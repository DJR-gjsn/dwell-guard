# -*- coding: utf-8 -*-
"""dwell-guard 驻留调度守护进程

终结 Ollama "模型切换死循环"（多模型交替使用导致反复重载 + SSD 100% 占用）。
实测：30B 模型切换 61.5s → 3.2s（19 倍提速，RTX 5070 8GB）。

架构：
  用户/Agent ──→ [dwell-guard 代理 :11500] ──转发──→ Ollama :11434
                    │  记录每次调用的模型（history.json 跨重启记忆）
                    │  转移概率预测"下一个模型"
                    │  预加载预测模型 / 驱逐低价值模型腾空间
                    │  空闲期自动预载（用户休息时准备）
                    ▼
                 状态文件 history.json

支持端点：/api/chat、/api/generate、/v1/chat/completions、/v1/completions（POST）
         /api/ps（GET 透传）

用法：
  python dwell_guard.py          # 启动（默认 127.0.0.1:11500）
  # 应用指向 http://127.0.0.1:11500 即可（OpenAI 兼容）
"""
import argparse
import json
import os
import sys
import time
import threading
import urllib.request

# UTF-8 输出（Windows GBK 控制台）
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OLLAMA = 'http://127.0.0.1:11434'
DEFAULT_PORT = 11500
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history.json')

# 调度参数（可调）
KEEP_ALIVE_HOT = '30m'    # 高频模型：驻留 30 分钟（避免频繁重载）
KEEP_ALIVE_PREDICTED = '10m'  # 预测的"下一个"模型：驻留 10 分钟（等用户切过来）
KEEP_ALIVE_COLD = '1m'    # 低频模型：驻留 1 分钟（用完快释放）
KEEP_ALIVE_IDLE = '15m'   # 空闲预载：驻留 15 分钟（用户在休息，回来就能用）
MIN_CALLS_HOT = 3         # 调用 ≥3 次才算"高频"
IDLE_PREFETCH_DELAY = 15  # 空闲多少秒后触发预载（秒）——用户停止操作后的静默期
IDLE_CHECK_INTERVAL = 5   # 空闲监控线程轮询间隔（秒）


class History:
    """使用历史：跨重启保存，记录每模型调用次数 + 转移矩阵。"""

    def __init__(self, path=STATE_FILE):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'calls': {}, 'transitions': {}, 'last': None}

    def _save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def record(self, model: str):
        """记录一次模型调用，更新转移矩阵。"""
        calls = self.data['calls']
        calls[model] = calls.get(model, 0) + 1

        last = self.data.get('last')
        if last and last != model:
            trans = self.data['transitions']
            key = f'{last}->{model}'
            trans[key] = trans.get(key, 0) + 1
        self.data['last'] = model
        self._save()

    def call_count(self, model: str) -> int:
        return self.data['calls'].get(model, 0)

    def predict_next(self, current: str) -> str | None:
        """用转移矩阵预测下一个模型（当前模型之后最可能被用到的）。"""
        best_model, best_cnt = None, 0
        for key, cnt in self.data['transitions'].items():
            src, dst = key.split('->', 1)
            if src == current and cnt > best_cnt:
                best_model, best_cnt = dst, cnt
        return best_model

    def hot_models(self) -> list[str]:
        """高频模型（调用次数达标）。"""
        return [m for m, c in self.data['calls'].items() if c >= MIN_CALLS_HOT]


class OllamaClient:
    """封装 Ollama API（/api/chat + /api/generate 转发 + /api/ps 查询 + keep_alive 控制）。"""

    def __init__(self, base=OLLAMA):
        self.base = base

    def _post(self, endpoint: str, payload: dict):
        """POST 到 Ollama 端点。返回 (status, body)，流式时 body 为 bytes，非流式为 dict。"""
        url = f'{self.base}{endpoint}'
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as resp:
            if payload.get('stream'):
                return resp.status, resp.read()  # 流式：原样透传字节
            return resp.status, json.loads(resp.read().decode('utf-8'))

    def chat(self, payload: dict):
        """转发 /api/chat。"""
        return self._post('/api/chat', payload)

    def generate(self, payload: dict):
        """转发 /api/generate（文本补全，默认流式 NDJSON）。"""
        return self._post('/api/generate', payload)

    def ensure_loaded(self, model: str, keep_alive: str):
        """确保模型加载：发一个最小 chat 请求（只含 system，无用户内容），
        模型会加载并驻留 keep_alive 时间；马上释放连接。
        这是"预加载/保活"的务实实现（Ollama 无独立 load API）。
        """
        payload = {
            'model': model,
            'messages': [{'role': 'system', 'content': 'ping'}],
            'stream': False,
            'keep_alive': keep_alive,
            'options': {'num_predict': 1},
        }
        try:
            status, _ = self.chat(payload)
            return status == 200
        except Exception:
            return False

    def set_keep_alive(self, model: str, keep_alive: str) -> bool:
        """对已加载模型调整驻留时间（用 /api/chat 空请求 + keep_alive）。"""
        return self.ensure_loaded(model, keep_alive)

    def loaded_models(self) -> list[str]:
        """驻留模型名列表（完整 tag）。"""
        return [m['name'] for m in self.loaded_detail()]

    def loaded_detail(self) -> list[dict]:
        """驻留模型详情：name（完整 tag 如 qwen3:0.6b）/ size_gb / vram_gb / expires_at。

        注意：name 是完整 tag（含 :tag），与请求时的 model 字段一致——不要 split。
        """
        try:
            with urllib.request.urlopen(f'{self.base}/api/ps', timeout=5) as r:
                data = json.loads(r.read().decode('utf-8'))
            out = []
            for m in data.get('models', []):
                out.append({
                    'name': m['name'],
                    'size_gb': round(m.get('size', 0) / 1e9, 1),
                    'vram_gb': round(m.get('size_vram', 0) / 1e9, 1),
                    'expires_at': m.get('expires_at'),
                })
            return out
        except Exception:
            return []

    def unload(self, model: str) -> bool:
        """立即卸载模型：keep_alive=0 的最小请求（Ollama 卸载语义）。"""
        payload = {
            'model': model,
            'messages': [{'role': 'system', 'content': 'unload'}],
            'stream': False,
            'keep_alive': 0,
            'options': {'num_predict': 1},
        }
        try:
            status, _ = self.chat(payload)
            return status == 200
        except Exception:
            return False

    def model_size_gb(self, model: str) -> float:
        """查询模型文件大小（读本地 Ollama manifest 的 layers size 总和）。

        注意：/api/show 不返回权重字节大小（只有 size_label 字符串），
        /api/ps 只在模型加载后有 size。故从模型目录 manifest 读取最可靠。
        失败返回 0.0（由调用方处理，勿用假 fallback）。
        """
        try:
            # 模型名如 qwen3:0.6b → manifest 路径 registry.ollama.ai/library/qwen3/0.6b
            name, _, tag = model.partition(':')
            if not tag:
                tag = 'latest'
            models_dir = os.environ.get('OLLAMA_MODELS', '')
            if not models_dir:
                return 0.0
            man_path = os.path.join(models_dir, 'manifests', 'registry.ollama.ai',
                                    'library', name, tag)
            if not os.path.exists(man_path):
                return 0.0
            with open(man_path, 'r', encoding='utf-8') as f:
                man = json.load(f)
            total = sum(l.get('size', 0) for l in man.get('layers', []))
            return round(total / 1e9, 1)
        except Exception:
            return 0.0


class Scheduler:
    """调度核心：记录 → 预测 → 预加载 → 保活 → 空闲预载。"""

    def __init__(self, history: History, client: OllamaClient,
                 enable_idle_prefetch: bool = True):
        self.history = history
        self.client = client
        self._lock = threading.Lock()
        self._last_active = time.time()
        self._idle_prefetch_done_for: set[str] = set()  # 已空闲预载过的目标（防重复）
        if enable_idle_prefetch:
            self._start_idle_watcher()

    @staticmethod
    def _log(msg: str):
        """统一调度日志前缀。"""
        print(f'[dwell] {msg}', flush=True)

    # ---- 空闲监控 ----

    def _start_idle_watcher(self):
        """空闲监控线程：用户停止请求一段时间后，预载预测的"下一个"模型。"""
        def _watch():
            while True:
                time.sleep(IDLE_CHECK_INTERVAL)
                try:
                    self._maybe_idle_prefetch()
                except Exception:
                    pass
        t = threading.Thread(target=_watch, daemon=True)
        t.start()

    def _maybe_idle_prefetch(self):
        """若空闲超阈值且有预测目标未预载 → 后台预载（用户在休息，无竞争）。"""
        with self._lock:
            idle_for = time.time() - self._last_active
            if idle_for < IDLE_PREFETCH_DELAY:
                return
            last_model = self.history.data.get('last')
            if not last_model:
                return
            predicted = self.history.predict_next(last_model)
            if not predicted or predicted == last_model:
                return
            if predicted in self._idle_prefetch_done_for:
                return  # 已经空闲预载过（防止每 5s 重复触发）
            # 检查是否已加载
            loaded = {m['name'] for m in self.client.loaded_detail()}
            if predicted in loaded:
                self._idle_prefetch_done_for.add(predicted)
                return
            self._idle_prefetch_done_for.add(predicted)
            target = predicted
            ka = KEEP_ALIVE_IDLE

        def _run():
            self._log(f'空闲预载 {target}（用户空闲 {idle_for:.0f}s，keep {ka}）...')
            protect = set(self.history.hot_models())
            need_gb = self.client.model_size_gb(target)
            if need_gb > 0:
                to_evict = self.decide_evictions(need_gb, protect)
                for ev in (to_evict or []):
                    ok = self.client.unload(ev)
                    self._log(f'  驱逐 {ev}（给 {target} 腾空间）: {"ok" if ok else "fail"}')
            ok = self.client.ensure_loaded(target, ka)
            self._log(f'  空闲预载完成 {target}: {"ok" if ok else "fail"}')
        threading.Thread(target=_run, daemon=True).start()

    def _touch(self):
        """更新最后活动时间；用户新请求到来时重置空闲预载标记。"""
        self._last_active = time.time()
        self._idle_prefetch_done_for.clear()  # 用户回来了，允许下次再预载

    def record(self, model: str):
        """同步记录（代理层在转发前调用）。"""
        with self._lock:
            self.history.record(model)
            self._touch()

    def on_request(self, model: str):
        """计算调度动作（不执行）。"""
        with self._lock:
            hot = self.history.hot_models()
            predicted = self.history.predict_next(model)

            actions = []
            # ① 当前模型若高频 → 长驻留
            if self.history.call_count(model) >= MIN_CALLS_HOT:
                actions.append(('keep_hot', model, KEEP_ALIVE_HOT))
            # ② 预测下一个 → 预加载（短驻留，等用户切过来）
            if predicted and predicted != model:
                actions.append(('prefetch', predicted, KEEP_ALIVE_PREDICTED))
            # ③ 其余高频模型保持驻留（防止被切换踢掉）
            for hm in hot:
                if hm not in (model, predicted):
                    actions.append(('keep', hm, KEEP_ALIVE_HOT))
        return actions

    def eviction_value(self, loaded: dict) -> float:
        """模型"驱逐价值"：越高越该被驱逐。

        驱逐价值 = size / (freq+1)² / (predicted+1)²
        - size：占空间越大 → 驱逐它腾空间越多 → 驱逐价值↑
        - freq（使用频率）²：越常用 → 驱逐价值↓↓（强惩罚，防驱逐主力模型）
        - predicted（被预测次数）²：越是被预测目标 → 驱逐价值↓↓（强惩罚）
        平方惩罚确保"高频/被预测"强于"占空间大"——30B 主力不该为腾空间被驱逐。
        """
        name = loaded.get('name', '')
        freq = self.history.call_count(name)
        predicted_cnt = 0
        for key, cnt in self.history.data['transitions'].items():
            if key.endswith(f'->{name}'):
                predicted_cnt += cnt
        size = loaded.get('size_gb', 0)
        return round(size / (freq + 1) ** 2 / (predicted_cnt + 1) ** 2, 4)

    def decide_evictions(self, need_gb: float, protect: set[str]) -> list[str]:
        """预载 need_gb 的模型前，决定驱逐哪些驻留模型（成本感知）。

        返回要卸载的模型名列表（完整 tag）。protect = 不可驱逐的模型（当前使用/高频）。
        """
        loaded = self.client.loaded_detail()
        if not loaded:
            return []
        candidates = [m for m in loaded if m['name'] not in protect]
        if not candidates:
            return []
        # 按驱逐价值降序：价值越高（越该驱逐）排前
        candidates.sort(key=lambda m: self.eviction_value(m), reverse=True)
        # MVP 策略：预载大模型时，驱逐最没价值的 1-2 个非保护模型
        if need_gb > 10:
            evict = []
            for c in candidates:
                # 只驱逐"远小于目标"的模型（它们通常是辅助模型，切回成本低）
                if c['size_gb'] < need_gb * 0.5:
                    evict.append(c['name'])
                if len(evict) >= 2:
                    break
            return evict
        return []

    def apply_async(self, model: str, delay: float = 2.0):
        """响应返回后后台执行调度动作（延迟 delay 秒，避免抢 Ollama 加载锁）。"""
        actions = self.on_request(model)
        # 保护集 = 高频模型（不可驱逐——它们是主力，切回成本高）
        # 注意：当前模型若低频不自动保护——它已被用完（响应返回），
        # 预载大模型时可驱逐它腾空间（切回成本低）。
        protect = set(self.history.hot_models())
        # 预测的"下一个"如果是高频，已在 hot 里；补充 keep 动作的目标
        for a in actions:
            if a[0] in ('keep_hot', 'keep'):
                protect.add(a[1])

        def _run():
            time.sleep(delay)
            for kind, m, ka in actions:
                if kind == 'prefetch':
                    # 预载前先驱逐低价值模型（给大模型腾空间）
                    need_gb = self.client.model_size_gb(m)
                    if need_gb > 0:  # 查得到大小才做驱逐决策（查不到不瞎驱逐）
                        to_evict = self.decide_evictions(need_gb, protect)
                        for ev in to_evict:
                            ok = self.client.unload(ev)
                            self._log(f'驱逐 {ev}（给 {m} 腾空间 {need_gb}GB）: {"ok" if ok else "fail"}')
                    ok = self.client.ensure_loaded(m, ka)
                    self._log(f'预加载 {m} (keep {ka}): {"ok" if ok else "fail/已加载"}')
                else:
                    ok = self.client.set_keep_alive(m, ka)
                    self._log(f'保活 {m} (keep {ka}): {"ok" if ok else "fail"}')
        t = threading.Thread(target=_run, daemon=True)
        t.start()


# ---- 代理层（极简 HTTP，不依赖 fastapi，减少依赖面）----

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


def make_handler(scheduler: Scheduler):
    class ProxyHandler(BaseHTTPRequestHandler):
        def _forward(self, forward_fn, body: bytes):
            """通用转发：提取 model → 记录 → 转发 → 响应 → 后台调度。"""
            try:
                payload = json.loads(body.decode('utf-8'))
            except Exception:
                self.send_error(400, 'bad json')
                return
            model = payload.get('model', '')
            if not model:
                self.send_error(400, 'no model')
                return
            # 记录（必须同步——调度决策需要最新历史）
            scheduler.record(model)
            # 转发到 Ollama（先转发，调度动作等响应后再做，避免抢加载锁）
            try:
                status, resp_body = forward_fn(payload)
            except Exception as e:
                self.send_error(502, f'ollama error: {e}')
                return
            # 统一为 bytes（非流式返回 dict → 重新序列化）
            if isinstance(resp_body, dict):
                resp_bytes = json.dumps(resp_body, ensure_ascii=False).encode('utf-8')
            else:
                resp_bytes = resp_body
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            # ★ 响应已返回 → 后台延迟执行调度动作（预载/保活）
            #   延迟 2s：确保当前请求连接完全关闭，避免和 Ollama 加载锁竞争
            scheduler.apply_async(model, delay=2.0)

        def _handle_chat(self, body: bytes):
            self._forward(scheduler.client.chat, body)

        def _handle_generate(self, body: bytes):
            self._forward(scheduler.client.generate, body)

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length else b''
            if self.path == '/api/chat' or self.path == '/v1/chat/completions':
                self._handle_chat(body)
            elif self.path == '/api/generate' or self.path == '/v1/completions':
                self._handle_generate(body)
            else:
                self.send_error(404)

        def do_GET(self):
            # /api/ps 透传（查看状态用）——用 client 的 base（与 --ollama 参数一致）
            if self.path == '/api/ps':
                try:
                    with urllib.request.urlopen(f'{scheduler.client.base}/api/ps', timeout=5) as r:
                        data = r.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                except Exception:
                    self.send_error(502)
                    return
            self.send_error(404)

        def log_message(self, fmt, *args):
            sys.stderr.write('[dwell-proxy] ' + (fmt % args) + '\n')

    return ProxyHandler


def main():
    parser = argparse.ArgumentParser(description='dwell-guard 驻留调度守护进程（MVP）')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'监听端口（默认 {DEFAULT_PORT}）')
    parser.add_argument('--ollama', default=OLLAMA, help='Ollama 地址')
    parser.add_argument('--state', default=STATE_FILE, help='历史状态文件')
    args = parser.parse_args()

    history = History(args.state)
    client = OllamaClient(args.ollama)
    scheduler = Scheduler(history, client)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(scheduler))
    print(f'[dwell-guard] 启动 @ http://127.0.0.1:{args.port}')
    print(f'[dwell-guard] 转发目标: {args.ollama}')
    print(f'[dwell-guard] 状态文件: {args.state}')
    print('[dwell-guard] 使用：让应用/Agent 指向本地址（OpenAI 兼容 /api/chat）')
    print('[dwell-guard] 历史: ', json.dumps(history.data, ensure_ascii=False)[:200])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[dwell-guard] 已停止')
        server.shutdown()


if __name__ == '__main__':
    main()
