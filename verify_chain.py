# -*- coding: utf-8 -*-
"""驱逐+预载完整链路验证（独立脚本，排除代理层干扰）。

模拟守护在"0.6b 请求完成后"的动作：
  1. 加载 0.6b（模拟用户刚用完）
  2. 调用 decide_evictions → 驱逐 0.6b（低频）
  3. ensure_loaded 预载 30B
  4. 切 30B 测速（对比冷启动 61.5s / 无驱逐 30.7s）
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import dwell_guard as dg  # noqa: E402

OLLAMA = 'http://127.0.0.1:11434'


def stop_all():
    for m in ['qwen3:30b-a3b', 'qwen3:0.6b']:
        subprocess.run([r'D:\本地模型测试\Ollama\ollama.exe', 'stop', m],
                       capture_output=True)
    time.sleep(2)


def ps():
    r = json.loads(urllib.request.urlopen(OLLAMA + '/api/ps', timeout=5).read())
    return [(m['name'], round(m.get('size_vram', 0) / 1e9, 1)) for m in r.get('models', [])]


def chat_direct(model, timeout=200):
    p = {'model': model, 'messages': [{'role': 'user', 'content': 'hi'}],
         'stream': False, 'think': False, 'options': {'num_predict': 3}}
    d = json.dumps(p).encode()
    req = urllib.request.Request(OLLAMA + '/api/chat', data=d,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.time() - t0


def main():
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verify_state.json')
    if os.path.exists(tmp):
        os.remove(tmp)
    h = dg.History(tmp)
    h.data['calls'] = {'qwen3:30b-a3b': 3, 'qwen3:0.6b': 2}
    h.data['transitions'] = {'qwen3:30b-a3b->qwen3:0.6b': 2,
                             'qwen3:0.6b->qwen3:30b-a3b': 2}
    h.data['last'] = 'qwen3:30b-a3b'
    c = dg.OllamaClient(OLLAMA)
    sch = dg.Scheduler(h, c)

    stop_all()
    print('[1] 加载 0.6b（模拟用户刚用完）...', flush=True)
    c.ensure_loaded('qwen3:0.6b', '5m')
    time.sleep(1)
    print(f'    驻留: {ps()}', flush=True)

    print('[2] 驱逐决策 + 预载 30B（复刻守护动作）...', flush=True)
    protect = set(h.hot_models())  # {30B}
    need_gb = c.model_size_gb('qwen3:30b-a3b') or 17.0
    to_evict = sch.decide_evictions(need_gb, protect)
    print(f'    驱逐: {to_evict} (need {need_gb}GB)', flush=True)
    for ev in to_evict:
        ok = c.unload(ev)
        print(f'    卸载 {ev}: {ok}', flush=True)
    time.sleep(2)
    print(f'    驱逐后驻留: {ps()}', flush=True)

    t0 = time.time()
    ok = c.ensure_loaded('qwen3:30b-a3b', '10m')
    print(f'    预载 30B: {ok} ({time.time()-t0:.1f}s)', flush=True)
    time.sleep(1)
    print(f'    预载后驻留: {ps()}', flush=True)

    print('[3] 切到 30b 测速...', flush=True)
    t_switch = chat_direct('qwen3:30b-a3b')
    print(f'[3] 30b 切换: {t_switch:.1f}s', flush=True)
    print(f'\n=== 结果 ===')
    print(f'冷启动基线: 61.5s')
    print(f'无驱逐预载(历史实测): 30.7s')
    print(f'驱逐+预载(本次): {t_switch:.1f}s')
    print(f'理论最优(隔离): 3.1s')
    os.remove(tmp)


if __name__ == '__main__':
    main()
