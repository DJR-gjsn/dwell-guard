# -*- coding: utf-8 -*-
"""dwell-guard 压力测试：多次模型切换 + 长时间，测 TTFT / tok/s / 切换间隔，原生 vs 守护。

用法：
  python stress_test.py native    # 直连 Ollama，每次 stop 模拟换出（冷启动）
  python stress_test.py guarded   # 经 dwell-guard（需先启动守护 + 学习过）

指标：
  TTFT      首 token 时间（请求→首个 token）
  tok/s     生成速度（内容字符累计 / 生成耗时）
  切换间隔  每个请求的总耗时
"""
import json
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OLLAMA = 'http://127.0.0.1:11434'
GUARD = 'http://127.0.0.1:11500'
OLLAMA_EXE = r'D:\本地模型测试\Ollama\ollama.exe'

# 测试模型：小(0.6b) 中(30b) 大(35b) 轮换
MODELS = ['qwen3:0.6b', 'qwen3:30b-a3b', 'qwen3:0.6b', 'qwen3.5:35b-a3b']
ROUNDS = 2        # 完整轮数（每轮 4 次请求）
PROMPT = 'Write a short sentence about the ocean.'


def stream_chat(base, model):
    """流式请求，返回 (ttft_s, tok_per_s, total_s, eval_count)。

    TTFT = 首行到达时间；tok/s 用 Ollama done 行的权威字段
    (eval_count / eval_duration)，而非客户端字符统计（流式是攒批的，不准）。
    """
    p = {'model': model,
         'messages': [{'role': 'user', 'content': PROMPT}],
         'stream': True, 'think': False,
         'options': {'num_predict': 60}}
    d = json.dumps(p).encode()
    req = urllib.request.Request(base + '/api/chat', data=d,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    first_ts = None
    eval_count = 0
    eval_duration_ns = 0
    done_ts = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            now = time.time()
            if first_ts is None:
                first_ts = now  # 首个 chunk（≈首 token）
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get('done'):
                eval_count = obj.get('eval_count', 0)
                eval_duration_ns = obj.get('eval_duration', 0)
                done_ts = now
                break
    total = done_ts - t0
    ttft = first_ts - t0
    gen_s = eval_duration_ns / 1e9 if eval_duration_ns > 0 else max(total - ttft, 0.01)
    tok_s = eval_count / gen_s if gen_s > 0 else 0
    return ttft, tok_s, total, eval_count


def stop(model):
    subprocess.run([OLLAMA_EXE, 'stop', model], capture_output=True, timeout=15)


def run_native():
    """原生：每次请求前 stop（模拟内存紧张被迫换出 = 冷启动切换）。"""
    print('=== NATIVE（直连 Ollama，每次 stop 冷切换）===', flush=True)
    results = []
    for rnd in range(ROUNDS):
        for m in MODELS:
            stop(m)
            time.sleep(1)
            try:
                ttft, tok_s, total, chars = stream_chat(OLLAMA, m)
                results.append({'model': m, 'ttft': ttft, 'tok_s': tok_s,
                                'total': total, 'chars': chars})
                print(f'  [{rnd+1}] {m:<18} TTFT={ttft:6.1f}s tok/s={tok_s:5.1f} '
                      f'总={total:6.1f}s', flush=True)
            except Exception as e:
                print(f'  [{rnd+1}] {m} FAIL: {e}', flush=True)
    return results


def run_guarded():
    """经 dwell-guard：守护已学习模式，预载生效。"""
    print('=== GUARDED（经 dwell-guard）===', flush=True)
    results = []
    for rnd in range(ROUNDS):
        for m in MODELS:
            try:
                ttft, tok_s, total, chars = stream_chat(GUARD, m)
                results.append({'model': m, 'ttft': ttft, 'tok_s': tok_s,
                                'total': total, 'chars': chars})
                print(f'  [{rnd+1}] {m:<18} TTFT={ttft:6.1f}s tok/s={tok_s:5.1f} '
                      f'总={total:6.1f}s', flush=True)
            except Exception as e:
                print(f'  [{rnd+1}] {m} FAIL: {e}', flush=True)
    return results


def summarize(results, label):
    print(f'\n=== {label} 汇总 ===', flush=True)
    if not results:
        print('  无数据')
        return
    # 按模型分组
    by_model = {}
    for r in results:
        by_model.setdefault(r['model'], []).append(r)
    print(f'{"模型":<18} {"请求数":<6} {"平均TTFT":<10} {"平均tok/s":<10} {"平均总耗时":<10}', flush=True)
    for m, rs in by_model.items():
        avg_ttft = sum(x['ttft'] for x in rs) / len(rs)
        avg_tok = sum(x['tok_s'] for x in rs) / len(rs)
        avg_total = sum(x['total'] for x in rs) / len(rs)
        print(f'{m:<18} {len(rs):<6} {avg_ttft:<10.1f} {avg_tok:<10.1f} {avg_total:<10.1f}', flush=True)
    # 切到 30B/35B 的 TTFT（真实痛点：大模型切换）
    big = [r for r in results if r['model'] in ('qwen3:30b-a3b', 'qwen3.5:35b-a3b')]
    if big:
        avg_big_ttft = sum(x['ttft'] for x in big) / len(big)
        print(f'\n大模型(30B/35B)平均 TTFT: {avg_big_ttft:.1f}s（这是切换死循环的痛点）', flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'native'
    if mode == 'native':
        results = run_native()
        summarize(results, 'NATIVE 原生（冷切换）')
    elif mode == 'guarded':
        results = run_guarded()
        summarize(results, 'GUARDED（dwell-guard）')
    else:
        print('用法: stress_test.py [native|guarded]')


if __name__ == '__main__':
    main()
