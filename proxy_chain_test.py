# -*- coding: utf-8 -*-
"""代理链路完整复测（经 dwell-guard 守护，验证驱逐+预载达到 3.1s）。

流程：
  1. 学习：30B 高频×4、0.6b 低频×2（序列保证 0.6b 不进高频）
  2. 冷态清空
  3. 经守护请求 0.6b（低频）→ 守护应驱逐 0.6b + 预载 30B
  4. 切 30B 测速（预期 ~3.1s）
"""
import json
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = 'http://127.0.0.1:11500'
OLLAMA = 'http://127.0.0.1:11434'
OLLAMA_EXE = r'D:\本地模型测试\Ollama\ollama.exe'


def chat(base, model, timeout=200):
    p = {'model': model, 'messages': [{'role': 'user', 'content': 'hi'}],
         'stream': False, 'think': False, 'options': {'num_predict': 3}}
    d = json.dumps(p).encode()
    req = urllib.request.Request(base + '/api/chat', data=d,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.time() - t0


def stop(model):
    subprocess.run([OLLAMA_EXE, 'stop', model], capture_output=True)


def ps():
    r = json.loads(urllib.request.urlopen(OLLAMA + '/api/ps', timeout=5).read())
    return [m['name'] for m in r.get('models', [])]


def main():
    # ---- 1. 学习阶段 ----
    print('=== 阶段1：学习（30B×4 高频, 0.6b×2 低频）===', flush=True)
    seq = ['qwen3:30b-a3b', 'qwen3:0.6b', 'qwen3:30b-a3b',
           'qwen3:0.6b', 'qwen3:30b-a3b', 'qwen3:30b-a3b']
    for m in seq:
        dt = chat(BASE, m)
        print(f'  学习 {m}: {dt:.1f}s', flush=True)
        time.sleep(3)
    print('学习完成', flush=True)

    # ---- 2. 冷态 ----
    stop('qwen3:30b-a3b')
    stop('qwen3:0.6b')
    time.sleep(3)
    print(f'冷态驻留: {ps()}', flush=True)

    # ---- 3. 请求 0.6b → 等待驱逐+预载 ----
    print('=== 阶段2：请求 0.6b（低频）===', flush=True)
    t1 = chat(BASE, 'qwen3:0.6b', timeout=90)
    print(f'[1] 0.6b: {t1:.1f}s | 驻留: {ps()}', flush=True)
    print('等待守护：驱逐 0.6b + 预载 30B（~75s）...', flush=True)
    time.sleep(75)
    print(f'[2] 预载后驻留: {ps()}', flush=True)

    # ---- 4. 切 30B 测速 ----
    print('=== 阶段3：切 30B ===', flush=True)
    t2 = chat(BASE, 'qwen3:30b-a3b', timeout=200)
    print(f'[3] 30b 切换: {t2:.1f}s', flush=True)
    print(f'\n结果: 冷启动61.5s | 驱逐+预载代理链路={t2:.1f}s | 理论最优3.1s', flush=True)


if __name__ == '__main__':
    main()
