# -*- coding: utf-8 -*-
"""3+ 模型场景验证：35B 主力 + 30B/0.6b 辅助的真实多模型调度。

场景：35B(写作主力) → 0.6b(快速查询) → 35B → 30B(日常) → 35B
验证：
  1. 35B 高频 → 保持驻留不被驱逐
  2. 0.6b/30B 低频用完 → 被驱逐给 35B 腾空间（若需）
  3. 切回 35B 时间远低于冷启动（35B 冷启动 ~90s）

用法：需 dwell-guard 运行在 11500 + Ollama D盘 serve。
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

# 模型角色
MAIN = 'qwen3.5:35b-a3b'    # 主力（写作/重任务）
MID = 'qwen3:30b-a3b'        # 次用（日常对话）
LIGHT = 'qwen3:0.6b'         # 轻量（快速查询）


def chat(base, model, timeout=300):
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
    # 1. 学习阶段：35B 高频(×4)，30B/0.6b 低频(×1-2)
    print('=== 阶段1：学习（35B 主力高频）===', flush=True)
    seq = [MAIN, LIGHT, MAIN, MID, MAIN, LIGHT, MAIN]  # 35B×4, 0.6b×2, 30B×1
    for m in seq:
        dt = chat(BASE, m)
        print(f'  学习 {m}: {dt:.1f}s', flush=True)
        time.sleep(3)

    # 2. 冷态清空
    for m in [MAIN, MID, LIGHT]:
        stop(m)
    time.sleep(3)
    print(f'冷态: {ps()}', flush=True)

    # 3. 验证场景：模拟"刚用完轻模型切回主力"
    print('=== 阶段2：轻模型 → 切回主力 ===', flush=True)
    t1 = chat(BASE, LIGHT, timeout=90)
    print(f'[轻] {LIGHT}: {t1:.1f}s | 驻留: {ps()}', flush=True)
    print('等待守护动作（驱逐/预载 35B ~90s）...', flush=True)
    time.sleep(95)
    print(f'[守护后] 驻留: {ps()}', flush=True)
    t2 = chat(BASE, MAIN, timeout=300)
    print(f'[切主力] {MAIN}: {t2:.1f}s（35B 冷启动基线 ~90s）', flush=True)
    print(f'\n=== 结果 ===')
    print(f'冷启动基线(35B): ~90s | 调度后切主力: {t2:.1f}s', flush=True)


if __name__ == '__main__':
    main()
