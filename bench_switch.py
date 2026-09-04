# -*- coding: utf-8 -*-
"""dwell-guard before/after 切换对比基准（攒 README 数据）。

测法：交替调用模型 A(0.6b) → B(1.7b) → A → B（共 4 次，模拟真实切换工作流），
统计每次"请求到响应"的耗时。

- before：直连 Ollama（无守护）→ 每次切换都要冷启动
- after ：走 dwell-guard（守护学交替模式后预加载）→ 切换应显著变快

用法：
  python bench_switch.py before   # 直连 Ollama 测
  python bench_switch.py after    # 走 dwell-guard :11500 测（需先启动守护）
"""
import json
import os
import sys
import time
import urllib.request

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OLLAMA = 'http://127.0.0.1:11434'
GUARD = 'http://127.0.0.1:11500'
# A = 重模型（主力），B = 轻模型（偶尔用）——真实"大模型切换"场景
MODELS = ['qwen3:30b-a3b', 'qwen3:0.6b']
ROUNDS = 1  # 30B 冷启动 60s+，交替 1 轮 = 3 次调用已耗时 ~2 分钟


def stop_model(model):
    """强制卸载模型（ollama stop），模拟"内存不够被迫换出"的真实场景。"""
    import subprocess
    subprocess.run(['D:/本地模型测试/Ollama/ollama.exe', 'stop', model],
                   capture_output=True, timeout=10)


def chat(base, model, force_cold=False):
    """发一个最小请求，返回耗时秒。force_cold=True 时先卸载（模拟冷切换）。"""
    if force_cold:
        stop_model(model)
        time.sleep(0.5)
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'hi'}],
        'stream': False,
        'think': False,
        'options': {'num_predict': 3},
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(base + '/api/chat', data=data,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    return time.time() - t0


def run(base, label, force_cold=True):
    print(f'\n=== {label} ===')
    times = []
    for rnd in range(ROUNDS):
        for m in MODELS:
            dt = chat(base, m, force_cold=force_cold)
            times.append(dt)
            print(f'  {m:<15} -> {dt:5.1f}s')
    print(f'  总耗时: {sum(times):.1f}s | 平均: {sum(times)/len(times):.1f}s')
    return times


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'before'
    if mode == 'before':
        t = run(OLLAMA, 'BEFORE（直连 Ollama，每次冷切换）')
    elif mode == 'after':
        # 守护先学 A/B 交替模式（真实使用几轮，不强制卸载——守护靠 keep_alive 保活）
        print('守护学习阶段（真实交替调用）...')
        for _ in range(3):
            for m in MODELS:
                chat(GUARD, m)
        time.sleep(3)
        # 清空状态，从"冷"开始测 after（让守护的预加载发挥作用）
        for m in MODELS:
            stop_model(m)
        time.sleep(1)
        t = run(GUARD, 'AFTER（走 dwell-guard，预加载生效）', force_cold=False)
    else:
        print('用法: bench_switch.py [before|after]')
        return

    print(f'\n--- README 数据 ---')
    print(f'{mode}_total = {sum(t):.1f}s')
    print(f'{mode}_switches = {[f"{x:.1f}s" for x in t]}')


if __name__ == '__main__':
    main()
