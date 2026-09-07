# -*- coding: utf-8 -*-
"""GUARDED 压力测试：守护学习 + 压力序列（不 stop，守护管理驻留）。

用法：python stress_guarded.py
需 dwell-guard 运行在 11500 + 已学习过模式。
"""
import json
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, r'D:\大创探索\local-llm-deploy\dwell_guard')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import stress_test as st

GUARD = 'http://127.0.0.1:11500'
OLLAMA_EXE = r'D:\本地模型测试\Ollama\ollama.exe'

LEARN_SEQ = ['qwen3:0.6b', 'qwen3:30b-a3b', 'qwen3:0.6b', 'qwen3.5:35b-a3b',
             'qwen3:0.6b', 'qwen3:30b-a3b']  # 学 1.5 轮


def learn():
    print('=== 学习阶段（让守护学到切换模式）===', flush=True)
    for m in LEARN_SEQ:
        try:
            ttft, tok_s, total, chars = st.stream_chat(GUARD, m)
            print(f'  学习 {m:<18} TTFT={ttft:6.1f}s', flush=True)
        except Exception as e:
            print(f'  学习 {m} FAIL: {e}', flush=True)
        time.sleep(2)
    print('学习完成\n', flush=True)


def run_pressure():
    print('=== GUARDED 压力序列（2轮，不stop）===', flush=True)
    st.ROUNDS = 2
    results = []
    for rnd in range(st.ROUNDS):
        for m in st.MODELS:
            try:
                ttft, tok_s, total, chars = st.stream_chat(GUARD, m)
                results.append({'model': m, 'ttft': ttft, 'tok_s': tok_s,
                                'total': total, 'chars': chars})
                print(f'  [{rnd+1}] {m:<18} TTFT={ttft:6.1f}s tok/s={tok_s:5.1f} '
                      f'总={total:6.1f}s', flush=True)
            except Exception as e:
                print(f'  [{rnd+1}] {m} FAIL: {e}', flush=True)
            time.sleep(1)
    st.summarize(results, 'GUARDED（dwell-guard 2轮）')
    with open(r'D:\大创探索\local-llm-deploy\dwell_guard\guarded_results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('结果已存 guarded_results.json', flush=True)


if __name__ == '__main__':
    learn()
    run_pressure()
