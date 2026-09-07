# -*- coding: utf-8 -*-
"""聚焦对比：反复切 0.6b→30B（轻查询后切回主力）——守护预测应稳定命中。

NATIVE:   每次 stop（全冷切换）
GUARDED:  守护学习"0.6b→30B"模式后预载

各跑 N 轮，统计切到 30B 的 TTFT（这是死循环痛点的核心指标）。
"""
import json
import subprocess
import sys
import time

sys.path.insert(0, r'D:\大创探索\local-llm-deploy\dwell_guard')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import stress_test as st

OLLAMA_EXE = r'D:\本地模型测试\Ollama\ollama.exe'
LIGHT = 'qwen3:0.6b'
MAIN = 'qwen3:30b-a3b'


def stop(model):
    subprocess.run([OLLAMA_EXE, 'stop', model], capture_output=True, timeout=15)


def run_native(rounds=5):
    """每次切 MAIN 前 stop（全冷）。"""
    print(f'=== NATIVE 聚焦（{rounds} 轮，每次冷切 30B）===', flush=True)
    ttfts = []
    for i in range(rounds):
        # 轻查询（热或冷均可）
        st.stream_chat(st.OLLAMA, LIGHT)
        # 切主力：先 stop 模拟内存紧张换出
        stop(MAIN)
        time.sleep(0.5)
        ttft, tok_s, total, cnt = st.stream_chat(st.OLLAMA, MAIN)
        ttfts.append(ttft)
        print(f'  轮{i+1}: 切 30B TTFT={ttft:.1f}s', flush=True)
    avg = sum(ttfts) / len(ttfts)
    print(f'  平均 TTFT: {avg:.1f}s', flush=True)
    return {'mode': 'native', 'ttfts': ttfts, 'avg': avg}


def run_guarded(rounds=5):
    """经守护：先学习（0.6b→30B 交替 2 轮），再测。"""
    print(f'=== GUARDED 聚焦（学习后 {rounds} 轮）===', flush=True)
    # 学习 2 轮（0.6b→30B 交替）
    for i in range(2):
        st.stream_chat(st.GUARD, LIGHT)
        st.stream_chat(st.GUARD, MAIN)
    # 清冷态（模拟长时间后全卸载）
    stop(MAIN)
    stop(LIGHT)
    time.sleep(2)
    ttfts = []
    for i in range(rounds):
        st.stream_chat(st.GUARD, LIGHT)  # 轻查询 → 守护应预载 30B
        time.sleep(1)
        ttft, tok_s, total, cnt = st.stream_chat(st.GUARD, MAIN)
        ttfts.append(ttft)
        print(f'  轮{i+1}: 切 30B TTFT={ttft:.1f}s', flush=True)
    avg = sum(ttfts) / len(ttfts)
    print(f'  平均 TTFT: {avg:.1f}s', flush=True)
    return {'mode': 'guarded', 'ttfts': ttfts, 'avg': avg}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'native'
    if mode == 'native':
        r = run_native()
    elif mode == 'guarded':
        r = run_guarded()
    else:
        print('用法: stress_focus.py [native|guarded]')
        return
    with open(rf'D:\大创探索\local-llm-deploy\dwell_guard\focus_{mode}.json', 'w') as f:
        json.dump(r, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
