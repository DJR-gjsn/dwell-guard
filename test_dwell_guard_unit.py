# -*- coding: utf-8 -*-
"""dwell-guard 单元测试：History 记录/转移矩阵/预测/高频判定（无需真实模型）。

用法：python test_dwell_guard_unit.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import dwell_guard as dg  # noqa: E402


def test_history_and_prediction():
    tmp = os.path.join(tempfile.gettempdir(), 'dg_unit_test.json')
    if os.path.exists(tmp):
        os.remove(tmp)
    h = dg.History(tmp)
    # 模拟交替切换 A -> B -> A -> B -> A
    for m in ['A', 'B', 'A', 'B', 'A']:
        h.record(m)

    calls = h.data['calls']
    trans = h.data['transitions']
    print('calls:', calls)
    print('transitions:', trans)

    assert calls == {'A': 3, 'B': 2}, f'调用计数错误: {calls}'
    assert h.predict_next('A') == 'B', f'预测错误: {h.predict_next("A")}'
    assert h.predict_next('B') == 'A', f'预测错误: {h.predict_next("B")}'
    assert 'A' in h.hot_models() and 'B' not in h.hot_models(), '高频判定错误'
    print('✅ 单元测试通过：记录 / 转移矩阵 / 预测 / 高频判定')
    os.remove(tmp)


def test_scheduler_actions():
    """Scheduler.on_request 应产出预加载/保活动作。"""
    tmp = os.path.join(tempfile.gettempdir(), 'dg_unit_test2.json')
    if os.path.exists(tmp):
        os.remove(tmp)
    h = dg.History(tmp)
    # 先制造 A 高频 + A->B 强转移
    for m in ['A', 'B', 'A', 'B', 'A']:
        h.record(m)
    sch = dg.Scheduler(h, dg.OllamaClient('http://127.0.0.1:11434'))

    # 再请求 A：应产出 prefetch B + keep A（A 已高频）
    actions = sch.on_request('A')
    kinds = [a[0] for a in actions]
    print('A 请求后动作:', actions)
    assert 'prefetch' in kinds, '应预加载预测模型 B'
    assert any(a[1] == 'B' and a[0] == 'prefetch' for a in actions), '预加载对象应为 B'
    print('✅ Scheduler 动作测试通过：预测预加载 + 高频保活')
    os.remove(tmp)


if __name__ == '__main__':
    test_history_and_prediction()
    test_scheduler_actions()
    print('\n🎉 dwell-guard 单元测试全部通过')
