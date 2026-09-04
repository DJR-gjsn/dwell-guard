# -*- coding: utf-8 -*-
"""驱逐决策（成本感知）单元测试：验证 eviction_value / decide_evictions 逻辑。

用法：python test_eviction.py
（纯逻辑测试，不连 Ollama——用假 client 模拟驻留模型）
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import dwell_guard as dg  # noqa: E402


class FakeClient:
    """假 OllamaClient：只提供 loaded_detail（不真连）。"""

    def __init__(self, loaded):
        self._loaded = loaded

    def loaded_detail(self):
        return self._loaded

    def unload(self, model):
        return True


def make_history(calls=None, transitions=None):
    tmp = os.path.join(tempfile.gettempdir(), 'dg_evict_test.json')
    if os.path.exists(tmp):
        os.remove(tmp)
    h = dg.History(tmp)
    if calls:
        h.data['calls'] = calls
    if transitions:
        h.data['transitions'] = transitions
    return h


def test_eviction_value():
    """验证驱逐价值排序：不常用+不被预测+占空间大 → 高驱逐优先级。"""
    h = make_history(
        calls={'qwen3:30b': 5, 'qwen3:0.6b': 1, 'qwen3:8b': 2},
        transitions={'qwen3:0.6b->qwen3:30b': 3, 'qwen3:30b->qwen3:0.6b': 1},
    )
    sch = dg.Scheduler(h, FakeClient([]))

    # 30B：高频(5) + 被预测(3) + 大 → 驱逐价值应最低（最该留）
    # 0.6b：低频(1) + 不被预测(0) + 小 → 驱逐价值中等
    # 8b：低频(2) + 不被预测 + 中等 → 驱逐价值中等偏高（8b > 0.6b 因为占空间大）
    v30 = sch.eviction_value({'name': 'qwen3:30b', 'size_gb': 17.0})
    v06 = sch.eviction_value({'name': 'qwen3:0.6b', 'size_gb': 0.6})
    v8 = sch.eviction_value({'name': 'qwen3:8b', 'size_gb': 5.0})
    print(f'驱逐价值: 30B={v30}, 8B={v8}, 0.6B={v06}')
    assert v30 < v06, f'30B(高频被预测)驱逐价值应低于 0.6b: {v30} vs {v06}'
    assert v30 < v8, f'30B 驱逐价值应低于 8B: {v30} vs {v8}'
    print('✅ eviction_value：高频+被预测的大模型驱逐价值低（该留）')


def test_decide_evictions():
    """验证驱逐选择：预载大模型时应驱逐低价值小模型，且保护当前/高频。"""
    h = make_history(
        calls={'qwen3:30b': 5, 'qwen3:0.6b': 1},
        transitions={'qwen3:0.6b->qwen3:30b': 3},
    )
    loaded = [
        {'name': 'qwen3:0.6b', 'size_gb': 0.6},   # 低频辅助模型 → 该被驱逐
        {'name': 'qwen3:30b', 'size_gb': 17.0},   # 高频被预测 → 保护
    ]
    sch = dg.Scheduler(h, FakeClient(loaded))

    # 预载 30B 需要 17GB，protect={'qwen3:30b'}（当前正用）
    evict = sch.decide_evictions(need_gb=17.0, protect={'qwen3:30b'})
    print(f'驱逐列表: {evict}')
    assert 'qwen3:0.6b' in evict, '应驱逐低频的 0.6b'
    assert 'qwen3:30b' not in evict, '不应驱逐保护的 30B'
    print('✅ decide_evictions：驱逐低价值模型、保护高频/当前模型')

    # 预载小模型（need_gb=0.6）→ 不应触发驱逐
    evict2 = sch.decide_evictions(need_gb=0.6, protect={'qwen3:30b'})
    print(f'小模型预载驱逐列表: {evict2}')
    assert evict2 == [], '小模型预载不应驱逐'
    print('✅ decide_evictions：小模型预载不驱逐')


if __name__ == '__main__':
    test_eviction_value()
    test_decide_evictions()
    print('\n🎉 驱逐决策单元测试全部通过')
