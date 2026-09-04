# -*- coding: utf-8 -*-
"""空闲预载单元测试：验证 idle watcher 的触发/去重/重置逻辑。"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import dwell_guard as dg  # noqa: E402


class FakeClient:
    """假 OllamaClient：loaded_detail 返回指定驻留，ensure_loaded 记录调用。"""

    def __init__(self, loaded=None):
        self._loaded = loaded or []
        self.ensure_calls = []
        self.unload_calls = []

    def loaded_detail(self):
        return self._loaded

    def model_size_gb(self, model):
        return {'qwen3:30b-a3b': 18.6}.get(model, 0.5)

    def ensure_loaded(self, model, keep_alive):
        self.ensure_calls.append((model, keep_alive))
        return True

    def unload(self, model):
        self.unload_calls.append(model)
        return True


def make_history(calls, transitions, last):
    tmp = os.path.join(tempfile.gettempdir(), 'dg_idle_test.json')
    if os.path.exists(tmp):
        os.remove(tmp)
    h = dg.History(tmp)
    h.data['calls'] = calls
    h.data['transitions'] = transitions
    h.data['last'] = last
    return h


def test_idle_prefetch_triggers():
    """空闲超阈值且有预测目标 → 应触发预载（且只一次）。"""
    h = make_history(
        calls={'qwen3:30b-a3b': 4, 'qwen3:0.6b': 1},
        transitions={'qwen3:0.6b->qwen3:30b-a3b': 2},
        last='qwen3:0.6b',
    )
    fc = FakeClient()  # 无驻留
    sch = dg.Scheduler(h, fc, enable_idle_prefetch=False)  # 关闭自动线程，手动触发

    # 模拟空闲 20s（> IDLE_PREFETCH_DELAY=15）
    sch._last_active = time.time() - 20
    sch._maybe_idle_prefetch()
    time.sleep(0.5)  # 等后台线程执行

    assert fc.ensure_calls, '空闲后应预载'
    target = fc.ensure_calls[0][0]
    print(f'空闲预载目标: {target}, calls: {fc.ensure_calls}')
    assert target == 'qwen3:30b-a3b', f'应预载 30B，实际 {target}'
    print('✅ 空闲预载触发正确')


def test_idle_prefetch_no_duplicate():
    """同目标不应重复空闲预载。"""
    h = make_history(
        calls={'qwen3:30b-a3b': 4, 'qwen3:0.6b': 1},
        transitions={'qwen3:0.6b->qwen3:30b-a3b': 2},
        last='qwen3:0.6b',
    )
    fc = FakeClient()
    sch = dg.Scheduler(h, fc, enable_idle_prefetch=False)

    sch._last_active = time.time() - 20
    sch._maybe_idle_prefetch()
    time.sleep(0.5)
    n1 = len(fc.ensure_calls)
    # 再触发一次（同目标）→ 不应重复
    sch._last_active = time.time() - 20
    sch._maybe_idle_prefetch()
    time.sleep(0.5)
    n2 = len(fc.ensure_calls)
    print(f'第一次预载次数: {n1}, 第二次后: {n2}')
    assert n2 == n1, '同目标不应重复空闲预载'
    print('✅ 空闲预载去重正确')


def test_idle_prefetch_reset_on_activity():
    """用户新请求到来后，空闲预载标记应重置（允许下次再预载）。"""
    h = make_history(
        calls={'qwen3:30b-a3b': 4, 'qwen3:0.6b': 1},
        transitions={'qwen3:0.6b->qwen3:30b-a3b': 2},
        last='qwen3:0.6b',
    )
    fc = FakeClient()
    sch = dg.Scheduler(h, fc, enable_idle_prefetch=False)

    # 首次空闲预载
    sch._last_active = time.time() - 20
    sch._maybe_idle_prefetch()
    time.sleep(0.5)
    assert sch._idle_prefetch_done_for, '预载后应有标记'

    # 用户回来请求 → 重置
    sch.record('qwen3:0.6b')  # 内部调用 _touch 清空标记
    assert not sch._idle_prefetch_done_for, '用户回来后标记应清空'
    print('✅ 用户活动重置空闲预载标记')


if __name__ == '__main__':
    test_idle_prefetch_triggers()
    test_idle_prefetch_no_duplicate()
    test_idle_prefetch_reset_on_activity()
    print('\n🎉 空闲预载单元测试全部通过')
