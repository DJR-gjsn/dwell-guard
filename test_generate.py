# -*- coding: utf-8 -*-
"""/api/generate 透传测试：非流式 + 流式。"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = 'http://127.0.0.1:11500'


def main():
    # 1. 非流式 generate
    print('=== 非流式 /api/generate ===', flush=True)
    p = {'model': 'qwen3:0.6b', 'prompt': 'hi', 'stream': False,
         'think': False, 'options': {'num_predict': 5}}
    d = json.dumps(p).encode()
    req = urllib.request.Request(BASE + '/api/generate', data=d,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode())
        resp_text = resp.get('response', '')[:40]
        print(f'OK ({time.time()-t0:.1f}s): response={resp_text!r}, done={resp.get("done")}', flush=True)
    except Exception as e:
        print(f'FAIL: {e}', flush=True)

    # 2. 流式 generate（NDJSON 多行）
    print('=== 流式 /api/generate ===', flush=True)
    p['stream'] = True
    d = json.dumps(p).encode()
    req = urllib.request.Request(BASE + '/api/generate', data=d,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
        lines = [l for l in raw.strip().split('\n') if l.strip()]
        first = lines[0][:80] if lines else '(空)'
        print(f'OK: {len(lines)} 行 NDJSON, 首行: {first}...', flush=True)
    except Exception as e:
        print(f'FAIL: {e}', flush=True)


if __name__ == '__main__':
    main()
