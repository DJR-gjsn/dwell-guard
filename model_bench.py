# -*- coding: utf-8 -*-
"""模型基准对比 v2：实际场景问答 + 完整回答输出 MD 报告。

每个模型回答 5 个实际应用问题（写作/编程/翻译/总结/逻辑），记录：
  - 冷启动 TTFT / 热态 TTFT / tok/s
  - 每个问题的完整回答
最后生成 model_bench_report.md：性能表 + 逐模型逐问题回答对比。

用法：python model_bench.py
需 Ollama serve 在 11434（D盘目录）。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OLLAMA = 'http://127.0.0.1:11434'
OLLAMA_EXE = r'D:\本地模型测试\Ollama\ollama.exe'
OUT_MD = r'D:\大创探索\local-llm-deploy\dwell_guard\model_bench_report.md'

MODELS = [
    'qwen3:0.6b', 'llama3.2:3b', 'deepseek-r1:7b', 'qwen3:8b',
    'qwen3:14b', 'qwen3:32b', 'qwen3:30b-a3b',
    'qwen3.5:35b-a3b', 'qwen3.6:35b-a3b',
]

# 每个模型的特殊选项：
#  - think: qwen3 系列支持 think=false 关闭思考提速；deepseek-r1 必须 think=true
#    且给足预算（其模板强制先思考，预算被思考耗尽则 content 为空）
#  - num_gpu: 8GB 显存跑 qwen3:32b（18.8GB 稠密）时 Ollama 自动分层会 CUDA OOM，
#    必须显式指定 GPU 层数（实测 20 层稳定加载，~4.5 tok/s）
MODEL_OPTS = {
    'deepseek-r1:7b': {'think': True, 'num_predict': 4000},
    'qwen3:32b': {'think': False, 'num_predict': 300, 'num_gpu': 20},
}

# 实际应用场景问题
QUESTIONS = [
    {
        'id': '写作',
        'q': '帮我写一段 80 字左右的咖啡馆下午的描写，要有画面感。',
    },
    {
        'id': '编程',
        'q': '用 Python 写一个函数：输入一个整数列表，返回其中出现次数最多的元素（若并列返回任意一个）。只给代码。',
    },
    {
        'id': '翻译',
        'q': '把这句话翻译成英文："今天的会议推迟到明天下午三点，因为主讲人临时有事。"',
    },
    {
        'id': '总结',
        'q': '把下面这段总结成一句话：大语言模型通过在海量文本上训练学习语言规律，能完成翻译、写作、问答等任务，但也会产生不符合事实的"幻觉"内容，因此关键场景需要人工核查。',
    },
    {
        'id': '逻辑陷阱',
        'q': '一个农夫有17只羊，除了9只全部死了，还剩几只？只回答数字。',
    },
]


def stream_chat(model, question, opts=None, timeout=600):
    """流式请求，返回 (ttft, tok_s, total, answer)。

    ttft = 首个 *content* chunk 时间（思考模型的 thinking 不算用户可读答案）。
    opts: 覆盖默认请求字段，如 {'think': True, 'num_predict': 2000, 'num_gpu': 20}
    """
    opts = opts or {}
    p = {'model': model,
         'messages': [{'role': 'user', 'content': question}],
         'stream': True}
    # 默认值
    if 'think' not in opts:
        p['think'] = False
    if 'num_predict' not in opts:
        opts['num_predict'] = 200
    p['options'] = {'num_predict': opts['num_predict']}
    if 'num_gpu' in opts:
        p['options']['num_gpu'] = opts['num_gpu']
    if 'think' in opts:
        p['think'] = opts['think']
    d = json.dumps(p).encode()
    req = urllib.request.Request(OLLAMA + '/api/chat', data=d,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    first_ts = None
    eval_count = 0
    eval_dur = 0
    parts = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            now = time.time()
            try:
                obj = json.loads(line)
            except Exception:
                continue
            content = (obj.get('message') or {}).get('content', '')
            if content:
                if first_ts is None:
                    first_ts = now
                parts.append(content)
            if obj.get('done'):
                eval_count = obj.get('eval_count', 0)
                eval_dur = obj.get('eval_duration', 0)
                break
    total = time.time() - t0
    ttft = first_ts - t0 if first_ts else total
    gen_s = eval_dur / 1e9 if eval_dur > 0 else max(total - ttft, 0.01)
    tok_s = eval_count / gen_s if gen_s > 0 else 0
    return ttft, tok_s, total, ''.join(parts)


def stop(model):
    subprocess.run([OLLAMA_EXE, 'stop', model], capture_output=True, timeout=15)


def stop_all():
    """停掉所有测试模型，避免上一模型驻留 GPU 干扰冷启动/加载。"""
    for m in MODELS:
        stop(m)
    time.sleep(2)


def bench_model(model):
    """测一个模型：冷/热 TTFT + tok/s + 5 问完整回答。"""
    print(f'\n[{model}] 开始测试...', flush=True)
    row = {'model': model, 'answers': {}}
    opts = MODEL_OPTS.get(model, {})

    # 冷启动（用第一个问题）——先确保无其它驻留干扰
    stop_all()
    ttft_cold, _, total_cold, ans_warm = stream_chat(model, QUESTIONS[0]['q'], opts)
    print(f'  冷启动 TTFT: {ttft_cold:.1f}s (总 {total_cold:.1f}s)', flush=True)
    row['ttft_cold'] = round(ttft_cold, 1)
    row['answers'][QUESTIONS[0]['id']] = ans_warm

    # 热态指标（重问第一问拿 tok/s）
    ttft_hot, tok_s, total_hot, _ = stream_chat(model, QUESTIONS[0]['q'], opts)
    print(f'  热态 TTFT: {ttft_hot:.2f}s | tok/s: {tok_s:.1f}', flush=True)
    row['ttft_hot'] = round(ttft_hot, 2)
    row['tok_s'] = round(tok_s, 1)

    # 其余问题（热态）
    for qi in range(1, len(QUESTIONS)):
        q = QUESTIONS[qi]
        _, _, _, ans = stream_chat(model, q['q'], opts)
        row['answers'][q['id']] = ans
        print(f'  [{q["id"]}] 回答 {len(ans)} 字', flush=True)
    stop(model)  # 测完即停，避免驻留干扰下一模型
    return row


def md_escape(text):
    """转义 MD 特殊字符，保留换行。"""
    if not text:
        return '*(无输出)*'
    return text.replace('|', '\\|').strip()


def write_report(results):
    """生成 MD 报告：性能表 + 逐模型逐问回答。"""
    lines = []
    lines.append('# 本地模型对比测试报告')
    lines.append('')
    lines.append(f'> 硬件：RTX 5070 Laptop 8GB + 32GB 内存 ｜ Ollama 0.33.3 ｜ 日期：2026-09-07')
    lines.append(f'> 测试模型：{len(results)} 个（qwen3 系列 / llama3.2 / deepseek-r1）')
    lines.append('')
    lines.append('## 性能总表')
    lines.append('')
    lines.append('| 模型 | 冷启动 TTFT (s) | 热态 TTFT (s) | 生成速度 (tok/s) |')
    lines.append('|---|---|---|---|')
    for r in results:
        if 'error' in r:
            lines.append(f"| {r['model']} | ERROR: {r['error']} | - | - |")
        else:
            lines.append(f"| {r['model']} | {r['ttft_cold']} | {r['ttft_hot']} | {r['tok_s']} |")
    lines.append('')
    lines.append('> TTFT = 首个可读答案 token 时间（切换等待痛点）；生成速度为热态稳定值。')
    lines.append('> 注：deepseek-r1:7b 为思考模型（think=true，预算 2000 token），TTFT/tok/s 含思考过程，')
    lines.append('> 速度为其真实体感速度。qwen3:32b 为 18.8GB 稠密模型，8GB 显存下 Ollama 自动分层')
    lines.append('> 会 CUDA OOM，故显式 num_gpu=20（GPU 20 层 + CPU 44 层），速度即分层后真实值。')
    lines.append('')

    # 逐问题逐模型回答
    for qi, qobj in enumerate(QUESTIONS):
        lines.append(f'## 问题 {qi+1}：{qobj["id"]}')
        lines.append('')
        lines.append(f'**提问**：{qobj["q"]}')
        lines.append('')
        for r in results:
            if 'error' in r:
                continue
            ans = r['answers'].get(qobj['id'], '*(无回答)*')
            lines.append(f'### {r["model"]}')
            lines.append('')
            lines.append('```')
            lines.append(md_escape(ans))
            lines.append('```')
            lines.append('')
    lines.append('---')
    lines.append('*报告由 model_bench.py 自动生成*')
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n报告已生成: {OUT_MD}')


def main():
    # 用法：python model_bench.py [--only model1 model2 ...]
    # 结果缓存到 bench_cache.json；--only 时只测指定模型，其余从缓存合并
    args = sys.argv[1:]
    only = []
    if args and args[0] == '--only':
        only = args[1:]
        if not only:
            print('用法: python model_bench.py [--only model1 model2 ...]')
            sys.exit(1)

    CACHE = OUT_MD.replace('model_bench_report.md', 'bench_cache.json')
    cache = {}
    if os.path.exists(CACHE):
        try:
            with open(CACHE, 'r', encoding='utf-8') as f:
                cache = {r['model']: r for r in json.load(f)}
        except Exception:
            cache = {}

    targets = only if only else MODELS
    results = []
    for m in targets:
        try:
            results.append(bench_model(m))
        except Exception as e:
            print(f'  [{m}] FAIL: {e}', flush=True)
            results.append({'model': m, 'error': str(e)})

    # 合并：本次结果覆盖缓存，其余模型从缓存取
    merged = {r['model']: r for r in results}
    for m, r in cache.items():
        if m not in merged:
            merged[m] = r
    all_results = [merged[m] for m in MODELS if m in merged]

    with open(CACHE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)

    write_report(all_results)
    # 控制台也打一份简要汇总
    print(f'\n{"="*50}\n性能汇总\n{"="*50}')
    print(f'{"模型":<20} {"冷TTFT":<8} {"热TTFT":<8} {"tok/s":<8}')
    for r in all_results:
        if 'error' in r:
            print(f'{r["model"]:<20} ERROR')
        else:
            print(f'{r["model"]:<20} {r["ttft_cold"]:<8.1f} {r["ttft_hot"]:<8.2f} {r["tok_s"]:<8.1f}')


if __name__ == '__main__':
    main()
