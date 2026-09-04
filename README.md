# dwell-guard

> **终结 Ollama 的"模型切换死循环"**——自动学习你的模型使用模式，预载预测目标、驱逐低价值模型、空闲期自动准备。实测把 30B 模型切换从 **61.5s 降到 3.2s**。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ollama 在多模型交替使用时（编码模型 ↔ 翻译模型 ↔ 主力模型来回切）会**反复重载**：每次冷启动读全量权重（6-86 秒），反复换入换出导致响应极慢、SSD 高占用——社区称之为 *model-swap death spiral*。

**dwell-guard 是一个透明代理**：应用把 base_url 指向它，它自动学习你的使用习惯并调度驻留——**对上层应用完全无感**。

## 为什么不用 Ollama 自带 / llama-swap？

| | Ollama | llama-swap | **dwell-guard** |
|---|---|---|---|
| 驻留管理 | 手动 `keep_alive` | 手动规则（YAML 写死） | **自动学习使用模式** |
| 预测下一个模型 | ❌ | ❌ | ✅ 转移概率预测 |
| 预加载 | ❌（open [request #209](https://github.com/mostlygeek/llama-swap/discussions/209)） | ❌ | ✅ 请求后 + 空闲期双预载 |
| 驱逐决策 | LRU 换出 | 规则 | **成本感知**（size/freq²/predicted²） |
| 空闲利用 | 闲置不做事 | 闲置不做事 | ✅ 空闲时自动准备 |

## 实测数据（RTX 5070 Laptop 8GB）

| 场景 | 纯冷启动 | **dwell-guard 后** | 提速 |
|---|---|---|---|
| 切回 30B 主力（0.6b 后用） | 61.5s | **3.2s** | **19×** |
| 3 模型场景（35B+30B+0.6b）切主力 | 82s | **9.3s** | **8.8×** |

## 特性

- 🧠 **自动学习**：记录每次调用 → 转移矩阵 → 预测"你 A 之后大概率用 B"
- ⚡ **双时机预载**：请求响应后延迟预载 + **空闲 15s 静默期预载**（用户在休息，无竞争）
- 🎯 **成本感知驱逐**：预载大模型前驱逐低价值模型（公式 `size/(freq+1)²/(predicted+1)²`——平方惩罚保护主力模型）
- 🔌 **透明代理**：OpenAI 兼容，应用零改动
- 💾 **跨重启记忆**：使用历史存 `history.json`

## 快速开始

```bash
# 1. 启动（默认 127.0.0.1:11500，转发本机 Ollama）
python dwell_guard.py

# 2. 把应用指向本守护（OpenAI 兼容）：
#    Open WebUI / 任何 OpenAI SDK / Agent：base_url = http://127.0.0.1:11500
```

参数：

```bash
python dwell_guard.py --port 11500 --ollama http://127.0.0.1:11434 --state history.json
```

> 需要 Ollama 运行中。模型目录通过 `OLLAMA_MODELS` 环境变量定位（与 Ollama 一致）。

## 架构

```
用户/Agent ──→ [dwell-guard :11500] ──转发──→ Ollama :11434
                  │  记录每次调用（history.json 跨重启）
                  │  转移概率预测"下一个模型"
                  │  预载预测目标 / 驱逐低价值模型腾空间
                  │  空闲期自动预载（keep 15m）
                  ▼
               调度动作（后台异步，不阻塞请求）
```

**关键机制**：Ollama 无独立 load API——预载 = 发一个 `num_predict=1` 的最小请求让模型载入内存，配 `keep_alive` 控制驻留时长。

## 支持的端点

| 端点 | 说明 |
|---|---|
| `/api/chat` | Ollama 对话（流式 + 非流式） |
| `/v1/chat/completions` | OpenAI 兼容对话 |
| `/api/generate` | Ollama 文本补全（流式 NDJSON + 非流式） |
| `/v1/completions` | OpenAI 兼容补全 |
| `/api/ps` (GET) | 查看驻留状态 |

所有 POST 端点都记录模型 → 触发调度。

## 调度策略

| 情形 | 动作 |
|---|---|
| 模型调用 ≥3 次（高频） | 驻留 30m |
| 预测的"下一个"模型 | 预载 + 驻留 10m |
| 用户空闲 ≥15s | 空闲预载预测目标（驻留 15m） |
| 预载大模型需空间 | 驱逐低价值模型（成本感知） |

## 状态文件（`history.json`）

```json
{
  "calls": {"qwen3:8b": 5, "qwen3:30b": 2},
  "transitions": {"qwen3:8b->qwen3:30b": 2},
  "last": "qwen3:30b"
}
```

## 测试

```bash
# 单元测试（纯逻辑，无需 Ollama/模型）
python test_dwell_guard_unit.py    # 记录/转移/预测/高频
python test_eviction.py            # 驱逐决策（成本感知）
python test_idle_prefetch.py       # 空闲预载（触发/去重/重置）

# 集成测试（需 Ollama + 已 pull 的模型；含本机路径，按需修改）
python test_generate.py            # /api/generate 透传
python test_3models.py             # 3 模型场景
python bench_switch.py before      # 切换基准（直连）
```

## 文档

更详细的开发过程与设计笔记见项目文档（中文）。

## License

[MIT](LICENSE)
