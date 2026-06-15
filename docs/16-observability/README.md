# 第16章 可观测与调试（Tracing、日志、成本追踪）

> **「任务助手 Agent」装上黑匣子了**——第12章的 Observer 组件，本章给它接上真正的"飞行记录仪"。
> 你将学会用 **Tracing（链路追踪）** 记录每一步 LLM 调用和工具调用、用**结构化日志**
> 让机器可读、用**成本追踪**算清每次查询花了多少钱。学完本章，你的 Agent 不再是一个
> "出了 bug 不知道为什么"的黑盒，而是一个每一步都可追溯、可复盘、可优化的透明系统。

---

## TL;DR

> **30 秒速读**：给 Agent 装上"飞行记录仪"——用 TraceCollector 记录每一步 LLM 调用和工具调用的树状链路，用 CostTracker 按 token × 单价算清每次查询花了多少钱，用 ASCII 树在终端可视化。
> 
> **如果只记一件事**：没有 trace 的 Agent 不该上生产，因为你无法回答"它为什么这么做"。

---

## 本章目标

学完本章，你将能够：

1. **实现 Tracing（链路追踪）**：用 `TraceCollector` 记录 Agent 每一步（step → LLM 调用 → 工具调用）成树状结构
2. **写结构化日志**：用 JSON 格式记录每条 trace entry，让机器可解析、可检索、可聚合
3. **追踪成本与延迟**：按 `prompt_tokens × 输入价 + completion_tokens × 输出价` 算每步花了多少钱、多久
4. **做 trace 可视化**：用 ASCII 树打印 step → LLM call → tool call 的层级关系
5. **理解 LangSmith / Langfuse 做了什么**：它们不是魔法，就是把本章的 trace 上传到一个带 UI 的平台
6. **用可观测数据定位问题**：成本爆炸、延迟突增、工具失败——trace 一眼看出根因

> ⚠️ **前置条件**：先学第12章（Observer 组件接口）。本章是实现第12章 Observer 接口的
> `TracingObserver` + `CostTracker`——你不理解 Observer 是"纯旁路观察"，就无法理解 tracing。

---

## 为什么 Agent 需要可观测（黑盒难调试）

### 痛点：Agent 是一个多步、非确定、异步的黑盒

传统软件出 bug，你打个断点、看堆栈、读日志，10 分钟定位。Agent 出 bug，你面对的是：

```
用户问："帮我查下北京和上海的天气，对比哪里更热"
  ↓
Agent 内部发生了什么？
  - 第 1 步：调 LLM？还是直接调工具？
  - 第 2 步：调了 get_weather(北京) 还是 get_weather("Beijing")？
  - 第 3 步：LLM 返回的 tool_calls 里参数对不对？
  - 第 4 步：工具返回了什么？天气数据是 25°C 还是 "晴"？
  - 第 5 步：为什么 Agent 又重复调了一次 get_weather？
  - 第 6 步：最终答案是"上海更热"——但模型是怎么得出这个结论的？
```

**没有 tracing，你只能看到"输入"和"输出"，中间 6 步全是黑盒**。你猜不出：
- 为什么这次查询花了 5 块钱（平时只要 2 毛）？
- 为什么这次响应慢了 8 秒（平时只要 1 秒）？
- 为什么模型给了一个明显错误的答案（它推理链路哪一步错了）？

### 三个真实的调试场景

**场景 1：成本爆炸**
> 运营说："今天 API 账单 500 美元，平时只要 50 美元，怎么查？"

没有成本追踪：你只能干瞪眼。有了 tracing：拉出当天所有 trace，按成本降序排列，
发现 80% 的钱花在一个"反复调 get_weather 7 次"的请求上——那是 max_steps 保险丝没生效。

**场景 2：延迟突增**
> 用户说："你的 Agent 卡了 30 秒才回复，退钱！"

没有延迟追踪：你不知道卡在哪。有了 tracing：看 trace 树，发现 LLM 调用只花了 2 秒，
但 `search_web` 工具调了 25 秒——是外部搜索 API 慢，不是你的 Agent。

**场景 3：答案错误**
> 测试说："Agent 说北京 25°C、上海 28°C，但实际北京 30°C、上海 26°C，反了！"

没有 trace：你只能重跑，但非确定性让你很难复现。有了 tracing：看 trace 里工具返回的原始数据，
发现 `get_weather("北京")` 返回的确实是 25°C——工具的数据源就是错的，不是 Agent 推理错。

> 💡 **核心理念**：可观测性不是"出 bug 了才加日志"，而是**默认开启的飞行记录仪**。
> 像 OpenTelemetry 一样，trace 是系统的第一公民，每一次请求都自带完整链路。

---

## Tracing（链路追踪）

### 什么是 Trace

**Trace = 一次请求的完整执行路径**，由多个 **Span（跨度）** 组成树状结构：

```
Trace: 用户问"查北京和上海天气"
│
├── Span: step_1 (Agent 第 1 步)
│   ├── Span: llm_call (调 LLM 决策)
│   │   ├── 输入: 8 messages, 320 tokens
│   │   └── 输出: tool_calls=[get_weather("北京")], 45 tokens
│   └── Span: tool_call: get_weather("北京")
│       └── 结果: "北京今天晴 25°C", 耗时 120ms
│
├── Span: step_2 (Agent 第 2 步)
│   ├── Span: llm_call
│   │   └── 输出: tool_calls=[get_weather("上海")], 48 tokens
│   └── Span: tool_call: get_weather("上海")
│       └── 结果: "上海今天多云 28°C", 耗时 95ms
│
└── Span: step_3 (Agent 第 3 步)
    └── Span: llm_call (最终回答，无 tool_calls)
        └── 输出: "上海更热(28°C > 25°C)", 80 tokens
```

**三个核心概念**（借鉴 OpenTelemetry）：
- **Span**：一个操作单元（LLM 调用 / 工具调用 / 一个 step）
- **Parent/Child**：Span 有父子关系，形成树（step 是父，llm_call 和 tool_call 是子）
- **SpanContext**：每个 span 有 trace_id（同一次请求共享）、span_id（唯一）、parent_id（指向上游）

### TraceEntry：每个 Span 记录什么

```python
@dataclass
class TraceEntry:
    """一个 trace span（Agent 执行的一个原子操作）。"""
    span_id: str           # 唯一 ID（如 "span_001"）
    trace_id: str          # 同一次请求共享（如 "trace_abc"）
    parent_id: str | None  # 父 span ID（顶层 span 为 None）
    span_type: str         # "step" | "llm_call" | "tool_call" | "tool_result"
    name: str              # 人读名字（如 "get_weather" / "LLM 决策"）
    start_time: float      # 开始时间戳（秒）
    end_time: float        # 结束时间戳（秒）
    input_summary: str     # 输入摘要（如 "320 tokens, 8 messages"）
    output_summary: str    # 输出摘要（如 "tool_calls=[get_weather], 45 tokens"）
    prompt_tokens: int     # LLM 输入 token（工具调用为 0）
    completion_tokens: int # LLM 输出 token（工具调用为 0）
    metadata: dict         # 额外数据（如工具参数、错误信息）
```

### TraceCollector：收集 + 渲染

`TraceCollector` 是 Observer 的实现（第12章 Observer 接口），它：

1. **收集**：Agent 循环在每个关键点（`on_step_start` / `on_llm_call` / `on_tool_call`）调 TraceCollector，它创建一个 TraceEntry 并存到内部列表
2. **渲染**：循环结束后，调 `render_tree()` 把扁平的 TraceEntry 列表渲染成 ASCII 树（见下方"trace 可视化"）
3. **导出**：调 `to_json()` 导出结构化日志（见下方"结构化日志"）

> 💡 **TraceCollector 实现了第12章 Observer 接口**——这是"接口先行"的价值。第12章你只定义了
> `Observer` 的 5 个钩子方法（`on_step_start` / `on_llm_call` / `on_tool_call` / `on_tool_result` / `on_step_end`），
> 本章给这些钩子填上"创建 span"的具体逻辑。换一个 Observer 实现（如 `MetricsObserver`），
> Agent 循环代码一行都不用改——这就是**控制反转**。

---

## 结构化日志（JSON 格式）

### 为什么不用 print

```python
# ❌ 坏：print 是给人看的，机器难解析
print(f"step 1: called get_weather for 北京, got 25°C, took 120ms")
# 运营想统计"平均工具耗时" → 得写正则从这堆字符串里抠数字

# ✓ 好：结构化日志是给机器看的，一行 JSON = 一条事件
import json
print(json.dumps({
    "ts": "2026-06-14T10:30:00Z",
    "level": "INFO",
    "span_id": "span_001",
    "event": "tool_call",
    "tool": "get_weather",
    "args": {"city": "北京"},
    "duration_ms": 120,
    "result": "北京今天晴 25°C",
}, ensure_ascii=False))
# 运营想统计"平均工具耗时" → grep "tool_call" | jq '.duration_ms' | avg
```

### 结构化日志的四要素

| 字段 | 作用 | 示例 |
|------|------|------|
| `ts`（timestamp） | 时间戳，排序和时序分析 | `"2026-06-14T10:30:00.123Z"` |
| `level` | 日志级别（DEBUG/INFO/WARN/ERROR） | `"INFO"` |
| `event` | 事件类型（step/llm/tool） | `"tool_call"` |
| `payload` | 结构化数据（字典） | `{"tool": "get_weather", ...}` |

**关键**：`payload` 必须是**结构化的**（dict / 字段固定）。`"result": "北京今天晴 25°C"` 可以，`"result": "工具返回了北京天气，是晴天25度"` 不行——后者机器解析不了。

### 本章的结构化日志实现

每个 `TraceEntry` 都可以序列化成一行 JSON 日志：

```python
def to_log_line(self) -> str:
    """把 TraceEntry 序列化成一行 JSON 日志。"""
    return json.dumps({
        "ts": datetime.fromtimestamp(self.start_time).isoformat() + "Z",
        "level": "INFO",
        "trace_id": self.trace_id,
        "span_id": self.span_id,
        "parent_id": self.parent_id,
        "event": self.span_type,
        "name": self.name,
        "duration_ms": round((self.end_time - self.start_time) * 1000, 2),
        "prompt_tokens": self.prompt_tokens,
        "completion_tokens": self.completion_tokens,
        "input": self.input_summary,
        "output": self.output_summary,
    }, ensure_ascii=False)
```

这样导出的日志可以被任何日志分析工具（ELK / Loki / Datadog）直接消费——不需要写正则解析。

---

## 成本与延迟追踪（token × 单价）

### 定价模型

LLM API 按 token 计费，**输入和输出价格不同**（输出更贵，因为生成比理解更耗算力）：

| 模型 | 输入价（$/1M tokens） | 输出价（$/1M tokens） |
|------|---------------------|----------------------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |
| deepseek-chat | $0.14 | $0.28 |
| qwen-plus | $0.40 | $1.20 |

> 本章用 gpt-4o-mini 的价格做演示。生产中你的 `CostTracker` 应该支持多模型定价表。

### 成本计算公式

```
单次成本 = (prompt_tokens × 输入价 + completion_tokens × 输出价) / 1,000,000

例：gpt-4o-mini，prompt=320 tokens, completion=45 tokens
  输入成本 = 320 × 0.15 / 1,000,000 = $0.0000480
  输出成本 =  45 × 0.60 / 1,000,000 = $0.0000270
  总成本 = $0.0000750（约 0.0075 美分）
```

看起来很少？但 Agent 一轮循环可能调 5-10 次 LLM，每天上千次查询，月账单轻松到几百美元——不追踪成本，钱包会被悄无声息掏空。

### CostTracker 实现

```python
class CostTracker:
    """累计一个 trace 内所有 LLM 调用的成本。"""

    # gpt-4o-mini 定价（演示用，生产应支持多模型定价表）
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.entries: list[dict] = []  # 每次 LLM 调用的明细

    def add_llm_call(self, prompt_tokens: int, completion_tokens: int) -> float:
        """记录一次 LLM 调用，返回本次成本。"""
        price = self.PRICING[self.model]
        input_cost = prompt_tokens * price["input"] / 1_000_000
        output_cost = completion_tokens * price["output"] / 1_000_000
        cost = input_cost + output_cost

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost
        self.entries.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        })
        return cost

    def summary(self) -> str:
        """生成成本摘要。"""
        return (
            f"模型: {self.model}\n"
            f"LLM 调用次数: {len(self.entries)}\n"
            f"总输入 tokens: {self.total_prompt_tokens}\n"
            f"总输出 tokens: {self.total_completion_tokens}\n"
            f"总成本: ${self.total_cost_usd:.6f}"
        )
```

### 延迟追踪

每个 span 记录 `start_time` 和 `end_time`，差值就是耗时（见上方 `TraceEntry` 定义）。

**关键洞察**：Agent 循环是**串行**的（调 LLM → 等结果 → 调工具 → 等结果），总延迟是所有 span 耗时之和。如果某步特别慢，trace 树会一眼高亮——这是定位 P99 延迟的首选手段。

---

## 简易 Trace 可视化（ASCII 树）

### 为什么用 ASCII 树

生产环境用 LangSmith / Langfuse 的 Web UI，但教学/本地调试时 ASCII 树最直接——不依赖任何服务，终端直接打印。

### 渲染算法

把扁平的 `TraceEntry` 列表（每个有 `parent_id`）渲染成树：

```python
def render_tree(self) -> str:
    """把 TraceEntry 列表渲染成 ASCII 树。"""
    lines = []
    lines.append(f"Trace: {self.trace_id} ({len(self.entries)} spans)")
    lines.append("│")
    # 找出顶层 span（parent_id 为 None）
    roots = [e for e in self.entries if e.parent_id is None]
    for root in roots:
        self._render_span(root, prefix="", is_last=True, lines=lines)
    return "\n".join(lines)

def _render_span(self, entry, prefix, is_last, lines):
    """递归渲染一个 span 及其子 span。"""
    connector = "└── " if is_last else "├── "
    duration_ms = (entry.end_time - entry.start_time) * 1000
    # 标注 token 和成本（如果是 llm_call）
    token_info = ""
    if entry.span_type == "llm_call":
        token_info = f" [in:{entry.prompt_tokens} out:{entry.completion_tokens}]"
    lines.append(f"{prefix}{connector}{entry.span_type}: {entry.name} "
                 f"({duration_ms:.0f}ms){token_info}")
    # 递归子 span
    children = [e for e in self.entries if e.parent_id == entry.span_id]
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(children):
        self._render_span(child, child_prefix, i == len(children) - 1, lines)
```

### 输出效果

```
Trace: trace_demo (8 spans)
│
└── step: 第1步 (510ms) [in:320 out:45]
    ├── llm_call: LLM决策 (390ms) [in:320 out:45]
    └── tool_call: get_weather (120ms)
        └── tool_result: 北京今天晴25°C (0ms)
```

一眼看出：第 1 步花了 510ms，其中 LLM 决策 390ms（76%）、工具调用 120ms（24%）——瓶颈在 LLM 而非工具。

---

## 对比 LangSmith / Langfuse（点到为止）

> ⚠️ **本章不接 LangSmith / Langfuse 实际服务**——教学用自实现，理解原理。
> 生产中你直接用这两个平台，底层逻辑和本章一样。

### 它们做了什么

LangSmith（LangChain 出品）和 Langfuse（开源）都是 **LLM 可观测平台**，核心功能：

| 功能 | 本章自实现 | LangSmith / Langfuse |
|------|-----------|---------------------|
| Trace 收集 | `TraceCollector`（内存） | SDK 自动收集（上传到平台） |
| Trace 存储 | 内存列表 / JSON 文件 | 时序数据库（可查几个月） |
| Trace 可视化 | ASCII 树（终端） | 交互式 Web UI（折叠/展开/筛选） |
| 成本统计 | `CostTracker`（手动） | 自动按模型定价表算 |
| 延迟分析 | span 耗时差值 | P50/P95/P99 分位数图 |
| 检索/过滤 | 手动遍历 | SQL/MongoDB 查询 |
| 告警 | 无 | 成本/延迟超阈值自动告警 |

### 接入方式（伪代码，仅展示概念）

```python
# LangSmith（概念，不实际接入）
from langsmith import traceable

@traceable  # ← 一个装饰器，自动把函数内的 LLM 调用记录成 trace
def my_agent(task: str) -> str:
    resp = client.chat.completions.create(...)  # 自动被追踪
    return resp.choices[0].message.content

# Langfuse（概念，不实际接入）
from langfuse import Langfuse
langfuse = Langfuse()
with langfuse.start_as_current_span("agent_run") as span:
    span.update_input({"task": task})
    resp = client.chat.completions.create(...)
    span.update_output({"answer": answer})
```

**核心洞察**：LangSmith / Langfuse 的 SDK 本质上就是**自动版 `TraceCollector`**——在你的 LLM 调用前后自动插桩，收集本章手写的那些 span。理解了 `TraceEntry` 结构和树渲染，打开 LangSmith 的 trace UI 你会觉得"原来是这么回事"。

### 本教程为什么不接

1. **教学原则**："先原理后工具"。自实现 trace 看清每一行代码在做什么，接 LangSmith 只看到一个装饰器。本章代码 100% 离线可跑，不依赖外部服务
2. **迁移成本低**：理解原理后，接 LangSmith 就是把 `TraceCollector.add_entry()` 换成 `langfuse.span()`，5 分钟的事

---

## 如何用可观测数据定位问题

### 调试工作流（trace 驱动）

当 Agent 行为异常（成本高 / 慢 / 错），**永远先看 trace**：

```
1. 拉出异常请求的 trace
   ↓
2. 看 trace 树的哪一步"异常"
   - 成本异常？看哪个 llm_call 的 token 特别多 → 是不是 prompt 膨胀？
   - 延迟异常？看哪个 span 耗时占比最高 → 是 LLM 慢还是工具慢？
   - 结果错误？看 tool_result 的原始数据 → 数据源错了还是解析错了？
   ↓
3. 定位到具体 span → 看 input/output summary → 找根因
   ↓
4. 修复（改 prompt / 改工具 / 加缓存）→ 重跑 trace 验证
```

### 三个实战案例（用 trace 定位）

**案例 A：成本突然翻倍**
```
看 trace → 发现 step_3 的 llm_call 输入了 5000 tokens（平时只有 800）
→ 看输入摘要 → "messages 包含了 15 轮历史对话"
→ 根因：Memory 没做截断，上下文无限增长（第05/11章的问题）
→ 修复：加 TokenBudget（第11章）或 ConversationBuffer 截断
```

**案例 B：P99 延迟从 2s 飙到 15s**
```
看 trace → 发现 step_2 的 tool_call: search_web 耗时 12s（平时 0.5s）
→ 根因：外部搜索 API 限流/故障
→ 修复：加超时（第06章）+ 降级策略（超时就返回"搜索暂不可用"）
```

**案例 C：Agent 给了错误答案**
```
看 trace → step_1 tool_result 显示 get_weather("北京")="25°C"
→ 但用户说北京实际是 30°C
→ 根因：工具的数据源（天气 API）返回的本来就是 25°C → 工具 bug 不是 Agent bug
→ 修复：换天气数据源（工具层问题），Agent 推理逻辑没问题
```

> 💡 **trace 是"事后取证"的金标准**。传统调试靠复现，但 Agent 是非确定性的，复现极难。
> trace 把每次请求的完整链路持久化，出 bug 后回放 trace 就能定位——不需要复现。

---

## 反模式（什么不该做）

### ❌ 只 print 不结构化

```python
# ❌ 坏：print 自由文本，机器无法解析
print(f"step {step}: llm returned {len(content)} chars")

# 问题：
# - 运营想统计"平均返回长度" → 得写正则从日志里抠数字
# - 日志格式一变，所有下游脚本全崩
# - 无法聚合（"今天有多少 step 返回超过 500 字？" → 要写复杂 awk）

# ✓ 好：结构化日志，字段固定
logger.info("step_completed", extra={
    "step": step, "content_length": len(content), "span_id": span_id
})
```

**后果**：日志只能人眼读，无法做数据分析。Agent 上量后面对几十万行 print 日志，想统计什么都统计不了。

**原则**：**日志 = 数据**。任何你想事后分析的指标（耗时、token、工具名、成功/失败），
都必须是结构化字段，不能埋在自由文本里。

### ❌ 不追踪成本（"反正 API 便宜"）

```python
# ❌ 坏：不记录 token，月底看账单才发现超了
resp = client.chat.completions.create(...)
answer = resp.choices[0].message.content
# usage 里的 prompt_tokens / completion_tokens 直接扔了

# ✓ 好：每次 LLM 调用都记进 CostTracker
tracker.add_llm_call(
    prompt_tokens=resp.usage.prompt_tokens,
    completion_tokens=resp.usage.completion_tokens,
)
```

**后果**：
- 开发期"每次调用几毛钱"不心疼 → 上量后月账单 $10,000 → 老板震怒
- 无法做成本归因（哪个用户/功能烧的钱？），更无法做预算控制

**原则**：**成本追踪从第一天就要有**。不要等账单爆炸才补——那时的 trace 已经过期了，你无法回溯。

### ❌ Trace 存内存不持久化

```python
# ❌ 坏：trace 存在 collector._entries 列表里，进程一退出全没了
collector = TraceCollector()
agent.run(task)  # trace 进了内存
# 进程结束 → trace 丢失 → 出 bug 想回放 → 无 trace 可看

# ✓ 好：trace 持久化（文件 / 数据库 / 上传平台）
collector.to_json_file(f"traces/{request_id}.json")
# 或上传到 LangSmith / Langfuse
```

**后果**：生产环境的 trace 必须持久化（至少存 7-30 天），否则"事后取证"无从谈起。

**原则**：**生产 trace 必须落盘**。本章为了离线演示存内存，生产环境必须写到文件 / 数据库 / 平台。

### ❌ Observer 有副作用（违反第12章纯旁路原则）

```python
# ❌ 坏：TracingObserver 修改了主流程状态
class BadTracingObserver:
    def on_tool_result(self, name, result):
        if "error" in result:
            self.agent.memory.add("system", "工具失败，请重试")  # ❌ 改了 Memory！

# ✓ 好：TracingObserver 只记录，不干预
class GoodTracingObserver:
    def on_tool_result(self, name, result):
        self.add_entry(TraceEntry(...))  # ✅ 只记 trace
        # 如需干预，抛异常或返回中断信号，让主流程决策
```

**后果**：Observer 变成"隐藏的第二主流程"，状态被谁改的完全不可追溯——
这正是你需要 trace 来调试的问题，结果 trace 系统自己就是这个问题的来源。

**原则**：**Observer 纯旁路，只读不写**（第12章已强调）。TracingObserver 记录的是"发生了什么"，
不是"应该发生什么"。如需干预（如护栏），那是 `GuardrailObserver`（第17章）的职责。

### ❌ Trace 粒度太粗或太细

```python
# ❌ 太粗：整个 agent.run 只记一条 trace
trace.add("agent_completed", duration=5.0)  # 出 bug 不知道是哪步

# ❌ 太细：每个 JSON.parse、每个字符串拼接都记 span
trace.add("json_parse_start")  # 噪声淹没信号，trace 树深 20 层没法看

# ✓ 合适：step / llm_call / tool_call / tool_result 四级
trace.add_span("step", step=1)           # 顶层
trace.add_span("llm_call", tokens=320)   # step 的子
trace.add_span("tool_call", name=...)    # step 的子
trace.add_span("tool_result", ...)       # tool_call 的子
```

**后果**：太粗 → trace 没用（看不出哪一步错）；太细 → trace 不可读（信号被噪声淹没）。

**原则**：**四级粒度足够**（step → llm_call → tool_call → tool_result）。这是 OpenTelemetry
和 LangSmith 的事实标准粒度，不要自创更细的。

---

## 常见错误

> 概念懂了，实际写代码还是会踩坑。

| 错误 | 症状 | 解决 |
|------|------|------|
| trace 只存内存不落盘 | 进程重启后历史 trace 全丢，出 bug 无法回放 | 每次请求结束后 `to_json_file()` 写磁盘，生产上传平台 |
| 不记录 `usage` 里的 token 数 | 月底账单爆炸，不知道哪个请求烧的钱 | 每次 LLM 调用后立刻 `tracker.add_llm_call(prompt_tokens, completion_tokens)` |
| Observer 里改了 Memory 或 messages | Agent 行为莫名异常，trace 系统自己成了 bug 来源 | Observer 只读不写，需要干预时返回中断信号让主流程决策 |
| print 自由文本当日志 | 运营想统计平均工具耗时，得写正则从字符串里抠数字 | 用 `json.dumps()` 输出结构化日志，字段名固定 |
| trace 粒度太细（每个 JSON.parse 都记） | trace 树 20 层深，信号被噪声淹没 | 只记 step / llm_call / tool_call / tool_result 四级 |

---

## 本章代码说明

本章代码实现 3 大功能（Python + TypeScript 对等）：

| 文件 | 功能 |
|------|------|
| `python/main.py` | TraceCollector + CostTracker + ASCII 树可视化 + 离线 mock |
| `typescript/main.ts` | 对等实现（async 全链路） |
| `exercises/README.md` | 给第13章 Observer 组件加成本追踪 |

### 3 个 Demo

1. **Demo 1 — Tracing**：模拟一个 3 步 Agent（查北京天气 → 查上海天气 → 对比），用 `TraceCollector` 收集 trace，打印树状结构
2. **Demo 2 — 成本计算**：用 `CostTracker` 计算上述 trace 的多步总成本（按 gpt-4o-mini 定价）
3. **Demo 3 — Trace 可视化**：打印 ASCII 树，展示 step → llm_call → tool_call 层级

### 运行示例

```bash
cd ai-agent/16-observability
pip install -r python/requirements.txt
python3 python/main.py        # Python
npx tsx typescript/main.ts     # TypeScript
```

输出标记（便于 QA grep）：`OUT:trace:step{N}:`（trace entry）、`OUT:cost:`（成本）、`OUT:viz:`（ASCII 树）、`OUT:log:`（JSON 日志）

> 💡 **离线 mock**：本章 trace 数据全部是 mock 的（预设的 token 数和耗时），不依赖真实 API。
> try API 失败 → catch → mock fallback，exit 0。这样在无网络/无密钥环境下也能完整演示。

---

## 下一步

恭喜！你给「任务助手 Agent」装上了飞行记录仪。现在它每一步都可追溯、可复盘。

- **第15章 评估与测试**：trace 告诉你"发生了什么"，eval 告诉你"做得好不好"
- **第17章 安全护栏**：trace + 成本数据是护栏的决策依据（"成本超 $10 就熔断"）

学完 Part 6 的 3 章（评估 / 可观测 / 安全），你就有了把 Agent 推向生产的全部工具箱。

> 💡 **可观测性是生产化的底线**。一个没有 trace 的 Agent 不应该上生产——
> 因为你无法回答"它为什么这么做"，也就无法保证"它不会做错"。
> 本章教你的是这种"观测思维"：**默认记录一切，让透明成为习惯**。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
