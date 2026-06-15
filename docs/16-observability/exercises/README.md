# 第16章 练习 — 可观测与调试

> 动手给 Agent 装上飞行记录仪：Tracing、结构化日志、成本追踪、可视化仪表盘。
> 核心练习：**搭建一个完整的 Agent 可观测仪表盘**。

---

## 练习 1：给 Agent 循环加 Tracing（难度：★☆☆）

### 任务

本章的 `TraceCollector` 用 mock 数据演示了 trace 的收集和渲染。现在请你把它接入一个真实的 Agent 循环。

给第13章的 `DefaultAgentRunner` 加一个 `TracingObserver`，在每个关键节点创建 span：

1. `on_step_start` → 创建 `span_type="step"` 的 span
2. `on_llm_call` → 创建 `span_type="llm_call"` 的 span（parent 指向当前 step）
3. `on_tool_call` → 创建 `span_type="tool_call"` 的 span（parent 指向当前 step）
4. `on_tool_result` → 创建 `span_type="tool_result"` 的 span（parent 指向对应的 tool_call）
5. `on_step_end` → 关闭当前 step 的 span（设置 `end_time`）

运行后，调用 `render_tree()` 打印出完整的 trace 树。

### 参考答案

```python
import time
import uuid
from dataclasses import dataclass, field

@dataclass
class TraceEntry:
    span_id: str
    trace_id: str
    parent_id: str | None
    span_type: str
    name: str
    start_time: float
    end_time: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict = field(default_factory=dict)

class TracingObserver:
    """Observer 实现：收集 trace span，形成树状结构。"""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex[:8]}"
        self.entries: list[TraceEntry] = []
        self._current_step_span_id: str | None = None
        self._current_llm_span_id: str | None = None
        self._span_counter = 0

    def _next_span_id(self) -> str:
        self._span_counter += 1
        return f"span_{self._span_counter:03d}"

    def on_step_start(self, step: int, messages: list) -> None:
        span_id = self._next_span_id()
        self._current_step_span_id = span_id
        self.entries.append(TraceEntry(
            span_id=span_id,
            trace_id=self.trace_id,
            parent_id=None,  # 顶层 span
            span_type="step",
            name=f"第{step + 1}步",
            start_time=time.time(),
            input_summary=f"{len(messages)} 条消息",
        ))

    def on_llm_call(self, messages: list) -> None:
        span_id = self._next_span_id()
        self._current_llm_span_id = span_id
        self.entries.append(TraceEntry(
            span_id=span_id,
            trace_id=self.trace_id,
            parent_id=self._current_step_span_id,
            span_type="llm_call",
            name="LLM 决策",
            start_time=time.time(),
            input_summary=f"{len(messages)} 条消息",
        ))

    def on_tool_call(self, name: str, args: dict) -> None:
        span_id = self._next_span_id()
        self.entries.append(TraceEntry(
            span_id=span_id,
            trace_id=self.trace_id,
            parent_id=self._current_step_span_id,
            span_type="tool_call",
            name=name,
            start_time=time.time(),
            input_summary=str(args),
        ))

    def on_tool_result(self, name: str, result: str) -> None:
        # 找到对应的 tool_call span，补上 end_time
        for entry in reversed(self.entries):
            if entry.span_type == "tool_call" and entry.name == name and entry.end_time == 0.0:
                entry.end_time = time.time()
                entry.output_summary = result[:100]
                break
        # 记录 tool_result span
        self.entries.append(TraceEntry(
            span_id=self._next_span_id(),
            trace_id=self.trace_id,
            parent_id=self._current_step_span_id,
            span_type="tool_result",
            name=result[:50],
            start_time=time.time(),
            end_time=time.time(),
        ))

    def on_step_end(self, step: int) -> None:
        # 关闭 step span 和 llm_call span
        now = time.time()
        for entry in reversed(self.entries):
            if entry.end_time == 0.0:
                entry.end_time = now

    def render_tree(self) -> str:
        lines = [f"Trace: {self.trace_id} ({len(self.entries)} spans)", "│"]
        roots = [e for e in self.entries if e.parent_id is None]
        for root in roots:
            self._render_span(root, prefix="", is_last=True, lines=lines)
        return "\n".join(lines)

    def _render_span(self, entry, prefix, is_last, lines):
        connector = "└── " if is_last else "├── "
        duration_ms = (entry.end_time - entry.start_time) * 1000
        token_info = ""
        if entry.span_type == "llm_call" and entry.prompt_tokens > 0:
            token_info = f" [in:{entry.prompt_tokens} out:{entry.completion_tokens}]"
        lines.append(f"{prefix}{connector}{entry.span_type}: {entry.name} "
                     f"({duration_ms:.0f}ms){token_info}")
        children = [e for e in self.entries if e.parent_id == entry.span_id]
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            self._render_span(child, child_prefix, i == len(children) - 1, lines)
```

### 验证

把 `TracingObserver` 注入第13章的 `DefaultAgentRunner`，运行后检查：

1. 每个 step 有对应的 `step` span（顶层，`parent_id=None`）
2. 每个 step 下面挂了 `llm_call` 和 `tool_call` 子 span
3. `render_tree()` 输出的树结构层级正确
4. 每个 span 的 `end_time > start_time`（没有未关闭的 span）

**思考**：为什么 `on_tool_result` 不创建新的子 span，而是回去更新 `tool_call` 的 `end_time`？
（提示：tool_call 和 tool_result 是同一个操作的"开始"和"结束"，不是一个父子关系）

---

## 练习 2：结构化日志 + 关联 ID（难度：★★☆）

### 任务

本章的 `to_log_line()` 把每个 `TraceEntry` 序列化成一行 JSON。但它缺少一个关键字段：
**关联 ID（correlation_id）**——把同一用户会话的多次请求串起来。

请扩展日志格式，加入：

1. **correlation_id**：同一用户会话共享的 ID（跨多次 Agent 请求）
2. **user_id**：发起请求的用户标识
3. **session_id**：会话 ID（可选，用于多轮对话场景）
4. **status**：span 的状态（`ok` / `error` / `timeout`）

然后实现一个 `StructuredLogger`，它接收 `TraceEntry`，输出带关联 ID 的 JSON 日志行。

### 参考答案

```python
import json
from datetime import datetime, timezone

@dataclass
class LogContext:
    """请求级上下文，注入到 logger 中。"""
    correlation_id: str       # 跨请求关联（同一用户会话共享）
    user_id: str              # 用户标识
    session_id: str = ""      # 会话 ID（可选）

class StructuredLogger:
    """把 TraceEntry 输出为带关联 ID 的 JSON 日志行。"""

    def __init__(self, context: LogContext) -> None:
        self.context = context

    def format_entry(self, entry: TraceEntry) -> str:
        duration_ms = round((entry.end_time - entry.start_time) * 1000, 2)
        status = "ok"
        if entry.metadata.get("error"):
            status = "error"
        elif duration_ms > 30_000:  # 超过 30 秒算 timeout
            status = "timeout"

        log_record = {
            "ts": datetime.fromtimestamp(entry.start_time, tz=timezone.utc).isoformat(),
            "level": "ERROR" if status == "error" else "INFO",
            "correlation_id": self.context.correlation_id,
            "user_id": self.context.user_id,
            "session_id": self.context.session_id,
            "trace_id": entry.trace_id,
            "span_id": entry.span_id,
            "parent_id": entry.parent_id,
            "event": entry.span_type,
            "name": entry.name,
            "status": status,
            "duration_ms": duration_ms,
            "prompt_tokens": entry.prompt_tokens,
            "completion_tokens": entry.completion_tokens,
            "input": entry.input_summary,
            "output": entry.output_summary,
        }
        return json.dumps(log_record, ensure_ascii=False)

    def log_trace(self, entries: list[TraceEntry]) -> list[str]:
        """把整个 trace 的所有 span 输出为日志行列表。"""
        return [self.format_entry(e) for e in entries]

# 使用示例
context = LogContext(
    correlation_id="sess_abc123",
    user_id="user_42",
    session_id="conv_789",
)
logger = StructuredLogger(context)
log_lines = logger.log_trace(tracing_observer.entries)

for line in log_lines:
    print(line)
# {"ts":"2026-06-14T10:30:00+00:00","level":"INFO","correlation_id":"sess_abc123",...}
```

### 验证

1. 每行日志是合法 JSON（`json.loads()` 不报错）
2. 所有日志行共享同一个 `correlation_id`
3. `status` 字段正确：正常 span 为 `ok`，含 error metadata 的为 `error`
4. 可以用 `jq` 按 `correlation_id` 过滤：`cat logs.jsonl | jq 'select(.correlation_id=="sess_abc123")'`

**思考**：为什么 `correlation_id` 和 `trace_id` 是两个不同的字段？
（提示：一次用户会话可能触发多次 Agent 请求，每次请求有自己的 `trace_id`，但共享同一个 `correlation_id`。前者定位单次请求，后者串联整个会话。）

---

## 练习 3：多模型成本追踪与预算告警（难度：★★☆）

### 任务

本章的 `CostTracker` 只支持单一模型定价。生产环境往往混用多个模型
（gpt-4o 做复杂推理，gpt-4o-mini 做简单任务）。请扩展它：

1. **多模型定价表**：支持 gpt-4o、gpt-4o-mini、deepseek-chat、qwen-plus
2. **按模型分别统计**：每个模型的 token 用量和成本独立汇总
3. **预算告警**：当累计成本超过阈值时，打印 ⚠️ 告警
4. **成本归因**：每次 LLM 调用记录关联的 step 编号，方便定位"哪一步最烧钱"

### 参考答案

```python
@dataclass
class CostEntry:
    step: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

class MultiModelCostTracker:
    """支持多模型定价 + 预算告警的成本追踪器。"""

    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "qwen-plus": {"input": 0.40, "output": 1.20},
    }

    def __init__(self, budget_usd: float = 1.0) -> None:
        self.budget_usd = budget_usd
        self.entries: list[CostEntry] = []
        self._warned = False

    def add_llm_call(self, step: int, model: str,
                     prompt_tokens: int, completion_tokens: int) -> float:
        price = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000

        self.entries.append(CostEntry(
            step=step, model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        ))

        # 预算告警
        if not self._warned and self.total_cost > self.budget_usd:
            print(f"⚠️ [COST] 累计成本 ${self.total_cost:.4f} 已超过预算 ${self.budget_usd:.2f}！")
            self._warned = True

        return cost

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def summary_by_model(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for e in self.entries:
            if e.model not in result:
                result[e.model] = {"calls": 0, "prompt_tokens": 0,
                                   "completion_tokens": 0, "cost_usd": 0.0}
            m = result[e.model]
            m["calls"] += 1
            m["prompt_tokens"] += e.prompt_tokens
            m["completion_tokens"] += e.completion_tokens
            m["cost_usd"] += e.cost_usd
        return result

    def most_expensive_step(self) -> CostEntry | None:
        """找出单次最贵的 LLM 调用。"""
        return max(self.entries, key=lambda e: e.cost_usd) if self.entries else None

    def print_report(self) -> None:
        print(f"OUT:cost: 总成本: ${self.total_cost:.6f} (预算: ${self.budget_usd:.2f})")
        for model, stats in self.summary_by_model().items():
            print(f"  {model}: {stats['calls']} 次调用, "
                  f"${stats['cost_usd']:.6f}")
        step = self.most_expensive_step()
        if step:
            print(f"  最贵单次: step {step.step}, {step.model}, ${step.cost_usd:.6f}")

# 使用
tracker = MultiModelCostTracker(budget_usd=0.50)
tracker.add_llm_call(step=0, model="gpt-4o-mini", prompt_tokens=320, completion_tokens=45)
tracker.add_llm_call(step=1, model="gpt-4o-mini", prompt_tokens=400, completion_tokens=48)
tracker.add_llm_call(step=2, model="gpt-4o", prompt_tokens=1200, completion_tokens=200)
tracker.print_report()
```

### 验证

1. gpt-4o 的单次成本远高于 gpt-4o-mini（检查 `summary_by_model()` 输出）
2. 设置低预算（如 `$0.001`），触发 ⚠️ 告警
3. `most_expensive_step()` 正确返回 cost 最大的那条记录
4. 总成本 = 各模型成本之和（四舍五入后一致）

**思考**：为什么"最贵单次调用"比"总成本"更有调试价值？
（提示：总成本高可能只是调用次数多，但最贵单次往往暴露了 prompt 膨胀或上下文泄漏的根因）

---

## 练习 4：搭建 Agent 可观测仪表盘（难度：★★★，核心练习）

### 场景

你的 Agent 已经上线一周，积累了上千条 trace。运营每天问你：

- "今天成本多少？比昨天涨了多少？"
- "P99 延迟是多少？有没有超过 10 秒的请求？"
- "哪个工具调用失败率最高？"
- "哪个用户消耗了最多 token？"

你需要一个**仪表盘**，能从 trace 日志中提取这些指标。

### 任务

实现一个 `ObservabilityDashboard`，它读取一批 `TraceEntry`，输出以下指标：

1. **成本面板**：总成本、按模型分组、日均成本
2. **延迟面板**：P50 / P95 / P99 延迟、最慢的 3 个请求
3. **工具面板**：每个工具的调用次数、失败率、平均耗时
4. **用户面板**：按用户 ID 分组的 token 消耗排行

### 参考答案

```python
import statistics
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class TraceRecord:
    """一条完整的请求记录（简化版，实际从 JSON 日志解析）。"""
    trace_id: str
    user_id: str
    model: str
    total_cost_usd: float
    total_duration_ms: float
    tool_calls: list[dict]  # [{"name": "get_weather", "duration_ms": 120, "success": True}]
    prompt_tokens: int
    completion_tokens: int

class ObservabilityDashboard:
    """从 trace 记录中提取可观测指标。"""

    def __init__(self, records: list[TraceRecord]) -> None:
        self.records = records

    # ---- 成本面板 ----

    def cost_panel(self) -> dict:
        total = sum(r.total_cost_usd for r in self.records)
        by_model: dict[str, float] = defaultdict(float)
        for r in self.records:
            by_model[r.model] += r.total_cost_usd
        return {
            "total_cost_usd": round(total, 6),
            "by_model": {m: round(c, 6) for m, c in by_model.items()},
            "avg_cost_per_request": round(total / max(len(self.records), 1), 6),
            "request_count": len(self.records),
        }

    # ---- 延迟面板 ----

    def latency_panel(self) -> dict:
        durations = sorted(r.total_duration_ms for r in self.records)
        if not durations:
            return {"p50": 0, "p95": 0, "p99": 0, "slowest": []}

        def percentile(data: list[float], p: float) -> float:
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])

        slowest_3 = sorted(self.records, key=lambda r: r.total_duration_ms, reverse=True)[:3]
        return {
            "p50_ms": round(percentile(durations, 50), 1),
            "p95_ms": round(percentile(durations, 95), 1),
            "p99_ms": round(percentile(durations, 99), 1),
            "slowest": [{"trace_id": r.trace_id, "ms": round(r.total_duration_ms, 1)}
                        for r in slowest_3],
        }

    # ---- 工具面板 ----

    def tool_panel(self) -> dict[str, dict]:
        stats: dict[str, dict] = defaultdict(lambda: {"calls": 0, "failures": 0,
                                                       "total_ms": 0.0})
        for r in self.records:
            for tc in r.tool_calls:
                s = stats[tc["name"]]
                s["calls"] += 1
                s["total_ms"] += tc["duration_ms"]
                if not tc.get("success", True):
                    s["failures"] += 1

        result = {}
        for name, s in stats.items():
            result[name] = {
                "calls": s["calls"],
                "failure_rate": round(s["failures"] / max(s["calls"], 1), 3),
                "avg_duration_ms": round(s["total_ms"] / max(s["calls"], 1), 1),
            }
        return result

    # ---- 用户面板 ----

    def user_panel(self, top_n: int = 5) -> list[dict]:
        by_user: dict[str, dict] = defaultdict(lambda: {"requests": 0,
                                                         "total_tokens": 0, "total_cost": 0.0})
        for r in self.records:
            u = by_user[r.user_id]
            u["requests"] += 1
            u["total_tokens"] += r.prompt_tokens + r.completion_tokens
            u["total_cost"] += r.total_cost_usd

        ranked = sorted(by_user.items(), key=lambda x: x[1]["total_cost"], reverse=True)
        return [{"user_id": uid, **{k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in stats.items()}}
                for uid, stats in ranked[:top_n]]

    # ---- 汇总打印 ----

    def print_dashboard(self) -> None:
        print("=" * 60)
        print("  Agent 可观测仪表盘")
        print("=" * 60)

        cost = self.cost_panel()
        print(f"\n📊 成本面板")
        print(f"  总成本: ${cost['total_cost_usd']}")
        print(f"  请求数: {cost['request_count']}")
        print(f"  均成本: ${cost['avg_cost_per_request']}/请求")
        for model, c in cost["by_model"].items():
            print(f"  {model}: ${c}")

        lat = self.latency_panel()
        print(f"\n⏱️ 延迟面板")
        print(f"  P50: {lat['p50_ms']}ms | P95: {lat['p95_ms']}ms | P99: {lat['p99_ms']}ms")
        for s in lat["slowest"]:
            print(f"  最慢: {s['trace_id']} ({s['ms']}ms)")

        tools = self.tool_panel()
        print(f"\n🔧 工具面板")
        for name, s in tools.items():
            fail_tag = f" ⚠️ 失败率{s['failure_rate']*100:.1f}%" if s["failure_rate"] > 0 else ""
            print(f"  {name}: {s['calls']}次, 均{s['avg_duration_ms']}ms{fail_tag}")

        users = self.user_panel()
        print(f"\n👤 用户面板 (Top {len(users)})")
        for u in users:
            print(f"  {u['user_id']}: {u['requests']}次, "
                  f"{u['total_tokens']} tokens, ${u['total_cost']}")

# 使用：从 trace 记录构建仪表盘
records = [
    TraceRecord("t1", "user_A", "gpt-4o-mini", 0.0005, 1200,
                [{"name": "get_weather", "duration_ms": 120, "success": True}], 320, 45),
    TraceRecord("t2", "user_B", "gpt-4o", 0.015, 8500,
                [{"name": "search_web", "duration_ms": 7000, "success": False},
                 {"name": "get_weather", "duration_ms": 100, "success": True}], 1200, 200),
    TraceRecord("t3", "user_A", "gpt-4o-mini", 0.0003, 800,
                [{"name": "calculate", "duration_ms": 5, "success": True}], 200, 30),
]
dashboard = ObservabilityDashboard(records)
dashboard.print_dashboard()
```

### 验证

1. 成本面板：总成本 = 各记录成本之和，按模型分组正确
2. 延迟面板：P99 >= P95 >= P50，最慢记录排序正确
3. 工具面板：`search_web` 的失败率 > 0，`get_weather` 的调用次数 = 2
4. 用户面板：`user_B` 排第一（因为用了 gpt-4o，成本最高）

**思考**：这个仪表盘是"离线批处理"模式（读取历史数据）。如果要改成"实时流式"模式
（每条 trace 进来就更新指标），你会怎么改造？（提示：把 `print_dashboard()` 改成
增量更新的 `ingest(record)` 方法，维护 running state）

---

## 总结

| 练习 | 核心技能 | 难度 |
|------|----------|------|
| 1 | 给 Agent 循环加 Tracing | ★☆☆ |
| 2 | 结构化日志 + 关联 ID | ★★☆ |
| 3 | 多模型成本追踪 + 预算告警 | ★★☆ |
| 4 | **Agent 可观测仪表盘（核心）** | ★★★ |

做完这些练习，你就掌握了 Agent 可观测的完整链路：**采集（Tracing）→ 存储（结构化日志）→ 分析（成本/延迟/工具面板）→ 告警（预算超限）**。

> 🔍 **记住本章的核心理念**：可观测性不是"出了 bug 再加日志"，而是默认开启的飞行记录仪。
> 一个没有 trace 的 Agent 不应该上生产——因为你无法回答"它为什么这么做"。
