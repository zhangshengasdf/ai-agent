"""
第16章 可观测与调试（Tracing、日志、成本追踪）— Python 实现

3 大功能：
  1. Tracing — TraceCollector 收集 TraceEntry，记录 Agent 每步（step/llm/tool）成树状结构
  2. 成本计算 — CostTracker 按 prompt_tokens×输入价 + completion_tokens×输出价 计算
  3. Trace 可视化 — ASCII 树渲染 step→llm_call→tool_call 层级

概念上实现第12章 Observer 接口（on_step_start/on_llm_call/on_tool_call/on_tool_result/on_step_end），
但独立可运行（不 import 第13章框架）。

离线 mock：模拟一个 3 步 Agent（查北京天气→查上海天气→对比），用 mock token 数据演示，
try 真实 API → catch → mock fallback，exit 0。

运行方式：
  cd ai-agent/16-observability
  python3 python/main.py
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# 让章节代码能 import shared.config（T1 确立的约定）
# ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import get_config  # noqa: E402

from openai import OpenAI  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# 数据结构：TraceEntry（一个 trace span = 一个原子操作）
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TraceEntry:
    """一个 trace span，记录 Agent 执行的一个原子操作。

    span_type 取值：
      - "step"         — Agent 循环的一步（顶层 span）
      - "llm_call"     — 一次 LLM 调用（step 的子 span）
      - "tool_call"    — 一次工具调用（step 的子 span）
      - "tool_result"  — 工具返回结果（tool_call 的子 span）
    """

    span_id: str
    trace_id: str
    parent_id: str | None
    span_type: str  # "step" | "llm_call" | "tool_call" | "tool_result"
    name: str
    start_time: float
    end_time: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """耗时（毫秒）。"""
        if self.end_time <= 0:
            return 0.0
        return round((self.end_time - self.start_time) * 1000, 2)

    def to_log_line(self) -> str:
        """序列化成一行 JSON 结构化日志（机器可解析）。"""
        return json.dumps({
            "ts": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            "level": "INFO",
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "event": self.span_type,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "input": self.input_summary,
            "output": self.output_summary,
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# TraceCollector：收集 + 渲染 trace（实现第12章 Observer 接口）
# ═══════════════════════════════════════════════════════════════════════

class TraceCollector:
    """收集 Agent 执行的每个 span，渲染成树状结构。

    实现了第12章 Observer 接口的 5 个钩子：
      on_step_start / on_llm_call / on_tool_call / on_tool_result / on_step_end

    设计原则（第12章）：纯旁路观察（只读不写），不修改主流程状态。
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id: str = trace_id or f"trace_{uuid.uuid4().hex[:8]}"
        self.entries: list[TraceEntry] = []
        self._current_step_id: str | None = None
        self._span_counter: int = 0

    def _next_span_id(self) -> str:
        """生成下一个 span ID。"""
        self._span_counter += 1
        return f"span_{self._span_counter:03d}"

    # ── 第12章 Observer 接口实现 ──────────────────────────────────────

    def on_step_start(self, step: int, messages: list[dict[str, Any]]) -> TraceEntry:
        """每步开始：创建顶层 step span。"""
        entry = TraceEntry(
            span_id=self._next_span_id(),
            trace_id=self.trace_id,
            parent_id=None,
            span_type="step",
            name=f"第{step}步",
            start_time=time.time(),
            input_summary=f"{len(messages)} messages",
            metadata={"step": step},
        )
        self.entries.append(entry)
        self._current_step_id = entry.span_id
        return entry

    def on_llm_call(
        self,
        messages: list[dict[str, Any]],
        prompt_tokens: int,
        completion_tokens: int,
        output_summary: str,
        duration_ms: float = 0.0,
    ) -> TraceEntry:
        """调 LLM：创建 llm_call span（step 的子）。"""
        start = time.time()
        entry = TraceEntry(
            span_id=self._next_span_id(),
            trace_id=self.trace_id,
            parent_id=self._current_step_id,
            span_type="llm_call",
            name="LLM决策",
            start_time=start - duration_ms / 1000,
            end_time=start,
            input_summary=f"{prompt_tokens} tokens, {len(messages)} messages",
            output_summary=output_summary,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.entries.append(entry)
        return entry

    def on_tool_call(self, name: str, args: dict[str, Any], duration_ms: float = 0.0) -> TraceEntry:
        """调工具：创建 tool_call span（step 的子）。"""
        start = time.time()
        entry = TraceEntry(
            span_id=self._next_span_id(),
            trace_id=self.trace_id,
            parent_id=self._current_step_id,
            span_type="tool_call",
            name=name,
            start_time=start - duration_ms / 1000,
            end_time=start,
            input_summary=json.dumps(args, ensure_ascii=False),
            metadata={"args": args},
        )
        self.entries.append(entry)
        return entry

    def on_tool_result(self, tool_call_span_id: str, result: str, duration_ms: float = 0.0) -> TraceEntry:
        """工具返回：创建 tool_result span（tool_call 的子）。"""
        start = time.time()
        entry = TraceEntry(
            span_id=self._next_span_id(),
            trace_id=self.trace_id,
            parent_id=tool_call_span_id,
            span_type="tool_result",
            name="返回结果",
            start_time=start - duration_ms / 1000,
            end_time=start,
            output_summary=result,
        )
        self.entries.append(entry)
        return entry

    def on_step_end(self, step: int) -> TraceEntry | None:
        """每步结束：关闭当前 step span。"""
        if self._current_step_id is None:
            return None
        for entry in reversed(self.entries):
            if entry.span_id == self._current_step_id:
                entry.end_time = time.time()
                break
        self._current_step_id = None
        return None

    # ── 手动添加 span（离线 mock 用）──────────────────────────────────

    def add_span(
        self,
        span_type: str,
        name: str,
        parent_id: str | None,
        start_time: float,
        end_time: float,
        input_summary: str = "",
        output_summary: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEntry:
        """手动添加一个 span（离线 mock trace 用）。"""
        entry = TraceEntry(
            span_id=self._next_span_id(),
            trace_id=self.trace_id,
            parent_id=parent_id,
            span_type=span_type,
            name=name,
            start_time=start_time,
            end_time=end_time,
            input_summary=input_summary,
            output_summary=output_summary,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        return entry

    # ── 渲染 ──────────────────────────────────────────────────────────

    def render_tree(self) -> str:
        """把扁平的 entries 渲染成 ASCII 树（展示层级关系）。"""
        lines: list[str] = []
        lines.append(f"Trace: {self.trace_id} ({len(self.entries)} spans)")
        lines.append("│")
        roots = [e for e in self.entries if e.parent_id is None]
        for i, root in enumerate(roots):
            self._render_span(root, prefix="", is_last=(i == len(roots) - 1), lines=lines)
        return "\n".join(lines)

    def _render_span(self, entry: TraceEntry, prefix: str, is_last: bool, lines: list[str]) -> None:
        """递归渲染一个 span 及其子 span。"""
        connector = "└── " if is_last else "├── "
        token_info = ""
        if entry.span_type == "llm_call":
            token_info = f" [in:{entry.prompt_tokens} out:{entry.completion_tokens}]"
        lines.append(
            f"{prefix}{connector}{entry.span_type}: {entry.name} "
            f"({entry.duration_ms:.0f}ms){token_info}"
        )
        children = [e for e in self.entries if e.parent_id == entry.span_id]
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            self._render_span(child, child_prefix, i == len(children) - 1, lines)

    def to_json(self) -> str:
        """导出所有 span 为 JSON 字符串（持久化用）。"""
        return json.dumps([{
            "span_id": e.span_id,
            "trace_id": e.trace_id,
            "parent_id": e.parent_id,
            "span_type": e.span_type,
            "name": e.name,
            "start_time": e.start_time,
            "end_time": e.end_time,
            "duration_ms": e.duration_ms,
            "prompt_tokens": e.prompt_tokens,
            "completion_tokens": e.completion_tokens,
            "input_summary": e.input_summary,
            "output_summary": e.output_summary,
        } for e in self.entries], ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# CostTracker：成本追踪（token × 单价）
# ═══════════════════════════════════════════════════════════════════════

class CostTracker:
    """累计一个 trace 内所有 LLM 调用的成本。

    公式：单次成本 = (prompt_tokens × 输入价 + completion_tokens × 输出价) / 1,000,000
    gpt-4o-mini 定价（演示用，生产应支持多模型定价表）。
    """

    PRICING: dict[str, dict[str, float]] = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},   # $/1M tokens
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "qwen-plus": {"input": 0.40, "output": 1.20},
    }

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model: str = model
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.llm_call_count: int = 0

    def add_llm_call(self, prompt_tokens: int, completion_tokens: int) -> float:
        """记录一次 LLM 调用，返回本次成本（美元）。"""
        price = self.PRICING.get(self.model, self.PRICING["gpt-4o-mini"])
        input_cost = prompt_tokens * price["input"] / 1_000_000
        output_cost = completion_tokens * price["output"] / 1_000_000
        cost = input_cost + output_cost

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost
        self.llm_call_count += 1
        return cost

    def add_from_trace(self, collector: TraceCollector) -> None:
        """从 TraceCollector 中提取所有 llm_call span，累计成本。"""
        for entry in collector.entries:
            if entry.span_type == "llm_call":
                self.add_llm_call(entry.prompt_tokens, entry.completion_tokens)

    def summary(self) -> str:
        """生成成本摘要。"""
        price = self.PRICING.get(self.model, self.PRICING["gpt-4o-mini"])
        return (
            f"模型: {self.model} (输入 ${price['input']}/1M, 输出 ${price['output']}/1M)\n"
            f"LLM 调用次数: {self.llm_call_count}\n"
            f"总输入 tokens: {self.total_prompt_tokens}\n"
            f"总输出 tokens: {self.total_completion_tokens}\n"
            f"总成本: ${self.total_cost_usd:.6f} (${self.total_cost_usd * 100:.4f} 美分)"
        )


# ═══════════════════════════════════════════════════════════════════════
# 离线 mock：模拟一个 3 步 Agent 的完整 trace
# ═══════════════════════════════════════════════════════════════════════

def build_mock_trace() -> TraceCollector:
    """构建一个模拟的 3 步 Agent trace（查北京天气→查上海天气→对比）。

    模拟场景：
      Step 1: LLM 决策 → 调 get_weather("北京") → 工具返回 25°C
      Step 2: LLM 决策 → 调 get_weather("上海") → 工具返回 28°C
      Step 3: LLM 决策 → 给最终答案"上海更热"

    使用 mock 的 token 数和耗时数据，不依赖真实 API。
    """
    collector = TraceCollector(trace_id="trace_mock_demo")
    base_time = 1_700_000_000.0  # 固定基准时间（避免时间戳跳动）

    # ── Step 1: 调 get_weather("北京") ──────────────────────────────────
    step1 = collector.add_span(
        span_type="step", name="第1步", parent_id=None,
        start_time=base_time, end_time=base_time + 0.520,
        input_summary="8 messages",
    )
    collector.add_span(
        span_type="llm_call", name="LLM决策", parent_id=step1.span_id,
        start_time=base_time, end_time=base_time + 0.400,
        input_summary="320 tokens, 8 messages",
        output_summary="tool_calls=[get_weather(city='北京')]",
        prompt_tokens=320, completion_tokens=45,
    )
    tool1 = collector.add_span(
        span_type="tool_call", name="get_weather", parent_id=step1.span_id,
        start_time=base_time + 0.400, end_time=base_time + 0.520,
        input_summary='{"city": "北京"}',
    )
    collector.add_span(
        span_type="tool_result", name="返回结果", parent_id=tool1.span_id,
        start_time=base_time + 0.500, end_time=base_time + 0.520,
        output_summary="北京今天晴 25°C",
    )

    # ── Step 2: 调 get_weather("上海") ──────────────────────────────────
    step2_base = base_time + 0.600
    step2 = collector.add_span(
        span_type="step", name="第2步", parent_id=None,
        start_time=step2_base, end_time=step2_base + 0.480,
        input_summary="10 messages (含 step1 工具结果)",
    )
    collector.add_span(
        span_type="llm_call", name="LLM决策", parent_id=step2.span_id,
        start_time=step2_base, end_time=step2_base + 0.380,
        input_summary="415 tokens, 10 messages",
        output_summary="tool_calls=[get_weather(city='上海')]",
        prompt_tokens=415, completion_tokens=42,
    )
    tool2 = collector.add_span(
        span_type="tool_call", name="get_weather", parent_id=step2.span_id,
        start_time=step2_base + 0.380, end_time=step2_base + 0.480,
        input_summary='{"city": "上海"}',
    )
    collector.add_span(
        span_type="tool_result", name="返回结果", parent_id=tool2.span_id,
        start_time=step2_base + 0.460, end_time=step2_base + 0.480,
        output_summary="上海今天多云 28°C",
    )

    # ── Step 3: 给最终答案（无工具调用）──────────────────────────────────
    step3_base = base_time + 1.200
    step3 = collector.add_span(
        span_type="step", name="第3步", parent_id=None,
        start_time=step3_base, end_time=step3_base + 0.350,
        input_summary="12 messages (含 step1+step2 结果)",
    )
    collector.add_span(
        span_type="llm_call", name="LLM最终回答", parent_id=step3.span_id,
        start_time=step3_base, end_time=step3_base + 0.350,
        input_summary="510 tokens, 12 messages",
        output_summary="上海更热(28°C > 25°C)",
        prompt_tokens=510, completion_tokens=80,
    )

    return collector


# ═══════════════════════════════════════════════════════════════════════
# Demo 1：Tracing — 收集并打印 trace entry
# ═══════════════════════════════════════════════════════════════════════

def demo_tracing() -> TraceCollector:
    """Demo 1: 构建一个 mock trace，逐条打印 span entry。"""
    print("=" * 72)
    print("Demo 1: Tracing（链路追踪）")
    print("  模拟 3 步 Agent：查北京天气 → 查上海天气 → 对比")
    print("  每个操作记录成一个 TraceEntry（span），含类型/耗时/token")
    print("=" * 72)
    print()

    # try 真实 API（占位符 key 必失败）→ 降级 mock
    try:
        cfg = get_config()
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        print("OUT:trace:offline: 真实 API 可用，但本章用 mock 数据演示（固定 token/耗时）")
    except Exception as e:
        print(f"OUT:trace:offline: 真实 API 不可用（{type(e).__name__}），使用 mock trace 演示")
    print()

    collector = build_mock_trace()

    print(f"收集到 {len(collector.entries)} 个 span：")
    print("-" * 72)
    for entry in collector.entries:
        print(f"OUT:trace:step{entry.metadata.get('step', '?') if entry.span_type == 'step' else ''}:"
              f" [{entry.span_type}] {entry.name}")
        print(f"    span_id={entry.span_id}, parent_id={entry.parent_id}, "
              f"耗时={entry.duration_ms:.0f}ms")
        if entry.span_type == "llm_call":
            print(f"    tokens: in={entry.prompt_tokens}, out={entry.completion_tokens}")
        if entry.input_summary:
            print(f"    输入: {entry.input_summary}")
        if entry.output_summary:
            print(f"    输出: {entry.output_summary}")
        print()

    return collector


# ═══════════════════════════════════════════════════════════════════════
# Demo 2：成本计算 — 按 gpt-4o-mini 定价计算多步总成本
# ═══════════════════════════════════════════════════════════════════════

def demo_cost(collector: TraceCollector) -> CostTracker:
    """Demo 2: 用 CostTracker 从 trace 提取 LLM 调用，计算总成本。"""
    print("=" * 72)
    print("Demo 2: 成本计算（token × 单价）")
    print("  gpt-4o-mini 定价: 输入 $0.15/1M tokens, 输出 $0.60/1M tokens")
    print("  公式: 成本 = (in_tokens × 0.15 + out_tokens × 0.60) / 1,000,000")
    print("=" * 72)
    print()

    tracker = CostTracker(model="gpt-4o-mini")
    price = tracker.PRICING["gpt-4o-mini"]

    print("逐笔 LLM 调用成本明细：")
    print("-" * 72)
    call_idx = 0
    for entry in collector.entries:
        if entry.span_type != "llm_call":
            continue
        call_idx += 1
        input_cost = entry.prompt_tokens * price["input"] / 1_000_000
        output_cost = entry.completion_tokens * price["output"] / 1_000_000
        total = tracker.add_llm_call(entry.prompt_tokens, entry.completion_tokens)
        print(f"OUT:cost: 第{call_idx}笔 LLM 调用 ({entry.name})")
        print(f"    输入: {entry.prompt_tokens} tokens × ${price['input']}/1M = ${input_cost:.6f}")
        print(f"    输出: {entry.completion_tokens} tokens × ${price['output']}/1M = ${output_cost:.6f}")
        print(f"    小计: ${total:.6f}")
        print()

    print("OUT:cost: 汇总")
    print("-" * 72)
    for line in tracker.summary().split("\n"):
        print(f"    {line}")
    print()

    # 成本直觉参考
    print("成本直觉参考：")
    print(f"    本次查询成本 ${tracker.total_cost_usd:.6f}")
    daily_queries = 1000
    monthly_cost = tracker.total_cost_usd * daily_queries * 30
    print(f"    若每天 {daily_queries} 次查询 → 月成本 ≈ ${monthly_cost:.2f}")
    print(f"    （不追踪成本，这笔钱会悄悄花掉）")
    print()

    return tracker


# ═══════════════════════════════════════════════════════════════════════
# Demo 3：Trace 可视化 — ASCII 树 + 结构化日志
# ═══════════════════════════════════════════════════════════════════════

def demo_visualization(collector: TraceCollector) -> None:
    """Demo 3: ASCII 树可视化 + 结构化日志导出。"""
    print("=" * 72)
    print("Demo 3: Trace 可视化（ASCII 树）+ 结构化日志（JSON）")
    print("  ASCII 树展示 step→llm_call→tool_call 层级关系")
    print("  结构化日志让机器可解析（可导入 ELK/Loki/Datadog）")
    print("=" * 72)
    print()

    print("OUT:viz: ASCII 树（一眼看清执行链路）")
    print("-" * 72)
    tree = collector.render_tree()
    for line in tree.split("\n"):
        print(f"  {line}")
    print()
    print("  解读：")
    print("    - 3 个顶层 step span（第1步/第2步/第3步）")
    print("    - 每个 step 下有 llm_call（LLM 决策）")
    print("    - step1/step2 下还有 tool_call→tool_result（工具调用链）")
    print("    - [in:X out:Y] 标注 LLM 调用的 token 数")
    print()

    print("OUT:log: 结构化日志（JSON，机器可解析）")
    print("-" * 72)
    # 打印前 3 条日志（演示格式，避免刷屏）
    for i, entry in enumerate(collector.entries[:3]):
        print(f"  {entry.to_log_line()}")
    if len(collector.entries) > 3:
        print(f"  ... (共 {len(collector.entries)} 条，此处展示前 3 条)")
    print()

    print("OUT:log: 完整 trace JSON 导出（可持久化到文件/数据库）")
    print("-" * 72)
    full_json = collector.to_json()
    # 只打印前 500 字符预览
    preview = full_json[:500]
    print(f"  {preview}")
    print(f"  ... (完整 JSON 共 {len(full_json)} 字符)")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("第16章 可观测与调试（Tracing、日志、成本追踪）")
    print("  TraceCollector / CostTracker / ASCII 树可视化 / 结构化日志")
    print("  （概念上实现第12章 Observer 接口，独立可运行）")
    print("=" * 72)
    print()

    # Demo 1: Tracing
    collector = demo_tracing()

    # Demo 2: 成本计算
    demo_cost(collector)

    # Demo 3: 可视化
    demo_visualization(collector)

    print("=" * 72)
    print("✓ 本章完成：3 大可观测功能演示完毕。")
    print("  核心收获：Agent 不再是黑盒，每一步都可追溯、可复盘、可优化。")
    print("  生产建议：trace 持久化（文件/DB），接 LangSmith/Langfuse 拿 Web UI。")
    print("=" * 72)


if __name__ == "__main__":
    main()
