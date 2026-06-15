"""
第12章 从零造框架 — 架构设计（6 大核心组件）

本章只定义接口契约（Protocol/ABC），不实现具体逻辑（第13章才实现）。

6 大核心组件：
  1. AgentRunner   — Agent 循环引擎（observe→reason→act，max_steps 保险丝）
  2. ToolRegistry  — 工具注册表（自描述工具，register/get_schema/execute）
  3. LLMClient     — LLM 包装器（调用+重试+流式+结构化输出）
  4. Memory        — 记忆管理（对话缓冲+可选摘要+token预算）
  5. ActionParser  — 输出解析器（结构化→动作，解析 tool_calls）
  6. Observer      — 可观测钩子（每步日志/trace/成本追踪）

本文件做三件事：
  1. 用 @runtime_checkable Protocol 定义 6 个组件接口
  2. 打印每个接口的职责和方法签名（用 inspect 自省）
  3. 验证 isinstance 检查可用（runtime_checkable 的价值）

不调用任何真实 API —— 纯接口定义 + 自省演示。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Protocol, Type, runtime_checkable

# ──────────────────────────────────────────────────────────────────────
# 让章节代码能 import shared.config（即使本章不真正用 config，
# 也 import 验证路径正确 —— 这是 T1 确立的约定）
# ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import get_config  # noqa: E402, F401  (验证 import 路径)


# ═══════════════════════════════════════════════════════════════════════
# 6 大核心组件接口定义（Protocol —— 结构性子类型）
# ═══════════════════════════════════════════════════════════════════════

# ─── 组件 1：ToolRegistry — 工具注册表 ───────────────────────────────────
@runtime_checkable
class ToolRegistry(Protocol):
    """工具注册表：管理工具的元数据（名称/描述/参数 schema）和处理器。

    核心设计 —— 自描述工具（self-describing）：
    每个工具自带名称、描述、参数 schema。get_schema() 返回 OpenAI tools
    格式的 JSON Schema 列表，可直接塞进 client.chat.completions.create(tools=...)。

    职责边界：
      ✅ register / get_schema / execute
      ❌ 不决定"调哪个工具"（LLM 决策）
      ❌ 不解析 LLM 响应（ActionParser 的活）
      ❌ 不记录调用日志（Observer 的活）
    """

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., str],
    ) -> None:
        """注册一个工具：名字 + 描述 + JSON Schema 参数 + 处理函数。"""
        ...

    def get_schema(self) -> list[dict[str, Any]]:
        """返回 OpenAI tools 格式的 JSON Schema 列表，给 LLM 看。"""
        ...

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """按名字查找 handler 并执行，返回字符串结果。"""
        ...


# ─── 组件 2：LLMClient — LLM 包装器 ──────────────────────────────────────
@runtime_checkable
class LLMClient(Protocol):
    """LLM 包装器：封装所有与 LLM API 的交互。

    职责边界：
      ✅ chat（调用）+ chat_with_retry（带退避重试，第06章）
      ✅ （可选扩展）stream / structured_output
      ❌ 不管对话历史（Memory 的活）
      ❌ 不解析 tool_calls（ActionParser 的活）
      ❌ 不记录成本（Observer 的活）

    为什么需要包装器：
      1. 统一接口 —— 换提供商只换实现，上层无感
      2. 重试逻辑集中 —— 不用每个 Agent 重写退避重试
      3. mock 友好 —— 测试时注入 MockLLMClient，不依赖真实 API
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """发送消息列表，返回响应 dict（含 content 和 tool_calls）。"""
        ...

    def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        max_retries: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """带指数退避的重试封装（第06章 call_llm_with_retry 的接口化）。"""
        ...


# ─── 组件 3：Memory — 记忆管理 ───────────────────────────────────────────
@runtime_checkable
class Memory(Protocol):
    """记忆管理：决定"模型能看到什么"（上下文工程的载体）。

    职责边界：
      ✅ add（追加消息）/ get_messages（返回列表）/ clear（清空）
      ✅ （可选扩展）自动压缩 / token 预算（第05/11章 SummaryMemory/TokenBudget）
      ❌ 不调 LLM（即使 SummaryMemory 做摘要，也是注入 LLMClient）
      ❌ 不执行工具 / 不决定何时停止

    多态价值：
      第05章的 ConversationBuffer / SummaryMemory / VectorMemory 都可实现此接口。
      AgentRunner 不关心你用哪种记忆，只要能 add/get_messages/clear。
    """

    def add(self, role: str, content: str) -> None:
        """追加一条消息（role: system/user/assistant/tool）。"""
        ...

    def get_messages(self) -> list[dict[str, Any]]:
        """返回当前消息列表（给 LLMClient 用）。"""
        ...

    def clear(self) -> None:
        """清空对话历史（开始新对话）。"""
        ...


# ─── 组件 4：ActionParser — 输出解析器 ───────────────────────────────────
@runtime_checkable
class ActionParser(Protocol):
    """输出解析器：把 LLM 原始响应解析成结构化的"动作"。

    职责边界：
      ✅ parse_tool_calls（提取 [{name, args, id}]）/ has_tool_calls（判断）
      ❌ 不执行工具（ToolRegistry 的活）
      ❌ 不调 LLM / 不做格式重试（上层/第06章自我纠正）

    为什么单独抽出来：
      1. 响应格式多样 —— OpenAI tools API 返回 tool_calls 字段；显式 ReAct
         （第07章）返回纯文本要正则解析；某些模型返回自定义 JSON
      2. 解析是脆弱环节 —— 第07章格式错误、第06章幻觉工具名，集中处理好维护
    """

    def parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """从响应中提取工具调用列表 [{name, args, id}]。"""
        ...

    def has_tool_calls(self, response: dict[str, Any]) -> bool:
        """判断响应是否包含工具调用（决定循环是否继续）。"""
        ...


# ─── 组件 5：Observer — 可观测钩子 ───────────────────────────────────────
@runtime_checkable
class Observer(Protocol):
    """可观测钩子：横切关注点，不侵入主流程地记录每步状态。

    设计原则 —— 纯旁路观察（只读不写）：
      ✅ 记录日志 / trace / 成本追踪 / 性能指标
      ❌ 不修改主流程状态（Memory/messages）
      ❌ 不调 LLM / 不执行工具
      ❌ 不决定循环是否继续

    这是观察者模式（Observer Pattern）+ OpenTelemetry 的设计基础。
    第15章会实现 TracingObserver，第16章会加 GuardrailObserver。
    """

    def on_step_start(self, step: int, messages: list[dict[str, Any]]) -> None:
        """每步开始：可记录 step 编号、当前消息数。"""
        ...

    def on_llm_call(self, messages: list[dict[str, Any]]) -> None:
        """调 LLM 前：可记录 token 数、估算成本。"""
        ...

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """调工具前：可记录工具名、参数。"""
        ...

    def on_tool_result(self, name: str, result: str) -> None:
        """工具返回后：可记录结果、耗时。"""
        ...

    def on_step_end(self, step: int) -> None:
        """每步结束：可记录总耗时。"""
        ...


# ─── 组件 6：AgentRunner — Agent 循环引擎 ─────────────────────────────────
@runtime_checkable
class AgentRunner(Protocol):
    """Agent 循环引擎：驱动 observe→reason→act，框架的"心脏"。

    职责边界：
      ✅ 接收 task → 初始化 Memory → 循环 max_steps 次
      ✅ 每步：调 LLM → 解析 → 执行工具 → 更新 Memory → 触发 Observer
      ✅ 两个终止条件：模型不再调工具（完成）/ 达到 max_steps（保险丝）
      ❌ 不直接调 client.chat.completions.create（LLMClient 的活）
      ❌ 不直接执行工具函数（ToolRegistry 的活）
      ❌ 不解析 JSON/正则（ActionParser 的活）

    max_steps 保险丝（第04章）：必填，默认 10，永不设无限。
    实现类通过构造函数注入其他 5 个组件（依赖注入）。
    """

    def run(self, task: str, max_steps: int = 10) -> str:
        """运行 Agent 循环，返回最终答案。max_steps 是防无限循环的保险丝。"""
        ...


# ═══════════════════════════════════════════════════════════════════════
# 组件注册表（元数据，用于演示打印）
# ═══════════════════════════════════════════════════════════════════════

COMPONENTS: list[tuple[str, Type[Any], str]] = [
    (
        "ToolRegistry",
        ToolRegistry,
        "工具注册表：管理工具元数据 + handler，对外提供 register/get_schema/execute",
    ),
    (
        "LLMClient",
        LLMClient,
        "LLM 包装器：封装调用 + 退避重试，换提供商只换实现",
    ),
    (
        "Memory",
        Memory,
        "记忆管理：决定模型能看到什么（ConversationBuffer/SummaryMemory/VectorMemory 都可实现）",
    ),
    (
        "ActionParser",
        ActionParser,
        "输出解析器：把 LLM 响应解析成 [{name,args,id}]，屏蔽 OpenAI/ReAct/自定义 JSON 差异",
    ),
    (
        "Observer",
        Observer,
        "可观测钩子：纯旁路观察（只读不写），记录日志/trace/成本，不侵入主流程",
    ),
    (
        "AgentRunner",
        AgentRunner,
        "Agent 循环引擎：协调其他 5 个组件，max_steps 保险丝防无限循环",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# 演示辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _format_signature(func: Any) -> str:
    """提取方法的签名字符串（去掉 ... 方法体）。"""
    try:
        sig = inspect.signature(func)
        return f"{func.__name__}{sig}"
    except (ValueError, TypeError):
        return f"{func.__name__}(...)"


def _list_protocol_methods(protocol_cls: Type[Any]) -> list[str]:
    """提取 Protocol 里所有定义的方法的签名。"""
    signatures: list[str] = []
    for name in sorted(dir(protocol_cls)):
        if name.startswith("_"):
            continue
        attr = getattr(protocol_cls, name, None)
        if callable(attr):
            signatures.append(_format_signature(attr))
    return signatures


def _print_components() -> None:
    """打印每个组件的职责 + 方法签名。"""
    for name, proto, desc in COMPONENTS:
        print(f"OUT:component:{name}:")
        print(f"  职责: {desc}")
        methods = _list_protocol_methods(proto)
        for sig in methods:
            print(f"  方法: {sig}")
        print()


def _print_architecture() -> None:
    """打印 ASCII 架构图，显示组件关系。"""
    print("OUT:architecture:")
    print(
        """
  ┌─────────────────────────────────────────────────────────────────┐
  │                    AgentRunner（循环引擎）                       │
  │                                                                 │
  │   ┌────────┐        ┌──────────────────────────────────────┐   │
  │   │ Memory │◄──────►│  for step in range(max_steps):       │   │
  │   │ (消息) │        │    observer.on_step_start(step)      │   │
  │   └───┬────┘        │    resp = llm.chat_with_retry(...)   │   │
  │       │ messages    │    if not parser.has_tool_calls:     │   │
  │       ▼             │        return final_answer           │   │
  │   ┌──────────┐      │    for call in parser.parse(resp):   │   │
  │   │LLMClient │◄─────┤        result = tools.execute(call)  │   │
  │   │(调模型)  │      │        memory.add("tool", result)    │   │
  │   └────┬─────┘      │    observer.on_step_end(step)        │   │
  │        │ response   └──────────────┬───────────────────────┘   │
  │        ▼                          │ coordinates                │
  │   ┌────────────┐   actions       ┌─────────────┐              │
  │   │ActionParser│────────────────►│ToolRegistry │              │
  │   │(解析 tool_ │                 │(register/   │              │
  │   │ calls)     │                 │ get_schema/ │              │
  │   └────────────┘                 │ execute)    │              │
  │                                  └─────────────┘              │
  │   ┌────────────────────────────────────────────────────────┐  │
  │   │           Observer（横切所有组件，纯旁路观察）            │  │
  │   │  on_step_start / on_llm_call / on_tool_call /          │  │
  │   │  on_tool_result / on_step_end                          │  │
  │   └────────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────┘

  数据流: task → Memory → LLMClient → ActionParser → ToolRegistry → Memory (循环)
  Observer: 在每个关键点挂钩，只读不写，记录日志/trace/成本
"""
    )


def _print_verify() -> None:
    """验证 runtime_checkable 的 isinstance 检查可用。"""
    print("OUT:verify:")
    print("  验证 @runtime_checkable Protocol 的 isinstance 检查可用:")
    print()

    # 定义一个"形状匹配"的类（不显式 inherit Protocol）
    class FakeToolRegistry:
        def register(self, name: str, description: str,
                     parameters: dict, handler: Callable[..., str]) -> None:
            pass

        def get_schema(self) -> list:
            return []

        def execute(self, name: str, args: dict) -> str:
            return ""

    # 定义一个"形状不匹配"的类（缺方法）
    class IncompleteRegistry:
        def register(self, name: str) -> None:
            pass
        # 缺 get_schema 和 execute

    fake = FakeToolRegistry()
    incomplete = IncompleteRegistry()

    print("  FakeToolRegistry（实现了 register/get_schema/execute）:")
    print(f"    isinstance(fake, ToolRegistry) = {isinstance(fake, ToolRegistry)}")
    print()
    print("  IncompleteRegistry（只有 register，缺 get_schema/execute）:")
    print(
        f"    isinstance(incomplete, ToolRegistry) = "
        f"{isinstance(incomplete, ToolRegistry)}"
    )
    print()
    print("  结论: Protocol 的结构性子类型 —— 形状匹配即算实现，无需显式 inherit。")
    print()

    # 验证 6 个 Protocol 都可 isinstance 检查
    print("  6 大组件 Protocol 的 runtime_checkable 验证:")
    for name, proto, _desc in COMPONENTS:
        result = isinstance(object(), proto)  # object() 不会匹配任何 Protocol
        marker = "✓ 可检查" if result is False else "✗"
        print(f"    isinstance(object(), {name:14s}) → {str(result):5s} {marker}")
    print()
    print("  （object() 不匹配任何 Protocol 是正确的 —— 它没有那些方法）")


def _print_config_check() -> None:
    """打印配置加载验证（证明 shared.config import 路径正确）。"""
    print("OUT:verify:config")
    try:
        cfg = get_config()
        masked = (
            f"***{cfg.api_key[-4:]}" if len(cfg.api_key) > 4 else "(set)"
        )
        print(
            f"  shared.config import 路径正确 ✓ "
            f"(provider={cfg.provider}, model={cfg.model}, key={masked})"
        )
    except SystemExit:
        # 占位符 key 会 sys.exit —— 这不影响本章（纯接口定义不调 API）
        print("  shared.config import 路径正确 ✓ (API key 未配置，本章不需要)")


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("第12章 从零造框架 — 架构设计（6 大核心组件接口定义）")
    print("本章只定义接口契约（Protocol），不实现具体逻辑（第13章才实现）")
    print("=" * 72)
    print()

    # Demo 1: 6 大组件的职责 + 方法签名
    print("▎ Demo 1: 6 大核心组件接口")
    print("-" * 72)
    _print_components()

    # Demo 2: ASCII 架构图
    print("▎ Demo 2: 架构总览（组件关系图）")
    print("-" * 72)
    _print_architecture()

    # Demo 3: runtime_checkable 验证
    print("▎ Demo 3: 接口验证（runtime_checkable isinstance）")
    print("-" * 72)
    _print_verify()

    # Demo 4: 配置 import 路径验证
    print("▎ Demo 4: 配置 import 路径验证")
    print("-" * 72)
    _print_config_check()

    print()
    print("=" * 72)
    print("✓ 本章完成：6 大组件接口已定义。第13章会实现具体逻辑。")
    print("=" * 72)


if __name__ == "__main__":
    main()
