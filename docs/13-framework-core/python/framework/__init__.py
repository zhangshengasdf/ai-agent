"""
第13章 从零造框架 — 6 大核心组件实现

本模块实现第12章定义的 6 个 Protocol 接口，组合成完整 Agent。

6 大组件（对应第12章的 Protocol）：
  1. InMemoryToolRegistry   → 实现 ToolRegistry（工具注册表）
  2. DefaultLLMClient       → 实现 LLMClient（LLM 包装器 + 离线 mock 降级）
  3. ConversationMemory     → 实现 Memory（对话缓冲记忆）
  4. OpenAIToolCallParser   → 实现 ActionParser（tool_calls 解析器）
  5. LoggingObserver        → 实现 Observer（日志钩子，纯旁路观察）
  6. DefaultAgentRunner     → 实现 AgentRunner（循环引擎，协调其他 5 个组件）

设计原则（第12章确立）：
  - 依赖注入：AgentRunner 通过构造函数接收其他 5 个组件（不 new 具体实现）
  - 单向依赖：AgentRunner 依赖其他组件，其他组件互不依赖
  - Observer 横切关注点：纯旁路（只读不写），不修改主流程状态
  - max_steps 保险丝：必填，默认 10，永不设无限
  - 两个终止条件：(1) 模型不调工具 = 完成；(2) 达到 max_steps = 保险丝
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


# ═══════════════════════════════════════════════════════════════════════
# Component 1: InMemoryToolRegistry — 工具注册表
# ═══════════════════════════════════════════════════════════════════════

class InMemoryToolRegistry:
    """工具注册表：管理工具元数据（name/description/parameters）+ handler。

    实现 ToolRegistry Protocol（register / get_schema / execute）。

    核心设计 —— 自描述工具（self-describing）：
    每个工具自带 name/description/parameters schema。get_schema() 返回 OpenAI
    tools 格式 JSON Schema，可直接塞进 chat.completions.create(tools=...)。

    职责边界：
      ✅ register / get_schema / execute
      ❌ 不决定"调哪个工具"（LLM 决策）
      ❌ 不解析 LLM 响应（ActionParser 的活）
      ❌ 不记录调用日志（Observer 的活）
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., str],
    ) -> None:
        """注册一个工具：名字 + 描述 + JSON Schema 参数 + 处理函数。"""
        self._tools[name] = {
            "description": description,
            "parameters": parameters,
            "handler": handler,
        }

    def get_schema(self) -> list[dict[str, Any]]:
        """返回 OpenAI tools 格式的 JSON Schema 列表，给 LLM 看。

        格式: [{"type": "function", "function": {"name", "description", "parameters"}}]
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"],
                },
            }
            for name, info in self._tools.items()
        ]

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """按名字查找 handler 并执行，返回字符串结果。

        未知工具名 → 返回错误消息（不抛异常，让 Agent 能自我纠正）。
        工具抛异常 → 返回错误消息（同上，第06章机制 2）。
        """
        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            return f"[错误] 工具 '{name}' 不存在。可用工具: {available}"
        handler = self._tools[name]["handler"]
        try:
            return str(handler(**args))
        except Exception as e:
            return f"[工具执行失败] {name}: {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
# Component 2: DefaultLLMClient — LLM 包装器（含退避重试 + 离线 mock 降级）
# ═══════════════════════════════════════════════════════════════════════

class DefaultLLMClient:
    """LLM 包装器：封装 OpenAI 调用 + 退避重试 + 离线 mock 降级。

    实现 LLMClient Protocol（chat / chat_with_retry）。

    归一化响应格式（屏蔽 SDK 差异）：
      {"content": str|None, "tool_calls": [...]|None, "usage": {...}|None}

    离线 mock 设计（关键！）：
      .env 的 OPENAI_API_KEY=sk-REPLACE-ME → 真实 API 必失败（401/连接错误）。
      chat_with_retry 在重试耗尽/永久错误后，降级返回预设 mock 响应序列，
      完整演示多步 Agent 循环：
        step1: get_weather(北京) → step2: get_weather(上海)
        → step3: calculate(28-25) → step4: 最终回答（不调工具，循环终止）
    """

    def __init__(self, config: Any) -> None:
        """初始化 LLM 客户端。

        Args:
            config: shared.config.Config 实例（provider/base_url/api_key/model）。
        """
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self._model = config.model
        self._mock_index = 0
        # 预设 mock 响应序列（模拟"查天气+算温差"的完整 4 步循环）
        self._mock_sequence: list[dict[str, Any]] = [
            {
                "content": None,
                "tool_calls": [{
                    "id": "call_mock_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
                }],
            },
            {
                "content": None,
                "tool_calls": [{
                    "id": "call_mock_2", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "上海"}'},
                }],
            },
            {
                "content": None,
                "tool_calls": [{
                    "id": "call_mock_3", "type": "function",
                    "function": {"name": "calculate", "arguments": '{"expression": "28-25"}'},
                }],
            },
            {
                "content": (
                    "北京今天晴 25°C，上海今天多云 28°C。"
                    "温差为 3°C（上海比北京高 3 度）。"
                ),
                "tool_calls": None,
            },
        ]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """调 OpenAI API，返回归一化 dict {content, tool_calls, usage}。

        Raises:
            OpenAI SDK 的各种异常（由 chat_with_retry 处理重试/降级）。
        """
        api_kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**api_kwargs)
        msg = response.choices[0].message

        # 归一化 tool_calls（SDK 格式 → 统一 dict 格式）
        raw_calls: list[dict[str, Any]] | None = None
        if msg.tool_calls:
            raw_calls = []
            for tc in msg.tool_calls:
                if tc.type != "function":
                    continue
                raw_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {"content": msg.content, "tool_calls": raw_calls, "usage": usage}

    def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        max_retries: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """带指数退避重试 + 离线 mock 降级。

        重试逻辑（第06章机制 1+4）：
          - 可重试错误（超时/连接/限流/5xx）→ 退避重试最多 max_retries 次
          - 永久错误（认证 401/参数 400）→ 不重试，直接降级 mock
          - 重试耗尽 → 降级 mock

        离线降级（关键）：
          .env 用占位符 sk-REPLACE-ME 时，真实 API 必失败 → 降级 mock，
          保证无有效密钥也能完整演示 Agent 循环（exit 0）。
        """
        tools = kwargs.get("tools")
        backoff_scale = kwargs.get("_backoff_scale", 0.1)  # 演示用 0.1s

        for attempt in range(max_retries):
            try:
                return self.chat(messages, tools=tools)
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                # 可重试错误 → 退避重试
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) * backoff_scale
                    print(
                        f"OUT:framework:retry: 第 {attempt + 1}/{max_retries} 次"
                        f"失败（{type(e).__name__}），等待 {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    continue
                break  # 重试耗尽 → 降级 mock
            except (AuthenticationError, BadRequestError):
                break  # 永久错误 → 不重试，直接降级 mock
            except (APIError, Exception):
                break  # 其他错误 → 降级 mock

        # ── 降级：离线 mock ──
        print(f"OUT:framework:offline: API 不可用，降级为 mock 响应（第 {self._mock_index + 1} 步）")
        return self._next_mock()

    def _next_mock(self) -> dict[str, Any]:
        """返回下一个预设 mock 响应（按序列循环）。"""
        resp = self._mock_sequence[self._mock_index % len(self._mock_sequence)]
        self._mock_index += 1
        # 返回深拷贝，防止外部修改内部序列
        tool_calls_copy = None
        if resp["tool_calls"]:
            tool_calls_copy = [json.loads(json.dumps(tc)) for tc in resp["tool_calls"]]
        return {
            "content": resp["content"],
            "tool_calls": tool_calls_copy,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


# ═══════════════════════════════════════════════════════════════════════
# Component 3: ConversationMemory — 对话缓冲记忆
# ═══════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """对话缓冲记忆：list 存储，支持 system/user/assistant/tool 消息。

    实现 Memory Protocol（add / get_messages / clear）。

    add 方法支持额外字段（tool_calls/tool_call_id），以适配 OpenAI 多轮工具调用格式：
      - assistant 消息可能携带 tool_calls（模型决定调用哪些工具）
      - tool 消息需要 tool_call_id（对应哪个 tool_call 的结果）

    多态价值（第05章）：
      ConversationBuffer / SummaryMemory / VectorMemory 都可实现此接口。
      AgentRunner 不关心用哪种记忆，只要能 add/get_messages/clear。
    """

    def __init__(self, system_prompt: str = "") -> None:
        self._system_prompt = system_prompt
        self._messages: list[dict[str, Any]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def add(
        self,
        role: str,
        content: str,
        *,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """追加一条消息。

        Args:
            role: system / user / assistant / tool
            content: 消息内容
            tool_call_id: tool 消息的关联 ID（对应 assistant 的 tool_call.id）
            tool_calls: assistant 消息携带的工具调用列表
        """
        msg: dict[str, Any] = {"role": role, "content": content}
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def get_messages(self) -> list[dict[str, Any]]:
        """返回消息列表副本（防止外部修改内部状态）。"""
        return [dict(m) for m in self._messages]

    def clear(self) -> None:
        """清空对话历史，保留 system prompt。"""
        self._messages = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})


# ═══════════════════════════════════════════════════════════════════════
# Component 4: OpenAIToolCallParser — 输出解析器
# ═══════════════════════════════════════════════════════════════════════

class OpenAIToolCallParser:
    """解析 LLM 响应中的 tool_calls，归一化为 [{name, args, id}]。

    实现 ActionParser Protocol（parse_tool_calls / has_tool_calls）。

    屏蔽 OpenAI tool_calls 格式差异：
      原始: {"id": ..., "type": "function", "function": {"name": ..., "arguments": "..."}}
      归一化: {"name": ..., "args": <parsed dict>, "id": ...}

    职责边界：
      ✅ parse_tool_calls / has_tool_calls
      ❌ 不执行工具（ToolRegistry 的活）
      ❌ 不调 LLM / 不做格式重试（上层/第06章自我纠正）
    """

    def parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """从响应提取工具调用列表 [{name, args, id}]。"""
        result: list[dict[str, Any]] = []
        for tc in response.get("tool_calls") or []:
            if tc.get("type") != "function":
                continue
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            result.append({
                "name": func.get("name", ""),
                "args": args,
                "id": tc.get("id", ""),
            })
        return result

    def has_tool_calls(self, response: dict[str, Any]) -> bool:
        """判断响应是否包含工具调用（决定循环是否继续）。"""
        calls = response.get("tool_calls")
        return bool(calls)


# ═══════════════════════════════════════════════════════════════════════
# Component 5: LoggingObserver — 日志钩子（纯旁路，只读不写）
# ═══════════════════════════════════════════════════════════════════════

class LoggingObserver:
    """日志观察者：在关键点打印格式化日志（OUT:framework:step{N}: 前缀）。

    实现 Observer Protocol（5 个 on_* 钩子）。

    设计原则 —— 纯旁路观察（只读不写）：
      ✅ 记录日志（步骤/工具调用/结果）
      ❌ 不修改主流程状态（Memory/messages）
      ❌ 不调 LLM / 不执行工具
      ❌ 不决定循环是否继续

    这是观察者模式（Observer Pattern）+ OpenTelemetry 的设计基础。
    第15章会实现 TracingObserver，第16章会加 GuardrailObserver。
    """

    def __init__(self) -> None:
        self._step = 0

    def on_step_start(self, step: int, messages: list[dict[str, Any]]) -> None:
        """每步开始：打印步骤编号 + 当前消息数。"""
        self._step = step + 1
        print(f"OUT:framework:step{self._step}: ▶ 步骤开始（历史消息: {len(messages)} 条）")

    def on_llm_call(self, messages: list[dict[str, Any]]) -> None:
        """调 LLM 前：打印输入消息数。"""
        print(f"OUT:framework:step{self._step}: 🧠 调用 LLM（输入 {len(messages)} 条消息）")

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """调工具前：打印工具名 + 参数。"""
        print(f"OUT:framework:step{self._step}: 🔧 调用工具 {name}({args})")

    def on_tool_result(self, name: str, result: str) -> None:
        """工具返回后：打印结果（截断过长的）。"""
        preview = result[:60] + "..." if len(result) > 60 else result
        print(f"OUT:framework:step{self._step}: 📋 工具结果 {name} → {preview}")

    def on_step_end(self, step: int) -> None:
        """每步结束：打印完成标记。"""
        print(f"OUT:framework:step{self._step}: ✓ 步骤结束")


# ═══════════════════════════════════════════════════════════════════════
# Component 6: DefaultAgentRunner — Agent 循环引擎（框架的心脏）
# ═══════════════════════════════════════════════════════════════════════

class DefaultAgentRunner:
    """Agent 循环引擎：协调 5 个组件，驱动 observe→reason→act 循环。

    实现 AgentRunner Protocol（run(task, max_steps)）。

    依赖注入：通过构造函数接收 5 个组件（不 new 具体实现，保证可替换性）。
    max_steps 保险丝：必填，默认 10，永不设无限（第04章反模式 #1）。

    循环逻辑（第04章 agent_loop 的接口化版本）：
      1. memory.add("user", task)
      2. for step in range(max_steps):
           observer.on_step_start / on_llm_call
           response = llm.chat_with_retry(memory.get_messages(), tools=tools.get_schema())
           if not parser.has_tool_calls(response): return final_answer  # 终止条件 1
           memory.add("assistant", ..., tool_calls=...)  # 记录模型决策
           for call in parser.parse_tool_calls(response):
               observer.on_tool_call
               result = tools.execute(call.name, call.args)
               observer.on_tool_result
               memory.add("tool", result, tool_call_id=call.id)
           observer.on_step_end
      3. return "达到最大步数"  # 终止条件 2（保险丝）

    职责边界：
      ✅ 协调 5 个组件，驱动循环
      ❌ 不直接调 client.chat.completions.create（LLMClient 的活）
      ❌ 不直接执行工具函数（ToolRegistry 的活）
      ❌ 不解析 JSON/正则（ActionParser 的活）
    """

    def __init__(
        self,
        llm: DefaultLLMClient,
        tools: InMemoryToolRegistry,
        memory: ConversationMemory,
        parser: OpenAIToolCallParser,
        observer: LoggingObserver,
    ) -> None:
        """通过构造函数注入 5 个组件（依赖注入）。"""
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.parser = parser
        self.observer = observer

    def run(self, task: str, max_steps: int = 10) -> str:
        """运行 Agent 循环，返回最终答案。max_steps 是防无限循环的保险丝。"""
        # 1. 初始化 Memory（把用户任务存入对话历史）
        self.memory.add("user", task)

        # 2. 循环 max_steps 次
        for step in range(max_steps):
            self.observer.on_step_start(step, self.memory.get_messages())
            self.observer.on_llm_call(self.memory.get_messages())

            # ── Reason：调 LLM（带重试 + mock 降级）──
            response = self.llm.chat_with_retry(
                self.memory.get_messages(),
                tools=self.tools.get_schema(),
            )

            # ── 终止条件 1：模型不调工具 = 任务完成 ──
            if not self.parser.has_tool_calls(response):
                answer = response.get("content") or "(空回答)"
                self.observer.on_step_end(step)
                return answer

            # ── Act：记录 assistant 决策（含 tool_calls）到 Memory ──
            content = response.get("content") or ""
            extra: dict[str, Any] = {}
            if response.get("tool_calls"):
                extra["tool_calls"] = response["tool_calls"]
            self.memory.add("assistant", content, **extra)

            # ── 执行每个工具调用 ──
            for call in self.parser.parse_tool_calls(response):
                self.observer.on_tool_call(call["name"], call["args"])
                result = self.tools.execute(call["name"], call["args"])
                self.observer.on_tool_result(call["name"], result)
                # 工具结果以 role="tool" 追加到 Memory
                self.memory.add("tool", result, tool_call_id=call["id"])

            self.observer.on_step_end(step)

        # 3. 终止条件 2：达到 max_steps，强制停止
        return "(已达到最大步数，强制停止)"


# ═══════════════════════════════════════════════════════════════════════
# 导出（便于 from framework import ...）
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "InMemoryToolRegistry",
    "DefaultLLMClient",
    "ConversationMemory",
    "OpenAIToolCallParser",
    "LoggingObserver",
    "DefaultAgentRunner",
]
