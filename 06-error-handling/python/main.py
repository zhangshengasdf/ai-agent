"""第06章 错误处理与容错（Error Handling & Resilience）

本章在第04章 Agent 循环基础上，加入四大容错机制：

  机制 1：指数退避重试 —— 对可重试错误（超时/限流/连接），重试最多 3 次，
          每次等 2**attempt 秒（演示用 0.1s 缩放，真实用 1s）。
  机制 2：工具异常 + Agent 自我纠正 —— 工具抛异常时，把错误以 role="tool"
          反馈给 Agent，让它换工具或调整参数，而非崩溃。
  机制 3：幻觉工具名检测 —— 模型调了不存在的工具名 → 告知正确工具列表 →
          Agent 重新选择合法工具。
  机制 4：区分可重试错误 vs 永久错误 —— 网络错误重试，认证错误直接退出。

离线 mock 设计（关键）：
  .env 的 OPENAI_API_KEY=sk-REPLACE-ME 是占位符，真实 API 调用必失败（401）。
  所有 demo 先 try 真实 API（失败时降级），然后用离线 mock 100% 可靠地
  演示四大容错机制，保证 exit code 0。
"""

import json
import sys
import time
from pathlib import Path

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


# ════════════════════════════════════════════════════════════════════
# 工具实现（复用第03/04章的 mock 工具）
# ════════════════════════════════════════════════════════════════════


def get_weather(city: str) -> str:
    """查询指定城市的当前天气（mock 数据）。"""
    mock_data = {
        "北京": "北京今天晴, 25°C, 湿度 40%, 东北风 2 级",
        "上海": "上海今天多云, 28°C, 湿度 65%, 东南风 3 级",
        "深圳": "深圳今天小雨, 30°C, 湿度 80%, 南风 2 级",
        "东京": "东京今天阴, 22°C, 湿度 55%, 西风 1 级",
    }
    if city not in mock_data:
        raise ValueError(
            f"城市 '{city}' 不在数据库中。可用的城市：{', '.join(mock_data.keys())}"
        )
    return mock_data[city]


def calculate(expression: str) -> str:
    """安全的数学计算。只允许数字和基本运算符。"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"错误：表达式包含不允许的字符，只支持数字和 + - * / ( )"
    try:
        result = eval(expression)  # noqa: S307 — 受限字符集，教学用途
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def search_wiki(query: str) -> str:
    """模拟百科搜索（mock 知识库）。"""
    knowledge = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。",
        "机器学习": "机器学习是 AI 的分支，使计算机从数据中学习。",
        "agent": "AI Agent 是能感知环境、决策、行动的自主系统。",
        "火星": "火星是太阳系第四颗行星，表面温度约 -63°C，大气稀薄。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到与'{query}'相关的百科条目。"


# ════════════════════════════════════════════════════════════════════
# 工具定义（JSON Schema）
# ════════════════════════════════════════════════════════════════════

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气，返回温度、湿度和风力信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'、'东京'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如'2+3*4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": "搜索百科知识，返回与查询相关的简介信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如'python'、'火星'",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# 工具名 → 函数的映射（dispatch 模式）
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_wiki": search_wiki,
}

# 合法工具名集合（用于幻觉检测）
VALID_TOOL_NAMES = set(TOOL_FUNCTIONS.keys())

MAX_STEPS = 10
MAX_RETRIES = 3


# ════════════════════════════════════════════════════════════════════
# 机制 4：错误分类 —— 区分可重试错误 vs 永久错误
# ════════════════════════════════════════════════════════════════════


def is_retryable(error: Exception) -> bool:
    """判断一个错误是否值得重试。

    可重试错误：超时、连接错误、限流、服务端 5xx —— 瞬时故障，再试可能成功。
    永久错误：认证失败、参数错误 —— 再试 100 次也一样，立即失败。
    """
    if isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    # APIError 是基类，检查 status_code 判断 5xx 服务端错误
    if isinstance(error, APIError):
        status = getattr(error, "status_code", None)
        if status is not None and status >= 500:
            return True
    return False


# ════════════════════════════════════════════════════════════════════
# 机制 1：指数退避重试
# ════════════════════════════════════════════════════════════════════


def call_llm_with_retry(
    messages: list,
    *,
    tools_list: list | None = None,
    backoff_scale: float = 1.0,
) -> object:
    """带指数退避重试的 LLM 调用。

    只对"可重试错误"（超时/限流/连接）重试，最多 MAX_RETRIES 次。
    每次等待 2**attempt * backoff_scale 秒（演示用 0.1，真实用 1.0）。
    永久错误（认证/参数）直接抛出，不浪费时间重试。

    Args:
        messages: 消息列表。
        tools_list: 工具定义（可选）。
        backoff_scale: 退避缩放因子。演示用 0.1（0.1s/0.2s/0.4s），
                       生产用 1.0（1s/2s/4s）。

    Returns:
        OpenAI ChatCompletion 响应对象。

    Raises:
        原始异常（重试耗尽或遇到永久错误时）。
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs: dict = {"model": cfg.model, "messages": messages}
            if tools_list is not None:
                kwargs["tools"] = tools_list
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            last_error = e
            # 机制 4：只重试可重试错误
            if not is_retryable(e):
                # 永久错误（认证/参数）—— 立即抛出，不重试
                raise
            if attempt == MAX_RETRIES - 1:
                # 最后一次也失败了，抛出去
                print(
                    f"OUT:retry: 第 {attempt + 1}/{MAX_RETRIES} 次失败（{type(e).__name__}），"
                    f"已达最大重试次数，放弃。"
                )
                raise
            wait = (2 ** attempt) * backoff_scale
            print(
                f"OUT:retry: 第 {attempt + 1}/{MAX_RETRIES} 次失败（{type(e).__name__}），"
                f"等待 {wait:.1f}s 后重试..."
            )
            time.sleep(wait)
    # 理论上不会走到这里（循环内会 return 或 raise），但满足类型检查
    assert last_error is not None
    raise last_error


# ════════════════════════════════════════════════════════════════════
# 护栏：输入/输出校验（基础版）
# ════════════════════════════════════════════════════════════════════


def validate_input(user_message: str) -> str:
    """输入护栏：校验用户输入。深度安全护栏见第17章。"""
    if len(user_message) > 10_000:
        raise ValueError("输入过长（超过 10000 字符），请精简后重试")
    lower = user_message.lower()
    if "ignore previous instructions" in lower or "忽略以上所有指令" in user_message:
        raise ValueError("检测到疑似 prompt 注入，已拒绝")
    return user_message


def validate_output(answer: str) -> str:
    """输出护栏：校验 Agent 回答。"""
    if len(answer) > 5000:
        return answer[:5000] + "\n\n（回答过长，已截断）"
    return answer


# ════════════════════════════════════════════════════════════════════
# 核心：带容错的 Agent 循环（扩展第04章）
# ════════════════════════════════════════════════════════════════════


def resilient_agent_loop(user_message: str) -> str:
    """带四大容错机制的 Agent 循环。

    相比第04章的 agent_loop，新增：
      - 机制 1：LLM 调用带指数退避重试（call_llm_with_retry）
      - 机制 2：工具异常 → 反馈给 Agent 自我纠正
      - 机制 3：幻觉工具名 → 告知正确列表 → Agent 重新选择
      - 机制 4：永久错误（认证/参数）直接退出，可重试错误才重试
    """
    user_message = validate_input(user_message)  # 输入护栏

    messages: list = [
        {
            "role": "system",
            "content": (
                "你是一个任务助手 Agent。你可以查天气、做计算、搜百科。"
                "面对复杂任务，请一步步调用工具收集信息，最后给出综合回答。"
                "如果某个工具失败，请阅读错误信息并尝试换工具或调整参数。"
                "当信息足够回答时，直接给出最终回答。"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'=' * 60}")
    print(f"任务: {user_message}")
    print(f"{'=' * 60}")

    for step in range(1, MAX_STEPS + 1):
        print(f"OUT:step{step}: 思考中...")

        # ── 机制 1 + 4：带退避重试的 LLM 调用 ──────────────────────
        response = call_llm_with_retry(messages, tools_list=tools)
        assistant_msg = response.choices[0].message

        # 终止条件 1：模型不再调工具 = 任务完成
        if not assistant_msg.tool_calls:
            answer = assistant_msg.content or "(空回答)"
            print(f"OUT:step{step}: ✓ 任务完成！")
            print(f"OUT:step{step}: 回答: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            return validate_output(answer)

        messages.append(assistant_msg.model_dump())
        tool_names = [tc.function.name for tc in assistant_msg.tool_calls]
        print(f"OUT:step{step}: 决定调用工具: {', '.join(tool_names)}")

        # ── 执行每个工具调用（含机制 2 + 3）─────────────────────────
        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # JSON 解析失败：反馈给 Agent 让它重新生成参数
                args = {}
                result = (
                    f"[参数解析失败] 工具 '{func_name}' 的 arguments "
                    f"'{tc.function.arguments}' 不是合法 JSON。请重新生成。"
                )
                print(f"OUT:step{step}: ⚠️ JSON 解析失败，反馈给 Agent")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
                continue

            # ── 机制 3：幻觉工具名检测 ──────────────────────────────
            if func_name not in VALID_TOOL_NAMES:
                result = (
                    f"[错误] 工具 '{func_name}' 不存在。"
                    f"可用的工具有：{', '.join(sorted(VALID_TOOL_NAMES))}。"
                    f"请从上述列表中选择一个。"
                )
                print(f"OUT:step{step}: 🚫 幻觉工具检测：'{func_name}' 不存在，已告知 Agent")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
                continue

            # ── 机制 2：工具异常 → 反馈给 Agent 自我纠正 ─────────────
            print(f"OUT:step{step}: 执行 {func_name}({args})")
            func = TOOL_FUNCTIONS[func_name]
            try:
                result = func(**args)
                preview = result[:80] + ("..." if len(result) > 80 else "")
                print(f"OUT:step{step}: 观察结果: {preview}")
            except Exception as e:
                # ❌ 不崩溃：把错误"翻译"成 Agent 能理解的语言
                result = (
                    f"[工具执行失败] {func_name} 抛出异常："
                    f"{type(e).__name__}: {e}"
                )
                print(f"OUT:step{step}: ⚠️ 工具异常，反馈给 Agent：{type(e).__name__}")

            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": str(result),
            })

    print(f"OUT:max_steps: ⚠️ 达到最大步数 {MAX_STEPS}，强制停止！")
    return "(已达到最大步数)"


# ════════════════════════════════════════════════════════════════════
# 离线 mock：Demo A —— 指数退避重试序列
# ════════════════════════════════════════════════════════════════════


def demo_backoff_retry_sequence() -> None:
    """离线演示指数退避重试序列（不消耗 API 额度）。

    mock 一个"前 2 次失败、第 3 次成功"的 API 调用，
    展示退避序列：0.1s → 0.2s → 成功。
    """
    print(f"\n{'=' * 60}")
    print("Demo A: 指数退避重试序列（mock：前 2 次失败，第 3 次成功）")
    print(f"{'=' * 60}")

    call_count = [0]  # 用 list 包裹以便闭包修改

    def mock_flaky_api() -> str:
        """模拟一个不稳定的 API：前 2 次抛连接错误，第 3 次成功。"""
        call_count[0] += 1
        if call_count[0] <= 2:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]
        return "✓ API 调用成功，返回数据"

    backoff_scale = 0.1  # 演示用 0.1s 缩放（生产用 1.0s：1s/2s/4s）

    for attempt in range(MAX_RETRIES):
        try:
            result = mock_flaky_api()
            print(f"OUT:demoA: ✓ 第 {attempt + 1} 次尝试成功！{result}")
            break
        except APIConnectionError as e:
            if not is_retryable(e):
                print(f"OUT:demoA: 永久错误，不重试：{type(e).__name__}")
                raise
            if attempt == MAX_RETRIES - 1:
                print(f"OUT:demoA: 第 {attempt + 1}/{MAX_RETRIES} 次失败，已达上限，放弃。")
                break
            wait = (2 ** attempt) * backoff_scale
            print(
                f"OUT:demoA: 第 {attempt + 1}/{MAX_RETRIES} 次失败（{type(e).__name__}），"
                f"等待 {wait:.1f}s 后重试..."
            )
            time.sleep(wait)

    print(f"OUT:demoA: 💡 生产环境用 backoff_scale=1.0（等待 1s/2s/4s），本章用 0.1 演示。")
    print(f"OUT:demoA: 💡 只重试可重试错误（超时/限流/连接），认证错误立即退出。")


# ════════════════════════════════════════════════════════════════════
# 离线 mock：Demo B —— 工具异常 + Agent 自我纠正
# ════════════════════════════════════════════════════════════════════


def demo_tool_self_correction() -> None:
    """离线演示工具异常 → Agent 自我纠正（不消耗 API 额度）。

    场景：用户问"火星天气"，get_weather 抛 ValueError（火星不在数据库），
    Agent 看到错误后改用 search_wiki 查火星信息，最后诚实回答。
    """
    print(f"\n{'=' * 60}")
    print("Demo B: 工具异常 + Agent 自我纠正（mock 决策序列）")
    print(f"{'=' * 60}")
    print("[场景] 用户问'火星天气'，get_weather 失败，Agent 改用 search_wiki")

    # mock Agent 的决策序列（演示自我纠正逻辑）
    steps = [
        {
            "step": 1,
            "action": "call_tool",
            "tool": "get_weather",
            "args": {"city": "火星"},
        },
        {
            "step": 2,
            "action": "observe_error",
            "error": "ValueError: 城市 '火星' 不在数据库中。可用的城市：北京、上海、深圳、东京",
        },
        {
            "step": 3,
            "action": "call_tool",
            "tool": "search_wiki",
            "args": {"query": "火星"},
            "note": "Agent 看到错误后换工具",
        },
        {
            "step": 4,
            "action": "observe_result",
            "result": "火星是太阳系第四颗行星，表面温度约 -63°C，大气稀薄。",
        },
        {
            "step": 5,
            "action": "final_answer",
            "answer": "我查不到火星的实时天气（不在天气数据库中），但查到百科："
            "火星表面温度约 -63°C，大气稀薄。如果你需要地球城市的天气，请告诉我城市名。",
        },
    ]

    for s in steps:
        step_num = s["step"]
        if s["action"] == "call_tool":
            print(f"OUT:demoB:step{step_num}: 调用 {s['tool']}({s['args']})")
            if "note" in s:
                print(f"OUT:demoB:step{step_num}: 💡 {s['note']}")
        elif s["action"] == "observe_error":
            print(f"OUT:demoB:step{step_num}: ⚠️ 工具异常，反馈给 Agent：{s['error']}")
        elif s["action"] == "observe_result":
            preview = s["result"][:60] + ("..." if len(s["result"]) > 60 else "")
            print(f"OUT:demoB:step{step_num}: 观察结果: {preview}")
        elif s["action"] == "final_answer":
            print(f"OUT:demoB:step{step_num}: ✓ 自我纠正成功！最终回答：{s['answer'][:80]}...")

    print(f"OUT:demoB: 💡 关键：工具异常没让 Agent 崩溃，而是驱动它换工具。")


# ════════════════════════════════════════════════════════════════════
# 离线 mock：Demo C —— 幻觉工具名检测
# ════════════════════════════════════════════════════════════════════


def demo_hallucination_detection() -> None:
    """离线演示幻觉工具名检测 + 纠正（不消耗 API 额度）。

    场景：模型"编造"了一个不存在的工具 get_stock_price，
    代码检测到 → 告知正确工具列表 → 模型改用合法工具。
    """
    print(f"\n{'=' * 60}")
    print("Demo C: 幻觉工具名检测（mock：模型调了不存在的工具）")
    print(f"{'=' * 60}")

    # mock 模型的两次决策
    mock_tool_calls = [
        {
            "step": 1,
            "id": "call_1",
            "name": "get_stock_price",  # ← 幻觉！这个工具不存在
            "args": {"symbol": "AAPL"},
        },
        {
            "step": 2,
            "id": "call_2",
            "name": "search_wiki",  # ← 纠正后改用合法工具
            "args": {"query": "Apple Inc"},
        },
    ]

    for tc in mock_tool_calls:
        step_num = tc["step"]
        func_name = tc["name"]
        print(f"OUT:demoC:step{step_num}: 模型调用工具 '{func_name}'({tc['args']})")

        if func_name not in VALID_TOOL_NAMES:
            # 幻觉检测
            result = (
                f"[错误] 工具 '{func_name}' 不存在。"
                f"可用的工具有：{', '.join(sorted(VALID_TOOL_NAMES))}。"
            )
            print(f"OUT:demoC:step{step_num}: 🚫 幻觉检测：'{func_name}' 不存在！")
            print(f"OUT:demoC:step{step_num}: 告知 Agent：{result}")
        else:
            # 合法工具，正常执行
            result = TOOL_FUNCTIONS[func_name](**tc["args"])
            print(f"OUT:demoC:step{step_num}: ✓ 合法工具，执行成功：{result[:60]}...")

    print(f"OUT:demoC: 💡 关键：幻觉工具没让循环崩溃，而是告知列表让模型纠正。")


# ════════════════════════════════════════════════════════════════════
# 离线 mock：Demo D —— 错误分类（可重试 vs 永久）
# ════════════════════════════════════════════════════════════════════


def _make_mock_response(status_code: int) -> object:
    # SDK 的 APIStatusError 子类在 __init__ 里访问 response.request，
    # 所以需要一个带 .request 属性的真实 httpx.Response（不能传 None）。
    import httpx

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


def demo_error_classification() -> None:
    """离线演示错误分类逻辑（可重试 vs 永久）。

    展示如何用 is_retryable() 判断不同错误类型。
    """
    print(f"\n{'=' * 60}")
    print("Demo D: 错误分类（可重试 vs 永久）")
    print(f"{'=' * 60}")

    # 构造各类错误示例
    test_errors: list[tuple[str, Exception]] = []

    # 超时错误（request 参数可传 None）
    try:
        raise APITimeoutError(request=None)  # type: ignore[arg-type]
    except APITimeoutError as e:
        test_errors.append(("APITimeoutError（超时）", e))

    # 连接错误（request 参数可传 None）
    try:
        raise APIConnectionError(request=None)  # type: ignore[arg-type]
    except APIConnectionError as e:
        test_errors.append(("APIConnectionError（连接失败）", e))

    # 限流错误（需要真实 Response 对象，status_code=429）
    try:
        raise RateLimitError(
            message="Rate limit exceeded",
            response=_make_mock_response(429),  # type: ignore[arg-type]
            body=None,
        )
    except RateLimitError as e:
        test_errors.append(("RateLimitError（限流 429）", e))

    # 认证错误（status_code=401）
    try:
        raise AuthenticationError(
            message="Invalid API key",
            response=_make_mock_response(401),  # type: ignore[arg-type]
            body=None,
        )
    except AuthenticationError as e:
        test_errors.append(("AuthenticationError（认证 401）", e))

    # 参数错误（status_code=400）
    try:
        raise BadRequestError(
            message="Bad request",
            response=_make_mock_response(400),  # type: ignore[arg-type]
            body=None,
        )
    except BadRequestError as e:
        test_errors.append(("BadRequestError（参数 400）", e))

    # 工具异常（非 API 错误）
    try:
        raise ValueError("城市 '火星' 不在数据库中")
    except ValueError as e:
        test_errors.append(("ValueError（工具异常）", e))

    print(f"{'错误类型':<35} {'可重试？':<10} {'处理方式'}")
    print(f"{'-' * 75}")
    for name, err in test_errors:
        retryable = is_retryable(err)
        if retryable:
            action = "退避重试"
        elif isinstance(err, (AuthenticationError, BadRequestError)):
            action = "立即退出（永久错误）"
        else:
            action = "反馈给 Agent（机制 2）"
        flag = "✅ 是" if retryable else "❌ 否"
        print(f"OUT:demoD: {name:<33} {flag:<10} {action}")

    print(f"\nOUT:demoD: 💡 核心：只重试瞬时故障（超时/限流/连接），永久错误立即失败。")
    print(f"OUT:demoD: 💡 混为一谈会导致：认证错误重试 3 次纯属浪费，或网络抖动直接崩溃。")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print(f"[config] MAX_RETRIES={MAX_RETRIES}, MAX_STEPS={MAX_STEPS}")
    print(f"[config] 合法工具: {sorted(VALID_TOOL_NAMES)}")

    api_ok = True

    try:
        # ── 尝试真实 API（占位符密钥会 401，降级到离线 mock）─────────
        resilient_agent_loop(
            "帮我查一下北京、上海两个城市的天气，然后推荐哪个更适合旅行。"
        )
    except AuthenticationError:
        api_ok = False
        print(f"\n[提示] 认证失败（AuthenticationError）—— 这是永久错误，不重试。")
        print(f"[提示] 原因：OPENAI_API_KEY=sk-REPLACE-ME 是占位符。")
        print(f"[提示] 这是机制 4 的体现：永久错误直接退出，不浪费时间重试。")
        print(f"[提示] 已自动降级为离线 mock 演示四大容错机制。\n")
    except BadRequestError as e:
        api_ok = False
        print(f"\n[提示] 请求错误（BadRequestError）—— 永久错误：{e}")
        print(f"[提示] 可能是模型不支持 tools API（如 Ollama qwen2.5vl）。")
        print(f"[提示] 已自动降级为离线 mock 演示。\n")
    except (APITimeoutError, APIConnectionError, RateLimitError) as e:
        api_ok = False
        print(f"\n[提示] 可重试错误耗尽（{type(e).__name__}）—— 已重试 {MAX_RETRIES} 次仍失败。")
        print(f"[提示] 已自动降级为离线 mock 演示。\n")
    except Exception as e:
        api_ok = False
        error_msg = str(e)
        is_auth = (
            "401" in error_msg
            or "invalid_api_key" in error_msg
            or "Authentication" in error_msg
            or "sk-REPLACE-ME" in error_msg
        )
        print(f"\n[提示] API 调用失败（{type(e).__name__}）。")
        if is_auth:
            print(f"[提示] 原因：API 密钥为占位符 sk-REPLACE-ME。请编辑 ai-agent/.env。")
        else:
            print(f"[提示] 原因：{e}")
        print(f"[提示] 已自动降级为离线 mock 演示。\n")

    # ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）──────
    demo_backoff_retry_sequence()
    demo_tool_self_correction()
    demo_hallucination_detection()
    demo_error_classification()

    print(f"\n{'=' * 60}")
    if api_ok:
        print("所有演示完成！（含真实 API 容错 + 四大机制离线 mock）")
    else:
        print("离线演示完成！（真实 API 未配置，但四大容错机制已完整展示）")
    print(f"💡 四大机制：退避重试 / 工具自我纠正 / 幻觉检测 / 错误分类")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
