"""第04章 Agent 循环（The Agent Loop）

本章是全教程的核心概念：**单轮=工具调用，多轮=Agent**。

从第03章的"单轮工具调用"扩展为"多步循环"：
  - Agent 持续 observe→reason→act，直到模型给出最终回答（终止条件 1）
  - max_steps=10 兜底保护，防止无限循环（终止条件 2）

三个演示：
  Demo 1: 多步循环 — 查多城市天气并推荐（需要 3-4 步）
  Demo 2: 单步快速完成 — 简单问题模型直接回答（0 次工具调用）
  Demo 3: max_steps 防护 — mock 模型模拟无限循环，验证 step=10 优雅停止（离线，不耗 API）
"""

import json
import sys
from pathlib import Path

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

# ════════════════════════════════════════════════════════════════════
# 工具实现（复用第03章的 3 个 mock 工具，保持一致）
# ════════════════════════════════════════════════════════════════════


def get_weather(city: str) -> str:
    """查询指定城市的当前天气（mock 数据）。"""
    mock_data = {
        "北京": "北京今天晴, 25°C, 湿度 40%, 东北风 2 级",
        "上海": "上海今天多云, 28°C, 湿度 65%, 东南风 3 级",
        "深圳": "深圳今天小雨, 30°C, 湿度 80%, 南风 2 级",
        "东京": "东京今天阴, 22°C, 湿度 55%, 西风 1 级",
    }
    return mock_data.get(city, f"{city}今天晴, 23°C, 湿度 50%")


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
        "openai": "OpenAI 是 AI 研究公司，开发了 GPT 系列和 ChatGPT。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到与'{query}'相关的百科条目。"


# ════════════════════════════════════════════════════════════════════
# 工具定义（JSON Schema，与第03章一致）
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
                        "description": "城市名称，如'北京'、'上海'",
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
            "description": "执行数学计算，支持加减乘除和括号。例如：'2+3*4'、'(10-2)/4'",
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
                        "description": "搜索关键词，如'python'、'机器学习'",
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

# ⚠️ 必须有上限！这是 Agent 循环的"保险丝"。
MAX_STEPS = 10


# ════════════════════════════════════════════════════════════════════
# 核心：Agent 循环
# ════════════════════════════════════════════════════════════════════


def agent_loop(user_message: str) -> str:
    """Agent 循环：持续调用工具直到模型给出最终回答或达到 max_steps。

    每一步打印状态（step N, action, observation），让你看清 Agent 的决策链。

    终止条件：
      1. 模型不再调用工具 → 返回最终回答（正常完成）
      2. 达到 MAX_STEPS → 强制停止（保险丝）

    Args:
        user_message: 用户的问题/任务。

    Returns:
        Agent 的最终回答字符串。
    """
    # messages 在循环外初始化，循环内只"追加"——模型需要完整历史来决策。
    messages: list = [
        {
            "role": "system",
            "content": (
                "你是一个任务助手 Agent。你可以查天气、做计算、搜百科。"
                "面对复杂任务，请一步步调用工具收集信息，最后给出综合回答。"
                "当信息足够回答时，直接给出最终回答，不要继续调用工具。"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'='*60}")
    print(f"任务: {user_message}")
    print(f"{'='*60}")

    # ── 循环最多 MAX_STEPS 次 ──────────────────────────────────────
    for step in range(1, MAX_STEPS + 1):
        print(f"OUT:step{step}: 思考中... (观察历史，决定下一步)")

        # ── Reason：让 LLM 决定下一步 ──────────────────────────────
        response = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 让模型自己决定是否调工具
        )
        assistant_msg = response.choices[0].message

        # ── 终止条件 1：模型不再调工具 = 任务完成 ──────────────────
        if not assistant_msg.tool_calls:
            answer = assistant_msg.content or "(空回答)"
            print(f"OUT:step{step}: ✓ 任务完成！模型给出最终回答（未调用工具）")
            print(f"OUT:step{step}: 回答: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            return answer

        # ── Act：模型决定调用工具，执行并把结果反馈回去 ───────────
        # 先记住"模型做了什么决策"（assistant 消息含 tool_calls）
        messages.append(assistant_msg.model_dump())

        tool_names = [tc.function.name for tc in assistant_msg.tool_calls]
        print(f"OUT:step{step}: 决定调用工具: {', '.join(tool_names)}")

        # 执行每个工具调用（模型可能一次返回多个）
        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f"OUT:step{step}: 执行 {func_name}({args})")

            # 执行工具（dispatch 模式）
            func = TOOL_FUNCTIONS.get(func_name)
            if func is None:
                result = f"错误：未知工具 '{func_name}'"
            else:
                try:
                    result = func(**args)
                except Exception as e:
                    result = f"工具执行错误：{e}"

            print(f"OUT:step{step}: 观察结果: {result[:80]}{'...' if len(result) > 80 else ''}")

            # ── Observe：把工具结果以 role="tool" 追加到 messages ────
            # 下一轮循环时，模型会"看到"这个结果，并据此决定下一步。
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

        # 循环回到顶部：下一轮的 Reason 会基于更新后的 messages 决策。

    # ── 终止条件 2：达到 max_steps，强制停止 ──────────────────────
    print(f"OUT:max_steps: ⚠️ 达到最大步数 {MAX_STEPS}，强制停止！（防止无限循环）")
    return "(已达到最大步数，可能需要更具体的指令或更好的工具)"


# ════════════════════════════════════════════════════════════════════
# Demo 3：max_steps 防护（离线 mock，不消耗 API 额度）
# ════════════════════════════════════════════════════════════════════


def demo_max_steps_protection() -> None:
    """用 mock 响应演示 max_steps 防护（不消耗 API 额度）。

    模拟一个"总是返回 tool_calls 的模型"——它永远不会给最终回答，
    从而构造出无限循环场景。验证 Agent 循环在 step=10 时优雅停止。

    这个函数不调用真实 API，所以：
      - 不需要有效 API 密钥
      - 不消耗任何 API 额度
      - 可以 100% 可靠地验证 max_steps 逻辑
    """

    # 模拟"无限循环"模型：每次都返回 tool_calls，永远不给最终回答
    # （真实场景：模型能力弱、工具返回模糊结果、任务定义不清时会发生）
    def mock_infinite_loop_model(messages: list) -> dict:
        """Mock：总是返回 get_weather("北京") 的 tool_calls，模拟无限循环。"""
        return {
            "tool_calls": [
                {
                    "id": f"call_mock_{len(messages)}",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "北京"}',
                    },
                }
            ]
        }

    print(f"\n{'='*60}")
    print("Demo 3: max_steps 防护（mock 无限循环场景）")
    print(f"{'='*60}")
    print("[说明] 模拟一个'总在重复调工具'的坏模型，验证 max_steps 兜底。")

    messages: list = [
        {"role": "system", "content": "你是任务助手 Agent..."},
        {"role": "user", "content": "查北京天气（演示无限循环防护）"},
    ]

    # 这个循环结构与真实 agent_loop 完全一致，只是模型换成 mock
    for step in range(1, MAX_STEPS + 1):
        mock_response = mock_infinite_loop_model(messages)
        tool_calls = mock_response["tool_calls"]

        if not tool_calls:
            # mock 永远不会走到这里（它总返回 tool_calls）
            print(f"OUT:max_steps:step{step}: 任务完成（mock 不会走到这）")
            return

        print(f"OUT:max_steps:step{step}: 调用工具（mock 重复调用）")
        # 记录 assistant 决策
        messages.append({"role": "assistant", "tool_calls": tool_calls})

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            result = TOOL_FUNCTIONS[func_name](**args)
            print(f"OUT:max_steps:step{step}: 结果: {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result),
            })

    # ⚠️ 这里是关键：循环正常结束（step 用尽），优雅停止
    print(f"OUT:max_steps: ⚠️ 达到最大步数 {MAX_STEPS}，强制停止！")
    print(f"OUT:max_steps: ✓ 防护生效——没有无限循环，Agent 安全停下。")
    print(f"OUT:max_steps: 💡 真实场景：请检查工具返回是否模糊、任务是否清晰、模型是否足够强。")


# ════════════════════════════════════════════════════════════════════
# 离线 mock Agent 循环（API 不可用时演示完整循环逻辑）
# ════════════════════════════════════════════════════════════════════


def demo_offline_multi_step() -> None:
    """离线演示多步 Agent 循环逻辑（API 不可用时降级使用）。

    用预设的 mock 响应模拟"查三城市天气并推荐"的完整 4 步循环，
    让没有有效 API 密钥的学习者也能看清 Agent 循环的运作方式。
    """
    print(f"\n{'='*60}")
    print("离线演示：多步 Agent 循环（mock 4 步：查三城市 + 推荐）")
    print(f"{'='*60}")

    # 预设一个"聪明"的 mock 模型的决策序列
    mock_decisions = [
        # Step 1: 查北京
        {"action": "get_weather", "args": {"city": "北京"}},
        # Step 2: 查上海
        {"action": "get_weather", "args": {"city": "上海"}},
        # Step 3: 查深圳
        {"action": "get_weather", "args": {"city": "深圳"}},
        # Step 4: 信息够了，给最终回答（不调工具）
        {"action": None, "answer": "推荐北京旅行：晴朗 25°C，温度最宜人；上海多云 28°C 次之；深圳小雨 30°C 较闷热。"},
    ]

    for step, decision in enumerate(mock_decisions, start=1):
        print(f"OUT:offline:step{step}: 思考中...")

        if decision["action"] is None:
            # 终止条件 1：模型决定不调工具，给最终回答
            print(f"OUT:offline:step{step}: ✓ 信息足够，给出最终回答（不再调工具）")
            print(f"OUT:offline:step{step}: 回答: {decision['answer']}")
            break

        # 执行工具
        func = TOOL_FUNCTIONS[decision["action"]]
        result = func(**decision["args"])
        print(f"OUT:offline:step{step}: 调用 {decision['action']}({decision['args']})")
        print(f"OUT:offline:step{step}: 结果: {result}")
    else:
        print(f"OUT:offline: 达到最大步数（此例不会触发）")

    print(f"OUT:offline: ✓ 循环正常终止（模型自主决定停止），共 {step} 步。")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print(f"[config] 工具数量: {len(tools)}")
    print(f"[config] MAX_STEPS={MAX_STEPS}（Agent 循环保险丝）")

    api_ok = True

    try:
        # ── Demo 1: 多步循环（需要 3-4 步）─────────────────────────
        agent_loop(
            "帮我查一下北京、上海、深圳三个城市的天气，"
            "然后推荐哪个城市今天最适合旅行，说明理由。"
        )

        # ── Demo 2: 单步快速完成（简单问题，0 次工具调用）──────────
        agent_loop("你好，请用一句话介绍你自己。")

    except Exception as e:
        api_ok = False
        error_msg = str(e)
        is_auth_error = (
            "401" in error_msg
            or "invalid_api_key" in error_msg
            or "Authentication" in error_msg
            or "sk-REPLACE-ME" in error_msg
        )
        is_tool_unsupported = "does not support tools" in error_msg or (
            "400" in error_msg and "model" in error_msg.lower()
        )

        print(f"\n[提示] 真实 API 调用失败（{type(e).__name__}）。")
        if is_auth_error:
            print(f"[提示] 原因：API 密钥无效或为占位符。请编辑 ai-agent/.env 填入有效密钥。")
            print(f"[提示] 当前 provider={cfg.provider}，需要对应的 API 密钥。")
        elif is_tool_unsupported:
            print(f"[提示] 原因：当前模型 {cfg.model} 不支持 tools API。")
            print(f"[提示] Ollama qwen2.5vl:latest 不支持工具调用。")
            print(f"[提示] 请用支持 function calling 的模型，或在 .env 设 PROVIDER=openai/deepseek。")
        else:
            print(f"[提示] 原因：{e}")
        print(f"[提示] 已自动降级为离线 mock 演示，Agent 循环逻辑不受影响。\n")

    # ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）────
    # Demo 离线多步循环
    demo_offline_multi_step()

    # Demo 3: max_steps 防护（始终演示，不依赖 API）
    demo_max_steps_protection()

    print(f"\n{'='*60}")
    if api_ok:
        print("所有演示完成！（含真实 API 多步循环 + max_steps 防护）")
    else:
        print("离线演示完成！（真实 API 未配置，但 Agent 循环逻辑已完整展示）")
    print(f"💡 核心要点：单轮=工具调用，多步循环=Agent。max_steps 是保险丝。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
