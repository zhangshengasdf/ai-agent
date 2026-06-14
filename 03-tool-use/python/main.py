"""第03章 工具调用（Tool Use / Function Calling）

演示完整的单轮工具调用流程：
  Step 1: 发送 user 消息 + tools 定义 → 模型返回 tool_calls
  Step 2: 解析 tool_calls，执行对应工具函数，获取结果
  Step 3: 把工具结果以 role="tool" 消息追加到 messages
  Step 4: 再次调用 API → 模型基于工具结果返回最终文本回答
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
# 工具实现（全部 mock，不调真实 API）
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
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。"
                  "它以简洁易读的语法著称，广泛应用于 Web 开发、数据科学、AI 等领域。",
        "机器学习": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进，"
                    "而无需被显式编程。主要方法包括监督学习、无监督学习和强化学习。",
        "agent": "在 AI 领域，Agent（智能体）是指能够感知环境、做出决策并采取行动的自主系统。"
                 "一个典型的 AI Agent 包含 LLM（大脑）、工具（手）和循环控制（自主性）。",
        "openai": "OpenAI 是一家美国人工智能研究公司，成立于 2015 年。"
                  "它开发了 GPT 系列大语言模型和 ChatGPT，是当前 AI 领域最具影响力的公司之一。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到与'{query}'相关的百科条目。"


# ════════════════════════════════════════════════════════════════════
# 工具定义（JSON Schema 格式，告诉模型有哪些工具可用）
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

# ════════════════════════════════════════════════════════════════════
# 完整单轮工具调用流程
# ════════════════════════════════════════════════════════════════════


def run_tool_flow(user_message: str) -> str:
    """执行完整的单轮工具调用流程，返回最终回答。"""

    messages = [
        {
            "role": "system",
            "content": "你是一个任务助手 Agent。你可以查天气、做计算、搜百科。"
                       "当用户的问题需要用到这些能力时，主动调用对应工具。",
        },
        {"role": "user", "content": user_message},
    ]

    # ── Step 1: 发送请求，让模型决定是否调用工具 ─────────────────────
    print(f"\n{'='*60}")
    print(f"用户提问: {user_message}")
    print(f"{'='*60}")

    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # 让模型自己决定是否调用工具
    )

    assistant_msg = response.choices[0].message

    # ── 检测模型是否返回了 tool_calls ───────────────────────────────
    if not assistant_msg.tool_calls:
        # 模型直接返回了文本，没用工具
        print("OUT:step1: 模型决定不调用工具，直接回答")
        final_answer = assistant_msg.content or "(空回答)"
        print(f"OUT:step4: 最终回答: {final_answer}")
        return final_answer

    # ── Step 2: 解析 tool_calls，执行工具 ────────────────────────────
    print(f"OUT:step1: 模型决定调用工具:")
    for tc in assistant_msg.tool_calls:
        func_name = tc.function.name
        func_args = tc.function.arguments
        print(f"  → {func_name}({func_args})")

    # 把 assistant 的 tool_calls 消息追加到 messages
    messages.append(assistant_msg.model_dump())

    # 执行每个工具调用
    for tc in assistant_msg.tool_calls:
        func_name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}

        # 执行工具
        func = TOOL_FUNCTIONS.get(func_name)
        if func is None:
            result = f"错误：未知工具 '{func_name}'"
        else:
            try:
                result = func(**args)
            except Exception as e:
                result = f"工具执行错误：{e}"

        print(f"OUT:step2: 工具执行结果: {func_name} → {result}")

        # ── Step 3: 把工具结果以 role="tool" 追加到 messages ─────────
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })

    print("OUT:step3: 将结果反馈给模型...")

    # ── Step 4: 再次调用 API，模型基于工具结果给出最终回答 ──────────
    response2 = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
    )

    final_answer = response2.choices[0].message.content or "(空回答)"
    print(f"OUT:step4: 最终回答: {final_answer}")
    return final_answer


# ════════════════════════════════════════════════════════════════════
# 演示：三个不同类型的工具调用
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print(f"[config] 工具数量: {len(tools)}")
    print(f"[config] 可用工具: {', '.join(t['function']['name'] for t in tools)}")

    try:
        # 演示 1: 天气查询
        run_tool_flow("北京今天天气怎么样？")

        # 演示 2: 数学计算
        run_tool_flow("帮我算一下 (15 + 27) * 3 - 18 等于多少")

        # 演示 3: 百科搜索
        run_tool_flow("什么是 Agent？给我简单介绍一下")

        print(f"\n{'='*60}")
        print("所有演示完成！")
        print(f"{'='*60}")
    except Exception as e:
        error_msg = str(e)
        is_auth_error = "401" in error_msg or "invalid_api_key" in error_msg or "Authentication" in error_msg
        is_tool_unsupported = "does not support tools" in error_msg or "400" in error_msg

        if is_auth_error:
            print(f"\n[提示] API 密钥无效或未配置。请编辑 ai-agent/.env 填入有效的 API 密钥。")
            print(f"[提示] 当前 provider={cfg.provider}，需要对应的密钥。")
        elif is_tool_unsupported:
            print(f"\n[提示] 当前模型 {cfg.model} 不支持 tools API。")
            print(f"[提示] 请使用支持 function calling 的模型，如 gpt-4o-mini 或 deepseek-chat。")
            print(f"[提示] 可在 .env 中设置 PROVIDER=openai 或 PROVIDER=deepseek。")
        else:
            print(f"\n[错误] {e}")

        # 仍然演示工具函数本身可以工作
        print(f"\n{'='*60}")
        print("本地工具函数测试（无需 API）:")
        print(f"  get_weather('北京') → {get_weather('北京')}")
        print(f"  calculate('2+3*4') → {calculate('2+3*4')}")
        print(f"  search_wiki('python') → {search_wiki('python')}")
        print("工具函数全部正常！配置好 API 密钥后即可运行完整流程。")
        print(f"{'='*60}")
