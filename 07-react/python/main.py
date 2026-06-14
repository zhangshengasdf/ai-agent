"""第07章 ReAct 模式（Reasoning + Acting）

本章对比两种 Agent 推理范式：

  显式 ReAct（经典文本格式）：
    - 模型输出 "Thought: ... Action: tool_name[args]" 文本格式
    - 手动正则解析提取 Thought 和 Action
    - 执行工具，把结果以 "Observation: ..." 追加到 prompt
    - 循环直到模型输出 "Final Answer: ..."
    - 推理过程完全可见，任何模型都能用（不需要 tools API）

  隐式 ReAct（现代 tools API）：
    - 用 tools API，模型在内部推理后输出 tool_calls
    - 开发者看不到推理过程（黑盒）
    - 结构化输出，更稳定但可调试性低

本章三个演示：
  Demo 1: 显式 ReAct — Thought→Action→Observation 文本格式循环
  Demo 2: 隐式 ReAct — tools API 的 tool_calls 序列
  Demo 3: 对比输出 — 两种范式解决同一问题的并排对比

离线 mock 设计：
  .env 的 OPENAI_API_KEY=sk-REPLACE-ME 是占位符，真实 API 调用必失败。
  所有 demo 先 try 真实 API（失败时降级），然后用离线 mock 100% 可靠地
  演示完整 ReAct 流程，保证 exit code 0。
"""

import json
import re
import sys
from pathlib import Path

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI  # noqa: E402

from shared.config import get_config  # noqa: E402

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


# ════════════════════════════════════════════════════════════════════
# 工具实现（复用第03/04章的 mock 工具，保持一致）
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
        return "错误：表达式包含不允许的字符，只支持数字和 + - * / ( )"
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
        "北京": "北京是中国的首都，著名景点有故宫、长城。",
        "上海": "上海是中国最大的城市，著名景点有外滩、东方明珠。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到与'{query}'相关的百科条目。"


# 工具名 → 函数的映射（dispatch 模式）
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_wiki": search_wiki,
}

# ⚠️ 必须有上限！
MAX_STEPS = 10


# ════════════════════════════════════════════════════════════════════
# 工具定义（JSON Schema，用于隐式 ReAct）
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
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ════════════════════════════════════════════════════════════════════
# 显式 ReAct：ReAct System Prompt（含格式约束 + Few-shot 示例）
# ════════════════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = """\
你是一个任务助手 Agent。请严格使用以下 ReAct 格式回答问题。

可用工具（用 Action: 工具名[参数] 调用）：
- get_weather[城市名]: 查询城市天气，如 get_weather[北京]
- calculate[数学表达式]: 数学计算，如 calculate[28-25]
- search_wiki[关键词]: 搜索百科，如 search_wiki[北京]

格式规则（必须严格遵守）：
Thought: 你的推理过程（1-2 句话，说明你现在知道什么、下一步该干嘛）
Action: 工具名[参数]

（系统会自动追加 Observation，你不需要自己写 Observation）

当你通过工具调用收集到足够信息后，用以下格式给出最终答案：
Thought: 信息已足够，我现在知道答案。
Final Answer: 你的最终回答

示例：
问题: 上海和深圳哪个温度更高？
Thought: 我需要分别查两个城市的温度。先查上海。
Action: get_weather[上海]

Observation: 上海今天多云, 28°C, 湿度 65%, 东南风 3 级
Thought: 上海是 28°C。现在查深圳的温度。
Action: get_weather[深圳]

Observation: 深圳今天小雨, 30°C, 湿度 80%, 南风 2 级
Thought: 上海 28°C，深圳 30°C。深圳温度更高，高了 2°C。我现在知道答案了。
Final Answer: 深圳温度更高（30°C > 上海 28°C），高 2°C。

现在请回答以下问题："""


# ════════════════════════════════════════════════════════════════════
# 显式 ReAct：解析逻辑（正则提取 Thought / Action / Final Answer）
# ════════════════════════════════════════════════════════════════════


def parse_react_output(text: str) -> dict:
    """解析模型的 ReAct 文本输出。

    返回 dict，包含以下键之一：
      - {"type": "final_answer", "thought": str, "answer": str}
      - {"type": "action", "thought": str, "tool": str, "args": str}
      - {"type": "parse_error", "raw": str}
    """
    # ── 情况 1：模型输出 Final Answer → 任务完成 ──
    if "Final Answer:" in text:
        answer = text.split("Final Answer:")[1].strip()
        thought = ""
        if "Thought:" in text:
            thought_match = re.search(
                r"Thought:\s*(.*?)(?:\nFinal Answer:)", text, re.DOTALL
            )
            if thought_match:
                thought = thought_match.group(1).strip()
        return {"type": "final_answer", "thought": thought, "answer": answer}

    # ── 情况 2：模型输出 Thought + Action → 需要执行工具 ──
    match = re.search(
        r"Thought:\s*(.*?)\nAction:\s*(\w+)\[(.*?)\]", text, re.DOTALL
    )
    if match:
        return {
            "type": "action",
            "thought": match.group(1).strip(),
            "tool": match.group(2),
            "args": match.group(3),
        }

    # ── 情况 3：格式错误，既没有 Final Answer 也没有 Action ──
    return {"type": "parse_error", "raw": text}


# ════════════════════════════════════════════════════════════════════
# 显式 ReAct：主循环（文本格式，不使用 tools API）
# ════════════════════════════════════════════════════════════════════


def explicit_react_loop(user_message: str) -> str:
    """显式 ReAct 循环：模型输出 Thought/Action 文本，手动解析执行。

    与隐式 ReAct 的核心区别：
      - 不传 tools 参数给 API（纯文本补全）
      - 手动正则解析模型输出
      - 推理过程（Thought）完全可见

    Args:
        user_message: 用户的问题。

    Returns:
        Agent 的最终回答字符串。
    """
    # prompt 在循环外初始化，循环内只追加 Observation
    prompt = f"{REACT_SYSTEM_PROMPT}\n问题: {user_message}\n"

    print(f"\n{'='*60}")
    print(f"显式 ReAct 任务: {user_message}")
    print(f"{'='*60}")

    for step in range(1, MAX_STEPS + 1):
        # ── Reason：让模型输出 Thought/Action 文本（注意：不传 tools）──
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            stop=["Observation:"],  # 让模型停在 Action 后，不自己编造 Observation
        )
        model_text = response.choices[0].message.content or ""

        # ── 解析模型输出 ──
        parsed = parse_react_output(model_text)

        if parsed["type"] == "final_answer":
            print(f"OUT:explicit:step{step}: Thought: {parsed['thought'][:80]}")
            print(f"OUT:explicit:step{step}: ✓ 检测到 Final Answer，终止循环")
            print(f"OUT:explicit:step{step}: 最终答案: {parsed['answer'][:120]}")
            return parsed["answer"]

        if parsed["type"] == "action":
            thought = parsed["thought"]
            tool_name = parsed["tool"]
            args = parsed["args"]
            print(f"OUT:explicit:step{step}: Thought: {thought[:80]}")
            print(f"OUT:explicit:step{step}: Action: {tool_name}[{args}]")

            # 执行工具
            func = TOOL_FUNCTIONS.get(tool_name)
            if func is None:
                result = f"错误：未知工具 '{tool_name}'"
            else:
                try:
                    result = func(args)
                except Exception as e:
                    result = f"工具执行错误：{e}"

            print(f"OUT:explicit:step{step}: Observation: {result[:80]}")

            # 关键：把 Observation 追加到 prompt，让模型下一步看到结果
            prompt += f"Thought: {thought}\nAction: {tool_name}[{args}]\n\nObservation: {result}\n"
            continue

        # 格式错误：提醒模型重新格式化
        print(f"OUT:explicit:step{step}: ⚠️ 格式解析失败，提醒模型重新格式化")
        prompt += f"\n（格式错误。请用 Thought:/Action:/Final Answer: 格式重新回答。）\n{model_text}\n"

    print(f"OUT:explicit: ⚠️ 达到最大步数 {MAX_STEPS}，强制停止！")
    return "(已达到最大步数)"


# ════════════════════════════════════════════════════════════════════
# 隐式 ReAct：tools API 循环（复用第04章模式，对比用）
# ════════════════════════════════════════════════════════════════════


def implicit_react_loop(user_message: str) -> str:
    """隐式 ReAct 循环：用 tools API，模型内部推理后输出 tool_calls。

    与显式 ReAct 的核心区别：
      - 传 tools 参数给 API
      - 模型在内部推理，不输出 Thought 文本
      - 结构化 tool_calls，无需正则解析

    Args:
        user_message: 用户的问题。

    Returns:
        Agent 的最终回答字符串。
    """
    messages: list = [
        {
            "role": "system",
            "content": (
                "你是一个任务助手 Agent。你可以查天气、做计算、搜百科。"
                "面对复杂任务，请一步步调用工具收集信息，最后给出综合回答。"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'='*60}")
    print(f"隐式 ReAct 任务: {user_message}")
    print(f"{'='*60}")
    print("[说明] 模型在内部推理，我们只能看到 tool_calls（推理过程不可见）")

    for step in range(1, MAX_STEPS + 1):
        # ── Reason：模型内部推理后输出 tool_calls（或最终回答）──
        response = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_msg = response.choices[0].message

        # 终止条件：模型不调工具 = 任务完成
        if not assistant_msg.tool_calls:
            answer = assistant_msg.content or "(空回答)"
            print(f"OUT:implicit:step{step}: ✓ 模型给出最终回答（无 tool_calls）")
            print(f"OUT:implicit:step{step}: 最终答案: {answer[:120]}")
            return answer

        # 模型决定调工具
        messages.append(assistant_msg.model_dump())
        tool_names = [tc.function.name for tc in assistant_msg.tool_calls]
        print(f"OUT:implicit:step{step}: tool_calls: {', '.join(tool_names)}")
        print(f"OUT:implicit:step{step}: （推理过程不可见——模型在内部完成决策）")

        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            func = TOOL_FUNCTIONS.get(func_name)
            if func is None:
                result = f"错误：未知工具 '{func_name}'"
            else:
                try:
                    result = func(**args)
                except Exception as e:
                    result = f"工具执行错误：{e}"

            print(f"OUT:implicit:step{step}: 执行 {func_name}({args}) → {result[:60]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )

    print(f"OUT:implicit: ⚠️ 达到最大步数 {MAX_STEPS}，强制停止！")
    return "(已达到最大步数)"


# ════════════════════════════════════════════════════════════════════
# 离线 mock：显式 ReAct（API 不可用时演示完整 Thought→Action→Observation 循环）
# ════════════════════════════════════════════════════════════════════


def demo_explicit_react_offline() -> str:
    """离线演示显式 ReAct：预设模型的文本输出，展示完整解析+执行流程。

    模拟问题："北京和上海哪个温度更高？"
    预设 3 步轨迹：
      Step 1: Thought + Action: get_weather[北京] → Observation
      Step 2: Thought + Action: get_weather[上海] → Observation
      Step 3: Thought + Final Answer
    """
    print(f"\n{'='*60}")
    print("离线演示：显式 ReAct（Thought → Action → Observation）")
    print(f"{'='*60}")
    print("[说明] 预设模型输出，演示完整的 ReAct 文本解析 + 工具执行流程")

    question = "北京和上海哪个温度更高？"
    print(f"问题: {question}\n")

    # 预设模型每一步的文本输出（模拟一个遵循 ReAct 格式的"好模型"）
    mock_model_outputs = [
        # Step 1
        "Thought: 我需要分别查北京和上海的温度才能比较。先查北京。\n"
        "Action: get_weather[北京]",
        # Step 2（模型看到了 step1 的 Observation 后的输出）
        "Thought: 北京是 25°C。现在查上海的温度。\n"
        "Action: get_weather[上海]",
        # Step 3（模型看到了 step2 的 Observation 后的输出）
        "Thought: 北京 25°C，上海 28°C。上海温度更高，高了 3°C。我现在知道答案了。\n"
        "Final Answer: 上海温度更高（28°C > 北京 25°C），高 3°C。",
    ]

    # 模拟 prompt 追加（与真实 explicit_react_loop 逻辑一致）
    prompt = f"{REACT_SYSTEM_PROMPT}\n问题: {question}\n"
    final_answer = "(无)"

    for step, model_text in enumerate(mock_model_outputs, start=1):
        print(f"--- Step {step} ---")

        # 用真实的解析函数解析（验证解析逻辑正确）
        parsed = parse_react_output(model_text)
        print(f"OUT:explicit:step{step}: 模型原始输出:")
        for line in model_text.split("\n"):
            print(f"  │ {line}")

        if parsed["type"] == "final_answer":
            final_answer = parsed["answer"]
            print(f"OUT:explicit:step{step}: 解析结果: final_answer")
            print(f"OUT:explicit:step{step}: Thought: {parsed['thought']}")
            print(f"OUT:explicit:step{step}: ✓ 终止条件触发，返回最终答案")
            print(f"OUT:explicit:step{step}: 最终答案: {final_answer}")
            break

        if parsed["type"] == "action":
            print(f"OUT:explicit:step{step}: 解析结果: action")
            print(f"OUT:explicit:step{step}: Thought: {parsed['thought']}")
            print(f"OUT:explicit:step{step}: Action: {parsed['tool']}[{parsed['args']}]")

            # 执行真实工具
            func = TOOL_FUNCTIONS[parsed["tool"]]
            observation = func(parsed["args"])
            print(f"OUT:explicit:step{step}: Observation: {observation}")

            # 追加到 prompt（模拟真实循环）
            prompt += (
                f"Thought: {parsed['thought']}\n"
                f"Action: {parsed['tool']}[{parsed['args']}]\n\n"
                f"Observation: {observation}\n"
            )
            print(f"OUT:explicit:step{step}: 已将 Observation 追加到 prompt，继续下一步\n")

    print(f"\nOUT:explicit: ✓ 显式 ReAct 完成，共 {step} 步。")
    print(f"OUT:explicit: 推理过程完全可见（每步的 Thought 都在输出里）。")
    return final_answer


# ════════════════════════════════════════════════════════════════════
# 离线 mock：隐式 ReAct（API 不可用时演示 tool_calls 序列）
# ════════════════════════════════════════════════════════════════════


def demo_implicit_react_offline() -> str:
    """离线演示隐式 ReAct：预设 tool_calls 序列，展示 tools API 推理流程。

    模拟同一问题："北京和上海哪个温度更高？"
    对比显式版：这里看不到 Thought，只有结构化的 tool_calls。
    """
    print(f"\n{'='*60}")
    print("离线演示：隐式 ReAct（tools API，推理过程不可见）")
    print(f"{'='*60}")
    print("[说明] 预设 tool_calls 序列，演示隐式推理流程")

    question = "北京和上海哪个温度更高？"
    print(f"问题: {question}\n")

    # 预设 tool_calls 序列（模拟 tools API 的输出）
    mock_tool_calls_sequence: list = [
        # Step 1: 查北京
        [{"name": "get_weather", "arguments": {"city": "北京"}}],
        # Step 2: 查上海
        [{"name": "get_weather", "arguments": {"city": "上海"}}],
        # Step 3: 无 tool_calls = 最终回答
        None,
    ]

    final_answer = "(无)"

    for step, tool_calls in enumerate(mock_tool_calls_sequence, start=1):
        print(f"--- Step {step} ---")

        if tool_calls is None:
            # 终止条件：模型不调工具，给出最终回答
            final_answer = "上海温度更高（28°C > 北京 25°C），高 3°C。"
            print(f"OUT:implicit:step{step}: response.choices[0].message.tool_calls = null")
            print(f"OUT:implicit:step{step}: ✓ 终止条件触发（无 tool_calls = 任务完成）")
            print(f"OUT:implicit:step{step}: 最终答案: {final_answer}")
            break

        for tc in tool_calls:
            func_name = tc["name"]
            args = tc["arguments"]
            print(f"OUT:implicit:step{step}: tool_calls: [{func_name}({args})]")
            print(f"OUT:implicit:step{step}: （推理不可见——模型在内部决定先查这个城市）")

            # 执行真实工具
            result = TOOL_FUNCTIONS[func_name](**args)
            print(f"OUT:implicit:step{step}: 执行结果: {result}")
            print(f"OUT:implicit:step{step}: 已将结果以 role=tool 追加到 messages\n")

    print(f"\nOUT:implicit: ✓ 隐式 ReAct 完成，共 {step} 步。")
    print(f"OUT:implicit: 推理过程不可见（只有 tool_calls，没有 Thought）。")
    return final_answer


# ════════════════════════════════════════════════════════════════════
# 对比输出：显式 ReAct vs 隐式 ReAct 并排对比
# ════════════════════════════════════════════════════════════════════


def demo_comparison() -> None:
    """并排对比显式 ReAct 和隐式 ReAct 的核心差异。"""
    print(f"\n{'='*60}")
    print("对比：显式 ReAct vs 隐式 ReAct")
    print(f"{'='*60}")

    comparisons = [
        ("推理可见性", "✓ Thought 文本完全可见", "✗ 模型内部推理（黑盒）"),
        ("工具调用格式", "文本 Action: name[args]", "结构化 tool_calls (JSON)"),
        ("解析方式", "正则 re.search 手动解析", "SDK 自动解析（无正则）"),
        ("格式健壮性", "⚠️ 模型可能不遵循格式", "✓ JSON Schema 约束，更稳定"),
        ("可调试性", "✓ 高（看 Thought 排查逻辑）", "⚠️ 低（推理不可见）"),
        ("Token 成本", "⚠️ Thought 占额外 token", "✓ 无 Thought 开销"),
        ("模型兼容性", "✓ 任何模型（纯文本）", "⚠️ 需支持 tools API"),
        ("API 参数", "不传 tools（纯文本补全）", "传 tools + tool_choice"),
        ("适合场景", "教学/调试/小模型", "生产/大型应用"),
    ]

    # 表格输出
    header = f"{'维度':<14} │ {'显式 ReAct':<28} │ {'隐式 ReAct':<28}"
    print(f"OUT:compare: {header}")
    print(f"OUT:compare: {'─'*14}─┼─{'─'*28}─┼─{'─'*28}")
    for dim, explicit, implicit in comparisons:
        row = f"{dim:<14} │ {explicit:<28} │ {implicit:<28}"
        print(f"OUT:compare: {row}")

    print(f"\nOUT:compare: 核心洞察：")
    print(f"OUT:compare: • 显式 ReAct = 推理透明 + 格式脆弱（需 Prompt 工程）")
    print(f"OUT:compare: • 隐式 ReAct = 推理黑盒 + 结构稳定（需 tools API 支持）")
    print(f"OUT:compare: • 两者共享同一个循环骨架（for step in range(MAX_STEPS)）")
    print(f"OUT:compare: • 现代框架默认用隐式 ReAct，但理解显式版能看透底层逻辑")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print(f"[config] 工具数量: {len(tools)}")
    print(f"[config] MAX_STEPS={MAX_STEPS}")
    print(f"[config] 显式 ReAct: 纯文本格式（不传 tools 参数）")
    print(f"[config] 隐式 ReAct: tools API（传 tools 参数）")

    api_ok = True
    question = "北京和上海哪个温度更高？"

    try:
        # ── Demo 1: 显式 ReAct（真实 API）──
        print(f"\n{'#'*60}")
        print("# Demo 1: 显式 ReAct（Thought → Action → Observation）")
        print(f"{'#'*60}")
        explicit_react_loop(question)

        # ── Demo 2: 隐式 ReAct（真实 API）──
        print(f"\n{'#'*60}")
        print("# Demo 2: 隐式 ReAct（tools API）")
        print(f"{'#'*60}")
        implicit_react_loop(question)

    except Exception as e:
        api_ok = False
        error_msg = str(e)
        is_auth_error = (
            "401" in error_msg
            or "invalid_api_key" in error_msg
            or "Authentication" in error_msg
            or "sk-REPLACE-ME" in error_msg
        )

        print(f"\n[提示] 真实 API 调用失败（{type(e).__name__}）。")
        if is_auth_error:
            print(f"[提示] 原因：API 密钥无效或为占位符。请编辑 ai-agent/.env 填入有效密钥。")
            print(f"[提示] 当前 provider={cfg.provider}，需要对应的 API 密钥。")
        else:
            print(f"[提示] 原因：{e}")
        print(f"[提示] 已自动降级为离线 mock 演示，ReAct 逻辑不受影响。\n")

    # ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）────
    demo_explicit_react_offline()
    demo_implicit_react_offline()
    demo_comparison()

    print(f"\n{'='*60}")
    if api_ok:
        print("所有演示完成！（含真实 API + 离线 mock + 对比）")
    else:
        print("离线演示完成！（真实 API 未配置，但 ReAct 逻辑已完整展示）")
    print(f"💡 核心要点：显式 ReAct 推理可见，隐式 ReAct 结构稳定。")
    print(f"💡 两者共享同一个循环骨架，区别只在推理如何表达。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
