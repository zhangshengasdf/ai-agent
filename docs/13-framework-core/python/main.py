"""
第13章 从零造框架 — 组装 6 大组件，运行完整 Agent

本文件做三件事：
  1. 定义两个工具函数（get_weather / calculate）
  2. 组装 6 大组件（依赖注入）
  3. 运行 Agent：查北京/上海天气 → 算温差 → 给出最终答案

运行方式：
  cd ai-agent/13-framework-core
  python3 python/main.py

输出标记：
  OUT:framework:step{N}: — Observer 记录的每步状态
  OUT:final: — Agent 最终答案
"""

from __future__ import annotations

import ast
import operator
import re
import sys
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# 让章节代码能 import shared.config（T1 确立的约定）
# ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import get_config  # noqa: E402

# import 我们在第13章造的框架
from framework import (  # noqa: E402
    ConversationMemory,
    DefaultAgentRunner,
    DefaultLLMClient,
    InMemoryToolRegistry,
    LoggingObserver,
    OpenAIToolCallParser,
)


# ═══════════════════════════════════════════════════════════════════════
# 工具函数（注册到 ToolRegistry）
# ═══════════════════════════════════════════════════════════════════════

# 模拟天气数据库（演示用，不调真实 API）
_WEATHER_DB: dict[str, dict[str, str]] = {
    "北京": {"condition": "晴", "temp": "25°C"},
    "上海": {"condition": "多云", "temp": "28°C"},
    "广州": {"condition": "雷阵雨", "temp": "32°C"},
    "深圳": {"condition": "晴", "temp": "31°C"},
}


def get_weather(city: str) -> str:
    """查询指定城市的天气（模拟数据）。"""
    city = city.strip()
    if city not in _WEATHER_DB:
        available = "、".join(sorted(_WEATHER_DB.keys()))
        return f"[未找到] 城市 '{city}'。支持的城市：{available}"
    w = _WEATHER_DB[city]
    return f"{city}今天{w['condition']}，气温 {w['temp']}"


# 安全表达式求值：只允许数字 + 四则运算符，杜绝代码注入（第06章教训）
_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def calculate(expression: str) -> str:
    """安全计算数学表达式（只允许数字和四则运算符）。

    用 ast 解析而非 eval()，杜绝任意代码执行。
    """
    expr = expression.strip()
    if not expr:
        return "[错误] 表达式为空"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"[解析失败] {e.msg}: '{expr}'"

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_BIN_OPS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return _SAFE_BIN_OPS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    try:
        result = _eval(tree.body)
        # 整数结果去掉小数点
        if isinstance(result, float) and result.is_integer():
            return f"{expression} = {int(result)}"
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"[错误] 除零: {expression}"
    except (ValueError, TypeError) as e:
        return f"[计算失败] {e}"


# ═══════════════════════════════════════════════════════════════════════
# 工具 schema 定义（OpenAI function calling 格式）
# ═══════════════════════════════════════════════════════════════════════

WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "要查询天气的城市名，例如：北京、上海、广州、深圳",
        },
    },
    "required": ["city"],
}

CALCULATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "数学表达式，支持 + - * / % **，例如：28-25、(3+4)*2",
        },
    },
    "required": ["expression"],
}


# ═══════════════════════════════════════════════════════════════════════
# 组装框架（依赖注入）
# ═══════════════════════════════════════════════════════════════════════

def build_agent() -> DefaultAgentRunner:
    """组装 6 大组件，返回可运行的 AgentRunner 实例。

    依赖注入：每个组件通过构造函数接收它需要的依赖（不 new 具体实现）。
    换任何一块组件，只需改这里，不影响其他代码。
    """
    # 1. 工具注册表：注册 get_weather + calculate
    tools = InMemoryToolRegistry()
    tools.register(
        name="get_weather",
        description="查询指定城市的天气（天气状况 + 气温）",
        parameters=WEATHER_SCHEMA,
        handler=get_weather,
    )
    tools.register(
        name="calculate",
        description="计算数学表达式（支持 + - * / % **）",
        parameters=CALCULATE_SCHEMA,
        handler=calculate,
    )

    # 2. LLM 客户端：注入配置（provider/base_url/api_key/model）
    cfg = get_config()
    llm = DefaultLLMClient(cfg)

    # 3. 记忆：带 system prompt 定义 Agent 人格
    memory = ConversationMemory(
        system_prompt=(
            "你是任务助手 Agent。你可以查询天气、做数学计算。"
            "请根据用户需求，逐步调用工具完成任务，最后给出简洁的结论。"
        )
    )

    # 4. 解析器：解析 OpenAI tool_calls 格式
    parser = OpenAIToolCallParser()

    # 5. 观察者：打印每步日志（OUT:framework:step{N}: 前缀）
    observer = LoggingObserver()

    # 6. 循环引擎：通过构造函数注入上面 5 个组件
    runner = DefaultAgentRunner(
        llm=llm,
        tools=tools,
        memory=memory,
        parser=parser,
        observer=observer,
    )

    return runner


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("第13章 从零造框架 — 实现 6 大核心组件")
    print("把第12章的接口图纸，浇筑成能跑的 mini Agent 框架")
    print("=" * 72)
    print()

    # ── Demo 1：展示 6 大组件的组装 ──
    print("▎ Demo 1: 组装 6 大组件（依赖注入）")
    print("-" * 72)
    print("  1. InMemoryToolRegistry  ← 注册 get_weather + calculate")
    print("  2. DefaultLLMClient      ← 包装 OpenAI SDK + 离线 mock 降级")
    print("  3. ConversationMemory    ← 对话缓冲（带 system prompt）")
    print("  4. OpenAIToolCallParser  ← 解析 tool_calls → [{name, args, id}]")
    print("  5. LoggingObserver       ← 纯旁路日志（OUT:framework:step{N}:）")
    print("  6. DefaultAgentRunner    ← 循环引擎（协调上面 5 个组件）")
    print()

    runner = build_agent()
    print("  ✓ 框架组装完成，开始运行 Agent 循环...")
    print()

    # ── Demo 2：运行 Agent ──
    print("▎ Demo 2: 运行 Agent（查天气 + 算温差）")
    print("-" * 72)

    task = "帮我查一下北京和上海的天气，然后算一下两地温差。"
    print(f"  任务: {task}")
    print()

    answer = runner.run(task, max_steps=10)

    print()
    print(f"OUT:final: {answer}")
    print()

    # ── Demo 3：验证工具独立可用 ──
    print("▎ Demo 3: 工具独立验证（证明组件可单独使用）")
    print("-" * 72)
    print(f"  get_weather('北京')  → {get_weather('北京')}")
    print(f"  get_weather('上海')  → {get_weather('上海')}")
    print(f"  calculate('28-25')  → {calculate('28-25')}")
    print(f"  calculate('(3+4)*2') → {calculate('(3+4)*2')}")
    print(f"  get_weather('未知')  → {get_weather('未知')}")
    print(f"  calculate('1/0')    → {calculate('1/0')}")
    print()

    print("=" * 72)
    print("✓ 本章完成：mini Agent 框架已跑通。")
    print("  第14章会扩展高级特性（流式输出 / 并行工具 / 摘要记忆）。")
    print("=" * 72)


if __name__ == "__main__":
    main()
