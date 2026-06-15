"""
第14章 从零造框架 — 高级特性（流式 / 结构化输出 / 工具校验 / 现代框架对比）

本文件在第13章自造框架的概念基础上，演示 4 个生产级特性：

  Demo 1: 流式输出（Streaming）
          - 用 client.chat.completions.create(stream=True) 逐块输出
          - 离线 mock：模拟逐块 yield（time.sleep 制造延迟）
          - 输出标记：OUT:stream:chunk:

  Demo 2: 结构化输出强制（StructuredOutput）
          - response_format={"type": "json_object"} + Pydantic 校验
          - 校验失败 → 重试（最多 max_retries 次），把错误反馈给 LLM
          - 离线 mock：预设 JSON + 模拟"一次失败 + 重试成功"
          - 输出标记：OUT:structured:attempt: / OUT:structured:result: / OUT:structured:retry:

  Demo 3: 工具参数校验（ToolValidation）
          - 执行工具前用 JSON Schema 校验 args（类型 + 必填字段）
          - 类型不匹配 → 报错，反馈给 Agent
          - 纯逻辑，100% 离线可运行
          - 输出标记：OUT:validate:pass: / OUT:validate:fail:

  Demo 4: 现代框架对比（OpenAI Agents SDK / Pydantic AI）
          - try-import 现代框架；未安装则打印注释代码片段
          - 展示"现代框架用更少代码完成同样功能"
          - 输出标记：OUT:compare:

运行方式：
  cd ai-agent/14-framework-advanced
  python3 python/main.py

设计原则：
  - 概念上引用第13章框架（AgentRunner/ToolRegistry/LLMClient 6 大组件）
  - 不强制 import 第13章代码（独立可运行，便于单独学习）
  - .env 用占位符 sk-REPLACE-ME → 真实 API 必失败 → 降级离线 mock，保证 exit 0
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

# ──────────────────────────────────────────────────────────────────────
# 让章节代码能 import shared.config（T1 确立的约定）
# ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.config import get_config  # noqa: E402

from openai import (  # noqa: E402
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
)
from pydantic import BaseModel, Field, ValidationError  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# 工具：客户端初始化（复用 shared.config，与 T2-T14 一致）
# ═══════════════════════════════════════════════════════════════════════

def make_client() -> OpenAI:
    """从 shared.config 读 provider/base_url/api_key，初始化 OpenAI 客户端。"""
    cfg = get_config()
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


# ═══════════════════════════════════════════════════════════════════════
# Demo 1: 流式输出（Streaming）
# ═══════════════════════════════════════════════════════════════════════

def stream_real_api(client: OpenAI, model: str, prompt: str) -> str | None:
    """尝试真实流式 API。失败返回 None（由上层降级 mock）。"""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是任务助手 Agent。用 30 字内回答。"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        collected: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                # 逐块打印（演示流式效果）
                print(f"OUT:stream:chunk: {delta}", end="", flush=True)
                collected.append(delta)
        print()  # 换行收尾
        return "".join(collected)
    except (AuthenticationError, APIConnectionError, APIError) as e:
        print(f"OUT:stream:offline: 真实 API 不可用（{type(e).__name__}），降级 mock 流式")
        return None


def stream_mock(prompt: str) -> str:
    """离线 mock 流式：把完整回答切成字符块，逐块 yield，模拟流式效果。

    设计要点：
      - 用 time.sleep(0.05) 制造可见的"逐块到达"延迟（演示用，生产无延迟）
      - 按词切片（不是逐字符）→ 更接近真实 SDK 行为（真实 API 是按 token 块下发）
    """
    full_text = (
        f"收到任务：『{prompt}』。我是任务助手 Agent，"
        "已经准备好帮你查询天气、做计算、规划任务。请告诉我具体需求。"
    )
    # 按 2-4 字一组切片，模拟真实 API 的 token 块
    chunks = re.findall(r".{1,3}", full_text, flags=re.DOTALL)
    collected: list[str] = []
    for chunk in chunks:
        time.sleep(0.05)  # 演示用，让"逐块"可见
        print(f"OUT:stream:chunk: {chunk}", end="", flush=True)
        collected.append(chunk)
    print()
    return "".join(collected)


def demo_streaming() -> None:
    """Demo 1: 流式输出。"""
    print("=" * 72)
    print("Demo 1: 流式输出（Streaming）")
    print("  把 LLM 响应逐 token 块下发，而不是一次性返回。")
    print("  价值：用户体验好（首字延迟低）、可中断、长文本不卡 UI。")
    print("=" * 72)
    print()

    cfg = get_config()
    client = make_client()
    prompt = "你好，请简短介绍一下你能做什么。"

    print(f"  用户输入: {prompt}")
    print("  流式输出（逐块到达）↓")
    print("-" * 72)

    # 优先尝试真实 API，失败降级 mock
    result = stream_real_api(client, cfg.model, prompt)
    if result is None:
        result = stream_mock(prompt)

    print("-" * 72)
    print(f"OUT:stream:done: 共收到 {len(result)} 字符（流式完成）")
    print()
    print("  💡 在真实应用中，每个 chunk 可以直接 write 到前端 SSE/WebSocket，")
    print("     用户看到的是『打字机效果』，而不是『空白等待 → 一大段文字』。")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 2: 结构化输出强制（StructuredOutput + 失败重试）
# ═══════════════════════════════════════════════════════════════════════

class TaskSummary(BaseModel):
    """任务摘要结构（Pydantic 模型，定义 JSON Schema）。"""

    name: str = Field(..., description="任务名称（简短）")
    description: str = Field(..., description="任务详细描述")
    difficulty: str = Field(..., description="难度：easy | medium | hard")
    priority: int = Field(..., ge=1, le=5, description="优先级 1-5（5 最高）")
    estimated_hours: float = Field(..., gt=0, description="预估工时（小时）")


def structured_real_api(
    client: OpenAI, model: str, task: str, max_retries: int = 3
) -> TaskSummary | None:
    """真实 API + Pydantic 校验 + 失败重试。

    返回 None 表示 API 不可用（由上层降级 mock）。
    重试逻辑：校验失败 → 把错误反馈给 LLM → 让它修正输出。
    """
    system_prompt = (
        "你是任务分析助手。把用户的任务描述解析成结构化 JSON，包含字段："
        "name, description, difficulty (easy|medium|hard), priority (1-5 整数), "
        "estimated_hours (正浮点数)。只返回 JSON，不要多余文字。"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"任务：{task}"},
    ]

    for attempt in range(1, max_retries + 1):
        print(f"OUT:structured:attempt: 第 {attempt}/{max_retries} 次尝试...")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
        except (AuthenticationError, APIConnectionError, APIError) as e:
            print(f"OUT:structured:offline: API 不可用（{type(e).__name__}），降级 mock")
            return None

        try:
            result = TaskSummary.model_validate_json(raw)
            print(f"OUT:structured:result: ✓ 校验通过 → {result.model_dump()}")
            return result
        except ValidationError as ve:
            print(f"OUT:structured:retry: ✗ 校验失败（第 {attempt} 次）")
            print(f"  错误: {ve.errors()[:2]}")  # 只打印前 2 条错误
            # 关键：把错误反馈给 LLM，让它下次输出修正后的 JSON
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"上次输出校验失败：{ve.json()}。"
                    "请严格按 schema 重新输出 JSON。"
                ),
            })

    print("OUT:structured:fail: 重试耗尽，仍无法通过校验。")
    return None


def structured_mock(task: str) -> TaskSummary:
    """离线 mock：预设"一次失败 + 重试成功"的轨迹，演示重试机制。

    第 1 次返回缺字段的非法 JSON（触发 ValidationError），
    第 2 次返回完整合法 JSON（校验通过）。
    """
    max_retries = 3
    # mock 序列：第 1 次故意缺 priority 字段，第 2 次才完整
    mock_responses = [
        # ❌ 故意缺 priority 和 estimated_hours（触发校验失败）
        json.dumps({
            "name": "实现登录功能",
            "description": "完成用户登录的 API 和前端表单",
            "difficulty": "medium",
        }, ensure_ascii=False),
        # ✓ 完整合法
        json.dumps({
            "name": "实现登录功能",
            "description": "完成用户登录的 API（JWT 鉴权）和前端表单（含校验）",
            "difficulty": "medium",
            "priority": 4,
            "estimated_hours": 8.0,
        }, ensure_ascii=False),
    ]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "(mock) 任务分析助手"},
        {"role": "user", "content": f"任务：{task}"},
    ]

    for attempt in range(1, max_retries + 1):
        print(f"OUT:structured:attempt: 第 {attempt}/{max_retries} 次尝试（mock）...")
        time.sleep(0.1)  # 演示用，让重试过程可见

        # mock：第 1 次返回残缺 JSON，之后返回完整 JSON
        raw = mock_responses[0] if attempt == 1 else mock_responses[1]

        try:
            result = TaskSummary.model_validate_json(raw)
            print(f"OUT:structured:result: ✓ 校验通过 → {result.model_dump()}")
            return result
        except ValidationError as ve:
            print(f"OUT:structured:retry: ✗ 校验失败（第 {attempt} 次）")
            missing = [err["loc"][0] for err in ve.errors()]
            print(f"  缺失字段: {missing}（LLM 漏掉了这些字段）")
            # 把错误反馈给 LLM（模拟真实流程）
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"上次输出缺少字段 {missing}，请补全后重新输出。",
            })

    raise RuntimeError("mock 序列设计错误：应该在第 2 次成功")


def demo_structured_output() -> None:
    """Demo 2: 结构化输出强制 + 校验失败重试。"""
    print("=" * 72)
    print("Demo 2: 结构化输出强制（StructuredOutput）")
    print("  response_format=json_object + Pydantic 校验 + 失败重试。")
    print("  价值：LLM 输出 100% 符合 schema，下游代码可直接使用。")
    print("=" * 72)
    print()

    cfg = get_config()
    client = make_client()
    task = "实现用户登录功能（含 JWT 鉴权和前端表单校验）"

    print(f"  任务: {task}")
    print(f"  目标 schema: TaskSummary(name, description, difficulty, priority, estimated_hours)")
    print("-" * 72)

    # 优先真实 API，失败降级 mock
    result = structured_real_api(client, cfg.model, task)
    if result is None:
        print()
        result = structured_mock(task)

    print("-" * 72)
    print(f"OUT:structured:final: {result.name} | 难度={result.difficulty} | 优先级={result.priority} | 工时={result.estimated_hours}h")
    print()
    print("  💡 校验失败重试的核心：把 ValidationError 反馈给 LLM，让它『看到』自己错在哪。")
    print("     这比单纯报错丢给用户强 100 倍——LLM 通常一次就能修正。")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 3: 工具参数校验（ToolValidation）
# ═══════════════════════════════════════════════════════════════════════

def validate_tool_args(
    args: dict[str, Any], schema: dict[str, Any]
) -> list[str]:
    """用 JSON Schema 校验工具参数。返回错误列表（空列表 = 通过）。

    简化实现（不依赖 jsonschema 库）：只校验 type + required。
    生产建议用 `jsonschema.validate` 或 Pydantic 做完整校验。
    """
    errors: list[str] = []

    # 1. 检查 required 字段是否都存在
    required = schema.get("required", [])
    for field in required:
        if field not in args:
            errors.append(f"缺少必填字段: '{field}'")

    # 2. 检查每个提供的字段类型是否匹配
    properties = schema.get("properties", {})
    json_type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for field, value in args.items():
        if field not in properties:
            # 允许额外字段（生产中可用 additionalProperties: false 禁止）
            continue
        expected_type = properties[field].get("type")
        if expected_type and expected_type in json_type_map:
            # 特殊处理：bool 是 int 的子类，必须先排除
            if expected_type in ("integer", "number") and isinstance(value, bool):
                errors.append(f"字段 '{field}' 期望 {expected_type}，实际 boolean")
                continue
            py_type = json_type_map[expected_type]
            if not isinstance(value, py_type):
                errors.append(
                    f"字段 '{field}' 期望 {expected_type}，实际 {type(value).__name__}"
                )

    return errors


def safe_execute_tool(
    name: str,
    args: dict[str, Any],
    schema: dict[str, Any],
    handler: Any,
) -> str:
    """带参数校验的工具执行：先 validate → 通过才执行。

    对应第13章 InMemoryToolRegistry.execute 的"生产增强版"：
    第13章版直接 handler(**args)，缺少类型保护；
    本章版先用 schema 校验，类型不匹配就报错反馈给 Agent。
    """
    errors = validate_tool_args(args, schema)
    if errors:
        msg = "; ".join(errors)
        print(f"OUT:validate:fail: 工具 '{name}' 参数校验失败 → {msg}")
        # 关键：返回错误消息（不抛异常），让 Agent 能"看到"错误并自我纠正
        return f"[参数校验失败] {name}: {msg}"

    print(f"OUT:validate:pass: 工具 '{name}' 参数校验通过 → {args}")
    try:
        return str(handler(**args))
    except Exception as e:
        return f"[工具执行失败] {name}: {type(e).__name__}: {e}"


# 工具函数 + schema（与第13章一致）

_WEATHER_DB: dict[str, dict[str, str]] = {
    "北京": {"condition": "晴", "temp": "25°C"},
    "上海": {"condition": "多云", "temp": "28°C"},
}


def get_weather(city: str) -> str:
    """查询天气（演示用模拟数据）。"""
    city = city.strip()
    if city not in _WEATHER_DB:
        return f"[未找到] 城市 '{city}'"
    w = _WEATHER_DB[city]
    return f"{city}今天{w['condition']}，气温 {w['temp']}"


def calculate(expression: str) -> str:
    """安全计算（只允许数字和运算符，用 ast 解析，杜绝 eval 注入）。"""
    import ast
    import operator
    safe_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
    }
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in safe_ops:
            return safe_ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"不支持的节点: {type(node).__name__}")

    result = _eval(tree.body)
    return f"{expression} = {int(result) if isinstance(result, float) and result.is_integer() else result}"


WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "城市名"},
    },
    "required": ["city"],
}

CALCULATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "数学表达式"},
    },
    "required": ["expression"],
}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_weather": {"schema": WEATHER_SCHEMA, "handler": get_weather},
    "calculate": {"schema": CALCULATE_SCHEMA, "handler": calculate},
}


def demo_tool_validation() -> None:
    """Demo 3: 工具参数校验（100% 离线，纯逻辑）。"""
    print("=" * 72)
    print("Demo 3: 工具参数校验（ToolValidation）")
    print("  执行工具前用 JSON Schema 校验 args（类型 + 必填字段）。")
    print("  价值：在工具执行前拦截非法参数，避免崩溃或语义错误。")
    print("=" * 72)
    print()

    # 测试用例：合法 + 非法对比
    test_cases: list[dict[str, Any]] = [
        # ✓ 合法：get_weather + 正确类型
        {"name": "get_weather", "args": {"city": "北京"}, "expect": "pass"},
        # ✗ 非法：缺必填字段
        {"name": "get_weather", "args": {}, "expect": "fail"},
        # ✗ 非法：类型错误（city 应为 string，传了 int）
        {"name": "get_weather", "args": {"city": 12345}, "expect": "fail"},
        # ✓ 合法：calculate + 正确类型
        {"name": "calculate", "args": {"expression": "28-25"}, "expect": "pass"},
        # ✗ 非法：类型错误（expression 应为 string，传了 int）
        {"name": "calculate", "args": {"expression": 28}, "expect": "fail"},
        # ✗ 非法：未知工具（schema 查不到）
        {"name": "get_stock_price", "args": {"symbol": "AAPL"}, "expect": "fail"},
    ]

    print(f"  共 {len(test_cases)} 个测试用例（合法/非法各半）")
    print("-" * 72)

    pass_count = 0
    fail_count = 0
    for tc in test_cases:
        name = tc["name"]
        args = tc["args"]
        tool = TOOL_SCHEMAS.get(name)

        if tool is None:
            print(f"OUT:validate:fail: 工具 '{name}' 未注册（未知工具名）")
            fail_count += 1
            continue

        result = safe_execute_tool(name, args, tool["schema"], tool["handler"])
        if result.startswith("[参数校验失败]") or result.startswith("[工具执行失败]"):
            fail_count += 1
        else:
            pass_count += 1
        # 打印执行结果（截断过长）
        preview = result[:70] + "..." if len(result) > 70 else result
        print(f"  → 结果: {preview}")
        print()

    print("-" * 72)
    print(f"OUT:validate:summary: 通过 {pass_count} 个，失败 {fail_count} 个")
    print()
    print("  💡 在第13章框架中，把这个 validate 步骤插到 ToolRegistry.execute 开头，")
    print("     就能把『模型生成错参数』的 bug 在执行前拦截，而不是等到工具内部崩溃。")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 4: 现代框架对比（OpenAI Agents SDK / Pydantic AI）
# ═══════════════════════════════════════════════════════════════════════

def demo_modern_frameworks() -> None:
    """Demo 4: 现代框架对比（try-import，未安装则打印注释代码片段）。"""
    print("=" * 72)
    print("Demo 4: 现代框架对比（OpenAI Agents SDK / Pydantic AI）")
    print("  展示『现代框架用更少代码完成同样的流式/结构化/校验』。")
    print("  未安装的包会优雅降级为注释代码片段（不强制安装）。")
    print("=" * 72)
    print()

    # ── 1. OpenAI Agents SDK（Python） ──
    print("▎ 对比 1: OpenAI Agents SDK（openai-agents）")
    print("-" * 72)
    try:
        # 尝试 import；未安装则降级为注释代码
        import agents  # type: ignore[import-not-found]
        print(f"OUT:compare:agents_sdk: ✓ 已安装（版本: {getattr(agents, '__version__', 'unknown')}）")
        print("  等价代码（流式 + 结构化）:")
        print("""
        from agents import Agent, Runner
        from pydantic import BaseModel

        class TaskSummary(BaseModel):
            name: str
            difficulty: str

        agent = Agent(
            name="TaskAnalyzer",
            instructions="分析任务并输出结构化 JSON",
            output_type=TaskSummary,  # ← 一行搞定结构化输出（本章 Demo 2 的全部逻辑）
        )

        result = Runner.run_sync(agent, "实现登录功能")
        # result.final_output 已经是 TaskSummary 实例（自动校验 + 重试）
        """)
    except ImportError:
        print("OUT:compare:agents_sdk: ✗ 未安装 openai-agents（这是正常的，本教程不强制安装）")
        print("  等价代码（如果你安装了 openai-agents，可以这样写）:")
        print("""
        # pip install openai-agents
        from agents import Agent, Runner
        from pydantic import BaseModel

        class TaskSummary(BaseModel):
            name: str
            difficulty: str

        agent = Agent(
            name="TaskAnalyzer",
            instructions="分析任务并输出结构化 JSON",
            output_type=TaskSummary,  # ← 一行搞定结构化输出（本章 Demo 2 的全部逻辑）
        )

        result = Runner.run_sync(agent, "实现登录功能")
        # result.final_output 已经是 TaskSummary 实例（自动校验 + 重试）
        """)
    print()

    # ── 2. Pydantic AI ──
    print("▎ 对比 2: Pydantic AI")
    print("-" * 72)
    try:
        import pydantic_ai  # type: ignore[import-not-found]
        print(f"OUT:compare:pydantic_ai: ✓ 已安装（版本: {getattr(pydantic_ai, '__version__', 'unknown')}）")
        print("  等价代码（工具校验 + 结构化输出）:")
        print("""
        from pydantic import BaseModel
        from pydantic_ai import Agent

        class TaskSummary(BaseModel):
            name: str
            difficulty: str

        agent = Agent(
            "openai:gpt-4o-mini",
            output_type=TaskSummary,
            system_prompt="分析任务并输出结构化 JSON",
        )

        @agent.tool  # ← 工具自动从函数签名 + type hints 生成 schema（本章 Demo 3）
        def get_weather(ctx, city: str) -> str:
            \"\"\"查询城市天气。\"\"\"
            return f"{city}今天晴 25°C"

        result = agent.run_sync("查北京天气，然后分析任务难度")
        # result.output 是 TaskSummary；agent 自动处理工具调用 + 校验
        """)
    except ImportError:
        print("OUT:compare:pydantic_ai: ✗ 未安装 pydantic-ai（这是正常的，本教程不强制安装）")
        print("  等价代码（如果你安装了 pydantic-ai，可以这样写）:")
        print("""
        # pip install pydantic-ai
        from pydantic import BaseModel
        from pydantic_ai import Agent

        class TaskSummary(BaseModel):
            name: str
            difficulty: str

        agent = Agent(
            "openai:gpt-4o-mini",
            output_type=TaskSummary,
            system_prompt="分析任务并输出结构化 JSON",
        )

        @agent.tool  # ← 工具自动从函数签名 + type hints 生成 schema（本章 Demo 3）
        def get_weather(ctx, city: str) -> str:
            \"\"\"查询城市天气。\"\"\"
            return f"{city}今天晴 25°C"

        result = agent.run_sync("查北京天气，然后分析任务难度")
        # result.output 是 TaskSummary；agent 自动处理工具调用 + 校验
        """)
    print()

    # ── 3. 何时该用现代框架而非自造 ──
    print("▎ 决策：何时用现代框架，何时自造？")
    print("-" * 72)
    print("OUT:compare:decision:")
    print("""
    ┌─────────────────────────┬─────────────────────────────────────────┐
    │ ✅ 用现代框架            │ ✅ 自造（如本教程第12-14章）             │
    ├─────────────────────────┼─────────────────────────────────────────┤
    │ • 生产项目               │ • 学习原理（看透框架黑盒）               │
    │ • 需要流式/结构化/校验   │ • 极简场景（< 3 个工具，单步任务）       │
    │ • 需要 tracing/eval     │ • 定制需求（现代框架都不满足）           │
    │ • 团队协作（社区支持）   │ • 教学/演示（不想引入重依赖）           │
    │ • 长期维护               │ • 嵌入式/资源受限环境                   │
    └─────────────────────────┴─────────────────────────────────────────┘

    核心原则：先原理后工具。学完本教程，你打开任何框架源码都能 1 小时看懂。
    """)
    print()


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("第14章 从零造框架 — 高级特性")
    print("  流式输出 / 结构化输出强制 / 工具参数校验 / 现代框架对比")
    print("  （概念上引用第13章的 6 大组件，独立可运行）")
    print("=" * 72)
    print()

    # Demo 1: 流式输出
    demo_streaming()

    # Demo 2: 结构化输出
    demo_structured_output()

    # Demo 3: 工具校验
    demo_tool_validation()

    # Demo 4: 现代框架对比
    demo_modern_frameworks()

    print("=" * 72)
    print("✓ 本章完成：4 个高级特性全部演示完毕。")
    print("  核心收获：理解原理后，用现代框架时你能『看穿』它的每一行代码。")
    print("=" * 72)


if __name__ == "__main__":
    main()
