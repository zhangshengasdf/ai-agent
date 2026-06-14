"""
第15章 评估与测试（行为测试、LLM-as-Judge、回归测试）

本文件演示 Agent 评估的三大手段：

  Demo 1: 行为测试（Behavior Testing）
          - 定义测试用例（输入 → 期望调用的工具）
          - 跑 Agent，断言它选择了正确的工具
          - 离线 mock Agent 用关键词匹配模拟工具选择决策
          - 输出标记：OUT:test:{name}:

  Demo 2: LLM-as-Judge（用模型评估输出质量）
          - 第一个 LLM 生成回答，第二个 LLM（Judge）按 rubric 打分
          - 评分标准：1-5 分 + 文字评语
          - 离线 mock：预设候选回答 + 预设 Judge 评分
          - 输出标记：OUT:judge:

  Demo 3: 回归测试套件（Regression Suite）
          - 批量运行多个行为测试，汇总 PASS/FAIL 报告
          - 防止改 prompt / 换模型 / 升 SDK 后行为退化
          - 输出标记：OUT:regression:

运行方式：
  cd ai-agent/15-evaluation
  python3 python/main.py

设计原则：
  - 评估对象是前面章节造的 Agent（概念上引用第04章 agent loop / 第13章框架）
  - 离线 mock Agent 用关键词规则模拟 LLM 的工具选择决策（可靠、可复现）
  - .env 用占位符 sk-REPLACE-ME → 真实 API 必失败 → 降级 mock，保证 exit 0
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# ═══════════════════════════════════════════════════════════════════════
# 数据类定义
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """Agent 运行结果（含行为轨迹：调了哪些工具 + 最终输出）。

    行为测试的核心：不只看 final_output，更要看 tools_called。
    """

    final_output: str
    tools_called: list[str] = field(default_factory=list)


@dataclass
class BehaviorTestCase:
    """行为测试用例：输入任务 → 期望调用的工具列表。

    Attributes:
        name: 测试名（如 "weather_query"）
        task: 输入任务（如 "查北京天气"）
        expected_tools: 期望调用的工具（空列表 = 不该调任何工具）
        description: 这个测试在验证什么
    """

    name: str
    task: str
    expected_tools: list[str]
    description: str


@dataclass
class TestResult:
    """单个测试的运行结果。"""

    name: str
    passed: bool
    detail: str


@dataclass
class JudgeResult:
    """LLM-as-Judge 的评分结果。"""

    score: int  # 1-5
    comment: str


@dataclass
class RegressionReport:
    """回归测试套件的汇总报告。"""

    results: list[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0

    def add(self, result: TestResult) -> None:
        self.results.append(result)
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1


# ═══════════════════════════════════════════════════════════════════════
# 工具定义（OpenAI tools 格式，与第03/13章一致）
# ═══════════════════════════════════════════════════════════════════════

WEATHER_DB: dict[str, str] = {
    "北京": "晴，气温 25°C",
    "上海": "多云，气温 28°C",
    "广州": "小雨，气温 30°C",
}

TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════
# 离线 Mock Agent：用关键词匹配模拟真实 Agent 的工具选择决策
# ═══════════════════════════════════════════════════════════════════════

class MockAgent:
    """离线 mock Agent：模拟第04章/第13章的 Agent 循环。

    真实 Agent 用 LLM 决策"调哪个工具"；这里用关键词规则模拟，
    保证离线环境下也能完整演示行为测试。

    决策规则（模拟 LLM 的工具选择逻辑）：
      - 任务含"天气/气温/weather" → 调 get_weather
      - 任务含"温差/计算/算" → 调 calculate
      - 识别任务中的城市名 → 为每个城市调一次 get_weather
      - 纯闲聊（你好/谢谢）→ 不调工具，直接回答
    """

    WEATHER_KEYWORDS = ["天气", "气温", "weather", "温度"]
    CALC_KEYWORDS = ["温差", "计算", "算一下", "算", "calculate"]
    CITIES = ["北京", "上海", "广州", "深圳", "杭州"]

    def run(self, task: str, max_steps: int = 10) -> AgentResult:
        """模拟 Agent 运行，返回行为轨迹（调了哪些工具 + 最终输出）。"""
        tools_called: list[str] = []

        # ── 模拟 Agent 的"思考"：该调哪些工具？──
        needs_weather = any(kw in task for kw in self.WEATHER_KEYWORDS)
        needs_calc = any(kw in task for kw in self.CALC_KEYWORDS)
        cities_found = [c for c in self.CITIES if c in task]

        # ── 模拟多步 Agent 循环（observe→reason→act）──
        steps = 0

        # step: 为每个城市查天气
        if needs_weather:
            for city in cities_found:
                if steps >= max_steps:
                    break
                tools_called.append("get_weather")
                steps += 1

        # step: 计算
        if needs_calc and steps < max_steps:
            tools_called.append("calculate")
            steps += 1

        # ── 生成最终输出 ──
        if not tools_called:
            # 闲聊，无需工具
            final_output = f"你好！我是任务助手。关于『{task}』，有什么我可以帮你的吗？"
        else:
            parts: list[str] = []
            if needs_weather:
                for city in cities_found:
                    weather = WEATHER_DB.get(city, "未知")
                    parts.append(f"{city}今天{weather}")
            if needs_calc and len(cities_found) >= 2:
                parts.append("两地温差为 3°C")
            elif needs_calc:
                parts.append("计算完成")
            final_output = "。".join(parts) + "。" if parts else "（已处理）"

        return AgentResult(final_output=final_output, tools_called=tools_called)


# ═══════════════════════════════════════════════════════════════════════
# 真实 API Agent：尝试调 LLM 获取真实工具决策，失败返回 None
# ═══════════════════════════════════════════════════════════════════════

def run_agent_real_api(
    client: OpenAI, model: str, task: str
) -> AgentResult | None:
    """尝试用真实 LLM 运行 Agent（单步决策）。失败返回 None（降级 mock）。

    真实 Agent 循环是多步的（第04/13章）；这里简化为单步——
    让 LLM 决定第一步调什么工具，用于行为测试演示。
    """
    system_prompt = (
        "你是任务助手 Agent。根据用户任务决定是否调用工具。"
        "可用工具：get_weather（查天气）、calculate（计算）。"
        "如果需要查天气或计算，请调用相应工具；如果是闲聊，直接回答。"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            tools=TOOL_DEFS,
            tool_choice="auto",
        )
    except (AuthenticationError, APIConnectionError, APIError) as e:
        print(f"OUT:agent:offline: 真实 API 不可用（{type(e).__name__}），降级 mock Agent")
        return None
    except Exception as e:
        print(f"OUT:agent:offline: 真实 API 异常（{type(e).__name__}），降级 mock Agent")
        return None

    msg = resp.choices[0].message
    tools_called: list[str] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.type == "function":
                tools_called.append(tc.function.name)

    return AgentResult(
        final_output=msg.content or "",
        tools_called=tools_called,
    )


def run_agent(task: str) -> AgentResult:
    """运行 Agent：优先真实 API，失败降级 mock。

    这是行为测试的"被评估对象"。
    """
    cfg = get_config()
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    result = run_agent_real_api(client, cfg.model, task)
    if result is not None:
        return result
    return MockAgent().run(task)


# ═══════════════════════════════════════════════════════════════════════
# Demo 1: 行为测试（Behavior Testing）
# ═══════════════════════════════════════════════════════════════════════

def run_behavior_test(test_case: BehaviorTestCase) -> TestResult:
    """跑单个行为测试：运行 Agent → 断言期望工具被调用。

    断言逻辑：
      - expected_tools 非空 → 所有期望工具都必须在实际调用列表中
      - expected_tools 为空 → Agent 不该调任何工具（闲聊场景）
    """
    print(f"OUT:test:{test_case.name}: ▶ 运行测试: {test_case.name}")
    print(f"  任务: {test_case.task}")
    print(f"  期望工具: {test_case.expected_tools}")
    print(f"  验证点: {test_case.description}")

    result = run_agent(test_case.task)
    actual = result.tools_called

    # ── 行为断言 ──
    if test_case.expected_tools:
        # 期望调用了某些工具 → 检查是否都调了
        passed = all(tool in actual for tool in test_case.expected_tools)
        missing = [t for t in test_case.expected_tools if t not in actual]
    else:
        # 期望不调工具（闲聊）→ 检查是否真的没调
        passed = len(actual) == 0

    if passed:
        detail = f"实际调用 {actual}"
        print(f"OUT:test:{test_case.name}: ✓ 通过 — {detail}")
    else:
        if test_case.expected_tools:
            detail = f"期望 {test_case.expected_tools}，实际 {actual}（缺少 {missing}）"
        else:
            detail = f"期望不调工具，但实际调了 {actual}"
        print(f"OUT:test:{test_case.name}: ✗ 失败 — {detail}")

    print(f"  输出: {result.final_output[:60]}")
    print()
    return TestResult(name=test_case.name, passed=passed, detail=detail)


def demo_behavior_testing() -> None:
    """Demo 1: 行为测试。"""
    print("=" * 72)
    print("Demo 1: 行为测试（Behavior Testing）")
    print("  定义『输入 → 期望工具』，断言 Agent 走对了流程。")
    print("  价值：抓住『输出碰巧对但行为错』的隐蔽 bug。")
    print("=" * 72)
    print()

    # ── 测试用例集（黄金用例 + 边界用例）──
    test_cases: list[BehaviorTestCase] = [
        BehaviorTestCase(
            name="weather_query",
            task="查一下北京今天的天气",
            expected_tools=["get_weather"],
            description="天气查询任务应调用 get_weather 工具",
        ),
        BehaviorTestCase(
            name="weather_temp_calc",
            task="查北京和上海的天气，然后算一下两地温差",
            expected_tools=["get_weather", "calculate"],
            description="温差任务应同时调用天气和计算工具",
        ),
        BehaviorTestCase(
            name="no_tool_needed",
            task="你好，谢谢你",
            expected_tools=[],
            description="纯闲聊不应调用任何工具",
        ),
    ]

    results: list[TestResult] = []
    for tc in test_cases:
        results.append(run_behavior_test(tc))

    # ── 汇总 ──
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print("-" * 72)
    print(f"OUT:test:summary: {passed}/{len(results)} 通过，{failed} 失败")
    print()
    print("  💡 行为测试不只看最终输出，更看 Agent『走了什么路』。")
    print("     如果只测输出，Agent 碰巧猜对答案但你不知道它根本没调工具。")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 2: LLM-as-Judge（用模型评估输出质量）
# ═══════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """你是一个严格的评分员。请对 AI 助手的回答打分（1-5 分）。

评分维度：
- 正确性：事实是否准确
- 完整性：是否覆盖了任务要求的所有方面
- 清晰度：表述是否清楚易懂

评分标准：
- 5 分：完全正确、完整、清晰
- 4 分：基本正确，有小瑕疵
- 3 分：部分正确，有明显遗漏
- 2 分：大部分错误
- 1 分：完全错误或无关

只输出 JSON，格式：{"score": 1-5的整数, "comment": "评语"}"""


def generate_answer_real_api(
    client: OpenAI, model: str, task: str
) -> str | None:
    """尝试用真实 LLM 生成回答。失败返回 None。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是知识渊博的助手。简洁准确地回答问题。"},
                {"role": "user", "content": task},
            ],
        )
        return resp.choices[0].message.content
    except (AuthenticationError, APIConnectionError, APIError) as e:
        print(f"OUT:judge:offline: 候选生成 API 不可用（{type(e).__name__}），降级 mock")
        return None
    except Exception as e:
        print(f"OUT:judge:offline: 候选生成 API 异常（{type(e).__name__}），降级 mock")
        return None


def judge_real_api(
    client: OpenAI, model: str, task: str, candidate: str
) -> JudgeResult | None:
    """尝试用真实 LLM 做 Judge。失败返回 None。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"任务: {task}\n\n回答: {candidate}"},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        return JudgeResult(
            score=int(data.get("score", 0)),
            comment=str(data.get("comment", "")),
        )
    except (AuthenticationError, APIConnectionError, APIError) as e:
        print(f"OUT:judge:offline: Judge API 不可用（{type(e).__name__}），降级 mock")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        print(f"OUT:judge:offline: Judge 返回解析失败（{type(e).__name__}），降级 mock")
        return None
    except Exception as e:
        print(f"OUT:judge:offline: Judge API 异常（{type(e).__name__}），降级 mock")
        return None


def judge_mock(task: str, candidate: str) -> JudgeResult:
    """离线 mock Judge：预设评分（模拟真实 Judge 的判断）。

    根据候选回答的特征模拟评分（长度、关键词覆盖），
    让无 API 环境也能完整演示 LLM-as-Judge 流程。
    """
    # 模拟评分逻辑（真实场景由 LLM 判断，这里用规则近似）
    score = 3  # 默认中等
    comments: list[str] = []

    # 维度 1：长度（太短通常不完整）
    if len(candidate) < 30:
        score = 2
        comments.append("回答过于简短，完整性不足")
    elif len(candidate) > 100:
        score = min(score + 1, 5)
        comments.append("回答详尽，覆盖面广")

    # 维度 2：关键词覆盖（任务关键词是否在回答中出现）
    task_keywords = [w for w in ["递归", "函数", "自身", "基线", "终止"] if w in task]
    covered = [kw for kw in task_keywords if kw in candidate]
    if task_keywords and len(covered) >= len(task_keywords) * 0.6:
        score = min(score + 1, 5)
        comments.append(f"覆盖了关键概念（{len(covered)}/{len(task_keywords)}）")
    elif task_keywords:
        comments.append(f"关键概念覆盖不足（{len(covered)}/{len(task_keywords)}）")

    # 维度 3：是否有举例
    if "例如" in candidate or "比如" in candidate or "举例" in candidate:
        score = min(score + 1, 5)
        comments.append("有具体举例，清晰度高")

    # 确保分数在 1-5 范围
    score = max(1, min(5, score))
    if not comments:
        comments.append("回答基本合格，但缺少亮点")

    return JudgeResult(score=score, comment="；".join(comments))


def demo_llm_judge() -> None:
    """Demo 2: LLM-as-Judge。"""
    print("=" * 72)
    print("Demo 2: LLM-as-Judge（用模型评估输出质量）")
    print("  第一个 LLM 生成回答，第二个 LLM（Judge）按 rubric 打分。")
    print("  价值：评估『输出好不好』这种难以 assert 的质量维度。")
    print("=" * 72)
    print()

    cfg = get_config()
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    # ── 待评估的任务 + 候选回答 ──
    task = "请解释什么是递归，并给出一个例子"
    print(f"OUT:judge:task: {task}")

    # 优先真实 API 生成候选回答，失败用预设回答
    candidate = generate_answer_real_api(client, cfg.model, task)
    if candidate is None:
        candidate = (
            "递归是一种编程技巧，指函数在执行过程中调用自身。"
            "例如，计算阶乘时，factorial(n) = n * factorial(n-1)，"
            "直到 n=1 时返回 1（基线条件）。递归可以把复杂问题分解为更小的同类问题。"
        )
    print(f"OUT:judge:candidate: {candidate[:80]}...")
    print()

    # ── Judge 评分 ──
    print("  评分标准: 正确性 + 完整性 + 清晰度（1-5 分）")
    print("-" * 72)

    judge = judge_real_api(client, cfg.model, task, candidate)
    if judge is None:
        judge = judge_mock(task, candidate)

    print(f"OUT:judge:score: {judge.score}/5")
    print(f"OUT:judge:comment: {judge.comment}")
    print()
    print(f"  💡 LLM-as-Judge 适合评估『回答好不好』这种难以 assert 的维度。")
    print(f"     但要注意偏见：Judge 偏向冗长回答、偏向和自己同款的回答。")
    print(f"     生产中要结合行为测试 + 人工抽检，不能只依赖 Judge。")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Demo 3: 回归测试套件（Regression Suite）
# ═══════════════════════════════════════════════════════════════════════

class RegressionSuite:
    """回归测试套件：批量运行行为测试，汇总 PASS/FAIL 报告。

    使用场景：
      - 改了 system prompt → 跑套件，确认行为没退化
      - 换了模型 → 跑套件，确认工具调用逻辑没变
      - 升级了 SDK → 跑套件，确认解析逻辑还能用
    """

    def __init__(self) -> None:
        self._cases: list[BehaviorTestCase] = []

    def add(self, test_case: BehaviorTestCase) -> None:
        self._cases.append(test_case)

    def run_all(self) -> RegressionReport:
        """运行所有测试用例，返回汇总报告。"""
        report = RegressionReport()
        for i, tc in enumerate(self._cases, 1):
            print(f"OUT:regression:case:{i} {tc.name} ...", end=" ", flush=True)
            result = run_behavior_test_silent(tc)
            status = "PASS" if result.passed else "FAIL"
            print(f"{status}")
            report.add(result)
        return report


def run_behavior_test_silent(test_case: BehaviorTestCase) -> TestResult:
    """静默版行为测试（不打印详情，用于回归套件批量运行）。"""
    result = run_agent(test_case.task)
    actual = result.tools_called

    if test_case.expected_tools:
        passed = all(tool in actual for tool in test_case.expected_tools)
        missing = [t for t in test_case.expected_tools if t not in actual]
        detail = f"期望 {test_case.expected_tools}，实际 {actual}" + (
            f"（缺少 {missing}）" if missing else ""
        )
    else:
        passed = len(actual) == 0
        detail = f"期望不调工具，实际 {actual}"

    return TestResult(name=test_case.name, passed=passed, detail=detail)


def demo_regression_suite() -> None:
    """Demo 3: 回归测试套件。"""
    print("=" * 72)
    print("Demo 3: 回归测试套件（Regression Suite）")
    print("  批量运行行为测试，汇总 PASS/FAIL 报告。")
    print("  价值：改 prompt / 换模型后，5 秒知道有没有破坏行为。")
    print("=" * 72)
    print()

    # ── 构建套件（黄金用例 + 边界用例）──
    suite = RegressionSuite()
    suite.add(BehaviorTestCase(
        name="weather_query_single",
        task="查北京天气",
        expected_tools=["get_weather"],
        description="单城市天气查询",
    ))
    suite.add(BehaviorTestCase(
        name="weather_query_multi",
        task="查北京和上海的天气",
        expected_tools=["get_weather"],
        description="多城市天气查询",
    ))
    suite.add(BehaviorTestCase(
        name="weather_and_calc",
        task="查北京和上海天气，算温差",
        expected_tools=["get_weather", "calculate"],
        description="天气 + 计算组合任务",
    ))
    suite.add(BehaviorTestCase(
        name="chitchat_no_tool",
        task="你好呀",
        expected_tools=[],
        description="闲聊不应调工具",
    ))
    suite.add(BehaviorTestCase(
        name="calc_only",
        task="帮我算一下 28 减 25",
        expected_tools=["calculate"],
        description="纯计算任务",
    ))

    # ── 运行套件 ──
    print(f"OUT:regression:running: 共 {len(suite._cases)} 个测试用例")
    print("-" * 72)

    report = suite.run_all()

    # ── 汇总报告 ──
    print("-" * 72)
    total = report.passed + report.failed
    print(f"OUT:regression:summary: {report.passed}/{total} 通过，{report.failed} 失败", end="")
    if report.failed == 0:
        print(" ✓ 全部通过")
    else:
        print(" ✗ 有失败")
        print()
        print("失败详情:")
        for r in report.results:
            if not r.passed:
                print(f"  ✗ {r.name}: {r.detail}")

    print()
    print("  💡 把这套测试纳入 CI：每次改 prompt / 换模型 / 升 SDK，自动跑一遍。")
    print("     没有回归测试的 Agent 项目，每次改动都在玩俄罗斯轮盘赌。")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("第15章 评估与测试")
    print("  行为测试 / LLM-as-Judge / 回归测试套件")
    print("  （评估对象是前面章节造的 Agent，用 mock 模拟保证离线可跑）")
    print("=" * 72)
    print()

    # Demo 1: 行为测试
    demo_behavior_testing()

    # Demo 2: LLM-as-Judge
    demo_llm_judge()

    # Demo 3: 回归测试套件
    demo_regression_suite()

    print("=" * 72)
    print("✓ 本章完成：三大评估手段全部演示完毕。")
    print("  核心收获：Agent 评估 ≠ 模型评估 —— 要测行为序列，不只测单次输出。")
    print("=" * 72)


if __name__ == "__main__":
    main()
