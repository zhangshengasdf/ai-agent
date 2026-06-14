"""第08章 规划模式（Plan-and-Execute、CoT、Reflection）

本章演示两种高级推理模式，让 Agent 能处理复杂的多阶段任务：

  Plan-and-Execute（规划-执行-汇总）：
    - Phase 1 (Plan)：用结构化输出（json_object + Pydantic Plan）分解任务
    - Phase 2 (Execute)：逐步执行每个步骤（调工具/子查询），累积结果
    - Phase 3 (Synthesize)：汇总所有步骤结果，生成最终输出
    - 适合：步骤明确的复杂任务（写报告、制定计划、调研）

  Reflection（反思 / 自我批评）：
    - Round 1：Agent 生成初版答案
    - Round 2：同一 Agent 审视初版，指出不足（完整性/准确性/结构）
    - Round 3：Agent 根据反思改进答案
    - 适合：质量敏感的生成任务（写作、代码、重要决策）

离线 mock 设计：
  .env 的 OPENAI_API_KEY=sk-REPLACE-ME 是占位符，真实 API 调用必失败。
  所有功能先 try 真实 API（失败时降级），然后跑离线 mock，保证 exit 0。
  Plan-and-Execute mock：预设步骤列表 + mock 执行结果 + 汇总。
  Reflection mock：预设"初版→反思→改进版"文本轨迹。
"""

import sys
from pathlib import Path

from pydantic import BaseModel

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI  # noqa: E402

from shared.config import get_config  # noqa: E402

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


# ════════════════════════════════════════════════════════════════════
# Pydantic 模型：规划阶段的结构化输出
# ════════════════════════════════════════════════════════════════════


class Plan(BaseModel):
    """任务分解计划。steps 是有序的可执行步骤列表。"""

    steps: list[str]


# ════════════════════════════════════════════════════════════════════
# mock 工具：执行阶段的子查询（与第03/07章风格一致）
# ════════════════════════════════════════════════════════════════════


def mock_search(query: str) -> str:
    """模拟知识检索（mock 知识库）。"""
    knowledge = {
        "定义": "AI Agent 是能感知环境、自主决策、采取行动以实现目标的智能系统。",
        "应用": "AI Agent 应用于智能客服、编程助手、自动化研究、数据分析等场景。",
        "框架": "主流 AI Agent 框架有 LangChain、AutoGPT、OpenAI Agents SDK、CrewAI 等。",
        "趋势": "AI Agent 正向多 Agent 协作、长程任务自主执行、工具自学习方向发展。",
        "挑战": "AI Agent 面临可靠性、成本控制、安全对齐、评估困难等挑战。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower or key in query:
            return value
    return f"检索到与'{query}'相关的通用信息。"


# ════════════════════════════════════════════════════════════════════
# Plan-and-Execute 模式
# ════════════════════════════════════════════════════════════════════

PLAN_SYSTEM_PROMPT = """\
你是一个任务规划助手。用户会给你一个复杂任务，你需要：

1. 先理解任务的本质
2. 思考完成任务需要哪些信息或操作
3. 把任务分解成 3-6 个有序的、具体的、可执行的步骤

输出 JSON 格式：{"steps": ["步骤1", "步骤2", ...]}

要求：
- 每步要具体、可执行（能明确说出"做什么"）
- 步骤之间有序（前面的输出是后面的输入）
- 最后一步通常是"综合/总结/撰写"

示例：
任务：写一篇 AI Agent 调研报告
输出：{"steps": [
  "检索 AI Agent 的定义与核心特征",
  "检索 AI Agent 的典型应用场景",
  "检索主流 AI Agent 开发框架",
  "检索 AI Agent 的发展趋势与挑战",
  "综合以上信息撰写调研报告"
]}
"""


def plan_task(task: str) -> Plan:
    """Phase 1: 用结构化输出生成任务计划。

    用 response_format=json_object 强制 JSON，再用 Pydantic Plan 解析校验。
    """
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return Plan.model_validate_json(raw)


def execute_step(step: str, step_index: int) -> str:
    """Phase 2: 执行单个步骤。

    对教学示例：用 mock_search 模拟检索/工具调用。
    真实场景：这里可以调用任意工具、子 LLM、或外部 API。
    """
    # 教学用：把步骤描述当作检索 query
    return mock_search(step)


def synthesize(task: str, steps_and_results: list[tuple[str, str]]) -> str:
    """Phase 3: 汇总所有步骤结果，生成最终输出。"""
    context = "\n".join(
        f"步骤{i}: {step}\n结果: {result}"
        for i, (step, result) in enumerate(steps_and_results, 1)
    )
    prompt = (
        f"用户任务：{task}\n\n"
        f"已完成以下步骤：\n{context}\n\n"
        f"请基于以上信息，完成用户的原始任务。输出一份简洁的综合报告。"
    )
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "(空)"


def plan_and_execute(task: str) -> str:
    """Plan-and-Execute 完整三阶段流程（真实 API）。"""
    print(f"\n{'='*60}")
    print(f"Plan-and-Execute 任务: {task}")
    print(f"{'='*60}")

    # Phase 1: Plan
    print("\n--- Phase 1: Plan（规划）---")
    plan = plan_task(task)
    print(f"OUT:plan: 分解出 {len(plan.steps)} 个步骤:")
    for i, step in enumerate(plan.steps, 1):
        print(f"OUT:plan:step{i}: {step}")

    # Phase 2: Execute
    print("\n--- Phase 2: Execute（执行）---")
    steps_and_results: list[tuple[str, str]] = []
    for i, step in enumerate(plan.steps, 1):
        print(f"OUT:execute:step{i}: 执行: {step}")
        result = execute_step(step, i)
        steps_and_results.append((step, result))
        print(f"OUT:execute:step{i}: 结果: {result[:80]}")

    # Phase 3: Synthesize
    print("\n--- Phase 3: Synthesize（汇总）---")
    final = synthesize(task, steps_and_results)
    print("OUT:synthesize: 最终输出（前 200 字）:")
    print(f"OUT:synthesize: {final[:200]}")
    return final


# ════════════════════════════════════════════════════════════════════
# Reflection 模式（自我批评 + 改进）
# ════════════════════════════════════════════════════════════════════

REFLECTION_PROMPT = """\
你是一个严格的审稿人。请审视以下初版答案，指出它的不足。

用户问题：{question}
初版答案：{draft}

请从以下维度批评：
1. 完整性：有没有遗漏重要信息？
2. 准确性：有没有事实错误或逻辑漏洞？
3. 结构：组织是否清晰？

只指出不足（2-3 点），不要给出完整改进版。用简洁的要点格式。
"""

REVISE_PROMPT = """\
用户问题：{question}
初版答案：{draft}
审稿意见：{critique}

请根据审稿意见改进初版答案，输出最终版。保持简洁。
"""


def generate_draft(question: str) -> str:
    """Round 1: 生成初版答案。"""
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content or "(空)"


def reflect(question: str, draft: str) -> str:
    """Round 2: 反思 / 自我批评。"""
    prompt = REFLECTION_PROMPT.format(question=question, draft=draft)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "(空)"


def revise(question: str, draft: str, critique: str) -> str:
    """Round 3: 根据反思改进答案。"""
    prompt = REVISE_PROMPT.format(
        question=question, draft=draft, critique=critique
    )
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "(空)"


def reflection_flow(question: str) -> str:
    """Reflection 完整三轮流程（真实 API）。"""
    print(f"\n{'='*60}")
    print(f"Reflection 任务: {question}")
    print(f"{'='*60}")

    # Round 1: Draft
    print("\n--- Round 1: Draft（生成初版）---")
    draft = generate_draft(question)
    print("OUT:reflect:draft: 初版（前 200 字）:")
    print(f"OUT:reflect:draft: {draft[:200]}")

    # Round 2: Critique
    print("\n--- Round 2: Critique（反思）---")
    critique = reflect(question, draft)
    print("OUT:reflect:critique: 审稿意见:")
    print(f"OUT:reflect:critique: {critique[:200]}")

    # Round 3: Revise
    print("\n--- Round 3: Revise（改进）---")
    revised = revise(question, draft, critique)
    print("OUT:reflect:revised: 改进版（前 200 字）:")
    print(f"OUT:reflect:revised: {revised[:200]}")

    return revised


# ════════════════════════════════════════════════════════════════════
# 离线 mock：Plan-and-Execute（API 不可用时演示完整三阶段）
# ════════════════════════════════════════════════════════════════════


def demo_plan_and_execute_offline() -> str:
    """离线演示 Plan-and-Execute：预设步骤 + mock 执行 + 汇总。

    模拟任务："写一篇 AI Agent 调研报告"
    预设计划：4 个检索步骤 + 1 个汇总步骤。
    """
    print(f"\n{'='*60}")
    print("离线演示：Plan-and-Execute（规划→执行→汇总）")
    print(f"{'='*60}")
    print("[说明] 预设计划 + mock 执行，演示完整三阶段流程")

    task = "写一篇 AI Agent 调研报告"
    print(f"任务: {task}\n")

    # ── Phase 1: Plan（预设计划，模拟 plan_task() 的输出）──
    print("--- Phase 1: Plan（规划）---")
    mock_plan = Plan(steps=[
        "检索 AI Agent 的定义与核心特征",
        "检索 AI Agent 的典型应用场景",
        "检索主流 AI Agent 开发框架",
        "检索 AI Agent 的发展趋势与挑战",
        "综合以上信息撰写调研报告",
    ])
    print(f"OUT:plan: 分解出 {len(mock_plan.steps)} 个步骤:")
    for i, step in enumerate(mock_plan.steps, 1):
        print(f"OUT:plan:step{i}: {step}")

    # ── Phase 2: Execute（用 mock_search 真实执行检索步骤）──
    print("\n--- Phase 2: Execute（执行）---")
    steps_and_results: list[tuple[str, str]] = []
    for i, step in enumerate(mock_plan.steps, 1):
        print(f"OUT:execute:step{i}: 执行: {step}")
        if i < len(mock_plan.steps):
            # 前 4 步是检索，用 mock_search 执行
            result = execute_step(step, i)
        else:
            # 最后一步是汇总，留到 Phase 3
            result = "(汇总步骤，在 Phase 3 执行)"
        steps_and_results.append((step, result))
        print(f"OUT:execute:step{i}: 结果: {result[:80]}")

    # ── Phase 3: Synthesize（汇总检索结果，生成报告）──
    print("\n--- Phase 3: Synthesize（汇总）---")
    # 用真实检索结果拼接一个 mock 报告（不调 LLM）
    search_results = steps_and_results[:-1]  # 排除最后的汇总步骤
    report_lines = [f"# {task}\n"]
    for i, (step, result) in enumerate(search_results, 1):
        # 从步骤提取主题（如"定义与核心特征"→"定义"）
        topic = step.replace("检索 AI Agent 的", "").replace("的", "")
        report_lines.append(f"## {i}. {topic}")
        report_lines.append(result)
        report_lines.append("")
    final_report = "\n".join(report_lines)

    print("OUT:synthesize: 最终输出（前 300 字）:")
    print(f"OUT:synthesize: {final_report[:300]}")
    print(f"\nOUT:synthesize: ✓ Plan-and-Execute 完成，共 {len(mock_plan.steps)} 个步骤。")
    print("OUT:synthesize: 规划阶段一次性看清全貌，执行阶段机械检索，汇总阶段综合输出。")
    return final_report


# ════════════════════════════════════════════════════════════════════
# 离线 mock：Reflection（API 不可用时演示三轮反思改进）
# ════════════════════════════════════════════════════════════════════


def demo_reflection_offline() -> str:
    """离线演示 Reflection：预设"初版→反思→改进版"轨迹。

    模拟问题："什么是 AI Agent？请简要解释。"
    预设轨迹展示反思如何让答案从"简陋"变"完善"。
    """
    print(f"\n{'='*60}")
    print("离线演示：Reflection（初版→反思→改进版）")
    print(f"{'='*60}")
    print("[说明] 预设三轮轨迹，演示反思如何改进输出质量")

    question = "什么是 AI Agent？请简要解释。"
    print(f"问题: {question}\n")

    # ── Round 1: Draft（预设一个"简陋"的初版）──
    print("--- Round 1: Draft（生成初版）---")
    mock_draft = (
        "AI Agent 是一种人工智能系统，能自主完成任务。"
        "它可以调用工具、理解指令。"
    )
    print("OUT:reflect:draft: 初版:")
    print(f"OUT:reflect:draft: {mock_draft}")
    print("OUT:reflect:draft: [问题：太简略，缺少关键维度]")

    # ── Round 2: Critique（预设反思，指出 2-3 个不足）──
    print("\n--- Round 2: Critique（反思）---")
    mock_critique = (
        "初版存在以下不足：\n"
        "1. 完整性：只提了'能自主完成任务'，没解释 Agent 的核心组成"
        "（感知、决策、行动三要素）。\n"
        "2. 完整性：没有区分 Agent 和普通 LLM 的本质区别（工具调用、循环、自主性）。\n"
        "3. 结构：缺乏层次，信息密度低。"
    )
    print("OUT:reflect:critique: 审稿意见:")
    print(f"OUT:reflect:critique: {mock_critique}")

    # ── Round 3: Revise（预设改进版，体现反思的改进）──
    print("\n--- Round 3: Revise（改进）---")
    mock_revised = (
        "AI Agent 是能**感知环境、自主决策、采取行动**以实现目标的智能系统。"
        "它的三个核心要素：\n"
        "1. **感知**：接收用户输入或环境信号（如读取消息、监控数据）。\n"
        "2. **决策**：通过 LLM 推理决定下一步（如 ReAct 的 Thought）。\n"
        "3. **行动**：调用工具执行操作（如 function calling）。\n\n"
        "与普通 LLM 的区别：Agent 有**循环**（能多步执行）、**工具**（能调用外部能力）、"
        "**自主性**（能自己决定何时停止）。"
    )
    print("OUT:reflect:revised: 改进版:")
    print(f"OUT:reflect:revised: {mock_revised}")

    # ── 对比展示反思的价值 ──
    print("\n--- 反思价值对比 ---")
    print(f"OUT:reflect: 初版字数: {len(mock_draft)} | 改进版字数: {len(mock_revised)}")
    print("OUT:reflect: 改进点：补充了三要素、与 LLM 的区别、结构化呈现")
    print("\nOUT:reflect: ✓ Reflection 完成，三轮流程（Draft→Critique→Revise）。")
    print("OUT:reflect: 反思让答案从'简陋'变'完善'——代价是 3 次 LLM 调用的延迟。")
    return mock_revised


# ════════════════════════════════════════════════════════════════════
# 模式对比输出：Plan-and-Execute vs Reflection vs ReAct
# ════════════════════════════════════════════════════════════════════


def demo_comparison() -> None:
    """并排对比三种推理模式的核心差异。"""
    print(f"\n{'='*60}")
    print("对比：ReAct vs Plan-and-Execute vs Reflection")
    print(f"{'='*60}")

    comparisons = [
        ("核心思想", "边想边做", "先规划再执行", "做完反思改进"),
        ("流程", "Thought→Action 循环", "Plan→Execute→Synth", "Draft→Critique→Revise"),
        ("决策时机", "每步动态决策", "规划阶段一次决策", "执行后回顾决策"),
        ("适合任务", "步骤未知/需探索", "步骤明确/多阶段", "质量敏感的生成"),
        ("LLM 调用次数", "N 步 = N 次", "1 + N + 1 次", "3 次（固定）"),
        ("延迟", "中（取决于步数）", "中高（规划+执行）", "高（3 次调用）"),
        ("可预测性", "低（路径不确定）", "高（计划可审查）", "中"),
        ("可并行", "难（步骤间依赖）", "易（独立步骤并发）", "不适用"),
    ]

    header = (
        f"{'维度':<12} │ {'ReAct':<20} │ {'Plan-and-Execute':<20} │ {'Reflection':<20}"
    )
    print(f"OUT:compare: {header}")
    print(f"OUT:compare: {'─'*12}─┼─{'─'*20}─┼─{'─'*20}─┼─{'─'*20}")
    for dim, react, plan, reflect in comparisons:
        row = (
            f"{dim:<12} │ {react:<20} │ {plan:<20} │ {reflect:<20}"
        )
        print(f"OUT:compare: {row}")

    print("\nOUT:compare: 核心洞察：")
    print("OUT:compare: • ReAct = 灵活探索（适合步骤未知的任务）")
    print("OUT:compare: • Plan-and-Execute = 战略规划（适合多阶段复杂任务）")
    print("OUT:compare: • Reflection = 质量打磨（适合对输出质量要求高的场景）")
    print("OUT:compare: • 三者可组合：Plan→Execute（ReAct 执行每步）→Reflect（打磨汇总）")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print("[config] 模式 1: Plan-and-Execute（规划→执行→汇总）")
    print("[config] 模式 2: Reflection（初版→反思→改进）")
    print("[config] 输出标记: OUT:plan:, OUT:execute:step{N}:, OUT:synthesize:, OUT:reflect:")

    api_ok = True
    plan_task_input = "写一篇 AI Agent 调研报告"
    reflect_question = "什么是 AI Agent？请简要解释。"

    try:
        # ── Demo 1: Plan-and-Execute（真实 API）──
        print(f"\n{'#'*60}")
        print("# Demo 1: Plan-and-Execute（规划→执行→汇总）")
        print(f"{'#'*60}")
        plan_and_execute(plan_task_input)

        # ── Demo 2: Reflection（真实 API）──
        print(f"\n{'#'*60}")
        print("# Demo 2: Reflection（初版→反思→改进）")
        print(f"{'#'*60}")
        reflection_flow(reflect_question)

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
            print("[提示] 原因：API 密钥无效或为占位符。请编辑 ai-agent/.env 填入有效密钥。")
            print(f"[提示] 当前 provider={cfg.provider}，需要对应的 API 密钥。")
        else:
            print(f"[提示] 原因：{e}")
        print("[提示] 已自动降级为离线 mock 演示，规划逻辑不受影响。\n")

    # ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）────
    demo_plan_and_execute_offline()
    demo_reflection_offline()
    demo_comparison()

    print(f"\n{'='*60}")
    if api_ok:
        print("所有演示完成！（含真实 API + 离线 mock + 对比）")
    else:
        print("离线演示完成！（真实 API 未配置，但规划逻辑已完整展示）")
    print("💡 核心要点：复杂任务先规划（减少返工），质量任务加反思（提升质量）。")
    print("💡 简单任务别过度规划——会增加延迟和成本。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
