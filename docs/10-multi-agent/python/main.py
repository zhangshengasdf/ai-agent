"""第10章 多 Agent 编排（Supervisor-Worker、Handoffs、共享记忆）

本章演示两种多 Agent 协作模式（不引入任何框架，每个 Worker 就是一次 LLM 调用）：

  模式 1: Supervisor-Worker（调度-执行）
    - Supervisor 接收任务，用结构化输出分解成 [子任务+Worker] 列表
    - 把子任务依次分派给专门 Worker（Researcher/Writer/Coder）
    - Worker 各自有独立的 system prompt + 工具集，执行后返回结果
    - Supervisor 收集所有结果并汇总
    - 适合：任务可预先分解、角色多样、工具过载

  模式 2: Agent Handoff（任务转交）
    - 客服 Agent 处理用户问题（退货/咨询）
    - 检测到"技术关键词"（代码/bug/部署）时触发 Handoff
    - 把完整对话上下文传给技术 Agent
    - 技术 Agent 接管，继续处理并返回结果
    - 适合：任务进行中发现需要专家（客服→技术）

  共享记忆策略：
    - Supervisor-Worker 用"消息传递"（每个 Worker 返回独立结果，Supervisor 收集）
    - Handoff 用"共享对话历史"（换 system prompt，保留 user/assistant 消息）

离线 mock 设计：
  .env 的 OPENAI_API_KEY=sk-REPLACE-ME 是占位符，真实 API 调用必失败。
  所有功能先 try 真实 API（失败时降级），然后跑离线 mock，保证 exit 0。
  Supervisor-Worker mock：预设分派决策序列 + mock 执行 + 汇总。
  Handoff mock：预设"客服答→触发handoff→技术Agent答"轨迹。
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
# 协作协议（Pydantic 模型）—— Agent 间消息的"格式契约"
# ════════════════════════════════════════════════════════════════════


class Assignment(BaseModel):
    """Supervisor → Worker 的分派消息：哪个 Worker 干什么。"""

    worker: str  # Worker 名（"Researcher"/"Writer"/"Coder"）
    subtask: str  # 子任务描述


class AssignmentPlan(BaseModel):
    """Supervisor 输出的完整分派计划。"""

    assignments: list[Assignment]


# ════════════════════════════════════════════════════════════════════
# mock 工具：Worker 执行阶段用（复用第03/07章风格）
# ════════════════════════════════════════════════════════════════════


def search_wiki(query: str) -> str:
    """模拟百科搜索（mock 知识库）。"""
    knowledge = {
        "定义": "AI Agent 是能感知环境、自主决策、采取行动以实现目标的智能系统。",
        "应用": "AI Agent 应用于智能客服、编程助手、自动化研究、数据分析等场景。",
        "框架": "主流 AI Agent 框架有 LangChain、OpenAI Agents SDK、CrewAI、AutoGen 等。",
        "趋势": "AI Agent 正向多 Agent 协作、长程任务自主执行、工具自学习方向发展。",
        "挑战": "AI Agent 面临可靠性、成本控制、安全对齐、评估困难等挑战。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower or key in query:
            return value
    return f"检索到与'{query}'相关的通用信息。"


# ════════════════════════════════════════════════════════════════════
# Worker 定义：每个 Worker = name + system prompt + 工具集
# ════════════════════════════════════════════════════════════════════


class Worker:
    """一个专门化的 Agent：带特定 system prompt + 工具集的 LLM 调用。

    真实场景每个 Worker 可以是一个完整的 Agent 循环（第04章），
    本章为了教学清晰，简化为"单次 LLM 调用 + 可选工具"。
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: dict[str, object],
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools  # 工具名 → 可调用函数

    def execute(self, subtask: str, context: str = "") -> str:
        """执行子任务：用自己的 system prompt + 工具调 LLM。

        Args:
            subtask: Supervisor 分派的子任务描述。
            context: 前序 Worker 的结果（消息传递的上下文）。

        Returns:
            Worker 的输出文本。
        """
        user_content = subtask
        if context:
            user_content = f"前序结果：\n{context}\n\n你的任务：{subtask}"

        # 简单实现：单次调用。真实场景这里可以是一个 Agent 循环。
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or "(空)"

    def execute_with_tool(self, subtask: str, context: str = "") -> str:
        """Researcher 专用：先用工具检索，再调 LLM 整理。

        教学简化：直接用 mock 工具检索，把结果拼进 prompt 给 LLM。
        """
        tool_result = ""
        for tool_name, tool_fn in self.tools.items():
            tool_result = tool_fn(subtask)  # type: ignore[operator]
            break  # 只用第一个工具（教学简化）

        user_content = f"检索结果：{tool_result}\n\n你的任务：{subtask}"
        if context:
            user_content = f"前序结果：\n{context}\n\n{user_content}"

        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or "(空)"


# Worker 的 system prompt —— 每个 Worker 角色边界清晰、互斥
RESEARCHER_PROMPT = "你是研究员，负责检索信息。基于检索结果返回事实，不要编造。保持简洁。"
WRITER_PROMPT = "你是撰稿人，负责把信息整理成文。注重结构清晰和可读性。输出 Markdown 格式。"
CODER_PROMPT = "你是程序员，负责写代码片段。输出带语法标注的代码块，附简要说明。"

# 构建 Worker 团队
WORKERS: dict[str, Worker] = {
    "Researcher": Worker("Researcher", RESEARCHER_PROMPT, {"search_wiki": search_wiki}),
    "Writer": Worker("Writer", WRITER_PROMPT, {}),
    "Coder": Worker("Coder", CODER_PROMPT, {}),
}


# ════════════════════════════════════════════════════════════════════
# Supervisor-Worker 模式（真实 API）
# ════════════════════════════════════════════════════════════════════

SUPERVISOR_PROMPT = """\
你是一个任务调度者（Supervisor）。用户给你一个复杂任务，你需要把它分解并分派给专门的 Worker。

可用 Worker：
- Researcher：负责检索信息（有 search_wiki 工具）
- Writer：负责把信息整理成文
- Coder：负责写代码片段

输出 JSON 格式：
{"assignments": [{"worker": "Worker名", "subtask": "子任务描述"}, ...]}

要求：
- 按执行顺序列出分派（前面 Worker 的输出是后面 Worker 的输入）
- 每个 subtask 要具体、可执行
- 通常以 Researcher 开头（先调研），以 Writer 结尾（成文）
"""


def supervisor_decompose(task: str) -> AssignmentPlan:
    """Supervisor Phase 1：用结构化输出分解任务。

    用 response_format=json_object 强制 JSON，再用 Pydantic 解析校验。
    """
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": task},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return AssignmentPlan.model_validate_json(raw)


def worker_execute(worker_name: str, subtask: str, context: str = "") -> str:
    """Supervisor Phase 2：把子任务派给对应 Worker 执行。"""
    worker = WORKERS[worker_name]
    # Researcher 用带工具的执行；其他 Worker 用纯 LLM 执行
    if worker.tools:
        return worker.execute_with_tool(subtask, context)
    return worker.execute(subtask, context)


def supervisor_synthesize(task: str, results: list[tuple[str, str]]) -> str:
    """Supervisor Phase 3：收集所有 Worker 结果并汇总。

    真实场景这里可以是一个独立的 Synthesizer Agent。本章简化为单次 LLM 调用。
    """
    context = "\n".join(
        f"[{worker}] 的输出：\n{result}" for worker, result in results
    )
    prompt = (
        f"用户原始任务：{task}\n\n"
        f"各 Worker 的执行结果：\n{context}\n\n"
        f"请综合以上结果，完成用户的原始任务。输出最终成果。"
    )
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "(空)"


def supervisor_worker_flow(task: str) -> str:
    """Supervisor-Worker 完整三阶段流程（真实 API）。"""
    print(f"\n{'='*60}")
    print(f"Supervisor-Worker 任务: {task}")
    print(f"{'='*60}")

    # Phase 1: Supervisor 分解
    print("\n--- Phase 1: Supervisor 分解任务 ---")
    plan = supervisor_decompose(task)
    print(f"OUT:supervisor: 分解出 {len(plan.assignments)} 个分派:")
    for i, a in enumerate(plan.assignments, 1):
        print(f"OUT:supervisor:assignment{i}: → {a.worker}: {a.subtask}")

    # Phase 2: Worker 依次执行（消息传递：前一个的结果作为后一个的 context）
    print("\n--- Phase 2: Worker 执行 ---")
    results: list[tuple[str, str]] = []
    cumulative_context = ""
    for i, assignment in enumerate(plan.assignments, 1):
        print(f"OUT:supervisor: 分派给 {assignment.worker}: {assignment.subtask}")
        result = worker_execute(assignment.worker, assignment.subtask, cumulative_context)
        results.append((assignment.worker, result))
        # 消息传递：累积 context 给下一个 Worker
        cumulative_context += f"\n[{assignment.worker}] {result[:100]}..."
        print(f"OUT:worker:{assignment.worker}: 执行完成（前 80 字）: {result[:80]}")

    # Phase 3: Supervisor 汇总
    print("\n--- Phase 3: Supervisor 汇总 ---")
    final = supervisor_synthesize(task, results)
    print("OUT:supervisor: 最终汇总（前 200 字）:")
    print(f"OUT:supervisor: {final[:200]}")
    return final


# ════════════════════════════════════════════════════════════════════
# Agent Handoff 模式（真实 API）
# ════════════════════════════════════════════════════════════════════

# 客服 Agent 和技术 Agent 的 system prompt（角色边界清晰）
CUSTOMER_SERVICE_PROMPT = (
    "你是客服 Agent，负责处理退货、订单查询、退款等客服问题。"
    "回答要友好、简洁。如果遇到技术问题（代码/bug/部署/报错），"
    "请说 [HANDOFF_TECH] 并简要说明原因。"
)

TECH_EXPERT_PROMPT = (
    "你是技术专家 Agent，负责排查代码 bug、部署问题、系统错误。"
    "你从客服 Agent 接手了这个对话，已有完整对话历史。"
    "请基于上下文给出技术解决方案。"
)

# Handoff 触发关键词（简单可靠，不依赖 API）
TECH_KEYWORDS = ["代码", "bug", "部署", "错误码", "报错", "异常", "崩溃", "500", "404"]


def needs_handoff(user_message: str) -> bool:
    """检测用户消息是否包含技术关键词 → 需要转交给技术 Agent。"""
    return any(kw in user_message for kw in TECH_KEYWORDS)


def customer_service_respond(messages: list[dict[str, str]]) -> str:
    """客服 Agent 回复。"""
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
    )
    return response.choices[0].message.content or "(空)"


def tech_expert_resolve(messages: list[dict[str, str]]) -> str:
    """技术 Agent 回复（接手完整对话历史）。"""
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
    )
    return response.choices[0].message.content or "(空)"


def handoff_flow(conversation: list[tuple[str, str]]) -> str:
    """Agent Handoff 完整流程（真实 API）。

    Args:
        conversation: [(speaker, message), ...] 模拟多轮对话。
                      speaker 是 "user" 或 "assistant"。
    """
    print(f"\n{'='*60}")
    print("Agent Handoff 演示（客服 → 技术专家）")
    print(f"{'='*60}")

    # 共享对话历史（Handoff 的关键：换 system prompt，保留 user/assistant 消息）
    messages: list[dict[str, str]] = [{"role": "system", "content": CUSTOMER_SERVICE_PROMPT}]
    current_agent = "CustomerService"
    final_answer = "(未解决)"

    for speaker, text in conversation:
        if speaker == "user":
            messages.append({"role": "user", "content": text})
            print(f"\n用户: {text}")

            # 检测是否需要 Handoff
            if needs_handoff(text):
                print(f"OUT:handoff: 检测到技术关键词 → 触发 Handoff")
                # Handoff：替换 system prompt，保留对话历史
                messages = [
                    {"role": "system", "content": TECH_EXPERT_PROMPT},
                    *messages[1:],  # 去掉旧 system，保留 user/assistant
                ]
                current_agent = "TechExpert"
                print(f"OUT:handoff: 控制权转移: CustomerService → TechExpert")
                print(f"OUT:handoff: 对话历史（{len(messages)-1} 条）已传递给 TechExpert")
                # 技术 Agent 立即响应这个技术问题
                reply = tech_expert_resolve(messages)
                messages.append({"role": "assistant", "content": reply})
                print(f"OUT:resolve: TechExpert: {reply[:120]}")
                final_answer = reply
            else:
                # 客服 Agent 正常回复
                reply = customer_service_respond(messages)
                messages.append({"role": "assistant", "content": reply})
                print(f"OUT:worker:CustomerService: {reply[:120]}")
                final_answer = reply
        # speaker == "assistant" 的预设消息直接跳过（教学用，模拟已有对话）

    print(f"\nOUT:resolve: 最终由 {current_agent} 处理完成")
    print(f"OUT:resolve: 最终回答（前 150 字）: {final_answer[:150]}")
    return final_answer


# ════════════════════════════════════════════════════════════════════
# 离线 mock：Supervisor-Worker（API 不可用时演示完整流程）
# ════════════════════════════════════════════════════════════════════


def demo_supervisor_worker_offline() -> str:
    """离线演示 Supervisor-Worker：预设分派 + mock 执行 + 汇总。

    模拟任务："调研 AI Agent 并写成报告"
    预设分派：Researcher 检索定义 → Researcher 检索应用 → Writer 写报告。
    """
    print(f"\n{'='*60}")
    print("离线演示：Supervisor-Worker（分解→分派→执行→汇总）")
    print(f"{'='*60}")
    print("[说明] 预设分派决策 + mock 执行，演示完整三阶段流程")

    task = "调研 AI Agent 并写成报告"
    print(f"任务: {task}\n")

    # ── Phase 1: Supervisor 分解（预设分派计划）──
    print("--- Phase 1: Supervisor 分解任务 ---")
    mock_plan = AssignmentPlan(assignments=[
        Assignment(worker="Researcher", subtask="检索 AI Agent 的定义与核心特征"),
        Assignment(worker="Researcher", subtask="检索 AI Agent 的典型应用场景"),
        Assignment(worker="Writer", subtask="把检索结果写成一份结构清晰的报告"),
    ])
    print(f"OUT:supervisor: 分解出 {len(mock_plan.assignments)} 个分派:")
    for i, a in enumerate(mock_plan.assignments, 1):
        print(f"OUT:supervisor:assignment{i}: → {a.worker}: {a.subtask}")

    # ── Phase 2: Worker 执行（用真实 mock 工具 + 预设输出）──
    print("\n--- Phase 2: Worker 执行 ---")
    results: list[tuple[str, str]] = []
    research_context = ""

    for i, assignment in enumerate(mock_plan.assignments, 1):
        print(f"OUT:supervisor: 分派给 {assignment.worker}: {assignment.subtask}")
        if assignment.worker == "Researcher":
            # Researcher 用 search_wiki 真实检索
            tool_result = search_wiki(assignment.subtask)
            result = f"[Researcher 整理] {tool_result}"
            research_context += f"\n{tool_result}"
        elif assignment.worker == "Writer":
            # Writer 基于检索结果"写报告"（mock，不调 LLM）
            result = _mock_writer_output(task, research_context)
        else:
            result = f"[{assignment.worker}] (mock 执行)"
        results.append((assignment.worker, result))
        print(f"OUT:worker:{assignment.worker}: 执行完成（前 80 字）: {result[:80]}")

    # ── Phase 3: Supervisor 汇总 ──
    print("\n--- Phase 3: Supervisor 汇总 ---")
    final_report = _mock_supervisor_synthesis(task, results)
    print("OUT:supervisor: 最终汇总（前 300 字）:")
    print(f"OUT:supervisor: {final_report[:300]}")
    print(f"\nOUT:supervisor: ✓ Supervisor-Worker 完成，{len(mock_plan.assignments)} 个分派。")
    print("OUT:supervisor: Supervisor 只做路由，Researcher 检索，Writer 写作——职责分离。")
    return final_report


def _mock_writer_output(task: str, research_context: str) -> str:
    """离线 mock：Writer 基于检索结果生成报告（不调 LLM）。"""
    return (
        f"# {task}\n\n"
        f"## 1. 定义\nAI Agent 是能感知环境、自主决策、采取行动以实现目标的智能系统。\n\n"
        f"## 2. 应用\nAI Agent 应用于智能客服、编程助手、自动化研究、数据分析等场景。\n\n"
        f"## 3. 框架\n主流框架有 LangChain、OpenAI Agents SDK、CrewAI、AutoGen 等。\n\n"
        f"## 4. 趋势\nAI Agent 正向多 Agent 协作、长程任务自主执行方向发展。"
    )


def _mock_supervisor_synthesis(task: str, results: list[tuple[str, str]]) -> str:
    """离线 mock：Supervisor 汇总各 Worker 结果（不调 LLM）。"""
    # 直接用 Writer 的输出作为最终报告
    for worker, result in results:
        if worker == "Writer":
            return result
    return "(汇总失败：未找到 Writer 输出)"


# ════════════════════════════════════════════════════════════════════
# 离线 mock：Agent Handoff（API 不可用时演示客服→技术转交）
# ════════════════════════════════════════════════════════════════════


def demo_handoff_offline() -> str:
    """离线演示 Agent Handoff：预设"客服答→触发handoff→技术Agent答"轨迹。

    模拟对话：
      用户：我想退货
      客服：好的，请提供订单号（客服正常处理）
      用户：退货页面报了 500 错误，代码 ERR_DEPLOY_123（触发技术关键词）
      → Handoff → 技术专家：这是部署问题，我帮你排查...
    """
    print(f"\n{'='*60}")
    print("离线演示：Agent Handoff（客服 → 技术专家）")
    print(f"{'='*60}")
    print("[说明] 预设对话轨迹，演示 Handoff 的上下文传递")

    # ── Turn 1: 用户咨询退货（客服正常处理）──
    print("\n--- Turn 1: 用户咨询退货 ---")
    user_msg_1 = "你好，我想退货，订单号 ORD-2024-001"
    print(f"用户: {user_msg_1}")
    # 关键词检测
    if needs_handoff(user_msg_1):
        print("OUT:handoff: [未触发] 无技术关键词，客服继续处理")
    else:
        print("OUT:handoff: [未触发] 无技术关键词，客服继续处理")
    cs_reply_1 = "您好！收到您的退货请求（订单 ORD-2024-001）。请告诉我退货原因，我帮您处理。"
    print(f"OUT:worker:CustomerService: {cs_reply_1}")

    # ── Turn 2: 用户追问技术问题（触发 Handoff）──
    print("\n--- Turn 2: 用户追问技术问题（触发 Handoff）---")
    user_msg_2 = "退货页面报了 500 错误，错误代码 ERR_DEPLOY_123，好像是部署问题"
    print(f"用户: {user_msg_2}")
    triggered = needs_handoff(user_msg_2)
    print(f"OUT:handoff: 关键词检测: {'触发' if triggered else '未触发'}")
    if triggered:
        matched = [kw for kw in TECH_KEYWORDS if kw in user_msg_2]
        print(f"OUT:handoff: 命中关键词: {matched}")

    # ── Handoff：客服 → 技术专家 ──
    print("\n--- Handoff 执行 ---")
    print("OUT:handoff: 客服 Agent 判断：这是技术问题，超出客服职责范围")
    print("OUT:handoff: 触发 Handoff: CustomerService → TechExpert")
    print(f"OUT:handoff: 传递对话历史（2 轮 user + 1 轮 assistant）")
    print("OUT:handoff: 替换 system prompt: 客服角色 → 技术专家角色")

    # ── 技术 Agent 接管并解决 ──
    print("\n--- TechExpert 接管 ---")
    tech_reply = (
        "我看到你遇到了 ERR_DEPLOY_123 错误（500 内部服务器错误）。\n"
        "这是已知的部署问题——最新版本的一个回滚配置缺失。\n"
        "解决方案：\n"
        "1. 清除浏览器缓存后重试（临时方案）\n"
        "2. 我已通知运维团队紧急修复（预计 15 分钟内恢复）\n"
        "3. 你也可以先联系客服走人工退货流程作为备选\n"
        "抱歉给你带来不便，我们会跟进直到问题解决。"
    )
    print(f"OUT:worker:TechExpert: (基于完整上下文响应)")
    print(f"OUT:resolve: TechExpert: {tech_reply[:150]}")

    # ── Handoff 价值展示 ──
    print("\n--- Handoff 价值 ---")
    print("OUT:resolve: ✓ Handoff 完成。客服处理不了的技术问题，转交技术专家解决。")
    print("OUT:resolve: 关键：技术专家看到了完整对话历史，知道用户是来退货的，不是凭空出现。")
    print("OUT:resolve: 对比：如果没有 Handoff，客服只能'我帮您反馈一下'，用户体验差。")
    return tech_reply


# ════════════════════════════════════════════════════════════════════
# 模式对比输出：Supervisor-Worker vs Handoff
# ════════════════════════════════════════════════════════════════════


def demo_comparison() -> None:
    """并排对比 Supervisor-Worker 和 Handoff 两种多 Agent 模式。"""
    print(f"\n{'='*60}")
    print("对比：Supervisor-Worker vs Agent Handoff")
    print(f"{'='*60}")

    comparisons = [
        ("核心思想", "Supervisor 调度 Worker 分工", "Agent 把任务转交给专家"),
        ("触发方式", "Supervisor 主动分派", "当前 Agent 判断超范围时被动触发"),
        ("控制流", "Supervisor 始终主导", "控制权完全转移（A → B）"),
        ("上下文", "分派时传递 subtask（精简）", "传递完整对话历史（完整）"),
        ("角色关系", "Supervisor + 多个平行 Worker", "通用 Agent + 专用专家"),
        ("适合场景", "任务可预先分解", "任务进行中发现需专家"),
        ("类比", "项目经理分派任务给组员", "客服把电话转给技术支持"),
        ("失败模式", "分派错误 Worker / Worker 失败", "无限踢皮球（反模式5）"),
    ]

    header = (
        f"{'维度':<12} │ {'Supervisor-Worker':<28} │ {'Agent Handoff':<28}"
    )
    print(f"OUT:compare: {header}")
    print(f"OUT:compare: {'─'*12}─┼─{'─'*28}─┼─{'─'*28}")
    for dim, sw, ho in comparisons:
        row = f"{dim:<12} │ {sw:<28} │ {ho:<28}"
        print(f"OUT:compare: {row}")

    print("\nOUT:compare: 核心洞察：")
    print("OUT:compare: • Supervisor-Worker = 团队分工（适合可分解的复杂任务）")
    print("OUT:compare: • Handoff = 专家路由（适合进行中发现的专业问题）")
    print("OUT:compare: • 两者可组合：Supervisor 分派时，某 Worker 内部可触发 Handoff")
    print("OUT:compare: • 记住：简单任务别上多 Agent（Anthropic 共识）")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print("[config] 模式 1: Supervisor-Worker（调度-执行）")
    print("[config] 模式 2: Agent Handoff（任务转交）")
    print("[config] 输出标记: OUT:supervisor:, OUT:worker:{name}:, OUT:handoff:, OUT:resolve:")

    api_ok = True
    sw_task = "调研 AI Agent 并写成报告"
    handoff_conversation = [
        ("user", "你好，我想退货，订单号 ORD-2024-001"),
        ("user", "退货页面报了 500 错误，错误代码 ERR_DEPLOY_123，好像是部署问题"),
    ]

    try:
        # ── Demo 1: Supervisor-Worker（真实 API）──
        print(f"\n{'#'*60}")
        print("# Demo 1: Supervisor-Worker（分解→分派→执行→汇总）")
        print(f"{'#'*60}")
        supervisor_worker_flow(sw_task)

        # ── Demo 2: Agent Handoff（真实 API）──
        print(f"\n{'#'*60}")
        print("# Demo 2: Agent Handoff（客服 → 技术专家）")
        print(f"{'#'*60}")
        handoff_flow(handoff_conversation)

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
        print("[提示] 已自动降级为离线 mock 演示，多 Agent 逻辑不受影响。\n")

    # ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）────
    demo_supervisor_worker_offline()
    demo_handoff_offline()
    demo_comparison()

    print(f"\n{'='*60}")
    if api_ok:
        print("所有演示完成！（含真实 API + 离线 mock + 对比）")
    else:
        print("离线演示完成！（真实 API 未配置，但多 Agent 逻辑已完整展示）")
    print("💡 核心要点：复杂任务用 Supervisor-Worker 分工，专业问题用 Handoff 转交。")
    print("💡 但记住 Anthropic 共识：从最简单的方案开始，许多场景只需优化单次 LLM 调用。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
