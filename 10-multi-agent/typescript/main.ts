/**
 * 第10章 多 Agent 编排（Supervisor-Worker、Handoffs、共享记忆）
 *
 * 本章演示两种多 Agent 协作模式（不引入任何框架，每个 Worker 就是一次 LLM 调用）：
 *
 *   模式 1: Supervisor-Worker（调度-执行）
 *     - Supervisor 接收任务，用结构化输出分解成 [子任务+Worker] 列表
 *     - 把子任务依次分派给专门 Worker（Researcher/Writer/Coder）
 *     - Worker 各自有独立的 system prompt + 工具集，执行后返回结果
 *     - Supervisor 收集所有结果并汇总
 *
 *   模式 2: Agent Handoff（任务转交）
 *     - 客服 Agent 处理用户问题（退货/咨询）
 *     - 检测到"技术关键词"（代码/bug/部署）时触发 Handoff
 *     - 把完整对话上下文传给技术 Agent
 *     - 技术 Agent 接管，继续处理并返回结果
 *
 * 离线 mock 设计：
 *   .env 的 OPENAI_API_KEY=sk-REPLACE-ME 是占位符，真实 API 调用必失败。
 *   所有功能先 try 真实 API（失败时降级），然后跑离线 mock，保证 exit 0。
 *   Supervisor-Worker mock：预设分派决策序列 + mock 执行 + 汇总。
 *   Handoff mock：预设"客服答→触发handoff→技术Agent答"轨迹。
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 协作协议（interface）—— Agent 间消息的"格式契约"
// ════════════════════════════════════════════════════════════════════

/** Supervisor → Worker 的分派消息：哪个 Worker 干什么。 */
interface Assignment {
  worker: string; // Worker 名（"Researcher"/"Writer"/"Coder"）
  subtask: string; // 子任务描述
}

/** Supervisor 输出的完整分派计划。 */
interface AssignmentPlan {
  assignments: Assignment[];
}

/** Worker 执行结果（worker 名 + 输出文本）。 */
interface WorkerResult {
  worker: string;
  result: string;
}

/** 一个 Worker 的工具集：工具名 → 可调用函数。 */
type ToolMap = Record<string, (...args: string[]) => string>;

/** 对话消息（OpenAI Chat 格式）。 */
interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

// ════════════════════════════════════════════════════════════════════
// Worker 类型定义（type Worker = { name, systemPrompt, tools }）
// ════════════════════════════════════════════════════════════════════

/**
 * 一个专门化的 Agent：带特定 system prompt + 工具集的 LLM 调用。
 * 真实场景每个 Worker 可以是一个完整的 Agent 循环（第04章），
 * 本章为了教学清晰，简化为"单次 LLM 调用 + 可选工具"。
 */
type Worker = {
  name: string;
  systemPrompt: string;
  tools: ToolMap;
};

// ════════════════════════════════════════════════════════════════════
// mock 工具：Worker 执行阶段用（复用第03/07章风格）
// ════════════════════════════════════════════════════════════════════

function searchWiki(...args: string[]): string {
  /**模拟百科搜索（mock 知识库）。*/
  const knowledge: Record<string, string> = {
    定义: "AI Agent 是能感知环境、自主决策、采取行动以实现目标的智能系统。",
    应用: "AI Agent 应用于智能客服、编程助手、自动化研究、数据分析等场景。",
    框架: "主流 AI Agent 框架有 LangChain、OpenAI Agents SDK、CrewAI、AutoGen 等。",
    趋势: "AI Agent 正向多 Agent 协作、长程任务自主执行、工具自学习方向发展。",
    挑战: "AI Agent 面临可靠性、成本控制、安全对齐、评估困难等挑战。",
  };
  const query = args.join(" ");
  const queryLower = query.toLowerCase();
  for (const [key, value] of Object.entries(knowledge)) {
    if (queryLower.includes(key) || query.includes(key)) {
      return value;
    }
  }
  return `检索到与'${query}'相关的通用信息。`;
}

// Worker 的 system prompt —— 每个 Worker 角色边界清晰、互斥
const RESEARCHER_PROMPT =
  "你是研究员，负责检索信息。基于检索结果返回事实，不要编造。保持简洁。";
const WRITER_PROMPT =
  "你是撰稿人，负责把信息整理成文。注重结构清晰和可读性。输出 Markdown 格式。";
const CODER_PROMPT =
  "你是程序员，负责写代码片段。输出带语法标注的代码块，附简要说明。";

// 构建 Worker 团队
const WORKERS: Record<string, Worker> = {
  Researcher: { name: "Researcher", systemPrompt: RESEARCHER_PROMPT, tools: { search_wiki: searchWiki } },
  Writer: { name: "Writer", systemPrompt: WRITER_PROMPT, tools: {} },
  Coder: { name: "Coder", systemPrompt: CODER_PROMPT, tools: {} },
};

// ════════════════════════════════════════════════════════════════════
// Supervisor-Worker 模式（真实 API）
// ════════════════════════════════════════════════════════════════════

const SUPERVISOR_PROMPT = `\
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
`;

async function supervisorDecompose(task: string): Promise<AssignmentPlan> {
  /** Supervisor Phase 1：用结构化输出分解任务。
   * 用 response_format=json_object 强制 JSON，再用 JSON.parse 解析。*/
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [
      { role: "system", content: SUPERVISOR_PROMPT },
      { role: "user", content: task },
    ],
    response_format: { type: "json_object" },
  });
  const raw = response.choices[0].message.content ?? "{}";
  const parsed = JSON.parse(raw) as AssignmentPlan;
  // 基本校验：assignments 必须是数组
  if (!Array.isArray(parsed.assignments)) {
    throw new Error(
      `Invalid plan: assignments is not an array (got ${typeof parsed.assignments})`,
    );
  }
  return parsed;
}

async function workerExecute(
  workerName: string,
  subtask: string,
  context = "",
): Promise<string> {
  /** Supervisor Phase 2：把子任务派给对应 Worker 执行。*/
  const worker = WORKERS[workerName];
  // Researcher 用带工具的执行；其他 Worker 用纯 LLM 执行
  if (Object.keys(worker.tools).length > 0) {
    return workerExecuteWithTool(worker, subtask, context);
  }
  return workerExecutePure(worker, subtask, context);
}

async function workerExecutePure(
  worker: Worker,
  subtask: string,
  context: string,
): Promise<string> {
  /**纯 LLM 执行（Writer/Coder）。*/
  let userContent = subtask;
  if (context) {
    userContent = `前序结果：\n${context}\n\n你的任务：${subtask}`;
  }
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [
      { role: "system", content: worker.systemPrompt },
      { role: "user", content: userContent },
    ],
  });
  return response.choices[0].message.content ?? "(空)";
}

async function workerExecuteWithTool(
  worker: Worker,
  subtask: string,
  context: string,
): Promise<string> {
  /**Researcher 专用：先用工具检索，再调 LLM 整理。
   * 教学简化：直接用 mock 工具检索，把结果拼进 prompt 给 LLM。*/
  let toolResult = "";
  for (const toolFn of Object.values(worker.tools)) {
    toolResult = toolFn(subtask);
    break; // 只用第一个工具（教学简化）
  }

  let userContent = `检索结果：${toolResult}\n\n你的任务：${subtask}`;
  if (context) {
    userContent = `前序结果：\n${context}\n\n${userContent}`;
  }

  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [
      { role: "system", content: worker.systemPrompt },
      { role: "user", content: userContent },
    ],
  });
  return response.choices[0].message.content ?? "(空)";
}

async function supervisorSynthesize(
  task: string,
  results: WorkerResult[],
): Promise<string> {
  /** Supervisor Phase 3：收集所有 Worker 结果并汇总。*/
  const context = results
    .map((r) => `[${r.worker}] 的输出：\n${r.result}`)
    .join("\n");
  const prompt =
    `用户原始任务：${task}\n\n` +
    `各 Worker 的执行结果：\n${context}\n\n` +
    `请综合以上结果，完成用户的原始任务。输出最终成果。`;
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [{ role: "user", content: prompt }],
  });
  return response.choices[0].message.content ?? "(空)";
}

async function supervisorWorkerFlow(task: string): Promise<string> {
  /** Supervisor-Worker 完整三阶段流程（真实 API）。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log(`Supervisor-Worker 任务: ${task}`);
  console.log("=".repeat(60));

  // Phase 1: Supervisor 分解
  console.log("\n--- Phase 1: Supervisor 分解任务 ---");
  const plan = await supervisorDecompose(task);
  console.log(`OUT:supervisor: 分解出 ${plan.assignments.length} 个分派:`);
  plan.assignments.forEach((a, i) => {
    console.log(`OUT:supervisor:assignment${i + 1}: → ${a.worker}: ${a.subtask}`);
  });

  // Phase 2: Worker 依次执行（消息传递：前一个的结果作为后一个的 context）
  console.log("\n--- Phase 2: Worker 执行 ---");
  const results: WorkerResult[] = [];
  let cumulativeContext = "";
  for (const assignment of plan.assignments) {
    console.log(`OUT:supervisor: 分派给 ${assignment.worker}: ${assignment.subtask}`);
    const result = await workerExecute(
      assignment.worker,
      assignment.subtask,
      cumulativeContext,
    );
    results.push({ worker: assignment.worker, result });
    // 消息传递：累积 context 给下一个 Worker
    cumulativeContext += `\n[${assignment.worker}] ${result.slice(0, 100)}...`;
    console.log(
      `OUT:worker:${assignment.worker}: 执行完成（前 80 字）: ${result.slice(0, 80)}`,
    );
  }

  // Phase 3: Supervisor 汇总
  console.log("\n--- Phase 3: Supervisor 汇总 ---");
  const final = await supervisorSynthesize(task, results);
  console.log(`OUT:supervisor: 最终汇总（前 200 字）:`);
  console.log(`OUT:supervisor: ${final.slice(0, 200)}`);
  return final;
}

// ════════════════════════════════════════════════════════════════════
// Agent Handoff 模式（真实 API）
// ════════════════════════════════════════════════════════════════════

// 客服 Agent 和技术 Agent 的 system prompt（角色边界清晰）
const CUSTOMER_SERVICE_PROMPT =
  "你是客服 Agent，负责处理退货、订单查询、退款等客服问题。" +
  "回答要友好、简洁。如果遇到技术问题（代码/bug/部署/报错），" +
  "请说 [HANDOFF_TECH] 并简要说明原因。";

const TECH_EXPERT_PROMPT =
  "你是技术专家 Agent，负责排查代码 bug、部署问题、系统错误。" +
  "你从客服 Agent 接手了这个对话，已有完整对话历史。" +
  "请基于上下文给出技术解决方案。";

// Handoff 触发关键词（简单可靠，不依赖 API）
const TECH_KEYWORDS = [
  "代码", "bug", "部署", "错误码", "报错", "异常", "崩溃", "500", "404",
];

function needsHandoff(userMessage: string): boolean {
  /**检测用户消息是否包含技术关键词 → 需要转交给技术 Agent。*/
  return TECH_KEYWORDS.some((kw) => userMessage.includes(kw));
}

async function customerServiceRespond(
  messages: ChatMessage[],
): Promise<string> {
  /**客服 Agent 回复。*/
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
  });
  return response.choices[0].message.content ?? "(空)";
}

async function techExpertResolve(messages: ChatMessage[]): Promise<string> {
  /**技术 Agent 回复（接手完整对话历史）。*/
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
  });
  return response.choices[0].message.content ?? "(空)";
}

/** 对话条目：说话者 + 消息内容（模拟多轮对话）。 */
interface ConversationTurn {
  speaker: "user" | "assistant";
  message: string;
}

async function handoffFlow(conversation: ConversationTurn[]): Promise<string> {
  /** Agent Handoff 完整流程（真实 API）。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("Agent Handoff 演示（客服 → 技术专家）");
  console.log("=".repeat(60));

  // 共享对话历史（Handoff 的关键：换 system prompt，保留 user/assistant 消息）
  let messages: ChatMessage[] = [
    { role: "system", content: CUSTOMER_SERVICE_PROMPT },
  ];
  let currentAgent = "CustomerService";
  let finalAnswer = "(未解决)";

  for (const turn of conversation) {
    if (turn.speaker === "user") {
      messages.push({ role: "user", content: turn.message });
      console.log(`\n用户: ${turn.message}`);

      // 检测是否需要 Handoff
      if (needsHandoff(turn.message)) {
        console.log(`OUT:handoff: 检测到技术关键词 → 触发 Handoff`);
        // Handoff：替换 system prompt，保留对话历史
        messages = [
          { role: "system", content: TECH_EXPERT_PROMPT },
          ...messages.slice(1), // 去掉旧 system，保留 user/assistant
        ];
        currentAgent = "TechExpert";
        console.log(`OUT:handoff: 控制权转移: CustomerService → TechExpert`);
        console.log(
          `OUT:handoff: 对话历史（${messages.length - 1} 条）已传递给 TechExpert`,
        );
        // 技术 Agent 立即响应这个技术问题
        const reply = await techExpertResolve(messages);
        messages.push({ role: "assistant", content: reply });
        console.log(`OUT:resolve: TechExpert: ${reply.slice(0, 120)}`);
        finalAnswer = reply;
      } else {
        // 客服 Agent 正常回复
        const reply = await customerServiceRespond(messages);
        messages.push({ role: "assistant", content: reply });
        console.log(`OUT:worker:CustomerService: ${reply.slice(0, 120)}`);
        finalAnswer = reply;
      }
    }
    // speaker === "assistant" 的预设消息直接跳过（教学用，模拟已有对话）
  }

  console.log(`\nOUT:resolve: 最终由 ${currentAgent} 处理完成`);
  console.log(`OUT:resolve: 最终回答（前 150 字）: ${finalAnswer.slice(0, 150)}`);
  return finalAnswer;
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：Supervisor-Worker（API 不可用时演示完整流程）
// ════════════════════════════════════════════════════════════════════

function demoSupervisorWorkerOffline(): string {
  /**离线演示 Supervisor-Worker：预设分派 + mock 执行 + 汇总。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("离线演示：Supervisor-Worker（分解→分派→执行→汇总）");
  console.log("=".repeat(60));
  console.log("[说明] 预设分派决策 + mock 执行，演示完整三阶段流程");

  const task = "调研 AI Agent 并写成报告";
  console.log(`任务: ${task}\n`);

  // ── Phase 1: Supervisor 分解（预设分派计划）──
  console.log("--- Phase 1: Supervisor 分解任务 ---");
  const mockPlan: AssignmentPlan = {
    assignments: [
      { worker: "Researcher", subtask: "检索 AI Agent 的定义与核心特征" },
      { worker: "Researcher", subtask: "检索 AI Agent 的典型应用场景" },
      { worker: "Writer", subtask: "把检索结果写成一份结构清晰的报告" },
    ],
  };
  console.log(`OUT:supervisor: 分解出 ${mockPlan.assignments.length} 个分派:`);
  mockPlan.assignments.forEach((a, i) => {
    console.log(`OUT:supervisor:assignment${i + 1}: → ${a.worker}: ${a.subtask}`);
  });

  // ── Phase 2: Worker 执行（用真实 mock 工具 + 预设输出）──
  console.log("\n--- Phase 2: Worker 执行 ---");
  const results: WorkerResult[] = [];
  let researchContext = "";

  for (const assignment of mockPlan.assignments) {
    console.log(`OUT:supervisor: 分派给 ${assignment.worker}: ${assignment.subtask}`);
    let result: string;
    if (assignment.worker === "Researcher") {
      // Researcher 用 searchWiki 真实检索
      const toolResult = searchWiki(assignment.subtask);
      result = `[Researcher 整理] ${toolResult}`;
      researchContext += `\n${toolResult}`;
    } else if (assignment.worker === "Writer") {
      // Writer 基于检索结果"写报告"（mock，不调 LLM）
      result = mockWriterOutput(task, researchContext);
    } else {
      result = `[${assignment.worker}] (mock 执行)`;
    }
    results.push({ worker: assignment.worker, result });
    console.log(
      `OUT:worker:${assignment.worker}: 执行完成（前 80 字）: ${result.slice(0, 80)}`,
    );
  }

  // ── Phase 3: Supervisor 汇总 ──
  console.log("\n--- Phase 3: Supervisor 汇总 ---");
  const finalReport = mockSupervisorSynthesis(task, results);
  console.log(`OUT:supervisor: 最终汇总（前 300 字）:`);
  console.log(`OUT:supervisor: ${finalReport.slice(0, 300)}`);
  console.log(
    `\nOUT:supervisor: ✓ Supervisor-Worker 完成，${mockPlan.assignments.length} 个分派。`,
  );
  console.log(
    `OUT:supervisor: Supervisor 只做路由，Researcher 检索，Writer 写作——职责分离。`,
  );
  return finalReport;
}

function mockWriterOutput(task: string, _researchContext: string): string {
  /**离线 mock：Writer 基于检索结果生成报告（不调 LLM）。*/
  return (
    `# ${task}\n\n` +
    `## 1. 定义\nAI Agent 是能感知环境、自主决策、采取行动以实现目标的智能系统。\n\n` +
    `## 2. 应用\nAI Agent 应用于智能客服、编程助手、自动化研究、数据分析等场景。\n\n` +
    `## 3. 框架\n主流框架有 LangChain、OpenAI Agents SDK、CrewAI、AutoGen 等。\n\n` +
    `## 4. 趋势\nAI Agent 正向多 Agent 协作、长程任务自主执行方向发展。`
  );
}

function mockSupervisorSynthesis(
  _task: string,
  results: WorkerResult[],
): string {
  /**离线 mock：Supervisor 汇总各 Worker 结果（不调 LLM）。*/
  // 直接用 Writer 的输出作为最终报告
  for (const r of results) {
    if (r.worker === "Writer") {
      return r.result;
    }
  }
  return "(汇总失败：未找到 Writer 输出)";
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：Agent Handoff（API 不可用时演示客服→技术转交）
// ════════════════════════════════════════════════════════════════════

function demoHandoffOffline(): string {
  /**离线演示 Agent Handoff：预设"客服答→触发handoff→技术Agent答"轨迹。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("离线演示：Agent Handoff（客服 → 技术专家）");
  console.log("=".repeat(60));
  console.log("[说明] 预设对话轨迹，演示 Handoff 的上下文传递");

  // ── Turn 1: 用户咨询退货（客服正常处理）──
  console.log("\n--- Turn 1: 用户咨询退货 ---");
  const userMsg1 = "你好，我想退货，订单号 ORD-2024-001";
  console.log(`用户: ${userMsg1}`);
  // 关键词检测
  if (needsHandoff(userMsg1)) {
    console.log("OUT:handoff: [未触发] 无技术关键词，客服继续处理");
  } else {
    console.log("OUT:handoff: [未触发] 无技术关键词，客服继续处理");
  }
  const csReply1 =
    "您好！收到您的退货请求（订单 ORD-2024-001）。请告诉我退货原因，我帮您处理。";
  console.log(`OUT:worker:CustomerService: ${csReply1}`);

  // ── Turn 2: 用户追问技术问题（触发 Handoff）──
  console.log("\n--- Turn 2: 用户追问技术问题（触发 Handoff）---");
  const userMsg2 =
    "退货页面报了 500 错误，错误代码 ERR_DEPLOY_123，好像是部署问题";
  console.log(`用户: ${userMsg2}`);
  const triggered = needsHandoff(userMsg2);
  console.log(`OUT:handoff: 关键词检测: ${triggered ? "触发" : "未触发"}`);
  if (triggered) {
    const matched = TECH_KEYWORDS.filter((kw) => userMsg2.includes(kw));
    console.log(`OUT:handoff: 命中关键词: ${JSON.stringify(matched)}`);
  }

  // ── Handoff：客服 → 技术专家 ──
  console.log("\n--- Handoff 执行 ---");
  console.log("OUT:handoff: 客服 Agent 判断：这是技术问题，超出客服职责范围");
  console.log("OUT:handoff: 触发 Handoff: CustomerService → TechExpert");
  console.log(`OUT:handoff: 传递对话历史（2 轮 user + 1 轮 assistant）`);
  console.log("OUT:handoff: 替换 system prompt: 客服角色 → 技术专家角色");

  // ── 技术 Agent 接管并解决 ──
  console.log("\n--- TechExpert 接管 ---");
  const techReply =
    "我看到你遇到了 ERR_DEPLOY_123 错误（500 内部服务器错误）。\n" +
    "这是已知的部署问题——最新版本的一个回滚配置缺失。\n" +
    "解决方案：\n" +
    "1. 清除浏览器缓存后重试（临时方案）\n" +
    "2. 我已通知运维团队紧急修复（预计 15 分钟内恢复）\n" +
    "3. 你也可以先联系客服走人工退货流程作为备选\n" +
    "抱歉给你带来不便，我们会跟进直到问题解决。";
  console.log(`OUT:worker:TechExpert: (基于完整上下文响应)`);
  console.log(`OUT:resolve: TechExpert: ${techReply.slice(0, 150)}`);

  // ── Handoff 价值展示 ──
  console.log("\n--- Handoff 价值 ---");
  console.log(
    `OUT:resolve: ✓ Handoff 完成。客服处理不了的技术问题，转交技术专家解决。`,
  );
  console.log(
    `OUT:resolve: 关键：技术专家看到了完整对话历史，知道用户是来退货的，不是凭空出现。`,
  );
  console.log(
    `OUT:resolve: 对比：如果没有 Handoff，客服只能'我帮您反馈一下'，用户体验差。`,
  );
  return techReply;
}

// ════════════════════════════════════════════════════════════════════
// 模式对比输出：Supervisor-Worker vs Handoff
// ════════════════════════════════════════════════════════════════════

function demoComparison(): void {
  /**并排对比 Supervisor-Worker 和 Handoff 两种多 Agent 模式。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("对比：Supervisor-Worker vs Agent Handoff");
  console.log("=".repeat(60));

  const comparisons: Array<[string, string, string]> = [
    ["核心思想", "Supervisor 调度 Worker 分工", "Agent 把任务转交给专家"],
    ["触发方式", "Supervisor 主动分派", "当前 Agent 判断超范围时被动触发"],
    ["控制流", "Supervisor 始终主导", "控制权完全转移（A → B）"],
    ["上下文", "分派时传递 subtask（精简）", "传递完整对话历史（完整）"],
    ["角色关系", "Supervisor + 多个平行 Worker", "通用 Agent + 专用专家"],
    ["适合场景", "任务可预先分解", "任务进行中发现需专家"],
    ["类比", "项目经理分派任务给组员", "客服把电话转给技术支持"],
    ["失败模式", "分派错误 Worker / Worker 失败", "无限踢皮球（反模式5）"],
  ];

  const header =
    `维度           │ Supervisor-Worker            │ Agent Handoff`;
  console.log(`OUT:compare: ${header}`);
  console.log(
    `OUT:compare: ${"─".repeat(12)}─┼─${"─".repeat(28)}─┼─${"─".repeat(28)}`,
  );
  for (const [dim, sw, ho] of comparisons) {
    const row =
      `${dim.padEnd(12)} │ ${sw.padEnd(28)} │ ${ho.padEnd(28)}`;
    console.log(`OUT:compare: ${row}`);
  }

  console.log(`\nOUT:compare: 核心洞察：`);
  console.log(`OUT:compare: • Supervisor-Worker = 团队分工（适合可分解的复杂任务）`);
  console.log(`OUT:compare: • Handoff = 专家路由（适合进行中发现的专业问题）`);
  console.log(`OUT:compare: • 两者可组合：Supervisor 分派时，某 Worker 内部可触发 Handoff`);
  console.log(`OUT:compare: • 记住：简单任务别上多 Agent（Anthropic 共识）`);
}

// ════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] 模式 1: Supervisor-Worker（调度-执行）`);
  console.log(`[config] 模式 2: Agent Handoff（任务转交）`);
  console.log(`[config] 输出标记: OUT:supervisor:, OUT:worker:{name}:, OUT:handoff:, OUT:resolve:`);

  let apiOk = true;
  const swTask = "调研 AI Agent 并写成报告";
  const handoffConversation: ConversationTurn[] = [
    { speaker: "user", message: "你好，我想退货，订单号 ORD-2024-001" },
    {
      speaker: "user",
      message: "退货页面报了 500 错误，错误代码 ERR_DEPLOY_123，好像是部署问题",
    },
  ];

  try {
    // ── Demo 1: Supervisor-Worker（真实 API）──
    console.log(`\n${"#".repeat(60)}`);
    console.log("# Demo 1: Supervisor-Worker（分解→分派→执行→汇总）");
    console.log("#".repeat(60));
    await supervisorWorkerFlow(swTask);

    // ── Demo 2: Agent Handoff（真实 API）──
    console.log(`\n${"#".repeat(60)}`);
    console.log("# Demo 2: Agent Handoff（客服 → 技术专家）");
    console.log("#".repeat(60));
    await handoffFlow(handoffConversation);
  } catch (err) {
    apiOk = false;
    const errorMsg = String(err);
    const isAuthError =
      errorMsg.includes("401") ||
      errorMsg.includes("invalid_api_key") ||
      errorMsg.includes("Authentication") ||
      errorMsg.includes("sk-REPLACE-ME");

    console.log(
      `\n[提示] 真实 API 调用失败（${err instanceof Error ? err.constructor.name : "Error"}）。`,
    );
    if (isAuthError) {
      console.log(`[提示] 原因：API 密钥无效或为占位符。请编辑 ai-agent/.env 填入有效密钥。`);
      console.log(`[提示] 当前 provider=${cfg.provider}，需要对应的 API 密钥。`);
    } else {
      console.log(`[提示] 原因：${err}`);
    }
    console.log(`[提示] 已自动降级为离线 mock 演示，多 Agent 逻辑不受影响。\n`);
  }

  // ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）────
  demoSupervisorWorkerOffline();
  demoHandoffOffline();
  demoComparison();

  console.log(`\n${"=".repeat(60)}`);
  if (apiOk) {
    console.log("所有演示完成！（含真实 API + 离线 mock + 对比）");
  } else {
    console.log("离线演示完成！（真实 API 未配置，但多 Agent 逻辑已完整展示）");
  }
  console.log(`💡 核心要点：复杂任务用 Supervisor-Worker 分工，专业问题用 Handoff 转交。`);
  console.log(`💡 但记住 Anthropic 共识：从最简单的方案开始，许多场景只需优化单次 LLM 调用。`);
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
