/**
 * 第08章 规划模式（Plan-and-Execute、CoT、Reflection）
 *
 * 本章演示两种高级推理模式，让 Agent 能处理复杂的多阶段任务：
 *
 *   Plan-and-Execute（规划-执行-汇总）：
 *     - Phase 1 (Plan)：用结构化输出（json_object + interface Plan）分解任务
 *     - Phase 2 (Execute)：逐步执行每个步骤（调工具/子查询），累积结果
 *     - Phase 3 (Synthesize)：汇总所有步骤结果，生成最终输出
 *     - 适合：步骤明确的复杂任务（写报告、制定计划、调研）
 *
 *   Reflection（反思 / 自我批评）：
 *     - Round 1：Agent 生成初版答案
 *     - Round 2：同一 Agent 审视初版，指出不足（完整性/准确性/结构）
 *     - Round 3：Agent 根据反思改进答案
 *     - 适合：质量敏感的生成任务（写作、代码、重要决策）
 *
 * 离线 mock 设计：
 *   .env 的 OPENAI_API_KEY=sk-REPLACE-ME 是占位符，真实 API 调用必失败。
 *   所有功能先 try 真实 API（失败时降级），然后跑离线 mock，保证 exit 0。
 *   Plan-and-Execute mock：预设步骤列表 + mock 执行结果 + 汇总。
 *   Reflection mock：预设"初版→反思→改进版"文本轨迹。
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 结构化输出：规划阶段的类型（用 interface + JSON.parse 解析）
// ════════════════════════════════════════════════════════════════════

/** 任务分解计划。steps 是有序的可执行步骤列表。 */
interface Plan {
  steps: string[];
}

/** 一步的执行结果（步骤描述 + 结果文本）。 */
interface StepResult {
  step: string;
  result: string;
}

// ════════════════════════════════════════════════════════════════════
// mock 工具：执行阶段的子查询（与第03/07章风格一致）
// ════════════════════════════════════════════════════════════════════

function mockSearch(query: string): string {
  /**模拟知识检索（mock 知识库）。*/
  const knowledge: Record<string, string> = {
    定义: "AI Agent 是能感知环境、自主决策、采取行动以实现目标的智能系统。",
    应用: "AI Agent 应用于智能客服、编程助手、自动化研究、数据分析等场景。",
    框架: "主流 AI Agent 框架有 LangChain、AutoGPT、OpenAI Agents SDK、CrewAI 等。",
    趋势: "AI Agent 正向多 Agent 协作、长程任务自主执行、工具自学习方向发展。",
    挑战: "AI Agent 面临可靠性、成本控制、安全对齐、评估困难等挑战。",
  };
  const queryLower = query.toLowerCase();
  for (const [key, value] of Object.entries(knowledge)) {
    if (queryLower.includes(key) || query.includes(key)) {
      return value;
    }
  }
  return `检索到与'${query}'相关的通用信息。`;
}

// ════════════════════════════════════════════════════════════════════
// Plan-and-Execute 模式
// ════════════════════════════════════════════════════════════════════

const PLAN_SYSTEM_PROMPT = `\
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
`;

async function planTask(task: string): Promise<Plan> {
  /** Phase 1: 用结构化输出生成任务计划。
   * 用 response_format=json_object 强制 JSON，再用 JSON.parse 解析。*/
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [
      { role: "system", content: PLAN_SYSTEM_PROMPT },
      { role: "user", content: task },
    ],
    response_format: { type: "json_object" },
  });
  const raw = response.choices[0].message.content ?? "{}";
  const parsed = JSON.parse(raw) as Plan;
  // 基本校验：steps 必须是数组
  if (!Array.isArray(parsed.steps)) {
    throw new Error(`Invalid plan: steps is not an array (got ${typeof parsed.steps})`);
  }
  return parsed;
}

async function executeStep(step: string, _stepIndex: number): Promise<string> {
  /** Phase 2: 执行单个步骤。
   * 教学示例：用 mockSearch 模拟检索/工具调用。
   * 真实场景：这里可以调用任意工具、子 LLM、或外部 API。*/
  return mockSearch(step);
}

async function synthesize(
  task: string,
  stepsAndResults: StepResult[],
): Promise<string> {
  /** Phase 3: 汇总所有步骤结果，生成最终输出。*/
  const context = stepsAndResults
    .map(
      (sr, i) =>
        `步骤${i + 1}: ${sr.step}\n结果: ${sr.result}`,
    )
    .join("\n");
  const prompt =
    `用户任务：${task}\n\n` +
    `已完成以下步骤：\n${context}\n\n` +
    `请基于以上信息，完成用户的原始任务。输出一份简洁的综合报告。`;
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [{ role: "user", content: prompt }],
  });
  return response.choices[0].message.content ?? "(空)";
}

async function planAndExecute(task: string): Promise<string> {
  /** Plan-and-Execute 完整三阶段流程（真实 API）。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log(`Plan-and-Execute 任务: ${task}`);
  console.log("=".repeat(60));

  // Phase 1: Plan
  console.log("\n--- Phase 1: Plan（规划）---");
  const plan = await planTask(task);
  console.log(`OUT:plan: 分解出 ${plan.steps.length} 个步骤:`);
  plan.steps.forEach((step, i) => {
    console.log(`OUT:plan:step${i + 1}: ${step}`);
  });

  // Phase 2: Execute
  console.log("\n--- Phase 2: Execute（执行）---");
  const stepsAndResults: StepResult[] = [];
  for (let i = 0; i < plan.steps.length; i++) {
    const step = plan.steps[i];
    console.log(`OUT:execute:step${i + 1}: 执行: ${step}`);
    const result = await executeStep(step, i);
    stepsAndResults.push({ step, result });
    console.log(`OUT:execute:step${i + 1}: 结果: ${result.slice(0, 80)}`);
  }

  // Phase 3: Synthesize
  console.log("\n--- Phase 3: Synthesize（汇总）---");
  const final = await synthesize(task, stepsAndResults);
  console.log(`OUT:synthesize: 最终输出（前 200 字）:`);
  console.log(`OUT:synthesize: ${final.slice(0, 200)}`);
  return final;
}

// ════════════════════════════════════════════════════════════════════
// Reflection 模式（自我批评 + 改进）
// ════════════════════════════════════════════════════════════════════

const REFLECTION_PROMPT = `\
你是一个严格的审稿人。请审视以下初版答案，指出它的不足。

用户问题：{question}
初版答案：{draft}

请从以下维度批评：
1. 完整性：有没有遗漏重要信息？
2. 准确性：有没有事实错误或逻辑漏洞？
3. 结构：组织是否清晰？

只指出不足（2-3 点），不要给出完整改进版。用简洁的要点格式。
`;

const REVISE_PROMPT = `\
用户问题：{question}
初版答案：{draft}
审稿意见：{critique}

请根据审稿意见改进初版答案，输出最终版。保持简洁。
`;

async function generateDraft(question: string): Promise<string> {
  /** Round 1: 生成初版答案。*/
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [{ role: "user", content: question }],
  });
  return response.choices[0].message.content ?? "(空)";
}

async function reflect(question: string, draft: string): Promise<string> {
  /** Round 2: 反思 / 自我批评。*/
  const prompt = REFLECTION_PROMPT.replace("{question}", question).replace(
    "{draft}",
    draft,
  );
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [{ role: "user", content: prompt }],
  });
  return response.choices[0].message.content ?? "(空)";
}

async function revise(
  question: string,
  draft: string,
  critique: string,
): Promise<string> {
  /** Round 3: 根据反思改进答案。*/
  const prompt = REVISE_PROMPT.replace("{question}", question)
    .replace("{draft}", draft)
    .replace("{critique}", critique);
  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [{ role: "user", content: prompt }],
  });
  return response.choices[0].message.content ?? "(空)";
}

async function reflectionFlow(question: string): Promise<string> {
  /** Reflection 完整三轮流程（真实 API）。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log(`Reflection 任务: ${question}`);
  console.log("=".repeat(60));

  // Round 1: Draft
  console.log("\n--- Round 1: Draft（生成初版）---");
  const draft = await generateDraft(question);
  console.log(`OUT:reflect:draft: 初版（前 200 字）:`);
  console.log(`OUT:reflect:draft: ${draft.slice(0, 200)}`);

  // Round 2: Critique
  console.log("\n--- Round 2: Critique（反思）---");
  const critique = await reflect(question, draft);
  console.log(`OUT:reflect:critique: 审稿意见:`);
  console.log(`OUT:reflect:critique: ${critique.slice(0, 200)}`);

  // Round 3: Revise
  console.log("\n--- Round 3: Revise（改进）---");
  const revised = await revise(question, draft, critique);
  console.log(`OUT:reflect:revised: 改进版（前 200 字）:`);
  console.log(`OUT:reflect:revised: ${revised.slice(0, 200)}`);

  return revised;
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：Plan-and-Execute（API 不可用时演示完整三阶段）
// ════════════════════════════════════════════════════════════════════

function demoPlanAndExecuteOffline(): string {
  /** 离线演示 Plan-and-Execute：预设步骤 + mock 执行 + 汇总。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("离线演示：Plan-and-Execute（规划→执行→汇总）");
  console.log("=".repeat(60));
  console.log("[说明] 预设计划 + mock 执行，演示完整三阶段流程");

  const task = "写一篇 AI Agent 调研报告";
  console.log(`任务: ${task}\n`);

  // ── Phase 1: Plan（预设计划，模拟 planTask() 的输出）──
  console.log("--- Phase 1: Plan（规划）---");
  const mockPlan: Plan = {
    steps: [
      "检索 AI Agent 的定义与核心特征",
      "检索 AI Agent 的典型应用场景",
      "检索主流 AI Agent 开发框架",
      "检索 AI Agent 的发展趋势与挑战",
      "综合以上信息撰写调研报告",
    ],
  };
  console.log(`OUT:plan: 分解出 ${mockPlan.steps.length} 个步骤:`);
  mockPlan.steps.forEach((step, i) => {
    console.log(`OUT:plan:step${i + 1}: ${step}`);
  });

  // ── Phase 2: Execute（用 mockSearch 真实执行检索步骤）──
  console.log("\n--- Phase 2: Execute（执行）---");
  const stepsAndResults: StepResult[] = [];
  for (let i = 0; i < mockPlan.steps.length; i++) {
    const step = mockPlan.steps[i];
    console.log(`OUT:execute:step${i + 1}: 执行: ${step}`);
    let result: string;
    if (i < mockPlan.steps.length - 1) {
      // 前 4 步是检索，用 mockSearch 执行
      result = executeStepSync(step, i);
    } else {
      // 最后一步是汇总，留到 Phase 3
      result = "(汇总步骤，在 Phase 3 执行)";
    }
    stepsAndResults.push({ step, result });
    console.log(`OUT:execute:step${i + 1}: 结果: ${result.slice(0, 80)}`);
  }

  // ── Phase 3: Synthesize（汇总检索结果，生成报告）──
  console.log("\n--- Phase 3: Synthesize（汇总）---");
  const searchResults = stepsAndResults.slice(0, -1); // 排除最后的汇总步骤
  const reportLines: string[] = [`# ${task}\n`];
  searchResults.forEach((sr, i) => {
    const topic = sr.step
      .replace("检索 AI Agent 的", "")
      .replace("的", "");
    reportLines.push(`## ${i + 1}. ${topic}`);
    reportLines.push(sr.result);
    reportLines.push("");
  });
  const finalReport = reportLines.join("\n");

  console.log(`OUT:synthesize: 最终输出（前 300 字）:`);
  console.log(`OUT:synthesize: ${finalReport.slice(0, 300)}`);
  console.log(
    `\nOUT:synthesize: ✓ Plan-and-Execute 完成，共 ${mockPlan.steps.length} 个步骤。`,
  );
  console.log(
    `OUT:synthesize: 规划阶段一次性看清全貌，执行阶段机械检索，汇总阶段综合输出。`,
  );
  return finalReport;
}

// 同步版 executeStep（离线 mock 用，mockSearch 本身是同步的）
function executeStepSync(step: string, _stepIndex: number): string {
  return mockSearch(step);
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：Reflection（API 不可用时演示三轮反思改进）
// ════════════════════════════════════════════════════════════════════

function demoReflectionOffline(): string {
  /** 离线演示 Reflection：预设"初版→反思→改进版"轨迹。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("离线演示：Reflection（初版→反思→改进版）");
  console.log("=".repeat(60));
  console.log("[说明] 预设三轮轨迹，演示反思如何改进输出质量");

  const question = "什么是 AI Agent？请简要解释。";
  console.log(`问题: ${question}\n`);

  // ── Round 1: Draft（预设一个"简陋"的初版）──
  console.log("--- Round 1: Draft（生成初版）---");
  const mockDraft =
    "AI Agent 是一种人工智能系统，能自主完成任务。" +
    "它可以调用工具、理解指令。";
  console.log(`OUT:reflect:draft: 初版:`);
  console.log(`OUT:reflect:draft: ${mockDraft}`);
  console.log(`OUT:reflect:draft: [问题：太简略，缺少关键维度]`);

  // ── Round 2: Critique（预设反思，指出 2-3 个不足）──
  console.log("\n--- Round 2: Critique（反思）---");
  const mockCritique =
    "初版存在以下不足：\n" +
    "1. 完整性：只提了'能自主完成任务'，没解释 Agent 的核心组成" +
    "（感知、决策、行动三要素）。\n" +
    "2. 完整性：没有区分 Agent 和普通 LLM 的本质区别（工具调用、循环、自主性）。\n" +
    "3. 结构：缺乏层次，信息密度低。";
  console.log(`OUT:reflect:critique: 审稿意见:`);
  console.log(`OUT:reflect:critique: ${mockCritique}`);

  // ── Round 3: Revise（预设改进版，体现反思的改进）──
  console.log("\n--- Round 3: Revise（改进）---");
  const mockRevised =
    "AI Agent 是能**感知环境、自主决策、采取行动**以实现目标的智能系统。" +
    "它的三个核心要素：\n" +
    "1. **感知**：接收用户输入或环境信号（如读取消息、监控数据）。\n" +
    "2. **决策**：通过 LLM 推理决定下一步（如 ReAct 的 Thought）。\n" +
    "3. **行动**：调用工具执行操作（如 function calling）。\n\n" +
    "与普通 LLM 的区别：Agent 有**循环**（能多步执行）、**工具**（能调用外部能力）、" +
    "**自主性**（能自己决定何时停止）。";
  console.log(`OUT:reflect:revised: 改进版:`);
  console.log(`OUT:reflect:revised: ${mockRevised}`);

  // ── 对比展示反思的价值 ──
  console.log(`\n--- 反思价值对比 ---`);
  console.log(
    `OUT:reflect: 初版字数: ${mockDraft.length} | 改进版字数: ${mockRevised.length}`,
  );
  console.log(
    `OUT:reflect: 改进点：补充了三要素、与 LLM 的区别、结构化呈现`,
  );
  console.log(
    `\nOUT:reflect: ✓ Reflection 完成，三轮流程（Draft→Critique→Revise）。`,
  );
  console.log(
    `OUT:reflect: 反思让答案从'简陋'变'完善'——代价是 3 次 LLM 调用的延迟。`,
  );
  return mockRevised;
}

// ════════════════════════════════════════════════════════════════════
// 模式对比输出：Plan-and-Execute vs Reflection vs ReAct
// ════════════════════════════════════════════════════════════════════

function demoComparison(): void {
  /** 并排对比三种推理模式的核心差异。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("对比：ReAct vs Plan-and-Execute vs Reflection");
  console.log("=".repeat(60));

  const comparisons: Array<[string, string, string, string]> = [
    ["核心思想", "边想边做", "先规划再执行", "做完反思改进"],
    ["流程", "Thought→Action 循环", "Plan→Execute→Synth", "Draft→Critique→Revise"],
    ["决策时机", "每步动态决策", "规划阶段一次决策", "执行后回顾决策"],
    ["适合任务", "步骤未知/需探索", "步骤明确/多阶段", "质量敏感的生成"],
    ["LLM 调用次数", "N 步 = N 次", "1 + N + 1 次", "3 次（固定）"],
    ["延迟", "中（取决于步数）", "中高（规划+执行）", "高（3 次调用）"],
    ["可预测性", "低（路径不确定）", "高（计划可审查）", "中"],
    ["可并行", "难（步骤间依赖）", "易（独立步骤并发）", "不适用"],
  ];

  const header =
    `维度           │ ReAct                │ Plan-and-Execute     │ Reflection`;
  console.log(`OUT:compare: ${header}`);
  console.log(
    `OUT:compare: ${"─".repeat(12)}─┼─${"─".repeat(20)}─┼─${"─".repeat(20)}─┼─${"─".repeat(20)}`,
  );
  for (const [dim, react, plan, reflect] of comparisons) {
    const row =
      `${dim.padEnd(12)} │ ${react.padEnd(20)} │ ${plan.padEnd(20)} │ ${reflect}`;
    console.log(`OUT:compare: ${row}`);
  }

  console.log(`\nOUT:compare: 核心洞察：`);
  console.log(`OUT:compare: • ReAct = 灵活探索（适合步骤未知的任务）`);
  console.log(`OUT:compare: • Plan-and-Execute = 战略规划（适合多阶段复杂任务）`);
  console.log(`OUT:compare: • Reflection = 质量打磨（适合对输出质量要求高的场景）`);
  console.log(`OUT:compare: • 三者可组合：Plan→Execute（ReAct 执行每步）→Reflect（打磨汇总）`);
}

// ════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] 模式 1: Plan-and-Execute（规划→执行→汇总）`);
  console.log(`[config] 模式 2: Reflection（初版→反思→改进）`);
  console.log(`[config] 输出标记: OUT:plan:, OUT:execute:step{N}:, OUT:synthesize:, OUT:reflect:`);

  let apiOk = true;
  const planTaskInput = "写一篇 AI Agent 调研报告";
  const reflectQuestion = "什么是 AI Agent？请简要解释。";

  try {
    // ── Demo 1: Plan-and-Execute（真实 API）──
    console.log(`\n${"#".repeat(60)}`);
    console.log("# Demo 1: Plan-and-Execute（规划→执行→汇总）");
    console.log("#".repeat(60));
    await planAndExecute(planTaskInput);

    // ── Demo 2: Reflection（真实 API）──
    console.log(`\n${"#".repeat(60)}`);
    console.log("# Demo 2: Reflection（初版→反思→改进）");
    console.log("#".repeat(60));
    await reflectionFlow(reflectQuestion);
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
    console.log(`[提示] 已自动降级为离线 mock 演示，规划逻辑不受影响。\n`);
  }

  // ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）────
  demoPlanAndExecuteOffline();
  demoReflectionOffline();
  demoComparison();

  console.log(`\n${"=".repeat(60)}`);
  if (apiOk) {
    console.log("所有演示完成！（含真实 API + 离线 mock + 对比）");
  } else {
    console.log("离线演示完成！（真实 API 未配置，但规划逻辑已完整展示）");
  }
  console.log(`💡 核心要点：复杂任务先规划（减少返工），质量任务加反思（提升质量）。`);
  console.log(`💡 简单任务别过度规划——会增加延迟和成本。`);
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
