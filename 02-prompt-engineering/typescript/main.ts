/**
 * 第02章 Prompt 工程 — 4 种核心技术演示（TypeScript 版）。
 *
 * 围绕「任务助手 Agent」展开：
 * 1. 无 system vs 有 system — 同一问题，截然不同的回答
 * 2. Few-shot 分类 — 用示例教模型做情感分类
 * 3. Chain-of-Thought — 引导模型逐步推理
 * 4. 结构化输出 — response_format + Zod 解析
 *
 * 运行：cd 02-prompt-engineering/typescript && npm install && npx tsx main.ts
 */

import OpenAI from "openai";
import { z } from "zod";
import { getConfig } from "../../shared/config";

// ── 初始化客户端 ───────────────────────────────────────────────────
const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

const SEPARATOR = "=".repeat(60);

// ── Zod Schema：用于场景 4 的结构化解析 ───────────────────────────
const TaskSchema = z.object({
  title: z.string(),
  priority: z.enum(["high", "medium", "low"]),
  description: z.string(),
});

type TaskInfo = z.infer<typeof TaskSchema>;

// ── 离线 mock 数据 ─────────────────────────────────────────────────
const MOCK_NO_SYS = "建议你先列出会议议程，准备相关材料，然后预留时间写报告。";
const MOCK_WITH_SYS =
  '{"title": "准备明天会议", "priority": "high", "description": "明天要开会，需要提前准备议程和材料"}';
const MOCK_FEW_SHOT = ["正面", "负面", "中性"];
const MOCK_COT_DIRECT = "C（今天下午截止）→ A（明天截止）→ B（下周截止）";
const MOCK_COT_REASONING =
  "让我逐一分析：\n" +
  "1. 任务 C：截止今天下午，预计 4 小时。今天有 4 小时可用，刚好够完成，必须第一个做。\n" +
  "2. 任务 A：截止明天，预计 2 小时。今天时间已被 C 占满，但明天还有时间，优先级第二。\n" +
  "3. 任务 B：截止下周，预计 30 分钟。时间最充裕，可以最后做。\n\n" +
  "排序：C → A → B";
const MOCK_STRUCTURED_JSON =
  '{"title": "提交项目报告", "priority": "medium", "description": "下周三之前完成，需要整理数据和写总结"}';

// ═══════════════════════════════════════════════════════════════════
// 场景 1：无 System Prompt vs 有 System Prompt
// ═══════════════════════════════════════════════════════════════════
async function demoSystemPrompt(): Promise<void> {
  const userInput = "我明天要开会，帮我准备一下";

  // ── 1a. 无 system prompt ───────────────────────────────────────
  console.log(`\n${SEPARATOR}`);
  console.log("场景 1：无 System Prompt");
  console.log(SEPARATOR);
  let answerNoSys: string;
  try {
    const respNoSys = await client.chat.completions.create({
      model: cfg.model,
      messages: [{ role: "user", content: userInput }],
    });
    answerNoSys = respNoSys.choices[0].message.content ?? "";
  } catch {
    console.log("OUT: [提示] API 不可用，使用离线 mock 演示");
    answerNoSys = MOCK_NO_SYS;
  }
  console.log(`OUT: ${answerNoSys.slice(0, 200)}`);

  // ── 1b. 有 system prompt（任务助手人格）────────────────────────
  console.log(`\n${SEPARATOR}`);
  console.log("场景 1：有 System Prompt（任务助手）");
  console.log(SEPARATOR);
  const systemPrompt =
    '你是任务管理助手。用户提到任何事项，你都要提取为结构化任务。' +
    '返回 JSON 格式：{"title": "任务标题", "priority": "high|medium|low", "description": "任务描述"}。' +
    "优先级规则：紧急=high，重要=medium，其他=low。只返回 JSON，不要其他文字。";

  let answerWithSys: string;
  try {
    const respWithSys = await client.chat.completions.create({
      model: cfg.model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userInput },
      ],
      response_format: { type: "json_object" },
    });
    answerWithSys = respWithSys.choices[0].message.content ?? "";
  } catch {
    console.log("OUT: [提示] API 不可用，使用离线 mock 演示");
    answerWithSys = MOCK_WITH_SYS;
  }
  console.log(`OUT: ${answerWithSys}`);
}

// ═══════════════════════════════════════════════════════════════════
// 场景 2：Few-shot 情感分类
// ═══════════════════════════════════════════════════════════════════
async function demoFewShot(): Promise<void> {
  console.log(`\n${SEPARATOR}`);
  console.log("场景 2：Few-shot 情感分类");
  console.log(SEPARATOR);

  const fewShotTemplate = `你是一个情感分类器。根据用户输入判断情感倾向，只回答分类结果。

示例 1：
输入：这家餐厅的菜太好吃了，下次还来！
分类：正面

示例 2：
输入：等了一个小时才上菜，服务态度还很差。
分类：负面

示例 3：
输入：餐厅在商场三楼，营业到晚上10点。
分类：中性

现在请分类：
输入：{user_input}
分类：`;

  const testInputs = [
    "这个产品用起来太顺手了，强烈推荐！",
    "包装破损，客服还推卸责任。",
    "商品重量约 500 克，保质期 12 个月。",
  ];

  let apiFailed = false;
  for (let i = 0; i < testInputs.length; i++) {
    const inp = testInputs[i];
    const prompt = fewShotTemplate.replace("{user_input}", inp);
    let label: string;
    try {
      const resp = await client.chat.completions.create({
        model: cfg.model,
        messages: [{ role: "user", content: prompt }],
      });
      label = (resp.choices[0].message.content ?? "").trim();
    } catch {
      if (!apiFailed) {
        console.log("OUT: [提示] API 不可用，使用离线 mock 演示");
        apiFailed = true;
      }
      label = MOCK_FEW_SHOT[i];
    }
    console.log(`OUT: 输入：${inp}`);
    console.log(`OUT: 分类：${label}`);
    console.log();
  }
}

// ═══════════════════════════════════════════════════════════════════
// 场景 3：Chain-of-Thought 推理
// ═══════════════════════════════════════════════════════════════════
async function demoChainOfThought(): Promise<void> {
  console.log(`\n${SEPARATOR}`);
  console.log("场景 3：Chain-of-Thought 推理");
  console.log(SEPARATOR);

  const question =
    "一个任务助手需要处理以下优先级排序问题：\n" +
    "有 3 个任务：A（截止明天，预计 2 小时），B（截止下周，预计 30 分钟），" +
    "C（截止今天下午，预计 4 小时）。\n" +
    "如果今天是上午，且每天只有 4 小时处理任务，应该如何排序？";

  // ── 直接回答 ───────────────────────────────────────────────────
  console.log("\n--- 直接回答（不引导 CoT）---");
  let directAnswer: string;
  try {
    const respDirect = await client.chat.completions.create({
      model: cfg.model,
      messages: [{ role: "user", content: question + "\n请直接给出排序结果。" }],
    });
    directAnswer = (respDirect.choices[0].message.content ?? "").slice(0, 300);
  } catch {
    console.log("OUT: [提示] API 不可用，使用离线 mock 演示");
    directAnswer = MOCK_COT_DIRECT;
  }
  console.log(`OUT: ${directAnswer}`);

  // ── CoT 引导 ───────────────────────────────────────────────────
  console.log("\n--- CoT 引导（请一步一步思考）---");
  let cotAnswer: string;
  try {
    const respCot = await client.chat.completions.create({
      model: cfg.model,
      messages: [
        {
          role: "user",
          content:
            question +
            "\n请一步一步思考，分析每个任务的紧急程度和所需时间，然后给出排序。",
        },
      ],
    });
    cotAnswer = (respCot.choices[0].message.content ?? "").slice(0, 500);
  } catch {
    console.log("OUT: [提示] API 不可用，使用离线 mock 演示");
    cotAnswer = MOCK_COT_REASONING;
  }
  console.log(`OUT: ${cotAnswer}`);
}

// ═══════════════════════════════════════════════════════════════════
// 场景 4：结构化输出（response_format + Zod）
// ═══════════════════════════════════════════════════════════════════
async function demoStructuredOutput(): Promise<void> {
  console.log(`\n${SEPARATOR}`);
  console.log("场景 4：结构化输出（response_format + Zod）");
  console.log(SEPARATOR);

  const systemPrompt =
    "你是任务管理助手。用户会描述一个任务，你需要提取为 JSON。\n" +
    'JSON 格式：{"title": "任务标题", "priority": "high|medium|low", "description": "任务描述"}\n' +
    "优先级规则：截止时间紧迫=high，重要但不急=medium，其他=low。\n" +
    "只返回 JSON，不要其他文字。";

  const userInput = "下周三之前要提交项目报告，需要整理数据和写总结";

  let rawJson: string;
  try {
    const resp = await client.chat.completions.create({
      model: cfg.model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userInput },
      ],
      response_format: { type: "json_object" },
    });
    rawJson = resp.choices[0].message.content ?? "{}";
  } catch {
    console.log("OUT: [提示] API 不可用，使用离线 mock 演示");
    rawJson = MOCK_STRUCTURED_JSON;
  }

  console.log(`OUT: 原始 JSON：${rawJson}`);

  // Zod 解析 + 校验
  const task: TaskInfo = TaskSchema.parse(JSON.parse(rawJson));
  console.log(`OUT: title=${task.title}, priority=${task.priority}`);
  console.log(`OUT: description=${task.description}`);

  // 验证类型安全
  console.assert(
    ["high", "medium", "low"].includes(task.priority),
    `未知优先级: ${task.priority}`,
  );
  console.log("OUT: Zod 解析成功，类型校验通过！");
}

// ═══════════════════════════════════════════════════════════════════
// 主入口
// ═══════════════════════════════════════════════════════════════════
async function main(): Promise<void> {
  console.log("第02章 Prompt 工程 — 4 种核心技术演示（TypeScript）");
  console.log(`提供商: ${cfg.provider} | 模型: ${cfg.model}`);

  await demoSystemPrompt();
  await demoFewShot();
  await demoChainOfThought();
  await demoStructuredOutput();

  console.log(`\n${SEPARATOR}`);
  console.log("OUT: 全部场景演示完成！");
  console.log(SEPARATOR);
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
