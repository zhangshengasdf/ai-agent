/**
 * 第01章 · LLM 基础 —— 与模型对话的第一步
 *
 * 演示内容：
 * 1. 单轮对话 —— 用 system prompt 定义「任务助手 Agent」人格
 * 2. Token 用量 —— 打印 prompt_tokens / completion_tokens
 * 3. 温度对比 —— 同一问题用 temperature=0.0 和 1.0 各调一次
 * 4. 流式输出 —— 逐 token 打印模型回答
 *
 * 所有输出以 "OUT:" 前缀标记，便于 QA 脚本过滤 tsx 的 dotenvx 横幅。
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";

// ── 初始化客户端 ────────────────────────────────────────────────────
const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ── 任务助手 Agent 的 system prompt ─────────────────────────────────
const SYSTEM_PROMPT =
  "你是一个「任务助手 Agent」——一个简洁、高效的任务管理助手。" +
  "用户会问你关于待办、日程、任务优先级的问题。" +
  "回答要简明扼要，直接给出建议，不要废话。" +
  "如果用户的问题与任务管理无关，简短回答后提醒他你只擅长任务管理。";

const USER_MESSAGE =
  "我今天有三个会要开，还有一个报告要写，怎么安排优先级？";

// ═══════════════════════════════════════════════════════════════════
// Demo 1: 单轮对话 + Demo 2: Token 用量
// ═══════════════════════════════════════════════════════════════════
async function demoSingleTurn(): Promise<void> {
  console.log("=".repeat(60));
  console.log("OUT: [Demo 1] 单轮对话 —— 任务助手 Agent");
  console.log("=".repeat(60));

  const response = await client.chat.completions.create({
    model: cfg.model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: USER_MESSAGE },
    ],
  });

  const answer = response.choices[0].message.content;
  console.log(`OUT: \n[用户] ${USER_MESSAGE}`);
  console.log(`OUT: \n[任务助手] ${answer}`);

  // ── Token 用量 ────────────────────────────────────────────────
  console.log("\n" + "=".repeat(60));
  console.log("OUT: [Demo 2] Token 用量");
  console.log("=".repeat(60));
  const usage = response.usage!;
  console.log(`OUT: prompt_tokens     = ${usage.prompt_tokens}`);
  console.log(`OUT: completion_tokens = ${usage.completion_tokens}`);
  console.log(`OUT: total_tokens      = ${usage.total_tokens}`);
}

// ═══════════════════════════════════════════════════════════════════
// Demo 3: 温度对比
// ═══════════════════════════════════════════════════════════════════
async function demoTemperatureComparison(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("OUT: [Demo 3] 温度对比 —— 同一问题，不同温度");
  console.log("=".repeat(60));

  const question = "用一句话解释什么是 Agent。";

  for (const temp of [0.0, 1.0]) {
    const response = await client.chat.completions.create({
      model: cfg.model,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: question },
      ],
      temperature: temp,
    });
    const answer = response.choices[0].message.content;
    console.log(`OUT: \n[temperature=${temp}] ${answer}`);
  }
}

// ═══════════════════════════════════════════════════════════════════
// Demo 4: 流式输出
// ═══════════════════════════════════════════════════════════════════
async function demoStreaming(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("OUT: [Demo 4] 流式输出 —— 逐 token 打印");
  console.log("=".repeat(60));
  process.stdout.write("OUT: \n[任务助手] ");

  const stream = await client.chat.completions.create({
    model: cfg.model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: "流式输出的好处是什么？用两句话回答。" },
    ],
    stream: true,
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      process.stdout.write(content);
    }
  }

  console.log(); // 换行
}

// ═══════════════════════════════════════════════════════════════════
// 入口
// ═══════════════════════════════════════════════════════════════════
async function main(): Promise<void> {
  console.log(`OUT: 提供商: ${cfg.provider} | 模型: ${cfg.model}`);
  console.log();

  await demoSingleTurn(); // Demo 1 + 2
  await demoTemperatureComparison(); // Demo 3
  await demoStreaming(); // Demo 4

  console.log("\nOUT: ✅ 所有演示完成！");
}

main().catch((err) => {
  console.error("错误:", err.message);
  process.exit(1);
});
