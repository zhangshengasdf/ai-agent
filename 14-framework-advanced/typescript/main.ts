/**
 * 第14章 从零造框架 — 高级特性（TypeScript 版）
 *
 * 对等 Python 实现。4 个 demo：
 *   Demo 1: 流式输出（Streaming）—— async iterator 逐块消费
 *   Demo 2: 结构化输出强制（StructuredOutput）—— JSON.parse + 手动类型校验 + 失败重试
 *   Demo 3: 工具参数校验（ToolValidation）—— JSON Schema 类型/必填校验
 *   Demo 4: 现代框架对比（OpenAI Agents SDK / Mastra / Vercel AI SDK）
 *
 * 运行方式：
 *   cd ai-agent/14-framework-advanced
 *   npx tsx typescript/main.ts
 *
 * 设计：
 *   - 概念上引用第13章框架，独立可运行（不 import 第13章代码）
 *   - .env 占位符 sk-REPLACE-ME → 真实 API 必失败 → 降级 mock，exit 0
 *   - 全链路 async（TS SDK 只有异步接口，T5 教训）
 *   - 不用 `as any` / `@ts-ignore`；类型不匹配用特定断言或类型守卫
 */

import OpenAI, {
  APIConnectionError,
  APIError,
  AuthenticationError,
} from "openai";
import { getConfig } from "../../shared/config";

// ═══════════════════════════════════════════════════════════════════════
// 共享类型
// ═══════════════════════════════════════════════════════════════════════

/** 简单消息（流式 demo 用，与第13章 Message 结构兼容）。 */
interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/** Demo 2 的结构化输出目标类型。 */
interface TaskSummary {
  name: string;
  description: string;
  difficulty: "easy" | "medium" | "hard";
  priority: number; // 1-5
  estimated_hours: number;
}

/** 工具 schema（JSON Schema 子集）。 */
interface JsonSchemaProperty {
  type?: string;
  description?: string;
}

interface JsonSchema {
  type: "object";
  properties: Record<string, JsonSchemaProperty>;
  required?: string[];
}

type ToolHandler = (...args: string[]) => string;

interface ToolDef {
  schema: JsonSchema;
  handler: ToolHandler;
}

// ═══════════════════════════════════════════════════════════════════════
// 客户端初始化
// ═══════════════════════════════════════════════════════════════════════

function makeClient(): OpenAI {
  const cfg = getConfig();
  return new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 1: 流式输出（Streaming）
// ═══════════════════════════════════════════════════════════════════════

async function streamRealApi(
  client: OpenAI,
  model: string,
  prompt: string,
): Promise<string | null> {
  /** 尝试真实流式 API。失败返回 null（由上层降级 mock）。 */
  try {
    const stream = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: "你是任务助手 Agent。用 30 字内回答。" },
        { role: "user", content: prompt },
      ],
      stream: true,
    });

    let collected = "";
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) {
        process.stdout.write(`OUT:stream:chunk: ${delta}`);
        collected += delta;
      }
    }
    process.stdout.write("\n");
    return collected;
  } catch (err) {
    const errName = (err as Error).constructor.name;
    console.log(
      `OUT:stream:offline: 真实 API 不可用（${errName}），降级 mock 流式`,
    );
    return null;
  }
}

async function streamMock(prompt: string): Promise<string> {
  /** 离线 mock 流式：把回答切成字符块，逐块下发，模拟流式效果。 */
  const fullText =
    `收到任务：『${prompt}』。我是任务助手 Agent，` +
    "已经准备好帮你查询天气、做计算、规划任务。请告诉我具体需求。";

  // 按 2-3 字一组切片，模拟真实 API 的 token 块
  const chunks = fullText.match(/.{1,3}/gs) ?? [fullText];
  let collected = "";
  for (const chunk of chunks) {
    await sleep(50); // 演示用，让"逐块"可见
    process.stdout.write(`OUT:stream:chunk: ${chunk}`);
    collected += chunk;
  }
  process.stdout.write("\n");
  return collected;
}

async function demoStreaming(): Promise<void> {
  console.log("=".repeat(72));
  console.log("Demo 1: 流式输出（Streaming）");
  console.log("  把 LLM 响应逐 token 块下发，而不是一次性返回。");
  console.log("  价值：用户体验好（首字延迟低）、可中断、长文本不卡 UI。");
  console.log("=".repeat(72));
  console.log();

  const cfg = getConfig();
  const client = makeClient();
  const prompt = "你好，请简短介绍一下你能做什么。";

  console.log(`  用户输入: ${prompt}`);
  console.log("  流式输出（逐块到达）↓");
  console.log("-".repeat(72));

  const result = (await streamRealApi(client, cfg.model, prompt)) ??
    (await streamMock(prompt));

  console.log("-".repeat(72));
  console.log(`OUT:stream:done: 共收到 ${result.length} 字符（流式完成）`);
  console.log();
  console.log("  💡 在真实应用中，每个 chunk 可以直接 write 到前端 SSE/WebSocket，");
  console.log("     用户看到的是『打字机效果』，而不是『空白等待 → 一大段文字』。");
  console.log();
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 2: 结构化输出强制（StructuredOutput + 失败重试）
// ═══════════════════════════════════════════════════════════════════════

/**
 * 手动类型校验：把 unknown 收窄到 TaskSummary。
 * 失败时返回错误列表（空列表 = 通过）。
 *
 * 这是 TS 没有 Pydantic/Zod 时的"原生校验"方案——
 * 用 type guard + Array.isArray + typeof 逐字段检查。
 */
function validateTaskSummary(raw: unknown): {
  ok: true;
  value: TaskSummary;
} | { ok: false; errors: string[] } {
  const errors: string[] = [];
  if (typeof raw !== "object" || raw === null) {
    return { ok: false, errors: ["顶层不是对象"] };
  }
  const obj = raw as Record<string, unknown>;

  // name: string
  if (typeof obj.name !== "string") {
    errors.push("字段 'name' 期望 string");
  }
  // description: string
  if (typeof obj.description !== "string") {
    errors.push("字段 'description' 期望 string");
  }
  // difficulty: "easy" | "medium" | "hard"
  if (
    typeof obj.difficulty !== "string" ||
    !["easy", "medium", "hard"].includes(obj.difficulty)
  ) {
    errors.push("字段 'difficulty' 期望 'easy' | 'medium' | 'hard'");
  }
  // priority: number 1-5（注意排除 bool，因为 typeof true === "boolean" 但能通过 number 检查）
  if (typeof obj.priority !== "number" || !Number.isInteger(obj.priority)) {
    errors.push("字段 'priority' 期望整数");
  } else if (obj.priority < 1 || obj.priority > 5) {
    errors.push("字段 'priority' 应在 1-5 范围");
  }
  // estimated_hours: positive number
  if (typeof obj.estimated_hours !== "number" || obj.estimated_hours <= 0) {
    errors.push("字段 'estimated_hours' 期望正数");
  }

  if (errors.length > 0) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    value: {
      name: obj.name as string,
      description: obj.description as string,
      difficulty: obj.difficulty as "easy" | "medium" | "hard",
      priority: obj.priority as number,
      estimated_hours: obj.estimated_hours as number,
    },
  };
}

async function structuredRealApi(
  client: OpenAI,
  model: string,
  task: string,
  maxRetries = 3,
): Promise<TaskSummary | null> {
  /** 真实 API + 类型校验 + 失败重试。返回 null 表示 API 不可用。 */
  const systemPrompt =
    "你是任务分析助手。把用户的任务描述解析成结构化 JSON，包含字段：" +
    "name, description, difficulty (easy|medium|hard), priority (1-5 整数), " +
    "estimated_hours (正浮点数)。只返回 JSON，不要多余文字。";

  const messages: ChatMessage[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: `任务：${task}` },
  ];

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    console.log(`OUT:structured:attempt: 第 ${attempt}/${maxRetries} 次尝试...`);
    let raw: string;
    try {
      const resp = await client.chat.completions.create({
        model,
        messages: messages as OpenAI.ChatCompletionMessageParam[],
        response_format: { type: "json_object" },
      });
      raw = resp.choices[0].message.content ?? "{}";
    } catch (err) {
      if (
        err instanceof AuthenticationError ||
        err instanceof APIConnectionError ||
        err instanceof APIError
      ) {
        console.log(
          `OUT:structured:offline: API 不可用（${(err as Error).constructor.name}），降级 mock`,
        );
        return null;
      }
      throw err;
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = {};
    }
    const result = validateTaskSummary(parsed);
    if (result.ok) {
      console.log(
        `OUT:structured:result: ✓ 校验通过 → ${JSON.stringify(result.value)}`,
      );
      return result.value;
    }
    console.log(`OUT:structured:retry: ✗ 校验失败（第 ${attempt} 次）`);
    console.log(`  错误: ${result.errors.slice(0, 2).join("; ")}`);
    // 把错误反馈给 LLM（关键：让模型"看到"自己错在哪）
    messages.push({ role: "assistant", content: raw });
    messages.push({
      role: "user",
      content: `上次输出校验失败：${result.errors.join("; ")}。请严格按 schema 重新输出 JSON。`,
    });
  }

  console.log("OUT:structured:fail: 重试耗尽，仍无法通过校验。");
  return null;
}

async function structuredMock(task: string): Promise<TaskSummary> {
  /** 离线 mock：预设"一次失败 + 重试成功"的轨迹。 */
  console.log(`  （mock）分析任务：${task}`);
  const maxRetries = 3;
  // mock 序列：第 1 次故意缺字段，第 2 次完整
  const mockResponses = [
    // ❌ 缺 priority 和 estimated_hours
    JSON.stringify({
      name: "实现登录功能",
      description: "完成用户登录的 API 和前端表单",
      difficulty: "medium",
    }),
    // ✓ 完整合法
    JSON.stringify({
      name: "实现登录功能",
      description: "完成用户登录的 API（JWT 鉴权）和前端表单（含校验）",
      difficulty: "medium",
      priority: 4,
      estimated_hours: 8.0,
    }),
  ];

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    console.log(`OUT:structured:attempt: 第 ${attempt}/${maxRetries} 次尝试（mock）...`);
    await sleep(100);
    const raw = attempt === 1 ? mockResponses[0] : mockResponses[1];

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = {};
    }
    const result = validateTaskSummary(parsed);
    if (result.ok) {
      console.log(
        `OUT:structured:result: ✓ 校验通过 → ${JSON.stringify(result.value)}`,
      );
      return result.value;
    }
    console.log(`OUT:structured:retry: ✗ 校验失败（第 ${attempt} 次）`);
    console.log(`  缺失字段: ${result.errors.join("; ")}`);
  }
  throw new Error("mock 序列设计错误：应该在第 2 次成功");
}

async function demoStructuredOutput(): Promise<void> {
  console.log("=".repeat(72));
  console.log("Demo 2: 结构化输出强制（StructuredOutput）");
  console.log("  response_format=json_object + 类型校验 + 失败重试。");
  console.log("  价值：LLM 输出 100% 符合 schema，下游代码可直接使用。");
  console.log("=".repeat(72));
  console.log();

  const cfg = getConfig();
  const client = makeClient();
  const task = "实现用户登录功能（含 JWT 鉴权和前端表单校验）";

  console.log(`  任务: ${task}`);
  console.log(
    "  目标 schema: TaskSummary(name, description, difficulty, priority, estimated_hours)",
  );
  console.log("-".repeat(72));

  let result = await structuredRealApi(client, cfg.model, task);
  if (result === null) {
    console.log();
    result = await structuredMock(task);
  }

  console.log("-".repeat(72));
  console.log(
    `OUT:structured:final: ${result.name} | 难度=${result.difficulty} | 优先级=${result.priority} | 工时=${result.estimated_hours}h`,
  );
  console.log();
  console.log("  💡 校验失败重试的核心：把错误反馈给 LLM，让它『看到』自己错在哪。");
  console.log("     这比单纯报错丢给用户强 100 倍——LLM 通常一次就能修正。");
  console.log();
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 3: 工具参数校验（ToolValidation）
// ═══════════════════════════════════════════════════════════════════════

/**
 * 用 JSON Schema 校验工具参数。返回错误列表（空 = 通过）。
 * 简化实现：只校验 type + required（生产建议用 ajv 或 zod 做完整校验）。
 */
function validateToolArgs(
  args: Record<string, unknown>,
  schema: JsonSchema,
): string[] {
  const errors: string[] = [];

  // 1. 检查 required 字段
  const required = schema.required ?? [];
  for (const field of required) {
    if (!(field in args)) {
      errors.push(`缺少必填字段: '${field}'`);
    }
  }

  // 2. 检查每个提供的字段类型
  const jsonTypeMap: Record<string, string> = {
    string: "string",
    integer: "number",
    number: "number",
    boolean: "boolean",
    array: "object",
    object: "object",
  };
  for (const [field, value] of Object.entries(args)) {
    if (!(field in schema.properties)) continue;
    const expected = schema.properties[field].type;
    if (!expected) continue;

    // 特殊处理：typeof true === "boolean"，但 bool 不应通过 integer/number 校验
    if (typeof value === "boolean" && (expected === "integer" || expected === "number")) {
      errors.push(`字段 '${field}' 期望 ${expected}，实际 boolean`);
      continue;
    }
    // 特殊处理：integer 要求 Number.isInteger
    if (expected === "integer" && (typeof value !== "number" || !Number.isInteger(value))) {
      errors.push(`字段 '${field}' 期望 integer，实际 ${typeof value}`);
      continue;
    }
    const expectedTypeof = jsonTypeMap[expected];
    if (expectedTypeof && typeof value !== expectedTypeof) {
      errors.push(`字段 '${field}' 期望 ${expected}，实际 ${typeof value}`);
    }
  }

  return errors;
}

/** 带参数校验的工具执行：先 validate → 通过才执行。 */
function safeExecuteTool(
  name: string,
  args: Record<string, unknown>,
  tool: ToolDef,
): string {
  const errors = validateToolArgs(args, tool.schema);
  if (errors.length > 0) {
    const msg = errors.join("; ");
    console.log(`OUT:validate:fail: 工具 '${name}' 参数校验失败 → ${msg}`);
    return `[参数校验失败] ${name}: ${msg}`;
  }
  console.log(`OUT:validate:pass: 工具 '${name}' 参数校验通过 → ${JSON.stringify(args)}`);
  try {
    // Object.values 按插入顺序，与 schema 参数顺序一致（T7/T8 教训）
    const stringArgs = Object.values(args).map(String);
    return String(tool.handler(...stringArgs));
  } catch (e) {
    return `[工具执行失败] ${name}: ${(e as Error).constructor.name}: ${(e as Error).message}`;
  }
}

// ── 工具函数 + schema ──

const WEATHER_DB: Record<string, { condition: string; temp: string }> = {
  北京: { condition: "晴", temp: "25°C" },
  上海: { condition: "多云", temp: "28°C" },
};

function getWeather(...args: string[]): string {
  const city = (args[0] ?? "").trim();
  if (!(city in WEATHER_DB)) {
    return `[未找到] 城市 '${city}'`;
  }
  const w = WEATHER_DB[city];
  return `${city}今天${w.condition}，气温 ${w.temp}`;
}

function calculate(...args: string[]): string {
  const expression = (args[0] ?? "").trim();
  // 安全求值：白名单字符 + 受控求值（不引入新依赖）
  if (!/^[\d\s+\-*/().]+$/.test(expression)) {
    return `[错误] 表达式含非法字符: '${expression}'`;
  }
  try {
    // 受控 eval：白名单已过滤，只有数字和运算符
    const result = Function(`"use strict"; return (${expression});`)() as number;
    const formatted = Number.isInteger(result) ? result.toString() : String(result);
    return `${expression} = ${formatted}`;
  } catch (e) {
    return `[计算失败] ${(e as Error).message}`;
  }
}

const WEATHER_SCHEMA: JsonSchema = {
  type: "object",
  properties: { city: { type: "string", description: "城市名" } },
  required: ["city"],
};

const CALCULATE_SCHEMA: JsonSchema = {
  type: "object",
  properties: { expression: { type: "string", description: "数学表达式" } },
  required: ["expression"],
};

const TOOL_REGISTRY: Record<string, ToolDef> = {
  get_weather: { schema: WEATHER_SCHEMA, handler: getWeather },
  calculate: { schema: CALCULATE_SCHEMA, handler: calculate },
};

interface TestCase {
  name: string;
  args: Record<string, unknown>;
  expect: "pass" | "fail";
}

function demoToolValidation(): void {
  console.log("=".repeat(72));
  console.log("Demo 3: 工具参数校验（ToolValidation）");
  console.log("  执行工具前用 JSON Schema 校验 args（类型 + 必填字段）。");
  console.log("  价值：在工具执行前拦截非法参数，避免崩溃或语义错误。");
  console.log("=".repeat(72));
  console.log();

  const testCases: TestCase[] = [
    { name: "get_weather", args: { city: "北京" }, expect: "pass" },
    { name: "get_weather", args: {}, expect: "fail" },
    { name: "get_weather", args: { city: 12345 }, expect: "fail" },
    { name: "calculate", args: { expression: "28-25" }, expect: "pass" },
    { name: "calculate", args: { expression: 28 }, expect: "fail" },
    { name: "get_stock_price", args: { symbol: "AAPL" }, expect: "fail" },
  ];

  console.log(`  共 ${testCases.length} 个测试用例（合法/非法各半）`);
  console.log("-".repeat(72));

  let passCount = 0;
  let failCount = 0;
  for (const tc of testCases) {
    const tool = TOOL_REGISTRY[tc.name];
    if (!tool) {
      console.log(`OUT:validate:fail: 工具 '${tc.name}' 未注册（未知工具名）`);
      failCount++;
      continue;
    }
    const result = safeExecuteTool(tc.name, tc.args, tool);
    if (
      result.startsWith("[参数校验失败]") ||
      result.startsWith("[工具执行失败]")
    ) {
      failCount++;
    } else {
      passCount++;
    }
    const preview = result.length > 70 ? result.slice(0, 70) + "..." : result;
    console.log(`  → 结果: ${preview}`);
    console.log();
  }

  console.log("-".repeat(72));
  console.log(`OUT:validate:summary: 通过 ${passCount} 个，失败 ${failCount} 个`);
  console.log();
  console.log("  💡 在第13章框架中，把这个 validate 步骤插到 ToolRegistry.execute 开头，");
  console.log("     就能把『模型生成错参数』的 bug 在执行前拦截，而不是等到工具内部崩溃。");
  console.log();
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 4: 现代框架对比（OpenAI Agents SDK / Mastra / Vercel AI SDK）
// ═══════════════════════════════════════════════════════════════════════

/** try-require 一个可选依赖，未安装返回 null（不抛异常）。 */
function tryRequire(moduleName: string): unknown | null {
  try {
    // require 是 CommonJS 全局函数；tsx 支持
    return require(moduleName);
  } catch {
    return null;
  }
}

function demoModernFrameworks(): void {
  console.log("=".repeat(72));
  console.log("Demo 4: 现代框架对比（OpenAI Agents SDK / Mastra / Vercel AI SDK）");
  console.log("  展示『现代框架用更少代码完成同样的流式/结构化/校验』。");
  console.log("  未安装的包会优雅降级为注释代码片段（不强制安装）。");
  console.log("=".repeat(72));
  console.log();

  // ── 1. Vercel AI SDK ──
  console.log("▎ 对比 1: Vercel AI SDK（ai + @ai-sdk/openai）");
  console.log("-".repeat(72));
  const aiPkg = tryRequire("ai");
  if (aiPkg !== null) {
    console.log("OUT:compare:ai_sdk: ✓ 已安装 ai");
  } else {
    console.log("OUT:compare:ai_sdk: ✗ 未安装 ai（这是正常的，本教程不强制安装）");
  }
  console.log("  等价代码（流式 + 结构化）:");
  console.log(`
  // npm install ai @ai-sdk/openai zod
  import { generateObject, streamText } from "ai";
  import { openai } from "@ai-sdk/openai";
  import { z } from "zod";

  // 流式（对应 Demo 1）
  const result = await streamText({
    model: openai("gpt-4o-mini"),
    prompt: "你好",
  });
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);  // 逐块输出
  }

  // 结构化（对应 Demo 2，一行 schema 搞定校验+重试）
  const { object } = await generateObject({
    model: openai("gpt-4o-mini"),
    schema: z.object({
      name: z.string(),
      difficulty: z.enum(["easy", "medium", "hard"]),
      priority: z.number().int().min(1).max(5),
    }),
    prompt: "分析任务：实现登录功能",
  });
  // object 已是强类型，无需手动校验
  `);
  console.log();

  // ── 2. Mastra（TS Agent 框架） ──
  console.log("▎ 对比 2: Mastra（TS Agent 框架）");
  console.log("-".repeat(72));
  const mastraPkg = tryRequire("@mastra/core");
  if (mastraPkg !== null) {
    console.log("OUT:compare:mastra: ✓ 已安装 @mastra/core");
  } else {
    console.log("OUT:compare:mastra: ✗ 未安装 @mastra/core（这是正常的，本教程不强制安装）");
  }
  console.log("  等价代码（工具 + Agent + 流式）:");
  console.log(`
  // npm install @mastra/core
  import { Mastra } from "@mastra/core";

  const weatherTool = {
    id: "get_weather",
    description: "查询天气",
    inputSchema: { type: "object", properties: { city: { type: "string" } } },
    execute: async ({ city }) => \`\${city}今天晴 25°C\`,
  };

  const agent = new Mastra({
    agents: {
      assistant: {
        name: "任务助手",
        instructions: "你是任务助手",
        model: { provider: "OPENAI", name: "gpt-4o-mini" },
        tools: { get_weather: weatherTool },
      },
    },
  });

  const stream = await agent.getAgent("assistant").stream("查北京天气");
  // 自动处理工具调用 + 校验（对应 Demo 3）
  `);
  console.log();

  // ── 3. 决策表 ──
  console.log("▎ 决策：何时用现代框架，何时自造？");
  console.log("-".repeat(72));
  console.log("OUT:compare:decision:");
  console.log(`
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
  `);
  console.log();
}

// ═══════════════════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log("=".repeat(72));
  console.log("第14章 从零造框架 — 高级特性");
  console.log("  流式输出 / 结构化输出强制 / 工具参数校验 / 现代框架对比");
  console.log("  （概念上引用第13章的 6 大组件，独立可运行）");
  console.log("=".repeat(72));
  console.log();

  await demoStreaming();
  await demoStructuredOutput();
  demoToolValidation();
  demoModernFrameworks();

  console.log("=".repeat(72));
  console.log("✓ 本章完成：4 个高级特性全部演示完毕。");
  console.log("  核心收获：理解原理后，用现代框架时你能『看穿』它的每一行代码。");
  console.log("=".repeat(72));
}

main().catch((err: unknown) => {
  console.error("未捕获错误:", err);
  process.exit(1);
});
