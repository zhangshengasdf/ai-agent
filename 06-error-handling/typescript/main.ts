/**
 * 第06章 错误处理与容错（Error Handling & Resilience）
 *
 * 本章在第04章 Agent 循环基础上，加入四大容错机制：
 *
 *   机制 1：指数退避重试 —— 对可重试错误（超时/限流/连接），重试最多 3 次，
 *           每次等 2**attempt 秒（演示用 100ms 缩放，真实用 1000ms）。
 *   机制 2：工具异常 + Agent 自我纠正 —— 工具抛异常时，把错误以 role="tool"
 *           反馈给 Agent，让它换工具或调整参数，而非崩溃。
 *   机制 3：幻觉工具名检测 —— 模型调了不存在的工具名 → 告知正确工具列表 →
 *           Agent 重新选择合法工具。
 *   机制 4：区分可重试错误 vs 永久错误 —— 网络错误重试，认证错误直接退出。
 *
 * 离线 mock 设计：.env 的 OPENAI_API_KEY=sk-REPLACE-ME 是占位符，真实 API
 * 调用必失败（401）。所有 demo 先 try 真实 API（失败时降级），然后用离线
 * mock 100% 可靠地演示四大容错机制，保证 exit code 0。
 */

import OpenAI, {
  APIConnectionError,
  APIConnectionTimeoutError,
  APIError,
  AuthenticationError,
  BadRequestError,
  InternalServerError,
  RateLimitError,
} from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 工具实现（复用第03/04章的 mock 工具）
// ════════════════════════════════════════════════════════════════════

const WEATHER_DATA: Record<string, string> = {
  北京: "北京今天晴, 25°C, 湿度 40%, 东北风 2 级",
  上海: "上海今天多云, 28°C, 湿度 65%, 东南风 3 级",
  深圳: "深圳今天小雨, 30°C, 湿度 80%, 南风 2 级",
  东京: "东京今天阴, 22°C, 湿度 55%, 西风 1 级",
};

function getWeather(city: string): string {
  if (!(city in WEATHER_DATA)) {
    const available = Object.keys(WEATHER_DATA).join("、");
    throw new Error(`城市 '${city}' 不在数据库中。可用的城市：${available}`);
  }
  return WEATHER_DATA[city];
}

function calculate(expression: string): string {
  const allowed = new Set("0123456789+-*/.() ");
  for (const c of expression) {
    if (!allowed.has(c)) {
      return "错误：表达式包含不允许的字符，只支持数字和 + - * / ( )";
    }
  }
  try {
    // eslint-disable-next-line no-eval
    const result = eval(expression);
    return String(result);
  } catch (e) {
    return `计算错误：${e}`;
  }
}

function searchWiki(query: string): string {
  const knowledge: Record<string, string> = {
    python: "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。",
    "机器学习": "机器学习是 AI 的分支，使计算机从数据中学习。",
    agent: "AI Agent 是能感知环境、决策、行动的自主系统。",
    火星: "火星是太阳系第四颗行星，表面温度约 -63°C，大气稀薄。",
  };
  const queryLower = query.toLowerCase();
  for (const [key, value] of Object.entries(knowledge)) {
    if (queryLower.includes(key)) {
      return value;
    }
  }
  return `未找到与'${query}'相关的百科条目。`;
}

// ════════════════════════════════════════════════════════════════════
// 工具定义（JSON Schema）
// ════════════════════════════════════════════════════════════════════

const tools: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "查询指定城市的当前天气，返回温度、湿度和风力信息",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string", description: "城市名称，如'北京'、'上海'、'东京'" },
        },
        required: ["city"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description: "执行数学计算，支持加减乘除和括号",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "数学表达式，如'2+3*4'" },
        },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_wiki",
      description: "搜索百科知识，返回与查询相关的简介信息",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "搜索关键词，如'python'、'火星'" },
        },
        required: ["query"],
      },
    },
  },
];

const TOOL_FUNCTIONS: Record<string, (...args: string[]) => string> = {
  get_weather: (city: string) => getWeather(city),
  calculate: (expression: string) => calculate(expression),
  search_wiki: (query: string) => searchWiki(query),
};

const VALID_TOOL_NAMES = new Set(Object.keys(TOOL_FUNCTIONS));

const MAX_STEPS = 10;
const MAX_RETRIES = 3;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ════════════════════════════════════════════════════════════════════
// 机制 4：错误分类 —— 区分可重试错误 vs 永久错误
// ════════════════════════════════════════════════════════════════════

function isRetryable(error: unknown): boolean {
  // APIConnectionError 覆盖 APIConnectionTimeoutError（后者继承前者）
  if (
    error instanceof APIConnectionError ||
    error instanceof RateLimitError ||
    error instanceof InternalServerError
  ) {
    return true;
  }
  // 通用 APIError：检查 status 是否为 5xx
  if (error instanceof APIError) {
    const status = error.status;
    if (typeof status === "number" && status >= 500) {
      return true;
    }
  }
  return false;
}

// ════════════════════════════════════════════════════════════════════
// 机制 1：指数退避重试
// ════════════════════════════════════════════════════════════════════

async function callLlmWithRetry(
  messages: OpenAI.ChatCompletionMessageParam[],
  toolsList?: OpenAI.ChatCompletionTool[],
  backoffScaleMs = 1000,
): Promise<OpenAI.ChatCompletion> {
  let lastError: unknown;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const params: OpenAI.ChatCompletionCreateParamsNonStreaming = {
        model: cfg.model,
        messages,
      };
      if (toolsList) {
        params.tools = toolsList;
        params.tool_choice = "auto";
      }
      return await client.chat.completions.create(params);
    } catch (e) {
      lastError = e;
      if (!isRetryable(e)) {
        // 永久错误（认证/参数）—— 立即抛出，不重试
        throw e;
      }
      if (attempt === MAX_RETRIES - 1) {
        const errName = e instanceof Error ? e.constructor.name : "Error";
        console.log(
          `OUT:retry: 第 ${attempt + 1}/${MAX_RETRIES} 次失败（${errName}），已达最大重试次数，放弃。`,
        );
        throw e;
      }
      const waitMs = Math.pow(2, attempt) * backoffScaleMs;
      const errName = e instanceof Error ? e.constructor.name : "Error";
      console.log(
        `OUT:retry: 第 ${attempt + 1}/${MAX_RETRIES} 次失败（${errName}），等待 ${(waitMs / 1000).toFixed(1)}s 后重试...`,
      );
      await sleep(waitMs);
    }
  }
  throw lastError;
}

// ════════════════════════════════════════════════════════════════════
// 护栏：输入/输出校验（基础版）
// ════════════════════════════════════════════════════════════════════

function validateInput(userMessage: string): string {
  if (userMessage.length > 10_000) {
    throw new Error("输入过长（超过 10000 字符），请精简后重试");
  }
  const lower = userMessage.toLowerCase();
  if (lower.includes("ignore previous instructions") || userMessage.includes("忽略以上所有指令")) {
    throw new Error("检测到疑似 prompt 注入，已拒绝");
  }
  return userMessage;
}

function validateOutput(answer: string): string {
  if (answer.length > 5000) {
    return answer.slice(0, 5000) + "\n\n（回答过长，已截断）";
  }
  return answer;
}

// ════════════════════════════════════════════════════════════════════
// 核心：带容错的 Agent 循环（扩展第04章）
// ════════════════════════════════════════════════════════════════════

async function resilientAgentLoop(userMessage: string): Promise<string> {
  validateInput(userMessage);

  const messages: OpenAI.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content:
        "你是一个任务助手 Agent。你可以查天气、做计算、搜百科。" +
        "面对复杂任务，请一步步调用工具收集信息，最后给出综合回答。" +
        "如果某个工具失败，请阅读错误信息并尝试换工具或调整参数。" +
        "当信息足够回答时，直接给出最终回答。",
    },
    { role: "user", content: userMessage },
  ];

  console.log(`\n${"=".repeat(60)}`);
  console.log(`任务: ${userMessage}`);
  console.log("=".repeat(60));

  for (let step = 1; step <= MAX_STEPS; step++) {
    console.log(`OUT:step${step}: 思考中...`);

    // ── 机制 1 + 4：带退避重试的 LLM 调用 ──────────────────────
    const response = await callLlmWithRetry(messages, tools);
    const assistantMsg = response.choices[0].message;

    // 终止条件 1：模型不再调工具 = 任务完成
    if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
      const answer = assistantMsg.content ?? "(空回答)";
      console.log(`OUT:step${step}: ✓ 任务完成！`);
      const preview = answer.length > 120 ? answer.slice(0, 120) + "..." : answer;
      console.log(`OUT:step${step}: 回答: ${preview}`);
      return validateOutput(answer);
    }

    messages.push(assistantMsg);
    const toolNames = assistantMsg.tool_calls
      .filter((t) => t.type === "function")
      .map((t) => t.function.name);
    console.log(`OUT:step${step}: 决定调用工具: ${toolNames.join(", ")}`);

    // ── 执行每个工具调用（含机制 2 + 3）─────────────────────────
    for (const tc of assistantMsg.tool_calls) {
      // ⚠️ TS discriminated union：访问 .function 前必须检查 type
      if (tc.type !== "function") continue;
      const funcName = tc.function.name;

      let args: Record<string, string>;
      try {
        args = JSON.parse(tc.function.arguments);
      } catch {
        // JSON 解析失败：反馈给 Agent 让它重新生成参数
        args = {};
        const result =
          `[参数解析失败] 工具 '${funcName}' 的 arguments ` +
          `'${tc.function.arguments}' 不是合法 JSON。请重新生成。`;
        console.log(`OUT:step${step}: ⚠️ JSON 解析失败，反馈给 Agent`);
        messages.push({ role: "tool", tool_call_id: tc.id, content: result });
        continue;
      }

      // ── 机制 3：幻觉工具名检测 ──────────────────────────────
      if (!VALID_TOOL_NAMES.has(funcName)) {
        const result =
          `[错误] 工具 '${funcName}' 不存在。` +
          `可用的工具有：${[...VALID_TOOL_NAMES].sort().join(", ")}。` +
          `请从上述列表中选择一个。`;
        console.log(`OUT:step${step}: 🚫 幻觉工具检测：'${funcName}' 不存在，已告知 Agent`);
        messages.push({ role: "tool", tool_call_id: tc.id, content: result });
        continue;
      }

      // ── 机制 2：工具异常 → 反馈给 Agent 自我纠正 ─────────────
      console.log(`OUT:step${step}: 执行 ${funcName}(${JSON.stringify(args)})`);
      const func = TOOL_FUNCTIONS[funcName];
      let result: string;
      try {
        result = func(...Object.values(args));
        const preview = result.length > 80 ? result.slice(0, 80) + "..." : result;
        console.log(`OUT:step${step}: 观察结果: ${preview}`);
      } catch (e) {
        const errName = e instanceof Error ? e.constructor.name : "Error";
        const errMsg = e instanceof Error ? e.message : String(e);
        result = `[工具执行失败] ${funcName} 抛出异常：${errName}: ${errMsg}`;
        console.log(`OUT:step${step}: ⚠️ 工具异常，反馈给 Agent：${errName}`);
      }

      messages.push({ role: "tool", tool_call_id: tc.id, content: result });
    }
  }

  console.log(`OUT:max_steps: ⚠️ 达到最大步数 ${MAX_STEPS}，强制停止！`);
  return "(已达到最大步数)";
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：Demo A —— 指数退避重试序列
// ════════════════════════════════════════════════════════════════════

async function demoBackoffRetrySequence(): Promise<void> {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo A: 指数退避重试序列（mock：前 2 次失败，第 3 次成功）");
  console.log("=".repeat(60));

  let callCount = 0;

  function mockFlakyApi(): string {
    callCount++;
    if (callCount <= 2) {
      throw new APIConnectionError({ message: "connection failed" });
    }
    return "✓ API 调用成功，返回数据";
  }

  const backoffScaleMs = 100; // 演示用 100ms（生产用 1000ms：1s/2s/4s）

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const result = mockFlakyApi();
      console.log(`OUT:demoA: ✓ 第 ${attempt + 1} 次尝试成功！${result}`);
      break;
    } catch (e) {
      if (!isRetryable(e)) {
        console.log(`OUT:demoA: 永久错误，不重试：${e instanceof Error ? e.constructor.name : "Error"}`);
        throw e;
      }
      if (attempt === MAX_RETRIES - 1) {
        const errName = e instanceof Error ? e.constructor.name : "Error";
        console.log(`OUT:demoA: 第 ${attempt + 1}/${MAX_RETRIES} 次失败（${errName}），已达上限，放弃。`);
        break;
      }
      const waitMs = Math.pow(2, attempt) * backoffScaleMs;
      const errName = e instanceof Error ? e.constructor.name : "Error";
      console.log(
        `OUT:demoA: 第 ${attempt + 1}/${MAX_RETRIES} 次失败（${errName}），等待 ${(waitMs / 1000).toFixed(1)}s 后重试...`,
      );
      await sleep(waitMs);
    }
  }

  console.log(`OUT:demoA: 💡 生产环境用 backoffScale=1000ms（等待 1s/2s/4s），本章用 100ms 演示。`);
  console.log(`OUT:demoA: 💡 只重试可重试错误（超时/限流/连接），认证错误立即退出。`);
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：Demo B —— 工具异常 + Agent 自我纠正
// ════════════════════════════════════════════════════════════════════

interface MockStep {
  step: number;
  action: "call_tool" | "observe_error" | "observe_result" | "final_answer";
  tool?: string;
  args?: Record<string, string>;
  error?: string;
  result?: string;
  answer?: string;
  note?: string;
}

function demoToolSelfCorrection(): void {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo B: 工具异常 + Agent 自我纠正（mock 决策序列）");
  console.log("=".repeat(60));
  console.log("[场景] 用户问'火星天气'，get_weather 失败，Agent 改用 search_wiki");

  const steps: MockStep[] = [
    { step: 1, action: "call_tool", tool: "get_weather", args: { city: "火星" } },
    {
      step: 2,
      action: "observe_error",
      error: "Error: 城市 '火星' 不在数据库中。可用的城市：北京、上海、深圳、东京",
    },
    {
      step: 3,
      action: "call_tool",
      tool: "search_wiki",
      args: { query: "火星" },
      note: "Agent 看到错误后换工具",
    },
    {
      step: 4,
      action: "observe_result",
      result: "火星是太阳系第四颗行星，表面温度约 -63°C，大气稀薄。",
    },
    {
      step: 5,
      action: "final_answer",
      answer:
        "我查不到火星的实时天气（不在天气数据库中），但查到百科：" +
        "火星表面温度约 -63°C，大气稀薄。如果你需要地球城市的天气，请告诉我城市名。",
    },
  ];

  for (const s of steps) {
    if (s.action === "call_tool") {
      console.log(`OUT:demoB:step${s.step}: 调用 ${s.tool}(${JSON.stringify(s.args)})`);
      if (s.note) console.log(`OUT:demoB:step${s.step}: 💡 ${s.note}`);
    } else if (s.action === "observe_error") {
      console.log(`OUT:demoB:step${s.step}: ⚠️ 工具异常，反馈给 Agent：${s.error}`);
    } else if (s.action === "observe_result") {
      const r = s.result ?? "";
      const preview = r.length > 60 ? r.slice(0, 60) + "..." : r;
      console.log(`OUT:demoB:step${s.step}: 观察结果: ${preview}`);
    } else if (s.action === "final_answer") {
      const a = s.answer ?? "";
      const preview = a.length > 80 ? a.slice(0, 80) + "..." : a;
      console.log(`OUT:demoB:step${s.step}: ✓ 自我纠正成功！最终回答：${preview}`);
    }
  }

  console.log(`OUT:demoB: 💡 关键：工具异常没让 Agent 崩溃，而是驱动它换工具。`);
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：Demo C —— 幻觉工具名检测
// ════════════════════════════════════════════════════════════════════

interface MockToolCall {
  step: number;
  id: string;
  name: string;
  args: Record<string, string>;
}

function demoHallucinationDetection(): void {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo C: 幻觉工具名检测（mock：模型调了不存在的工具）");
  console.log("=".repeat(60));

  const mockToolCalls: MockToolCall[] = [
    { step: 1, id: "call_1", name: "get_stock_price", args: { symbol: "AAPL" } },
    { step: 2, id: "call_2", name: "search_wiki", args: { query: "Apple Inc" } },
  ];

  for (const tc of mockToolCalls) {
    console.log(`OUT:demoC:step${tc.step}: 模型调用工具 '${tc.name}'(${JSON.stringify(tc.args)})`);

    if (!VALID_TOOL_NAMES.has(tc.name)) {
      const result =
        `[错误] 工具 '${tc.name}' 不存在。` +
        `可用的工具有：${[...VALID_TOOL_NAMES].sort().join(", ")}。`;
      console.log(`OUT:demoC:step${tc.step}: 🚫 幻觉检测：'${tc.name}' 不存在！`);
      console.log(`OUT:demoC:step${tc.step}: 告知 Agent：${result}`);
    } else {
      const result = TOOL_FUNCTIONS[tc.name](...Object.values(tc.args));
      const preview = result.length > 60 ? result.slice(0, 60) + "..." : result;
      console.log(`OUT:demoC:step${tc.step}: ✓ 合法工具，执行成功：${preview}`);
    }
  }

  console.log(`OUT:demoC: 💡 关键：幻觉工具没让循环崩溃，而是告知列表让模型纠正。`);
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：Demo D —— 错误分类（可重试 vs 永久）
// ════════════════════════════════════════════════════════════════════

interface ErrorSample {
  name: string;
  error: unknown;
}

function demoErrorClassification(): void {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo D: 错误分类（可重试 vs 永久）");
  console.log("=".repeat(60));

  const samples: ErrorSample[] = [
    { name: "APIConnectionTimeoutError（超时）", error: new APIConnectionTimeoutError() },
    {
      name: "APIConnectionError（连接失败）",
      error: new APIConnectionError({ message: "conn fail" }),
    },
    {
      name: "RateLimitError（限流 429）",
      error: new RateLimitError(429, { error: "rate" }, "rate limit", new Headers()),
    },
    {
      name: "AuthenticationError（认证 401）",
      error: new AuthenticationError(401, { error: "bad key" }, "auth fail", new Headers()),
    },
    {
      name: "BadRequestError（参数 400）",
      error: new BadRequestError(400, { error: "bad" }, "bad request", new Headers()),
    },
    { name: "Error（工具异常）", error: new Error("城市 '火星' 不在数据库中") },
  ];

  console.log(`${"错误类型".padEnd(38)}${"可重试？".padEnd(12)}处理方式`);
  console.log("-".repeat(75));
  for (const { name, error } of samples) {
    const retryable = isRetryable(error);
    let action: string;
    if (retryable) {
      action = "退避重试";
    } else if (error instanceof AuthenticationError || error instanceof BadRequestError) {
      action = "立即退出（永久错误）";
    } else {
      action = "反馈给 Agent（机制 2）";
    }
    const flag = retryable ? "✅ 是" : "❌ 否";
    console.log(`OUT:demoD: ${name.padEnd(36)}${flag.padEnd(12)}${action}`);
  }

  console.log(`\nOUT:demoD: 💡 核心：只重试瞬时故障（超时/限流/连接），永久错误立即失败。`);
  console.log(`OUT:demoD: 💡 混为一谈会导致：认证错误重试 3 次纯属浪费，或网络抖动直接崩溃。`);
}

// ════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] MAX_RETRIES=${MAX_RETRIES}, MAX_STEPS=${MAX_STEPS}`);
  console.log(`[config] 合法工具: ${[...VALID_TOOL_NAMES].sort()}`);

  let apiOk = true;

  try {
    await resilientAgentLoop(
      "帮我查一下北京、上海两个城市的天气，然后推荐哪个更适合旅行。",
    );
  } catch (err) {
    apiOk = false;
    if (err instanceof AuthenticationError) {
      console.log(`\n[提示] 认证失败（AuthenticationError）—— 这是永久错误，不重试。`);
      console.log(`[提示] 原因：OPENAI_API_KEY=sk-REPLACE-ME 是占位符。`);
      console.log(`[提示] 这是机制 4 的体现：永久错误直接退出，不浪费时间重试。`);
      console.log(`[提示] 已自动降级为离线 mock 演示四大容错机制。\n`);
    } else if (err instanceof BadRequestError) {
      console.log(`\n[提示] 请求错误（BadRequestError）—— 永久错误：${err.message}`);
      console.log(`[提示] 可能是模型不支持 tools API（如 Ollama qwen2.5vl）。`);
      console.log(`[提示] 已自动降级为离线 mock 演示。\n`);
    } else if (err instanceof APIConnectionError || err instanceof RateLimitError) {
      console.log(`\n[提示] 可重试错误耗尽（${err.constructor.name}）—— 已重试 ${MAX_RETRIES} 次仍失败。`);
      console.log(`[提示] 已自动降级为离线 mock 演示。\n`);
    } else {
      const errorMsg = err instanceof Error ? err.message : String(err);
      const isAuth =
        errorMsg.includes("401") ||
        errorMsg.includes("invalid_api_key") ||
        errorMsg.includes("Authentication") ||
        errorMsg.includes("sk-REPLACE-ME");
      console.log(`\n[提示] API 调用失败（${err instanceof Error ? err.constructor.name : "Error"}）。`);
      if (isAuth) {
        console.log(`[提示] 原因：API 密钥为占位符 sk-REPLACE-ME。请编辑 ai-agent/.env。`);
      } else {
        console.log(`[提示] 原因：${errorMsg}`);
      }
      console.log(`[提示] 已自动降级为离线 mock 演示。\n`);
    }
  }

  // ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）──────
  await demoBackoffRetrySequence();
  demoToolSelfCorrection();
  demoHallucinationDetection();
  demoErrorClassification();

  console.log(`\n${"=".repeat(60)}`);
  if (apiOk) {
    console.log("所有演示完成！（含真实 API 容错 + 四大机制离线 mock）");
  } else {
    console.log("离线演示完成！（真实 API 未配置，但四大容错机制已完整展示）");
  }
  console.log(`💡 四大机制：退避重试 / 工具自我纠正 / 幻觉检测 / 错误分类`);
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
