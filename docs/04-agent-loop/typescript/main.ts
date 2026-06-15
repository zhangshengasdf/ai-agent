/**
 * 第04章 Agent 循环（The Agent Loop）
 *
 * 本章是全教程的核心概念：**单轮=工具调用，多轮=Agent**。
 *
 * 从第03章的"单轮工具调用"扩展为"多步循环"：
 *   - Agent 持续 observe→reason→act，直到模型给出最终回答（终止条件 1）
 *   - max_steps=10 兜底保护，防止无限循环（终止条件 2）
 *
 * 三个演示：
 *   Demo 1: 多步循环 — 查多城市天气并推荐（需要 3-4 步）
 *   Demo 2: 单步快速完成 — 简单问题模型直接回答（0 次工具调用）
 *   Demo 3: max_steps 防护 — mock 模型模拟无限循环，验证 step=10 优雅停止（离线，不耗 API）
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 工具实现（复用第03章的 3 个 mock 工具，保持一致）
// ════════════════════════════════════════════════════════════════════

function getWeather(city: string): string {
  const mockData: Record<string, string> = {
    北京: "北京今天晴, 25°C, 湿度 40%, 东北风 2 级",
    上海: "上海今天多云, 28°C, 湿度 65%, 东南风 3 级",
    深圳: "深圳今天小雨, 30°C, 湿度 80%, 南风 2 级",
    东京: "东京今天阴, 22°C, 湿度 55%, 西风 1 级",
  };
  return mockData[city] ?? `${city}今天晴, 23°C, 湿度 50%`;
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
    openai: "OpenAI 是 AI 研究公司，开发了 GPT 系列和 ChatGPT。",
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
// 工具定义（JSON Schema，与第03章一致）
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
          city: { type: "string", description: "城市名称，如'北京'、'上海'" },
        },
        required: ["city"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description: "执行数学计算，支持加减乘除和括号。例如：'2+3*4'、'(10-2)/4'",
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
          query: { type: "string", description: "搜索关键词，如'python'、'机器学习'" },
        },
        required: ["query"],
      },
    },
  },
];

// 工具名 → 函数的映射（dispatch 模式）
const TOOL_FUNCTIONS: Record<string, (...args: string[]) => string> = {
  get_weather: (city: string) => getWeather(city),
  calculate: (expression: string) => calculate(expression),
  search_wiki: (query: string) => searchWiki(query),
};

// ⚠️ 必须有上限！这是 Agent 循环的"保险丝"。
const MAX_STEPS = 10;

// ════════════════════════════════════════════════════════════════════
// 核心：Agent 循环
// ════════════════════════════════════════════════════════════════════

async function agentLoop(userMessage: string): Promise<string> {
  /**
   * Agent 循环：持续调用工具直到模型给出最终回答或达到 max_steps。
   *
   * 终止条件：
   *   1. 模型不再调用工具 → 返回最终回答（正常完成）
   *   2. 达到 MAX_STEPS → 强制停止（保险丝）
   */
  // messages 在循环外初始化，循环内只"追加"——模型需要完整历史来决策。
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content:
        "你是一个任务助手 Agent。你可以查天气、做计算、搜百科。" +
        "面对复杂任务，请一步步调用工具收集信息，最后给出综合回答。" +
        "当信息足够回答时，直接给出最终回答，不要继续调用工具。",
    },
    { role: "user", content: userMessage },
  ];

  console.log(`\n${"=".repeat(60)}`);
  console.log(`任务: ${userMessage}`);
  console.log("=".repeat(60));

  // ── 循环最多 MAX_STEPS 次 ──────────────────────────────────────
  for (let step = 1; step <= MAX_STEPS; step++) {
    console.log(`OUT:step${step}: 思考中... (观察历史，决定下一步)`);

    // ── Reason：让 LLM 决定下一步 ──────────────────────────────
    const response = await client.chat.completions.create({
      model: cfg.model,
      messages,
      tools,
      tool_choice: "auto",
    });
    const assistantMsg = response.choices[0].message;

    // ── 终止条件 1：模型不再调工具 = 任务完成 ──────────────────
    if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
      const answer = assistantMsg.content ?? "(空回答)";
      console.log(`OUT:step${step}: ✓ 任务完成！模型给出最终回答（未调用工具）`);
      const preview = answer.length > 120 ? answer.slice(0, 120) + "..." : answer;
      console.log(`OUT:step${step}: 回答: ${preview}`);
      return answer;
    }

    // ── Act：模型决定调用工具，执行并把结果反馈回去 ───────────
    messages.push(assistantMsg);

    const toolNames = assistantMsg.tool_calls
      .filter((t) => t.type === "function")
      .map((t) => t.function.name);
    console.log(`OUT:step${step}: 决定调用工具: ${toolNames.join(", ")}`);

    // 执行每个工具调用（模型可能一次返回多个）
    // ⚠️ TS 类型安全：ChatCompletionMessageToolCall 是 discriminated union，
    //    访问 .function 前必须检查 tc.type === "function"
    for (const tc of assistantMsg.tool_calls) {
      if (tc.type !== "function") continue;
      const funcName = tc.function.name;
      let args: Record<string, string> = {};
      try {
        args = JSON.parse(tc.function.arguments);
      } catch {
        args = {};
      }

      console.log(`OUT:step${step}: 执行 ${funcName}(${JSON.stringify(args)})`);

      // 执行工具（dispatch 模式）
      const func = TOOL_FUNCTIONS[funcName];
      let result: string;
      if (!func) {
        result = `错误：未知工具 '${funcName}'`;
      } else {
        try {
          result = func(...Object.values(args));
        } catch (e) {
          result = `工具执行错误：${e}`;
        }
      }

      const preview = result.length > 80 ? result.slice(0, 80) + "..." : result;
      console.log(`OUT:step${step}: 观察结果: ${preview}`);

      // ── Observe：把工具结果以 role="tool" 追加到 messages ────
      messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: result,
      });
    }
    // 循环回到顶部：下一轮的 Reason 会基于更新后的 messages 决策。
  }

  // ── 终止条件 2：达到 max_steps，强制停止 ──────────────────────
  console.log(`OUT:max_steps: ⚠️ 达到最大步数 ${MAX_STEPS}，强制停止！（防止无限循环）`);
  return "(已达到最大步数，可能需要更具体的指令或更好的工具)";
}

// ════════════════════════════════════════════════════════════════════
// Demo 3：max_steps 防护（离线 mock，不消耗 API 额度）
// ════════════════════════════════════════════════════════════════════

interface MockToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

function demoMaxStepsProtection(): void {
  /**
   * 用 mock 响应演示 max_steps 防护（不消耗 API 额度）。
   *
   * 模拟一个"总是返回 tool_calls 的模型"——它永远不会给最终回答，
   * 从而构造出无限循环场景。验证 Agent 循环在 step=10 时优雅停止。
   */

  // 模拟"无限循环"模型：每次都返回 tool_calls，永远不给最终回答
  const mockInfiniteLoopModel = (messages: unknown[]): MockToolCall[] => {
    return [
      {
        id: `call_mock_${messages.length}`,
        type: "function" as const,
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city: "北京" }),
        },
      },
    ];
  };

  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 3: max_steps 防护（mock 无限循环场景）");
  console.log("=".repeat(60));
  console.log("[说明] 模拟一个'总在重复调工具'的坏模型，验证 max_steps 兜底。");

  const messages: unknown[] = [
    { role: "system", content: "你是任务助手 Agent..." },
    { role: "user", content: "查北京天气（演示无限循环防护）" },
  ];

  for (let step = 1; step <= MAX_STEPS; step++) {
    const toolCalls = mockInfiniteLoopModel(messages);

    if (toolCalls.length === 0) {
      console.log(`OUT:max_steps:step${step}: 任务完成（mock 不会走到这）`);
      return;
    }

    console.log(`OUT:max_steps:step${step}: 调用工具（mock 重复调用）`);
    messages.push({ role: "assistant", tool_calls: toolCalls });

    for (const tc of toolCalls) {
      const funcName = tc.function.name;
      const args: Record<string, string> = JSON.parse(tc.function.arguments);
      const result = TOOL_FUNCTIONS[funcName](...Object.values(args));
      console.log(`OUT:max_steps:step${step}: 结果: ${result}`);
      messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: result,
      });
    }
  }

  // ⚠️ 这里是关键：循环正常结束（step 用尽），优雅停止
  console.log(`OUT:max_steps: ⚠️ 达到最大步数 ${MAX_STEPS}，强制停止！`);
  console.log(`OUT:max_steps: ✓ 防护生效——没有无限循环，Agent 安全停下。`);
  console.log(`OUT:max_steps: 💡 真实场景：请检查工具返回是否模糊、任务是否清晰、模型是否足够强。`);
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock Agent 循环（API 不可用时演示完整循环逻辑）
// ════════════════════════════════════════════════════════════════════

interface MockDecision {
  action: string | null;
  args?: Record<string, string>;
  answer?: string;
}

function demoOfflineMultiStep(): void {
  /**
   * 离线演示多步 Agent 循环逻辑（API 不可用时降级使用）。
   * 用预设的 mock 响应模拟"查三城市天气并推荐"的完整 4 步循环。
   */
  console.log(`\n${"=".repeat(60)}`);
  console.log("离线演示：多步 Agent 循环（mock 4 步：查三城市 + 推荐）");
  console.log("=".repeat(60));

  // 预设一个"聪明"的 mock 模型的决策序列
  const mockDecisions: MockDecision[] = [
    { action: "get_weather", args: { city: "北京" } },
    { action: "get_weather", args: { city: "上海" } },
    { action: "get_weather", args: { city: "深圳" } },
    {
      action: null,
      answer:
        "推荐北京旅行：晴朗 25°C，温度最宜人；上海多云 28°C 次之；深圳小雨 30°C 较闷热。",
    },
  ];

  let step = 0;
  for (const decision of mockDecisions) {
    step++;
    console.log(`OUT:offline:step${step}: 思考中...`);

    if (decision.action === null) {
      // 终止条件 1：模型决定不调工具，给最终回答
      console.log(`OUT:offline:step${step}: ✓ 信息足够，给出最终回答（不再调工具）`);
      console.log(`OUT:offline:step${step}: 回答: ${decision.answer}`);
      break;
    }

    const func = TOOL_FUNCTIONS[decision.action];
    const result = func(...Object.values(decision.args ?? {}));
    console.log(`OUT:offline:step${step}: 调用 ${decision.action}(${JSON.stringify(decision.args)})`);
    console.log(`OUT:offline:step${step}: 结果: ${result}`);
  }

  console.log(`OUT:offline: ✓ 循环正常终止（模型自主决定停止），共 ${step} 步。`);
}

// ════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] 工具数量: ${tools.length}`);
  console.log(`[config] MAX_STEPS=${MAX_STEPS}（Agent 循环保险丝）`);

  let apiOk = true;

  try {
    // ── Demo 1: 多步循环（需要 3-4 步）─────────────────────────
    await agentLoop(
      "帮我查一下北京、上海、深圳三个城市的天气，" +
        "然后推荐哪个城市今天最适合旅行，说明理由。",
    );

    // ── Demo 2: 单步快速完成（简单问题，0 次工具调用）──────────
    await agentLoop("你好，请用一句话介绍你自己。");
  } catch (err) {
    apiOk = false;
    const errorMsg = String(err);
    const isAuthError =
      errorMsg.includes("401") ||
      errorMsg.includes("invalid_api_key") ||
      errorMsg.includes("Authentication") ||
      errorMsg.includes("sk-REPLACE-ME");
    const isToolUnsupported =
      errorMsg.includes("does not support tools") ||
      (errorMsg.includes("400") && errorMsg.toLowerCase().includes("model"));

    console.log(`\n[提示] 真实 API 调用失败（${err instanceof Error ? err.constructor.name : "Error"}）。`);
    if (isAuthError) {
      console.log(`[提示] 原因：API 密钥无效或为占位符。请编辑 ai-agent/.env 填入有效密钥。`);
      console.log(`[提示] 当前 provider=${cfg.provider}，需要对应的 API 密钥。`);
    } else if (isToolUnsupported) {
      console.log(`[提示] 原因：当前模型 ${cfg.model} 不支持 tools API。`);
      console.log(`[提示] Ollama qwen2.5vl:latest 不支持工具调用。`);
      console.log(`[提示] 请用支持 function calling 的模型，或在 .env 设 PROVIDER=openai/deepseek。`);
    } else {
      console.log(`[提示] 原因：${err}`);
    }
    console.log(`[提示] 已自动降级为离线 mock 演示，Agent 循环逻辑不受影响。\n`);
  }

  // ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）────
  demoOfflineMultiStep();
  demoMaxStepsProtection();

  console.log(`\n${"=".repeat(60)}`);
  if (apiOk) {
    console.log("所有演示完成！（含真实 API 多步循环 + max_steps 防护）");
  } else {
    console.log("离线演示完成！（真实 API 未配置，但 Agent 循环逻辑已完整展示）");
  }
  console.log(`💡 核心要点：单轮=工具调用，多步循环=Agent。max_steps 是保险丝。`);
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
