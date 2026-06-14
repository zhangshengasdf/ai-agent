/**
 * 第03章 工具调用（Tool Use / Function Calling）
 *
 * 演示完整的单轮工具调用流程：
 *   Step 1: 发送 user 消息 + tools 定义 → 模型返回 tool_calls
 *   Step 2: 解析 tool_calls，执行对应工具函数，获取结果
 *   Step 3: 把工具结果以 role="tool" 消息追加到 messages
 *   Step 4: 再次调用 API → 模型基于工具结果返回最终文本回答
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 工具实现（全部 mock，不调真实 API）
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
    python:
      "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。" +
      "它以简洁易读的语法著称，广泛应用于 Web 开发、数据科学、AI 等领域。",
    "机器学习":
      "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进，" +
      "而无需被显式编程。主要方法包括监督学习、无监督学习和强化学习。",
    agent:
      "在 AI 领域，Agent（智能体）是指能够感知环境、做出决策并采取行动的自主系统。" +
      "一个典型的 AI Agent 包含 LLM（大脑）、工具（手）和循环控制（自主性）。",
    openai:
      "OpenAI 是一家美国人工智能研究公司，成立于 2015 年。" +
      "它开发了 GPT 系列大语言模型和 ChatGPT，是当前 AI 领域最具影响力的公司之一。",
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
// 工具定义（JSON Schema 格式，告诉模型有哪些工具可用）
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
          city: {
            type: "string",
            description: "城市名称，如'北京'、'上海'",
          },
        },
        required: ["city"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description:
        "执行数学计算，支持加减乘除和括号。例如：'2+3*4'、'(10-2)/4'",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "数学表达式，如'2+3*4'",
          },
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
          query: {
            type: "string",
            description: "搜索关键词，如'python'、'机器学习'",
          },
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

// ════════════════════════════════════════════════════════════════════
// 完整单轮工具调用流程
// ════════════════════════════════════════════════════════════════════

async function runToolFlow(userMessage: string): Promise<string> {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content:
        "你是一个任务助手 Agent。你可以查天气、做计算、搜百科。" +
        "当用户的问题需要用到这些能力时，主动调用对应工具。",
    },
    { role: "user", content: userMessage },
  ];

  // ── Step 1: 发送请求，让模型决定是否调用工具 ─────────────────────
  console.log(`\n${"=".repeat(60)}`);
  console.log(`用户提问: ${userMessage}`);
  console.log("=".repeat(60));

  const response = await client.chat.completions.create({
    model: cfg.model,
    messages,
    tools,
    tool_choice: "auto",
  });

  const assistantMsg = response.choices[0].message;

  // ── 检测模型是否返回了 tool_calls ───────────────────────────────
  if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
    // 模型直接返回了文本，没用工具
    console.log("OUT:step1: 模型决定不调用工具，直接回答");
    const finalAnswer = assistantMsg.content ?? "(空回答)";
    console.log(`OUT:step4: 最终回答: ${finalAnswer}`);
    return finalAnswer;
  }

  // ── Step 2: 解析 tool_calls，执行工具 ────────────────────────────
  console.log("OUT:step1: 模型决定调用工具:");
  for (const tc of assistantMsg.tool_calls) {
    if (tc.type === "function") {
      console.log(`  → ${tc.function.name}(${tc.function.arguments})`);
    }
  }

  // 把 assistant 的 tool_calls 消息追加到 messages
  messages.push(assistantMsg);

  // 执行每个工具调用
  for (const tc of assistantMsg.tool_calls) {
    if (tc.type !== "function") continue;
    const funcName = tc.function.name;
    let args: Record<string, string> = {};
    try {
      args = JSON.parse(tc.function.arguments);
    } catch {
      args = {};
    }

    // 执行工具
    const func = TOOL_FUNCTIONS[funcName];
    let result: string;
    if (!func) {
      result = `错误：未知工具 '${funcName}'`;
    } else {
      try {
        const argValues = Object.values(args);
        result = func(...argValues);
      } catch (e) {
        result = `工具执行错误：${e}`;
      }
    }

    console.log(`OUT:step2: 工具执行结果: ${funcName} → ${result}`);

    // ── Step 3: 把工具结果以 role="tool" 追加到 messages ─────────
    messages.push({
      role: "tool",
      tool_call_id: tc.id,
      content: result,
    });
  }

  console.log("OUT:step3: 将结果反馈给模型...");

  // ── Step 4: 再次调用 API，模型基于工具结果给出最终回答 ──────────
  const response2 = await client.chat.completions.create({
    model: cfg.model,
    messages,
  });

  const finalAnswer = response2.choices[0].message.content ?? "(空回答)";
  console.log(`OUT:step4: 最终回答: ${finalAnswer}`);
  return finalAnswer;
}

// ════════════════════════════════════════════════════════════════════
// 演示：三个不同类型的工具调用
// ════════════════════════════════════════════════════════════════════

async function main() {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] 工具数量: ${tools.length}`);
  console.log(
    `[config] 可用工具: ${tools.filter((t) => t.type === "function").map((t) => t.function.name).join(", ")}`,
  );

  try {
    // 演示 1: 天气查询
    await runToolFlow("北京今天天气怎么样？");

    // 演示 2: 数学计算
    await runToolFlow("帮我算一下 (15 + 27) * 3 - 18 等于多少");

    // 演示 3: 百科搜索
    await runToolFlow("什么是 Agent？给我简单介绍一下");

    console.log(`\n${"=".repeat(60)}`);
    console.log("所有演示完成！");
    console.log("=".repeat(60));
  } catch (err) {
    const errorMsg = String(err);
    const isAuthError =
      errorMsg.includes("401") ||
      errorMsg.includes("invalid_api_key") ||
      errorMsg.includes("Authentication");
    const isToolUnsupported =
      errorMsg.includes("does not support tools") || errorMsg.includes("400");

    if (isAuthError) {
      console.error(
        `\n[提示] API 密钥无效或未配置。请编辑 ai-agent/.env 填入有效的 API 密钥。`,
      );
    } else if (isToolUnsupported) {
      console.error(`\n[提示] 当前模型 ${cfg.model} 不支持 tools API。`);
      console.error(
        `[提示] 请使用支持 function calling 的模型，如 gpt-4o-mini 或 deepseek-chat。`,
      );
    } else {
      console.error(`\n[错误] ${err}`);
    }

    // 仍然演示工具函数本身可以工作
    console.log(`\n${"=".repeat(60)}`);
    console.log("本地工具函数测试（无需 API）:");
    console.log(`  get_weather('北京') → ${getWeather("北京")}`);
    console.log(`  calculate('2+3*4') → ${calculate("2+3*4")}`);
    console.log(`  search_wiki('python') → ${searchWiki("python")}`);
    console.log("工具函数全部正常！配置好 API 密钥后即可运行完整流程。");
    console.log("=".repeat(60));
  }
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
