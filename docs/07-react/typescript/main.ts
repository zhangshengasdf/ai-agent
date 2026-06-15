/**
 * 第07章 ReAct 模式（Reasoning + Acting）
 *
 * 本章对比两种 Agent 推理范式：
 *
 *   显式 ReAct（经典文本格式）：
 *     - 模型输出 "Thought: ... Action: tool_name[args]" 文本格式
 *     - 手动正则解析提取 Thought 和 Action
 *     - 执行工具，把结果以 "Observation: ..." 追加到 prompt
 *     - 循环直到模型输出 "Final Answer: ..."
 *     - 推理过程完全可见，任何模型都能用（不需要 tools API）
 *
 *   隐式 ReAct（现代 tools API）：
 *     - 用 tools API，模型在内部推理后输出 tool_calls
 *     - 开发者看不到推理过程（黑盒）
 *     - 结构化输出，更稳定但可调试性低
 *
 * 离线 mock 设计：
 *   .env 的 OPENAI_API_KEY=sk-REPLACE-ME 是占位符，真实 API 调用必失败。
 *   所有 demo 先 try 真实 API（失败时降级），然后用离线 mock 100% 可靠地
 *   演示完整 ReAct 流程，保证 exit code 0。
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 工具实现（复用第03/04章的 mock 工具，保持一致）
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
    北京: "北京是中国的首都，著名景点有故宫、长城。",
    上海: "上海是中国最大的城市，著名景点有外滩、东方明珠。",
  };
  const queryLower = query.toLowerCase();
  for (const [key, value] of Object.entries(knowledge)) {
    if (queryLower.includes(key)) {
      return value;
    }
  }
  return `未找到与'${query}'相关的百科条目。`;
}

// 工具名 → 函数的映射（dispatch 模式）
// 注意：用 rest 参数 (...args: string[]) 而非单参数，这样 Object.values(args) 展开调用才类型安全
const TOOL_FUNCTIONS: Record<string, (...args: string[]) => string> = {
  get_weather: (city: string) => getWeather(city),
  calculate: (expr: string) => calculate(expr),
  search_wiki: (query: string) => searchWiki(query),
};

// ⚠️ 必须有上限！
const MAX_STEPS = 10;

// ════════════════════════════════════════════════════════════════════
// 工具定义（JSON Schema，用于隐式 ReAct）
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
          query: { type: "string", description: "搜索关键词" },
        },
        required: ["query"],
      },
    },
  },
];

// ════════════════════════════════════════════════════════════════════
// 显式 ReAct：ReAct System Prompt（含格式约束 + Few-shot 示例）
// ════════════════════════════════════════════════════════════════════

const REACT_SYSTEM_PROMPT = `\
你是一个任务助手 Agent。请严格使用以下 ReAct 格式回答问题。

可用工具（用 Action: 工具名[参数] 调用）：
- get_weather[城市名]: 查询城市天气，如 get_weather[北京]
- calculate[数学表达式]: 数学计算，如 calculate[28-25]
- search_wiki[关键词]: 搜索百科，如 search_wiki[北京]

格式规则（必须严格遵守）：
Thought: 你的推理过程（1-2 句话，说明你现在知道什么、下一步该干嘛）
Action: 工具名[参数]

（系统会自动追加 Observation，你不需要自己写 Observation）

当你通过工具调用收集到足够信息后，用以下格式给出最终答案：
Thought: 信息已足够，我现在知道答案。
Final Answer: 你的最终回答

示例：
问题: 上海和深圳哪个温度更高？
Thought: 我需要分别查两个城市的温度。先查上海。
Action: get_weather[上海]

Observation: 上海今天多云, 28°C, 湿度 65%, 东南风 3 级
Thought: 上海是 28°C。现在查深圳的温度。
Action: get_weather[深圳]

Observation: 深圳今天小雨, 30°C, 湿度 80%, 南风 2 级
Thought: 上海 28°C，深圳 30°C。深圳温度更高，高了 2°C。我现在知道答案了。
Final Answer: 深圳温度更高（30°C > 上海 28°C），高 2°C。

现在请回答以下问题：`;

// ════════════════════════════════════════════════════════════════════
// 显式 ReAct：解析逻辑（正则提取 Thought / Action / Final Answer）
// ════════════════════════════════════════════════════════════════════

type ParsedReAct =
  | { type: "final_answer"; thought: string; answer: string }
  | { type: "action"; thought: string; tool: string; args: string }
  | { type: "parse_error"; raw: string };

function parseReactOutput(text: string): ParsedReAct {
  /**解析模型的 ReAct 文本输出。*/
  // ── 情况 1：模型输出 Final Answer → 任务完成 ──
  if (text.includes("Final Answer:")) {
    const answer = text.split("Final Answer:")[1].trim();
    let thought = "";
    if (text.includes("Thought:")) {
      const thoughtMatch = text.match(
        /Thought:\s*([\s\S]*?)\nFinal Answer:/,
      );
      if (thoughtMatch) {
        thought = thoughtMatch[1].trim();
      }
    }
    return { type: "final_answer", thought, answer };
  }

  // ── 情况 2：模型输出 Thought + Action → 需要执行工具 ──
  // 注意：用 `s` flag (dotAll) 让 . 匹配换行符
  const match = text.match(/Thought:\s*(.*?)\nAction:\s*(\w+)\[(.*?)\]/s);
  if (match) {
    return {
      type: "action",
      thought: match[1].trim(),
      tool: match[2],
      args: match[3],
    };
  }

  // ── 情况 3：格式错误 ──
  return { type: "parse_error", raw: text };
}

// ════════════════════════════════════════════════════════════════════
// 显式 ReAct：主循环（文本格式，不使用 tools API）
// ════════════════════════════════════════════════════════════════════

async function explicitReactLoop(userMessage: string): Promise<string> {
  /**显式 ReAct 循环：模型输出 Thought/Action 文本，手动解析执行。*/
  let prompt = `${REACT_SYSTEM_PROMPT}\n问题: ${userMessage}\n`;

  console.log(`\n${"=".repeat(60)}`);
  console.log(`显式 ReAct 任务: ${userMessage}`);
  console.log("=".repeat(60));

  for (let step = 1; step <= MAX_STEPS; step++) {
    // ── Reason：让模型输出 Thought/Action 文本（注意：不传 tools）──
    const response = await client.chat.completions.create({
      model: cfg.model,
      messages: [{ role: "user", content: prompt }],
      stop: ["Observation:"],
    });
    const modelText = response.choices[0].message.content ?? "";

    // ── 解析模型输出 ──
    const parsed = parseReactOutput(modelText);

    if (parsed.type === "final_answer") {
      console.log(`OUT:explicit:step${step}: Thought: ${parsed.thought.slice(0, 80)}`);
      console.log(`OUT:explicit:step${step}: ✓ 检测到 Final Answer，终止循环`);
      console.log(`OUT:explicit:step${step}: 最终答案: ${parsed.answer.slice(0, 120)}`);
      return parsed.answer;
    }

    if (parsed.type === "action") {
      const { thought, tool: toolName, args } = parsed;
      console.log(`OUT:explicit:step${step}: Thought: ${thought.slice(0, 80)}`);
      console.log(`OUT:explicit:step${step}: Action: ${toolName}[${args}]`);

      // 执行工具
      const func = TOOL_FUNCTIONS[toolName];
      let result: string;
      if (!func) {
        result = `错误：未知工具 '${toolName}'`;
      } else {
        try {
          result = func(args);
        } catch (e) {
          result = `工具执行错误：${e}`;
        }
      }

      console.log(`OUT:explicit:step${step}: Observation: ${result.slice(0, 80)}`);

      // 关键：把 Observation 追加到 prompt，让模型下一步看到结果
      prompt += `Thought: ${thought}\nAction: ${toolName}[${args}]\n\nObservation: ${result}\n`;
      continue;
    }

    // 格式错误：提醒模型重新格式化
    console.log(`OUT:explicit:step${step}: ⚠️ 格式解析失败，提醒模型重新格式化`);
    prompt += `\n（格式错误。请用 Thought:/Action:/Final Answer: 格式重新回答。）\n${modelText}\n`;
  }

  console.log(`OUT:explicit: ⚠️ 达到最大步数 ${MAX_STEPS}，强制停止！`);
  return "(已达到最大步数)";
}

// ════════════════════════════════════════════════════════════════════
// 隐式 ReAct：tools API 循环（复用第04章模式，对比用）
// ════════════════════════════════════════════════════════════════════

async function implicitReactLoop(userMessage: string): Promise<string> {
  /**隐式 ReAct 循环：用 tools API，模型内部推理后输出 tool_calls。*/
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content:
        "你是一个任务助手 Agent。你可以查天气、做计算、搜百科。" +
        "面对复杂任务，请一步步调用工具收集信息，最后给出综合回答。",
    },
    { role: "user", content: userMessage },
  ];

  console.log(`\n${"=".repeat(60)}`);
  console.log(`隐式 ReAct 任务: ${userMessage}`);
  console.log("=".repeat(60));
  console.log("[说明] 模型在内部推理，我们只能看到 tool_calls（推理过程不可见）");

  for (let step = 1; step <= MAX_STEPS; step++) {
    // ── Reason：模型内部推理后输出 tool_calls（或最终回答）──
    const response = await client.chat.completions.create({
      model: cfg.model,
      messages,
      tools,
      tool_choice: "auto",
    });
    const assistantMsg = response.choices[0].message;

    // 终止条件：模型不调工具 = 任务完成
    if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
      const answer = assistantMsg.content ?? "(空回答)";
      console.log(`OUT:implicit:step${step}: ✓ 模型给出最终回答（无 tool_calls）`);
      console.log(`OUT:implicit:step${step}: 最终答案: ${answer.slice(0, 120)}`);
      return answer;
    }

    // 模型决定调工具
    messages.push(assistantMsg);
    const toolNames = assistantMsg.tool_calls
      .filter((t) => t.type === "function")
      .map((t) => t.function.name);
    console.log(`OUT:implicit:step${step}: tool_calls: ${toolNames.join(", ")}`);
    console.log(`OUT:implicit:step${step}: （推理过程不可见——模型在内部完成决策）`);

    for (const tc of assistantMsg.tool_calls) {
      // ⚠️ discriminated union：访问 .function 前必须检查 type
      if (tc.type !== "function") continue;
      const funcName = tc.function.name;
      let args: Record<string, string> = {};
      try {
        args = JSON.parse(tc.function.arguments);
      } catch {
        args = {};
      }

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

      console.log(`OUT:implicit:step${step}: 执行 ${funcName}(${JSON.stringify(args)}) → ${result.slice(0, 60)}`);
      messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: result,
      });
    }
  }

  console.log(`OUT:implicit: ⚠️ 达到最大步数 ${MAX_STEPS}，强制停止！`);
  return "(已达到最大步数)";
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：显式 ReAct
// ════════════════════════════════════════════════════════════════════

function demoExplicitReactOffline(): string {
  /**离线演示显式 ReAct：预设模型输出，展示完整解析+执行流程。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("离线演示：显式 ReAct（Thought → Action → Observation）");
  console.log("=".repeat(60));
  console.log("[说明] 预设模型输出，演示完整的 ReAct 文本解析 + 工具执行流程");

  const question = "北京和上海哪个温度更高？";
  console.log(`问题: ${question}\n`);

  // 预设模型每一步的文本输出
  const mockModelOutputs: string[] = [
    "Thought: 我需要分别查北京和上海的温度才能比较。先查北京。\n" +
      "Action: get_weather[北京]",
    "Thought: 北京是 25°C。现在查上海的温度。\n" +
      "Action: get_weather[上海]",
    "Thought: 北京 25°C，上海 28°C。上海温度更高，高了 3°C。我现在知道答案了。\n" +
      "Final Answer: 上海温度更高（28°C > 北京 25°C），高 3°C。",
  ];

  let prompt = `${REACT_SYSTEM_PROMPT}\n问题: ${question}\n`;
  let finalAnswer = "(无)";
  let step = 0;

  for (const modelText of mockModelOutputs) {
    step++;
    console.log(`--- Step ${step} ---`);

    console.log(`OUT:explicit:step${step}: 模型原始输出:`);
    for (const line of modelText.split("\n")) {
      console.log(`  │ ${line}`);
    }

    const parsed = parseReactOutput(modelText);

    if (parsed.type === "final_answer") {
      finalAnswer = parsed.answer;
      console.log(`OUT:explicit:step${step}: 解析结果: final_answer`);
      console.log(`OUT:explicit:step${step}: Thought: ${parsed.thought}`);
      console.log(`OUT:explicit:step${step}: ✓ 终止条件触发，返回最终答案`);
      console.log(`OUT:explicit:step${step}: 最终答案: ${finalAnswer}`);
      break;
    }

    if (parsed.type === "action") {
      console.log(`OUT:explicit:step${step}: 解析结果: action`);
      console.log(`OUT:explicit:step${step}: Thought: ${parsed.thought}`);
      console.log(`OUT:explicit:step${step}: Action: ${parsed.tool}[${parsed.args}]`);

      const func = TOOL_FUNCTIONS[parsed.tool];
      const observation = func(parsed.args);
      console.log(`OUT:explicit:step${step}: Observation: ${observation}`);

      prompt += `Thought: ${parsed.thought}\nAction: ${parsed.tool}[${parsed.args}]\n\nObservation: ${observation}\n`;
      console.log(`OUT:explicit:step${step}: 已将 Observation 追加到 prompt，继续下一步\n`);
    }
  }

  console.log(`\nOUT:explicit: ✓ 显式 ReAct 完成，共 ${step} 步。`);
  console.log(`OUT:explicit: 推理过程完全可见（每步的 Thought 都在输出里）。`);
  return finalAnswer;
}

// ════════════════════════════════════════════════════════════════════
// 离线 mock：隐式 ReAct
// ════════════════════════════════════════════════════════════════════

interface MockToolCall {
  name: string;
  arguments: Record<string, string>;
}

function demoImplicitReactOffline(): string {
  /**离线演示隐式 ReAct：预设 tool_calls 序列，展示 tools API 推理流程。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("离线演示：隐式 ReAct（tools API，推理过程不可见）");
  console.log("=".repeat(60));
  console.log("[说明] 预设 tool_calls 序列，演示隐式推理流程");

  const question = "北京和上海哪个温度更高？";
  console.log(`问题: ${question}\n`);

  // 预设 tool_calls 序列（null = 无 tool_calls = 最终回答）
  const mockToolCallsSequence: (MockToolCall[] | null)[] = [
    [{ name: "get_weather", arguments: { city: "北京" } }],
    [{ name: "get_weather", arguments: { city: "上海" } }],
    null,
  ];

  let finalAnswer = "(无)";
  let step = 0;

  for (const toolCalls of mockToolCallsSequence) {
    step++;
    console.log(`--- Step ${step} ---`);

    if (toolCalls === null) {
      finalAnswer = "上海温度更高（28°C > 北京 25°C），高 3°C。";
      console.log(`OUT:implicit:step${step}: response.choices[0].message.tool_calls = null`);
      console.log(`OUT:implicit:step${step}: ✓ 终止条件触发（无 tool_calls = 任务完成）`);
      console.log(`OUT:implicit:step${step}: 最终答案: ${finalAnswer}`);
      break;
    }

    for (const tc of toolCalls) {
      const funcName = tc.name;
      const args = tc.arguments;
      console.log(`OUT:implicit:step${step}: tool_calls: [${funcName}(${JSON.stringify(args)})]`);
      console.log(`OUT:implicit:step${step}: （推理不可见——模型在内部决定先查这个城市）`);

      const result = TOOL_FUNCTIONS[funcName](...Object.values(args));
      console.log(`OUT:implicit:step${step}: 执行结果: ${result}`);
      console.log(`OUT:implicit:step${step}: 已将结果以 role=tool 追加到 messages\n`);
    }
  }

  console.log(`\nOUT:implicit: ✓ 隐式 ReAct 完成，共 ${step} 步。`);
  console.log(`OUT:implicit: 推理过程不可见（只有 tool_calls，没有 Thought）。`);
  return finalAnswer;
}

// ════════════════════════════════════════════════════════════════════
// 对比输出：显式 ReAct vs 隐式 ReAct 并排对比
// ════════════════════════════════════════════════════════════════════

function demoComparison(): void {
  /**并排对比显式 ReAct 和隐式 ReAct 的核心差异。*/
  console.log(`\n${"=".repeat(60)}`);
  console.log("对比：显式 ReAct vs 隐式 ReAct");
  console.log("=".repeat(60));

  const comparisons: Array<[string, string, string]> = [
    ["推理可见性", "✓ Thought 文本完全可见", "✗ 模型内部推理（黑盒）"],
    ["工具调用格式", "文本 Action: name[args]", "结构化 tool_calls (JSON)"],
    ["解析方式", "正则 re.search 手动解析", "SDK 自动解析（无正则）"],
    ["格式健壮性", "⚠️ 模型可能不遵循格式", "✓ JSON Schema 约束，更稳定"],
    ["可调试性", "✓ 高（看 Thought 排查逻辑）", "⚠️ 低（推理不可见）"],
    ["Token 成本", "⚠️ Thought 占额外 token", "✓ 无 Thought 开销"],
    ["模型兼容性", "✓ 任何模型（纯文本）", "⚠️ 需支持 tools API"],
    ["API 参数", "不传 tools（纯文本补全）", "传 tools + tool_choice"],
    ["适合场景", "教学/调试/小模型", "生产/大型应用"],
  ];

  const header = `维度             │ 显式 ReAct                   │ 隐式 ReAct`;
  console.log(`OUT:compare: ${header}`);
  console.log(`OUT:compare: ${"─".repeat(14)}─┼─${"─".repeat(28)}─┼─${"─".repeat(28)}`);
  for (const [dim, explicit, implicit] of comparisons) {
    console.log(`OUT:compare: ${dim.padEnd(14)} │ ${explicit.padEnd(28)} │ ${implicit}`);
  }

  console.log(`\nOUT:compare: 核心洞察：`);
  console.log(`OUT:compare: • 显式 ReAct = 推理透明 + 格式脆弱（需 Prompt 工程）`);
  console.log(`OUT:compare: • 隐式 ReAct = 推理黑盒 + 结构稳定（需 tools API 支持）`);
  console.log(`OUT:compare: • 两者共享同一个循环骨架（for step in range(MAX_STEPS)）`);
  console.log(`OUT:compare: • 现代框架默认用隐式 ReAct，但理解显式版能看透底层逻辑`);
}

// ════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] 工具数量: ${tools.length}`);
  console.log(`[config] MAX_STEPS=${MAX_STEPS}`);
  console.log(`[config] 显式 ReAct: 纯文本格式（不传 tools 参数）`);
  console.log(`[config] 隐式 ReAct: tools API（传 tools 参数）`);

  let apiOk = true;
  const question = "北京和上海哪个温度更高？";

  try {
    // ── Demo 1: 显式 ReAct（真实 API）──
    console.log(`\n${"#".repeat(60)}`);
    console.log("# Demo 1: 显式 ReAct（Thought → Action → Observation）");
    console.log("#".repeat(60));
    await explicitReactLoop(question);

    // ── Demo 2: 隐式 ReAct（真实 API）──
    console.log(`\n${"#".repeat(60)}`);
    console.log("# Demo 2: 隐式 ReAct（tools API）");
    console.log("#".repeat(60));
    await implicitReactLoop(question);
  } catch (err) {
    apiOk = false;
    const errorMsg = String(err);
    const isAuthError =
      errorMsg.includes("401") ||
      errorMsg.includes("invalid_api_key") ||
      errorMsg.includes("Authentication") ||
      errorMsg.includes("sk-REPLACE-ME");

    console.log(`\n[提示] 真实 API 调用失败（${err instanceof Error ? err.constructor.name : "Error"}）。`);
    if (isAuthError) {
      console.log(`[提示] 原因：API 密钥无效或为占位符。请编辑 ai-agent/.env 填入有效密钥。`);
      console.log(`[提示] 当前 provider=${cfg.provider}，需要对应的 API 密钥。`);
    } else {
      console.log(`[提示] 原因：${err}`);
    }
    console.log(`[提示] 已自动降级为离线 mock 演示，ReAct 逻辑不受影响。\n`);
  }

  // ── 无论 API 是否可用，都演示离线 mock（保证学习体验完整）────
  demoExplicitReactOffline();
  demoImplicitReactOffline();
  demoComparison();

  console.log(`\n${"=".repeat(60)}`);
  if (apiOk) {
    console.log("所有演示完成！（含真实 API + 离线 mock + 对比）");
  } else {
    console.log("离线演示完成！（真实 API 未配置，但 ReAct 逻辑已完整展示）");
  }
  console.log(`💡 核心要点：显式 ReAct 推理可见，隐式 ReAct 结构稳定。`);
  console.log(`💡 两者共享同一个循环骨架，区别只在推理如何表达。`);
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
