/**
 * 第13章 从零造框架 — 组装 6 大组件，运行完整 Agent（TypeScript 版）
 *
 * 本文件做三件事：
 *   1. 定义两个工具函数（getWeather / calculate）
 *   2. 组装 6 大组件（依赖注入）
 *   3. 运行 Agent：查北京/上海天气 → 算温差 → 给出最终答案
 *
 * 运行方式：
 *   cd ai-agent/13-framework-core
 *   npx tsx typescript/main.ts
 *
 * 输出标记：
 *   OUT:framework:step{N}: — Observer 记录的每步状态
 *   OUT:final: — Agent 最终答案
 */

import { getConfig } from "../../shared/config";
import {
  ConversationMemory,
  DefaultAgentRunner,
  DefaultLLMClient,
  InMemoryToolRegistry,
  LoggingObserver,
  OpenAIToolCallParser,
  type FrameworkDeps,
} from "./framework";

// ═══════════════════════════════════════════════════════════════════════
// 工具函数（注册到 ToolRegistry）
// ═══════════════════════════════════════════════════════════════════════

// 模拟天气数据库（演示用，不调真实 API）
const WEATHER_DB: Record<string, { condition: string; temp: string }> = {
  北京: { condition: "晴", temp: "25°C" },
  上海: { condition: "多云", temp: "28°C" },
  广州: { condition: "雷阵雨", temp: "32°C" },
  深圳: { condition: "晴", temp: "31°C" },
};

function getWeather(city: string): string {
  const trimmed = city.trim();
  if (!(trimmed in WEATHER_DB)) {
    const available = Object.keys(WEATHER_DB).sort().join("、");
    return `[未找到] 城市 '${trimmed}'。支持的城市：${available}`;
  }
  const w = WEATHER_DB[trimmed];
  return `${trimmed}今天${w.condition}，气温 ${w.temp}`;
}

// 安全表达式求值：只允许数字 + 四则运算符，杜绝代码注入（第06章教训）
const SAFE_BIN_OPS: Record<string, (a: number, b: number) => number> = {
  "+": (a, b) => a + b,
  "-": (a, b) => a - b,
  "*": (a, b) => a * b,
  "/": (a, b) => a / b,
  "%": (a, b) => a % b,
  "**": (a, b) => Math.pow(a, b),
};

function calculate(expression: string): string {
  const expr = expression.trim();
  if (!expr) return "[错误] 表达式为空";

  // 只允许数字、运算符、括号、空格、小数点
  const allowed = new Set("0123456789+-*/%.() ");
  for (const c of expr) {
    if (!allowed.has(c)) {
      return `[错误] 表达式含非法字符 '${c}'，只支持数字和 + - * / % ** ( )`;
    }
  }

  try {
    // 用 Function 构造器安全求值（比 eval 稍好，且输入已白名单过滤）
    // 先把 ** 替换为 Math.pow 不现实，这里用 Function 求值已过滤的表达式
    const result = Function(`"use strict"; return (${expr});`)() as number;
    if (!Number.isFinite(result)) {
      return `[错误] 计算结果不是有效数字: ${expr}`;
    }
    // 整数结果去掉小数点
    if (Number.isInteger(result)) {
      return `${expression} = ${result}`;
    }
    return `${expression} = ${result}`;
  } catch (e) {
    if (Number(expr) === Infinity || expr.includes("/0")) {
      return `[错误] 除零: ${expression}`;
    }
    return `[计算失败] ${(e as Error).message}`;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 工具 schema 定义（OpenAI function calling 格式）
// ═══════════════════════════════════════════════════════════════════════

const WEATHER_SCHEMA: Record<string, unknown> = {
  type: "object",
  properties: {
    city: {
      type: "string",
      description: "要查询天气的城市名，例如：北京、上海、广州、深圳",
    },
  },
  required: ["city"],
};

const CALCULATE_SCHEMA: Record<string, unknown> = {
  type: "object",
  properties: {
    expression: {
      type: "string",
      description: "数学表达式，支持 + - * / % **，例如：28-25、(3+4)*2",
    },
  },
  required: ["expression"],
};

// ═══════════════════════════════════════════════════════════════════════
// 组装框架（依赖注入）
// ═══════════════════════════════════════════════════════════════════════

function buildAgent(): DefaultAgentRunner {
  // 1. 工具注册表：注册 getWeather + calculate
  const tools = new InMemoryToolRegistry();
  tools.register(
    "get_weather",
    "查询指定城市的天气（天气状况 + 气温）",
    WEATHER_SCHEMA,
    (city: string) => getWeather(city),
  );
  tools.register(
    "calculate",
    "计算数学表达式（支持 + - * / % **）",
    CALCULATE_SCHEMA,
    (expression: string) => calculate(expression),
  );

  // 2. LLM 客户端：注入配置（provider/baseUrl/apiKey/model）
  const cfg = getConfig();
  const llm = new DefaultLLMClient(cfg);

  // 3. 记忆：带 system prompt 定义 Agent 人格
  const memory = new ConversationMemory(
    "你是任务助手 Agent。你可以查询天气、做数学计算。" +
      "请根据用户需求，逐步调用工具完成任务，最后给出简洁的结论。",
  );

  // 4. 解析器：解析 OpenAI tool_calls 格式
  const parser = new OpenAIToolCallParser();

  // 5. 观察者：打印每步日志（OUT:framework:step{N}: 前缀）
  const observer = new LoggingObserver();

  // 6. 循环引擎：通过构造函数注入上面 5 个组件
  const deps: FrameworkDeps = {
    llm,
    tools,
    memory,
    parser,
    observer,
  };
  return new DefaultAgentRunner(deps);
}

// ═══════════════════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log("=".repeat(72));
  console.log("第13章 从零造框架 — 实现 6 大核心组件");
  console.log("把第12章的接口图纸，浇筑成能跑的 mini Agent 框架");
  console.log("=".repeat(72));
  console.log();

  // ── Demo 1：展示 6 大组件的组装 ──
  console.log("▎ Demo 1: 组装 6 大组件（依赖注入）");
  console.log("-".repeat(72));
  console.log("  1. InMemoryToolRegistry  ← 注册 get_weather + calculate");
  console.log("  2. DefaultLLMClient      ← 包装 OpenAI SDK + 离线 mock 降级");
  console.log("  3. ConversationMemory    ← 对话缓冲（带 system prompt）");
  console.log("  4. OpenAIToolCallParser  ← 解析 tool_calls → [{name, args, id}]");
  console.log("  5. LoggingObserver       ← 纯旁路日志（OUT:framework:step{N}:）");
  console.log("  6. DefaultAgentRunner    ← 循环引擎（协调上面 5 个组件）");
  console.log();

  const runner = buildAgent();
  console.log("  ✓ 框架组装完成，开始运行 Agent 循环...");
  console.log();

  // ── Demo 2：运行 Agent ──
  console.log("▎ Demo 2: 运行 Agent（查天气 + 算温差）");
  console.log("-".repeat(72));

  const task = "帮我查一下北京和上海的天气，然后算一下两地温差。";
  console.log(`  任务: ${task}`);
  console.log();

  const answer = await runner.run(task, 10);

  console.log();
  console.log(`OUT:final: ${answer}`);
  console.log();

  // ── Demo 3：验证工具独立可用 ──
  console.log("▎ Demo 3: 工具独立验证（证明组件可单独使用）");
  console.log("-".repeat(72));
  console.log(`  getWeather('北京')  → ${getWeather("北京")}`);
  console.log(`  getWeather('上海')  → ${getWeather("上海")}`);
  console.log(`  calculate('28-25')  → ${calculate("28-25")}`);
  console.log(`  calculate('(3+4)*2') → ${calculate("(3+4)*2")}`);
  console.log(`  getWeather('未知')  → ${getWeather("未知")}`);
  console.log(`  calculate('1/0')    → ${calculate("1/0")}`);
  console.log();

  console.log("=".repeat(72));
  console.log("✓ 本章完成：mini Agent 框架已跑通。");
  console.log("  第14章会扩展高级特性（流式输出 / 并行工具 / 摘要记忆）。");
  console.log("=".repeat(72));
}

main().catch((err: unknown) => {
  console.error("运行失败:", err);
  process.exit(1);
});
