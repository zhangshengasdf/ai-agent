/**
 * 第16章 可观测与调试（Tracing、日志、成本追踪）— TypeScript 实现
 *
 * 对等 Python 实现。3 大功能：
 *   1. Tracing — TraceCollector 收集 TraceEntry，记录 Agent 每步成树状结构
 *   2. 成本计算 — CostTracker 按 token×单价 计算
 *   3. Trace 可视化 — ASCII 树渲染 step→llm_call→tool_call 层级
 *
 * 概念上实现第12章 Observer 接口，但独立可运行（不 import 第13章框架）。
 *
 * 运行方式：
 *   cd ai-agent/16-observability
 *   npx tsx typescript/main.ts
 *
 * 设计：
 *   - .env 占位符 sk-REPLACE-ME → 真实 API 必失败 → 降级 mock，exit 0
 *   - 全链路 async（TS SDK 只有异步接口，T5 教训）
 *   - 不用 `as any` / `@ts-ignore`；错误用 `error instanceof Error` 收窄
 */

import OpenAI, { APIConnectionError } from "openai";
import { getConfig } from "../../shared/config";

// ═══════════════════════════════════════════════════════════════════════
// 数据结构：TraceEntry（一个 trace span = 一个原子操作）
// ═══════════════════════════════════════════════════════════════════════

/** Span 类型（discriminated union 的 type 字段）。 */
type SpanType = "step" | "llm_call" | "tool_call" | "tool_result";

/** 一个 trace span，记录 Agent 执行的一个原子操作。 */
interface TraceEntry {
  spanId: string;
  traceId: string;
  parentId: string | null;
  spanType: SpanType;
  name: string;
  startTime: number;
  endTime: number;
  inputSummary: string;
  outputSummary: string;
  promptTokens: number;
  completionTokens: number;
  metadata: Record<string, unknown>;
}

/** 计算 span 耗时（毫秒）。 */
function getDurationMs(entry: TraceEntry): number {
  if (entry.endTime <= 0) return 0;
  return Math.round((entry.endTime - entry.startTime) * 1000 * 100) / 100;
}

/** 序列化成一行 JSON 结构化日志（机器可解析）。 */
function toLogLine(entry: TraceEntry): string {
  return JSON.stringify({
    ts: new Date(entry.startTime * 1000).toISOString(),
    level: "INFO",
    trace_id: entry.traceId,
    span_id: entry.spanId,
    parent_id: entry.parentId,
    event: entry.spanType,
    name: entry.name,
    duration_ms: getDurationMs(entry),
    prompt_tokens: entry.promptTokens,
    completion_tokens: entry.completionTokens,
    input: entry.inputSummary,
    output: entry.outputSummary,
  });
}

// ═══════════════════════════════════════════════════════════════════════
// TraceCollector：收集 + 渲染 trace（实现第12章 Observer 接口）
// ═══════════════════════════════════════════════════════════════════════

/**
 * 收集 Agent 执行的每个 span，渲染成树状结构。
 *
 * 实现了第12章 Observer 接口的 5 个钩子：
 *   onStepStart / onLLMCall / onToolCall / onToolResult / onStepEnd
 *
 * 设计原则（第12章）：纯旁路观察（只读不写），不修改主流程状态。
 */
class TraceCollector {
  traceId: string;
  entries: TraceEntry[] = [];
  private currentStepId: string | null = null;
  private spanCounter = 0;

  constructor(traceId?: string) {
    this.traceId = traceId ?? `trace_${Math.random().toString(16).slice(2, 10)}`;
  }

  private nextSpanId(): string {
    this.spanCounter += 1;
    return `span_${String(this.spanCounter).padStart(3, "0")}`;
  }

  // ── 手动添加 span（离线 mock 用）──────────────────────────────────

  addSpan(params: {
    spanType: SpanType;
    name: string;
    parentId: string | null;
    startTime: number;
    endTime: number;
    inputSummary?: string;
    outputSummary?: string;
    promptTokens?: number;
    completionTokens?: number;
    metadata?: Record<string, unknown>;
  }): TraceEntry {
    const entry: TraceEntry = {
      spanId: this.nextSpanId(),
      traceId: this.traceId,
      parentId: params.parentId,
      spanType: params.spanType,
      name: params.name,
      startTime: params.startTime,
      endTime: params.endTime,
      inputSummary: params.inputSummary ?? "",
      outputSummary: params.outputSummary ?? "",
      promptTokens: params.promptTokens ?? 0,
      completionTokens: params.completionTokens ?? 0,
      metadata: params.metadata ?? {},
    };
    this.entries.push(entry);
    return entry;
  }

  // ── 渲染 ──────────────────────────────────────────────────────────

  renderTree(): string {
    const lines: string[] = [];
    lines.push(`Trace: ${this.traceId} (${this.entries.length} spans)`);
    lines.push("│");
    const roots = this.entries.filter((e) => e.parentId === null);
    roots.forEach((root, i) => {
      this.renderSpan(root, "", i === roots.length - 1, lines);
    });
    return lines.join("\n");
  }

  private renderSpan(entry: TraceEntry, prefix: string, isLast: boolean, lines: string[]): void {
    const connector = isLast ? "└── " : "├── ";
    let tokenInfo = "";
    if (entry.spanType === "llm_call") {
      tokenInfo = ` [in:${entry.promptTokens} out:${entry.completionTokens}]`;
    }
    lines.push(
      `${prefix}${connector}${entry.spanType}: ${entry.name} ` +
        `(${Math.round(getDurationMs(entry))}ms)${tokenInfo}`,
    );
    const children = this.entries.filter((e) => e.parentId === entry.spanId);
    const childPrefix = prefix + (isLast ? "    " : "│   ");
    children.forEach((child, i) => {
      this.renderSpan(child, childPrefix, i === children.length - 1, lines);
    });
  }

  toJson(): string {
    return JSON.stringify(
      this.entries.map((e) => ({
        span_id: e.spanId,
        trace_id: e.traceId,
        parent_id: e.parentId,
        span_type: e.spanType,
        name: e.name,
        start_time: e.startTime,
        end_time: e.endTime,
        duration_ms: getDurationMs(e),
        prompt_tokens: e.promptTokens,
        completion_tokens: e.completionTokens,
        input_summary: e.inputSummary,
        output_summary: e.outputSummary,
      })),
      null,
      2,
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════
// CostTracker：成本追踪（token × 单价）
// ═══════════════════════════════════════════════════════════════════════

/** 模型定价表（$/1M tokens）。 */
const PRICING: Record<string, { input: number; output: number }> = {
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
  "gpt-4o": { input: 2.5, output: 10.0 },
  "deepseek-chat": { input: 0.14, output: 0.28 },
  "qwen-plus": { input: 0.4, output: 1.2 },
};

/**
 * 累计一个 trace 内所有 LLM 调用的成本。
 *
 * 公式：单次成本 = (promptTokens × 输入价 + completionTokens × 输出价) / 1,000,000
 */
class CostTracker {
  model: string;
  totalPromptTokens = 0;
  totalCompletionTokens = 0;
  totalCostUsd = 0;
  llmCallCount = 0;

  constructor(model = "gpt-4o-mini") {
    this.model = model;
  }

  private price(): { input: number; output: number } {
    return PRICING[this.model] ?? PRICING["gpt-4o-mini"];
  }

  addLlmCall(promptTokens: number, completionTokens: number): number {
    const price = this.price();
    const inputCost = (promptTokens * price.input) / 1_000_000;
    const outputCost = (completionTokens * price.output) / 1_000_000;
    const cost = inputCost + outputCost;

    this.totalPromptTokens += promptTokens;
    this.totalCompletionTokens += completionTokens;
    this.totalCostUsd += cost;
    this.llmCallCount += 1;
    return cost;
  }

  addFromTrace(collector: TraceCollector): void {
    for (const entry of collector.entries) {
      if (entry.spanType === "llm_call") {
        this.addLlmCall(entry.promptTokens, entry.completionTokens);
      }
    }
  }

  summary(): string {
    const price = this.price();
    return (
      `模型: ${this.model} (输入 $${price.input}/1M, 输出 $${price.output}/1M)\n` +
      `LLM 调用次数: ${this.llmCallCount}\n` +
      `总输入 tokens: ${this.totalPromptTokens}\n` +
      `总输出 tokens: ${this.totalCompletionTokens}\n` +
      `总成本: $${this.totalCostUsd.toFixed(6)} ($${(this.totalCostUsd * 100).toFixed(4)} 美分)`
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 离线 mock：模拟一个 3 步 Agent 的完整 trace
// ═══════════════════════════════════════════════════════════════════════

/**
 * 构建一个模拟的 3 步 Agent trace（查北京天气→查上海天气→对比）。
 * 使用 mock 的 token 数和耗时数据，不依赖真实 API。
 */
function buildMockTrace(): TraceCollector {
  const collector = new TraceCollector("trace_mock_demo");
  const baseTime = 1_700_000_000.0;

  // ── Step 1: 调 get_weather("北京") ──────────────────────────────────
  const step1 = collector.addSpan({
    spanType: "step", name: "第1步", parentId: null,
    startTime: baseTime, endTime: baseTime + 0.52,
    inputSummary: "8 messages",
  });
  collector.addSpan({
    spanType: "llm_call", name: "LLM决策", parentId: step1.spanId,
    startTime: baseTime, endTime: baseTime + 0.4,
    inputSummary: "320 tokens, 8 messages",
    outputSummary: "tool_calls=[get_weather(city='北京')]",
    promptTokens: 320, completionTokens: 45,
  });
  const tool1 = collector.addSpan({
    spanType: "tool_call", name: "get_weather", parentId: step1.spanId,
    startTime: baseTime + 0.4, endTime: baseTime + 0.52,
    inputSummary: '{"city": "北京"}',
  });
  collector.addSpan({
    spanType: "tool_result", name: "返回结果", parentId: tool1.spanId,
    startTime: baseTime + 0.5, endTime: baseTime + 0.52,
    outputSummary: "北京今天晴 25°C",
  });

  // ── Step 2: 调 get_weather("上海") ──────────────────────────────────
  const step2Base = baseTime + 0.6;
  const step2 = collector.addSpan({
    spanType: "step", name: "第2步", parentId: null,
    startTime: step2Base, endTime: step2Base + 0.48,
    inputSummary: "10 messages (含 step1 工具结果)",
  });
  collector.addSpan({
    spanType: "llm_call", name: "LLM决策", parentId: step2.spanId,
    startTime: step2Base, endTime: step2Base + 0.38,
    inputSummary: "415 tokens, 10 messages",
    outputSummary: "tool_calls=[get_weather(city='上海')]",
    promptTokens: 415, completionTokens: 42,
  });
  const tool2 = collector.addSpan({
    spanType: "tool_call", name: "get_weather", parentId: step2.spanId,
    startTime: step2Base + 0.38, endTime: step2Base + 0.48,
    inputSummary: '{"city": "上海"}',
  });
  collector.addSpan({
    spanType: "tool_result", name: "返回结果", parentId: tool2.spanId,
    startTime: step2Base + 0.46, endTime: step2Base + 0.48,
    outputSummary: "上海今天多云 28°C",
  });

  // ── Step 3: 给最终答案（无工具调用）──────────────────────────────────
  const step3Base = baseTime + 1.2;
  const step3 = collector.addSpan({
    spanType: "step", name: "第3步", parentId: null,
    startTime: step3Base, endTime: step3Base + 0.35,
    inputSummary: "12 messages (含 step1+step2 结果)",
  });
  collector.addSpan({
    spanType: "llm_call", name: "LLM最终回答", parentId: step3.spanId,
    startTime: step3Base, endTime: step3Base + 0.35,
    inputSummary: "510 tokens, 12 messages",
    outputSummary: "上海更热(28°C > 25°C)",
    promptTokens: 510, completionTokens: 80,
  });

  return collector;
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 1：Tracing — 收集并打印 trace entry
// ═══════════════════════════════════════════════════════════════════════

async function demoTracing(): Promise<TraceCollector> {
  console.log("=".repeat(72));
  console.log("Demo 1: Tracing（链路追踪）");
  console.log("  模拟 3 步 Agent：查北京天气 → 查上海天气 → 对比");
  console.log("  每个操作记录成一个 TraceEntry（span），含类型/耗时/token");
  console.log("=".repeat(72));
  console.log();

  // try 真实 API（占位符 key 必失败）→ 降级 mock
  try {
    const cfg = getConfig();
    const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });
    await client.chat.completions.create({
      model: cfg.model,
      messages: [{ role: "user", content: "ping" }],
      max_tokens: 1,
    });
    console.log("OUT:trace:offline: 真实 API 可用，但本章用 mock 数据演示（固定 token/耗时）");
  } catch (err) {
    const errName = err instanceof APIConnectionError
      ? "APIConnectionError"
      : err instanceof Error ? err.constructor.name : "Unknown";
    console.log(`OUT:trace:offline: 真实 API 不可用（${errName}），使用 mock trace 演示`);
  }
  console.log();

  const collector = buildMockTrace();

  console.log(`收集到 ${collector.entries.length} 个 span：`);
  console.log("-".repeat(72));
  for (const entry of collector.entries) {
    const stepMarker = entry.spanType === "step" ? String(entry.metadata.step ?? "?") : "";
    console.log(
      `OUT:trace:step${stepMarker}: [${entry.spanType}] ${entry.name}`,
    );
    console.log(
      `    span_id=${entry.spanId}, parent_id=${entry.parentId}, ` +
        `耗时=${Math.round(getDurationMs(entry))}ms`,
    );
    if (entry.spanType === "llm_call") {
      console.log(`    tokens: in=${entry.promptTokens}, out=${entry.completionTokens}`);
    }
    if (entry.inputSummary) {
      console.log(`    输入: ${entry.inputSummary}`);
    }
    if (entry.outputSummary) {
      console.log(`    输出: ${entry.outputSummary}`);
    }
    console.log();
  }

  return collector;
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 2：成本计算 — 按 gpt-4o-mini 定价计算多步总成本
// ═══════════════════════════════════════════════════════════════════════

function demoCost(collector: TraceCollector): CostTracker {
  console.log("=".repeat(72));
  console.log("Demo 2: 成本计算（token × 单价）");
  console.log("  gpt-4o-mini 定价: 输入 $0.15/1M tokens, 输出 $0.60/1M tokens");
  console.log("  公式: 成本 = (in_tokens × 0.15 + out_tokens × 0.60) / 1,000,000");
  console.log("=".repeat(72));
  console.log();

  const tracker = new CostTracker("gpt-4o-mini");
  const price = PRICING["gpt-4o-mini"];

  console.log("逐笔 LLM 调用成本明细：");
  console.log("-".repeat(72));
  let callIdx = 0;
  for (const entry of collector.entries) {
    if (entry.spanType !== "llm_call") continue;
    callIdx += 1;
    const inputCost = (entry.promptTokens * price.input) / 1_000_000;
    const outputCost = (entry.completionTokens * price.output) / 1_000_000;
    const total = tracker.addLlmCall(entry.promptTokens, entry.completionTokens);
    console.log(`OUT:cost: 第${callIdx}笔 LLM 调用 (${entry.name})`);
    console.log(`    输入: ${entry.promptTokens} tokens × $${price.input}/1M = $${inputCost.toFixed(6)}`);
    console.log(`    输出: ${entry.completionTokens} tokens × $${price.output}/1M = $${outputCost.toFixed(6)}`);
    console.log(`    小计: $${total.toFixed(6)}`);
    console.log();
  }

  console.log("OUT:cost: 汇总");
  console.log("-".repeat(72));
  for (const line of tracker.summary().split("\n")) {
    console.log(`    ${line}`);
  }
  console.log();

  // 成本直觉参考
  console.log("成本直觉参考：");
  console.log(`    本次查询成本 $${tracker.totalCostUsd.toFixed(6)}`);
  const dailyQueries = 1000;
  const monthlyCost = tracker.totalCostUsd * dailyQueries * 30;
  console.log(`    若每天 ${dailyQueries} 次查询 → 月成本 ≈ $${monthlyCost.toFixed(2)}`);
  console.log("    （不追踪成本，这笔钱会悄悄花掉）");
  console.log();

  return tracker;
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 3：Trace 可视化 — ASCII 树 + 结构化日志
// ═══════════════════════════════════════════════════════════════════════

function demoVisualization(collector: TraceCollector): void {
  console.log("=".repeat(72));
  console.log("Demo 3: Trace 可视化（ASCII 树）+ 结构化日志（JSON）");
  console.log("  ASCII 树展示 step→llm_call→tool_call 层级关系");
  console.log("  结构化日志让机器可解析（可导入 ELK/Loki/Datadog）");
  console.log("=".repeat(72));
  console.log();

  console.log("OUT:viz: ASCII 树（一眼看清执行链路）");
  console.log("-".repeat(72));
  const tree = collector.renderTree();
  for (const line of tree.split("\n")) {
    console.log(`  ${line}`);
  }
  console.log();
  console.log("  解读：");
  console.log("    - 3 个顶层 step span（第1步/第2步/第3步）");
  console.log("    - 每个 step 下有 llm_call（LLM 决策）");
  console.log("    - step1/step2 下还有 tool_call→tool_result（工具调用链）");
  console.log("    - [in:X out:Y] 标注 LLM 调用的 token 数");
  console.log();

  console.log("OUT:log: 结构化日志（JSON，机器可解析）");
  console.log("-".repeat(72));
  // 打印前 3 条日志（演示格式，避免刷屏）
  const first3 = collector.entries.slice(0, 3);
  for (const entry of first3) {
    console.log(`  ${toLogLine(entry)}`);
  }
  if (collector.entries.length > 3) {
    console.log(`  ... (共 ${collector.entries.length} 条，此处展示前 3 条)`);
  }
  console.log();

  console.log("OUT:log: 完整 trace JSON 导出（可持久化到文件/数据库）");
  console.log("-".repeat(72));
  const fullJson = collector.toJson();
  const preview = fullJson.slice(0, 500);
  console.log(`  ${preview}`);
  console.log(`  ... (完整 JSON 共 ${fullJson.length} 字符)`);
  console.log();
}

// ═══════════════════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log("=".repeat(72));
  console.log("第16章 可观测与调试（Tracing、日志、成本追踪）");
  console.log("  TraceCollector / CostTracker / ASCII 树可视化 / 结构化日志");
  console.log("  （概念上实现第12章 Observer 接口，独立可运行）");
  console.log("=".repeat(72));
  console.log();

  // Demo 1: Tracing
  const collector = await demoTracing();

  // Demo 2: 成本计算
  demoCost(collector);

  // Demo 3: 可视化
  demoVisualization(collector);

  console.log("=".repeat(72));
  console.log("✓ 本章完成：3 大可观测功能演示完毕。");
  console.log("  核心收获：Agent 不再是黑盒，每一步都可追溯、可复盘、可优化。");
  console.log("  生产建议：trace 持久化（文件/DB），接 LangSmith/Langfuse 拿 Web UI。");
  console.log("=".repeat(72));
}

main().catch((err: unknown) => {
  console.error("未捕获错误:", err);
  process.exit(1);
});
