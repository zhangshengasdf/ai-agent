/**
 * 项目1 · 深度研究助手（Plan-and-Execute + RAG + 多工具）
 *
 * 综合实战：把 Agent 循环、规划、RAG 缝合成一个能"查资料→做笔记→出报告"的研究助手。
 *
 * 核心组件：
 *   - Plan-and-Execute：LLM 规划研究步骤 → 逐步执行
 *   - RAG：从 data/ 加载文档 → 分块 → 词频向量化 → 余弦相似度 top-k
 *   - 多工具：search_knowledge / write_note / get_summary
 *   - Trace + Cost：每步记录耗时/token/费用，最后打印 trace 树和总成本
 *   - 离线 Mock：API 不可用 → 预设计划 + 本地检索 + mock 报告，exit 0
 */

import OpenAI from "openai";
import { getConfig } from "../../../shared/config";
import { readFileSync, readdirSync } from "node:fs";
import { resolve, join } from "node:path";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

const DATA_DIR = resolve(__dirname, "..", "data");

// 模型价格（USD per 1K tokens，用于成本估算）
const INPUT_PRICE = 0.00015;
const OUTPUT_PRICE = 0.0006;

// ════════════════════════════════════════════════════════════════════
// 1. Embedding + 相似度（复用第09章 RAG 模式）
// ════════════════════════════════════════════════════════════════════

type Embedding = Record<string, number>;

function simpleEmbedding(text: string): Embedding {
  let cleaned = text.toLowerCase();
  for (const ch of "，。！？,.!?;:\"'()[]{}（）【】\n\r\t#*-`>") {
    cleaned = cleaned.split(ch).join(" ");
  }
  const words = cleaned.split(/\s+/).filter(Boolean);

  const vec: Embedding = {};
  for (const w of words) {
    if (w.length > 1 && /^[\u4e00-\u9fff]+$/.test(w)) {
      for (const c of w) {
        vec[c] = (vec[c] ?? 0) + 1;
      }
    } else {
      vec[w] = (vec[w] ?? 0) + 1;
    }
  }
  return vec;
}

function cosineSimilarity(a: Embedding, b: Embedding): number {
  let dot = 0;
  for (const k in a) {
    dot += a[k] * (b[k] ?? 0);
  }
  let normA = 0;
  for (const k in a) normA += a[k] * a[k];
  let normB = 0;
  for (const k in b) normB += b[k] * b[k];
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function chunkText(text: string, chunkSize = 200, overlap = 50): string[] {
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    const chunk = text.slice(start, start + chunkSize).trim();
    if (chunk) chunks.push(chunk);
    let step = chunkSize - overlap;
    if (step <= 0) step = chunkSize;
    start += step;
  }
  return chunks;
}

// ════════════════════════════════════════════════════════════════════
// 2. 知识库加载 + 索引构建
// ════════════════════════════════════════════════════════════════════

interface Document {
  filename: string;
  content: string;
}

function loadDocuments(dataDir: string): Document[] {
  const docs: Document[] = [];
  try {
    const entries = readdirSync(dataDir).sort();
    for (const name of entries) {
      if (name.endsWith(".md") || name.endsWith(".txt")) {
        const content = readFileSync(join(dataDir, name), "utf-8");
        docs.push({ filename: name, content });
      }
    }
  } catch {
    return [];
  }
  return docs;
}

interface IndexEntry {
  chunk: string;
  embedding: Embedding;
}

function buildIndex(
  documents: Document[],
  chunkSize = 200,
  overlap = 50,
): IndexEntry[] {
  const index: IndexEntry[] = [];
  for (const doc of documents) {
    for (const chunk of chunkText(doc.content, chunkSize, overlap)) {
      index.push({ chunk, embedding: simpleEmbedding(chunk) });
    }
  }
  return index;
}

// ════════════════════════════════════════════════════════════════════
// 3. Trace + Cost 追踪
// ════════════════════════════════════════════════════════════════════

interface TraceStep {
  name: string;
  durationMs: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  detail: string;
}

class Tracer {
  steps: TraceStep[] = [];

  record(
    name: string,
    durationMs: number,
    inputTokens = 0,
    outputTokens = 0,
    detail = "",
  ): void {
    const cost =
      (inputTokens * INPUT_PRICE + outputTokens * OUTPUT_PRICE) / 1000;
    this.steps.push({
      name,
      durationMs,
      inputTokens,
      outputTokens,
      costUsd: cost,
      detail,
    });
  }

  printTree(): void {
    console.log("\nOUT:trace: ── Trace Tree ──");
    let totalCost = 0;
    let totalMs = 0;
    for (let i = 0; i < this.steps.length; i++) {
      const s = this.steps[i];
      const prefix = i < this.steps.length - 1 ? "  ├─" : "  └─";
      let tokensInfo = "";
      if (s.inputTokens > 0 || s.outputTokens > 0) {
        tokensInfo = ` | tokens: ${s.inputTokens}+${s.outputTokens}`;
      }
      const costInfo =
        s.costUsd > 0 ? ` | cost: $${s.costUsd.toFixed(6)}` : "";
      console.log(
        `OUT:trace: ${prefix} [${i + 1}] ${s.name} ` +
          `(${s.durationMs.toFixed(0)}ms${tokensInfo}${costInfo})`,
      );
      if (s.detail) {
        const detailPrefix = i < this.steps.length - 1 ? "  │  " : "     ";
        console.log(`OUT:trace: ${detailPrefix}    ${s.detail}`);
      }
      totalCost += s.costUsd;
      totalMs += s.durationMs;
    }
    console.log("OUT:trace:");
    console.log(
      `OUT:cost: 总耗时: ${totalMs.toFixed(0)}ms | 总成本: $${totalCost.toFixed(6)}`,
    );
  }
}

// ════════════════════════════════════════════════════════════════════
// 4. 工具定义
// ════════════════════════════════════════════════════════════════════

interface PlanStep {
  step: number;
  action: string;
  query?: string;
  goal?: string;
  content?: string;
}

const MOCK_PLAN_STEPS: PlanStep[] = [
  {
    step: 1,
    action: "search_knowledge",
    query: "涌现能力 大语言模型",
    goal: "了解 LLM 涌现能力的定义和例子",
  },
  {
    step: 2,
    action: "search_knowledge",
    query: "RAG 检索增强生成 原理",
    goal: "了解 RAG 的核心流程和优势",
  },
  {
    step: 3,
    action: "search_knowledge",
    query: "多 Agent 系统 协作",
    goal: "了解多 Agent 架构和协作模式",
  },
];

// ════════════════════════════════════════════════════════════════════
// 5. LLM 调用封装（带 try/catch 降级）
// ════════════════════════════════════════════════════════════════════

interface LLMResult {
  content: string;
  toolCalls: { name: string; arguments: string }[];
}

async function llmChat(
  messages: OpenAI.ChatCompletionMessageParam[],
  tools?: OpenAI.ChatCompletionTool[],
  tracer?: Tracer,
  stepName = "llm_call",
): Promise<LLMResult> {
  const t0 = performance.now();
  try {
    const resp = await client.chat.completions.create({
      model: cfg.model,
      messages,
      ...(tools ? { tools } : {}),
    });
    const elapsed = performance.now() - t0;
    const msg = resp.choices[0].message;
    const usage = resp.usage;
    const inTok = usage?.prompt_tokens ?? 0;
    const outTok = usage?.completion_tokens ?? 0;
    tracer?.record(stepName, elapsed, inTok, outTok);

    const result: LLMResult = { content: msg.content ?? "", toolCalls: [] };
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.type !== "function") continue;
        result.toolCalls.push({
          name: tc.function.name,
          arguments: tc.function.arguments,
        });
      }
    }
    return result;
  } catch {
    const elapsed = performance.now() - t0;
    tracer?.record(stepName, elapsed, 0, 0, "(offline mock)");
    return { content: "", toolCalls: [] };
  }
}

// ════════════════════════════════════════════════════════════════════
// 6. 工具执行
// ════════════════════════════════════════════════════════════════════

function searchKnowledge(
  query: string,
  index: IndexEntry[],
  topK = 3,
): string {
  const qEmb = simpleEmbedding(query);
  const scored = index.map((entry) => ({
    chunk: entry.chunk,
    score: cosineSimilarity(qEmb, entry.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  const results = scored.slice(0, topK);
  if (results.length === 0 || results[0].score === 0) {
    return "（未找到相关内容）";
  }
  return results
    .map(
      (r, i) => `[片段${i + 1}] (相似度:${r.score.toFixed(3)}) ${r.chunk}`,
    )
    .join("\n");
}

// ════════════════════════════════════════════════════════════════════
// 7. Plan-and-Execute 主流程
// ════════════════════════════════════════════════════════════════════

async function planResearch(
  topic: string,
  tracer: Tracer,
): Promise<PlanStep[]> {
  const systemPrompt =
    "你是一个研究规划助手。给定研究主题，输出一个 JSON 数组，每个元素包含 " +
    "step(编号)、action(工具名: search_knowledge/write_note/get_summary)、" +
    "query(action 为 search_knowledge 时的检索词)、goal(这步的目标)。" +
    "输出 3-5 步，只输出 JSON，不要其他文字。";

  console.log("\nOUT:plan: ══ 研究规划 ══");
  console.log(`OUT:plan: 主题: ${topic}`);

  const resp = await llmChat(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: `研究主题: ${topic}` },
    ],
    undefined,
    tracer,
    "plan",
  );

  // 尝试解析 JSON
  const content = resp.content;
  try {
    let jsonStr = content;
    if (content.includes("```")) {
      for (const line of content.split("\n")) {
        const trimmed = line.trim();
        if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
          jsonStr = trimmed;
          break;
        }
      }
    }
    const steps = JSON.parse(jsonStr) as PlanStep[];
    if (Array.isArray(steps) && steps.length > 0) {
      console.log(`OUT:plan: LLM 生成了 ${steps.length} 个研究步骤`);
      for (const s of steps) {
        console.log(
          `OUT:plan:   步骤${s.step}: ${s.goal ?? s.query ?? ""}`,
        );
      }
      return steps;
    }
  } catch {
    // fallthrough to mock
  }

  // 降级：使用预设计划
  console.log(
    `OUT:plan: (offline) 使用预设研究计划，共 ${MOCK_PLAN_STEPS.length} 步`,
  );
  for (const s of MOCK_PLAN_STEPS) {
    console.log(`OUT:plan:   步骤${s.step}: ${s.goal ?? ""}`);
  }
  return MOCK_PLAN_STEPS;
}

async function executeResearch(
  steps: PlanStep[],
  index: IndexEntry[],
  notes: string[],
  tracer: Tracer,
): Promise<void> {
  console.log("\nOUT:step: ══ 逐步执行 ══");

  for (const stepDef of steps) {
    const stepNum = stepDef.step;
    const action = stepDef.action;
    const query = stepDef.query ?? "";
    const goal = stepDef.goal ?? "";

    console.log(`\nOUT:step: ── 步骤 ${stepNum}: ${goal} ──`);

    if (action === "search_knowledge") {
      const t0 = performance.now();
      const result = searchKnowledge(query, index);
      const elapsed = performance.now() - t0;
      tracer.record(`search_knowledge(${query})`, elapsed);
      console.log(`OUT:search: 查询: ${query}`);
      const preview =
        result.length > 200 ? result.slice(0, 200) + "..." : result;
      console.log(`OUT:search: 结果: ${preview}`);

      // 自动写笔记
      const noteContent = `[${goal}] ${result.slice(0, 150)}`;
      notes.push(noteContent);
      console.log(
        `OUT:note: 记录笔记: ${noteContent.slice(0, 80)}...`,
      );
    } else if (action === "write_note") {
      const content = stepDef.content ?? query;
      notes.push(content);
      console.log(`OUT:note: 手动笔记: ${content.slice(0, 80)}`);
    } else if (action === "get_summary") {
      const summary = notes
        .map((n, i) => `  ${i + 1}. ${n.slice(0, 60)}`)
        .join("\n");
      console.log(
        `OUT:note: 笔记汇总 (${notes.length} 条):\n${summary}`,
      );
    }
  }
}

async function generateReport(
  topic: string,
  notes: string[],
  tracer: Tracer,
): Promise<string> {
  console.log("\nOUT:report: ══ 生成报告 ══");

  const notesText = notes.map((n) => `- ${n}`).join("\n");
  const systemPrompt =
    "你是一个研究报告撰写助手。根据以下研究笔记，撰写一份简洁的研究报告摘要。";
  const userPrompt =
    `研究主题: ${topic}\n\n研究笔记:\n${notesText}\n\n请输出报告摘要（200字以内）。`;

  const resp = await llmChat(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    undefined,
    tracer,
    "report",
  );

  let report = resp.content;
  if (!report) {
    // 离线 mock：用笔记拼接
    report = `【${topic} - 研究报告摘要】\n\n`;
    for (let i = 0; i < notes.length; i++) {
      report += `${i + 1}. ${notes[i].slice(0, 80)}\n`;
    }
    report += "\n(离线模式：基于检索笔记自动生成)";
    console.log("OUT:report: (offline) 使用笔记拼接 mock 报告");
  } else {
    console.log("OUT:report: LLM 生成了研究报告");
  }

  console.log(`OUT:report: ${report.slice(0, 300)}`);
  return report;
}

// ════════════════════════════════════════════════════════════════════
// 8. 主函数
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  const topic = "大语言模型、RAG 与多 Agent 系统的融合趋势";
  const tracer = new Tracer();
  const notes: string[] = [];

  // 加载知识库
  console.log("OUT: ══ 深度研究助手 ══");
  console.log(`OUT: 研究主题: ${topic}`);

  const documents = loadDocuments(DATA_DIR);
  console.log(`OUT: 加载了 ${documents.length} 篇文档`);

  const t0 = performance.now();
  const index = buildIndex(documents);
  const buildMs = performance.now() - t0;
  tracer.record("build_index", buildMs);
  console.log(
    `OUT: 构建索引: ${index.length} 个文本块 (${buildMs.toFixed(0)}ms)`,
  );

  // 阶段1：规划
  const steps = await planResearch(topic, tracer);

  // 阶段2：执行
  await executeResearch(steps, index, notes, tracer);

  // 阶段3：报告
  await generateReport(topic, notes, tracer);

  // 打印 trace 树
  tracer.printTree();

  console.log("\nOUT: ══ 研究完成 ══");
}

main();
