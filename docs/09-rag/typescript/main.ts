/**
 * 第09章 RAG 检索（Retrieval-Augmented Generation）
 *
 * 本章实现两种 RAG：
 *   1. 基础 RAG pipeline：固定管道 检索→注入→回答（无条件检索）
 *   2. Agentic RAG：把检索作为工具，Agent 自主决定是否检索、检索什么
 *
 * 核心组件（复用第05章 VectorMemory 的模式）：
 *   - simpleEmbedding：纯 TS 词频向量（中文拆单字），不依赖 embedding API
 *   - cosineSimilarity：纯 TS 余弦相似度（稀疏 Record<string, number> 向量）
 *   - chunkText：文档分块（chunk_size + overlap 滑窗）
 *
 * 离线设计：
 *   - 基础 RAG：embedding 纯 TS，回答阶段 try API 失败降级 mock
 *   - Agentic RAG：预设 mock 决策序列演示 Agent 自主检索
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";
import { readFileSync, readdirSync } from "node:fs";
import { resolve, join } from "node:path";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

const DATA_DIR = resolve(__dirname, "..", "data");

// ════════════════════════════════════════════════════════════════════
// 1. Embedding + 相似度（复用第05章 VectorMemory 模式）
// ════════════════════════════════════════════════════════════════════

type Embedding = Record<string, number>;

function simpleEmbedding(text: string): Embedding {
  // 词频向量模拟 embedding：分词 + 中文拆单字
  let cleaned = text.toLowerCase();
  for (const ch of "，。！？,.!?;:\"'()[]{}（）【】\n\r\t#*-`>") {
    cleaned = cleaned.split(ch).join(" ");
  }
  const words = cleaned.split(/\s+/).filter(Boolean);

  const vec: Embedding = {};
  for (const w of words) {
    if (w.length > 1 && /^[\u4e00-\u9fff]+$/.test(w)) {
      // 连续中文字符 → 拆成单字，模拟基础分词
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

// ════════════════════════════════════════════════════════════════════
// 2. 文档分块（Chunking）
// ════════════════════════════════════════════════════════════════════

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
// 3. 知识库加载 + 索引构建
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
    const chunks = chunkText(doc.content, chunkSize, overlap);
    for (const chunk of chunks) {
      index.push({ chunk, embedding: simpleEmbedding(chunk) });
    }
  }
  return index;
}

interface RetrieveResult {
  chunk: string;
  score: number;
}

function retrieve(
  query: string,
  index: IndexEntry[],
  topK = 3,
): RetrieveResult[] {
  if (index.length === 0) return [];
  const qEmb = simpleEmbedding(query);
  const scored: RetrieveResult[] = index.map((entry) => ({
    chunk: entry.chunk,
    score: cosineSimilarity(qEmb, entry.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

// ════════════════════════════════════════════════════════════════════
// 4. 基础 RAG：检索 → 注入 → 回答
// ════════════════════════════════════════════════════════════════════

async function ragAnswer(
  query: string,
  index: IndexEntry[],
  topK = 3,
): Promise<string> {
  const results = retrieve(query, index, topK);
  console.log(`OUT:retrieve: 检索到 ${results.length} 个相关分块:`);
  results.forEach((r, i) => {
    const preview = r.chunk.slice(0, 60).replace(/\n/g, " ");
    console.log(`OUT:retrieve:   [${i + 1}] score=${r.score.toFixed(3)} | ${preview}...`);
  });

  const contextParts = results.map(
    (r, i) => `[片段${i + 1}] ${r.chunk}`,
  );
  const context = contextParts.join("\n\n");

  const prompt =
    `请根据以下背景知识回答问题。\n\n` +
    `背景知识：\n${context}\n\n` +
    `问题：${query}`;

  return generateAnswer(query, context, prompt);
}

async function generateAnswer(
  query: string,
  context: string,
  prompt: string,
): Promise<string> {
  try {
    const resp = await client.chat.completions.create({
      model: cfg.model,
      messages: [
        {
          role: "system",
          content:
            "你是任务助手 Agent，请基于提供的背景知识回答用户问题。" +
            "如果背景知识中没有答案，请如实说明。",
        },
        { role: "user", content: prompt },
      ],
      max_tokens: 300,
    });
    return resp.choices[0].message.content ?? "(空回答)";
  } catch (e) {
    const errName = e instanceof Error ? e.constructor.name : "Error";
    console.log(
      `OUT:answer: [离线模式] API 不可用（${errName}），降级 mock 回答。`,
    );
    return mockAnswer(query, context);
  }
}

function mockAnswer(query: string, context: string): string {
  const snippets = context.split("[片段");
  const relevant: string[] = [];
  for (let i = 1; i < snippets.length; i++) {
    const lines = snippets[i].trim();
    if (lines) {
      const idx = lines.indexOf("]");
      const text = idx >= 0 ? lines.slice(idx + 1).trim() : lines;
      relevant.push(text.slice(0, 80));
    }
  }
  const summary = relevant.join(" ").slice(0, 200);
  return `[基于检索结果的 mock 回答] 关于「${query}」：根据知识库，${summary}...`;
}

// ════════════════════════════════════════════════════════════════════
// 5. Agentic RAG：检索作为工具，Agent 自主决定
// ════════════════════════════════════════════════════════════════════

const RAG_TOOLS: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search_knowledge_base",
      description:
        "在知识库中检索与查询相关的文档片段。" +
        "当用户问与知识库内容（如 Python、AI Agent、LLM）相关的问题时使用。" +
        "对于常识问题（如数学计算）不需要使用此工具。",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "检索关键词，如 'Python 特点' 或 'Agent 概念'",
          },
        },
        required: ["query"],
      },
    },
  },
];

const MAX_STEPS = 6;

function searchKnowledgeBase(query: string, index: IndexEntry[]): string {
  const results = retrieve(query, index, 3);
  if (results.length === 0) return "知识库为空，未找到相关信息。";
  const parts = results.map(
    (r, i) => `[片段${i + 1} 相关度=${r.score.toFixed(3)}] ${r.chunk}`,
  );
  return parts.join("\n\n");
}

async function agenticRag(
  query: string,
  index: IndexEntry[],
): Promise<string> {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content:
        "你是任务助手 Agent。你有一个工具 search_knowledge_base 可以检索知识库。" +
        "当用户问题与知识库内容（Python、AI Agent、LLM 等）相关时，调用工具检索。" +
        "对于常识问题（如数学计算、打招呼），直接回答，不需要检索。" +
        "检索到信息后，基于检索结果给出准确回答。",
    },
    { role: "user", content: query },
  ];

  for (let step = 1; step <= MAX_STEPS; step++) {
    console.log(`OUT:agentic:step${step}: 思考中...`);
    let response: OpenAI.Chat.Completions.ChatCompletion;
    try {
      response = await client.chat.completions.create({
        model: cfg.model,
        messages,
        tools: RAG_TOOLS,
        tool_choice: "auto",
      });
    } catch (e) {
      const errName = e instanceof Error ? e.constructor.name : "Error";
      console.log(
        `OUT:agentic: [离线模式] API 不可用（${errName}），降级 mock 决策演示。`,
      );
      return agenticRagMock(query, index);
    }

    const assistantMsg = response.choices[0].message;

    if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
      const answer = assistantMsg.content ?? "(空回答)";
      console.log(
        `OUT:agentic:step${step}: ✓ Agent 决定直接回答（未调用检索工具）`,
      );
      const preview = answer.length > 120 ? answer.slice(0, 120) : answer;
      console.log(`OUT:agentic:step${step}: 回答: ${preview}`);
      return answer;
    }

    messages.push(assistantMsg);
    for (const tc of assistantMsg.tool_calls) {
      if (tc.type !== "function") continue;
      let args: Record<string, string> = {};
      try {
        args = JSON.parse(tc.function.arguments);
      } catch {
        args = {};
      }

      const searchQuery = args.query ?? query;
      console.log(
        `OUT:agentic:step${step}: → 调用 ${tc.function.name}(query='${searchQuery}')`,
      );
      const result = searchKnowledgeBase(searchQuery, index);
      const preview = result.slice(0, 80).replace(/\n/g, " ");
      console.log(`OUT:agentic:step${step}: ← 检索结果: ${preview}...`);

      messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: String(result),
      });
    }
  }

  console.log(`OUT:agentic: ⚠️ 达到最大步数 ${MAX_STEPS}，停止。`);
  return "(已达到最大步数)";
}

function agenticRagMock(query: string, index: IndexEntry[]): string {
  console.log(`OUT:agentic: [离线 mock] 用规则模拟 Agent 的检索决策。`);

  const kbKeywords = [
    "python",
    "agent",
    "llm",
    "大模型",
    "智能体",
    "语言模型",
    "编程",
  ];
  const queryLower = query.toLowerCase();
  const needsRetrieval = kbKeywords.some((kw) => queryLower.includes(kw));

  if (needsRetrieval) {
    console.log(`OUT:agentic:step1: 思考中...`);
    console.log(
      `OUT:agentic:step1: Agent 判断：问题与知识库相关 → 调用检索工具`,
    );
    console.log(`OUT:agentic:step1: → search_knowledge_base(query='${query}')`);
    const result = searchKnowledgeBase(query, index);
    const preview = result.slice(0, 80).replace(/\n/g, " ");
    console.log(`OUT:agentic:step1: ← 检索结果: ${preview}...`);

    console.log(`OUT:agentic:step2: 思考中...`);
    console.log(
      `OUT:agentic:step2: Agent 判断：信息足够 → 基于检索结果回答（不再检索）`,
    );
    const answer = mockAnswer(query, result);
    const answerPreview = answer.length > 120 ? answer.slice(0, 120) : answer;
    console.log(`OUT:agentic:step2: 回答: ${answerPreview}`);
    return answer;
  }

  console.log(`OUT:agentic:step1: 思考中...`);
  console.log(
    `OUT:agentic:step1: Agent 判断：这是常识问题 → 不检索，直接回答`,
  );
  const answer = directAnswer(query);
  const preview = answer.length > 120 ? answer.slice(0, 120) : answer;
  console.log(`OUT:agentic:step1: 回答: ${preview}`);
  return answer;
}

function directAnswer(query: string): string {
  const mathMatch = query.match(/([\d\s+\-*/().]+)/);
  if (mathMatch) {
    const expr = mathMatch[1].trim();
    if (expr && /[+\-*/]/.test(expr)) {
      try {
        const result = eval(expr);
        return `${expr} = ${result}`;
      } catch {
        // fall through
      }
    }
  }
  if (query.includes("你好") || query.toLowerCase().includes("hello")) {
    return "你好！我是任务助手 Agent，有什么可以帮你的？";
  }
  return `这是常识问题，我直接回答：${query}`;
}

// ════════════════════════════════════════════════════════════════════
// Demo 1: 基础 RAG pipeline
// ════════════════════════════════════════════════════════════════════

async function demoBasicRag(): Promise<void> {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 1: 基础 RAG pipeline（检索 → 注入 → 回答）");
  console.log("=".repeat(60));
  console.log("[说明] embedding 用纯 TS 词频向量；回答 try API，失败降级 mock。");

  const documents = loadDocuments(DATA_DIR);
  console.log(`\nOUT:chunk: 加载 ${documents.length} 个文档:`);
  for (const doc of documents) {
    console.log(`OUT:chunk:   - ${doc.filename} (${doc.content.length} 字符)`);
  }

  const index = buildIndex(documents, 200, 50);
  console.log(
    `OUT:chunk: 分块完成（chunk_size=200, overlap=50）→ 共 ${index.length} 个分块`,
  );
  console.log(`OUT:chunk: 前 3 个分块预览:`);
  index.slice(0, 3).forEach((entry, i) => {
    const preview = entry.chunk.slice(0, 50).replace(/\n/g, " ");
    console.log(`OUT:chunk:   [${i + 1}] ${preview}...`);
  });

  const sampleEmb = simpleEmbedding("Python 编程语言");
  console.log(`\nOUT:embed: simpleEmbedding('Python 编程语言') = ${JSON.stringify(sampleEmb)}`);
  const a = simpleEmbedding("Python");
  const b = simpleEmbedding("Python 语言");
  const c = simpleEmbedding("天气");
  console.log(
    `OUT:embed: cosine('Python', 'Python 语言') = ${cosineSimilarity(a, b).toFixed(3)} (应较高)`,
  );
  console.log(
    `OUT:embed: cosine('Python', '天气') = ${cosineSimilarity(a, c).toFixed(3)} (应较低)`,
  );

  const query = "Python 有什么特点？";
  console.log(`\nOUT:retrieve: 查询: ${query}`);
  const answer = await ragAnswer(query, index, 3);
  const answerPreview = answer.slice(0, 200);
  console.log(`\nOUT:answer: 回答: ${answerPreview}`);

  console.log(`\nOUT:answer: ✓ 基础 RAG 完成（固定管道：无条件检索 → 注入 → 回答）`);
}

// ════════════════════════════════════════════════════════════════════
// Demo 2: Agentic RAG
// ════════════════════════════════════════════════════════════════════

async function demoAgenticRag(): Promise<void> {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 2: Agentic RAG（Agent 自主决定是否检索）");
  console.log("=".repeat(60));
  console.log("[说明] 把检索作为工具，Agent 自主决定调用与否。");
  console.log("[说明] 先试真实 API，失败降级 mock 决策序列。");

  const documents = loadDocuments(DATA_DIR);
  const index = buildIndex(documents, 200, 50);

  console.log(`\n--- 场景 A：问知识库相关问题 ---`);
  const queryA = "Python 语言有什么特点？";
  console.log(`OUT:agentic: 问题: ${queryA}`);
  await agenticRag(queryA, index);
  console.log(`OUT:agentic: ✓ Agent 对知识库相关问题进行了检索`);

  console.log(`\n--- 场景 B：问常识问题 ---`);
  const queryB = "1+1 等于几？";
  console.log(`OUT:agentic: 问题: ${queryB}`);
  await agenticRag(queryB, index);
  console.log(`OUT:agentic: ✓ Agent 对常识问题直接回答（未检索）`);

  console.log(`\n--- 场景 C：问 Agent 概念 ---`);
  const queryC = "什么是 AI Agent？";
  console.log(`OUT:agentic: 问题: ${queryC}`);
  await agenticRag(queryC, index);
  console.log(`OUT:agentic: ✓ Agent 根据问题性质做出了检索决策`);

  console.log(`\nOUT:agentic: 💡 对比：基础 RAG 对所有问题都检索；Agentic RAG 按需检索。`);
}

// ════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] 章节主题: RAG 检索（基础 RAG + Agentic RAG）`);
  console.log(`[config] 知识库目录: ${DATA_DIR}`);

  await demoBasicRag();
  await demoAgenticRag();

  console.log(`\n${"=".repeat(60)}`);
  console.log("所有演示完成！");
  console.log("💡 核心要点：");
  console.log("   - 基础 RAG：固定管道，无条件检索（简单但浪费）");
  console.log("   - Agentic RAG：Agent 自主决定检索（灵活但复杂）");
  console.log("   - Embedding 用纯 TS 词频向量模拟，真实项目换 embedding API");
  console.log("=".repeat(60));
}

main().catch((err: unknown) => {
  console.error("[fatal]", err);
  process.exit(1);
});
