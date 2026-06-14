/**
 * 第05章 记忆系统（Memory Systems）
 *
 * LLM 本身无状态，每次 API 调用独立——所谓"记忆"是开发者替模型管理 messages
 * 列表的机制。本章实现三种记忆系统：
 *   1. ConversationBuffer: 完整保留所有对话（短期记忆）
 *   2. SummaryMemory: 超阈值时摘要压缩旧历史（中等对话）
 *   3. VectorMemory: 词频向量 + 余弦相似度检索（长期记忆，纯 TS，不依赖向量库）
 *
 * 离线设计：
 *   - Demo 1 (Buffer): 纯内存，100% 离线
 *   - Demo 2 (Summary): 先试真实 API 摘要，失败降级 mock 摘要
 *   - Demo 3 (Vector): 纯 TS 词频向量，不调 embedding API，100% 离线
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 通用类型
// ════════════════════════════════════════════════════════════════════

/** API 消息格式。 */
interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

/** 稀疏向量：词 → 权重。教学用 embedding 表示。 */
type Embedding = Record<string, number>;

// ════════════════════════════════════════════════════════════════════
// 1. ConversationBuffer — 完整保留所有对话（短期记忆）
// ════════════════════════════════════════════════════════════════════

class ConversationBuffer {
  /** 完整保留所有对话消息的短期记忆。 */
  private messages: Message[] = [];

  /** 追加一条消息。 */
  add(role: Message["role"], content: string): void {
    this.messages.push({ role, content });
  }

  /** 返回所有消息的副本（防外部修改内部状态）。 */
  getMessages(): Message[] {
    return this.messages.map((m) => ({ ...m }));
  }

  /** 返回消息数量。 */
  count(): number {
    return this.messages.length;
  }

  /** 清空所有记忆。 */
  clear(): void {
    this.messages = [];
  }
}

// ════════════════════════════════════════════════════════════════════
// 2. SummaryMemory — 超阈值时摘要压缩（中等对话）
// ════════════════════════════════════════════════════════════════════

class SummaryMemory {
  /**
   * 超阈值时把旧消息摘要压缩的记忆系统。
   *
   * 当消息数超过 maxMessages 时，把最早的一批送去 LLM 摘要，
   * 压缩成一段累积摘要，替换掉那批原文。
   */
  private readonly max: number;
  private messages: Message[] = [];
  private summary: string = "";
  private readonly systemPrompt: string;

  constructor(maxMessages = 6, systemPrompt = "") {
    this.max = maxMessages;
    this.systemPrompt = systemPrompt;
  }

  /** 追加消息，超阈值时自动触发摘要。 */
  async add(role: Message["role"], content: string): Promise<void> {
    this.messages.push({ role, content });
    if (this.messages.length > this.max) {
      await this.summarizeOldest();
    }
  }

  /** 把最早 2 条消息送去摘要，压缩进 summary。 */
  private async summarizeOldest(): Promise<void> {
    const toSummarize = this.messages.slice(0, 2);
    this.messages = this.messages.slice(2);

    const chunkText = toSummarize
      .map((m) => `[${m.role}] ${m.content}`)
      .join("\n");

    const newSummary = await this.llmSummarize(chunkText);
    if (this.summary) {
      this.summary = await this.llmSummarize(this.summary + "\n" + newSummary);
    } else {
      this.summary = newSummary;
    }
  }

  /** 调 LLM 摘要，失败时降级为离线 mock 摘要。 */
  private async llmSummarize(text: string): Promise<string> {
    try {
      const resp = await client.chat.completions.create({
        model: cfg.model,
        messages: [
          {
            role: "system",
            content: "请用一句话（不超过50字）总结以下对话的要点：",
          },
          { role: "user", content: text },
        ],
        max_tokens: 100,
      });
      return resp.choices[0].message.content ?? "(摘要为空)";
    } catch {
      return SummaryMemory.mockSummarize(text);
    }
  }

  /** 离线 mock 摘要：提取关键词模拟压缩。 */
  private static mockSummarize(text: string): string {
    const keywords = [
      "小明", "北京", "上海", "Python", "天气", "偏好", "喜欢", "用户",
    ].filter((kw) => text.includes(kw));
    const kwStr = keywords.length > 0 ? keywords.join("、") : "对话内容";
    const snippet = text.slice(0, 30).replace(/\n/g, " ");
    return `[摘要] 涉及${kwStr}。原文片段: ${snippet}...`;
  }

  /** 返回 [摘要 system msg（如有）] + [最近 N 条原文]。 */
  getMessages(): Message[] {
    const result: Message[] = [];
    if (this.systemPrompt) {
      result.push({ role: "system", content: this.systemPrompt });
    }
    if (this.summary) {
      result.push({
        role: "system",
        content: `[之前对话摘要] ${this.summary}`,
      });
    }
    result.push(...this.messages.map((m) => ({ ...m })));
    return result;
  }

  /** 返回当前累积摘要（调试用）。 */
  getSummary(): string {
    return this.summary;
  }

  /** 返回未压缩的原文消息数。 */
  count(): number {
    return this.messages.length;
  }

  /** 清空所有记忆和摘要。 */
  clear(): void {
    this.messages = [];
    this.summary = "";
  }
}

// ════════════════════════════════════════════════════════════════════
// 3. VectorMemory — 词频向量 + 余弦相似度检索（长期记忆）
// ════════════════════════════════════════════════════════════════════

/** 词频向量模拟 embedding（教学用，非真实语义）。 */
function simpleEmbedding(text: string): Embedding {
  let cleaned = text.toLowerCase();
  for (const ch of "，。！？,.!?;:\"'()[]{}（）【】") {
    cleaned = cleaned.split(ch).join(" ");
  }
  const words = cleaned.split(/\s+/).filter((w) => w.length > 0);

  const vec: Embedding = {};
  for (const w of words) {
    if (w.length > 1 && /^[\u4e00-\u9fff]+$/.test(w)) {
      // 连续中文字符（无空格分隔）→ 拆成单字，模拟基础分词
      for (const c of w) {
        vec[c] = (vec[c] ?? 0) + 1;
      }
    } else {
      vec[w] = (vec[w] ?? 0) + 1;
    }
  }
  return vec;
}

/** 两个稀疏向量的余弦相似度，范围 [-1, 1]，1 = 完全相同。纯 TS 实现。 */
function cosineSimilarity(a: Embedding, b: Embedding): number {
  // 点积：只在共有的词上累加
  let dot = 0;
  for (const w in a) {
    dot += a[w] * (b[w] ?? 0);
  }
  let normA = 0;
  for (const w in a) {
    normA += a[w] * a[w];
  }
  let normB = 0;
  for (const w in b) {
    normB += b[w] * b[w];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) {
    return 0;
  }
  return dot / (normA * normB);
}

interface SearchResult {
  text: string;
  score: number;
}

class VectorMemory {
  /** 用词频向量 + 余弦相似度检索的长期记忆。纯 TS 实现。 */
  private store: Array<{ text: string; embedding: Embedding }> = [];

  /** 添加一条文本，自动计算 embedding 并存储。 */
  add(text: string): void {
    const embedding = simpleEmbedding(text);
    this.store.push({ text, embedding });
  }

  /** 检索与 query 最相似的 topK 条文本，返回 [{text, score}] 按相似度降序。 */
  search(query: string, topK = 3): SearchResult[] {
    const qEmb = simpleEmbedding(query);
    const scored: SearchResult[] = this.store.map(({ text, embedding }) => ({
      text,
      score: cosineSimilarity(qEmb, embedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  /** 返回存储的文本数量。 */
  count(): number {
    return this.store.length;
  }

  /** 清空所有存储。 */
  clear(): void {
    this.store = [];
  }
}

// ════════════════════════════════════════════════════════════════════
// Demo 1: ConversationBuffer — 多轮对话记忆
// ════════════════════════════════════════════════════════════════════

function demoConversationBuffer(): void {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 1: ConversationBuffer — 多轮对话记忆");
  console.log("=".repeat(60));

  const buffer = new ConversationBuffer();

  buffer.add("system", "你是任务助手 Agent，会记住用户信息。");
  buffer.add("user", "你好，我叫小明。");
  buffer.add("assistant", "你好小明！有什么可以帮你的？");
  buffer.add("user", "我最喜欢用 Python 编程。");
  buffer.add("assistant", "记住了！Python 是一门优秀的语言。");

  console.log(`OUT:buffer: 消息总数: ${buffer.count()}`);
  console.log(`OUT:buffer: 记忆内容:`);
  buffer.getMessages().forEach((msg, i) => {
    const preview = msg.content.slice(0, 50);
    console.log(`OUT:buffer:   [${i + 1}] ${msg.role}: ${preview}`);
  });

  const history = buffer.getMessages();
  console.log(`OUT:buffer: ✓ getMessages() 返回 ${history.length} 条（完整历史）`);

  buffer.clear();
  console.log(`OUT:buffer: ✓ clear() 后消息数: ${buffer.count()}`);

  buffer.add("user", "新对话开始。");
  console.log(`OUT:buffer: ✓ 新对话后消息数: ${buffer.count()}（独立于旧对话）`);

  console.log(`OUT:buffer: 💡 Buffer 适合短对话（<20轮），长对话需 Summary 或 Vector。`);
}

// ════════════════════════════════════════════════════════════════════
// Demo 2: SummaryMemory — 超阈值自动摘要压缩
// ════════════════════════════════════════════════════════════════════

async function demoSummaryMemory(): Promise<void> {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 2: SummaryMemory — 超阈值自动摘要压缩");
  console.log("=".repeat(60));
  console.log("[说明] 设 maxMessages=6，超过时把最早 2 条送去摘要。");

  const memory = new SummaryMemory(6, "你是任务助手 Agent。");

  const conversation: Array<[Message["role"], string]> = [
    ["user", "你好，我叫小明，住在北京。"],
    ["assistant", "你好小明！北京是个好地方。"],
    ["user", "我喜欢用 Python 编程，特别是做数据分析。"],
    ["assistant", "Python 在数据分析领域很强大！"],
    ["user", "我最近在学机器学习，用 scikit-learn。"],
    ["assistant", "scikit-learn 是经典 ML 库，选择不错。"],
    ["user", "能推荐一个 Python 的可视化库吗？"],
    ["assistant", "推荐 matplotlib 和 seaborn，适合数据分析。"],
  ];

  console.log(`\nOUT:summary: 逐条添加对话（共 ${conversation.length} 条，阈值 6）:`);
  for (const [role, content] of conversation) {
    const beforeCount = memory.count();
    await memory.add(role, content);
    const afterCount = memory.count();
    const summaryLen = memory.getSummary().length;
    const compressed =
      beforeCount >= 6 && afterCount < beforeCount ? "触发了摘要！" : "";
    console.log(
      `OUT:summary: +[${role}] ${content.slice(0, 30)}... ` +
        `(原文数: ${beforeCount}→${afterCount}, 摘要长度: ${summaryLen}) ${compressed}`,
    );
  }

  console.log(`\nOUT:summary: 最终 getMessages() 返回:`);
  memory.getMessages().forEach((msg, i) => {
    const preview = msg.content.slice(0, 60);
    console.log(`OUT:summary:   [${i + 1}] ${msg.role}: ${preview}`);
  });

  console.log(`\nOUT:summary: 累积摘要内容:`);
  console.log(`OUT:summary:   ${memory.getSummary().slice(0, 100)}`);
  console.log(`OUT:summary: ✓ 旧历史被压缩进摘要，近期消息保留原文。`);
  console.log(`OUT:summary: 💡 Summary 平衡了上下文长度与信息密度。`);
}

// ════════════════════════════════════════════════════════════════════
// Demo 3: VectorMemory — 语义检索（词频向量 + 余弦相似度）
// ════════════════════════════════════════════════════════════════════

function demoVectorMemory(): void {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 3: VectorMemory — 语义检索（词频向量+余弦相似度）");
  console.log("=".repeat(60));
  console.log("[说明] 用词频向量模拟 embedding，纯 TS，不调 embedding API。");

  const vm = new VectorMemory();

  const knowledgeBase = [
    "小明喜欢用 Python 编程",
    "北京今天的天气是晴天 25 度",
    "机器学习是人工智能的分支",
    "Python 是 Guido 创建的编程语言",
    "上海今天下雨 30 度",
    "用户偏好用 Python 做数据分析",
  ];

  console.log(`\nOUT:vector: 添加 ${knowledgeBase.length} 条知识:`);
  for (const text of knowledgeBase) {
    vm.add(text);
    console.log(`OUT:vector:   + ${text}`);
  }

  console.log(`\nOUT:vector: 检索 1: query='Python 编程'`);
  let results = vm.search("Python 编程", 3);
  for (const { text, score } of results) {
    console.log(`OUT:vector:   [${score.toFixed(3)}] ${text}`);
  }

  console.log(`OUT:vector: 检索 2: query='今天天气怎么样'`);
  results = vm.search("今天天气怎么样", 3);
  for (const { text, score } of results) {
    console.log(`OUT:vector:   [${score.toFixed(3)}] ${text}`);
  }

  console.log(`OUT:vector: 检索 3: query='音乐推荐'（无关查询）`);
  results = vm.search("音乐推荐", 3);
  for (const { text, score } of results) {
    console.log(`OUT:vector:   [${score.toFixed(3)}] ${text}`);
  }

  console.log(`\nOUT:vector: 余弦相似度验证:`);
  const a = simpleEmbedding("Python 编程");
  const b = simpleEmbedding("Python 编程");
  console.log(
    `OUT:vector:   cosine('Python 编程', 'Python 编程') = ${cosineSimilarity(a, b).toFixed(3)} (应为1.0)`,
  );
  const c = simpleEmbedding("天气");
  const d = simpleEmbedding("Python 编程");
  console.log(
    `OUT:vector:   cosine('天气', 'Python 编程') = ${cosineSimilarity(c, d).toFixed(3)} (应较低)`,
  );
  console.log(`OUT:vector: ✓ 词频向量能捕捉关键词重叠，近似语义检索。`);
  console.log(`OUT:vector: 💡 真实项目换成 embedding API，检索质量会大幅提升。`);
}

// ════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] 章节主题: 记忆系统（ConversationBuffer + SummaryMemory + VectorMemory）`);

  // Demo 1: ConversationBuffer（纯内存，不调 API）
  demoConversationBuffer();

  // Demo 2: SummaryMemory（先试真实 API，失败降级 mock）
  await demoSummaryMemory();

  // Demo 3: VectorMemory（纯 TS 词频向量，100% 离线）
  demoVectorMemory();

  console.log(`\n${"=".repeat(60)}`);
  console.log("所有演示完成！三种记忆系统均已展示。");
  console.log("💡 核心要点：LLM 无状态，记忆=你替模型管理 messages 的机制。");
  console.log("   - Buffer: 完整保留（短对话）");
  console.log("   - Summary: 超阈值摘要压缩（中等对话）");
  console.log("   - Vector: 语义检索召回（长期/跨会话）");
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error("运行出错:", err);
  process.exit(1);
});
