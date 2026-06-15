/**
 * 第11章 上下文工程（Context Engineering）
 *
 * Prompt 工程的进化——当 Agent 跑几十步、调上百次工具时，"写好一个 prompt"不够了。
 * 真正决定质量的是：你如何管理喂给模型的上下文。
 *
 * 本章实现上下文工程的三大支柱：
 *   1. ContextCompactor: 上下文压缩——超 token 阈值时把旧轨迹摘要成一条 system 消息
 *   2. SubAgent 隔离: 主 Agent 派子 Agent 干重活，只收回摘要（不看全量轨迹）
 *   3. TokenBudget: token 预算管理——每轮检查，接近上限时自动触发压缩
 *
 * 离线设计：
 *   - Token 估算用纯 TS（字符数 // 3），不依赖 gpt-tokenizer，100% 离线
 *   - 压缩/子 Agent：先试真实 API，失败降级 mock（预设文本），保证演示完整
 *   - 预算循环：纯本地模拟对话，100% 离线
 */

import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 通用类型 & Token 估算
// ════════════════════════════════════════════════════════════════════

interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

/** 估算 messages 的 token 数（粗略，1 token ≈ 3 字符）。纯 TS，不依赖 gpt-tokenizer。 */
function estimateTokens(messages: Message[]): number {
  const text = JSON.stringify(messages);
  return Math.floor(text.length / 3);
}

// ════════════════════════════════════════════════════════════════════
// 1. ContextCompactor — 上下文压缩
// ════════════════════════════════════════════════════════════════════

class ContextCompactor {
  /** 超 token 阈值时把旧轨迹摘要压缩的上下文管理器。 */
  private readonly threshold: number;
  private readonly keepRecent: number;
  private messages: Message[] = [];
  private summary: string = "";

  constructor(threshold = 2000, keepRecent = 6) {
    this.threshold = threshold;
    this.keepRecent = keepRecent;
  }

  /** 追加消息，返回是否触发了压缩。 */
  async add(message: Message): Promise<boolean> {
    this.messages.push(message);
    if (estimateTokens(this.messages) > this.threshold) {
      await this.compact();
      return true;
    }
    return false;
  }

  /** 把旧消息摘要，保留最近 keepRecent 条原文。 */
  private async compact(): Promise<void> {
    const split = Math.max(this.keepRecent, 1);
    const old = this.messages.slice(0, -split);
    const recent = this.messages.slice(-split);

    if (old.length === 0) return;

    const newSummary = await this.llmSummarize(old);
    if (this.summary) {
      this.summary = `${this.summary}\n${newSummary}`;
    } else {
      this.summary = newSummary;
    }
    this.messages = recent;
  }

  /** 调 LLM 摘要，失败时降级为离线 mock 摘要。 */
  private async llmSummarize(messages: Message[]): Promise<string> {
    try {
      const text = messages
        .map((m) => `[${m.role}] ${m.content}`)
        .join("\n");
      const resp = await client.chat.completions.create({
        model: cfg.model,
        messages: [
          {
            role: "system",
            content: "请用一段话（不超过80字）总结以下对话的要点：",
          },
          { role: "user", content: text },
        ],
        max_tokens: 150,
      });
      return resp.choices[0].message.content || "(摘要为空)";
    } catch {
      return ContextCompactor.mockSummarize(messages);
    }
  }

  /** 离线 mock 摘要：提取关键词 + 片段模拟压缩结果。 */
  private static mockSummarize(messages: Message[]): string {
    const keywords: string[] = [];
    const allText = messages.map((m) => m.content).join(" ");
    const kwList = [
      "Python", "Agent", "框架", "LangChain", "工具", "记忆",
      "小明", "北京", "天气", "研究", "压缩", "上下文",
    ];
    for (const kw of kwList) {
      if (allText.includes(kw)) keywords.push(kw);
    }
    const kwStr = keywords.slice(0, 5).join("、") || "对话内容";
    const snippet = allText.slice(0, 40).replace(/\n/g, " ");
    return `[摘要] 涉及${kwStr}。要点片段: ${snippet}...`;
  }

  /** 返回 [摘要 system msg（如有）] + [最近 N 条原文]。 */
  getMessages(): Message[] {
    const result: Message[] = [];
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

  /** 返回原文消息数（不含摘要）。 */
  count(): number {
    return this.messages.length;
  }
}

// ════════════════════════════════════════════════════════════════════
// 2. SubAgent 隔离 — 主 Agent 派子 Agent，只收回摘要
// ════════════════════════════════════════════════════════════════════

class SubAgent {
  /** 模拟一个子 Agent：内部有多步轨迹，但只向外暴露摘要。 */
  private readonly name: string;
  private trace: Message[] = [];

  constructor(name: string) {
    this.name = name;
  }

  /** 模拟子 Agent 研究一个主题（多步轨迹），返回摘要。 */
  async runResearch(topic: string): Promise<string> {
    const steps: Array<[string, string]> = [
      ["search", `搜索 '${topic}' 的基础信息...`],
      ["read", `阅读关于 ${topic} 的 3 篇核心文章...`],
      ["analyze", `分析 ${topic} 的关键特征和应用场景...`],
      ["search", `补充搜索 '${topic} 最新趋势 2025'...`],
      ["synthesize", `整理 ${topic} 的要点：定义、应用、趋势...`],
      ["output", `输出 ${topic} 的研究摘要。`],
    ];
    this.trace.push({
      role: "system",
      content: `子Agent[${this.name}] 开始研究: ${topic}`,
    });
    for (const [action, detail] of steps) {
      this.trace.push({
        role: "assistant",
        content: `[${action}] ${detail}`,
      });
      this.trace.push({
        role: "tool",
        content: `${topic}相关数据: ${detail.repeat(5)}`,
      });
    }

    return this.summarizeForParent(topic);
  }

  /** 生成给主 Agent 的摘要（1-2 句话）。 */
  private async summarizeForParent(topic: string): Promise<string> {
    try {
      const traceText = this.trace.map((m) => m.content).join("\n");
      const resp = await client.chat.completions.create({
        model: cfg.model,
        messages: [
          {
            role: "system",
            content: `你刚完成对'${topic}'的研究。请用1-2句话总结核心发现（不超过60字）：`,
          },
          { role: "user", content: traceText },
        ],
        max_tokens: 100,
      });
      return resp.choices[0].message.content || SubAgent.mockSummary(topic);
    } catch {
      return SubAgent.mockSummary(topic);
    }
  }

  /** 离线 mock 摘要（预设文本，模拟 LLM 压缩结果）。 */
  private static mockSummary(topic: string): string {
    return `${topic}的核心：它是当前Agent领域的关键技术，已有多款主流框架支持，2025年趋势是工具调用+记忆融合。`;
  }

  /** 返回子 Agent 完整轨迹的 token 数（用于对比展示）。 */
  traceTokenCount(): number {
    return estimateTokens(this.trace);
  }

  /** 返回子 Agent 轨迹的消息条数。 */
  traceStepCount(): number {
    return this.trace.length;
  }
}

interface MainAgentResult {
  mainTokens: number;
  subagentTotalTokens: number;
  mainMsgCount: number;
}

/** 模拟主 Agent 用子 Agent 隔离研究 3 个主题。 */
async function runMainAgentWithSubagents(): Promise<MainAgentResult> {
  const topics = ["LangChain 框架", "ReAct 推理模式", "向量记忆系统"];
  const mainMessages: Message[] = [
    { role: "system", content: "你是研究助手 Agent。" },
    { role: "user", content: `研究这3个主题并汇总: ${topics.join(", ")}` },
  ];

  let subagentTotalTokens = 0;

  for (const topic of topics) {
    const sub = new SubAgent(`researcher-${topic.slice(0, 4)}`);
    const summary = await sub.runResearch(topic);
    const traceTokens = sub.traceTokenCount();
    subagentTotalTokens += traceTokens;

    mainMessages.push({
      role: "assistant",
      content: `子Agent研究了'${topic}'，发现: ${summary}`,
    });
  }

  mainMessages.push({
    role: "assistant",
    content: "三个主题研究完毕，共同点是都涉及Agent的核心能力。",
  });

  return {
    mainTokens: estimateTokens(mainMessages),
    subagentTotalTokens,
    mainMsgCount: mainMessages.length,
  };
}

// ════════════════════════════════════════════════════════════════════
// 3. TokenBudget — token 预算管理
// ════════════════════════════════════════════════════════════════════

interface BudgetStatus {
  tokens: number;
  budget: number;
  usage: number;
  compacted: boolean;
}

class TokenBudget {
  /** token 预算管理器：每轮检查，接近上限时自动触发压缩。 */
  readonly budget: number;
  readonly thresholdRatio: number;
  readonly threshold: number;
  private readonly compactor: ContextCompactor;
  compactionCount = 0;

  constructor(budget = 4000, thresholdRatio = 0.8, keepRecent = 6) {
    this.budget = budget;
    this.thresholdRatio = thresholdRatio;
    this.threshold = Math.floor(budget * thresholdRatio);
    this.compactor = new ContextCompactor(this.threshold, keepRecent);
  }

  /** 添加消息，返回当前 token 状态 + 是否压缩了。 */
  async add(message: Message): Promise<BudgetStatus> {
    const compacted = await this.compactor.add(message);
    if (compacted) this.compactionCount++;
    const tokens = this.currentTokens();
    return {
      tokens,
      budget: this.budget,
      usage: tokens / this.budget,
      compacted,
    };
  }

  /** 返回当前上下文（含摘要）的 token 数。 */
  currentTokens(): number {
    return estimateTokens(this.compactor.getMessages());
  }

  getMessages(): Message[] {
    return this.compactor.getMessages();
  }
}

// ════════════════════════════════════════════════════════════════════
// Demo 1: Token 估算（纯离线）
// ════════════════════════════════════════════════════════════════════

function demoTokenEstimation(): void {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 1: Token 估算（纯字符数 // 3，不依赖 gpt-tokenizer）");
  console.log("=".repeat(60));

  const testCases: Message[][] = [
    [{ role: "user", content: "你好" }],
    [{ role: "user", content: "请用Python写一个快速排序算法，并解释其时间复杂度。" }],
    [
      { role: "system", content: "你是助手。" },
      { role: "user", content: "研究AI Agent框架的趋势，包括LangChain、LangGraph等。" },
    ],
  ];

  console.log(`OUT:token: 测试 ${testCases.length} 组消息:`);
  testCases.forEach((msgs, i) => {
    const tokens = estimateTokens(msgs);
    const chars = JSON.stringify(msgs).length;
    const preview = msgs[msgs.length - 1].content.slice(0, 30);
    console.log(`OUT:token:   [${i + 1}] ${tokens} tokens (${chars} chars) | ${preview}...`);
  });

  console.log(`\nOUT:token: 消息数增长 → token 增长（模拟对话累积）:`);
  const growing: Message[] = [{ role: "system", content: "你是任务助手 Agent。" }];
  for (const n of [1, 5, 10, 20, 50]) {
    while (growing.length < n + 1) {
      growing.push({ role: "user", content: `第${growing.length}轮对话：请帮我处理任务。` });
      growing.push({ role: "assistant", content: `好的，我来处理第${growing.length}轮的任务。` });
    }
    const tokens = estimateTokens(growing);
    console.log(`OUT:token:   ${growing.length.toString().padStart(3)} 条消息 → ${tokens.toString().padStart(5)} tokens`);
  }

  console.log(`OUT:token: ✓ 纯字符估算，零依赖，可离线验证上下文规模。`);
  console.log(`OUT:token: 💡 生产环境换 gpt-tokenizer 可获得精确值（误差 <1%）。`);
}

// ════════════════════════════════════════════════════════════════════
// Demo 2: ContextCompactor — 上下文压缩
// ════════════════════════════════════════════════════════════════════

async function demoContextCompaction(): Promise<void> {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 2: ContextCompactor — 上下文压缩");
  console.log("=".repeat(60));
  console.log("[说明] 设 threshold=2000 tokens，keep_recent=6。");
  console.log("[说明] 先试真实 API 摘要，失败降级 mock 摘要。");

  const compactor = new ContextCompactor(2000, 6);

  const bigResult1 = (
    "搜索结果: asyncio是Python 3.4引入的异步IO库。核心组件包括 event loop、" +
    "coroutine、task、future。用 async def 定义协程函数，await 等待协程完成。" +
    "与多线程相比，asyncio 单线程并发，无锁，适合IO密集场景。" +
    "常见用法：aiohttp异步HTTP、aiofiles异步文件、asyncpg异步PG。"
  ).repeat(10);
  const bigResult2 = (
    "搜索结果: LangChain的Agent支持异步工具。用 @tool 装饰器可定义 async 工具，" +
    "AgentExecutor 内部用 asyncio.gather 并发执行独立工具调用。" +
    "LangGraph 进一步支持流式执行和中断恢复。" +
    "注意：同步工具和异步工具混用时，框架会自动适配，但推荐统一用 async。"
  ).repeat(10);
  const bigResult3 = (
    "搜索结果: 向量记忆系统的 embedding API 调用是IO密集操作，应该用 async。" +
    "OpenAI SDK 支持 async client：AsyncOpenAI。检索时用 await client.embeddings.create。" +
    "批量 embedding 用 asyncio.gather 并发，比串行快 5-10 倍。" +
    "上下文压缩同理：调LLM摘要用 async，不阻塞 event loop 上的其他工具。"
  ).repeat(10);

  const conversation: Array<[Message["role"], string]> = [
    ["user", "我想了解 Python 的异步编程，asyncio 怎么用？"],
    ["assistant", `asyncio 是 Python 的异步IO库。我来查详细资料。\n${bigResult1}`],
    ["user", "能解释一下 event loop 吗？它和线程有什么区别？"],
    ["assistant", `event loop 是核心。补充搜索结果:\n${bigResult2}`],
    ["user", "Agent 开发里怎么用异步？我听说 LangChain 支持异步工具。"],
    ["assistant", `LangChain 支持。详细资料:\n${bigResult2}`],
    ["user", "那记忆系统呢？VectorMemory 的检索可以异步吗？"],
    ["assistant", `可以异步。详细:\n${bigResult3}`],
    ["user", "上下文压缩也是这个原理吧？压缩时主 Agent 可以等。"],
    ["assistant", "对。压缩是IO密集操作（调LLM摘要），用 async 不阻塞其他工具。"],
  ];

  console.log(`\nOUT:compact: 逐条添加对话（共 ${conversation.length} 条，阈值 2000 tokens）:`);
  for (const [role, content] of conversation) {
    const beforeTokens = estimateTokens(compactor.getMessages());
    const compacted = await compactor.add({ role, content });
    const afterTokens = estimateTokens(compactor.getMessages());
    const flag = compacted ? " ⚡触发了压缩!" : "";
    console.log(
      `OUT:compact: +[${role.padEnd(9)}] tokens: ${String(beforeTokens).padStart(4)}→${String(afterTokens).padStart(4)}` +
      ` (原文${compactor.count()}条)${flag}`,
    );
  }

  console.log(`\nOUT:compact: 最终上下文（摘要 + 最近6条原文）:`);
  const finalMsgs = compactor.getMessages();
  finalMsgs.forEach((msg, i) => {
    const preview = msg.content.slice(0, 60);
    console.log(`OUT:compact:   [${i + 1}] ${msg.role}: ${preview}`);
  });

  console.log(`\nOUT:compact: 累积摘要:`);
  console.log(`OUT:compact:   ${compactor.getSummary().slice(0, 120)}`);
  const finalTokens = estimateTokens(finalMsgs);
  console.log(`OUT:compact: ✓ 最终 ${finalTokens} tokens，旧轨迹被压缩进摘要。`);
  console.log(`OUT:compact: 💡 压缩把 token 从'线性增长'变成'有上限'，避免质量衰退。`);
}

// ════════════════════════════════════════════════════════════════════
// Demo 3: SubAgent 隔离
// ════════════════════════════════════════════════════════════════════

async function demoSubagentIsolation(): Promise<void> {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 3: SubAgent 隔离 — 主 Agent 只收摘要");
  console.log("=".repeat(60));
  console.log("[说明] 主 Agent 派 3 个子 Agent 研究，每个子 Agent 内部有 6 步轨迹。");
  console.log("[说明] 主 Agent 只收每子 Agent 的 1 句摘要。");

  const sub = new SubAgent("demo-researcher");
  const summary = await sub.runResearch("LangChain 框架");
  console.log(`\nOUT:subagent: 单个子 Agent 内部轨迹（主 Agent 看不到）:`);
  console.log(`OUT:subagent:   轨迹消息数: ${sub.traceStepCount()} 条`);
  console.log(`OUT:subagent:   轨迹 token 数: ${sub.traceTokenCount()} tokens`);
  console.log(`OUT:subagent:   主 Agent 收到的摘要: ${summary}`);
  console.log(`OUT:subagent:   摘要 token 数: ${estimateTokens([{ role: "assistant", content: summary }])}`);

  console.log(`\nOUT:subagent: 主 Agent 派 3 个子 Agent 研究（对比上下文大小）:`);
  const result = await runMainAgentWithSubagents();
  console.log(`OUT:subagent:   主 Agent 上下文: ${result.mainTokens} tokens (${result.mainMsgCount} 条消息)`);
  console.log(`OUT:subagent:   3个子Agent总轨迹: ${result.subagentTotalTokens} tokens（被隔离）`);
  console.log(`OUT:subagent:   隔离节省: ${result.subagentTotalTokens - result.mainTokens} tokens 不进主上下文`);

  const ratio = (result.mainTokens / Math.max(result.subagentTotalTokens, 1)) * 100;
  console.log(`OUT:subagent:   主上下文仅为子轨迹的 ${ratio.toFixed(1)}%`);
  console.log(`OUT:subagent: ✓ 隔离让主 Agent 上下文保持干净，只看摘要不看原始搜索结果。`);
  console.log(`OUT:subagent: 💡 反模式：把子 Agent 全量轨迹塞回主 Agent = 丧失隔离意义。`);
}

// ════════════════════════════════════════════════════════════════════
// Demo 4: TokenBudget — 完整的"对话→检查→压缩→继续"循环
// ════════════════════════════════════════════════════════════════════

async function demoTokenBudget(): Promise<void> {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Demo 4: TokenBudget — 预算管理与自动压缩循环");
  console.log("=".repeat(60));
  console.log("[说明] budget=1500 tokens（demo 缩小值，便于 12 轮内观察），超 80%（1200）触发压缩。");
  console.log("[说明] 模拟含大段解释的真实对话，观察 token 变化和自动压缩。");

  const budget = new TokenBudget(1500, 0.8, 6);

  const detail1 = (
    "详细回答: 工具调用是让 LLM 决定调用哪个函数。你定义工具的 JSON Schema，" +
    "传给 API 的 tools 参数。模型分析用户意图后，返回结构化的 tool_calls 字段，" +
    "包含函数名和参数。你执行该函数，把结果以 role=tool 追加到 messages，" +
    "再调一次 API 让模型看结果。和普通函数调用的区别：调用决策由模型做，不是硬编码。"
  ).repeat(8);
  const detail2 = (
    "详细回答: Agent 循环让模型多步推理。结构：for step in range(MAX_STEPS)，" +
    "每步调 LLM→看有无 tool_calls→有则执行并追加结果→无则终止。" +
    "为什么不能单次调用：复杂任务需要多步（查天气+查日历+综合判断），" +
    "单次调用模型无法获得工具结果反馈。max_steps 防止无限循环。"
  ).repeat(8);
  const detail3 = (
    "详细回答: ReAct=Reason+Act。显式版模型输出 Thought/Action/Observation 文本，" +
    "你用正则解析。隐式版用 tools API，模型输出结构化 tool_calls。" +
    "显式版推理过程可见可调试，但格式脆弱。隐式版结构稳定但推理黑盒。" +
    "现代框架默认用隐式，但理解显式能看透底层。"
  ).repeat(8);
  const detail4 = (
    "详细回答: 记忆系统选择：ConversationBuffer 完整保留，适合短对话（<20轮）。" +
    "SummaryMemory 超阈值摘要压缩，适合中等对话（20-100轮）。" +
    "VectorMemory 词频/embedding 向量+余弦相似度检索，适合长期/知识库。" +
    "组合使用最常见：当前会话用 Buffer，用户画像用 Summary，知识库用 Vector。"
  ).repeat(8);

  const conversation: Array<[Message["role"], string]> = [
    ["user", "你好，我想学习 AI Agent 开发，从哪里开始？"],
    ["assistant", "建议从基础开始：先学 LLM API 调用，再学工具调用，最后学 Agent 循环。"],
    ["user", "工具调用是什么意思？和普通函数调用有什么区别？"],
    ["assistant", detail1],
    ["user", "Agent 循环又是啥？为什么不能单次调用搞定？"],
    ["assistant", detail2],
    ["user", "ReAct 推理是什么？和隐式工具调用有什么不同？"],
    ["assistant", detail3],
    ["user", "记忆系统怎么选？Buffer、Summary、Vector 各适合什么？"],
    ["assistant", detail4],
    ["user", "上下文工程又是什么？它和Prompt工程啥关系？"],
    ["assistant", "上下文工程是Prompt工程的进化：主动管理每次调用模型看到什么。"],
  ];

  console.log(`\nOUT:budget: 预算循环演示（${conversation.length} 轮对话）:`);
  for (let i = 0; i < conversation.length; i++) {
    const [role, content] = conversation[i];
    const status = await budget.add({ role, content });
    const barFilled = Math.floor(status.usage * 20);
    const bar = "█".repeat(barFilled) + "░".repeat(20 - barFilled);
    const flag = status.compacted ? " ⚡压缩!" : "";
    const pct = Math.floor(status.usage * 100);
    console.log(
      `OUT:budget: 轮${String(i + 1).padStart(2)} [${role.padEnd(9)}] ${bar} ${String(status.tokens).padStart(4)}/${status.budget}` +
      ` (${pct}%)${flag}`,
    );
  }

  console.log(`\nOUT:budget: 总压缩次数: ${budget.compactionCount}`);
  const final = budget.getMessages();
  const finalTokens = budget.currentTokens();
  console.log(`OUT:budget: 最终上下文: ${final.length} 条消息, ${finalTokens} tokens`);
  console.log(`OUT:budget: ✓ 预算循环让上下文始终在健康范围内，自动避免超限。`);
  console.log(`OUT:budget: 💡 这是所有长任务 Agent 的基础设施——没有它，Agent 跑久了必然衰退。`);
}

// ════════════════════════════════════════════════════════════════════
// 主入口
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log(`[config] provider=${cfg.provider}, model=${cfg.model}`);
  console.log(`[config] 章节主题: 上下文工程（压缩 + 子Agent隔离 + Token预算）`);

  demoTokenEstimation();
  await demoContextCompaction();
  await demoSubagentIsolation();
  await demoTokenBudget();

  console.log(`\n${"=".repeat(60)}`);
  console.log("所有演示完成！上下文工程三大支柱均已展示。");
  console.log("💡 核心要点：上下文是有限昂贵的资源，必须主动管理。");
  console.log("   - 压缩: 超阈值摘要旧轨迹（本章 ContextCompactor）");
  console.log("   - 隔离: 子 Agent 只回摘要（本章 SubAgent）");
  console.log("   - 预算: 超上限自动触发压缩（本章 TokenBudget）");
  console.log("=".repeat(60));
}

main().catch((err: unknown) => {
  console.error("[fatal]", err instanceof Error ? err.message : String(err));
  process.exit(1);
});
