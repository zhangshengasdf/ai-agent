/**
 * 第13章 从零造框架 — 6 大核心组件实现（TypeScript 版）
 *
 * 本模块实现第12章定义的 6 个 interface，组合成完整 Agent。
 *
 * 6 大组件（对应第12章的 interface）：
 *   1. InMemoryToolRegistry   → 实现 ToolRegistry（工具注册表）
 *   2. DefaultLLMClient       → 实现 LLMClient（LLM 包装器 + 离线 mock 降级）
 *   3. ConversationMemory     → 实现 Memory（对话缓冲记忆）
 *   4. OpenAIToolCallParser   → 实现 ActionParser（tool_calls 解析器）
 *   5. LoggingObserver        → 实现 Observer（日志钩子，纯旁路观察）
 *   6. DefaultAgentRunner     → 实现 AgentRunner（循环引擎，协调其他 5 个组件）
 *
 * 设计原则（第12章确立）：
 *   - 依赖注入：AgentRunner 通过构造函数接收其他 5 个组件（不 new 具体实现）
 *   - 单向依赖：AgentRunner 依赖其他组件，其他组件互不依赖
 *   - Observer 横切关注点：纯旁路（只读不写），不修改主流程状态
 *   - maxSteps 保险丝：必填，默认 10，永不设无限
 *   - 两个终止条件：(1) 模型不调工具 = 完成；(2) 达到 maxSteps = 保险丝
 */

import OpenAI, {
  APIConnectionError,
  APIError,
  AuthenticationError,
  BadRequestError,
  RateLimitError,
} from "openai";

// ═══════════════════════════════════════════════════════════════════════
// 辅助类型（与第12章定义保持一致，框架内部使用）
// ═══════════════════════════════════════════════════════════════════════

/** 对话消息（system/user/assistant/tool 共用一种结构）。 */
export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  /** assistant 消息可能附带 tool_calls；tool 消息可能带 tool_call_id。 */
  tool_calls?: unknown[];
  tool_call_id?: string;
}

/** LLM 响应（统一格式，屏蔽不同 SDK 的差异）。 */
export interface ChatResponse {
  content: string | null;
  tool_calls: RawToolCall[] | null;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/** OpenAI 原始 tool_call 格式（解析前）。 */
export interface RawToolCall {
  id: string;
  type: string;
  function: { name: string; arguments: string };
}

/** 解析后的工具调用（结构化动作）。 */
export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  id: string;
}

/** 自描述工具：名字 + 描述 + JSON Schema 参数 + 处理器。 */
export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  /** rest 参数类型，保证 spread 调用类型安全（T7/T8 教训）。 */
  handler: (...args: string[]) => string;
}

/** OpenAI tools 格式的 JSON Schema。 */
export interface ToolSchema {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

/** LLMClient 需要的最小配置（shared.config.Config 结构兼容）。 */
export interface LLMConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

// ═══════════════════════════════════════════════════════════════════════
// Component 1: InMemoryToolRegistry — 工具注册表
// ═══════════════════════════════════════════════════════════════════════

/**
 * 工具注册表：管理工具元数据 + handler。
 *
 * 核心设计 —— 自描述工具（self-describing）：
 * 每个工具自带 name/description/parameters schema。getSchema() 返回 OpenAI
 * tools 格式 JSON Schema，可直接塞进 chat.completions.create({ tools })。
 */
export class InMemoryToolRegistry {
  private readonly _tools = new Map<string, Tool>();

  register(
    name: string,
    description: string,
    parameters: Record<string, unknown>,
    handler: (...args: string[]) => string,
  ): void {
    this._tools.set(name, { name, description, parameters, handler });
  }

  getSchema(): ToolSchema[] {
    const schemas: ToolSchema[] = [];
    for (const tool of this._tools.values()) {
      schemas.push({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      });
    }
    return schemas;
  }

  execute(name: string, args: Record<string, unknown>): string {
    const tool = this._tools.get(name);
    if (!tool) {
      const available = Array.from(this._tools.keys()).sort().join(", ");
      return `[错误] 工具 '${name}' 不存在。可用工具: ${available}`;
    }
    try {
      // Object.values 按插入顺序返回，与 schema 参数顺序一致。
      // .map(String) 把 unknown[] → string[]，匹配 handler 的 rest params 类型。
      const stringArgs = Object.values(args).map(String);
      return String(tool.handler(...stringArgs));
    } catch (e) {
      return `[工具执行失败] ${name}: ${(e as Error).constructor.name}: ${(e as Error).message}`;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Component 2: DefaultLLMClient — LLM 包装器（含退避重试 + 离线 mock 降级）
// ═══════════════════════════════════════════════════════════════════════

/**
 * LLM 包装器：封装 OpenAI 调用 + 退避重试 + 离线 mock 降级。
 *
 * 归一化响应格式（屏蔽 SDK 差异）：
 *   { content, tool_calls, usage }
 *
 * 离线 mock 设计（关键！）：
 *   .env 的 OPENAI_API_KEY=sk-REPLACE-ME → 真实 API 必失败（401/连接错误）。
 *   chatWithRetry 在重试耗尽/永久错误后，降级返回预设 mock 响应序列，
 *   完整演示多步 Agent 循环：
 *     step1: get_weather(北京) → step2: get_weather(上海)
 *     → step3: calculate(28-25) → step4: 最终回答（不调工具，循环终止）
 */
export class DefaultLLMClient {
  private readonly _client: OpenAI;
  private readonly _model: string;
  private _mockIndex = 0;
  private readonly _mockSequence: ChatResponse[] = [
    {
      content: null,
      tool_calls: [
        {
          id: "call_mock_1",
          type: "function",
          function: { name: "get_weather", arguments: '{"city": "北京"}' },
        },
      ],
    },
    {
      content: null,
      tool_calls: [
        {
          id: "call_mock_2",
          type: "function",
          function: { name: "get_weather", arguments: '{"city": "上海"}' },
        },
      ],
    },
    {
      content: null,
      tool_calls: [
        {
          id: "call_mock_3",
          type: "function",
          function: { name: "calculate", arguments: '{"expression": "28-25"}' },
        },
      ],
    },
    {
      content:
        "北京今天晴 25°C，上海今天多云 28°C。温差为 3°C（上海比北京高 3 度）。",
      tool_calls: null,
    },
  ];

  constructor(config: LLMConfig) {
    this._client = new OpenAI({
      baseURL: config.baseUrl,
      apiKey: config.apiKey,
    });
    this._model = config.model;
  }

  async chat(
    messages: Message[],
    tools?: ToolSchema[],
  ): Promise<ChatResponse> {
    const apiParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming = {
      model: this._model,
      messages: messages as OpenAI.ChatCompletionMessageParam[],
    };
    if (tools && tools.length > 0) {
      apiParams.tools = tools as OpenAI.ChatCompletionTool[];
      apiParams.tool_choice = "auto";
    }

    const response = await this._client.chat.completions.create(apiParams);
    const msg = response.choices[0].message;

    // 归一化 tool_calls（SDK 格式 → 统一 RawToolCall 格式）
    let rawCalls: RawToolCall[] | null = null;
    if (msg.tool_calls) {
      rawCalls = [];
      for (const tc of msg.tool_calls) {
        // 判别联合类型：只处理 type === "function"（T9 教训）
        if (tc.type !== "function") continue;
        rawCalls.push({
          id: tc.id,
          type: "function",
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments,
          },
        });
      }
    }

    let usage: ChatResponse["usage"];
    if (response.usage) {
      usage = {
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        total_tokens: response.usage.total_tokens,
      };
    }

    return {
      content: msg.content ?? null,
      tool_calls: rawCalls,
      usage,
    };
  }

  async chatWithRetry(
    messages: Message[],
    tools?: ToolSchema[],
    maxRetries = 3,
  ): Promise<ChatResponse> {
    const backoffScale = 0.1; // 演示用 0.1s（真实用 1.0s）

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await this.chat(messages, tools);
      } catch (err) {
        const isRetryable =
          err instanceof APIConnectionError || // 含 APIConnectionTimeoutError
          err instanceof RateLimitError ||
          (err instanceof APIError &&
            typeof err.status === "number" &&
            err.status >= 500);

        if (isRetryable && attempt < maxRetries - 1) {
          const wait = Math.pow(2, attempt) * backoffScale;
          console.log(
            `OUT:framework:retry: 第 ${attempt + 1}/${maxRetries} 次失败` +
              `（${(err as Error).constructor.name}），等待 ${wait.toFixed(1)}s...`,
          );
          await sleep(wait);
          continue;
        }
        break; // 永久错误 or 重试耗尽 → 降级 mock
      }
    }

    // ── 降级：离线 mock ──
    console.log(
      `OUT:framework:offline: API 不可用，降级为 mock 响应（第 ${this._mockIndex + 1} 步）`,
    );
    return this._nextMock();
  }

  private _nextMock(): ChatResponse {
    const resp = this._mockSequence[this._mockIndex % this._mockSequence.length];
    this._mockIndex++;
    // 返回深拷贝，防止外部修改内部序列
    const toolCallsCopy: RawToolCall[] | null = resp.tool_calls
      ? resp.tool_calls.map((tc) => ({
          id: tc.id,
          type: tc.type,
          function: { name: tc.function.name, arguments: tc.function.arguments },
        }))
      : null;
    return {
      content: resp.content,
      tool_calls: toolCallsCopy,
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Component 3: ConversationMemory — 对话缓冲记忆
// ═══════════════════════════════════════════════════════════════════════

/**
 * 对话缓冲记忆：数组存储，支持 system/user/assistant/tool 消息。
 *
 * add 方法支持额外字段（tool_calls/tool_call_id），以适配 OpenAI 多轮工具调用格式。
 */
export class ConversationMemory {
  private readonly _systemPrompt: string;
  private _messages: Message[] = [];

  constructor(systemPrompt = "") {
    this._systemPrompt = systemPrompt;
    if (systemPrompt) {
      this._messages.push({ role: "system", content: systemPrompt });
    }
  }

  add(
    role: Message["role"],
    content: string,
    extra?: { tool_calls?: unknown[]; tool_call_id?: string },
  ): void {
    const msg: Message = { role, content };
    if (extra?.tool_call_id !== undefined) {
      msg.tool_call_id = extra.tool_call_id;
    }
    if (extra?.tool_calls !== undefined) {
      msg.tool_calls = extra.tool_calls;
    }
    this._messages.push(msg);
  }

  getMessages(): Message[] {
    // 返回浅拷贝列表，防止外部修改内部状态
    return this._messages.map((m) => ({ ...m }));
  }

  clear(): void {
    this._messages = [];
    if (this._systemPrompt) {
      this._messages.push({ role: "system", content: this._systemPrompt });
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Component 4: OpenAIToolCallParser — 输出解析器
// ═══════════════════════════════════════════════════════════════════════

/**
 * 解析 LLM 响应中的 tool_calls，归一化为 [{name, args, id}]。
 *
 * 屏蔽 OpenAI tool_calls 格式差异：
 *   原始: { id, type: "function", function: { name, arguments: "..." } }
 *   归一化: { name, args: <parsed object>, id }
 */
export class OpenAIToolCallParser {
  parseToolCalls(response: ChatResponse): ToolCall[] {
    const result: ToolCall[] = [];
    if (!response.tool_calls) return result;

    for (const tc of response.tool_calls) {
      if (tc.type !== "function") continue; // 判别联合类型
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(tc.function.arguments || "{}");
      } catch {
        args = {};
      }
      result.push({
        name: tc.function.name,
        args,
        id: tc.id,
      });
    }
    return result;
  }

  hasToolCalls(response: ChatResponse): boolean {
    return response.tool_calls !== null && response.tool_calls.length > 0;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Component 5: LoggingObserver — 日志钩子（纯旁路，只读不写）
// ═══════════════════════════════════════════════════════════════════════

/**
 * 日志观察者：在关键点打印格式化日志（OUT:framework:step{N}: 前缀）。
 *
 * 设计原则 —— 纯旁路观察（只读不写）：
 *   ✅ 记录日志（步骤/工具调用/结果）
 *   ❌ 不修改主流程状态（Memory/messages）
 *   ❌ 不调 LLM / 不执行工具
 *   ❌ 不决定循环是否继续
 */
export class LoggingObserver {
  private _step = 0;

  onStepStart(step: number, messages: Message[]): void {
    this._step = step + 1;
    console.log(
      `OUT:framework:step${this._step}: ▶ 步骤开始（历史消息: ${messages.length} 条）`,
    );
  }

  onLLMCall(messages: Message[]): void {
    console.log(
      `OUT:framework:step${this._step}: 🧠 调用 LLM（输入 ${messages.length} 条消息）`,
    );
  }

  onToolCall(name: string, args: Record<string, unknown>): void {
    console.log(`OUT:framework:step${this._step}: 🔧 调用工具 ${name}(${JSON.stringify(args)})`);
  }

  onToolResult(name: string, result: string): void {
    const preview = result.length > 60 ? result.slice(0, 60) + "..." : result;
    console.log(`OUT:framework:step${this._step}: 📋 工具结果 ${name} → ${preview}`);
  }

  onStepEnd(_step: number): void {
    console.log(`OUT:framework:step${this._step}: ✓ 步骤结束`);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Component 6: DefaultAgentRunner — Agent 循环引擎（框架的心脏）
// ═══════════════════════════════════════════════════════════════════════

/** 依赖注入：AgentRunner 构造函数接收 5 个组件。 */
export interface FrameworkDeps {
  llm: DefaultLLMClient;
  tools: InMemoryToolRegistry;
  memory: ConversationMemory;
  parser: OpenAIToolCallParser;
  observer: LoggingObserver;
}

/**
 * Agent 循环引擎：协调 5 个组件，驱动 observe→reason→act 循环。
 *
 * 依赖注入：通过构造函数接收 5 个组件（不 new 具体实现，保证可替换性）。
 * maxSteps 保险丝：必填，默认 10，永不设无限（第04章反模式 #1）。
 */
export class DefaultAgentRunner {
  private readonly _llm: DefaultLLMClient;
  private readonly _tools: InMemoryToolRegistry;
  private readonly _memory: ConversationMemory;
  private readonly _parser: OpenAIToolCallParser;
  private readonly _observer: LoggingObserver;

  constructor(deps: FrameworkDeps) {
    this._llm = deps.llm;
    this._tools = deps.tools;
    this._memory = deps.memory;
    this._parser = deps.parser;
    this._observer = deps.observer;
  }

  async run(task: string, maxSteps = 10): Promise<string> {
    // 1. 初始化 Memory（把用户任务存入对话历史）
    this._memory.add("user", task);

    // 2. 循环 maxSteps 次
    for (let step = 0; step < maxSteps; step++) {
      this._observer.onStepStart(step, this._memory.getMessages());
      this._observer.onLLMCall(this._memory.getMessages());

      // ── Reason：调 LLM（带重试 + mock 降级）──
      const response = await this._llm.chatWithRetry(
        this._memory.getMessages(),
        this._tools.getSchema(),
      );

      // ── 终止条件 1：模型不调工具 = 任务完成 ──
      if (!this._parser.hasToolCalls(response)) {
        const answer = response.content ?? "(空回答)";
        this._observer.onStepEnd(step);
        return answer;
      }

      // ── Act：记录 assistant 决策（含 tool_calls）到 Memory ──
      this._memory.add("assistant", response.content ?? "", {
        tool_calls: response.tool_calls ?? undefined,
      });

      // ── 执行每个工具调用 ──
      for (const call of this._parser.parseToolCalls(response)) {
        this._observer.onToolCall(call.name, call.args);
        const result = this._tools.execute(call.name, call.args);
        this._observer.onToolResult(call.name, result);
        this._memory.add("tool", result, { tool_call_id: call.id });
      }

      this._observer.onStepEnd(step);
    }

    // 3. 终止条件 2：达到 maxSteps，强制停止
    return "(已达到最大步数，强制停止)";
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 辅助函数
// ═══════════════════════════════════════════════════════════════════════

function sleep(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}
