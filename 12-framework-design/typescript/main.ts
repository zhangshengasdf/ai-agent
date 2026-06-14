/**
 * 第12章 从零造框架 — 架构设计（6 大核心组件）
 *
 * 本章只定义接口契约（interface），不实现具体逻辑（第13章才实现）。
 *
 * 6 大核心组件：
 *   1. AgentRunner   — Agent 循环引擎（observe→reason→act，max_steps 保险丝）
 *   2. ToolRegistry  — 工具注册表（自描述工具，register/getSchema/execute）
 *   3. LLMClient     — LLM 包装器（调用+重试+流式+结构化输出）
 *   4. Memory        — 记忆管理（对话缓冲+可选摘要+token预算）
 *   5. ActionParser  — 输出解析器（结构化→动作，解析 tool_calls）
 *   6. Observer      — 可观测钩子（每步日志/trace/成本追踪）
 *
 * 本文件做三件事：
 *   1. 用 interface 定义 6 个组件契约 + 辅助类型
 *   2. 打印每个接口的职责和方法签名（手写元数据，因为 TS interface 运行时擦除）
 *   3. 打印架构图 + 接口验证说明
 *
 * 不调用任何真实 API —— 纯接口定义 + 自省演示。
 */

import { getConfig } from "../../shared/config";

// 验证 import 路径正确（本章不真正用 config 调 API，只验证路径）
const _cfg = getConfig();

// ═══════════════════════════════════════════════════════════════════════
// 辅助类型（被多个接口复用）
// ═══════════════════════════════════════════════════════════════════════

/** 对话消息（system/user/assistant/tool 共用一种结构）。 */
interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  /** assistant 消息可能附带 tool_calls；tool 消息可能带 tool_call_id。 */
  tool_calls?: unknown[];
  tool_call_id?: string;
}

/** LLM 响应（统一格式，屏蔽不同 SDK 的差异）。 */
interface ChatResponse {
  content: string | null;
  tool_calls: ToolCall[] | null;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/** 解析后的工具调用（结构化动作）。 */
interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  id: string;
}

/** 自描述工具：名字 + 描述 + JSON Schema 参数 + 处理器。 */
interface Tool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  /** rest 参数类型，保证 spread 调用类型安全（T7/T8 教训）。 */
  handler: (...args: string[]) => string;
}

/** OpenAI tools 格式的 JSON Schema（getSchema() 返回值）。 */
interface ToolSchema {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

// ═══════════════════════════════════════════════════════════════════════
// 6 大核心组件接口定义（interface —— 声明性约束）
// ═══════════════════════════════════════════════════════════════════════

// ─── 组件 1：ToolRegistry — 工具注册表 ───────────────────────────────────
/**
 * 工具注册表：管理工具的元数据（名称/描述/参数 schema）和处理器。
 *
 * 核心设计 —— 自描述工具（self-describing）：
 * 每个工具自带名称、描述、参数 schema。getSchema() 返回 OpenAI tools
 * 格式的 JSON Schema 列表，可直接塞进 client.chat.completions.create(tools=...)。
 *
 * 职责边界：
 *   ✅ register / getSchema / execute
 *   ❌ 不决定"调哪个工具"（LLM 决策）
 *   ❌ 不解析 LLM 响应（ActionParser 的活）
 *   ❌ 不记录调用日志（Observer 的活）
 */
interface ToolRegistry {
  /** 注册一个工具：名字 + 描述 + JSON Schema 参数 + 处理函数。 */
  register(
    name: string,
    description: string,
    parameters: Record<string, unknown>,
    handler: (...args: string[]) => string,
  ): void;

  /** 返回 OpenAI tools 格式的 JSON Schema 列表，给 LLM 看。 */
  getSchema(): ToolSchema[];

  /** 按名字查找 handler 并执行，返回字符串结果。 */
  execute(name: string, args: Record<string, unknown>): string;
}

// ─── 组件 2：LLMClient — LLM 包装器 ──────────────────────────────────────
/**
 * LLM 包装器：封装所有与 LLM API 的交互。
 *
 * 职责边界：
 *   ✅ chat（调用）+ chatWithRetry（带退避重试，第06章）
 *   ✅ （可选扩展）stream / structuredOutput
 *   ❌ 不管对话历史（Memory 的活）
 *   ❌ 不解析 tool_calls（ActionParser 的活）
 *   ❌ 不记录成本（Observer 的活）
 *
 * 为什么需要包装器：
 *   1. 统一接口 —— 换提供商只换实现，上层无感
 *   2. 重试逻辑集中 —— 不用每个 Agent 重写退避重试
 *   3. mock 友好 —— 测试时注入 MockLLMClient，不依赖真实 API
 */
interface LLMClient {
  /** 发送消息列表，返回统一格式的响应（含 content 和 tool_calls）。 */
  chat(
    messages: Message[],
    tools?: ToolSchema[],
  ): Promise<ChatResponse>;

  /** 带指数退避的重试封装（第06章 callLLMWithRetry 的接口化）。 */
  chatWithRetry(
    messages: Message[],
    maxRetries?: number,
  ): Promise<ChatResponse>;
}

// ─── 组件 3：Memory — 记忆管理 ───────────────────────────────────────────
/**
 * 记忆管理：决定"模型能看到什么"（上下文工程的载体）。
 *
 * 职责边界：
 *   ✅ add（追加消息）/ getMessages（返回列表）/ clear（清空）
 *   ✅ （可选扩展）自动压缩 / token 预算（第05/11章 SummaryMemory/TokenBudget）
 *   ❌ 不调 LLM（即使 SummaryMemory 做摘要，也是注入 LLMClient）
 *   ❌ 不执行工具 / 不决定何时停止
 *
 * 多态价值：
 *   第05章的 ConversationBuffer / SummaryMemory / VectorMemory 都可实现此接口。
 *   AgentRunner 不关心你用哪种记忆，只要能 add/getMessages/clear。
 */
interface Memory {
  /** 追加一条消息（role: system/user/assistant/tool）。 */
  add(role: Message["role"], content: string): void;

  /** 返回当前消息列表（给 LLMClient 用）。 */
  getMessages(): Message[];

  /** 清空对话历史（开始新对话）。 */
  clear(): void;
}

// ─── 组件 4：ActionParser — 输出解析器 ───────────────────────────────────
/**
 * 输出解析器：把 LLM 原始响应解析成结构化的"动作"。
 *
 * 职责边界：
 *   ✅ parseToolCalls（提取 [{name, args, id}]）/ hasToolCalls（判断）
 *   ❌ 不执行工具（ToolRegistry 的活）
 *   ❌ 不调 LLM / 不做格式重试（上层/第06章自我纠正）
 *
 * 为什么单独抽出来：
 *   1. 响应格式多样 —— OpenAI tools API 返回 tool_calls 字段；显式 ReAct
 *      （第07章）返回纯文本要正则解析；某些模型返回自定义 JSON
 *   2. 解析是脆弱环节 —— 第07章格式错误、第06章幻觉工具名，集中处理好维护
 */
interface ActionParser {
  /** 从响应中提取工具调用列表 [{name, args, id}]。 */
  parseToolCalls(response: ChatResponse): ToolCall[];

  /** 判断响应是否包含工具调用（决定循环是否继续）。 */
  hasToolCalls(response: ChatResponse): boolean;
}

// ─── 组件 5：Observer — 可观测钩子 ───────────────────────────────────────
/**
 * 可观测钩子：横切关注点，不侵入主流程地记录每步状态。
 *
 * 设计原则 —— 纯旁路观察（只读不写）：
 *   ✅ 记录日志 / trace / 成本追踪 / 性能指标
 *   ❌ 不修改主流程状态（Memory/messages）
 *   ❌ 不调 LLM / 不执行工具
 *   ❌ 不决定循环是否继续
 *
 * 这是观察者模式（Observer Pattern）+ OpenTelemetry 的设计基础。
 * 第15章会实现 TracingObserver，第16章会加 GuardrailObserver。
 */
interface Observer {
  /** 每步开始：可记录 step 编号、当前消息数。 */
  onStepStart(step: number, messages: Message[]): void;

  /** 调 LLM 前：可记录 token 数、估算成本。 */
  onLLMCall(messages: Message[]): void;

  /** 调工具前：可记录工具名、参数。 */
  onToolCall(name: string, args: Record<string, unknown>): void;

  /** 工具返回后：可记录结果、耗时。 */
  onToolResult(name: string, result: string): void;

  /** 每步结束：可记录总耗时。 */
  onStepEnd(step: number): void;
}

// ─── 组件 6：AgentRunner — Agent 循环引擎 ─────────────────────────────────
/**
 * Agent 循环引擎：驱动 observe→reason→act，框架的"心脏"。
 *
 * 职责边界：
 *   ✅ 接收 task → 初始化 Memory → 循环 maxSteps 次
 *   ✅ 每步：调 LLM → 解析 → 执行工具 → 更新 Memory → 触发 Observer
 *   ✅ 两个终止条件：模型不再调工具（完成）/ 达到 maxSteps（保险丝）
 *   ❌ 不直接调 client.chat.completions.create（LLMClient 的活）
 *   ❌ 不直接执行工具函数（ToolRegistry 的活）
 *   ❌ 不解析 JSON/正则（ActionParser 的活）
 *
 * maxSteps 保险丝（第04章）：必填，默认 10，永不设无限。
 * 实现类通过构造函数注入其他 5 个组件（依赖注入）。
 */
interface AgentRunner {
  /** 运行 Agent 循环，返回最终答案。maxSteps 是防无限循环的保险丝。 */
  run(task: string, maxSteps?: number): Promise<string>;
}

// ═══════════════════════════════════════════════════════════════════════
// 依赖注入容器：引用所有 6 大接口，展示第13章如何组装组件
// ═══════════════════════════════════════════════════════════════════════

/** AgentRunner 实现类通过构造函数接收这 5 个组件（依赖注入）。 */
interface FrameworkDeps {
  llm: LLMClient;
  memory: Memory;
  tools: ToolRegistry;
  parser: ActionParser;
  observer: Observer;
}

/** 第13章的 createRunner(deps) 工厂函数签名。 */
type AgentRunnerFactory = (deps: FrameworkDeps) => AgentRunner;

// ═══════════════════════════════════════════════════════════════════════
// 组件注册表（元数据，用于演示打印）
// TS interface 在运行时擦除，所以方法签名用手写字符串记录（不像 Python 能自省）
// ═══════════════════════════════════════════════════════════════════════

interface ComponentMeta {
  name: string;
  description: string;
  methods: string[];
}

const COMPONENTS: ComponentMeta[] = [
  {
    name: "ToolRegistry",
    description:
      "工具注册表：管理工具元数据 + handler，对外提供 register/getSchema/execute",
    methods: [
      "register(name, description, parameters, handler): void",
      "getSchema(): ToolSchema[]",
      "execute(name, args): string",
    ],
  },
  {
    name: "LLMClient",
    description:
      "LLM 包装器：封装调用 + 退避重试，换提供商只换实现",
    methods: [
      "chat(messages, tools?): Promise<ChatResponse>",
      "chatWithRetry(messages, maxRetries?): Promise<ChatResponse>",
    ],
  },
  {
    name: "Memory",
    description:
      "记忆管理：决定模型能看到什么（ConversationBuffer/SummaryMemory/VectorMemory 都可实现）",
    methods: [
      "add(role, content): void",
      "getMessages(): Message[]",
      "clear(): void",
    ],
  },
  {
    name: "ActionParser",
    description:
      "输出解析器：把 LLM 响应解析成 [{name,args,id}]，屏蔽 OpenAI/ReAct/自定义 JSON 差异",
    methods: [
      "parseToolCalls(response): ToolCall[]",
      "hasToolCalls(response): boolean",
    ],
  },
  {
    name: "Observer",
    description:
      "可观测钩子：纯旁路观察（只读不写），记录日志/trace/成本，不侵入主流程",
    methods: [
      "onStepStart(step, messages): void",
      "onLLMCall(messages): void",
      "onToolCall(name, args): void",
      "onToolResult(name, result): void",
      "onStepEnd(step): void",
    ],
  },
  {
    name: "AgentRunner",
    description:
      "Agent 循环引擎：协调其他 5 个组件，maxSteps 保险丝防无限循环",
    methods: ["run(task, maxSteps?): Promise<string>"],
  },
];

// ═══════════════════════════════════════════════════════════════════════
// 演示辅助函数
// ═══════════════════════════════════════════════════════════════════════

function printComponents(): void {
  for (const comp of COMPONENTS) {
    console.log(`OUT:component:${comp.name}:`);
    console.log(`  职责: ${comp.description}`);
    for (const sig of comp.methods) {
      console.log(`  方法: ${sig}`);
    }
    console.log();
  }
}

function printArchitecture(): void {
  console.log("OUT:architecture:");
  console.log(
    `
  ┌─────────────────────────────────────────────────────────────────┐
  │                    AgentRunner（循环引擎）                       │
  │                                                                 │
  │   ┌────────┐        ┌──────────────────────────────────────┐   │
  │   │ Memory │◄──────►│  for (step = 0; step < maxSteps) {   │   │
  │   │ (消息) │        │    observer.onStepStart(step)        │   │
  │   └───┬────┘        │    resp = await llm.chatWithRetry()  │   │
  │       │ messages    │    if (!parser.hasToolCalls(resp))   │   │
  │       ▼             │        return finalAnswer            │   │
  │   ┌──────────┐      │    for (call of parser.parse(resp))  │   │
  │   │LLMClient │◄─────┤        result = tools.execute(call)  │   │
  │   │(调模型)  │      │        memory.add("tool", result)    │   │
  │   └────┬─────┘      │    observer.onStepEnd(step)          │   │
  │        │ response   │  }                                   │   │
  │        ▼            │  └───────────────────────────────────┘   │
  │   ┌────────────┐   actions       ┌─────────────┐              │
  │   │ActionParser│────────────────►│ToolRegistry │              │
  │   │(解析 tool_ │                 │(register/   │              │
  │   │ calls)     │                 │ getSchema/  │              │
  │   └────────────┘                 │ execute)    │              │
  │                                  └─────────────┘              │
  │   ┌────────────────────────────────────────────────────────┐  │
  │   │           Observer（横切所有组件，纯旁路观察）            │  │
  │   │  onStepStart / onLLMCall / onToolCall /                │  │
  │   │  onToolResult / onStepEnd                              │  │
  │   └────────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────┘

  数据流: task → Memory → LLMClient → ActionParser → ToolRegistry → Memory (循环)
  Observer: 在每个关键点挂钩，只读不写，记录日志/trace/成本
`,
  );
}

function printVerify(): void {
  console.log("OUT:verify:");
  console.log("  验证 interface 的编译期类型检查（TS 与 Python Protocol 的差异）:");
  console.log();
  console.log("  TypeScript interface 在运行时擦除，不能用 instanceof 检查。");
  console.log("  但可以用 'implements' 关键字在编译期强制约束：");
  console.log();

  // 演示 implements 的编译期检查（类型层面的验证）
  // 这个类必须实现 ToolRegistry 的全部方法，否则 tsc 报错
  class FakeToolRegistry implements ToolRegistry {
    private readonly store: Tool[] = [];

    register(
      name: string,
      description: string,
      parameters: Record<string, unknown>,
      handler: (...args: string[]) => string,
    ): void {
      this.store.push({ name, description, parameters, handler });
    }

    getSchema(): ToolSchema[] {
      return [];
    }

    execute(_name: string, _args: Record<string, unknown>): string {
      return "";
    }
  }

  const fake = new FakeToolRegistry();
  fake.register(
    "demo",
    "演示工具",
    { type: "object" },
    () => "ok",
  );
  console.log("  class FakeToolRegistry implements ToolRegistry { ... }");
  console.log("  → 编译通过 ✓ （实现了 register/getSchema/execute 全部方法）");
  console.log(`  → 实例化成功: ${fake.constructor.name}`);
  console.log();

  const registry: ToolRegistry = fake;
  console.log(`  const registry: ToolRegistry = fake; (getSchema() → ${registry.getSchema().length} schemas)`);
  console.log("  → 类型兼容 ✓ （结构匹配，TS 是结构性类型系统）");
  console.log();

  const createStubRunner: AgentRunnerFactory = (_deps) => ({
    run: async (_task: string, _maxSteps?: number) => "(stub)",
  });
  console.log("  AgentRunnerFactory 类型引用（第13章 createRunner 的签名预览）:");
  console.log(`  → typeof createStubRunner = ${typeof createStubRunner}`);
  console.log();

  console.log("  6 大组件 interface 编译期约束说明:");
  for (const comp of COMPONENTS) {
    console.log(
      `    ${comp.name.padEnd(14)} → class Foo implements ${comp.name} 时，`,
    );
    console.log(
      `                    缺任何方法都会在 tsc 报错 TS2420 (Class incorrectly implements)`,
    );
  }
  console.log();
  console.log("  结论: TS interface 用 'implements' 做编译期契约约束；");
  console.log("        运行时不保留接口信息（与 Python @runtime_checkable 不同）,");
  console.log("        但能用类型系统在编译期挡住不完整的实现。");
}

function printConfigCheck(): void {
  console.log("OUT:verify:config");
  const masked =
    _cfg.apiKey.length > 4 ? `***${_cfg.apiKey.slice(-4)}` : "(set)";
  console.log(
    `  shared.config import 路径正确 ✓ (provider=${_cfg.provider}, model=${_cfg.model}, key=${masked})`,
  );
}

// ═══════════════════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log("=".repeat(72));
  console.log("第12章 从零造框架 — 架构设计（6 大核心组件接口定义）");
  console.log("本章只定义接口契约（interface），不实现具体逻辑（第13章才实现）");
  console.log("=".repeat(72));
  console.log();

  // Demo 1: 6 大组件的职责 + 方法签名
  console.log("▎ Demo 1: 6 大核心组件接口");
  console.log("-".repeat(72));
  printComponents();

  // Demo 2: ASCII 架构图
  console.log("▎ Demo 2: 架构总览（组件关系图）");
  console.log("-".repeat(72));
  printArchitecture();

  // Demo 3: interface 编译期约束验证
  console.log("▎ Demo 3: 接口验证（interface implements 约束）");
  console.log("-".repeat(72));
  printVerify();
  console.log();

  // Demo 4: 配置 import 路径验证
  console.log("▎ Demo 4: 配置 import 路径验证");
  console.log("-".repeat(72));
  printConfigCheck();

  console.log();
  console.log("=".repeat(72));
  console.log("✓ 本章完成：6 大组件接口已定义。第13章会实现具体逻辑。");
  console.log("=".repeat(72));
}

main().catch((err: unknown) => {
  console.error("运行失败:", err);
  process.exit(1);
});
