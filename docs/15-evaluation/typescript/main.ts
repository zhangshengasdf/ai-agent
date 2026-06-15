/**
 * 第15章 评估与测试（TypeScript 版）
 *
 * 对等 Python 实现。3 个 demo：
 *   Demo 1: 行为测试（Behavior Testing）—— 定义『输入 → 期望工具』，断言 Agent 走对流程
 *   Demo 2: LLM-as-Judge —— 第一个 LLM 生成回答，第二个 LLM 按 rubric 打分
 *   Demo 3: 回归测试套件（Regression Suite）—— 批量运行行为测试，汇总 PASS/FAIL
 *
 * 运行方式：
 *   cd ai-agent/15-evaluation
 *   npx tsx typescript/main.ts
 *
 * 设计：
 *   - 评估对象是前面章节造的 Agent（概念上引用第04/13章）
 *   - 离线 mock Agent 用关键词匹配模拟工具选择决策（可靠、可复现）
 *   - .env 占位符 sk-REPLACE-ME → 真实 API 必失败 → 降级 mock，exit 0
 *   - 全链路 async（TS SDK 只有异步接口）
 *   - 不用 `as any` / `@ts-ignore`；类型不匹配用类型守卫收窄
 */

import OpenAI, {
  APIConnectionError,
  APIError,
  AuthenticationError,
} from "openai";
import { getConfig } from "../../shared/config";

// ═══════════════════════════════════════════════════════════════════════
// 辅助：从 unknown 错误提取类型名（用 instanceof 窄化，避免 e.constructor.name）
// ═══════════════════════════════════════════════════════════════════════

function errorName(e: unknown): string {
  // 用 instanceof 逐一检查（消费 import，符合 TS 类型收窄惯例）
  if (e instanceof APIConnectionError) return "APIConnectionError";
  if (e instanceof AuthenticationError) return "AuthenticationError";
  if (e instanceof APIError) return "APIError";
  return e instanceof Error ? e.constructor.name : "Unknown";
}

// ═══════════════════════════════════════════════════════════════════════
// 类型定义（与 Python dataclass 对等）
// ═══════════════════════════════════════════════════════════════════════

/** Agent 运行结果（含行为轨迹：调了哪些工具 + 最终输出）。 */
interface AgentResult {
  finalOutput: string;
  toolsCalled: string[];
}

/** 行为测试用例：输入任务 → 期望调用的工具列表。 */
interface BehaviorTestCase {
  name: string;
  task: string;
  expectedTools: string[];
  description: string;
}

/** 单个测试的运行结果。 */
interface TestResult {
  name: string;
  passed: boolean;
  detail: string;
}

/** LLM-as-Judge 的评分结果。 */
interface JudgeResult {
  score: number;
  comment: string;
}

/** 回归测试套件的汇总报告。 */
interface RegressionReport {
  results: TestResult[];
  passed: number;
  failed: number;
}

/** Judge 返回的 JSON 结构。 */
interface JudgeJson {
  score: number;
  comment: string;
}

// ═══════════════════════════════════════════════════════════════════════
// 工具定义（OpenAI tools 格式，与第03/13章一致）
// ═══════════════════════════════════════════════════════════════════════

const WEATHER_DB: Record<string, string> = {
  北京: "晴，气温 25°C",
  上海: "多云，气温 28°C",
  广州: "小雨，气温 30°C",
};

const TOOL_DEFS = [
  {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "查询指定城市的天气",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string", description: "城市名" },
        },
        required: ["city"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "calculate",
      description: "计算数学表达式",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "数学表达式" },
        },
        required: ["expression"],
      },
    },
  },
];

// ═══════════════════════════════════════════════════════════════════════
// 离线 Mock Agent：用关键词匹配模拟真实 Agent 的工具选择决策
// ═══════════════════════════════════════════════════════════════════════

const WEATHER_KEYWORDS = ["天气", "气温", "weather", "温度"];
const CALC_KEYWORDS = ["温差", "计算", "算一下", "算", "calculate"];
const CITIES = ["北京", "上海", "广州", "深圳", "杭州"];

class MockAgent {
  /** 离线 mock Agent：模拟第04章/第13章的 Agent 循环。
   *
   * 真实 Agent 用 LLM 决策"调哪个工具"；这里用关键词规则模拟，
   * 保证离线环境下也能完整演示行为测试。
   */

  async run(task: string, maxSteps = 10): Promise<AgentResult> {
    const toolsCalled: string[] = [];

    // ── 模拟 Agent 的"思考"：该调哪些工具？──
    const needsWeather = WEATHER_KEYWORDS.some((kw) => task.includes(kw));
    const needsCalc = CALC_KEYWORDS.some((kw) => task.includes(kw));
    const citiesFound = CITIES.filter((c) => task.includes(c));

    // ── 模拟多步 Agent 循环（observe→reason→act）──
    let steps = 0;

    // step: 为每个城市查天气（用索引循环避免未使用的变量）
    if (needsWeather) {
      const weatherCalls = Math.min(citiesFound.length, maxSteps - steps);
      for (let i = 0; i < weatherCalls; i++) {
        toolsCalled.push("get_weather");
        steps += 1;
      }
    }

    // step: 计算
    if (needsCalc && steps < maxSteps) {
      toolsCalled.push("calculate");
      steps += 1;
    }

    // ── 生成最终输出 ──
    let finalOutput: string;
    if (toolsCalled.length === 0) {
      finalOutput = `你好！我是任务助手。关于『${task}』，有什么我可以帮你的吗？`;
    } else {
      const parts: string[] = [];
      if (needsWeather) {
        for (const city of citiesFound) {
          const weather = WEATHER_DB[city] ?? "未知";
          parts.push(`${city}今天${weather}`);
        }
      }
      if (needsCalc && citiesFound.length >= 2) {
        parts.push("两地温差为 3°C");
      } else if (needsCalc) {
        parts.push("计算完成");
      }
      finalOutput = parts.length > 0 ? parts.join("。") + "。" : "（已处理）";
    }

    return { finalOutput, toolsCalled };
  }
}

// ═══════════════════════════════════════════════════════════════════════
// 真实 API Agent：尝试调 LLM 获取真实工具决策，失败返回 null
// ═══════════════════════════════════════════════════════════════════════

async function runAgentRealApi(
  client: OpenAI,
  model: string,
  task: string,
): Promise<AgentResult | null> {
  /** 尝试用真实 LLM 运行 Agent（单步决策）。失败返回 null（降级 mock）。 */
  const systemPrompt =
    "你是任务助手 Agent。根据用户任务决定是否调用工具。" +
    "可用工具：get_weather（查天气）、calculate（计算）。" +
    "如果需要查天气或计算，请调用相应工具；如果是闲聊，直接回答。";
  try {
    const resp = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: task },
      ],
      tools: TOOL_DEFS,
      tool_choice: "auto",
    });

    const msg = resp.choices[0].message;
    const toolsCalled: string[] = [];
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        // discriminated union: 必须检查 type === "function"
        if (tc.type !== "function") continue;
        toolsCalled.push(tc.function.name);
      }
    }

    return {
      finalOutput: msg.content ?? "",
      toolsCalled,
    };
  } catch (e) {
    console.log(
      `OUT:agent:offline: 真实 API 不可用（${errorName(e)}），降级 mock Agent`,
    );
    return null;
  }
}

async function runAgent(task: string): Promise<AgentResult> {
  /** 运行 Agent：优先真实 API，失败降级 mock。 */
  const cfg = getConfig();
  const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });
  const result = await runAgentRealApi(client, cfg.model, task);
  if (result !== null) {
    return result;
  }
  return new MockAgent().run(task);
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 1: 行为测试（Behavior Testing）
// ═══════════════════════════════════════════════════════════════════════

async function runBehaviorTest(
  testCase: BehaviorTestCase,
): Promise<TestResult> {
  /** 跑单个行为测试：运行 Agent → 断言期望工具被调用。 */
  console.log(`OUT:test:${testCase.name}: ▶ 运行测试: ${testCase.name}`);
  console.log(`  任务: ${testCase.task}`);
  console.log(`  期望工具: [${testCase.expectedTools.join(", ")}]`);
  console.log(`  验证点: ${testCase.description}`);

  const result = await runAgent(testCase.task);
  const actual = result.toolsCalled;

  // ── 行为断言 ──
  let passed: boolean;
  let detail: string;

  if (testCase.expectedTools.length > 0) {
    // 期望调用了某些工具 → 检查是否都调了
    const missing = testCase.expectedTools.filter(
      (t) => !actual.includes(t),
    );
    passed = missing.length === 0;
    detail = `期望 ${JSON.stringify(testCase.expectedTools)}，实际 ${JSON.stringify(actual)}` +
      (missing.length > 0 ? `（缺少 ${JSON.stringify(missing)}）` : "");
  } else {
    // 期望不调工具（闲聊）→ 检查是否真的没调
    passed = actual.length === 0;
    detail = `期望不调工具，实际 ${JSON.stringify(actual)}`;
  }

  if (passed) {
    console.log(`OUT:test:${testCase.name}: ✓ 通过 — 实际调用 [${actual.join(", ")}]`);
  } else {
    console.log(`OUT:test:${testCase.name}: ✗ 失败 — ${detail}`);
  }

  console.log(`  输出: ${result.finalOutput.slice(0, 60)}`);
  console.log("");
  return { name: testCase.name, passed, detail };
}

async function demoBehaviorTesting(): Promise<void> {
  /** Demo 1: 行为测试。 */
  console.log("=".repeat(72));
  console.log("Demo 1: 行为测试（Behavior Testing）");
  console.log("  定义『输入 → 期望工具』，断言 Agent 走对了流程。");
  console.log("  价值：抓住『输出碰巧对但行为错』的隐蔽 bug。");
  console.log("=".repeat(72));
  console.log("");

  // ── 测试用例集（黄金用例 + 边界用例）──
  const testCases: BehaviorTestCase[] = [
    {
      name: "weather_query",
      task: "查一下北京今天的天气",
      expectedTools: ["get_weather"],
      description: "天气查询任务应调用 get_weather 工具",
    },
    {
      name: "weather_temp_calc",
      task: "查北京和上海的天气，然后算一下两地温差",
      expectedTools: ["get_weather", "calculate"],
      description: "温差任务应同时调用天气和计算工具",
    },
    {
      name: "no_tool_needed",
      task: "你好，谢谢你",
      expectedTools: [],
      description: "纯闲聊不应调用任何工具",
    },
  ];

  const results: TestResult[] = [];
  for (const tc of testCases) {
    results.push(await runBehaviorTest(tc));
  }

  // ── 汇总 ──
  const passed = results.filter((r) => r.passed).length;
  const failed = results.length - passed;
  console.log("-".repeat(72));
  console.log(`OUT:test:summary: ${passed}/${results.length} 通过，${failed} 失败`);
  console.log("");
  console.log("  💡 行为测试不只看最终输出，更看 Agent『走了什么路』。");
  console.log("     如果只测输出，Agent 碰巧猜对答案但你不知道它根本没调工具。");
  console.log("");
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 2: LLM-as-Judge（用模型评估输出质量）
// ═══════════════════════════════════════════════════════════════════════

const JUDGE_SYSTEM_PROMPT = `你是一个严格的评分员。请对 AI 助手的回答打分（1-5 分）。

评分维度：
- 正确性：事实是否准确
- 完整性：是否覆盖了任务要求的所有方面
- 清晰度：表述是否清楚易懂

评分标准：
- 5 分：完全正确、完整、清晰
- 4 分：基本正确，有小瑕疵
- 3 分：部分正确，有明显遗漏
- 2 分：大部分错误
- 1 分：完全错误或无关

只输出 JSON，格式：{"score": 1-5的整数, "comment": "评语"}`;

async function generateAnswerRealApi(
  client: OpenAI,
  model: string,
  task: string,
): Promise<string | null> {
  /** 尝试用真实 LLM 生成回答。失败返回 null。 */
  try {
    const resp = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: "你是知识渊博的助手。简洁准确地回答问题。" },
        { role: "user", content: task },
      ],
    });
    return resp.choices[0].message.content ?? "";
  } catch (e) {
    console.log(
      `OUT:judge:offline: 候选生成 API 不可用（${errorName(e)}），降级 mock`,
    );
    return null;
  }
}

/** 手动 type guard：收窄 unknown → JudgeJson（TS 无运行时校验，需手动检查）。 */
function isJudgeJson(raw: unknown): raw is JudgeJson {
  if (typeof raw !== "object" || raw === null) return false;
  const obj = raw as Record<string, unknown>;
  return (
    typeof obj["score"] === "number" &&
    typeof obj["comment"] === "string"
  );
}

async function judgeRealApi(
  client: OpenAI,
  model: string,
  task: string,
  candidate: string,
): Promise<JudgeResult | null> {
  /** 尝试用真实 LLM 做 Judge。失败返回 null。 */
  try {
    const resp = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: JUDGE_SYSTEM_PROMPT },
        { role: "user", content: `任务: ${task}\n\n回答: ${candidate}` },
      ],
      response_format: { type: "json_object" },
    });
    const raw = resp.choices[0].message.content ?? "{}";
    const parsed: unknown = JSON.parse(raw);
    if (!isJudgeJson(parsed)) {
      console.log("OUT:judge:offline: Judge 返回格式不符，降级 mock");
      return null;
    }
    return { score: parsed.score, comment: parsed.comment };
  } catch (e) {
    console.log(
      `OUT:judge:offline: Judge API 不可用（${errorName(e)}），降级 mock`,
    );
    return null;
  }
}

function judgeMock(task: string, candidate: string): JudgeResult {
  /** 离线 mock Judge：预设评分（模拟真实 Judge 的判断）。 */
  let score = 3; // 默认中等
  const comments: string[] = [];

  // 维度 1：长度（太短通常不完整）
  if (candidate.length < 30) {
    score = 2;
    comments.push("回答过于简短，完整性不足");
  } else if (candidate.length > 100) {
    score = Math.min(score + 1, 5);
    comments.push("回答详尽，覆盖面广");
  }

  // 维度 2：关键词覆盖
  const taskKeywords = ["递归", "函数", "自身", "基线", "终止"].filter((w) =>
    task.includes(w)
  );
  const covered = taskKeywords.filter((kw) => candidate.includes(kw));
  if (taskKeywords.length > 0 && covered.length >= taskKeywords.length * 0.6) {
    score = Math.min(score + 1, 5);
    comments.push(`覆盖了关键概念（${covered.length}/${taskKeywords.length}）`);
  } else if (taskKeywords.length > 0) {
    comments.push(`关键概念覆盖不足（${covered.length}/${taskKeywords.length}）`);
  }

  // 维度 3：是否有举例
  if (candidate.includes("例如") || candidate.includes("比如") || candidate.includes("举例")) {
    score = Math.min(score + 1, 5);
    comments.push("有具体举例，清晰度高");
  }

  score = Math.max(1, Math.min(5, score));
  if (comments.length === 0) {
    comments.push("回答基本合格，但缺少亮点");
  }

  return { score, comment: comments.join("；") };
}

async function demoLlmJudge(): Promise<void> {
  /** Demo 2: LLM-as-Judge。 */
  console.log("=".repeat(72));
  console.log("Demo 2: LLM-as-Judge（用模型评估输出质量）");
  console.log("  第一个 LLM 生成回答，第二个 LLM（Judge）按 rubric 打分。");
  console.log("  价值：评估『输出好不好』这种难以 assert 的质量维度。");
  console.log("=".repeat(72));
  console.log("");

  const cfg = getConfig();
  const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

  // ── 待评估的任务 + 候选回答 ──
  const task = "请解释什么是递归，并给出一个例子";
  console.log(`OUT:judge:task: ${task}`);

  // 优先真实 API 生成候选回答，失败用预设回答
  let candidate = await generateAnswerRealApi(client, cfg.model, task);
  if (candidate === null) {
    candidate =
      "递归是一种编程技巧，指函数在执行过程中调用自身。" +
      "例如，计算阶乘时，factorial(n) = n * factorial(n-1)，" +
      "直到 n=1 时返回 1（基线条件）。递归可以把复杂问题分解为更小的同类问题。";
  }
  console.log(`OUT:judge:candidate: ${candidate.slice(0, 80)}...`);
  console.log("");

  // ── Judge 评分 ──
  console.log("  评分标准: 正确性 + 完整性 + 清晰度（1-5 分）");
  console.log("-".repeat(72));

  let judge = await judgeRealApi(client, cfg.model, task, candidate);
  if (judge === null) {
    judge = judgeMock(task, candidate);
  }

  console.log(`OUT:judge:score: ${judge.score}/5`);
  console.log(`OUT:judge:comment: ${judge.comment}`);
  console.log("");
  console.log("  💡 LLM-as-Judge 适合评估『回答好不好』这种难以 assert 的维度。");
  console.log("     但要注意偏见：Judge 偏向冗长回答、偏向和自己同款的回答。");
  console.log("     生产中要结合行为测试 + 人工抽检，不能只依赖 Judge。");
  console.log("");
}

// ═══════════════════════════════════════════════════════════════════════
// Demo 3: 回归测试套件（Regression Suite）
// ═══════════════════════════════════════════════════════════════════════

class RegressionSuite {
  /** 回归测试套件：批量运行行为测试，汇总 PASS/FAIL 报告。 */

  private cases: BehaviorTestCase[] = [];

  add(testCase: BehaviorTestCase): void {
    this.cases.push(testCase);
  }

  get length(): number {
    return this.cases.length;
  }

  async runAll(): Promise<RegressionReport> {
    const report: RegressionReport = { results: [], passed: 0, failed: 0 };
    for (let i = 0; i < this.cases.length; i++) {
      const tc = this.cases[i];
      process.stdout.write(
        `OUT:regression:case:${i + 1} ${tc.name} ... `,
      );
      const result = await runBehaviorTestSilent(tc);
      const status = result.passed ? "PASS" : "FAIL";
      console.log(`${status}`);
      report.results.push(result);
      if (result.passed) {
        report.passed += 1;
      } else {
        report.failed += 1;
      }
    }
    return report;
  }
}

async function runBehaviorTestSilent(
  testCase: BehaviorTestCase,
): Promise<TestResult> {
  /** 静默版行为测试（不打印详情，用于回归套件批量运行）。 */
  const result = await runAgent(testCase.task);
  const actual = result.toolsCalled;

  if (testCase.expectedTools.length > 0) {
    const missing = testCase.expectedTools.filter(
      (t) => !actual.includes(t),
    );
    const passed = missing.length === 0;
    const detail = `期望 ${JSON.stringify(testCase.expectedTools)}，实际 ${JSON.stringify(actual)}` +
      (missing.length > 0 ? `（缺少 ${JSON.stringify(missing)}）` : "");
    return { name: testCase.name, passed, detail };
  }

  const passed = actual.length === 0;
  const detail = `期望不调工具，实际 ${JSON.stringify(actual)}`;
  return { name: testCase.name, passed, detail };
}

async function demoRegressionSuite(): Promise<void> {
  /** Demo 3: 回归测试套件。 */
  console.log("=".repeat(72));
  console.log("Demo 3: 回归测试套件（Regression Suite）");
  console.log("  批量运行行为测试，汇总 PASS/FAIL 报告。");
  console.log("  价值：改 prompt / 换模型后，5 秒知道有没有破坏行为。");
  console.log("=".repeat(72));
  console.log("");

  // ── 构建套件（黄金用例 + 边界用例）──
  const suite = new RegressionSuite();
  suite.add({
    name: "weather_query_single",
    task: "查北京天气",
    expectedTools: ["get_weather"],
    description: "单城市天气查询",
  });
  suite.add({
    name: "weather_query_multi",
    task: "查北京和上海的天气",
    expectedTools: ["get_weather"],
    description: "多城市天气查询",
  });
  suite.add({
    name: "weather_and_calc",
    task: "查北京和上海天气，算温差",
    expectedTools: ["get_weather", "calculate"],
    description: "天气 + 计算组合任务",
  });
  suite.add({
    name: "chitchat_no_tool",
    task: "你好呀",
    expectedTools: [],
    description: "闲聊不应调工具",
  });
  suite.add({
    name: "calc_only",
    task: "帮我算一下 28 减 25",
    expectedTools: ["calculate"],
    description: "纯计算任务",
  });

  // ── 运行套件 ──
  console.log(`OUT:regression:running: 共 ${suite.length} 个测试用例`);
  console.log("-".repeat(72));

  const report = await suite.runAll();

  // ── 汇总报告 ──
  console.log("-".repeat(72));
  const total = report.passed + report.failed;
  if (report.failed === 0) {
    console.log(
      `OUT:regression:summary: ${report.passed}/${total} 通过，${report.failed} 失败 ✓ 全部通过`,
    );
  } else {
    console.log(
      `OUT:regression:summary: ${report.passed}/${total} 通过，${report.failed} 失败 ✗ 有失败`,
    );
    console.log("");
    console.log("失败详情:");
    for (const r of report.results) {
      if (!r.passed) {
        console.log(`  ✗ ${r.name}: ${r.detail}`);
      }
    }
  }

  console.log("");
  console.log("  💡 把这套测试纳入 CI：每次改 prompt / 换模型 / 升 SDK，自动跑一遍。");
  console.log("     没有回归测试的 Agent 项目，每次改动都在玩俄罗斯轮盘赌。");
  console.log("");
}

// ═══════════════════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log("=".repeat(72));
  console.log("第15章 评估与测试");
  console.log("  行为测试 / LLM-as-Judge / 回归测试套件");
  console.log("  （评估对象是前面章节造的 Agent，用 mock 模拟保证离线可跑）");
  console.log("=".repeat(72));
  console.log("");

  // Demo 1: 行为测试
  await demoBehaviorTesting();

  // Demo 2: LLM-as-Judge
  await demoLlmJudge();

  // Demo 3: 回归测试套件
  await demoRegressionSuite();

  console.log("=".repeat(72));
  console.log("✓ 本章完成：三大评估手段全部演示完毕。");
  console.log("  核心收获：Agent 评估 ≠ 模型评估 —— 要测行为序列，不只测单次输出。");
  console.log("=".repeat(72));
}

main().catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err);
  console.error(`[fatal] ${message}`);
  process.exit(1);
});
