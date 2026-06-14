/**
 * 项目2 · 编程/代码 Agent（工具调用 + 代码执行沙箱）
 *
 * 综合实战：构建一个能"读代码→写代码→运行测试→看输出→修 bug→循环"的编程 Agent。
 *
 * 核心组件：
 *   - Agent Loop：LLM 决定调用工具 → 执行 → 反馈 → 循环
 *   - 沙箱执行：child_process.execSync + timeout + 危险代码拦截
 *   - 4 个工具：read_file / write_file / run_test / list_files
 *   - 自我纠正：测试失败 → 把错误反馈给 LLM → 修 bug 重试
 *   - 离线 Mock：预设 mock 代码+测试，演示完整修复流程
 */

import OpenAI from "openai";
import { getConfig } from "../../../shared/config";
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { resolve, join } from "node:path";
import { execSync } from "node:child_process";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// 工作区目录（workspace/ 在项目根目录下）
const WORKSPACE = resolve(__dirname, "..", "workspace");

// 最大 Agent 循环步数
const MAX_STEPS = 10;

// ── 沙箱安全：危险代码关键词 ─────────────────────────────────────
const DANGEROUS_KEYWORDS: readonly string[] = [
  "rm -rf", "rm /", "rmdir /", "shutil.rmtree",
  "os.system", "os.popen", "os.exec", "os.spawn",
  "import subprocess", "from subprocess",
  "import socket", "from socket",
  "urllib.request.urlopen",
  "sys.exit", "os._exit",
  "__import__('os')", "__import__('subprocess')",
  "eval(", "exec(",
  "open('/etc", "open('/proc",
];

function checkCodeSafety(code: string): boolean {
  const lower = code.toLowerCase();
  for (const keyword of DANGEROUS_KEYWORDS) {
    if (lower.includes(keyword.toLowerCase())) {
      return false;
    }
  }
  return true;
}

// ════════════════════════════════════════════════════════════════════
// 1. 沙箱代码执行
// ════════════════════════════════════════════════════════════════════

interface SandboxResult {
  stdout: string;
  stderr: string;
  returnCode: number;
  timedOut: boolean;
}

function sandboxExecute(code: string, timeoutSec = 5, shell = false): SandboxResult {
  if (!checkCodeSafety(code)) {
    return {
      stdout: "",
      stderr: "BLOCKED: 代码包含危险操作，已被沙箱拦截。",
      returnCode: -1,
      timedOut: false,
    };
  }

  const cmd = shell ? code : `python3 -c ${JSON.stringify(code)}`;

  try {
    const output = execSync(cmd, {
      cwd: WORKSPACE,
      timeout: timeoutSec * 1000,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    return { stdout: output, stderr: "", returnCode: 0, timedOut: false };
  } catch (err: unknown) {
    const error = err as { stderr?: string; stdout?: string; message?: string; status?: number };
    if (error.message?.includes("ETIMEDOUT") || error.message?.includes("timeout")) {
      return {
        stdout: "",
        stderr: `TIMEOUT: 代码执行超过 ${timeoutSec} 秒。`,
        returnCode: -1,
        timedOut: true,
      };
    }
    return {
      stdout: error.stdout ?? "",
      stderr: error.stderr ?? error.message ?? "Unknown error",
      returnCode: error.status ?? -1,
      timedOut: false,
    };
  }
}

// ════════════════════════════════════════════════════════════════════
// 2. 工具定义 + 实现
// ════════════════════════════════════════════════════════════════════

const TOOLS: OpenAI.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "read_file",
      description: "读取工作区中的文件内容。",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "相对于工作区的文件路径" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description: "向工作区写入文件内容（覆盖已有内容）。",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "相对于工作区的文件路径" },
          content: { type: "string", description: "要写入的文件内容" },
        },
        required: ["path", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_test",
      description: "在沙箱中运行测试命令并返回输出。",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "要执行的测试命令" },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_files",
      description: "列出工作区目录下的文件。",
      parameters: {
        type: "object",
        properties: {
          dir: { type: "string", description: "相对于工作区的目录路径，默认 '.'" },
        },
      },
    },
  },
];

function executeTool(name: string, args: Record<string, unknown>): string {
  if (name === "read_file") {
    const path = String(args.path ?? "");
    const fullPath = join(WORKSPACE, path);
    if (!existsSync(fullPath)) {
      return `ERROR: 文件不存在: ${path}`;
    }
    const content = readFileSync(fullPath, "utf-8");
    console.log(`OUT:tool: 读取文件: ${path} (${content.length} 字符)`);
    return content;
  }

  if (name === "write_file") {
    const path = String(args.path ?? "");
    const content = String(args.content ?? "");
    const fullPath = join(WORKSPACE, path);
    const dir = fullPath.substring(0, fullPath.lastIndexOf("/"));
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    writeFileSync(fullPath, content, "utf-8");
    console.log(`OUT:tool: 写入文件: ${path} (${content.length} 字符)`);
    return `OK: 已写入 ${path}`;
  }

  if (name === "run_test") {
    const command = String(args.command ?? "");
    console.log(`OUT:tool: 运行测试: ${command}`);
    const result = sandboxExecute(command, 5, true);
    let output = "";
    if (result.stdout) output += result.stdout;
    if (result.stderr) output += (output ? "\n" : "") + result.stderr;
    if (result.returnCode !== 0) {
      output = `EXIT CODE: ${result.returnCode}\n${output}`;
    }
    return output || "(无输出)";
  }

  if (name === "list_files") {
    const dirPath = String(args.dir ?? ".");
    const fullPath = join(WORKSPACE, dirPath);
    if (!existsSync(fullPath)) {
      return `ERROR: 目录不存在: ${dirPath}`;
    }
    const entries = readdirSync(fullPath).sort();
    console.log(`OUT:tool: 列出文件: ${dirPath} (${entries.length} 项)`);
    return entries.join("\n");
  }

  return `ERROR: 未知工具: ${name}`;
}

// ════════════════════════════════════════════════════════════════════
// 3. LLM 调用封装（带 try/catch 降级）
// ════════════════════════════════════════════════════════════════════

interface LLMResult {
  content: string;
  toolCalls: { name: string; arguments: string }[];
}

async function llmChat(
  messages: OpenAI.ChatCompletionMessageParam[],
  tools?: OpenAI.ChatCompletionTool[],
): Promise<LLMResult> {
  try {
    const resp = await client.chat.completions.create({
      model: cfg.model,
      messages,
      ...(tools ? { tools } : {}),
    });
    const msg = resp.choices[0].message;
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
    return { content: "", toolCalls: [] };
  }
}

// ════════════════════════════════════════════════════════════════════
// 4. Agent Loop（核心）
// ════════════════════════════════════════════════════════════════════

async function codingAgent(task: string, maxSteps = MAX_STEPS): Promise<string> {
  console.log("\nOUT:agent: ══ 编程 Agent ══");
  console.log(`OUT:agent: 任务: ${task}`);
  console.log(`OUT:agent: 最大步数: ${maxSteps}`);

  const systemPrompt =
    "你是一个编程助手 Agent。你的任务是通过工具调用来完成编程任务。\n" +
    "可用工具：\n" +
    "  - read_file(path): 读取文件\n" +
    "  - write_file(path, content): 写入文件\n" +
    "  - run_test(command): 运行测试\n" +
    "  - list_files(dir): 列出文件\n\n" +
    "工作流程：\n" +
    "1. 先用 list_files 了解项目结构\n" +
    "2. 用 read_file 读取相关文件\n" +
    "3. 用 write_file 写入代码\n" +
    "4. 用 run_test 运行测试\n" +
    "5. 如果测试失败，分析错误并修复代码，然后重新测试\n" +
    "6. 重复直到测试通过\n\n" +
    "重要：每次只做一步操作，观察结果后再决定下一步。";

  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: task },
  ];

  let finalOutput = "";

  for (let step = 1; step <= maxSteps; step++) {
    console.log(`\nOUT:agent: ── 步骤 ${step}/${maxSteps} ──`);

    const resp = await llmChat(messages, TOOLS);
    const content = resp.content;
    const toolCalls = resp.toolCalls;

    if (content) {
      console.log(`OUT:agent: LLM: ${content.slice(0, 200)}`);
      finalOutput = content;
    }

    if (toolCalls.length === 0) {
      console.log("OUT:agent: 无工具调用，结束循环");
      break;
    }

    // 执行工具调用
    for (const tc of toolCalls) {
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(tc.arguments) as Record<string, unknown>;
      } catch {
        // empty args
      }

      console.log(
        `OUT:agent: 调用工具: ${tc.name}(${JSON.stringify(args).slice(0, 100)})`,
      );
      const result = executeTool(tc.name, args);

      // 把工具结果反馈给 LLM
      messages.push({ role: "assistant", content: content || "" });
      messages.push({
        role: "user",
        content: `工具 ${tc.name} 的输出:\n${result}`,
      });

      // 检查测试是否通过
      if (tc.name === "run_test" && result.includes("All tests passed")) {
        console.log("OUT:agent: ✅ 测试通过!");
        return result;
      }
    }
  }

  return finalOutput;
}

// ════════════════════════════════════════════════════════════════════
// 5. 离线 Mock Agent（完整演示修复流程）
// ════════════════════════════════════════════════════════════════════

function mockCodingAgent(): void {
  console.log("\nOUT:mock: ══ 离线 Mock Agent ══");
  console.log("OUT:mock: 任务: 实现 add 函数，通过所有测试");

  // 步骤1：列出文件
  console.log("\nOUT:mock: ── 步骤 1: 列出工作区文件 ──");
  const files = executeTool("list_files", { dir: "." });
  console.log(`OUT:mock: 文件列表:\n${files}`);

  // 步骤2：读取测试文件
  console.log("\nOUT:mock: ── 步骤 2: 读取测试文件 ──");
  const testContent = executeTool("read_file", { path: "test_add.py" });
  console.log(`OUT:mock: 测试内容:\n${testContent}`);

  // 步骤3：读取当前 main.py（初始空实现）
  console.log("\nOUT:mock: ── 步骤 3: 读取当前实现 ──");
  const mainContent = executeTool("read_file", { path: "main.py" });
  console.log(`OUT:mock: 当前 main.py:\n${mainContent}`);

  // 步骤4：写入第一个实现（故意有 bug）
  console.log("\nOUT:mock: ── 步骤 4: 写入实现（有 bug） ──");
  const buggyCode = "def add(a, b):\n    return a - b  # bug: 应该是 +\n";
  executeTool("write_file", { path: "main.py", content: buggyCode });

  // 步骤5：运行测试（预期失败）
  console.log("\nOUT:mock: ── 步骤 5: 运行测试（预期失败） ──");
  const failResult = executeTool("run_test", { command: "python3 test_add.py" });
  console.log(`OUT:mock: 测试结果:\n${failResult}`);

  // 步骤6：分析错误并修复
  console.log("\nOUT:mock: ── 步骤 6: 分析错误并修复 ──");
  console.log("OUT:mock: LLM 分析: add(2,3) 返回 -1 而不是 5，应该是 a + b 而非 a - b");
  const fixedCode = "def add(a, b):\n    return a + b\n";
  executeTool("write_file", { path: "main.py", content: fixedCode });

  // 步骤7：重新运行测试（预期通过）
  console.log("\nOUT:mock: ── 步骤 7: 重新运行测试 ──");
  const passResult = executeTool("run_test", { command: "python3 test_add.py" });
  console.log(`OUT:mock: 测试结果:\n${passResult}`);

  console.log("\nOUT:mock: ══ Mock 完成 ══");
}

// ════════════════════════════════════════════════════════════════════
// 6. 沙箱演示
// ════════════════════════════════════════════════════════════════════

function demoSandbox(): void {
  console.log("\nOUT:sandbox: ══ 沙箱安全演示 ══");

  // 安全代码
  console.log("\nOUT:sandbox: ── 安全代码执行 ──");
  const safeCodes = [
    "print(1 + 1)",
    "print('Hello, Agent!')",
    "import math; print(math.sqrt(16))",
  ];
  for (const code of safeCodes) {
    console.log(`\nOUT:sandbox: 代码: ${code}`);
    const result = sandboxExecute(code);
    console.log(`OUT:sandbox: 输出: ${result.stdout.trim()}`);
    console.log(`OUT:sandbox: 状态: ${result.returnCode === 0 ? "✅ 安全" : "❌ 失败"}`);
  }

  // 危险代码
  console.log("\nOUT:sandbox: ── 危险代码拦截 ──");
  const dangerousCodes = [
    "import os; os.system('rm -rf /')",
    "import subprocess; subprocess.run(['rm', '-rf', '/'])",
    "from socket import socket",
    "exec('import os; os.system(\"ls\")')",
  ];
  for (const code of dangerousCodes) {
    console.log(`\nOUT:sandbox: 代码: ${code}`);
    const result = sandboxExecute(code);
    console.log(`OUT:sandbox: 输出: ${result.stderr.slice(0, 100)}`);
    console.log(`OUT:sandbox: 状态: ${result.returnCode === -1 ? "🛡️ 已拦截" : "⚠️ 未拦截"}`);
  }
}

// ════════════════════════════════════════════════════════════════════
// 7. 主函数
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  console.log("OUT: ══ 编程/代码 Agent ══");

  // 1. 沙箱演示
  demoSandbox();

  // 2. 确保 workspace 存在且有 mock 文件
  if (!existsSync(WORKSPACE)) {
    console.log("\nOUT:agent: workspace 不存在，跳过 Agent 演示");
    return;
  }

  // 3. 尝试真实 LLM，失败则用 mock
  console.log("\nOUT:agent: 尝试调用 LLM...");
  try {
    const resp = await client.chat.completions.create({
      model: cfg.model,
      messages: [{ role: "user", content: "say ok" }],
      max_tokens: 5,
    });
    console.log(`OUT:agent: LLM 连接成功: ${resp.choices[0].message.content}`);
    // 真实 Agent 循环
    await codingAgent(
      "实现 workspace/main.py 中的 add 函数，使 test_add.py 中的所有测试通过。",
    );
  } catch {
    console.log("OUT:agent: LLM 不可用，进入离线 mock 模式");
    mockCodingAgent();
  }

  console.log("\nOUT: ══ 编程 Agent 完成 ══");
}

main();
