/**
 * 项目3 · 多 Agent 代码审查系统（Supervisor-Worker 协作）
 *
 * 综合实战：Supervisor 接收代码 → 分派给 3 个 Reviewer Agent → 各自从专门维度审查 → 汇总报告。
 *
 * 核心组件：
 *   - Supervisor：协调者，分派任务、收集结果、汇总排序
 *   - SecurityReviewer：检查 SQL 注入、硬编码密码、XSS
 *   - PerformanceReviewer：检查 O(n²) 循环、不必要拷贝
 *   - StyleReviewer：检查命名规范、注释、代码结构
 *   - 离线 Mock：预设含问题的代码片段，各 Reviewer 用正则规则审查，exit 0
 */

import OpenAI from "openai";
import { getConfig } from "../../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// ════════════════════════════════════════════════════════════════════
// 1. 数据结构
// ════════════════════════════════════════════════════════════════════

enum Severity {
  CRITICAL = 1,
  WARNING = 2,
  INFO = 3,
}

interface Finding {
  severity: Severity;
  category: string;   // security | performance | style
  rule: string;
  line: number;       // 0 表示全局
  message: string;
  suggestion: string;
}

interface ReviewResult {
  reviewer: string;
  findings: Finding[];
  durationMs: number;
  usedLlm: boolean;
}

// ════════════════════════════════════════════════════════════════════
// 2. 预设含问题的代码片段（离线 mock 用）
// ════════════════════════════════════════════════════════════════════

const MOCK_CODE_SNIPPET = `\
import sqlite3

def get_user(user_id):
    """获取用户信息。"""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    # SQL 注入风险：直接拼接用户输入
    cursor.execute("SELECT * FROM users WHERE id = '%s'" % user_id)
    row = cursor.fetchone()
    conn.close()
    return row

password = "admin123"

def authenticate(username, pwd):
    if pwd == password:
        return True
    return False

def find_duplicates(items):
    """查找重复元素 — O(n²) 复杂度。"""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

def process(data):
    temp = []
    for x in data:
        temp.append(x * 2)
    result = temp
    return result

def render_page(title, content):
    """渲染 HTML 页面。"""
    html = "<h1>" + title + "</h1>"
    html += "<div>" + content + "</div>"
    return html
`;

// ════════════════════════════════════════════════════════════════════
// 3. LLM 调用封装（带 try/catch 降级）
// ════════════════════════════════════════════════════════════════════

async function llmChat(
  messages: OpenAI.ChatCompletionMessageParam[],
): Promise<string> {
  try {
    const resp = await client.chat.completions.create({
      model: cfg.model,
      messages,
    });
    return resp.choices[0].message.content ?? "";
  } catch {
    return "";
  }
}

// ════════════════════════════════════════════════════════════════════
// 4. Reviewer Agent 基类 + 3 个专门 Reviewer
// ════════════════════════════════════════════════════════════════════

abstract class ReviewerAgent {
  abstract readonly name: string;
  abstract readonly systemPrompt: string;

  async review(code: string): Promise<ReviewResult> {
    const t0 = performance.now();
    const findings = await this.tryLlmReview(code);
    const usedLlm = findings.length > 0;
    const finalFindings = usedLlm ? findings : this.mockReview(code);
    const durationMs = performance.now() - t0;
    return { reviewer: this.name, findings: finalFindings, durationMs, usedLlm };
  }

  private async tryLlmReview(code: string): Promise<Finding[]> {
    const resp = await llmChat([
      { role: "system", content: this.systemPrompt },
      {
        role: "user",
        content:
          "请审查以下代码，输出 JSON 数组，每个元素包含 " +
          "severity(Critical/Warning/Info)、rule、line、message、suggestion。\n\n" +
          "```python\n" + code + "\n```",
      },
    ]);
    if (!resp) return [];

    try {
      let jsonStr = resp;
      if (resp.includes("```")) {
        for (const line of resp.split("\n")) {
          const trimmed = line.trim();
          if (trimmed.startsWith("[")) {
            jsonStr = trimmed;
            break;
          }
        }
      }
      const items = JSON.parse(jsonStr) as LlmFinding[];
      if (!Array.isArray(items)) return [];

      return items.map((item) => {
        const sevStr = (item.severity ?? "Info").toUpperCase();
        const sev =
          sevStr === "CRITICAL"
            ? Severity.CRITICAL
            : sevStr === "WARNING"
              ? Severity.WARNING
              : Severity.INFO;
        return {
          severity: sev,
          category: this.name,
          rule: item.rule ?? "unknown",
          line: item.line ?? 0,
          message: item.message ?? "",
          suggestion: item.suggestion ?? "",
        };
      });
    } catch {
      return [];
    }
  }

  protected abstract mockReview(code: string): Finding[];
}

interface LlmFinding {
  severity?: string;
  rule?: string;
  line?: number;
  message?: string;
  suggestion?: string;
}

// ── SecurityReviewer ──────────────────────────────────────────────

class SecurityReviewer extends ReviewerAgent {
  readonly name = "security";
  readonly systemPrompt =
    "你是一个安全审查专家。专门检查以下安全问题：\n" +
    "1. SQL 注入：字符串拼接构造 SQL\n" +
    "2. 硬编码密码/密钥\n" +
    "3. XSS：未转义的用户输入拼接 HTML\n" +
    "输出 JSON 数组格式的审查结果。";

  protected mockReview(code: string): Finding[] {
    const findings: Finding[] = [];
    const lines = code.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineNum = i + 1;

      // SQL 注入：execute + 字符串格式化
      if (/execute\s*\(.*%/.test(line) || /execute\s*\(.*\.format/.test(line)) {
        findings.push({
          severity: Severity.CRITICAL,
          category: "security",
          rule: "sql-injection",
          line: lineNum,
          message: "SQL 注入风险：使用字符串格式化拼接 SQL 查询",
          suggestion:
            "使用参数化查询：cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        });
      }

      // 硬编码密码
      if (
        /(password|passwd|secret|api_key|token)\s*=\s*["'][^"']{3,}["']/i.test(
          line,
        )
      ) {
        findings.push({
          severity: Severity.CRITICAL,
          category: "security",
          rule: "hardcoded-secret",
          line: lineNum,
          message: "硬编码密码/密钥：敏感信息不应直接写在代码中",
          suggestion:
            "使用环境变量或密钥管理服务：os.environ.get('DB_PASSWORD')",
        });
      }
    }

    return findings;
  }
}

// ── PerformanceReviewer ──────────────────────────────────────────

class PerformanceReviewer extends ReviewerAgent {
  readonly name = "performance";
  readonly systemPrompt =
    "你是一个性能审查专家。专门检查以下性能问题：\n" +
    "1. O(n²) 或更高复杂度的嵌套循环\n" +
    "2. 不必要的列表拷贝或重复计算\n" +
    "3. 可用集合/字典优化的线性查找\n" +
    "输出 JSON 数组格式的审查结果。";

  protected mockReview(code: string): Finding[] {
    const findings: Finding[] = [];
    const lines = code.split("\n");

    // 检测嵌套 for 循环（O(n²)）
    const forPattern = /^(\s*)for\s+.+\s+in\s+/;
    let outerIndent = -1;
    let outerLine = -1;

    for (let i = 0; i < lines.length; i++) {
      const match = forPattern.exec(lines[i]);
      if (match) {
        const indent = match[1].length;
        if (outerIndent === -1) {
          outerIndent = indent;
          outerLine = i + 1;
        } else if (indent > outerIndent) {
          findings.push({
            severity: Severity.WARNING,
            category: "performance",
            rule: "nested-loop-on2",
            line: outerLine,
            message: `O(n²) 嵌套循环：第 ${outerLine} 行和第 ${i + 1} 行的双重循环`,
            suggestion:
              "考虑使用集合(set)去重，或将内层查找优化为 O(1) 字典查找",
          });
          outerIndent = -1;
          outerLine = -1;
        } else {
          outerIndent = indent;
          outerLine = i + 1;
        }
      }
    }

    // 检测不必要的变量赋值拷贝
    for (let i = 0; i < lines.length; i++) {
      if (/result\s*=\s*temp\s*$/.test(lines[i].trim())) {
        findings.push({
          severity: Severity.INFO,
          category: "performance",
          rule: "unnecessary-copy",
          line: i + 1,
          message: "不必要的变量赋值拷贝：result = temp 只是引用复制",
          suggestion:
            "直接返回 temp，或使用 list comprehension: return [x * 2 for x in data]",
        });
      }
    }

    // 检测 not in list（O(n) 查找）
    for (let i = 0; i < lines.length; i++) {
      if (/\bnot\s+in\s+\w+/.test(lines[i])) {
        findings.push({
          severity: Severity.WARNING,
          category: "performance",
          rule: "linear-search-in-list",
          line: i + 1,
          message: "线性查找 'not in list'：对于频繁查找应使用 set",
          suggestion: "将 duplicates 改为 set 类型：duplicates = set()",
        });
      }
    }

    return findings;
  }
}

// ── StyleReviewer ────────────────────────────────────────────────

class StyleReviewer extends ReviewerAgent {
  readonly name = "style";
  readonly systemPrompt =
    "你是一个代码风格审查专家。专门检查以下问题：\n" +
    "1. 模糊命名：temp、data、result、item 等无意义变量名\n" +
    "2. 缺少类型注解\n" +
    "3. 函数缺少 docstring\n" +
    "4. 代码结构问题\n" +
    "输出 JSON 数组格式的审查结果。";

  private static readonly VAGUE_NAMES =
    /\b(temp|data|result|item|val|obj|arr|lst|tmp|res)\s*=/i;

  protected mockReview(code: string): Finding[] {
    const findings: Finding[] = [];
    const lines = code.split("\n");

    for (let i = 0; i < lines.length; i++) {
      const stripped = lines[i].trim();
      const lineNum = i + 1;

      // 模糊命名
      const nameMatch = StyleReviewer.VAGUE_NAMES.exec(stripped);
      if (nameMatch) {
        findings.push({
          severity: Severity.INFO,
          category: "style",
          rule: "vague-naming",
          line: lineNum,
          message: `模糊变量名 '${nameMatch[1]}'：无法从名称推断用途`,
          suggestion:
            "使用更具描述性的名称，如 doubled_items、processed_records 等",
        });
      }

      // 缺少类型注解的函数定义
      if (/^\s*def\s+\w+\([^)]*\)\s*:/.test(stripped)) {
        if (!stripped.startsWith("def __")) {
          findings.push({
            severity: Severity.INFO,
            category: "style",
            rule: "missing-type-hints",
            line: lineNum,
            message: "函数缺少参数和返回值类型注解",
            suggestion:
              "添加类型注解：def get_user(user_id: str) -> Optional[User]:",
          });
        }
      }

      // 缺少 docstring
      if (/^\s*def\s+\w+/.test(stripped)) {
        const nextLine = i + 1 < lines.length ? lines[i + 1].trim() : "";
        if (!nextLine.startsWith('"""') && !nextLine.startsWith("'''")) {
          findings.push({
            severity: Severity.INFO,
            category: "style",
            rule: "missing-docstring",
            line: lineNum,
            message: "函数缺少 docstring",
            suggestion: "添加 docstring 说明函数用途、参数和返回值",
          });
        }
      }
    }

    return findings;
  }
}

// ════════════════════════════════════════════════════════════════════
// 5. Supervisor Agent
// ════════════════════════════════════════════════════════════════════

class Supervisor {
  private readonly reviewers: ReviewerAgent[] = [
    new SecurityReviewer(),
    new PerformanceReviewer(),
    new StyleReviewer(),
  ];

  async reviewCode(code: string): Promise<ReviewResult[]> {
    console.log("OUT:supervisor: ══ 多 Agent 代码审查系统 ══");
    console.log("OUT:supervisor: Supervisor 启动，准备分派审查任务");
    console.log(`OUT:supervisor: 代码行数: ${code.split("\n").length}`);
    console.log(
      `OUT:supervisor: 分派给 ${this.reviewers.length} 个 Reviewer Agent`,
    );

    const results: ReviewResult[] = [];
    for (const reviewer of this.reviewers) {
      console.log(
        `OUT:supervisor: → 分派给 ${reviewer.name} Reviewer...`,
      );
      const result = await reviewer.review(code);
      results.push(result);
      const mode = result.usedLlm ? "LLM" : "mock";
      console.log(
        `OUT:supervisor: ← ${reviewer.name} Reviewer 完成 ` +
          `(${result.findings.length} 个发现, ${mode} 模式, ` +
          `${result.durationMs.toFixed(0)}ms)`,
      );
    }

    return results;
  }

  generateReport(results: ReviewResult[]): Finding[] {
    const allFindings: Finding[] = [];
    for (const result of results) {
      allFindings.push(...result.findings);
    }

    // 按严重程度排序（Critical → Warning → Info），同级别按行号排序
    allFindings.sort((a, b) => a.severity - b.severity || a.line - b.line);
    return allFindings;
  }

  printReport(findings: Finding[]): void {
    console.log("\nOUT:report: ══ 审查汇总报告 ══");

    const critical = findings.filter(
      (f) => f.severity === Severity.CRITICAL,
    );
    const warning = findings.filter(
      (f) => f.severity === Severity.WARNING,
    );
    const info = findings.filter((f) => f.severity === Severity.INFO);

    console.log(
      `OUT:report: 发现 ${findings.length} 个问题 ` +
        `(🔴 Critical: ${critical.length}, 🟡 Warning: ${warning.length}, ` +
        `🔵 Info: ${info.length})`,
    );
    console.log();

    const severityLabels: Record<Severity, string> = {
      [Severity.CRITICAL]: "🔴 CRITICAL",
      [Severity.WARNING]: "🟡 WARNING",
      [Severity.INFO]: "🔵 INFO",
    };

    for (let i = 0; i < findings.length; i++) {
      const f = findings[i];
      const label = severityLabels[f.severity];
      const lineInfo = f.line > 0 ? `L${f.line}` : "全局";
      console.log(
        `OUT:report: [${i + 1}] ${label} [${f.category}] ` +
          `${f.rule} (${lineInfo})`,
      );
      console.log(`OUT:report:     问题: ${f.message}`);
      console.log(`OUT:report:     建议: ${f.suggestion}`);
      console.log();
    }
  }
}

// ════════════════════════════════════════════════════════════════════
// 6. 主函数
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  const code = MOCK_CODE_SNIPPET;

  console.log("OUT: ══ 多 Agent 代码审查系统（Supervisor-Worker）══");
  console.log(
    `OUT: 审查目标: 内置示例代码 (${code.split("\n").length} 行)`,
  );
  console.log();

  // Supervisor 协调审查
  const supervisor = new Supervisor();
  const results = await supervisor.reviewCode(code);

  // 各 Reviewer 输出自己的发现
  for (const result of results) {
    console.log(
      `\nOUT:reviewer:${result.reviewer}: ── ${result.reviewer} Reviewer 审查结果 ` +
        `(${result.findings.length} 个发现) ──`,
    );
    for (const f of result.findings) {
      const sev =
        f.severity === Severity.CRITICAL
          ? "CRITICAL"
          : f.severity === Severity.WARNING
            ? "WARNING"
            : "INFO";
      const lineInfo = f.line > 0 ? `L${f.line}` : "全局";
      console.log(
        `OUT:reviewer:${result.reviewer}:   [${sev}] ${f.rule} (${lineInfo}): ${f.message}`,
      );
    }
  }

  // Supervisor 汇总
  const allFindings = supervisor.generateReport(results);
  supervisor.printReport(allFindings);

  console.log("OUT: ══ 审查完成 ══");
}

main();
