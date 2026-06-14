"""项目3 · 多 Agent 代码审查系统（Supervisor-Worker 协作）

综合实战：Supervisor 接收代码 → 分派给 3 个 Reviewer Agent → 各自从专门维度审查 → 汇总报告。

核心组件：
  - Supervisor：协调者，分派任务、收集结果、汇总排序
  - SecurityReviewer：检查 SQL 注入、硬编码密码、XSS
  - PerformanceReviewer：检查 O(n²) 循环、不必要拷贝
  - StyleReviewer：检查命名规范、注释、代码结构
  - 离线 Mock：预设含问题的代码片段，各 Reviewer 用正则规则审查，exit 0
"""

import re
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


# ════════════════════════════════════════════════════════════════════
# 1. 数据结构
# ════════════════════════════════════════════════════════════════════


class Severity(IntEnum):
    """问题严重程度（数值越小越严重）。"""
    CRITICAL = 1
    WARNING = 2
    INFO = 3


@dataclass(frozen=True)
class Finding:
    """一条审查发现。"""
    severity: Severity
    category: str       # security | performance | style
    rule: str           # 规则名称
    line: int           # 行号（0 表示全局）
    message: str        # 描述
    suggestion: str     # 修复建议


@dataclass
class ReviewResult:
    """单个 Reviewer 的审查结果。"""
    reviewer: str
    findings: List[Finding] = field(default_factory=list)
    duration_ms: float = 0.0
    used_llm: bool = False


# ════════════════════════════════════════════════════════════════════
# 2. 预设含问题的代码片段（离线 mock 用）
# ════════════════════════════════════════════════════════════════════

MOCK_CODE_SNIPPET = '''\
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
'''


# ════════════════════════════════════════════════════════════════════
# 3. LLM 调用封装（带 try/catch 降级）
# ════════════════════════════════════════════════════════════════════


def llm_chat(messages: List[Dict[str, str]]) -> str:
    """调用 LLM，失败时返回空字符串。"""
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════════
# 4. Reviewer Agent 基类 + 3 个专门 Reviewer
# ════════════════════════════════════════════════════════════════════


class ReviewerAgent:
    """Reviewer 基类：定义接口 + LLM 增强。"""

    name: str = "base"
    system_prompt: str = ""

    def review(self, code: str) -> ReviewResult:
        """审查代码，返回 ReviewResult。先尝试 LLM，失败则走 mock。"""
        t0 = time.time()
        result = ReviewResult(reviewer=self.name)

        # 尝试 LLM 审查
        llm_findings = self._try_llm_review(code)
        if llm_findings:
            result.findings = llm_findings
            result.used_llm = True
        else:
            # 离线 mock：规则审查
            result.findings = self._mock_review(code)

        result.duration_ms = (time.time() - t0) * 1000
        return result

    def _try_llm_review(self, code: str) -> List[Finding]:
        """尝试用 LLM 审查代码。"""
        resp = llm_chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请审查以下代码，输出 JSON 数组，每个元素包含 "
             f"severity(Critical/Warning/Info)、rule、line、message、suggestion。\n\n"
             f"```python\n{code}\n```"},
        ])
        if not resp:
            return []
        # 尝试解析 JSON（简化处理）
        try:
            import json
            json_str = resp
            if "```" in resp:
                for line in resp.split("\n"):
                    line = line.strip()
                    if line.startswith("["):
                        json_str = line
                        break
            items = json.loads(json_str)
            if not isinstance(items, list):
                return []
            findings = []
            for item in items:
                sev_str = item.get("severity", "Info").upper()
                sev = {"CRITICAL": Severity.CRITICAL, "WARNING": Severity.WARNING}.get(
                    sev_str, Severity.INFO
                )
                findings.append(Finding(
                    severity=sev,
                    category=self.name,
                    rule=item.get("rule", "unknown"),
                    line=item.get("line", 0),
                    message=item.get("message", ""),
                    suggestion=item.get("suggestion", ""),
                ))
            return findings
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _mock_review(self, code: str) -> List[Finding]:
        """离线 mock 审查（子类实现）。"""
        raise NotImplementedError


class SecurityReviewer(ReviewerAgent):
    """安全审查 Agent：检查 SQL 注入、硬编码密码、XSS。"""

    name = "security"
    system_prompt = (
        "你是一个安全审查专家。专门检查以下安全问题：\n"
        "1. SQL 注入：字符串拼接构造 SQL\n"
        "2. 硬编码密码/密钥\n"
        "3. XSS：未转义的用户输入拼接 HTML\n"
        "输出 JSON 数组格式的审查结果。"
    )

    def _mock_review(self, code: str) -> List[Finding]:
        findings: List[Finding] = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # SQL 注入：execute + 字符串格式化
            if re.search(r"execute\s*\(.*%", line) or re.search(r"execute\s*\(.*\.format", line):
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category="security",
                    rule="sql-injection",
                    line=i,
                    message="SQL 注入风险：使用字符串格式化拼接 SQL 查询",
                    suggestion="使用参数化查询：cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                ))

            # 硬编码密码
            if re.search(r'(password|passwd|secret|api_key|token)\s*=\s*["\'][^"\']{3,}["\']', line, re.IGNORECASE):
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category="security",
                    rule="hardcoded-secret",
                    line=i,
                    message="硬编码密码/密钥：敏感信息不应直接写在代码中",
                    suggestion="使用环境变量或密钥管理服务：os.environ.get('DB_PASSWORD')",
                ))

            # XSS：未转义的 HTML 拼接
            if re.search(r'["\']<[a-z]+>["\'].*\+', line) or re.search(r'\+\s*["\']<[a-z]+>["\']', line):
                pass  # 单独的 HTML 标签不算

            if re.search(r'["\']<\w+>["\'] *\+ *\w+', line) and "escape" not in line.lower():
                if re.search(r'["\']</?\w+[^>]*>', line):
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        category="security",
                        rule="xss-risk",
                        line=i,
                        message="XSS 风险：未转义的用户输入直接拼接 HTML",
                        suggestion="使用模板引擎或转义函数：html.escape(user_input)",
                    ))

        return findings


class PerformanceReviewer(ReviewerAgent):
    """性能审查 Agent：检查 O(n²) 循环、不必要拷贝。"""

    name = "performance"
    system_prompt = (
        "你是一个性能审查专家。专门检查以下性能问题：\n"
        "1. O(n²) 或更高复杂度的嵌套循环\n"
        "2. 不必要的列表拷贝或重复计算\n"
        "3. 可用集合/字典优化的线性查找\n"
        "输出 JSON 数组格式的审查结果。"
    )

    def _mock_review(self, code: str) -> List[Finding]:
        findings: List[Finding] = []
        lines = code.split("\n")

        # 检测嵌套 for 循环（O(n²)）
        for_pattern = re.compile(r"^(\s*)for\s+.+\s+in\s+")
        outer_indent = -1
        outer_line = -1

        for i, line in enumerate(lines, 1):
            match = for_pattern.match(line)
            if match:
                indent = len(match.group(1))
                if outer_indent == -1:
                    outer_indent = indent
                    outer_line = i
                elif indent > outer_indent:
                    # 嵌套 for 循环
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        category="performance",
                        rule="nested-loop-on2",
                        line=outer_line,
                        message=f"O(n²) 嵌套循环：第 {outer_line} 行和第 {i} 行的双重循环",
                        suggestion="考虑使用集合(set)去重，或将内层查找优化为 O(1) 字典查找",
                    ))
                    outer_indent = -1
                    outer_line = -1
                else:
                    outer_indent = indent
                    outer_line = i

        # 检测不必要的列表拷贝模式
        for i, line in enumerate(lines, 1):
            if re.search(r"result\s*=\s*temp\s*$", line.strip()):
                findings.append(Finding(
                    severity=Severity.INFO,
                    category="performance",
                    rule="unnecessary-copy",
                    line=i,
                    message="不必要的变量赋值拷贝：result = temp 只是引用复制",
                    suggestion="直接返回 temp，或使用 list comprehension: return [x * 2 for x in data]",
                ))

        # 检测 not in list（O(n) 查找）
        for i, line in enumerate(lines, 1):
            if re.search(r"\bnot\s+in\s+\w+\s*\]", line) or re.search(r"\bnot\s+in\s+duplicates\b", line):
                findings.append(Finding(
                    severity=Severity.WARNING,
                    category="performance",
                    rule="linear-search-in-list",
                    line=i,
                    message="线性查找 'not in list'：对于频繁查找应使用 set",
                    suggestion="将 duplicates 改为 set 类型：duplicates = set()",
                ))

        return findings


class StyleReviewer(ReviewerAgent):
    """风格审查 Agent：检查命名规范、注释、代码结构。"""

    name = "style"
    system_prompt = (
        "你是一个代码风格审查专家。专门检查以下问题：\n"
        "1. 模糊命名：temp、data、result、item 等无意义变量名\n"
        "2. 缺少类型注解\n"
        "3. 函数缺少 docstring\n"
        "4. 代码结构问题\n"
        "输出 JSON 数组格式的审查结果。"
    )

    # 模糊命名模式
    VAGUE_NAMES = re.compile(
        r"\b(temp|data|result|item|val|obj|arr|lst|tmp|res)\s*=", re.IGNORECASE
    )

    def _mock_review(self, code: str) -> List[Finding]:
        findings: List[Finding] = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # 模糊命名
            match = self.VAGUE_NAMES.search(stripped)
            if match:
                name = match.group(1)
                findings.append(Finding(
                    severity=Severity.INFO,
                    category="style",
                    rule="vague-naming",
                    line=i,
                    message=f"模糊变量名 '{name}'：无法从名称推断用途",
                    suggestion=f"使用更具描述性的名称，如 doubled_items、processed_records 等",
                ))

            # 缺少类型注解的函数定义
            if re.match(r"^\s*def\s+\w+\([^)]*\)\s*:", stripped):
                if "def __" not in stripped:  # 跳过魔术方法
                    findings.append(Finding(
                        severity=Severity.INFO,
                        category="style",
                        rule="missing-type-hints",
                        line=i,
                        message="函数缺少参数和返回值类型注解",
                        suggestion="添加类型注解：def get_user(user_id: str) -> Optional[User]:",
                    ))

            # 缺少 docstring
            if re.match(r"^\s*def\s+\w+", stripped):
                # 检查下一行是否是 docstring
                if i < len(lines):
                    next_line = lines[i].strip() if i < len(lines) else ""
                    if not next_line.startswith('"""') and not next_line.startswith("'''"):
                        findings.append(Finding(
                            severity=Severity.INFO,
                            category="style",
                            rule="missing-docstring",
                            line=i,
                            message="函数缺少 docstring",
                            suggestion='添加 docstring 说明函数用途、参数和返回值',
                        ))

        return findings


# ════════════════════════════════════════════════════════════════════
# 5. Supervisor Agent
# ════════════════════════════════════════════════════════════════════


class Supervisor:
    """Supervisor Agent：协调多个 Reviewer，汇总审查结果。"""

    def __init__(self) -> None:
        self.reviewers: List[ReviewerAgent] = [
            SecurityReviewer(),
            PerformanceReviewer(),
            StyleReviewer(),
        ]

    def review_code(self, code: str) -> List[ReviewResult]:
        """分派代码给各 Reviewer，收集结果。"""
        print("OUT:supervisor: ══ 多 Agent 代码审查系统 ══")
        print("OUT:supervisor: Supervisor 启动，准备分派审查任务")
        print(f"OUT:supervisor: 代码行数: {len(code.splitlines())}")
        print(f"OUT:supervisor: 分派给 {len(self.reviewers)} 个 Reviewer Agent")

        results: List[ReviewResult] = []
        for reviewer in self.reviewers:
            print(f"OUT:supervisor: → 分派给 {reviewer.name} Reviewer...")
            result = reviewer.review(code)
            results.append(result)
            mode = "LLM" if result.used_llm else "mock"
            print(
                f"OUT:supervisor: ← {reviewer.name} Reviewer 完成 "
                f"({len(result.findings)} 个发现, {mode} 模式, "
                f"{result.duration_ms:.0f}ms)"
            )

        return results

    def generate_report(self, results: List[ReviewResult]) -> List[Finding]:
        """汇总所有 Reviewer 结果，按严重程度排序。"""
        all_findings: List[Finding] = []
        for result in results:
            all_findings.extend(result.findings)

        # 按严重程度排序（Critical → Warning → Info），同级别按行号排序
        all_findings.sort(key=lambda f: (f.severity, f.line))

        return all_findings

    def print_report(self, findings: List[Finding]) -> None:
        """打印汇总报告。"""
        print("\nOUT:report: ══ 审查汇总报告 ══")

        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        warning = [f for f in findings if f.severity == Severity.WARNING]
        info = [f for f in findings if f.severity == Severity.INFO]

        print(f"OUT:report: 发现 {len(findings)} 个问题 "
              f"(🔴 Critical: {len(critical)}, 🟡 Warning: {len(warning)}, "
              f"🔵 Info: {len(info)})")
        print()

        severity_labels = {
            Severity.CRITICAL: "🔴 CRITICAL",
            Severity.WARNING: "🟡 WARNING",
            Severity.INFO: "🔵 INFO",
        }

        for i, finding in enumerate(findings, 1):
            label = severity_labels[finding.severity]
            line_info = f"L{finding.line}" if finding.line > 0 else "全局"
            print(f"OUT:report: [{i}] {label} [{finding.category}] "
                  f"{finding.rule} ({line_info})")
            print(f"OUT:report:     问题: {finding.message}")
            print(f"OUT:report:     建议: {finding.suggestion}")
            print()


# ════════════════════════════════════════════════════════════════════
# 6. 主函数
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    """运行多 Agent 代码审查。"""
    # 使用预设的含问题代码片段
    code = MOCK_CODE_SNIPPET

    print("OUT: ══ 多 Agent 代码审查系统（Supervisor-Worker）══")
    print(f"OUT: 审查目标: 内置示例代码 ({len(code.splitlines())} 行)")
    print()

    # Supervisor 协调审查
    supervisor = Supervisor()
    results = supervisor.review_code(code)

    # 各 Reviewer 输出自己的发现
    for result in results:
        print(f"\nOUT:reviewer:{result.reviewer}: ── {result.reviewer} Reviewer 审查结果 "
              f"({len(result.findings)} 个发现) ──")
        for f in result.findings:
            sev = {Severity.CRITICAL: "CRITICAL", Severity.WARNING: "WARNING",
                   Severity.INFO: "INFO"}[f.severity]
            line_info = f"L{f.line}" if f.line > 0 else "全局"
            print(f"OUT:reviewer:{result.reviewer}:   [{sev}] {f.rule} ({line_info}): {f.message}")

    # Supervisor 汇总
    all_findings = supervisor.generate_report(results)
    supervisor.print_report(all_findings)

    print("OUT: ══ 审查完成 ══")


if __name__ == "__main__":
    main()
