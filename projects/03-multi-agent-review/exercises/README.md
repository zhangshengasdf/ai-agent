# 练习题 — 多 Agent 代码审查系统

## 练习1：添加 ComplexityReviewer Agent

**目标**：添加一个新的 Reviewer Agent，专门检查代码复杂度问题。

**说明**：当前系统有 Security/Performance/Style 三个 Reviewer。添加一个 ComplexityReviewer 可以检测圈复杂度过高、函数过长等问题。

**要求**：
- 创建 `ComplexityReviewer` 类，继承 `ReviewerAgent`
- 检测规则：
  - 函数超过 20 行（过长函数）
  - 单个函数中 if/elif 分支超过 5 个（高圈复杂度）
  - 嵌套层级超过 3 层（深嵌套）
- 在 Supervisor 的 reviewers 列表中注册新 Agent

<details>
<summary>参考答案（Python）</summary>

```python
class ComplexityReviewer(ReviewerAgent):
    """复杂度审查 Agent：检查函数长度、分支数量、嵌套层级。"""

    name = "complexity"
    system_prompt = (
        "你是一个代码复杂度审查专家。检查以下问题：\n"
        "1. 函数过长（超过 20 行）\n"
        "2. 分支过多（if/elif 超过 5 个）\n"
        "3. 嵌套过深（超过 3 层）\n"
        "输出 JSON 数组格式的审查结果。"
    )

    def _mock_review(self, code: str) -> List[Finding]:
        findings: List[Finding] = []
        lines = code.split("\n")
        func_start = -1
        func_name = ""
        indent_level = 0

        for i, line in enumerate(lines, 1):
            # 检测函数开始
            match = re.match(r"^(\s*)def\s+(\w+)", line)
            if match:
                # 检查上一个函数的长度
                if func_start > 0:
                    length = i - func_start
                    if length > 20:
                        findings.append(Finding(
                            severity=Severity.WARNING,
                            category="complexity",
                            rule="function-too-long",
                            line=func_start,
                            message=f"函数 '{func_name}' 过长（{length} 行）",
                            suggestion="拆分为更小的子函数，每个函数职责单一",
                        ))
                func_start = i
                func_name = match.group(2)

            # 检测嵌套层级
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                current_indent = len(line) - len(line.lstrip())
                nesting = current_indent // 4
                if nesting > 3:
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        category="complexity",
                        rule="deep-nesting",
                        line=i,
                        message=f"嵌套层级过深（{nesting} 层）",
                        suggestion="使用 early return 或提取子函数减少嵌套",
                    ))

        return findings
```
</details>

<details>
<summary>参考答案（TypeScript）</summary>

```typescript
class ComplexityReviewer extends ReviewerAgent {
  readonly name = "complexity";
  readonly systemPrompt =
    "你是一个代码复杂度审查专家。检查以下问题：\n" +
    "1. 函数过长（超过 20 行）\n" +
    "2. 分支过多（if/elif 超过 5 个）\n" +
    "3. 嵌套过深（超过 3 层）\n" +
    "输出 JSON 数组格式的审查结果。";

  protected mockReview(code: string): Finding[] {
    const findings: Finding[] = [];
    const lines = code.split("\n");
    let funcStart = -1;
    let funcName = "";

    for (let i = 0; i < lines.length; i++) {
      const lineNum = i + 1;
      const match = /^(\s*)def\s+(\w+)/.exec(lines[i]);
      if (match) {
        if (funcStart > 0) {
          const length = lineNum - funcStart;
          if (length > 20) {
            findings.push({
              severity: Severity.WARNING,
              category: "complexity",
              rule: "function-too-long",
              line: funcStart,
              message: `函数 '${funcName}' 过长（${length} 行）`,
              suggestion: "拆分为更小的子函数，每个函数职责单一",
            });
          }
        }
        funcStart = lineNum;
        funcName = match[2];
      }

      // 检测嵌套层级
      const stripped = lines[i].trim();
      if (stripped && !stripped.startsWith("#")) {
        const currentIndent = lines[i].length - lines[i].trimStart().length;
        const nesting = Math.floor(currentIndent / 4);
        if (nesting > 3) {
          findings.push({
            severity: Severity.WARNING,
            category: "complexity",
            rule: "deep-nesting",
            line: lineNum,
            message: `嵌套层级过深（${nesting} 层）`,
            suggestion: "使用 early return 或提取子函数减少嵌套",
          });
        }
      }
    }

    return findings;
  }
}
```
</details>

---

## 练习2：添加自定义代码输入

**目标**：让程序支持从文件或标准输入读取待审查的代码。

**说明**：当前实现只审查内置的 `MOCK_CODE_SNIPPET`。添加命令行参数支持，可以审查任意 Python 文件。

**要求**：
- 支持命令行参数：`python main.py <file.py>`
- 无参数时使用内置 mock 代码片段
- 有参数时读取指定文件内容
- 文件不存在时报错并退出

<details>
<summary>参考答案（Python）</summary>

```python
import sys

def load_code() -> str:
    """从命令行参数或内置片段加载代码。"""
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"OUT: 错误: 文件不存在: {file_path}", file=sys.stderr)
            sys.exit(1)
        return file_path.read_text(encoding="utf-8")
    return MOCK_CODE_SNIPPET

def main() -> None:
    code = load_code()
    # ... 其余逻辑不变
```
</details>

<details>
<summary>参考答案（TypeScript）</summary>

```typescript
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

function loadCode(): string {
  const args = process.argv.slice(2);
  if (args.length > 0) {
    const filePath = resolve(args[0]);
    if (!existsSync(filePath)) {
      console.error(`OUT: 错误: 文件不存在: ${filePath}`);
      process.exit(1);
    }
    return readFileSync(filePath, "utf-8");
  }
  return MOCK_CODE_SNIPPET;
}

async function main(): Promise<void> {
  const code = loadCode();
  // ... 其余逻辑不变
}
```
</details>

---

## 练习3：实现并行审查

**目标**：将 3 个 Reviewer 的审查改为并行执行，提高效率。

**说明**：当前实现是串行执行各 Reviewer。由于各 Reviewer 独立工作，可以并行执行以减少总耗时。

**要求**：
- Python：使用 `concurrent.futures.ThreadPoolExecutor` 并行执行
- TypeScript：使用 `Promise.all` 并行执行
- 保持输出顺序一致（按 Security → Performance → Style）
- 在 trace 中记录并行执行的总耗时

<details>
<summary>参考答案（Python）</summary>

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class Supervisor:
    def review_code(self, code: str) -> List[ReviewResult]:
        """并行分派代码给各 Reviewer。"""
        print("OUT:supervisor: ══ 多 Agent 代码审查系统 ══")
        print(f"OUT:supervisor: 并行分派给 {len(self.reviewers)} 个 Reviewer")

        results: List[ReviewResult] = []
        with ThreadPoolExecutor(max_workers=len(self.reviewers)) as executor:
            future_to_reviewer = {
                executor.submit(reviewer.review, code): reviewer
                for reviewer in self.reviewers
            }
            for future in as_completed(future_to_reviewer):
                reviewer = future_to_reviewer[future]
                result = future.result()
                results.append(result)
                print(f"OUT:supervisor: ← {reviewer.name} 完成")

        # 按原始顺序排序
        order = {r.name: i for i, r in enumerate(self.reviewers)}
        results.sort(key=lambda r: order.get(r.reviewer, 99))
        return results
```
</details>

<details>
<summary>参考答案（TypeScript）</summary>

```typescript
class Supervisor {
  async reviewCode(code: string): Promise<ReviewResult[]> {
    console.log("OUT:supervisor: ══ 多 Agent 代码审查系统 ══");
    console.log(
      `OUT:supervisor: 并行分派给 ${this.reviewers.length} 个 Reviewer`,
    );

    // 并行执行所有 Reviewer
    const promises = this.reviewers.map(async (reviewer) => {
      console.log(`OUT:supervisor: → 分派给 ${reviewer.name} Reviewer...`);
      const result = await reviewer.review(code);
      console.log(`OUT:supervisor: ← ${reviewer.name} 完成`);
      return result;
    });

    return Promise.all(promises);
  }
}
```
</details>
