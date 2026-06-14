# 练习题 — 编程/代码 Agent

## 练习1：添加代码解释工具

**目标**：给 Agent 添加第 5 个工具 `explain_code(code)`，让 LLM 解释一段代码的功能。

**说明**：编程 Agent 不仅要能写代码，还要能解释代码。这个工具让 Agent 在读到不理解的代码时可以请求解释。

**要求**：
- 添加 `explain_code` 工具定义（参数：`code` 字符串）
- 工具实现：把 code 发给 LLM，让它用简单语言解释
- 离线模式：返回 "（离线模式：无法解释代码）"
- 在 Agent 循环中注册这个工具

<details>
<summary>参考答案（Python）</summary>

```python
# 工具定义
{
    "type": "function",
    "function": {
        "name": "explain_code",
        "description": "解释一段代码的功能和逻辑。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要解释的代码"}
            },
            "required": ["code"],
        },
    },
},

# 工具实现
def explain_code(code: str) -> str:
    """用 LLM 解释代码。"""
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": "用简单语言解释以下代码的功能。"},
                {"role": "user", "content": code},
            ],
        )
        return resp.choices[0].message.content or "（无法解释）"
    except Exception:
        return "（离线模式：无法解释代码）"
```
</details>

<details>
<summary>参考答案（TypeScript）</summary>

```typescript
// 工具定义
{
  type: "function",
  function: {
    name: "explain_code",
    description: "解释一段代码的功能和逻辑。",
    parameters: {
      type: "object",
      properties: {
        code: { type: "string", description: "要解释的代码" },
      },
      required: ["code"],
    },
  },
},

// 工具实现
async function explainCode(code: string): Promise<string> {
  try {
    const resp = await client.chat.completions.create({
      model: cfg.model,
      messages: [
        { role: "system", content: "用简单语言解释以下代码的功能。" },
        { role: "user", content: code },
      ],
    });
    return resp.choices[0].message.content ?? "（无法解释）";
  } catch {
    return "（离线模式：无法解释代码）";
  }
}
```
</details>

---

## 练习2：添加多文件编辑支持

**目标**：让 Agent 能在一个循环中编辑多个文件（如同时修改 `main.py` 和 `utils.py`）。

**说明**：真实编程任务往往涉及多个文件。当前 Agent 的 `write_file` 工具已支持任意路径，但 Agent 循环需要能跟踪多个文件的变更。

**要求**：
- 添加 `modified_files: Set[str]` 跟踪已修改的文件
- 在 trace 中记录每个文件的修改次数
- 最后打印所有修改过的文件列表

<details>
<summary>参考答案（Python 核心逻辑）</summary>

```python
modified_files: set[str] = set()

# 在 write_file 工具实现中
def write_file(path: str, content: str) -> str:
    full_path = WORKSPACE / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    modified_files.add(path)  # 跟踪修改
    return f"OUT:tool: 写入文件: {path} ({len(content)} 字符)"

# 在 main() 结束时
print(f"\nOUT:agent: 修改了 {len(modified_files)} 个文件:")
for f in sorted(modified_files):
    print(f"OUT:agent:   - {f}")
```
</details>

---

## 练习3：添加测试覆盖率报告

**目标**：让 `run_test` 工具在运行测试时同时收集覆盖率信息。

**说明**：知道代码"通过测试"是不够的，还需要知道"测试覆盖了多少代码"。

**要求**：
- 修改 `run_test`：用 `coverage run` 替代直接 `python3` 执行
- 测试后运行 `coverage report` 获取覆盖率
- 在输出中显示覆盖率百分比
- 离线模式：mock 一个覆盖率数字

<details>
<summary>参考答案（Python 核心逻辑）</summary>

```python
def run_test(command: str) -> str:
    """沙箱执行测试命令，带覆盖率。"""
    # 安全检查
    if not check_code_safety(command):
        return "BLOCKED: 命令包含危险操作"

    # 用 coverage 包装
    if command.startswith("python3 ") and not "coverage" in command:
        script = command.replace("python3 ", "", 1)
        command = f"python3 -m coverage run {script}"

    result = subprocess.run(
        ["python3", "-c", command],
        capture_output=True, text=True, timeout=10,
    )

    output = result.stdout + result.stderr

    # 获取覆盖率报告
    try:
        cov = subprocess.run(
            ["python3", "-m", "coverage", "report"],
            capture_output=True, text=True, timeout=5,
        )
        output += "\n" + cov.stdout
    except Exception:
        pass

    return output if result.returncode == 0 else f"FAILED:\n{output}"
```
</details>

<details>
<summary>参考答案（TypeScript 核心逻辑）</summary>

```typescript
async function runTest(command: string): Promise<string> {
  if (!checkCodeSafety(command)) {
    return "BLOCKED: 命令包含危险操作";
  }

  // 用 coverage 包装
  let cmd = command;
  if (command.startsWith("python3 ") && !command.includes("coverage")) {
    const script = command.replace("python3 ", "");
    cmd = `python3 -m coverage run ${script}`;
  }

  try {
    const result = execSync(cmd, {
      cwd: WORKSPACE,
      timeout: 10_000,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });

    // 获取覆盖率
    try {
      const cov = execSync("python3 -m coverage report", {
        cwd: WORKSPACE,
        timeout: 5_000,
        encoding: "utf-8",
      });
      return result + "\n" + cov;
    } catch {
      return result;
    }
  } catch (err) {
    const error = err as { stderr?: string; stdout?: string; message?: string };
    return `FAILED:\n${(error.stdout ?? "") + (error.stderr ?? "")}`;
  }
}
```
</details>
