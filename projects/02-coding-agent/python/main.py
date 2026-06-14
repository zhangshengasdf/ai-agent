"""项目2 · 编程/代码 Agent（工具调用 + 代码执行沙箱）

综合实战：构建一个能"读代码→写代码→运行测试→看输出→修 bug→循环"的编程 Agent。

核心组件：
  - Agent Loop：LLM 决定调用工具 → 执行 → 反馈 → 循环
  - 沙箱执行：subprocess.run + timeout + 危险代码拦截
  - 4 个工具：read_file / write_file / run_test / list_files
  - 自我纠正：测试失败 → 把错误反馈给 LLM → 修 bug 重试
  - 离线 Mock：预设 mock 代码+测试，演示完整修复流程
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

# 工作区目录（workspace/ 在项目根目录下）
WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"

# 最大 Agent 循环步数
MAX_STEPS = 10

# ── 沙箱安全：危险代码关键词 ─────────────────────────────────────
DANGEROUS_KEYWORDS = [
    "rm -rf", "rm /", "rmdir /", "shutil.rmtree",
    "os.system", "os.popen", "os.exec", "os.spawn",
    "import subprocess", "from subprocess",
    "import socket", "from socket",
    "urllib.request.urlopen",
    "sys.exit", "os._exit",
    "__import__('os')", "__import__('subprocess')",
    "eval(", "exec(",
    "open('/etc", "open('/proc",
]


def check_code_safety(code: str) -> bool:
    """检查代码是否包含危险操作。返回 True 表示安全。"""
    code_lower = code.lower()
    for keyword in DANGEROUS_KEYWORDS:
        if keyword.lower() in code_lower:
            return False
    return True


# ════════════════════════════════════════════════════════════════════
# 1. 沙箱代码执行
# ════════════════════════════════════════════════════════════════════


def sandbox_execute(
    code: str, timeout: int = 5, *, shell: bool = False
) -> Dict[str, Any]:
    """在沙箱中安全执行代码。

    Args:
        code: 要执行的代码字符串（inline Python）或 shell 命令。
        timeout: 超时秒数（默认 5 秒）。
        shell: True 时作为 shell 命令执行，False 时作为 python3 -c 执行。

    Returns:
        包含 stdout, stderr, returncode, timed_out 的字典。
    """
    if not check_code_safety(code):
        return {
            "stdout": "",
            "stderr": "BLOCKED: 代码包含危险操作，已被沙箱拦截。",
            "returncode": -1,
            "timed_out": False,
        }

    cmd = code if shell else [sys.executable, "-c", code]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKSPACE),
            shell=shell,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"TIMEOUT: 代码执行超过 {timeout} 秒。",
            "returncode": -1,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"ERROR: {e}",
            "returncode": -1,
            "timed_out": False,
        }


# ════════════════════════════════════════════════════════════════════
# 2. 工具定义 + 实现
# ════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作区中的文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作区的文件路径"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "向工作区写入文件内容（覆盖已有内容）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对于工作区的文件路径"},
                    "content": {"type": "string", "description": "要写入的文件内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_test",
            "description": "在沙箱中运行测试命令并返回输出。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的测试命令，如 'python3 test_add.py'"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出工作区目录下的文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir": {"type": "string", "description": "相对于工作区的目录路径，默认 '.'"}
                },
            },
        },
    },
]


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """执行工具并返回结果字符串。"""
    if name == "read_file":
        path = args.get("path", "")
        full_path = WORKSPACE / path
        if not full_path.exists():
            return f"ERROR: 文件不存在: {path}"
        content = full_path.read_text(encoding="utf-8")
        print(f"OUT:tool: 读取文件: {path} ({len(content)} 字符)")
        return content

    elif name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        full_path = WORKSPACE / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        print(f"OUT:tool: 写入文件: {path} ({len(content)} 字符)")
        return f"OK: 已写入 {path}"

    elif name == "run_test":
        command = args.get("command", "")
        print(f"OUT:tool: 运行测试: {command}")
        result = sandbox_execute(command, shell=True)
        output = ""
        if result["stdout"]:
            output += result["stdout"]
        if result["stderr"]:
            output += ("\n" if output else "") + result["stderr"]
        if result["returncode"] != 0:
            output = f"EXIT CODE: {result['returncode']}\n{output}"
        return output or "(无输出)"

    elif name == "list_files":
        dir_path = args.get("dir", ".")
        full_path = WORKSPACE / dir_path
        if not full_path.exists():
            return f"ERROR: 目录不存在: {dir_path}"
        entries = sorted(os.listdir(full_path))
        print(f"OUT:tool: 列出文件: {dir_path} ({len(entries)} 项)")
        return "\n".join(entries)

    else:
        return f"ERROR: 未知工具: {name}"


# ════════════════════════════════════════════════════════════════════
# 3. LLM 调用封装（带 try/catch 降级）
# ════════════════════════════════════════════════════════════════════


def llm_chat(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """调用 LLM，失败时返回空响应。"""
    try:
        kwargs: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        result: Dict[str, Any] = {"content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]
        return result
    except Exception:
        return {"content": "", "tool_calls": []}


# ════════════════════════════════════════════════════════════════════
# 4. Agent Loop（核心）
# ════════════════════════════════════════════════════════════════════


def coding_agent(task: str, max_steps: int = MAX_STEPS) -> str:
    """编程 Agent 主循环。

    给定编程任务，Agent 通过工具调用完成：读文件→写代码→运行测试→修 bug→循环。

    Args:
        task: 编程任务描述。
        max_steps: 最大循环步数。

    Returns:
        Agent 的最终输出文本。
    """
    print(f"\nOUT:agent: ══ 编程 Agent ══")
    print(f"OUT:agent: 任务: {task}")
    print(f"OUT:agent: 最大步数: {max_steps}")

    system_prompt = (
        "你是一个编程助手 Agent。你的任务是通过工具调用来完成编程任务。\n"
        "可用工具：\n"
        "  - read_file(path): 读取文件\n"
        "  - write_file(path, content): 写入文件\n"
        "  - run_test(command): 运行测试\n"
        "  - list_files(dir): 列出文件\n\n"
        "工作流程：\n"
        "1. 先用 list_files 了解项目结构\n"
        "2. 用 read_file 读取相关文件\n"
        "3. 用 write_file 写入代码\n"
        "4. 用 run_test 运行测试\n"
        "5. 如果测试失败，分析错误并修复代码，然后重新测试\n"
        "6. 重复直到测试通过\n\n"
        "重要：每次只做一步操作，观察结果后再决定下一步。"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    final_output = ""

    for step in range(1, max_steps + 1):
        print(f"\nOUT:agent: ── 步骤 {step}/{max_steps} ──")

        resp = llm_chat(messages, tools=TOOLS)
        content = resp.get("content", "")
        tool_calls = resp.get("tool_calls", [])

        if content:
            print(f"OUT:agent: LLM: {content[:200]}")
            final_output = content

        if not tool_calls:
            print("OUT:agent: 无工具调用，结束循环")
            break

        # 执行工具调用
        for tc in tool_calls:
            name = tc["name"]
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                args = {}

            print(f"OUT:agent: 调用工具: {name}({json.dumps(args, ensure_ascii=False)[:100]})")
            result = execute_tool(name, args)

            # 把工具结果反馈给 LLM
            messages.append({"role": "assistant", "content": content or ""})
            messages.append({
                "role": "user",
                "content": f"工具 {name} 的输出:\n{result}",
            })

            # 检查测试是否通过
            if name == "run_test" and "EXIT CODE: 0" not in result and "All tests passed" in result:
                print("OUT:agent: ✅ 测试通过!")
                return result

    return final_output


# ════════════════════════════════════════════════════════════════════
# 5. 离线 Mock Agent（完整演示修复流程）
# ════════════════════════════════════════════════════════════════════


def mock_coding_agent() -> None:
    """离线模式：模拟 Agent 的完整"写代码→测试→修 bug"流程。"""
    print("\nOUT:mock: ══ 离线 Mock Agent ══")
    print("OUT:mock: 任务: 实现 add 函数，通过所有测试")

    # 步骤1：列出文件
    print("\nOUT:mock: ── 步骤 1: 列出工作区文件 ──")
    files = execute_tool("list_files", {"dir": "."})
    print(f"OUT:mock: 文件列表:\n{files}")

    # 步骤2：读取测试文件
    print("\nOUT:mock: ── 步骤 2: 读取测试文件 ──")
    test_content = execute_tool("read_file", {"path": "test_add.py"})
    print(f"OUT:mock: 测试内容:\n{test_content}")

    # 步骤3：读取当前 main.py（初始空实现）
    print("\nOUT:mock: ── 步骤 3: 读取当前实现 ──")
    main_content = execute_tool("read_file", {"path": "main.py"})
    print(f"OUT:mock: 当前 main.py:\n{main_content}")

    # 步骤4：写入第一个实现（故意有 bug）
    print("\nOUT:mock: ── 步骤 4: 写入实现（有 bug） ──")
    buggy_code = "def add(a, b):\n    return a - b  # bug: 应该是 +\n"
    execute_tool("write_file", {"path": "main.py", "content": buggy_code})

    # 步骤5：运行测试（预期失败）
    print("\nOUT:mock: ── 步骤 5: 运行测试（预期失败） ──")
    test_result = execute_tool("run_test", {"command": "python3 test_add.py"})
    print(f"OUT:mock: 测试结果:\n{test_result}")

    # 步骤6：分析错误并修复
    print("\nOUT:mock: ── 步骤 6: 分析错误并修复 ──")
    print("OUT:mock: LLM 分析: add(2,3) 返回 -1 而不是 5，应该是 a + b 而非 a - b")
    fixed_code = "def add(a, b):\n    return a + b\n"
    execute_tool("write_file", {"path": "main.py", "content": fixed_code})

    # 步骤7：重新运行测试（预期通过）
    print("\nOUT:mock: ── 步骤 7: 重新运行测试 ──")
    test_result = execute_tool("run_test", {"command": "python3 test_add.py"})
    print(f"OUT:mock: 测试结果:\n{test_result}")

    print("\nOUT:mock: ══ Mock 完成 ══")


# ════════════════════════════════════════════════════════════════════
# 6. 沙箱演示
# ════════════════════════════════════════════════════════════════════


def demo_sandbox() -> None:
    """演示沙箱的安全执行和危险代码拦截。"""
    print("\nOUT:sandbox: ══ 沙箱安全演示 ══")

    # 安全代码
    print("\nOUT:sandbox: ── 安全代码执行 ──")
    safe_codes = [
        "print(1 + 1)",
        "print('Hello, Agent!')",
        "import math; print(math.sqrt(16))",
    ]
    for code in safe_codes:
        print(f"\nOUT:sandbox: 代码: {code}")
        result = sandbox_execute(code)
        print(f"OUT:sandbox: 输出: {result['stdout'].strip()}")
        print(f"OUT:sandbox: 状态: {'✅ 安全' if result['returncode'] == 0 else '❌ 失败'}")

    # 危险代码
    print("\nOUT:sandbox: ── 危险代码拦截 ──")
    dangerous_codes = [
        "import os; os.system('rm -rf /')",
        "import subprocess; subprocess.run(['rm', '-rf', '/'])",
        "from socket import socket",
        "exec('import os; os.system(\"ls\")')",
    ]
    for code in dangerous_codes:
        print(f"\nOUT:sandbox: 代码: {code}")
        result = sandbox_execute(code)
        print(f"OUT:sandbox: 输出: {result['stderr'][:100]}")
        print(f"OUT:sandbox: 状态: {'🛡️ 已拦截' if result['returncode'] == -1 else '⚠️ 未拦截'}")


# ════════════════════════════════════════════════════════════════════
# 7. 主函数
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print("OUT: ══ 编程/代码 Agent ══")

    # 1. 沙箱演示
    demo_sandbox()

    # 2. 确保 workspace 存在且有 mock 文件
    if not WORKSPACE.exists():
        print("\nOUT:agent: workspace 不存在，跳过 Agent 演示")
        return

    # 3. 尝试真实 LLM，失败则用 mock
    print("\nOUT:agent: 尝试调用 LLM...")
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "say ok"}],
            max_tokens=5,
        )
        print(f"OUT:agent: LLM 连接成功: {resp.choices[0].message.content}")
        # 真实 Agent 循环
        coding_agent("实现 workspace/main.py 中的 add 函数，使 test_add.py 中的所有测试通过。")
    except Exception:
        print("OUT:agent: LLM 不可用，进入离线 mock 模式")
        mock_coding_agent()

    print("\nOUT: ══ 编程 Agent 完成 ══")


if __name__ == "__main__":
    main()
