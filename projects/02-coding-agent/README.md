# 项目2 · 编程/代码 Agent（工具调用 + 代码执行沙箱）

> **综合实战**：构建一个能"读代码→写代码→运行测试→看输出→修 bug→循环"的编程 Agent。

---

## 你会学到什么

1. **编程 Agent 循环**：给编程任务 → 读文件(tool) → 写代码(tool) → 运行测试(沙箱) → 看输出 → 修 bug → 循环直到通过或 max_steps
2. **沙箱代码执行**：`subprocess.run(["python3", "-c", code], timeout=5)` 安全执行 + 危险代码拦截
3. **自我纠正**：测试失败时把错误信息反馈给 Agent，让它修 bug 重试
4. **4 个工具**：`read_file(path)` / `write_file(path, content)` / `run_test(command)` / `list_files(dir)`
5. **离线 Mock**：预设 mock 代码 + 测试文件，Agent 在离线模式下演示完整"写代码→测试→修 bug"流程

---

## 架构概览

```
编程任务（如"实现 add 函数"）
   │
   ▼
┌──────────────┐
│  Agent Loop  │  ← LLM 决定调用哪个工具
│  (max_steps) │
└──────┬───────┘
       │ tools
       ▼
┌──────────────────────────────────────────┐
│  read_file(path)       读取文件内容        │
│  write_file(path,code) 写入代码文件        │
│  run_test(cmd)         沙箱执行测试        │
│  list_files(dir)       列出目录文件        │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   Sandbox    │  ← subprocess + 超时 + 危险代码拦截
└──────────────┘
       │
       ▼
  测试通过? ──No──► 把错误反馈给 LLM ──► 回到 Agent Loop
     │
    Yes
     ▼
   完成 ✅
```

---

## 运行方式

```bash
cd ai-agent/projects/02-coding-agent

# Python
python3 python/main.py

# TypeScript
npx tsx typescript/main.ts
```

输出前缀：`OUT:sandbox:` / `OUT:agent:` / `OUT:tool:` / `OUT:mock:`

---

## 离线设计

`.env` 中 API 密钥为占位符 `sk-REPLACE-ME` 时：

1. **沙箱演示**：安全执行 `print(1+1)` + 拦截危险代码（`rm -rf`、`os.system` 等）
2. **Mock Agent 循环**：
   - 预设 `workspace/test_add.py`（含 `def test_add(): assert add(2,3)==5`）
   - 预设 `workspace/main.py`（初始空实现 `pass`）
   - Agent 按预设步骤：读测试 → 写实现 → 运行测试(失败) → 修 bug → 运行测试(通过)
3. 全程 **不依赖真实 API**，`exit 0`

---

## 沙箱安全

沙箱拦截的危险关键词：

| 类别 | 关键词 |
|------|--------|
| 文件删除 | `rm `, `rm -rf`, `rmdir`, `shutil.rmtree` |
| 系统调用 | `os.system`, `os.popen`, `os.exec`, `os.spawn` |
| 子进程 | `import subprocess`, `from subprocess` |
| 网络 | `import socket`, `from socket`, `urllib.request.urlopen` |
| 退出 | `sys.exit`, `os._exit` |

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
- [Mock 工作区](./workspace/)
