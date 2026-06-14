# 第04章 Agent 循环（The Agent Loop）

> **「任务助手 Agent」获得了"自主性"**——从第03章的"单轮调用工具"，进化为"多步循环调工具直到完成任务"。
> 这是本教程的核心概念：**单轮=工具调用，多轮=Agent**。

## TL;DR

> **30 秒速读**：Agent 循环就是一个 `for` 循环，每轮让 LLM 决定下一步行动、执行工具、把结果反馈回去，直到模型不再调用工具或达到 max_steps 上限。
> 
> **如果只记一件事**：永远给 Agent 循环加 `max_steps` 上限（建议 10 到 20），没有上限的循环是生产事故的源头，一个失控 Agent 能在几分钟内烧掉几百块 API 费用。

---

## 本章目标

学完本章，你将理解：

1. **Agent 循环的本质**：一个 `while` 循环，持续 observe→reason→act，直到任务完成
2. **终止条件**：模型不再调用工具 = 任务完成；`max_steps` 兜底 = 防止无限循环
3. **observe→reason→act 三段式**：Agent 的每一步都是这个经典循环的一次迭代
4. **max_steps 防护**：为什么"无上限的循环"是生产事故的源头
5. **"工具调用"与"Agent"的本质区别**：循环让 Agent 具备了自主决策的连续性

---

## 核心概念：单轮 ≠ Agent，多轮 = Agent

这是本章最重要的认知：

> **工具调用（第03章）** 是一次性的：你问、模型调一次工具、给结果，结束。
> **Agent（本章）** 是持续的：模型**自主决定**"我现在该调什么工具"，调完**继续看结果**，**再决定**下一步，直到它认为任务完成。

一句话区分：

| | 单轮工具调用 | Agent 循环 |
|---|---|---|
| **谁决定何时停？** | 你的代码（固定走 1 次工具 + 1 次总结） | **模型自己**（不再调工具时即停） |
| **工具调用次数** | 固定 1 次（或预设） | **不固定**，0 次到 N 次，由模型决定 |
| **失败模式** | 一次调用失败 = 整个流程失败 | 可以中途纠错、换工具、重试思路 |
| **类比** | 计算器（按一次出一个结果） | 实习生（给个任务，他自己反复查资料做完） |

**Agent 循环是"Agent"区别于"工具调用"的本质。** 没有循环，就只是一个增强版的函数调用器。

---

## observe → reason → act：Agent 循环的三段式

每一步 Agent 迭代都遵循这个经典模式（来自经典的智能体架构）：

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
   ┌─────────┐    ┌─────────┐    ┌─────────┐          │
   │ Observe │───▶│ Reason  │───▶│   Act   │───┐      │
   │ 观察环境 │    │ LLM推理 │    │ 执行动作 │   │      │
   └─────────┘    └─────────┘    └─────────┘   │      │
        ▲                              │        │      │
        │                              ▼        │      │
        │                         ┌─────────┐   │      │
        │                         │ 结果反馈 │   │      │
        │                         │（新观察）│   │      │
        │                         └────┬────┘   │      │
        └──────────────────────────────┘        │      │
                                                │      │
                          不再调工具 / 达到上限 ─┴──▶ 终止
```

在我们用 OpenAI tools API 实现的版本里：

| 经典阶段 | 我们的实现 | 说明 |
|----------|------------|------|
| **Observe（观察）** | 读取 `messages` 列表（含之前的工具结果） | 看看任务进展到哪了、工具返回了什么 |
| **Reason（推理）** | 调用 LLM（`chat.completions.create`） | 模型决定"下一步该干嘛" |
| **Act（行动）** | 执行 `tool_calls`（运行对应工具函数） | 把模型的决策落地成真实操作 |

> 📌 **隐式推理 vs 显式推理**：本章用 OpenAI tools API 的**隐式**推理——模型内部想，我们只看它的行动。
> 第07章 ReAct 会教你**显式**推理（让模型把思考过程写出来：Thought → Action → Observation）。

---

## Agent 循环的伪代码

这是本章的核心代码骨架，所有 Agent 框架（无论 LangChain、OpenAI Agents SDK 还是我们第12章自造的）本质都是它：

```python
MAX_STEPS = 10  # ⚠️ 必须有上限！

def agent_loop(user_message, tools, tool_functions):
    messages = [
        {"role": "system", "content": "你是任务助手 Agent..."},
        {"role": "user", "content": user_message},
    ]

    for step in range(1, MAX_STEPS + 1):
        # ── Reason：让 LLM 决定下一步 ──
        response = client.chat.completions.create(
            model=cfg.model, messages=messages, tools=tools,
        )
        assistant_msg = response.choices[0].message

        # ── 终止条件 1：模型不再调工具 = 任务完成 ──
        if not assistant_msg.tool_calls:
            return assistant_msg.content  # 最终回答

        # ── Act：执行模型决定的工具调用 ──
        messages.append(assistant_msg)  # 记住"我做了什么决策"
        for tc in assistant_msg.tool_calls:
            result = tool_functions[tc.function.name](**parse(tc))
            messages.append({"role": "tool", "content": result})

        # ── Observe：下一轮循环开始时，messages 里已经有了新结果 ──

    # ── 终止条件 2：达到 max_steps，强制停止 ──
    return "(已达到最大步数)"
```

### 两个终止条件（缺一不可）

| 条件 | 触发时机 | 含义 |
|------|----------|------|
| **模型不调工具** | `response.choices[0].message.tool_calls` 为空 | 模型认为信息够了，给出最终回答 |
| **达到 `max_steps`** | 循环变量 `step` 超过上限 | 兜底保护，防止无限循环 |

**第一个条件是"正常完成"，第二个是"保险丝"。** 永远不要写没有 `max_steps` 的 Agent 循环——下一节解释为什么。

---

## 为什么 `max_steps` 是必需的（不是可选的）

这是新手最容易忽视、但最容易引发生产事故的一点。

### 场景 1：模型陷入"无限确认"循环

```
Step 1: 模型调 get_weather("北京")
Step 2: 工具返回 "北京晴 25°C"
Step 3: 模型又调 get_weather("北京")  ← 觉得"再看一次确认下"
Step 4: 工具返回 "北京晴 25°C"
Step 5: 模型又调 get_weather("北京")  ← 死循环
...
```

这真实会发生，尤其在：
- 模型能力较弱（小模型、温度过高）
- 工具返回结果模糊（如"需要更多信息"）
- 任务定义不清（模型不知道"完成"长什么样）

### 场景 2：烧钱

每个 GPT-4o 请求约 ¥0.1-0.5。一个失控的 Agent 循环跑 1000 步 = 烧掉几百块，而且你毫不知情。

### 场景 3：卡死整个系统

没有 `max_steps` + 没有单步超时 = 一个 Agent 卡住会拖垮整个后端。

### 正确做法

```python
MAX_STEPS = 10  # 合理上限：大部分任务 3-8 步完成

for step in range(1, MAX_STEPS + 1):
    # ... 循环体 ...

print(f"⚠️ 达到最大步数 {MAX_STEPS}，强制停止")
```

> 💡 **`max_steps` 该设多少？** 经验值：简单查询 5，中等任务 10，复杂研究 20-30。**永远不要设 `float('inf')`**。

---

## 反模式（什么不该做）

### ❌ 无 `max_steps` → 无限循环烧钱

```python
# 坏：没有上限，模型可以无限循环
while True:
    response = call_llm(messages)
    if not response.tool_calls:
        break
    # ...
```

**后果**：模型卡在循环里，你的 API 账单像火箭一样飞涨。

### ❌ 无单步超时 → 单步卡住整个 Agent

```python
# 坏：工具调用没有超时，万一工具卡住，整个 Agent 卡住
result = tool_functions[name](**args)  # 万一这里 hang 住？
```

**正确**：给工具调用加超时（第06章错误处理会详细讲，本章先用简单 try/except）。

### ❌ 循环体内副作用未清理 → 状态污染

```python
# 坏：循环里修改了全局状态，下一轮被污染
GLOBAL_CACHE.clear()  # 每轮都清，但如果中途异常就留下脏数据
```

**正确**：Agent 循环里的状态应该是**本次循环的局部变量**（`messages` 列表）。跨会话的记忆是第05章的事。

### ❌ 把"模型不调工具"当成错误

```python
# 坏：以为模型不调工具 = 出错了
if not assistant_msg.tool_calls:
    raise Exception("模型没调工具！")  # ❌ 错！这才是正常终止！
```

**正确**：模型不调工具是**正常完成信号**，返回它的 content 即可。

### ❌ 每步都重建 messages → 丢失上下文

```python
# 坏：每步 messages = [...] 重新开始，模型记不住之前调了什么
for step in range(MAX_STEPS):
    messages = [{"role": "user", "content": query}]  # ❌ 丢历史！
    response = call_llm(messages)
```

**正确**：`messages` 在循环外初始化，循环内只**追加**，不重建。模型需要完整历史来决策。

## 常见错误

> 概念懂了，实际写代码还是会踩坑。这些是初学者最常犯的错误。

| 错误 | 症状 | 解决 |
|------|------|------|
| 循环内重建 messages | Agent 每步都"失忆"，反复调同一个工具 | messages 在循环外初始化，循环内只 append，不重新赋值 |
| 模型不调工具时抛异常 | 正常完成时报错，Agent 把终止信号当错误 | `if not tool_calls` 是正常终止，直接返回 content |
| max_steps 设太大 | Agent 跑几十步才停，API 费用飙升 | 简单任务 5，中等任务 10，复杂任务 20 到 30，绝不用 `float('inf')` |
| 工具结果没追加到 messages | 模型看不到工具返回值，反复调同一个工具 | 执行工具后必须 `messages.append({"role": "tool", ...})` |

---

## 完整流程图（4 步多城市天气推荐）

以"查北京、上海、深圳天气，推荐最适合旅行的城市"为例：

```
Step 1: model → get_weather("北京")
        tool  → "北京晴 25°C"
Step 2: model → get_weather("上海")
        tool  → "上海多云 28°C"
Step 3: model → get_weather("深圳")
        tool  → "深圳小雨 30°C"
Step 4: model → "推荐北京，因为晴朗且温度宜人..."  ← 不再调工具，终止
```

注意：**模型自己决定**先查哪个、查完三个再总结。你的代码只负责"循环 + 执行 + 反馈"。

> 💡 **并行工具调用**：OpenAI 较新的模型支持**单步返回多个 tool_calls**（如一步查三个城市）。本章代码已支持——`for tc in tool_calls` 循环处理所有。但为了演示清晰，本例让模型一步步查。

---

## 运行示例

```bash
# Python
cd ai-agent/04-agent-loop
python3 python/main.py

# TypeScript
cd ai-agent/04-agent-loop
npx tsx typescript/main.ts
```

输出会用 `OUT:step{N}:` 前缀标记每一步，方便你追踪 Agent 的决策链。

### 关于 max_steps 防护演示

为了**不依赖真实 API** 就能可靠验证 max_steps 逻辑，本章实现了一个 **mock agent loop**——用预设的"永远返回 tool_calls"的假模型响应，模拟无限循环场景。

这样即使没有有效的 API 密钥，你也能看到 Agent 在 step=10 时优雅停止。运行 `main.py` / `main.ts` 时观察 `OUT:max_steps:` 标记的输出。

---

## 兼容性注意

- **Ollama `qwen2.5vl:latest` 不支持 tools API**（返回 400）。本章代码用 try/catch 优雅处理：API 失败时自动降级为**离线 mock 演示**，依然展示完整的 Agent 循环逻辑。
- **`.env` 是占位符密钥**（`OPENAI_API_KEY=sk-REPLACE-ME`）→ 真实 API 调用会 401。代码会捕获并演示本地循环逻辑。

---

## 下一步

本章你让「任务助手 Agent」获得了**自主性**——它能多步循环调工具直到完成任务。但有个问题：

> **每开一个新的 `agent_loop`，模型都"失忆"了**——它不记得上一次对话。

第05章「记忆系统」解决这个问题：把 `messages` 列表持久化（短期记忆）或用摘要/向量库（长期记忆），让 Agent 跨会话记住上下文。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
