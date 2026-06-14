# 第11章 上下文工程（Context Engineering）

> **Prompt 工程的进化**——当 Agent 动辄调用几十次工具、跑上百步循环时，"写好一个 prompt"已经不够了。
> 真正决定 Agent 质量的，是**你如何管理喂给模型的上下文**：压缩什么、隔离什么、为每次调用分配多少 token 预算。
> 这就是 2025 年后兴起的「上下文工程」。

---

## 本章目标

学完本章，你将理解：

1. **上下文工程是什么**，它为何是 Prompt 工程的进化形态
2. **上下文不是免费的**：越长检索越差（lost in the middle）、成本越高（O(n²)）
3. **上下文压缩（Compaction）**：超阈值时把旧轨迹摘要成一条 system 消息
4. **子 Agent 隔离**：主 Agent 派子 Agent 干重活，只收回摘要，不看全量轨迹
5. **Token 预算管理**：每次调用前估算 token，接近上限时自动触发压缩
6. **笔记式记忆**：边执行边记要点，而非堆叠原始轨迹
7. **两大反模式**：无限堆叠上下文、把子 Agent 全量输出塞回主 Agent

---

## 1. 上下文工程 = Prompt 工程的进化

### Prompt 工程的局限

2023 年的 Prompt 工程教你怎么写一个好的 system prompt、怎么 few-shot、怎么 CoT。
它隐含的假设是：**每次调用是独立的、上下文是短小的、你手动控制一切。**

但真实的 Agent 不是这样运行的：

```python
# 一个研究 Agent 可能跑 30 步，每步都有工具结果
messages = [
    {"role": "system", "content": system_prompt},        # 200 tokens
    {"role": "user", "content": "研究 AI Agent 框架趋势"}, # 20 tokens
    {"role": "assistant", "content": "我先搜索...", "tool_calls": [...]},  # 150 tokens
    {"role": "tool", "content": search_results_page_1},   # 2000 tokens !!
    {"role": "assistant", "content": "找到了，继续...", "tool_calls": [...]}, # 150 tokens
    {"role": "tool", "content": search_results_page_2},   # 1800 tokens !!
    # ... 重复 20 次 ...
    # 第 30 步时，messages 已经 40000+ tokens
]
```

**问题**：第 30 步时，模型要重新阅读前面 29 步的全部原始轨迹——包括那些早已过时的中间搜索结果。
这不仅**烧钱**（每步都付全量 token），还**降低质量**（模型在长上下文中"迷失"，忽略关键信息）。

### 上下文工程的核心命题

> **上下文工程 = 主动设计"每次 LLM 调用时，模型实际看到什么"。**

它把上下文当作一个**有限、昂贵的资源**来管理，而不是"把所有东西塞进去让模型自己看着办"。
三大支柱：

| 支柱 | 一句话 | 解决的问题 |
|------|--------|-----------|
| **压缩（Compaction）** | 把旧的、已消化完的轨迹摘要成一句话 | 长对话/长任务的质量衰退 + 成本爆炸 |
| **子 Agent 隔离** | 把脏活累活外包给子 Agent，只收回摘要 | 主 Agent 上下文被工具结果淹没 |
| **Token 预算管理** | 每次调用前估算 token，接近上限就压缩 | 无主动管理导致超限报错或静默截断 |

> 💡 **Claude Code 的 `/compact` 命令、Cursor 的长上下文摘要、Devin 的会话压缩**——
> 这些 2024-2025 年流行的 Agent 功能，底层全是上下文工程。原理一致，只是工程化程度不同。

---

## 2. 上下文不是免费的（为什么必须管理）

很多人以为"上下文窗口 128K，塞多少都行"。这是危险的误解。

### 2.1 成本是 O(n²) 的

一个 N 轮的对话，用 ConversationBuffer（完整保留历史）时：

```
第 1 轮传 1 条消息
第 2 轮传 2 条消息
...
第 N 轮传 N 条消息
```

累计传的 token 数 = 1 + 2 + ... + N = **N(N+1)/2 = O(N²)**。

| 对话轮数 | 每轮平均 token | 累计成本（tokens） |
|----------|---------------|-------------------|
| 10 轮 | 100 | 5,500 |
| 50 轮 | 100 | 127,500 |
| 100 轮 | 100 | 505,000 |

100 轮对话累计 50 万 token——按 gpt-4o-mini 的价格（$0.15/百万 input），这是 $0.075；
按 gpt-4o（$2.5/百万 input），这是 **$1.26**。一个用户的 100 轮对话就烧掉一美元。

### 2.2 质量随长度下降（Lost in the Middle）

即使窗口够大，模型在长上下文中的表现也会下降。这是著名的
["Lost in the Middle" 论文](https://arxiv.org/abs/2307.03172)（Liu et al., 2023）的结论：

> 把关键信息放在上下文的**开头或结尾**时，模型召回率高；放在**中间**时，召回率显著下降。

```
[质量高] system prompt ... [质量最低 ...中间的信息...] 最近几轮 [质量高]
```

这意味着：**你堆在中间的旧工具结果，模型基本"看不见"**。堆得越多，噪声越大，越容易答错。

### 2.3 延迟随长度增长

输入越长，模型的首 token 延迟（TTFT）越高。40K tokens 的输入可能比 4K tokens 慢 2-3 秒。
对于一个跑 30 步的 Agent，每步多 2 秒 = 整体多 1 分钟。

> ⚠️ **结论**：上下文窗口是"上限"，不是"推荐用量"。**主动控制上下文长度 = 同时省钱、提质量、降延迟。**

---

## 3. 上下文压缩（Compaction）

### 3.1 核心思想

当上下文超过阈值时，**把已经"消化完"的旧轨迹交给 LLM 摘要**，压缩成一条 system 消息，
只保留最近 N 轮的原文。

```
压缩前（40 条消息，12000 tokens）：
  [system] [user] [asst+tool] [tool] [asst+tool] [tool] ... × 19 ... [最近几轮]

压缩后（8 条消息，2000 tokens）：
  [system] [system: 之前对话摘要: 用户问了X，我们做了Y，得到Z...] [最近 6 轮原文]
```

### 3.2 触发机制

```python
class ContextCompactor:
    def __init__(self, threshold=2000, keep_recent=6):
        self.threshold = threshold  # token 阈值
        self.keep_recent = keep_recent  # 保留最近几轮原文
        self.summary = ""  # 累积摘要
        self.messages = []

    def add(self, message):
        self.messages.append(message)
        if self.estimate_tokens(self.messages) > self.threshold:
            self.compact()

    def compact(self):
        """把旧消息摘要，保留最近 N 轮。"""
        old = self.messages[:-self.keep_recent]
        recent = self.messages[-self.keep_recent:]
        new_summary = self.llm_summarize(old)
        self.summary = self.summary + "\n" + new_summary if self.summary else new_summary
        self.messages = recent
```

### 3.3 何时压缩

- **按 token 阈值**（推荐）：`estimate_tokens(messages) > threshold` 时触发。直接控制成本。
- **按消息条数**：简单但粗略——一条长工具结果可能等于 10 条短对话。
- **按步骤数**：Agent 循环每 N 步压缩一次。适合固定节奏的任务。

> 💡 本章用 **token 阈值**，因为它直接对应"成本"和"lost in the middle"风险。

### 3.4 Token 估算（不依赖 tiktoken）

精确计 token 需要 tokenizer（如 `tiktoken`），但本章刻意不装它——用**字符数估算**即可：

```python
def estimate_tokens(messages):
    """估算 messages 的 token 数（粗略，1 token ≈ 3 字符）。"""
    text = json.dumps(messages, ensure_ascii=False)
    return len(text) // 3
```

这个估算对教学足够（误差 ±20%），且**零依赖、纯 Python、100% 离线**。
生产环境换成 `tiktoken.encoding_for_model(model).encode(text)` 即可。

---

## 4. 子 Agent 隔离（Sub-Agent Isolation）

### 4.1 问题：主 Agent 上下文被工具结果淹没

假设主 Agent 要"研究三个主题并汇总"。如果它自己直接调搜索工具：

```
主 Agent 上下文（直接调工具）：
  [system] [user: 研究A,B,C]
  [asst: 我先搜A] [tool: A的1000字搜索结果]
  [asst: 再搜B] [tool: B的1000字搜索结果]
  [asst: 再搜C] [tool: C的1000字搜索结果]
  [asst: 汇总...]  ← 主 Agent 要重新阅读上面 3000 字的原始结果
```

主 Agent 的上下文被原始搜索结果塞满——**它要做的是"汇总"，不需要重新阅读每个原始结果**。

### 4.2 方案：派子 Agent 干脏活，只收回摘要

```python
# 主 Agent 派子 Agent 去研究主题 A
subagent_trace = run_subagent("深入研究主题 A")
# subagent_trace 内部有 6 步工具调用 + 3000 字原始结果
# 但主 Agent 只收回一段摘要：
summary_of_a = subagent_trace.summarize()  # "主题A的核心是X，关键发现是Y..."

# 主 Agent 上下文：
#   [system] [user: 研究A,B,C]
#   [asst: 子Agent研究了A: summary_of_a]  ← 只有一段摘要！
#   [asst: 子Agent研究了B: summary_of_b]
#   [asst: 子Agent研究了C: summary_of_c]
#   [asst: 汇总...]  ← 上下文干净，只有摘要
```

**隔离的核心**：子 Agent 的完整轨迹（几千 token）**不进入主 Agent 的上下文**，
只有它的摘要（几百 token）进入。这是"信息分层"——主 Agent 只看高层摘要，子 Agent 看原始细节。

### 4.3 上下文层级

```
主 Agent（高层上下文，几千 token）
  ├── 摘要1 ← 子Agent1（完整轨迹，几万 token，隔离）
  ├── 摘要2 ← 子Agent2（完整轨迹，几万 token，隔离）
  └── 摘要3 ← 子Agent3（完整轨迹，几万 token，隔离）
```

每个子 Agent 有**独立的上下文窗口**，它们的内部轨迹互不污染，也不污染主 Agent。
这就是为什么复杂的 Agent 系统（如 Devin、Claude Code 的子任务）天然需要多 Agent 架构——
**不是为了分工，而是为了上下文隔离**。

> 💡 **Claude Code 的 Task 工具、OpenAI Agents SDK 的 Handoff、LangGraph 的 subgraph**——
> 这些"子任务"机制，底层全是子 Agent 隔离。原理一致。

---

## 5. Token 预算管理

### 5.1 为什么需要预算

上下文窗口有硬上限（gpt-4o-mini 是 128K）。但"不超上限"不等于"健康"——
正如 2.2 节所说，超过一定长度后质量下降。所以我们要设一个**软预算**（如 4000 tokens），
主动控制在健康范围内。

### 5.2 预算循环

```python
class TokenBudget:
    def __init__(self, budget=4000, threshold_ratio=0.8):
        self.budget = budget
        self.threshold = int(budget * threshold_ratio)  # 80% 触发压缩
        self.messages = []

    def add_and_check(self, message):
        """添加消息，返回是否需要压缩。"""
        self.messages.append(message)
        tokens = estimate_tokens(self.messages)
        if tokens > self.threshold:
            return True  # 触发压缩
        return False
```

### 5.3 完整的"对话→检查→压缩→继续"循环

```python
budget = TokenBudget(budget=4000)
for user_input in conversation:
    budget.add({"role": "user", "content": user_input})
    response = call_llm(budget.messages)
    budget.add({"role": "assistant", "content": response})

    tokens = estimate_tokens(budget.messages)
    print(f"当前 {tokens}/{budget.budget} tokens ({tokens/budget.budget:.0%})")

    if tokens > budget.threshold:
        print(f"⚠️ 超过 {budget.threshold_ratio:.0%}，触发压缩")
        budget.compact()  # 压缩
        print(f"压缩后: {estimate_tokens(budget.messages)} tokens")
```

这个循环是所有长任务 Agent 的基础设施。**没有它，Agent 跑久了必然质量衰退或超限报错。**

---

## 6. 笔记式记忆（边执行边记要点）

压缩和隔离是"事后处理"——等上下文长了才压缩。还有一种更主动的策略：
**边执行边记笔记**，让 Agent 自己提炼要点，而非堆叠原始轨迹。

### 6.1 原理

```python
# 普通模式：堆叠原始轨迹
messages = [
    {"role": "user", "content": "研究 AI Agent 框架"},
    {"role": "tool", "content": "LangChain 是...（2000字）"},  # 原始结果
    {"role": "tool", "content": "LangGraph 是...（1800字）"},  # 原始结果
    # ... 堆叠 ...

# 笔记模式：每步提炼一句话要点
notes = []
for step in agent_steps:
    result = execute_tool(step)
    note = llm_extract_key_point(result)  # "LangChain 是主流框架，支持工具调用"
    notes.append(note)
    # 原始 result 不进主上下文，只有 note 进

# 最终上下文只有笔记，不是原始结果
messages = [{"role": "system", "content": "已知要点: " + "\n".join(notes)}]
```

### 6.2 与压缩的区别

| 特性 | 压缩（Compaction） | 笔记式记忆 |
|------|-------------------|-----------|
| 时机 | 事后（超阈值才触发） | 事中（每步都记） |
| 粒度 | 摘要整段旧对话 | 每步提炼一句要点 |
| 上下文质量 | 中（摘要可能丢细节） | 高（实时提炼，针对性强） |
| API 成本 | 低（偶尔摘要） | 高（每步多一次提炼调用） |

**实践建议**：简单任务用压缩（成本低），复杂长任务用笔记式（质量高），或两者结合。

---

## 7. 反模式（什么不该做）

### ❌ 反模式 1：无限堆叠上下文

```python
# 坏：把所有工具结果、所有中间思考都塞进 messages，从不压缩
messages = []
for step in range(100):
    messages.append({"role": "tool", "content": huge_search_result})
    response = client.chat.completions.create(messages=messages)  # 第100步传几万token
```

**后果**：
1. **成本爆炸**：O(n²) 增长，100 步可能烧掉几美元
2. **质量下降**：lost in the middle，模型忽略关键信息
3. **延迟增大**：输入越长，TTFT 越高
4. **最终超限**：硬报错 `context_length_exceeded`

**正确**：设 token 预算，超阈值时用 ContextCompactor 压缩，或用笔记式记忆边走边记。

### ❌ 反模式 2：把子 Agent 全量输出塞回主 Agent

```python
# 坏：子 Agent 跑完了，把它的完整轨迹（含几千字工具结果）全塞回主 Agent
subagent_result = run_subagent("研究主题A")
for msg in subagent_result.full_trace:  # 6步轨迹 + 3000字原始结果
    main_messages.append(msg)  # 主 Agent 上下文被淹没
```

**后果**：完全丧失了子 Agent 隔离的意义。主 Agent 上下文照样爆炸，只是把"自己调工具"换成"看子 Agent 调工具的录像"。

**正确**：子 Agent 只返回**摘要**（1-2 句话），不返回完整轨迹。

```python
subagent_result = run_subagent("研究主题A")
main_messages.append({
    "role": "assistant",
    "content": f"子Agent研究发现: {subagent_result.summary}"  # 只要摘要
})
```

---

## 运行示例

```bash
# Python
cd ai-agent/11-context-engineering
python3 python/main.py

# TypeScript
cd ai-agent/11-context-engineering
npx tsx typescript/main.ts
```

输出用 `OUT:token:` / `OUT:compact:` / `OUT:subagent:` / `OUT:budget:` 前缀标记四个功能点。

### 离线可运行（关键设计）

本章四个 demo 的设计：
- **Token 估算**：纯字符数计算，不依赖任何 API，100% 离线
- **上下文压缩**：先试真实 API 摘要，失败降级 mock 摘要（预设文本）
- **子 Agent 隔离**：预设子 Agent 多步轨迹 + 摘要结果，纯离线
- **Token 预算循环**：纯本地模拟对话，100% 离线

所以即使 `.env` 是占位符 `OPENAI_API_KEY=sk-REPLACE-ME`，四个 demo 都能完整跑通，exit 0。

---

## 与前序章节的关系

- **第05章（记忆系统）**：SummaryMemory 是压缩的雏形（按消息条数触发）。本章 ContextCompactor 是它的进化版——**按 token 触发**，更直接控制成本。
- **第04章（Agent 循环）**：agent_loop 的 messages 列表会随步骤增长。本章教你怎么管住它。
- **第10章（多 Agent 编排）**：多 Agent 协作时，每个 Agent 的上下文要隔离。本章的子 Agent 隔离是这个思想的底层机制。

---

## 下一步

本章你学会了管理上下文的三大支柱。但真实的 Agent 框架需要把这些组件**组装成可复用的架构**——
第12章「从零造框架：架构设计」开始，我们把这些原理固化成一个 mini Agent 框架。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
