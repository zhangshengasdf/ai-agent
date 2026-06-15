# 第10章 多 Agent 编排（Supervisor-Worker、Handoffs、共享记忆）

> **「任务助手 Agent」从"孤胆英雄"变成了"团队"**——不再是一个 Agent 硬扛所有事，
> 而是让多个专门化的 Agent 分工协作：一个当 Supervisor（调度）、几个当 Worker（干活），
> 遇到超出自己能力范围的问题还能 Handoff（转交）给更专业的同事。

## TL;DR

> **30 秒速读**：多 Agent 用 Supervisor-Worker 模式分工协作（Supervisor 分派任务、Worker 执行）和 Handoff 模式转交对话（客服 → 技术专家），但最大的陷阱是"用早了"——大多数任务单 Agent 就能搞定。
> 
> **如果只记一件事**：先问"单个 Agent 为什么搞不定？"，能明确回答（工具过载 / 上下文过长 / 角色冲突 / 需要并行）才上多 Agent，否则回去优化单 Agent。

---

## 本章目标

学完本章，你将理解：

1. **何时需要多 Agent**——以及更重要的：**何时不需要**
2. **Supervisor-Worker 模式**：Supervisor 分解任务 → 分派 Worker → Worker 汇报 → Supervisor 汇总
3. **Agent Handoffs**：一个 Agent 把对话上下文转交给另一个 Agent
4. **共享记忆与消息传递**：多个 Agent 之间怎么共享上下文、避免并发写冲突
5. **协作协议设计**：Agent 之间的"通信契约"
6. **反模式**：过度工程、无明确协议、共享记忆无并发控制

---

## 0. 先泼一盆冷水：多 Agent 的最大陷阱是"用早了"

在讲"怎么编排多 Agent"之前，必须先讲 Anthropic 反复强调的一条共识：

> **"从最简单的方案开始，许多场景只需优化单次 LLM 调用就够了。"** —— Anthropic《Building Effective Agents》

```
❌ 错误顺序：
"我要做一个 Agent 系统" → "多 Agent 一定更强大" → 上 5 个 Agent → 复杂度爆炸 → 调不动

✅ 正确顺序：
"我要解决这个问题" → "先用单次 LLM 调用试试" →
够用？→ 交付，别加复杂度
不够？→ 加工具调用（第03章）
还不够？→ 加 Agent 循环（第04章）
还不够？→ 加 ReAct/规划（第07/08章）
都不够？→ 这时候才考虑多 Agent（本章）
```

多 Agent 引入的不是"能力"，而是**复杂度**：
- Agent 之间的通信协议要设计
- 上下文在 Agent 间传递（成本和延迟翻倍）
- 错误处理变难（一个 Worker 崩了，整个流程怎么办？）
- 调试变难（"为什么 Supervisor 分派给了错误的 Worker？"）

**判断标准**：如果你的任务能被一个 Agent（带工具 + 规划）处理，就**别**上多 Agent。
只有当单个 Agent 的上下文窗口、工具数量、角色职责"撑不住"时，多 Agent 才合理。

> 💡 **经验法则**：先问"单个 Agent 为什么搞不定？" 如果你能明确回答（工具过载 / 上下文过长 / 角色冲突 / 需要并行），再用多 Agent。答不上来，回去优化单 Agent。

---

## 1. 何时真的需要多 Agent

下面是四种**有充分理由**使用多 Agent 的场景。

### 1.1 工具过载（Tool Overload）

单个 Agent 的工具列表如果超过 **15-20 个**，模型的工具选择准确率会显著下降——它会"眼花"，选错工具或漏选。

```
单 Agent 装满 30 个工具：
  模型经常在 search_wiki 和 search_web 之间选错
  或者在 calculate 和 run_python 之间纠结

拆成多 Agent：
  ResearchAgent → 只装 search_wiki, search_web（2 个工具，准确率 95%+）
  CodeAgent     → 只装 run_python, write_file（2 个工具，准确率 95%+）
  Supervisor    → 决定派给谁（不需要工具，只做路由）
```

### 1.2 上下文过长（Context Window Pressure）

单个 Agent 把"对话历史 + 系统提示 + 工具结果"全塞进去，上下文窗口很快爆炸（成本飙升、"lost in the middle"导致中部信息检索精度下降、推理变慢）。

```
单 Agent 处理"研究 + 写作 + 审查"：
  上下文 = [研究阶段的 50 条检索结果] + [写作初稿] + [审查意见]
  → 几万 token，又贵又慢

拆成多 Agent：
  ResearchAgent 的上下文只有检索结果（用完即弃）
  WriterAgent 的上下文只有"研究摘要 + 写作任务"（精简）
  每个 Agent 的上下文窗口独立，互不污染
```

### 1.3 角色冲突（Role Confusion）

一个 Agent 同时扮演"作者"和"审稿人"，会陷入**自我矛盾**——写的时候想"审稿人会怎么挑刺"，审的时候又"放自己一马"。

```
单 Agent 既写又审：
  写作阶段：模型在"作者"模式
  审查阶段：模型切换到"审稿人"模式 → 但同一个模型容易"放水"

拆成多 Agent：
  WriterAgent  → 系统 prompt："你是撰稿人，目标是写出好文章"
  ReviewerAgent → 系统 prompt："你是严格的审稿人，目标是挑出所有问题"
  两个 Agent 互相独立，Reviewer 不会对 Writer "放水"
```

### 1.4 需要并行（Parallelism）

某些任务的子步骤是独立的，可以并行执行——单 Agent 是串行的，多 Agent 可以并发。

```
单 Agent 串行检索 4 个维度：4 × 3s = 12s
多 Agent 并行检索 4 个维度：max(3s, 3s, 3s, 3s) = 3s
```

> 本章为了教学清晰，主要演示**串行**的 Supervisor-Worker。并行是多 Agent 的高级用法，进阶项目会涉及。

---

## 2. Supervisor-Worker 模式

这是最常见的多 Agent 架构，也最容易理解。

### 2.1 架构图

```
┌──────────────────────────────────────────────────────┐
│                  用户任务                             │
│            "写一篇 AI Agent 调研报告"                 │
└──────────────────────┬───────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │   Supervisor    │  ← 分解任务，决定分派
              │   （调度者）     │     不直接干活，只做路由
              └────────┬────────┘
                       │ 分派任务（结构化输出）
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │Researcher│ │  Writer  │ │  Coder   │
    │ (研究员) │ │ (撰稿人) │ │ (程序员) │
    │          │ │          │ │          │
    │工具:     │ │工具:     │ │工具:     │
    │search_wiki│ │(无，纯写作)│ │(无，纯写码)│
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              ┌─────────────────┐
              │   Supervisor    │  ← 收集 Worker 结果，汇总
              │   （汇总者）     │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │    最终输出      │
              └─────────────────┘
```

### 2.2 每个 Worker 是什么

**关键认知**：每个 Worker 本质就是**一个带特定 system prompt + 工具集的 LLM 调用**。

```python
# Researcher = system prompt 决定角色 + 工具集决定能力
researcher = Worker(
    name="Researcher",
    system_prompt="你是研究员，负责检索信息。只返回事实，不要编造。",
    tools=[search_wiki],           # 只有搜索工具
)

# Writer = 另一个 system prompt + 可能无工具
writer = Worker(
    name="Writer",
    system_prompt="你是撰稿人，负责把信息整理成文。注重结构和可读性。",
    tools=[],                      # 纯写作，无需工具
)
```

**为什么有效**：
- **system prompt 专注**：每个 Worker 只需"扮演一个角色"，不会被"既要又要"搞乱
- **工具集精简**：每个 Worker 只看到自己需要的工具，不会选错
- **上下文隔离**：Worker A 的工作过程不污染 Worker B

> 💡 **本章实现故意不用框架**（不用 CrewAI / AutoGen / LangGraph）。每个 Worker 就是一次普通的 `chat.completions.create` 调用——这样你能看清"多 Agent"在底层到底是怎么运作的，没有任何魔法。理解了底层，再用框架时就不会被"框架的抽象"困住。

### 2.3 Supervisor 的核心：结构化决策

Supervisor 不直接干活，它的唯一职责是**决定把任务派给谁**。这个决策通常用**结构化输出**实现：

```python
# Supervisor 用 json_object 输出"分派决策"
SUPERVISOR_PROMPT = """\
你是一个任务调度者。用户给你一个任务，你需要决定把它分派给哪个 Worker。

可用 Worker：
- Researcher：负责检索信息（有 search_wiki 工具）
- Writer：负责把信息整理成文
- Coder：负责写代码片段

输出 JSON：{"assignments": [{"worker": "Worker名", "subtask": "子任务描述"}]}
"""

response = client.chat.completions.create(
    model=cfg.model,
    messages=[
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": task},
    ],
    response_format={"type": "json_object"},
)
plan = AssignmentPlan.model_validate_json(response.choices[0].message.content)
# plan.assignments = [
#   Assignment(worker="Researcher", subtask="检索 AI Agent 定义"),
#   Assignment(worker="Writer", subtask="把检索结果写成报告"),
# ]
```

这和第08章 Plan-and-Execute 的规划阶段几乎一样——区别在于：Plan-and-Execute 分解的是**步骤**，Supervisor 分解的是**任务 + 指定执行者**。

### 2.4 完整流程

```
1. 用户任务 → Supervisor
2. Supervisor 用结构化输出分解成 [子任务+Worker] 列表
3. 依次（或并行）把子任务派给对应 Worker
   - Worker 收到 subtask + 上下文，用 system prompt + tools 执行，返回结果
4. Supervisor 收集所有 Worker 的结果，汇总成最终输出
```

### 2.5 Supervisor-Worker vs Plan-and-Execute

你可能发现这和第08章的 Plan-and-Execute 很像。区别在哪？

| 维度 | Plan-and-Execute（第08章） | Supervisor-Worker（本章） |
|------|---------------------------|--------------------------|
| **分解对象** | 步骤（step1, step2...） | 任务+执行者（worker+subtask） |
| **执行者** | 同一个 Agent | 不同 Agent（每个 Worker 角色不同） |
| **角色多样性** | 单一角色 | 多角色（Researcher/Writer/Coder...） |
| **system prompt** | 全程一个 | 每个 Worker 独立 |
| **工具集** | 全程一个 | 每个 Worker 独立 |

一句话：**Plan-and-Execute 是"一个人分步干活"，Supervisor-Worker 是"一个团队分工干活"**。

---

## 3. Agent Handoffs（任务转交）

### 3.1 什么是 Handoff

Handoff 是另一种多 Agent 模式：一个 Agent 在处理任务时，发现"这事超出我的能力/职责范围"，于是**把整个对话上下文转交给另一个 Agent** 继续处理。

```
用户：我想退货
客服 Agent：好的，请提供订单号...（处理退货流程）
用户：退货页面报了 500 错误，代码 ERR_DEPLOY_123
客服 Agent：检测到技术关键词 → Handoff → 技术 Agent（带上完整对话上下文）
技术 Agent：我看到你遇到了 ERR_DEPLOY_123，这是部署问题，我帮你查...
```

### 3.2 Handoff vs Supervisor-Worker 的区别

| 维度 | Supervisor-Worker | Handoff |
|------|-------------------|---------|
| **触发** | Supervisor 主动分派 | 当前 Agent 判断"超出范围"时被动触发 |
| **控制流** | Supervisor 始终主导 | 控制权转移（A 全权交给 B） |
| **上下文** | 分派时传递 subtask | A 把完整对话历史交给 B |
| **适合场景** | 任务可预先分解 | 任务进行中发现需要专家 |
| **类比** | 项目经理分派任务 | 客服转接技术支持 |

### 3.3 Handoff 的触发机制

最常见的触发方式是**关键词检测**（简单可靠）或**LLM 判断**（更智能但更贵）。

```python
TECH_KEYWORDS = ["代码", "bug", "部署", "错误码", "报错", "异常", "崩溃"]

def needs_handoff(user_message: str) -> bool:
    """检测是否需要转交给技术 Agent。"""
    return any(kw in user_message for kw in TECH_KEYWORDS)
```

更高级的做法是让客服 Agent 自己判断（用结构化输出）：
```python
# 客服 Agent 输出：{"action": "answer" | "handoff_tech", "content": "..."}
```

本章教学用**关键词检测**（简单、可靠、不依赖 API），生产场景可以升级为 LLM 判断。

### 3.4 Handoff 的上下文传递

Handoff 的关键是**把对话历史完整传递**，让接手的 Agent "知道之前发生了什么"：

```python
# 客服 Agent 的对话历史
customer_service_messages = [
    {"role": "system", "content": CUSTOMER_SERVICE_PROMPT},
    {"role": "user", "content": "我想退货"},
    {"role": "assistant", "content": "好的，请提供订单号..."},
    {"role": "user", "content": "退货页面报了 500 错误，代码 ERR_DEPLOY_123"},
]

# Handoff：替换 system prompt，保留对话历史
tech_messages = [
    {"role": "system", "content": TECH_EXPERT_PROMPT},  # 换角色
    *customer_service_messages[1:],  # 保留 user/assistant 对话
]
```

> 💡 **为什么要替换 system prompt**：对话历史里的 user/assistant 消息是"事实记录"（发生了什么），但 system prompt 是"角色定义"（你是谁）。Handoff 后角色变了，必须换 system prompt，否则技术 Agent 还以为自己是客服。

---

## 4. 共享记忆与消息传递

多 Agent 之间需要共享信息。有三种常见方式：

### 4.1 消息传递（Message Passing）—— 最推荐

Agent 之间通过**显式的消息**传递信息，不共享可变状态。

```
Supervisor → Researcher: {"subtask": "检索 AI Agent 定义"}
Researcher → Supervisor: {"result": "AI Agent 是..."}
Supervisor → Writer: {"subtask": "写成报告", "context": "AI Agent 是..."}
Writer → Supervisor: {"result": "# AI Agent 调研报告\n..."}
```

**优点**：
- **无并发问题**：每个 Agent 只读写自己范围内的数据
- **可追溯**：消息流就是执行日志，好调试
- **可重放**：记录消息序列就能重现整个流程

**缺点**：信息要在消息里"搬来搬去"，有一定开销。

### 4.2 共享黑板（Shared Blackboard）

所有 Agent 读写同一个**共享数据结构**（"黑板"）。

```
shared_context = {"research_results": [], "draft": "", "critique": ""}

# Researcher 写
shared_context["research_results"].append("AI Agent 是...")
# Writer 读 + 写
shared_context["draft"] = write(shared_context["research_results"])
```

**优点**：Agent 不用关心"谁给的数据"，只管读写黑板。
**缺点**：**并发写冲突**——多个 Agent 同时写黑板会互相覆盖。

### 4.3 共享对话历史（Shared Conversation）

所有 Agent 共享同一个 `messages` 列表，像"群聊"一样。

```
shared_messages = [
    {"role": "system", "content": "你是团队..."},
    {"role": "user", "content": "写报告"},
    {"role": "assistant", "name": "Researcher", "content": "查到..."},
    {"role": "assistant", "name": "Writer", "content": "写成..."},
]
```

**优点**：上下文最完整，每个 Agent 都能看到全貌。
**缺点**：所有 Agent 的输出堆在一起，很快超窗口。

### 4.4 本章的选择

本章的 Supervisor-Worker 用**消息传递**（最简单、最可靠、无并发问题）。Handoff 用**共享对话历史**（因为 Handoff 本质就是"换角色继续同一对话"）。

> ⚠️ **并发控制**：如果你用共享黑板或共享历史，且 Worker 是**并行**执行的，**必须**加锁（`threading.Lock` / `mutex`）。本章为了教学清晰用串行执行，不涉及并发。生产场景的多 Agent 并行必须处理这个问题。

---

## 5. 协作协议设计

多 Agent 系统的成败，很大程度上取决于**Agent 之间的通信协议**设计得有多清晰。

### 5.1 什么是协作协议

协议 = Agent 之间传递消息的**格式契约**。好的协议应该：
- **字段明确**：接收方知道每个字段什么意思
- **版本化**：协议升级时不破坏旧 Agent
- **可校验**：用 Pydantic / Zod / JSON Schema 校验消息合法性

### 5.2 本章的协议设计

```python
# Supervisor → Worker 的分派消息
class Assignment(BaseModel):
    worker: str        # Worker 名（"Researcher"/"Writer"/"Coder"）
    subtask: str       # 子任务描述

class AssignmentPlan(BaseModel):
    assignments: list[Assignment]

# Worker → Supervisor 的汇报消息（隐式：直接返回字符串结果）
# Worker 收到 subtask，返回 result（字符串）
```

### 5.3 反面教材：没有协议

```python
# 坏：Supervisor 随便发个字符串，Worker 要猜格式
supervisor.send("Researcher 帮我查一下 AI Agent")

# 好：用结构化消息
supervisor.send(Assignment(worker="Researcher", subtask="查 AI Agent 定义"))
```

没有协议的多 Agent 系统，就像"没有 API 文书的微服务"——能跑但维护噩梦。

---

## 6. 反模式（什么不该做）

### ❌ 反模式 1：为简单任务强上多 Agent（过度工程）

```python
# 坏：查天气这种单步任务，硬要拆成 Supervisor + WeatherWorker
task = "北京今天天气？"
plan = supervisor.decompose(task)  # plan = [{"worker": "WeatherWorker", "subtask": "查北京天气"}]
result = weather_worker.execute(plan[0].subtask)
final = supervisor.synthesize([result])
# 为一个 1 次工具调用能解决的问题，引入了 3 次 LLM 调用（Supervisor分解 + Worker执行 + Supervisor汇总）
```

**后果**：延迟 3 倍、成本 3 倍、调试时要在 3 个 Agent 之间跳。

**正确**：先用第04章的单 Agent 循环。只有当单 Agent 撑不住（工具过载/上下文过长/角色冲突）时，才上多 Agent。

### ❌ 反模式 2：Agent 间无明确协议

```python
# 坏：Worker 返回自由文本，Supervisor 要猜结构
result = worker.execute("查 AI Agent 定义")
# result 可能是 "AI Agent 是..." 也可能是 "查询失败：网络错误" 也可能是 ""
supervisor.synthesize([result])  # Supervisor 不知道哪个是错误、哪个是结果
```

**后果**：Agent 之间的"误解"导致流程出错，且难以定位。

**正确**：用 Pydantic / Zod 定义清晰的消息结构，Worker 返回结构化结果（含 status 字段）。

### ❌ 反模式 3：共享记忆无并发控制

```python
# 坏：多个 Worker 并行写同一个 shared_context，无锁
shared_context = {"results": []}

def worker_task(worker, subtask):
    result = worker.execute(subtask)
    shared_context["results"].append(result)  # ❌ 并发 append 可能丢数据！

# 并行执行
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as ex:
    futures = [ex.submit(worker_task, w, s) for w, s in assignments]
```

**后果**：Python 有 GIL 保护 `list.append`，但更复杂的操作（read-modify-write）会丢更新。TS/JS 完全没保护，并发写必然出问题。

**正确**：
- 优先用**消息传递**（每个 Worker 返回独立结果，Supervisor 收集）
- 如果非要用共享黑板，**必须加锁**（`threading.Lock` / `Mutex`）
- 或者用 Actor 模型（每个 Agent 有独立信箱，不共享可变状态）

### ❌ 反模式 4：Supervisor 自己也干活

```python
# 坏：Supervisor 既调度又执行
def supervisor(task):
    plan = decompose(task)
    for assignment in plan:
        if assignment.worker == "Researcher":
            result = do_research(assignment.subtask)  # Supervisor 自己干 Researcher 的活
        else:
            result = workers[assignment.worker].execute(assignment.subtask)
```

**后果**：Supervisor 既是调度者又是执行者，违背了"职责分离"。

**正确**：Supervisor **只做调度**，所有执行都交给 Worker。没有对应的 Worker，说明团队设计不全，应该**增加 Worker** 而不是让 Supervisor 代劳。

### ❌ 反模式 5：Handoff 无限链（踢皮球）

```python
# 坏：Agent A → B → C → A → B... 无限转交
customer_service → tech_support → billing → customer_service → ...
```

**后果**：用户被来回踢，永远得不到解决，体验极差。

**正确**：
- **限制 Handoff 次数**（如最多 2 次）
- **Handoff 必须有"终点 Agent"**——某个 Agent 被设计为"兜底"，它不能再 Handoff，必须给出答案
- 记录 Handoff 链，超过限制就升级人工

### ❌ 反模式 6：Worker 的 system prompt 重叠

```python
# 坏：两个 Worker 的职责边界模糊
writer = Worker(prompt="你是撰稿人，写文章")
editor = Worker(prompt="你是编辑，也能写文章")  # 跟 writer 重叠
# Supervisor 分派时不知道该给 writer 还是 editor
```

**后果**：Supervisor 的分派决策不稳定（两个都能干的活，每次可能派给不同人），结果不可预测。

**正确**：每个 Worker 的 system prompt 要**边界清晰、互斥**。Researcher 只检索、Writer 只写作、Coder 只写码——没有重叠。

## 常见错误

> 概念懂了，实际写代码还是会踩坑。这些是初学者最常犯的错误。

| 错误 | 症状 | 解决 |
|------|------|------|
| Supervisor 的 JSON 输出没做 Pydantic 校验 | 模型偶尔输出 `{"assignments": "Researcher"}` 而非列表，后续 `for` 循环报 `TypeError` | 用 `AssignmentPlan.model_validate_json()` 校验，校验失败就重试或降级 |
| Handoff 后没替换 system prompt | 技术 Agent 还以为自己是客服，回答风格和能力都不对 | Handoff 时保留 user/assistant 历史，但**替换** system prompt 为新角色 |
| Worker 返回空字符串 | Supervisor 收到 `""`，不知道是成功还是失败，汇总时丢信息 | Worker 返回结构化结果 `{status: "ok", result: "..."}` 或 `{status: "error", message: "..."}` |
| 两个 Worker 的 system prompt 职责重叠 | Supervisor 分派不稳定，同一任务每次可能派给不同人，结果不可预测 | 每个 Worker 的 prompt 边界清晰互斥，用"你只负责X，不负责Y"明确排除 |

---

## 7. 完整流程图

### Supervisor-Worker（写调研报告）

```
Phase 1 (Supervisor 分解): Supervisor → {"assignments": [
  {"worker": "Researcher", "subtask": "检索 AI Agent 的定义"},
  {"worker": "Researcher", "subtask": "检索 AI Agent 的应用场景"},
  {"worker": "Writer", "subtask": "把检索结果写成报告"},
]}

Phase 2 (Worker 执行):
  Researcher → search_wiki("AI Agent 定义") → "AI Agent 是..."
  Researcher → search_wiki("AI Agent 应用") → "应用于..."
  Writer → (综合研究结果写作) → "# AI Agent 调研报告\n..."

Phase 3 (Supervisor 汇总): 收集所有结果 → 最终输出
```

### Handoff（客服 → 技术专家）

```
Turn 1: 用户"我想退货" → 客服 Agent → "好的，请提供订单号..."

Turn 2: 用户"退货页面报 500，代码 ERR_DEPLOY_123"
  客服 Agent → 检测到技术关键词 → Handoff（带上对话历史）→ 技术 Agent

Turn 3: 技术 Agent（看到完整历史）→ "你遇到的是部署问题 ERR_DEPLOY_123，我来排查..."
```

---

## 8. 多 Agent vs 单 Agent 决策框架

| 任务特征 | 建议 | 示例 |
|----------|------|------|
| 单步任务 | 单次 LLM 调用 | "翻译这句话" |
| 2-3 步，同质任务 | 单 Agent + 工具 | "查北京天气并推荐穿搭" |
| 步骤明确，单一角色 | Plan-and-Execute | "写报告"（第08章） |
| 工具 >15 个 | **多 Agent**（工具分组） | 全栈助手 |
| 上下文 >50K token | **多 Agent**（上下文隔离） | 长文档处理 |
| 角色冲突（写+审） | **多 Agent**（角色分离） | 代码审查 |
| 需要专家路由（客服/技术） | **Handoff** | 智能客服 |
| 简单任务 | **别用多 Agent** | （反模式 1） |

---

## 运行示例

```bash
# Python
cd ai-agent/10-multi-agent
python3 python/main.py

# TypeScript
cd ai-agent/10-multi-agent
npx tsx typescript/main.ts
```

代码会先用真实 API 尝试（占位符密钥会失败），然后**自动降级为离线 mock 演示**，100% 可靠地展示：

1. **Supervisor-Worker**（`OUT:supervisor:` / `OUT:worker:{name}:`）：Supervisor 分派 → Worker 执行 → Supervisor 汇总
2. **Handoff**（`OUT:handoff:` / `OUT:resolve:`）：客服处理退货 → 检测技术关键词 → Handoff 给技术 Agent

---

## 兼容性注意

- **`.env` 是占位符密钥**（`OPENAI_API_KEY=sk-REPLACE-ME`）→ 真实 API 会 401，代码自动降级为离线 mock。
- **本章不引入任何多 Agent 框架**——每个 Worker 就是一次普通的 `chat.completions.create` 调用，让你看清底层。
- **离线 mock 设计**：预设分派决策序列（Supervisor-Worker）+ 预设客服→技术转交轨迹（Handoff），不依赖真实 API。

---

## 下一步

本章你让「任务助手 Agent」进化成了**团队**——Supervisor 调度、Worker 分工、还能 Handoff 给专家。

但你可能注意到：**每个 Agent 的上下文窗口是有限的**。随着任务变复杂、Agent 变多，"怎么管好每个 Agent 的上下文预算"成了新挑战。

第11章「上下文工程」会解决这个问题：如何压缩、分块、检索、遗忘——让多个 Agent 在有限的上下文窗口里高效协作。

> 💡 **多 Agent 是手段，不是目的**。从最简单的方案开始，多 Agent 只在单 Agent 明确"撑不住"时才用。一个设计良好的单 Agent，往往胜过一个设计糟糕的多 Agent 系统。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
