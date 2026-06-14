# 第12章 从零造框架 — 架构设计（6 大核心组件）

> **「任务助手 Agent」长出骨架了**——从第01–11章散落在各文件里的函数和字典，被抽象成
> **6 个有清晰边界的组件接口**。本章只画图纸（定义 Protocol/ABC/interface），第13章才浇筑混凝土（实现）。
> 学完本章，你打开 LangChain、OpenAI Agents SDK、Pydantic AI、Mastra、Vercel AI SDK 的源码，
> 会发现它们底层都在做同一件事：把这 6 块拼起来。

---

## 本章目标

学完本章，你将理解：

1. **为什么要"从零造框架"**：不是要取代 LangChain，而是要看透所有框架的共同骨架
2. **现代框架的本质是什么**：把"散装的 Agent 代码"重构成"组件化 + 接口契约"的架构
3. **6 大核心组件**各自干什么、不干什么、边界在哪里
4. **接口（Protocol/interface）先于实现**：先定义契约，再写实现，这是工程化的第一步
5. **反模式**：过早抽象、组件紧耦合、无清晰接口契约

> ⚠️ **本章只定义接口，不实现具体逻辑**。所有 `Protocol`/`interface` 里的方法体都是 `...`（Python）或
> 纯声明（TS）——这是"架构设计"的含义。第13章「实现核心」才会填上真正的逻辑。

---

## 先泼一盆冷水：你真的需要"造框架"吗？

在开始设计 6 大组件之前，必须诚实回答这个问题。

> **"从零造框架"的目标不是让你在生产环境用自己造的框架，而是让你看懂所有框架。**

很多团队掉进一个陷阱：读了 LangChain 的 `AgentExecutor` 源码觉得"我也能写"，然后花三个月造了个
内部框架，结果 API 比 LangChain 还难用，文档为零，最后大家都回去用 LangChain 了。

**造框架的正确理由（教学/理解）**：
- ✅ 看清 Agent 的本质结构（6 块积木怎么拼）
- ✅ 出 bug 时知道是哪一块的问题（而不是面对一个黑盒）
- ✅ 框架过时后能快速迁移（换工具实现，不动接口）

**造框架的错误理由（生产）**：
- ❌ "LangChain 太重了" → 改用 OpenAI Agents SDK（500 行代码搞定），不要造轮子
- ❌ "我想定制" → 先试试能不能用现有框架的钩子/插件机制
- ❌ "NIH 综合征"（Not Invented Here）→ 这是工程灾难的开始

> 💡 **Anthropic 的共识**：从最简单的方案开始。单次 LLM 调用能搞定就别加工具；加了工具能搞定就别加
> Agent 循环；Agent 循环能搞定就别造框架。**框架是最后的抽象，不是第一个抽象。**

---

## 为什么从零造框架（理解 LangChain/OpenAI Agents SDK 在做什么）

### 痛点：第01–11章的代码"散装"了

回顾一下你写过的代码：

| 章节 | 核心代码 | 散落在哪里 |
|------|----------|------------|
| 第03章 | `TOOL_FUNCTIONS` 字典 + `get_schema()` | 各自的 `main.py` |
| 第04章 | `agent_loop()` 循环 + `MAX_STEPS` | 每章重写一遍 |
| 第05章 | `ConversationBuffer` / `SummaryMemory` 类 | 独立文件 |
| 第06章 | `call_llm_with_retry()` + 错误分类 | 函数 |
| 第07章 | `parse_react_output()` 解析器 | 函数 |
| 第08章 | `plan_task()` / `synthesize()` | 函数 |
| 第10章 | `Worker` 类 + `WORKERS` 字典 | 类 |
| 第11章 | `ContextCompactor` / `TokenBudget` | 类 |

问题在于：**每章都把 LLM 调用、工具执行、消息管理、循环控制混在一起**。比如 `agent_loop()` 里
既有调 LLM（第01章职责）、又有执行工具（第03章职责）、又有管消息（第05章职责）、又有计数
步数（循环控制职责）。当你想把"记忆"从 `ConversationBuffer` 换成 `VectorMemory` 时，得改
`agent_loop` 的内部逻辑——这就是**紧耦合**。

### 框架的本质：拆分 + 契约

**框架 = 把散装代码拆成独立组件 + 用接口约束组件之间的交互。**

拆分后，每个组件只干一件事，组件之间通过**接口**（而不是具体实现）通信：

```
# 散装代码（第01-11章）：
def agent_loop(task):
    messages = [...]                              # 记忆
    for step in range(MAX_STEPS):                 # 循环控制
        resp = client.chat.completions.create(...) # LLM 调用 + 重试
        tool_calls = resp.choices[0].message.tool_calls  # 解析
        if not tool_calls: return ...             # 终止判断
        for tc in tool_calls:
            result = TOOL_FUNCTIONS[tc.name](...) # 工具执行
            messages.append({...})                # 记忆
    # 没有日志、没有 trace、没有成本追踪（可观测性为零）

# 框架化后（本章定义接口，第13章实现）：
def agent_loop(runner):  # runner 持有所有组件的引用
    runner.memory.add("user", task)
    for step in range(runner.max_steps):
        runner.observer.on_step_start(step, runner.memory.get_messages())
        resp = runner.llm.chat_with_retry(runner.memory.get_messages(), tools=runner.tools.get_schema())
        runner.observer.on_llm_call(runner.memory.get_messages())
        if not runner.parser.has_tool_calls(resp):
            return runner.parser.parse_final_answer(resp)
        for call in runner.parser.parse_tool_calls(resp):
            runner.observer.on_tool_call(call.name, call.args)
            result = runner.tools.execute(call.name, call.args)
            runner.observer.on_tool_result(call.name, result)
            runner.memory.add("tool", result)
        runner.observer.on_step_end(step)
```

看出区别了吗？**框架化的 `agent_loop` 里不再有任何"具体实现"**——它只调用组件的接口方法。
换记忆、换 LLM、换工具、加日志，全都不用动 `agent_loop` 本身。这就是**控制反转**（IoC）。

### 现代框架的共同骨架

无论 LangChain、OpenAI Agents SDK、Pydantic AI、Mastra、Vercel AI SDK，底层都逃不出这 6 块：

| 我们的组件 | LangChain 对应 | OpenAI Agents SDK 对应 | Pydantic AI 对应 |
|-----------|----------------|----------------------|-----------------|
| AgentRunner | `AgentExecutor` | `Runner` | `Agent.run()` |
| ToolRegistry | `Tool` + `BaseToolkit` | `@tool` + `tools=[...]` | `@agent.tool` |
| LLMClient | `BaseChatModel` | `set_default_openai_client` | `model` 参数 |
| Memory | `BaseMemory` | `RunResultWrapper` | `message_history` |
| ActionParser | `OutputParser` | `tool_calls` 字段 | `result.data` |
| Observer | `Callbacks` | `AgentHook` / tracing | `InstrumentationSettings` |

名字不同，职责完全一样。**学会了这 6 块，你就能在一小时内读懂任何框架的架构图。**

---

## 架构总览（ASCII 图）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AgentRunner（循环引擎）                        │
│                                                                     │
│   ┌──────────┐  task  ┌─────────────────────────────────────────┐  │
│   │  Memory  │◄──────►│  for step in range(max_steps):          │  │
│   │  (消息)  │        │    observer.on_step_start(step)         │  │
│   └────┬─────┘        │    resp = llm.chat_with_retry(...)      │  │
│        │              │    if not parser.has_tool_calls(resp):  │  │
│        │ messages     │        return final_answer             │  │
│        ▼              │    for call in parser.parse(resp):     │  │
│   ┌────────────┐      │        result = tools.execute(call)    │  │
│   │ LLMClient  │◄─────┤        memory.add("tool", result)      │  │
│   │ (调模型)   │      │    observer.on_step_end(step)          │  │
│   └────┬───────┘      └──────────────────┬──────────────────────┘  │
│        │ response                        │ coordinates             │
│        ▼                                 ▼                         │
│   ┌──────────────┐              ┌─────────────────┐               │
│   │ActionParser  │──actions────►│  ToolRegistry   │               │
│   │(解析 tool_   │              │  (工具注册表)    │               │
│   │ calls)       │              │ register/get_   │               │
│   └──────────────┘              │ schema/execute  │               │
│                                 └─────────────────┘               │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              Observer（可观测钩子，横切所有组件）           │     │
│   │  on_step_start / on_llm_call / on_tool_call /            │     │
│   │  on_tool_result / on_step_end                            │     │
│   └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

**数据流**：用户 task → Memory 存储 → LLMClient 读取并调用模型 → ActionParser 解析响应 →
ToolRegistry 执行工具 → 结果回到 Memory → 循环。Observer 在每个关键点挂钩，做日志/trace/成本追踪。

**关键设计原则**：
1. **组件间只通过接口通信**，不直接访问彼此的内部状态
2. **AgentRunner 是协调者**（orchestrator），不包含业务逻辑
3. **Observer 是横切关注点**（cross-cutting concern），用钩子模式注入，不侵入主流程
4. **单向依赖**：AgentRunner 依赖其他 5 个组件，其他组件互不依赖（除了 Observer 被所有人调用）

---

## 6 大核心组件详解

### 组件 1：AgentRunner — Agent 循环引擎

**职责**：驱动 observe→reason→act 循环，持有所有组件的引用，是整个框架的"心脏"。

**它干什么**：
- 接收用户 task，初始化 Memory
- 循环 `max_steps` 次，每次：调 LLM → 解析响应 → 执行工具 → 更新 Memory
- 触发 Observer 的钩子（每步开始/结束、LLM 调用、工具调用）
- 两个终止条件：模型不再调用工具（任务完成）/ 达到 max_steps（保险丝）

**它不干什么**：
- ❌ 不直接调 `client.chat.completions.create()`（交给 LLMClient）
- ❌ 不直接执行工具函数（交给 ToolRegistry）
- ❌ 不解析 JSON/正则（交给 ActionParser）
- ❌ 不做日志/trace（交给 Observer）

**接口定义**：

```python
class AgentRunner(Protocol):
    def run(self, task: str, max_steps: int = 10) -> str: ...
```

看起来只有一个方法？没错——`run()` 是整个框架的入口。但 `AgentRunner` 的**实现类**会持有其他
5 个组件的引用（`self.llm`、`self.tools`、`self.memory`、`self.parser`、`self.observer`），
在 `run()` 内部协调它们。

> 💡 **max_steps 保险丝**（第04章学过）：必填参数，默认 10。**永远不要设成无限**——否则模型陷入
> "调工具→不满足→再调工具"的死循环，烧光预算。框架级别必须有这个保护。

---

### 组件 2：ToolRegistry — 工具注册表

**职责**：管理所有工具的元数据（名称/描述/参数 schema）和处理器（handler），对外提供统一的注册、查询、执行接口。

**它干什么**：
- `register(name, description, parameters, handler)`：注册一个工具
- `get_schema()`：返回 OpenAI `tools` 格式的 JSON Schema 列表（给 LLM 看）
- `execute(name, args)`：按名字查找 handler 并执行，返回字符串结果

**它不干什么**：
- ❌ 不决定"什么时候调哪个工具"（那是 LLM 的决策）
- ❌ 不解析 LLM 的响应（那是 ActionParser 的活）
- ❌ 不记录调用日志（那是 Observer 的活）

**接口定义**：

```python
class ToolRegistry(Protocol):
    def register(self, name: str, description: str,
                 parameters: dict, handler: Callable[..., str]) -> None: ...
    def get_schema(self) -> list[dict]: ...   # OpenAI tools 格式
    def execute(self, name: str, args: dict) -> str: ...
```

**自描述工具**（self-describing）是核心设计：每个工具自带名称、描述、参数 schema——LLM 通过
`get_schema()` 返回的 JSON 知道有哪些工具可用、每个工具接受什么参数。这正是第03章 function calling
的本质。

> 💡 **`get_schema()` 返回 OpenAI tools 格式**：`[{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]`。
> 这样 ToolRegistry 的输出可以直接塞进 `client.chat.completions.create(tools=...)`。

---

### 组件 3：LLMClient — LLM 包装器

**职责**：封装所有与 LLM API 的交互——调用、重试、流式、结构化输出，让上层不关心"用的是哪家 API"。

**它干什么**：
- `chat(messages, tools)`：发送消息列表，返回响应（dict，包含 content 和 tool_calls）
- `chat_with_retry(messages, max_retries)`：带指数退避的重试封装（第06章学过）
- （可选扩展）`stream()`：流式输出
- （可选扩展）`structured_output(schema)`：结构化输出（第02/08章学过）

**它不干什么**：
- ❌ 不管理对话历史（那是 Memory 的活）
- ❌ 不解析 tool_calls（那是 ActionParser 的活）
- ❌ 不记录成本（那是 Observer 的活）

**接口定义**：

```python
class LLMClient(Protocol):
    def chat(self, messages: list[dict],
             tools: list[dict] | None = None, **kwargs) -> dict: ...
    def chat_with_retry(self, messages: list[dict],
                        max_retries: int = 3, **kwargs) -> dict: ...
```

**为什么需要包装器**：
1. **统一接口**：换提供商（OpenAI→DeepSeek→Ollama）只换 LLMClient 实现，上层无感
2. **重试逻辑集中**：第06章的 `call_llm_with_retry()` 不用在每个 Agent 里重写
3. **mock 友好**：测试时用 `MockLLMClient` 替换，不依赖真实 API（第04/06/07章的离线 mock 模式）

---

### 组件 4：Memory — 记忆管理

**职责**：管理对话历史，决定"模型能看到什么"。这是上下文工程（第11章）的核心载体。

**它干什么**：
- `add(role, content)`：追加一条消息
- `get_messages()`：返回当前消息列表（给 LLMClient 用）
- `clear()`：清空（开始新对话）
- （可选扩展）自动压缩、token 预算控制（第05/11章的 SummaryMemory/TokenBudget）

**它不干什么**：
- ❌ 不调 LLM（即使 SummaryMemory 内部调 LLM 做摘要，也是注入 LLMClient，不自己持有 client）
- ❌ 不执行工具
- ❌ 不决定"什么时候停止"

**接口定义**：

```python
class Memory(Protocol):
    def add(self, role: str, content: str) -> None: ...
    def get_messages(self) -> list[dict]: ...
    def clear(self) -> None: ...
```

> 💡 **Memory 是策略接口**：第05章的 `ConversationBuffer`、`SummaryMemory`、`VectorMemory` 都可以
> 实现这个接口。框架上层（AgentRunner）不关心你用哪种记忆，只要能 `add`/`get_messages`/`clear`。
> 这就是**多态**的威力——换实现不改调用方。

---

### 组件 5：ActionParser — 输出解析器

**职责**：把 LLM 的原始响应（dict/JSON）解析成结构化的"动作"（tool calls 或 final answer）。

**它干什么**：
- `parse_tool_calls(response)`：从响应中提取工具调用列表 `[{name, args, id}]`
- `has_tool_calls(response)`：判断响应是否包含工具调用（决定循环是否继续）

**它不干什么**：
- ❌ 不执行工具（交给 ToolRegistry）
- ❌ 不调 LLM
- ❌ 不做格式验证/重试（交给上层，如第06章的自我纠正）

**接口定义**：

```python
class ActionParser(Protocol):
    def parse_tool_calls(self, response: dict) -> list[dict]: ...  # [{name, args, id}]
    def has_tool_calls(self, response: dict) -> bool: ...
```

**为什么单独抽出来**：
1. **响应格式多样**：OpenAI tools API 返回 `tool_calls` 字段；显式 ReAct（第07章）返回纯文本要正则解析；
   某些模型返回自定义 JSON。ActionParser 屏蔽这些差异。
2. **解析是脆弱环节**：第07章的格式错误、第06章的幻觉工具名，都需要在解析层处理。集中到一个组件，
   比散落在循环各处好维护。

---

### 组件 6：Observer — 可观测钩子

**职责**：在不侵入主流程的前提下，记录每一步的状态——日志、trace、成本追踪、性能指标。

**它干什么**：
- `on_step_start(step, messages)`：每步开始（可记录 step 编号、当前消息数）
- `on_llm_call(messages)`：调 LLM 前（可记录 token 数、估算成本）
- `on_tool_call(name, args)`：调工具前（可记录工具名、参数）
- `on_tool_result(name, result)`：工具返回后（可记录结果、耗时）
- `on_step_end(step)`：每步结束（可记录总耗时）

**它不干什么**：
- ❌ **不修改主流程的状态**（只读不写，纯旁路观察）
- ❌ 不调 LLM / 不执行工具
- ❌ 不决定循环是否继续

**接口定义**：

```python
class Observer(Protocol):
    def on_step_start(self, step: int, messages: list[dict]) -> None: ...
    def on_llm_call(self, messages: list[dict]) -> None: ...
    def on_tool_call(self, name: str, args: dict) -> None: ...
    def on_tool_result(self, name: str, result: str) -> None: ...
    def on_step_end(self, step: int) -> None: ...
```

> 💡 **Observer 是横切关注点**（cross-cutting concern）：日志、监控、trace 这些需求"横跨"所有组件，
> 如果不抽出来，你的 `agent_loop` 会被 `print()` 淹没。Observer 用**钩子模式**解决——主流程在
> 关键点"通知"Observer，Observer 自己决定怎么处理（打印/写文件/发 Datadog）。这就是
> **观察者模式**（Observer Pattern），也是 OpenTelemetry 的设计基础。
>
> 第15章「可观测调试」会实现一个真正的 `TracingObserver`，第16章会加 `GuardrailObserver`。

---

## 接口定义概览（一图速查）

```
ToolRegistry       LLMClient           Memory
─────────────      ───────────         ──────
register()         chat()              add()
get_schema()       chat_with_retry()   get_messages()
execute()                              clear()

ActionParser       Observer            AgentRunner
─────────────      ────────            ───────────
parse_tool_calls() on_step_start()     run(task, max_steps)
has_tool_calls()   on_llm_call()
                   on_tool_call()
                   on_tool_result()
                   on_step_end()
```

**Python 用 `Protocol`（结构性子类型）**：实现类不需要显式 `inherit`，只要方法签名匹配就算
"实现了接口"。`@runtime_checkable` 让你可以用 `isinstance()` 检查。

**TypeScript 用 `interface`（声明性约束）**：实现类需要 `class Foo implements Bar`，编译期检查。
运行时不保留接口信息（与 Python 的 Protocol 不同），所以 TS 没有对等的 `isinstance` 检查。

---

## 为什么要"接口先行"（Contract-First Design）

### 好处 1：并行开发

定义好接口后，**6 个组件可以由 6 个人并行实现**。只要大家都遵守接口契约，最后拼起来就能跑。
没有接口的话，得等一个人写完 Memory，另一个人才能写依赖 Memory 的 AgentRunner——串行，慢。

### 好处 2：可替换性

想让 Agent 支持流式输出？写个 `StreamingLLMClient` 实现 `LLMClient` 接口，换进去就行。
想从 `ConversationBuffer` 升级到 `SummaryMemory`？只要都实现 `Memory` 接口，一行代码替换。
**接口是"可替换性"的基石**。

### 好处 3：可测试性

测试 `AgentRunner` 时，注入 `MockLLMClient`（返回预设响应）+ `MockToolRegistry`，
不依赖真实 API。第15章「评估测试」会大量用到这个。没有接口，你没法 mock。

### 好处 4：文档化契约

接口本身就是文档。看到 `ToolRegistry` 有 `register/get_schema/execute` 三个方法，你就知道
这个组件干什么——不需要读实现代码。**接口 = 契约 = 文档**。

---

## 反模式（什么不该做）

### ❌ 过早抽象（Premature Abstraction）

```python
# 坏：只写过 1 个 Agent，就抽象出 6 个组件 + 4 层继承 + 3 种工厂模式
class BaseAgentRunner(AbstractAgentCore, MixinTimeout, MixinRetry): ...

# 正确：先写 3-5 个具体 Agent（第04-11章），发现重复模式后再抽象
# 本章的 6 个组件，是从第01-11章的真实代码里提炼出来的，不是拍脑袋设计的
```

**后果**：抽象脱离实际，接口设计错误，后面所有实现都在"削足适履"。

**原则**：**Rule of Three**——同一个模式出现 3 次以上才抽象。我们的 6 个组件都在第03-11章
出现过 3 次以上（agent loop 写了 8 次、工具注册写了 6 次、记忆管理写了 4 次），才抽象成接口。

### ❌ 组件间紧耦合（Tight Coupling）

```python
# 坏：AgentRunner 直接 import 具体实现，无法替换
class AgentRunner:
    def __init__(self):
        self.llm = OpenAIClient()                    # ❌ 写死 OpenAI
        self.memory = ConversationBuffer()           # ❌ 写死 Buffer
        self.tools = InMemoryToolRegistry()          # ❌ 写死内存版

# 正确：通过构造函数注入（依赖注入），接收接口类型
class AgentRunner:
    def __init__(self, llm: LLMClient, memory: Memory,
                 tools: ToolRegistry, parser: ActionParser,
                 observer: Observer): ...
```

**后果**：换任何一块都得改 AgentRunner 的源码，丧失"可替换性"——框架白造了。

**原则**：**依赖注入**（Dependency Injection）。组件通过构造函数接收依赖，类型声明为接口
（Protocol/interface），不是具体实现。

### ❌ 无清晰接口契约（Implicit Contract）

```python
# 坏：没有接口，组件间靠"默契"通信
class MyAgent:
    def run(self, task):
        resp = self.llm.chat(...)  # chat 返回什么？dict？object？不知道
        result = resp["choices"][0]["message"]["content"]  # 脆弱，依赖具体结构

# 正确：接口明确约定返回类型
class LLMClient(Protocol):
    def chat(self, messages: list[dict], ...) -> dict: ...  # 契约：返回 dict
```

**后果**：换一个 LLMClient 实现，返回类型变了（比如从 dict 变成 object），所有调用方全崩。

**原则**：**接口 = 显式契约**。返回类型、参数类型、异常都必须在接口里声明，实现类必须遵守。
Python 用 type hints + Protocol，TS 用 interface + strict mode。

### ❌ Observer 带副作用（违反纯观察原则）

```python
# 坏：Observer 修改了主流程状态
class BadObserver(Observer):
    def on_tool_result(self, name, result):
        if "error" in result:
            self.runner.memory.add("system", "工具失败了，请重试")  # ❌ 改了 Memory！

# 正确：Observer 只读不写，通过抛异常或返回信号让主流程决策
class GoodObserver(Observer):
    def on_tool_result(self, name, result):
        if "error" in result:
            logger.warning(f"工具 {name} 失败: {result}")  # ✅ 只记录
```

**后果**：Observer 变成"隐藏的第二个主流程"，调试时根本不知道状态被谁改了。

**原则**：**Observer 是纯旁路**。只读不写，只记录不决策。如果需要干预主流程（如护栏），
那是第16章 `GuardrailObserver` 的职责，而且要通过显式的"中断信号"机制，不是偷偷改状态。

---

## 本章代码说明

本章代码**只定义接口，不实现**：

- **Python `main.py`**：用 `@runtime_checkable Protocol` 定义 6 个组件接口，演示脚本打印
  每个接口的方法签名 + ASCII 架构图 + `isinstance` 验证（证明 Protocol 可运行时检查）。
- **TypeScript `main.ts`**：用 `interface` 定义对等的 6 个组件契约，演示脚本打印职责和方法签名。

两个文件都**不调用真实 API**——纯接口定义 + 自省演示。第13章才会填上具体实现。

### 运行示例

```bash
# Python
cd ai-agent/12-framework-design
python3 python/main.py

# TypeScript
cd ai-agent/12-framework-design
npx tsx typescript/main.ts
```

输出标记：
- `OUT:component:{name}:` — 每个组件的职责和方法签名
- `OUT:architecture:` — ASCII 架构图
- `OUT:verify:` — 接口验证（Python `isinstance`，TS 类型检查说明）

---

## 下一步

本章你定义了 6 个组件的**接口契约**——画好了框架的图纸。但图纸不能跑。

第13章「实现核心」会**浇筑混凝土**：给每个接口写真正的实现类——
`OpenAILLMClient`（包装 OpenAI SDK）、`InMemoryToolRegistry`（字典存储）、
`ConversationBuffer`（列表存储）、`OpenAIToolCallParser`（解析 tool_calls）、
`LoggingObserver`（打印日志）、`DefaultAgentRunner`（组装所有组件的循环引擎）。

最后你会得到一个**能真正跑起来的 mini Agent 框架**，而「任务助手 Agent」会从散装代码
被重构成这个框架的一个实例——Part 5 的"骨架"就此成型。

> 💡 **接口先行是工程化的第一步**。在真实项目里，先定义接口（架构评审），再并行实现，
> 是团队协作的标准流程。本章教你的是这种"架构思维"，不只是 6 个组件本身。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
