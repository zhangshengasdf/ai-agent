# 第12章 练习题 — 从零造框架（架构设计）

> 练习目标：加深对 6 大核心组件接口的理解，亲手画出自己版本的架构图。

---

## 练习 1：画出你理解的框架架构图（文字描述）

### 题目

本章给出了 6 大核心组件（AgentRunner / ToolRegistry / LLMClient / Memory / ActionParser / Observer）
和它们的关系图。现在请你**用自己的理解**重新画一张架构图（纯文字/ASCII 描述即可），并满足：

1. 标注每个组件的**输入**和**输出**（数据流方向）
2. 标注组件之间的**调用关系**（谁调用谁）
3. 说明 Observer 为什么是**横切关注点**（用箭头或注释表达）
4. 标出**两个终止条件**发生在哪里

### 参考答案

```
                    ┌─────────────────────────────────────────────┐
                    │            用户 task (字符串)                 │
                    └────────────────────┬────────────────────────┘
                                         │ 输入
                                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        AgentRunner.run(task)                         │
│                                                                      │
│  [终止条件 1] if (!parser.hasToolCalls(resp)) → return answer       │
│  [终止条件 2] if (step >= maxSteps) → return "达到最大步数"          │
│                                                                      │
│  ┌─────────┐  输出: messages[]   ┌──────────────┐                    │
│  │ Memory  │◄───────────────────│  (循环体)     │                    │
│  │         │───────────────────►│              │                    │
│  └─────────┘  输入: add(role)   └──────┬───────┘                    │
│                                     │ 调用                          │
│              ┌──────────────────────┼──────────────────┐            │
│              ▼                      ▼                  ▼            │
│      ┌────────────┐         ┌──────────────┐   ┌──────────────┐     │
│      │ LLMClient  │         │ ActionParser │   │ ToolRegistry │     │
│      │            │         │              │   │              │     │
│      │ 输入:      │         │ 输入:        │   │ 输入:        │     │
│      │  messages  │         │  response    │   │  name, args  │     │
│      │  + tools   │         │              │   │              │     │
│      │            │         │ 输出:        │   │ 输出:        │     │
│      │ 输出:      │         │  ToolCall[]  │   │  result(str) │     │
│      │  response  │────────►│              │──►│              │     │
│      └────────────┘         └──────────────┘   └──────────────┘     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │   Observer（横切：被上面所有组件调用，自己不调用别人）          │   │
│  │                                                              │   │
│  │   onStepStart ←── AgentRunner 每步开始时调用                 │   │
│  │   onLLMCall   ←── AgentRunner 调 LLMClient 前调用            │   │
│  │   onToolCall  ←── AgentRunner 调 ToolRegistry 前调用         │   │
│  │   onToolResult ←─ AgentRunner 工具返回后调用                 │   │
│  │   onStepEnd   ←── AgentRunner 每步结束时调用                 │   │
│  │                                                              │   │
│  │   Observer 只接收数据（只读），不返回数据，不修改主流程状态    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**关键说明**：

- **数据流方向**：task → Memory（存） → LLMClient（读 messages，返 response） →
  ActionParser（解析 response） → ToolRegistry（执行工具） → Memory（存结果） → 循环
- **调用关系**：AgentRunner 是唯一的协调者，它持有其他 5 个组件的引用并调用它们。
  其他组件之间**不直接调用**（ToolRegistry 不调 LLMClient，Memory 不调 ActionParser）。
- **Observer 是横切关注点**：所有组件（主要是 AgentRunner）在关键点"通知"Observer，
  但 Observer **不反向调用**任何组件——它是纯被动的接收者。这就是"不侵入主流程"的含义。
- **两个终止条件**：都在 AgentRunner 的循环体内——(1) `hasToolCalls` 返回 false（模型自主完成），
  (2) `step >= max_steps`（保险丝触发）。

---

## 练习 2：识别违反接口契约的反模式

### 题目

下面是一段"坏"的框架设计代码。请指出它违反了本章哪些设计原则（至少 3 处），并说明后果：

```python
class MyAgentRunner:
    def __init__(self):
        # 直接 new 具体实现
        self.llm = OpenAIClient(api_key="sk-xxx")           # ①
        self.memory = ConversationBuffer()                   # ②
        self.tools = {"get_weather": get_weather_handler}   # ③ 裸字典

    def run(self, task):
        messages = [{"role": "user", "content": task}]
        while True:                                          # ④ 无 max_steps
            resp = self.llm.chat(messages)                   # ⑤ 无重试
            # 直接解析 response（没有 ActionParser）
            if not resp["choices"][0]["message"]["tool_calls"]:
                return resp["choices"][0]["message"]["content"]
            for tc in resp["choices"][0]["message"]["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                result = self.tools[name](**args)            # ⑥ 无 Observer 记录
                messages.append({"role": "tool", "content": result})
        # 没有 Observer，没有日志，没有成本追踪
```

### 参考答案

这段代码违反了至少 **6 处**设计原则：

| # | 违反点 | 违反的原则 | 后果 |
|---|--------|-----------|------|
| ① | `OpenAIClient(api_key="sk-xxx")` 硬编码在构造函数 | **依赖注入** + **LLMClient 接口** | 想换 DeepSeek 得改源码；密钥泄露到代码 |
| ② | `ConversationBuffer()` 写死具体实现 | **依赖注入** + **Memory 接口** | 想升级 SummaryMemory 得改源码 |
| ③ | `self.tools = {...}` 裸字典，无 `get_schema()` | **ToolRegistry 接口** | 无法把工具 schema 传给 LLM；无统一 execute |
| ④ | `while True` 无 max_steps | **AgentRunner 接口**（max_steps 保险丝） | 无限循环烧光预算（第04章反模式 #1） |
| ⑤ | `self.llm.chat(messages)` 无重试 | **LLMClient.chat_with_retry** | 一次网络抖动就崩溃（第06章） |
| ⑥ | 直接 `resp["choices"][0]...` 硬编码解析 | **ActionParser 接口** | 换个模型/API 格式就得改循环逻辑 |

**额外问题**：
- **无 Observer**：没有日志、trace、成本追踪——出 bug 时完全是黑盒
- **无 Memory 接口隔离**：`messages` 是局部变量，Memory 组件形同虚设
- **密钥硬编码**：`api_key="sk-xxx"` 违反 shared/config 的安全约定

**正确写法**（对照本章接口）：

```python
class DefaultAgentRunner:  # 第13章会实现
    def __init__(self, llm: LLMClient, memory: Memory,
                 tools: ToolRegistry, parser: ActionParser,
                 observer: Observer):  # 依赖注入，接收接口类型
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.parser = parser
        self.observer = observer

    def run(self, task: str, max_steps: int = 10) -> str:  # max_steps 保险丝
        self.memory.add("user", task)
        for step in range(max_steps):
            self.observer.on_step_start(step, self.memory.get_messages())
            resp = self.llm.chat_with_retry(  # 带重试
                self.memory.get_messages(), tools=self.tools.get_schema()
            )
            if not self.parser.has_tool_calls(resp):  # 终止条件 1
                return self.parser.parse_final_answer(resp)
            for call in self.parser.parse_tool_calls(resp):  # ActionParser 解析
                self.observer.on_tool_call(call["name"], call["args"])
                result = self.tools.execute(call["name"], call["args"])
                self.observer.on_tool_result(call["name"], result)
                self.memory.add("tool", result)
            self.observer.on_step_end(step)
        return "达到最大步数，强制停止"  # 终止条件 2
```

---

## 练习 3（拓展）：为 6 大组件各想一个"扩展实现"

### 题目

接口的价值在于**可替换性**。本章的 6 个接口每个都可以有多种实现。请为每个组件想出至少 2 种
不同的实现（本章的接口定义 + 你对前面章节的理解），并说明各自的适用场景。

### 参考答案

| 接口 | 实现 1 | 实现 2 | 实现 3（拓展） |
|------|--------|--------|---------------|
| **LLMClient** | `OpenAILLMClient`（包装 OpenAI SDK，生产用） | `MockLLMClient`（返回预设响应，测试用，第04/06/07章离线 mock） | `StreamingLLMClient`（流式输出，聊天 UI 用） |
| **ToolRegistry** | `InMemoryToolRegistry`（字典存储，单进程） | `RemoteToolRegistry`（从配置文件/API 加载工具，微服务） | `MCPToolRegistry`（桥接 Model Context Protocol 工具） |
| **Memory** | `ConversationBuffer`（全量消息，第05章） | `SummaryMemory`（超阈值摘要，第05章） | `VectorMemory`（语义检索，第05/09章 RAG） |
| **ActionParser** | `OpenAIToolCallParser`（解析 tool_calls 字段） | `ReActTextParser`（正则解析 Thought/Action 文本，第07章） | `CustomJSONParser`（解析自定义 JSON 格式） |
| **Observer** | `LoggingObserver`（console.log，开发用） | `TracingObserver`（写 JSONL trace 文件，第15章） | `GuardrailObserver`（安全护栏，第16章） |
| **AgentRunner** | `DefaultAgentRunner`（顺序循环，第13章） | `ReActAgentRunner`（带显式 Thought，第07章风格） | `PlanExecuteRunner`（先规划后执行，第08章风格） |

**核心洞察**：同一个 `AgentRunner.run()` 接口，底层可以是 ReAct、Plan-Execute、Reflection
等不同模式——只要都遵守 `run(task, max_steps) -> str` 契约，上层调用方无感。这就是
**接口作为抽象屏障**的威力。

---

## 运行本章代码

```bash
# Python（打印 6 大组件接口 + 架构图 + isinstance 验证）
cd ai-agent/12-framework-design
python3 python/main.py

# TypeScript（对等实现，interface 编译期约束演示）
cd ai-agent/12-framework-design
npx tsx typescript/main.ts
```
