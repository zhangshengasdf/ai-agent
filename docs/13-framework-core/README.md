# 第13章 从零造框架 — 实现核心（6 大组件落地）

> **「任务助手 Agent」的骨架有了血肉**——第12章画好的 6 块接口图纸，
> 本章浇筑成能真正跑起来的 mini Agent 框架。你会亲手实现 6 个组件类，
> 用依赖注入组装它们，然后运行一个"查天气 + 算温差"的多步 Agent。
> **没有 LangChain，没有 AutoGen，只有你自己的代码**——学完后，任何框架的源码
> 你都能在一小时内看懂。

---

## TL;DR

> **30 秒速读**：把第12章的 6 个接口 Protocol/interface 逐一写成可运行的实现类，用依赖注入组装成一个完整的 mini Agent 框架，跑通"查天气 + 算温差"的多步循环。
> 
> **如果只记一件事**：依赖注入（构造函数接收接口类型）是所有现代框架的核心，掌握它就能在一小时内读懂任何框架源码。

---

## 本章目标

学完本章，你将能够：

1. **实现 6 大组件**：给第12章的每个 Protocol/interface 写具体的实现类
2. **掌握依赖注入**：理解"通过构造函数注入组件"为什么比"内部 new"更灵活
3. **理解归一化设计**：LLMClient 把 SDK 差异屏蔽掉，上层只看到统一格式
4. **写出离线降级**：API 不可用时自动切换 mock，保证演示 100% 可靠
5. **组装完整 Agent**：把 6 个组件像乐高积木一样拼成一个能干活的 Agent

> ⚠️ **前置条件**：先学第12章（接口定义）。本章的实现类全部对应第12章的 Protocol/interface，
> 不读第12章你会不知道每个方法在干什么。

---

## 从接口到实现：第12章 → 第13章

第12章定义了 6 个接口（只声明方法签名，方法体是 `...`）。本章给每个接口写**真正的实现类**：

| 第12章接口（Protocol/interface） | 第13章实现类 | 干什么 |
|------|------|------|
| `ToolRegistry` | `InMemoryToolRegistry` | 字典存储工具，register/get_schema/execute |
| `LLMClient` | `DefaultLLMClient` | 包装 OpenAI SDK + 退避重试 + 离线 mock |
| `Memory` | `ConversationMemory` | 列表存储消息，add/get_messages/clear |
| `ActionParser` | `OpenAIToolCallParser` | 解析 tool_calls → [{name, args, id}] |
| `Observer` | `LoggingObserver` | 打印每步日志（OUT:framework:step{N}:） |
| `AgentRunner` | `DefaultAgentRunner` | 循环引擎，协调其他 5 个组件 |

**文件结构**：

```
13-framework-core/
├── python/
│   ├── framework/
│   │   └── __init__.py    ← 6 大组件实现（~300 行）
│   ├── main.py            ← 组装 + 运行 demo
│   └── requirements.txt
├── typescript/
│   ├── framework/
│   │   └── index.ts       ← 6 大组件实现（~300 行）
│   └── main.ts            ← 组装 + 运行 demo
├── exercises/
│   └── README.md
└── README.md              ← 你在这里
```

---

## 核心设计：依赖注入（Dependency Injection）

本章最重要的工程模式不是某个组件的实现，而是**怎么把组件拼起来**。

### 反模式：内部 new（紧耦合）

```python
# ❌ 坏：AgentRunner 内部 new 具体实现，换任何一块都得改源码
class DefaultAgentRunner:
    def __init__(self):
        self.llm = DefaultLLMClient(get_config())         # 写死
        self.memory = ConversationMemory("...")           # 写死
        self.tools = InMemoryToolRegistry()               # 写死
        # ...
```

问题：想从 `ConversationMemory` 换成 `SummaryMemory`？得改 `AgentRunner` 的源码。
想测试时注入 `MockLLMClient`？做不到。框架白造了。

### 正确模式：构造函数注入

```python
# ✅ 正确：通过构造函数接收组件（依赖注入）
class DefaultAgentRunner:
    def __init__(self, llm, tools, memory, parser, observer):
        self.llm = llm          # 接口类型，不是具体实现
        self.tools = tools
        self.memory = memory
        self.parser = parser
        self.observer = observer
```

组装时在 `main.py` 里决定用哪些实现：

```python
def build_agent():
    tools = InMemoryToolRegistry()
    tools.register("get_weather", ..., handler=get_weather)

    llm = DefaultLLMClient(get_config())
    memory = ConversationMemory(system_prompt="你是任务助手...")
    parser = OpenAIToolCallParser()
    observer = LoggingObserver()

    return DefaultAgentRunner(llm, tools, memory, parser, observer)  # 注入
```

**好处**：
- 换实现只改 `build_agent()`，不动组件源码
- 测试时注入 mock 组件（`MockLLMClient` 返回预设响应）
- 每个组件可以独立开发、独立测试

> 💡 **依赖注入是所有现代框架的核心**。Spring（Java）、FastAPI（Python）、NestJS（TypeScript）
> 都在做同一件事：帮你把组件"注入"到需要它们的地方。本章我们手动注入，感受原理。

---

## 6 大组件实现详解

### 组件 1：InMemoryToolRegistry

用字典（Python `dict` / TS `Map`）存储工具。每个工具是一条记录：
`{name → {description, parameters, handler}}`。

```python
class InMemoryToolRegistry:
    def __init__(self):
        self._tools = {}          # name → tool info

    def register(self, name, description, parameters, handler):
        self._tools[name] = {"description": ..., "parameters": ..., "handler": handler}

    def get_schema(self):
        # 转成 OpenAI tools 格式：[{"type": "function", "function": {...}}]
        return [{"type": "function", "function": {"name": n, ...}} for n, info in self._tools.items()]

    def execute(self, name, args):
        if name not in self._tools:
            return f"[错误] 工具 '{name}' 不存在。可用: ..."
        return str(self._tools[name]["handler"](**args))
```

**关键设计**：
- `get_schema()` 的输出可以直接塞进 `client.chat.completions.create(tools=...)`
- `execute()` 对未知工具名返回错误消息（不抛异常），让 Agent 能自我纠正（第06章机制 3）
- 工具执行异常也返回错误消息（第06章机制 2），不崩溃

**TypeScript 特殊处理**：
handler 的类型是 `(...args: string[]) => string`（rest params），但 `execute` 收到的 `args`
是 `Record<string, unknown>`。需要用 `Object.values(args).map(String)` 转成位置参数数组：

```typescript
const stringArgs = Object.values(args).map(String);
return String(tool.handler(...stringArgs));
```

---

### 组件 2：DefaultLLMClient（最复杂的组件）

这个组件做了 4 件事：调用 API → 归一化响应 → 退避重试 → 离线降级。

#### 归一化响应格式

OpenAI SDK 返回的对象结构复杂（`response.choices[0].message.tool_calls[0].function.name`）。
我们把它**归一化**成简单的 dict：

```python
# 原始 SDK 格式（复杂、嵌套深）
msg = response.choices[0].message
for tc in msg.tool_calls:
    name = tc.function.name          # 还要判断 tc.type == "function"
    args = tc.function.arguments     # 是 JSON 字符串，要 json.loads

# 归一化格式（简单、统一）
{"content": "...", "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}], "usage": {...}}
```

这样 `ActionParser` 和 `AgentRunner` 都不用关心 SDK 的具体格式——**屏蔽差异**。

#### 退避重试（第06章机制 1+4）

```python
def chat_with_retry(self, messages, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return self.chat(messages, tools=tools)
        except (APITimeoutError, APIConnectionError, RateLimitError):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 0.1)  # 退避：0.1s, 0.2s, 0.4s
                continue
            break  # 重试耗尽 → 降级
        except (AuthenticationError, BadRequestError):
            break  # 永久错误 → 不重试，直接降级
    return self._next_mock()  # 降级
```

TypeScript 版用 `APIConnectionError` 统一捕获连接错误和超时错误
（`APIConnectionTimeoutError` 继承自 `APIConnectionError`）。

#### 离线 mock 降级（关键！）

`.env` 的 `OPENAI_API_KEY=sk-REPLACE-ME` 是占位符——真实 API 调用必失败（401 或连接错误）。
没有 mock 降级，教程根本跑不起来。

mock 设计：预设一个 4 步响应序列，模拟完整的 Agent 循环：

```python
self._mock_sequence = [
    {"tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}]},  # step 1
    {"tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "上海"}'}}]},  # step 2
    {"tool_calls": [{"function": {"name": "calculate", "arguments": '{"expression": "28-25"}'}}]},  # step 3
    {"content": "北京今天晴 25°C，上海今天多云 28°C。温差为 3°C..."},  # step 4: 最终回答
]
```

每次调用 `_next_mock()` 返回序列中的下一个响应，循环使用。这样即使没有有效 API 密钥，
也能完整演示 4 步 Agent 循环（查天气 → 查天气 → 算温差 → 给结论）。

> 💡 **为什么不用真实 API 做演示**？两个原因：(1) 教程要在无密钥/离线环境下跑通；
> (2) LLM 响应不确定，每次运行结果不同，不利于教学。mock 让输出 100% 可复现。

---

### 组件 3：ConversationMemory

最简单的组件——一个列表 + system prompt。但它有一个**扩展设计**值得注意：

`Memory` 接口（第12章）定义的 `add(role, content)` 只接受两个参数。但 OpenAI 多轮工具调用
需要额外字段：
- assistant 消息要带 `tool_calls`（模型决定调哪些工具）
- tool 消息要带 `tool_call_id`（对应哪个 tool_call 的结果）

我们在**不破坏接口**的前提下扩展：

```python
# 接口定义（第12章）：add(role, content)
# 实现扩展（第13章）：add(role, content, *, tool_calls=None, tool_call_id=None)
def add(self, role, content, *, tool_call_id=None, tool_calls=None):
    msg = {"role": role, "content": content}
    if tool_call_id is not None:
        msg["tool_call_id"] = tool_call_id
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    self._messages.append(msg)
```

Python 用关键字参数（`*` 强制后面的参数必须用关键字传），TypeScript 用可选的 `extra` 对象。
接口兼容（不传额外参数时行为不变），但支持了 OpenAI 的完整格式。

---

### 组件 4：OpenAIToolCallParser

把 SDK 的嵌套格式解析成扁平的 `[{name, args, id}]`：

```python
def parse_tool_calls(self, response):
    result = []
    for tc in response.get("tool_calls") or []:
        if tc.get("type") != "function":  # 判别联合类型：只处理 function
            continue
        args = json.loads(tc["function"]["arguments"])  # JSON 字符串 → dict
        result.append({"name": tc["function"]["name"], "args": args, "id": tc["id"]})
    return result
```

**TypeScript 注意**：`tc.type !== "function"` 的 `continue` 是处理判别联合类型（discriminated union）
的标准模式。OpenAI SDK 的 `tool_calls` 数组里理论上可能有非 function 类型，必须过滤。

---

### 组件 5：LoggingObserver（纯旁路）

5 个钩子方法，每个只打印一行日志。**绝对不修改主流程状态**：

```python
class LoggingObserver:
    def on_step_start(self, step, messages):
        print(f"OUT:framework:step{step+1}: ▶ 步骤开始（历史消息: {len(messages)} 条）")
    # ... 其他 4 个钩子类似
```

输出带 `OUT:framework:step{N}:` 前缀，方便 grep 过滤。

> 💡 **为什么 Observer 必须是纯旁路**？如果 Observer 改了 Memory 或 messages，
> 它就变成了"隐藏的第二个主流程"，调试时根本不知道状态被谁改了。第16章的
> `GuardrailObserver` 需要干预主流程时，会用显式的"中断信号"机制，不是偷偷改状态。

---

### 组件 6：DefaultAgentRunner（心脏）

把上面 5 个组件协调起来，驱动 observe→reason→act 循环：

```python
class DefaultAgentRunner:
    def __init__(self, llm, tools, memory, parser, observer):  # 依赖注入
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.parser = parser
        self.observer = observer

    def run(self, task, max_steps=10):
        self.memory.add("user", task)

        for step in range(max_steps):
            self.observer.on_step_start(step, self.memory.get_messages())
            self.observer.on_llm_call(self.memory.get_messages())

            response = self.llm.chat_with_retry(
                self.memory.get_messages(),
                tools=self.tools.get_schema(),
            )

            # 终止条件 1：模型不调工具 = 任务完成
            if not self.parser.has_tool_calls(response):
                return response["content"]

            # 记录 assistant 决策（含 tool_calls）
            self.memory.add("assistant", response["content"], tool_calls=response["tool_calls"])

            # 执行每个工具调用
            for call in self.parser.parse_tool_calls(response):
                self.observer.on_tool_call(call["name"], call["args"])
                result = self.tools.execute(call["name"], call["args"])
                self.observer.on_tool_result(call["name"], result)
                self.memory.add("tool", result, tool_call_id=call["id"])

            self.observer.on_step_end(step)

        return "(已达到最大步数，强制停止)"  # 终止条件 2：保险丝
```

注意：`AgentRunner` 里**没有任何"具体实现"**——它只调用组件的接口方法。
这就是第12章说的"控制反转"（IoC）：循环逻辑不再依赖具体实现，而是依赖接口。

---

## 运行示例

```bash
# Python
cd ai-agent/13-framework-core
pip install -r python/requirements.txt
python3 python/main.py

# TypeScript
cd ai-agent/13-framework-core
npx tsx typescript/main.ts
```

输出（Python 版，TS 版类似）：

```
▎ Demo 2: 运行 Agent（查天气 + 算温差）
  任务: 帮我查一下北京和上海的天气，然后算一下两地温差。

OUT:framework:step1: ▶ 步骤开始（历史消息: 2 条）
OUT:framework:step1: 🧠 调用 LLM（输入 2 条消息）
OUT:framework:offline: API 不可用，降级为 mock 响应（第 1 步）
OUT:framework:step1: 🔧 调用工具 get_weather({'city': '北京'})
OUT:framework:step1: 📋 工具结果 get_weather → 北京今天晴，气温 25°C
OUT:framework:step1: ✓ 步骤结束
OUT:framework:step2: ... get_weather(上海) ...
OUT:framework:step3: ... calculate(28-25) → 28-25 = 3 ...
OUT:framework:step4: ▶ 步骤开始（历史消息: 8 条）
OUT:framework:step4: 🧠 调用 LLM（输入 8 条消息）
OUT:framework:offline: API 不可用，降级为 mock 响应（第 4 步）
OUT:framework:step4: ✓ 步骤结束

OUT:final: 北京今天晴 25°C，上海今天多云 28°C。温差为 3°C（上海比北京高 3 度）。
```

完整的 4 步循环清晰可见：查北京天气 → 查上海天气 → 算温差 → 给结论。

---

## Python vs TypeScript 实现差异

两个版本功能完全对等，但有几个语言层面的差异值得注意：

| 差异点 | Python | TypeScript |
|--------|--------|------------|
| 异步 | 同步（`time.sleep`） | 全链路 async/await（`await sleep`） |
| 接口匹配 | Protocol 结构性子类型（不用 `implements`） | interface 结构性类型（不用 `implements`） |
| 错误捕获 | `except (APITimeoutError, ...)` | `err instanceof APIConnectionError \|\| ...` |
| 工具执行 | `handler(**args)` 解包字典 | `handler(...Object.values(args).map(String))` |
| 类型断言 | 不需要 | `messages as OpenAI.ChatCompletionMessageParam[]`（特定断言，非 `as any`） |

> ⚠️ **TypeScript 为什么不用 `implements` 关键字**？虽然第12章的 demo 用了 `implements` 做演示，
> 但在本章的实现中，`Memory.add` 方法有额外的可选参数（`extra` 对象），与第12章接口的
> `add(role, content)` 签名略有不同。TS 的结构类型系统允许这种兼容扩展，但加上 `implements`
> 可能触发 LSP 的签名严格检查。所以实现类不写 `implements`，靠结构匹配。

---

## 反模式（什么不该做）

### ❌ 在 AgentRunner 里 new 组件

```python
# 坏：紧耦合，换实现要改 Runner 源码
class DefaultAgentRunner:
    def __init__(self):
        self.llm = DefaultLLMClient(get_config())  # ❌

# 正确：构造函数注入
class DefaultAgentRunner:
    def __init__(self, llm: LLMClient):  # ✅ 接收接口类型
        self.llm = llm
```

### ❌ Observer 修改主流程状态

```python
# 坏：Observer 偷偷改 Memory
class BadObserver:
    def on_tool_result(self, name, result):
        if "error" in result:
            self.runner.memory.add("system", "重试！")  # ❌ 改了 Memory

# 正确：只读不写
class GoodObserver:
    def on_tool_result(self, name, result):
        if "error" in result:
            logger.warning(f"工具失败: {result}")  # ✅ 只记录
```

### ❌ 用 `as any` 绕过类型检查

```typescript
// 坏：用 as any 把类型系统关掉
const response = await this.client.chat.completions.create({
  messages: messages as any,  // ❌
});

// 正确：用特定类型断言
const response = await this.client.chat.completions.create({
  messages: messages as OpenAI.ChatCompletionMessageParam[],  // ✅
});
```

### ❌ 用 `eval()` 做表达式求值

```python
# 坏：代码注入风险
result = eval(expression)  # ❌ expression 可能是 "__import__('os').system('rm -rf /')"

# 正确：AST 解析 + 白名单运算符（Python 版），或白名单字符过滤（TS 版）
tree = ast.parse(expression, mode="eval")
# 只允许 BinOp + 数字 + 安全运算符
```

---

## 常见错误

> 概念懂了，实际写代码还是会踩坑。

| 错误 | 症状 | 解决 |
|------|------|------|
| 注入时传了具体类而不是接口 | 换实现时发现改一处崩一片，测试无法 mock | 构造函数参数类型写 Protocol/interface，不写 `DefaultLLMClient` |
| Observer 里偷偷改了 Memory | Agent 行为莫名异常，trace 和日志对不上 | Observer 只做 `print` / `add_entry`，需要干预时返回中断信号 |
| mock 序列步数不够 | Agent 循环跑到第 5 步时 mock 用完了，抛 IndexError | mock 序列长度 ≥ `max_steps`，或用 `itertools.cycle` 循环 |
| TypeScript 忘了 `for await` | 流式 chunk 拿到的是 Promise 对象而不是字符串 | SDK 的 stream 只有异步迭代器，必须 `for await...of` |
| 工具 handler 参数类型不匹配 | TS 里 `args` 是 `Record<string, unknown>`，直接展开报错 | 用 `Object.values(args).map(String)` 转成位置参数数组 |

---

## 本章代码说明

| 文件 | 内容 | 行数 |
|------|------|------|
| `python/framework/__init__.py` | 6 大组件实现（Python） | ~350 |
| `python/main.py` | 工具定义 + 组装 + 运行 demo | ~200 |
| `typescript/framework/index.ts` | 6 大组件实现（TS） | ~350 |
| `typescript/main.ts` | 工具定义 + 组装 + 运行 demo | ~200 |

框架代码约 300 行/语言，加上 demo 约 500 行/语言。这比 LangChain 的 `AgentExecutor`
（数千行）简单得多，但核心结构完全一样。

---

## 下一步

本章你把第12章的接口图纸浇筑成了**能跑的 mini Agent 框架**。「任务助手 Agent」从此有了骨架。

但这个框架还是个"最小可用版"——它缺少几个生产级特性：

- **流式输出**：LLM 响应逐 token 流式返回（用户体验更好）
- **并行工具调用**：模型一次调多个工具时，并行执行而非串行
- **摘要记忆**：对话太长时自动压缩（第05/11章的 SummaryMemory）
- **RAG 检索**：工具能搜索外部知识库（第09章）

第14章「高级特性」会给这个框架加上这些能力，让它从"教学玩具"进化为"可用工具"。

> 💡 **从零造框架的价值**：不是要用这个框架去生产，而是当你打开 OpenAI Agents SDK、
> Pydantic AI、Mastra 的源码时，你会发现它们底层就是这 6 块积木。你造过一遍，
> 就能在一小时内读懂任何框架的架构图——这就是"先原理后工具"的力量。

---

## 代码

- [Python 实现](./python/framework/__init__.py)（6 大组件）
- [Python demo](./python/main.py)（组装 + 运行）
- [TypeScript 实现](./typescript/framework/index.ts)（6 大组件）
- [TypeScript demo](./typescript/main.ts)（组装 + 运行）
- [练习题](./exercises/README.md)
