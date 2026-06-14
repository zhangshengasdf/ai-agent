# 第13章 练习题 — 从零造框架（实现核心）

> 练习目标：通过动手扩展 mini 框架，加深对 6 大组件实现细节和依赖注入的理解。

---

## 练习 1：添加第三个工具（搜索引擎）

### 题目

当前框架注册了 `get_weather` 和 `calculate` 两个工具。请添加第三个工具 `search_knowledge`，
它有一个模拟的知识库（第09章 RAG 的简化版），让 Agent 能回答"北京的人口是多少"这类问题。

**要求**：

1. 在 `main.py` / `main.ts` 中实现 `search_knowledge(query: str) -> str` 函数
2. 定义对应的 JSON Schema（参数：`query`，类型 `string`）
3. 用 `tools.register(...)` 注册它
4. 在 `_mock_sequence` / `_mockSequence` 里添加一个调用 `search_knowledge` 的步骤
5. 运行后能看到 Agent 调用了三个不同的工具

### 参考答案（Python 版）

```python
# 1. 知识库 + 工具函数
KNOWLEDGE_DB = {
    "北京": "北京人口约 2189 万（2023 年），面积 16410 平方公里。",
    "上海": "上海人口约 2487 万（2023 年），面积 6340 平方公里。",
    "广州": "广州人口约 1873 万（2023 年），面积 7434 平方公里。",
}

def search_knowledge(query: str) -> str:
    for key, value in KNOWLEDGE_DB.items():
        if key in query:
            return value
    return f"[未找到] 没有查到关于 '{query}' 的信息。"

# 2. Schema
KNOWLEDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "搜索关键词，例如城市名"},
    },
    "required": ["query"],
}

# 3. 注册
tools.register("search_knowledge", "搜索知识库（城市信息）", KNOWLEDGE_SCHEMA, search_knowledge)

# 4. 在 framework/__init__.py 的 _mock_sequence 插入新步骤
# 例如在 step 2 和 step 3 之间加：
{"tool_calls": [{"id": "call_mock_2b", "type": "function",
  "function": {"name": "search_knowledge", "arguments": '{"query": "北京"}'}}]},
```

**关键收获**：添加一个新工具不需要改 `AgentRunner`、`LLMClient`、`ActionParser` 的任何代码——
只需 `register()` + 改 mock。这就是**开闭原则**（对扩展开放，对修改关闭）。

---

## 练习 2：实现 CostTrackingObserver（成本追踪）

### 题目

当前的 `LoggingObserver` 只打印步骤日志。请实现一个新的 `CostTrackingObserver`，
它额外追踪每步的 token 消耗和估算成本。

**要求**：

1. 创建一个新类 `CostTrackingObserver`（可以在 `LoggingObserver` 基础上扩展，也可以独立实现）
2. 在 `on_llm_call` 钩子里记录 `usage.prompt_tokens` 和 `usage.completion_tokens`
3. 用一个简单的定价模型估算成本（例如：输入 $0.15/1M tokens，输出 $0.60/1M tokens）
4. 新增一个 `get_total_cost()` 方法返回累计成本
5. 在 `main.py` / `main.ts` 的 Agent 循环结束后打印总成本
6. **纯旁路原则**：不能修改主流程的 Memory 或 messages

### 参考答案（Python 版）

```python
class CostTrackingObserver(LoggingObserver):
    """扩展 LoggingObserver，额外追踪 token 成本。"""

    # 简化定价（GPT-4o-mini 的参考价格，美元/百万 tokens）
    INPUT_PRICE = 0.15
    OUTPUT_PRICE = 0.60

    def __init__(self):
        super().__init__()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_usage = None

    def on_llm_call(self, messages):
        super().on_llm_call(messages)  # 保留父类的日志
        # usage 在响应返回后才知道，这里先标记
        self._last_usage = None

    def on_step_end(self, step):
        super().on_step_end(step)
        # 注意：当前接口的 Observer 钩子不接收 response，
        # 所以实际项目中可能需要加一个 on_llm_response(response) 钩子。
        # 这里用简化方案：在 on_tool_result 里间接拿到 usage。

    def record_usage(self, usage: dict | None) -> None:
        """额外的方法：由 AgentRunner 在拿到 response 后调用。"""
        if usage is None:
            return
        self._total_input_tokens += usage.get("prompt_tokens", 0)
        self._total_output_tokens += usage.get("completion_tokens", 0)

    def get_total_cost(self) -> float:
        input_cost = self._total_input_tokens / 1_000_000 * self.INPUT_PRICE
        output_cost = self._total_output_tokens / 1_000_000 * self.OUTPUT_PRICE
        return round(input_cost + output_cost, 6)

    def print_summary(self) -> None:
        print(f"OUT:cost: 输入 tokens={self._total_input_tokens}, "
              f"输出 tokens={self._total_output_tokens}, "
              f"总成本≈${self.get_total_cost():.6f}")
```

**关键收获**：Observer 是可组合的——你可以同时挂 `LoggingObserver` 和 `CostTrackingObserver`，
或者用继承扩展。只要遵守"只读不写"原则，多个 Observer 可以叠加而不互相干扰。

> 💡 **思考题**：当前 `Observer` 接口没有 `on_llm_response(response)` 钩子，导致 `CostTrackingObserver`
> 拿不到 `usage` 数据。你会怎么修改第12章的接口定义来解决这个问题？（提示：加一个钩子，或者
> 让 `chat_with_retry` 把 `usage` 传给 Observer。）

---

## 练习 3：识别代码中的 bug

### 题目

下面是 `DefaultAgentRunner.run()` 的一个"有 bug"的实现。请找出至少 **3 个**bug，
说明每个的后果，并给出修正方案。

```python
def run(self, task, max_steps=10):
    self.memory.add("user", task)

    for step in range(max_steps):
        self.observer.on_step_start(step, self.memory.get_messages())

        response = self.llm.chat_with_retry(
            self.memory.get_messages(),
            tools=self.tools.get_schema(),
        )

        if not self.parser.has_tool_calls(response):
            return response["content"]

        # 记录 assistant 消息
        self.memory.add("assistant", response["content"])

        for call in self.parser.parse_tool_calls(response):
            self.observer.on_tool_call(call["name"], call["args"])
            result = self.tools.execute(call["name"], call["args"])
            self.memory.add("tool", result, tool_call_id=call["id"])

    return "达到最大步数"
```

### 参考答案

这段代码有 **4 个** bug：

| # | Bug | 后果 | 修正 |
|---|-----|------|------|
| ① | `response["content"]` 可能为 `None`（当模型只返回 tool_calls 时） | `return None` 导致上层拿不到字符串；`memory.add("assistant", None)` 存入 None 值 | `response.get("content") or "(空回答)"` |
| ② | assistant 消息没有带 `tool_calls` 字段 | OpenAI API 要求 assistant 消息携带 `tool_calls`，否则下一轮 API 调用会报 400 错误 | `self.memory.add("assistant", content, tool_calls=response["tool_calls"])` |
| ③ | 缺少 `on_llm_call` 和 `on_tool_result` 和 `on_step_end` 钩子调用 | Observer 记录不完整，日志缺失关键步骤 | 补上 3 个钩子调用 |
| ④ | `return "达到最大步数"` 在循环外，但如果循环正常 return（终止条件 1）时不会打印最后的 `on_step_end` | 最后一步的 Observer 日志不完整 | 在 return 前加 `self.observer.on_step_end(step)` |

**正确版本**（对照本章 `framework/__init__.py`）：

```python
def run(self, task, max_steps=10):
    self.memory.add("user", task)

    for step in range(max_steps):
        self.observer.on_step_start(step, self.memory.get_messages())
        self.observer.on_llm_call(self.memory.get_messages())          # 修复 ③

        response = self.llm.chat_with_retry(
            self.memory.get_messages(),
            tools=self.tools.get_schema(),
        )

        if not self.parser.has_tool_calls(response):
            answer = response.get("content") or "(空回答)"             # 修复 ①
            self.observer.on_step_end(step)                             # 修复 ④
            return answer

        content = response.get("content") or ""                         # 修复 ①
        self.memory.add("assistant", content,
                         tool_calls=response.get("tool_calls"))         # 修复 ②

        for call in self.parser.parse_tool_calls(response):
            self.observer.on_tool_call(call["name"], call["args"])     # 修复 ③
            result = self.tools.execute(call["name"], call["args"])
            self.observer.on_tool_result(call["name"], result)         # 修复 ③
            self.memory.add("tool", result, tool_call_id=call["id"])

        self.observer.on_step_end(step)                                # 修复 ③④

    return "(已达到最大步数，强制停止)"
```

**关键收获**：Agent 循环里每一步都有多个职责（调 LLM + 记 Memory + 触 Observer），
漏掉任何一步都会导致 bug。这正是为什么要把循环逻辑抽成框架——集中维护，一处修复。

---

## 练习 4（拓展）：实现 SummaryMemory

### 题目

当前框架用的是 `ConversationMemory`（全量消息存储）。当对话很长时，全量消息会超出
模型的上下文窗口（第05/11章的问题）。请实现一个 `SummaryMemory`，在消息超过阈值时
自动用 LLM 做摘要压缩。

**要求**：

1. 创建 `SummaryMemory` 类，实现与 `ConversationMemory` 相同的接口
2. 当消息数超过 `threshold`（例如 10 条）时，触发摘要
3. 摘要把旧消息压缩成一条 `{"role": "system", "content": "[摘要] ..."}` 消息
4. 需要注入 `LLMClient` 来做摘要（但注意：Memory 接口不应该依赖 LLMClient——
   你需要设计一个合理的注入方式）
5. 替换 `main.py` 中的 `ConversationMemory` 为 `SummaryMemory`，验证 Agent 仍然正常工作

### 设计提示

```python
class SummaryMemory:
    """超阈值时自动摘要的记忆实现。

    设计挑战：Memory 接口（第12章）说"不调 LLM"。但摘要需要调 LLM。
    解决方案：通过构造函数注入一个"摘要函数"（callable），而不是注入 LLMClient。
    这样 Memory 不依赖 LLMClient 接口，只依赖一个 callable。
    """
    def __init__(self, system_prompt: str, threshold: int = 10,
                 summarizer: Callable[[list[dict]], str] = None):
        self._summarizer = summarizer or self._default_summarizer
        # ...

    def _maybe_summarize(self):
        if len(self._messages) > self._threshold:
            summary = self._summarizer(self._messages)
            # 保留 system prompt + 摘要 + 最近几条消息
            # ...
```

**关键收获**：接口不是教条，而是契约。`SummaryMemory` 需要调 LLM 做摘要，
但通过注入 `callable` 而非 `LLMClient`，保持了 Memory 接口的纯洁性。这就是
**依赖倒置原则**的灵活应用。

---

## 运行本章代码

```bash
# Python（6 大组件实现 + 查天气算温差 demo）
cd ai-agent/13-framework-core
python3 python/main.py

# TypeScript（对等实现）
cd ai-agent/13-framework-core
npx tsx typescript/main.ts
```

完成后，尝试上面的练习。练习 1 最简单（加一个工具），练习 4 最有挑战（实现摘要记忆）。
