# 第03章 工具调用（Tool Use / Function Calling）

> **「任务助手 Agent」获得了"手"**——它能调用工具查天气、做计算、搜百科，不再只是空谈。

---

## 本章目标

学完本章，你将理解：

1. **工具调用的本质**：LLM 不直接执行工具，而是输出结构化指令（JSON），由客户端执行
2. **JSON Schema 定义工具**：如何告诉模型"你有哪些工具、每个工具接受什么参数"
3. **tool_calls 响应解析**：如何从模型响应中提取工具名和参数
4. **单轮完整流程**：发送请求 → 模型决定调用工具 → 执行工具 → 反馈结果 → 模型给出最终回答

---

## 核心概念：LLM 不会"调用"工具

这是最关键的认知转变：

> **LLM 本身不能执行任何代码。** 它不能查数据库、不能调 API、不能做计算。
> 所谓"工具调用"，是 LLM **输出一段结构化 JSON**，告诉客户端：
> "嘿，我觉得你应该帮我调用这个工具，参数是这些。"

整个流程是这样的：

```
┌─────────────────────────────────────────────────────────────────┐
│  用户: "北京今天天气怎么样？"                                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  客户端 → LLM API                                               │
│  messages: [{role:"user", content:"北京今天天气怎么样？"}]         │
│  tools: [{type:"function", function:{name:"get_weather", ...}}] │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM 返回（不是文本，而是 tool_calls）：                           │
│  {                                                              │
│    "tool_calls": [{                                             │
│      "id": "call_abc123",                                       │
│      "function": {                                              │
│        "name": "get_weather",                                   │
│        "arguments": "{\"city\": \"北京\"}"                       │
│      }                                                          │
│    }]                                                           │
│  }                                                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  客户端解析 tool_calls，执行对应函数：                             │
│  result = get_weather(city="北京")                               │
│  → "北京今天晴, 25°C"                                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  客户端把工具结果以 role="tool" 追加到 messages，再次调用 LLM：    │
│  messages: [                                                    │
│    {role:"user", content:"北京今天天气怎么样？"},                  │
│    {role:"assistant", tool_calls:[...]},                        │
│    {role:"tool", tool_call_id:"call_abc123",                    │
│     content:"北京今天晴, 25°C"}                                  │
│  ]                                                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM 基于工具结果，生成最终回答：                                  │
│  "北京今天天气晴朗，气温25°C，适合出行。"                          │
└─────────────────────────────────────────────────────────────────┘
```

### 关键洞察

| 误解 | 事实 |
|------|------|
| LLM 直接调用函数 | LLM 只输出 JSON 指令，客户端负责执行 |
| LLM 能访问互联网 | LLM 只能生成文本，所有"外部能力"都靠客户端中转 |
| 工具结果是 LLM 计算的 | 工具结果来自你的代码，LLM 只是"阅读"并总结 |

---

## JSON Schema：告诉模型有哪些工具

在调用 API 时，我们通过 `tools` 参数告诉模型它能使用哪些工具。每个工具用 JSON Schema 描述：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'"
                    }
                },
                "required": ["city"],
            },
        },
    },
]
```

### 工具定义的三要素

1. **`name`**：工具的函数名，必须与你代码中的函数名一致
2. **`description`**：自然语言描述，告诉模型这个工具做什么、什么时候该用它
3. **`parameters`**：JSON Schema 格式的参数定义，包含类型、描述、是否必填

> **描述越清晰，模型选择越准确。** "处理数据"是坏描述；"查询指定城市的当前天气，返回温度和天气状况"是好描述。

---

## 单轮完整流程（本章重点）

本章只做**单轮**工具调用——一次请求、一次工具执行、一次反馈。多轮循环（Agent 反复调用工具直到完成任务）是第04章的内容。

### 四步流程

```
Step 1: 发送 user 消息 + tools 定义 → 模型返回 tool_calls
Step 2: 解析 tool_calls，执行对应工具函数，获取结果
Step 3: 把工具结果以 role="tool" 消息追加到 messages
Step 4: 再次调用 API → 模型基于工具结果返回最终文本回答
```

### 工具执行的 Dispatch 模式

```python
# 工具名 → 函数的映射
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_wiki": search_wiki,
}

# 执行时
for tool_call in response.choices[0].message.tool_calls:
    func = TOOL_FUNCTIONS[tool_call.function.name]
    args = json.loads(tool_call.function.arguments)
    result = func(**args)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
    })
```

---

## 本章的三个工具

我们给「任务助手」装备三样"手"：

| 工具 | 功能 | 参数 |
|------|------|------|
| `get_weather` | 查询城市天气（mock） | `city: str` |
| `calculate` | 安全的数学计算 | `expression: str` |
| `search_wiki` | 搜索百科知识（mock） | `query: str` |

所有工具都返回 mock 数据，不调用真实 API——保证离线可运行。

---

## 兼容性注意

不同模型对 tools 的支持程度不同：

- **OpenAI 原生支持** tools API
- **DeepSeek 也支持** tools API
- **某些模型可能不返回 tool_calls**——你的代码必须检测并优雅处理

```python
if response.choices[0].message.tool_calls:
    # 模型决定调用工具
    ...
else:
    # 模型直接返回了文本回答（没用工具）
    print(response.choices[0].message.content)
```

---

## 反模式（什么不该做）

### ❌ 工具过多

3-5 个工具是最佳数量。工具太多，模型选择准确率下降，而且每次请求都要发送所有工具定义，浪费 tokens。

### ❌ 工具描述模糊

```python
# 坏：模型不知道什么时候该用
"description": "处理数据"

# 好：清晰说明功能和使用场景
"description": "查询指定城市的当前天气，返回温度和天气状况"
```

### ❌ 直接把原始 API 响应塞回上下文

工具返回的原始数据可能很长。应该截断或结构化后再反馈给模型：

```python
# 坏：把整个 API 响应（可能几千行）塞回去
content = json.dumps(raw_api_response)

# 好：只提取关键信息
content = f"城市: {data['city']}, 温度: {data['temp']}°C, 天气: {data['condition']}"
```

### ❌ 不验证 tool_call 的 arguments

模型输出的 arguments 是 JSON 字符串，但不保证格式正确。必须用 try/except 解析：

```python
try:
    args = json.loads(tool_call.function.arguments)
except json.JSONDecodeError:
    result = "错误：工具参数格式不正确"
```

---

## 运行示例

```bash
# Python
cd ai-agent/03-tool-use
python3 python/main.py

# TypeScript
cd ai-agent/03-tool-use
npx tsx typescript/main.ts
```

---

## 下一步

本章你学会了"单轮工具调用"——模型决定调用一次工具，执行后反馈结果。
但真实世界的 Agent 需要**反复调用工具**直到完成任务。这就是第04章「Agent 循环」要解决的问题。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
