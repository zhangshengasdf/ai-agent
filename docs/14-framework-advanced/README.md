# 第14章 从零造框架 — 高级特性（流式、结构化输出、工具校验）

> **「任务助手 Agent」从教学玩具走向生产可用**——第13章你造的 6 大组件框架，
> 本章给它装上 3 个生产级特性：**流式输出**（用户不再盯着空白等）、
> **结构化输出强制**（LLM 输出 100% 符合 schema）、**工具参数校验**
> （在执行前拦截非法参数）。最后对比 OpenAI Agents SDK / Pydantic AI / Mastra
> 如何用更少代码完成同样的事——理解原理后，你打开任何框架源码都能 1 小时看懂。

---

## TL;DR

> **30 秒速读**：给第13章的 mini 框架装上流式输出、结构化输出强制、工具参数校验三个生产级特性，然后对比 OpenAI Agents SDK / Pydantic AI / Vercel AI SDK 如何用一行代码搞定你 50 行的逻辑。
> 
> **如果只记一件事**：校验失败时把错误信息反馈给 LLM 让它自我修正，比直接抛异常给用户强 100 倍。

---

## 本章目标

学完本章，你将能够：

1. **实现流式输出**：用 `stream=True` / `for await` 逐 token 块下发，提升用户体验
2. **强制结构化输出**：`response_format=json_object` + Pydantic/Zod/手动校验 + 失败重试
3. **加工具参数校验**：执行工具前用 JSON Schema 校验 args，拦截类型错误
4. **理解现代框架的价值**：同样的特性，OpenAI Agents SDK 用 5 行搞定你 50 行的代码
5. **判断何时该切换**：明白"自造框架用于学原理，生产用现代框架"的边界

> ⚠️ **前置条件**：先学第13章（自造框架的 6 大核心组件）。本章的高级特性
> 都是在第13章框架概念上叠加的——不读第13章你会不知道这些特性"加在哪里"。

---

## 从第13章到第14章：教学玩具 → 生产工具

第13章你造了一个 mini Agent 框架（6 大组件），它能跑通"查天气 + 算温差"
的多步循环。但把它放到生产环境，三个问题立刻暴露：

| 问题 | 第13章表现 | 用户体验 | 本章解法 |
|------|-----------|----------|----------|
| **响应慢** | LLM 一次性返回完整回答 | 用户盯着 spinner 等 5 秒 | **流式输出**（首字延迟 < 500ms） |
| **输出不可控** | LLM 偶尔返回非 JSON / 缺字段 | 下游代码崩溃 | **结构化强制**（校验 + 重试） |
| **工具崩** | `handler(**args)` 直接执行 | `TypeError: unsupported operand` | **参数校验**（执行前拦截） |

本章的 4 个 demo 分别解这 3 个问题 + 一个现代框架对比。

```
14-framework-advanced/
├── README.md              ← 你在这里
├── python/
│   ├── main.py            ← 4 个 demo（流式/结构化/校验/对比）
│   └── requirements.txt   ← openai + pydantic
├── typescript/
│   └── main.ts            ← 对等实现
└── exercises/
    └── README.md          ← 练习 + 参考答案
```

---

## 特性 1：流式输出（Streaming）

### 为什么需要流式

非流式调用的体验问题：

```
用户输入 → [空白等待 5 秒] → 一次性显示 500 字回答
```

用户在这 5 秒里不知道 Agent 是不是崩了。流式输出解掉这个痛点：

```
用户输入 → [0.3 秒后首字到达] → 逐 token 显示（打字机效果）
```

**核心指标：首字延迟（Time-to-First-Token, TTFT）**。非流式 TTFT = 总生成时间；
流式 TTFT ≈ 模型开始输出的时间（通常 < 500ms）。用户体验提升不是线性的，是质的飞跃。

### 实现原理

OpenAI 兼容 API 用 `stream=True`（Python）/ `stream: true`（TS）开启流式：

```python
# Python
stream = client.chat.completions.create(
    model=cfg.model,
    messages=[...],
    stream=True,  # ← 关键
)
for chunk in stream:
    delta = chunk.choices[0].delta.content  # 增量内容
    if delta:
        print(delta, end="", flush=True)    # 立刻输出，不等缓冲
```

```typescript
// TypeScript（必须 async iterator）
const stream = await client.chat.completions.create({
  model: cfg.model,
  messages: [...],
  stream: true,
});
for await (const chunk of stream) {
  const delta = chunk.choices[0]?.delta?.content;
  if (delta) process.stdout.write(delta);
}
```

**关键技术点**：
- `chunk.choices[0].delta.content` 是**增量**（不是完整内容），每个 chunk 只有几个 token
- `end="", flush=True`（Python）/ `process.stdout.write`（TS）确保立即输出，不被缓冲
- TS 必须用 `for await...of` 异步迭代（SDK 只有异步接口，T5 教训）
- 流式结束后需要自己拼接完整文本（`collected.append(delta)` / `collected += delta`）

### 离线 mock 流式（教学关键）

`.env` 用占位符 `OPENAI_API_KEY=sk-REPLACE-ME` → 真实 API 必失败。如何演示流式？

**方案**：把完整回答切成字符块，逐块 `yield` + `time.sleep(0.05)` 模拟网络延迟：

```python
def stream_mock(prompt: str) -> str:
    full_text = f"收到任务：『{prompt}』。我是任务助手 Agent..."
    chunks = re.findall(r".{1,3}", full_text)  # 按 2-3 字一组切片
    for chunk in chunks:
        time.sleep(0.05)  # 演示用，让"逐块"可见
        print(f"OUT:stream:chunk: {chunk}", end="", flush=True)
    return "".join(chunks)
```

这让你在无网络/无密钥环境下也能完整看到"打字机效果"。

---

## 特性 2：结构化输出强制（StructuredOutput）

### 为什么需要强制

LLM 输出本质上是非确定性的。即使你 prompt 写了"请输出 JSON"，模型偶尔会：
- 在 JSON 前后加解释文字（"好的，这是你要的 JSON: {...}"）
- 漏字段（"忘了加 priority"）
- 类型错（把 `"priority": "4"` 写成字符串而不是数字）

下游代码（`json.loads` + 字段访问）一旦遇到这些情况就崩。

### 三层防御

| 层 | 技术 | 作用 |
|---|------|------|
| 1. **API 层** | `response_format={"type": "json_object"}` | 强制模型输出合法 JSON（不带前后文字） |
| 2. **校验层** | Pydantic / Zod / 手动 type guard | 校验字段类型、必填、范围 |
| 3. **重试层** | 校验失败 → 反馈错误给 LLM → 重试 | 让模型自我修正 |

### 实现：Pydantic 校验 + 失败重试

```python
from pydantic import BaseModel, Field, ValidationError

class TaskSummary(BaseModel):
    name: str = Field(..., description="任务名称")
    difficulty: str = Field(..., description="easy|medium|hard")
    priority: int = Field(..., ge=1, le=5, description="优先级 1-5")
    estimated_hours: float = Field(..., gt=0)

def structured_call(client, model, task, max_retries=3):
    messages = [{"role": "system", "content": "...输出 JSON..."}]
    for attempt in range(1, max_retries + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            response_format={"type": "json_object"},  # 层 1
        )
        raw = resp.choices[0].message.content
        try:
            return TaskSummary.model_validate_json(raw)  # 层 2
        except ValidationError as ve:
            # 层 3：把错误反馈给 LLM
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"校验失败：{ve.json()}，请修正"})
    raise RuntimeError("重试耗尽")
```

### 为什么"反馈错误"比"直接报错丢给用户"强 100 倍

LLM 不是故意输出错的——它不知道你要什么 schema。把 ValidationError 反馈回去，
等于告诉它"你漏了 priority 字段，请补上"。模型看到具体错误后，**通常一次就能修正**。

这是错误处理的黄金模式（第06章机制 3 的变体）：**把错误信息序列化进上下文，让 Agent 自我纠正**。

### 离线 mock：模拟"一次失败 + 重试成功"

```python
mock_responses = [
    # ❌ 第 1 次：故意缺 priority 和 estimated_hours
    '{"name": "登录", "description": "...", "difficulty": "medium"}',
    # ✓ 第 2 次：完整合法
    '{"name": "登录", "description": "...", "difficulty": "medium", "priority": 4, "estimated_hours": 8.0}',
]
```

跑离线 demo 时，你能看到第 1 次校验失败、错误反馈、第 2 次成功的完整轨迹——
这就是生产环境真实发生的流程。

---

## 特性 3：工具参数校验（ToolValidation）

### 第13章的隐患

第13章的 `InMemoryToolRegistry.execute` 直接调 `handler(**args)`：

```python
def execute(self, name, args):
    return str(self._tools[name]["handler"](**args))  # ← 无校验
```

问题：如果 LLM 生成了 `{"city": 12345}`（数字而不是字符串），`get_weather(12345)`
内部 `"北京".strip()` 会崩（`int` 没有 `.strip()` 方法）。Agent 收到 `TypeError`
只能干瞪眼——它不知道自己错在哪。

### 解法：执行前校验 args

```python
def validate_tool_args(args, schema):
    """用 JSON Schema 校验 args（类型 + 必填）。"""
    errors = []
    # 1. 必填字段
    for field in schema.get("required", []):
        if field not in args:
            errors.append(f"缺少必填字段: '{field}'")
    # 2. 类型检查
    for field, value in args.items():
        expected = schema["properties"].get(field, {}).get("type")
        if expected == "string" and not isinstance(value, str):
            errors.append(f"字段 '{field}' 期望 string，实际 {type(value).__name__}")
        # ... 其他类型
    return errors

def safe_execute_tool(name, args, schema, handler):
    errors = validate_tool_args(args, schema)
    if errors:
        return f"[参数校验失败] {name}: {'; '.join(errors)}"  # 反馈给 Agent
    return str(handler(**args))  # 校验通过才执行
```

### 关键设计：校验失败不抛异常，返回错误消息

```python
# ❌ 坏：抛异常 → Agent 循环崩溃
if errors:
    raise ValueError(errors)

# ✓ 好：返回错误消息 → Agent 能"看到"错误并自我纠正
if errors:
    return f"[参数校验失败] {name}: {'; '.join(errors)}"
```

这和第06章机制 2 一致：**工具异常 + Agent 自我纠正**。Agent 收到错误消息后，
下一轮可以重新调工具（这次传对参数），而不是整个循环崩掉。

### bool 是 int 的子类陷阱（Python 特有）

Python 里 `isinstance(True, int) == True`（历史包袱）。校验 `integer` 类型时
必须先排除 bool：

```python
if expected_type in ("integer", "number") and isinstance(value, bool):
    errors.append(f"字段 '{field}' 期望 {expected_type}，实际 boolean")
    continue
```

否则 `{"priority": True}` 会通过 `integer` 校验，下游崩溃。

---

## 特性 4：现代框架对比

本章的 3 个特性，现代框架用**几行代码**就能搞定。这就是为什么"学完原理要用现代框架"。

### OpenAI Agents SDK（Python）

```python
# pip install openai-agents
from agents import Agent, Runner
from pydantic import BaseModel

class TaskSummary(BaseModel):
    name: str
    difficulty: str

agent = Agent(
    name="TaskAnalyzer",
    instructions="分析任务并输出结构化 JSON",
    output_type=TaskSummary,  # ← 一行搞定结构化输出（本章 Demo 2 全部逻辑）
)

result = Runner.run_sync(agent, "实现登录功能")
# result.final_output 已经是 TaskSummary 实例（自动校验 + 重试）
```

`output_type=TaskSummary` 这一行 = 本章 Demo 2 的**全部代码**（response_format + 校验 + 重试 + 错误反馈）。

### Pydantic AI（Python）

```python
# pip install pydantic-ai
from pydantic_ai import Agent

agent = Agent("openai:gpt-4o-mini", output_type=TaskSummary)

@agent.tool  # ← 工具自动从函数签名生成 schema（本章 Demo 3）
def get_weather(ctx, city: str) -> str:
    """查询城市天气。"""
    return f"{city}今天晴 25°C"
```

`@agent.tool` + 函数 type hints = 自动 schema 生成 + 自动类型校验。你不用手写
JSON Schema，框架从函数签名读。

### Vercel AI SDK（TypeScript）

```typescript
// npm install ai @ai-sdk/openai zod
import { generateObject, streamText } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

// 流式（对应 Demo 1）
const result = await streamText({ model: openai("gpt-4o-mini"), prompt: "你好" });
for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}

// 结构化（对应 Demo 2）
const { object } = await generateObject({
  model: openai("gpt-4o-mini"),
  schema: z.object({
    name: z.string(),
    difficulty: z.enum(["easy", "medium", "hard"]),
    priority: z.number().int().min(1).max(5),
  }),
  prompt: "分析任务：实现登录功能",
});
```

### 现代框架的"少代码"从哪来

| 本章手写的部分 | 现代框架怎么省掉 |
|---------------|------------------|
| `response_format={"type": "json_object"}` | `output_type=Model` 自动加 |
| `Model.model_validate_json(raw)` + 手动重试 | 框架内部自动校验 + 重试 |
| `validate_tool_args(args, schema)` | 从函数 type hints 自动生成 schema |
| `for chunk in stream` + 拼接 | `streamText()` 返回 ready-to-pipe 流 |
| 错误反馈给 LLM（拼 messages） | 框架内部自动做 |

**核心洞察**：现代框架不是"魔法"，它就是把本章的手写逻辑封装成了一行 API。
你懂了本章，打开 OpenAI Agents SDK 源码会看到熟悉的循环（agent_loop）、
熟悉的校验（pydantic）、熟悉的重试——只是被包了一层漂亮的 DSL。

---

## 何时该用现代框架，何时自造？

### ✅ 用现代框架的场景

- **生产项目**：需要稳定、可维护、社区支持
- **需要流式/结构化/校验**：自己写容易出 bug，框架已踩过坑
- **需要 tracing/eval**（第15-17章）：现代框架内置 OpenTelemetry
- **团队协作**：框架是行业通用语言，新人上手快
- **长期维护**：框架跟随 LLM API 演进，你不用自己适配

### ✅ 自造的场景

- **学习原理**（本教程第12-14章的目的）：看透框架黑盒
- **极简场景**：< 3 个工具、单步任务，引入框架反而增加依赖
- **定制需求**：现代框架都不满足的边缘场景（如自定义重试策略）
- **教学/演示**：不想让学习者被框架抽象分心
- **嵌入式/资源受限**：框架依赖太重

### 反模式：自造框架过度膨胀

> ⚠️ **本教程第12-14章的框架是教学骨架，不是生产工具。**

最大的反模式是：学完本章后，把这个 mini 框架继续扩展成"自己的生产框架"——
加上并行工具、向量记忆、tracing、eval、guardrail……最后你造了一个简陋版 LangChain，
还没有社区支持、没有文档、没有跟随 LLM API 演进。

**正确做法**：学完原理，切到现代框架（OpenAI Agents SDK / Pydantic AI / Mastra / Vercel AI SDK）。
你因为懂原理，能用得很好、调得很深、出 bug 能定位——这才是本教程的终极目标。

---

## Python vs TypeScript 实现差异

| 差异点 | Python | TypeScript |
|--------|--------|------------|
| 流式迭代 | `for chunk in stream`（同步） | `for await (const chunk of stream)`（异步） |
| 结构化校验 | Pydantic `model_validate_json` | 手动 type guard（或 Zod） |
| 工具校验 | `isinstance(value, str)` | `typeof value === "string"` |
| bool 陷阱 | `isinstance(True, int) == True`（需排除） | `typeof true === "boolean"`（无陷阱） |
| 现代框架 | OpenAI Agents SDK / Pydantic AI | Vercel AI SDK / Mastra |
| try-import | `try: import x; except ImportError:` | `try { require(x) } catch {}` |

> 💡 **TS 为什么不用 Zod**：本教程后期的章节（第08章起）为减少依赖，都用
> 手动 type guard 代替 Zod。本章保持一致。如果你想用 Zod，看第02章的例子。

---

## 运行示例

```bash
# Python
cd ai-agent/14-framework-advanced
pip install -r python/requirements.txt
python3 python/main.py

# TypeScript
cd ai-agent/14-framework-advanced
npx tsx typescript/main.ts
```

输出（节选，`.env` 用占位符 sk-REPLACE-ME，自动降级 mock）：

```
========================================================================
Demo 1: 流式输出（Streaming）
========================================================================

  用户输入: 你好，请简短介绍一下你能做什么。
  流式输出（逐块到达）↓
------------------------------------------------------------------------
OUT:stream:offline: 真实 API 不可用（AuthenticationError），降级 mock 流式
OUT:stream:chunk: 收到任OUT:stream:chunk: 务：『OUT:stream:chunk: 你好，...
OUT:stream:done: 共收到 67 字符（流式完成）

========================================================================
Demo 2: 结构化输出强制（StructuredOutput）
========================================================================
OUT:structured:attempt: 第 1/3 次尝试（mock）...
OUT:structured:retry: ✗ 校验失败（第 1 次）
  缺失字段: ['priority', 'estimated_hours']
OUT:structured:attempt: 第 2/3 次尝试（mock）...
OUT:structured:result: ✓ 校验通过 → {'name': '实现登录功能', ...}
OUT:structured:final: 实现登录功能 | 难度=medium | 优先级=4 | 工时=8.0h

========================================================================
Demo 3: 工具参数校验（ToolValidation）
========================================================================
OUT:validate:pass: 工具 'get_weather' 参数校验通过 → {'city': '北京'}
OUT:validate:fail: 工具 'get_weather' 参数校验失败 → 缺少必填字段: 'city'
OUT:validate:fail: 工具 'get_weather' 参数校验失败 → 字段 'city' 期望 string，实际 int
OUT:validate:summary: 通过 2 个，失败 4 个

========================================================================
Demo 4: 现代框架对比
========================================================================
OUT:compare:agents_sdk: ✗ 未安装 openai-agents（这是正常的，本教程不强制安装）
OUT:compare:pydantic_ai: ✗ 未安装 pydantic-ai（这是正常的，本教程不强制安装）
OUT:compare:decision:
  ┌─────────────────────────┬─────────────────────────────────────┐
  │ ✅ 用现代框架            │ ✅ 自造（如本教程第12-14章）         │
  ...
```

---

## 反模式（什么不该做）

### ❌ 自造框架过度膨胀

学完本章后继续给 mini 框架加功能（并行工具、向量记忆、tracing、guardrail），
最后造出简陋版 LangChain。**正确做法**：学完原理切到现代框架。

### ❌ 结构化输出只靠 prompt，不靠 schema

```python
# ❌ 坏：只靠 prompt 说"请输出 JSON"，模型偶尔不听
messages = [{"role": "user", "content": "请输出 JSON 格式..."}]

# ✓ 好：response_format + Pydantic 双保险
response_format={"type": "json_object"}
TaskSummary.model_validate_json(raw)
```

### ❌ 校验失败直接抛异常

```python
# ❌ 坏：抛异常 → Agent 循环崩溃
if errors:
    raise ValueError(errors)

# ✓ 好：返回错误消息 → Agent 自我纠正
if errors:
    return f"[参数校验失败] {errors}"
```

### ❌ 流式输出攒完再显示

```python
# ❌ 坏：攒完所有 chunk 再 print，等于没流式
full = ""
for chunk in stream:
    full += chunk.choices[0].delta.content
print(full)

# ✓ 好：每 chunk 立刻 print
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### ❌ 把现代框架当黑盒用

不学原理直接用 LangChain / OpenAI Agents SDK，出 bug 完全不知道哪里错。
**正确做法**：先学原理（本教程第01-14章），再用框架——你能调得很深。

---

## 常见错误

> 概念懂了，实际写代码还是会踩坑。

| 错误 | 症状 | 解决 |
|------|------|------|
| 流式输出攒完再 `print` | 用户等了 5 秒才看到一大段文字，和没流式一样 | 每个 chunk 立刻 `print(delta, end="", flush=True)` |
| `response_format` 只设了 `json_object` 没做 Pydantic 校验 | LLM 返回合法 JSON 但缺字段，下游 `KeyError` 崩溃 | `json_object` + `model_validate_json` 双保险 |
| 校验失败直接 `raise ValueError` | Agent 循环直接崩掉，用户看到 500 错误 | 返回错误消息字符串，让 Agent 下一轮自我修正 |
| Python 校验 integer 没排除 bool | `{"priority": True}` 通过校验，下游做算术崩溃 | `isinstance(value, bool)` 放在 `isinstance(value, int)` 之前检查 |
| 手写框架加了太多功能不肯放手 | 造出简陋版 LangChain，没有社区、没有文档 | 学完原理切到现代框架，自造只用于学习和极简场景 |

---

## 本章代码说明

| 文件 | 内容 | 行数 |
|------|------|------|
| `python/main.py` | 4 个 demo（流式/结构化/校验/对比） | ~480 |
| `typescript/main.ts` | 对等实现（async 全链路） | ~470 |
| `exercises/README.md` | 3 个练习 + 参考答案 | - |

本章代码**不 import 第13章框架**——独立可运行，便于单独学习。
概念上引用第13章的 6 大组件（"把这个 validate 插到 ToolRegistry.execute 开头"）。

---

## 下一步

恭喜！你完成了 Part 5「从零造框架」的全部 3 章（第12章接口 → 第13章实现 → 第14章高级特性）。
「任务助手 Agent」从一个想法变成了一个有骨架、有血肉、有生产级特性的 mini 框架。

但还有最后一程：**让 Agent 可测、可观测、安全地跑在生产**。

- **第15章 评估与测试**：怎么衡量 Agent 输出"好不好"？跑 eval 套件、A/B 对比
- **第16章 可观测与调试**：怎么知道 Agent 在干什么？tracing、logging、metrics
- **第17章 安全护栏**：怎么防止 Agent 干坏事？guardrail、PII 脱敏、越狱防护

学完这 3 章，你就有了把 Agent 推向生产的全部工具箱。

> 💡 **Part 5 的终极收获**：不是"你有了一个 mini 框架"，而是"你有了看穿任何框架的能力"。
> 下次有人跟你聊 LangGraph、AutoGen、CrewAI、OpenAI Agents SDK，你能问出对的问题：
> "它的 AgentRunner 在哪？工具怎么校验？流式怎么实现？"——这些问题，没人能再忽悠你。

---

## 代码

- [Python 实现](./python/main.py)（4 个 demo）
- [TypeScript 实现](./typescript/main.ts)（对等）
- [练习题](./exercises/README.md)
