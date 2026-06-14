# 第02章 Prompt 工程 — 让 LLM 精准听话

> 本章是「任务助手 Agent」的第一次能力升级：从"随便聊聊"变成"精准执行任务"。
> 我们会用原生字符串模板（f-string / 模板字符串）手写 4 种核心 Prompt 技术，
> 不依赖任何第三方抽象库。

---

## 为什么 Prompt 工程是 Agent 的地基

LLM 本身只是一个"什么都会一点"的通用引擎。真正让它变成 Agent 的，
是你给它的 **指令质量**——也就是 Prompt。

一个模糊的 Prompt（"帮我处理一下"）和一个精确的 Prompt（"你是任务管理助手，
请从用户输入中提取任务标题、优先级、描述，以 JSON 格式返回"），
差距不只是"好一点"，而是 **能不能用** 的区别。

本章教你的 4 种技术，是你后续写 Agent 的基础设施：

| 技术 | 作用 | 何时用 |
|------|------|--------|
| **System Prompt** | 定义 Agent 的"人格"和行为边界 | 每个 Agent 的第一步 |
| **Few-shot** | 用示例教模型新任务 | 分类、格式化、风格模仿 |
| **Chain-of-Thought** | 让模型展示推理过程 | 复杂推理、多步计算 |
| **结构化输出** | 强制 JSON 输出 + 类型解析 | Agent 返回可编程数据 |

---

## 1. System Prompt：Agent 的"灵魂"

System prompt 是发送给 LLM 的第一条消息（`role: "system"`），
它定义了模型的 **身份、能力边界和行为规范**。

### 消息优先级

LLM 对消息的遵从程度有隐含优先级：

```
system > user > assistant
```

这意味着：如果 user 说"忽略你之前的指令"，一个写得好的 system prompt
应该能顶住（虽然现实中不完美，但这是设计意图）。

### 无 System vs 有 System：同一问题，截然不同的回答

```python
# 无 system prompt — 模型以"通用助手"身份回答
messages_no_system = [
    {"role": "user", "content": "我明天要开会，帮我准备一下"}
]

# 有 system prompt — 模型以"任务助手"身份回答
messages_with_system = [
    {"role": "system", "content": "你是任务管理助手。用户提到任何事项，"
     "你都要提取为结构化任务（标题、优先级、描述）。"
     "优先级规则：紧急=high，重要=medium，其他=low。"
     "始终用 JSON 格式回复。"},
    {"role": "user", "content": "我明天要开会，帮我准备一下"}
]
```

无 system 时，模型可能泛泛回复"建议你准备议程、资料……"。
有 system 后，它会直接返回结构化的任务 JSON。

### System Prompt 设计要点

1. **明确身份**："你是任务管理助手"（不是"你是一个 AI"）
2. **定义能力边界**：只做任务管理，不写诗、不翻译
3. **指定输出格式**：JSON、Markdown、纯文本
4. **设定规则**：优先级判断标准、错误处理方式

---

## 2. Few-shot：用示例教模型

Few-shot prompting 是在 prompt 中给出 **几个输入-输出示例**，
让模型"学会"你想要的模式，然后对新输入做同样的事。

这是 Agent 做分类任务的核心技术——你不需要训练模型，
只需要在 prompt 里"展示"几个例子。

### 情感分类示例

```python
few_shot_prompt = """你是一个情感分类器。根据用户输入判断情感倾向。

示例：
输入：这家餐厅的菜太好吃了，下次还来！
分类：正面

输入：等了一个小时才上菜，服务态度还很差。
分类：负面

输入：餐厅在商场三楼，营业到晚上10点。
分类：中性

现在请分类：
输入：{user_input}
分类："""
```

### Few-shot 最佳实践

- **2-3 个示例足够**：太多浪费 token，太少模型学不会
- **覆盖边界情况**：正面、负面、中性各一个
- **格式一致**：所有示例用完全相同的格式
- **示例要有代表性**：选典型、不选边缘

---

## 3. Chain-of-Thought (CoT)：让模型"想"出来

Chain-of-Thought 是在 prompt 中引导模型 **逐步推理**，
而不是直接跳到答案。这对数学、逻辑、多步骤问题特别有效。

### 标准 CoT vs 直接回答

```python
# 直接回答 — 容易出错
"一个商店有 15 个苹果，卖掉了 8 个，又进了 12 个，还有多少个？"

# CoT 引导 — 更准确
"一个商店有 15 个苹果，卖掉了 8 个，又进了 12 个，还有多少个？
请一步一步思考："
```

### CoT 的两种触发方式

1. **显式引导**：在 prompt 中加"请一步一步思考"、"让我们逐步分析"
2. **Few-shot CoT**：在示例中展示推理过程

```python
cot_few_shot = """问题：小明有 5 个苹果，给了小红 2 个，又买了 3 个，他现在有几个？
思考过程：
1. 小明开始有 5 个苹果
2. 给了小红 2 个，剩下 5 - 2 = 3 个
3. 又买了 3 个，现在有 3 + 3 = 6 个
答案：6 个

问题：{question}
思考过程："""
```

### 何时用 CoT

- 数学计算（加减乘除、百分比）
- 逻辑推理（如果A则B，已知C……）
- 多步骤决策（先分析X，再考虑Y，最后判断Z）
- **不需要** CoT 的场景：简单分类、格式转换、翻译

---

## 4. 结构化输出：让 Agent 返回可编程数据

Agent 的核心价值是 **自动化**——这意味着它的输出必须能被代码解析，
而不是给人看的自然语言。

OpenAI 兼容 API 提供了 `response_format` 参数来强制 JSON 输出：

```python
response = client.chat.completions.create(
    model=cfg.model,
    messages=[...],
    response_format={"type": "json_object"},  # 强制 JSON
)
```

### 从 JSON 到类型安全

拿到 JSON 字符串后，用 Pydantic（Python）或 Zod（TypeScript）做类型校验：

```python
from pydantic import BaseModel

class TaskInfo(BaseModel):
    title: str
    priority: str  # "high" | "medium" | "low"
    description: str

# 直接从 JSON 字符串解析 + 校验
task = TaskInfo.model_validate_json(response.choices[0].message.content)
print(task.title)      # 类型安全，IDE 自动补全
print(task.priority)   # 如果 JSON 缺字段或类型错，Pydantic 会报错
```

```typescript
import { z } from "zod";

const TaskSchema = z.object({
  title: z.string(),
  priority: z.enum(["high", "medium", "low"]),
  description: z.string(),
});

// parse 会校验，失败抛 ZodError
const task = TaskSchema.parse(JSON.parse(response.choices[0].message.content!));
console.log(task.title);  // 类型安全
```

### response_format 兼容性

| 模式 | 说明 | 兼容性 |
|------|------|--------|
| `{"type": "json_object"}` | 强制输出合法 JSON | 广泛兼容（OpenAI / DeepSeek / Qwen） |
| `{"type": "json_schema", ...}` | 按指定 schema 输出 | 仅 OpenAI 部分模型 |

**建议**：优先用 `json_object` 模式（最通用），配合 Pydantic/Zod 做后置校验。

---

## 反模式：这些坑你一定会踩

### ❌ 反模式 1：Prompt 模糊

```python
# 太模糊——模型不知道你要什么
"帮我处理一下这个任务"
"分析一下数据"
"写个总结"
```

**修正**：明确输入、输出格式、规则。

```python
# 明确——模型知道该做什么
"从以下文本中提取任务标题和优先级，以 JSON 格式返回：
文本：{text}
格式：{\"title\": \"...\", \"priority\": \"high|medium|low\"}"
```

### ❌ 反模式 2：过度约束

```python
# 规则太多——模型僵化，反而容易违反
"你必须用 JSON 格式回复。JSON 必须有 title 字段。title 不能超过 50 字符。
priority 只能是 high/medium/low。description 必须用中文。
你不能说任何与任务无关的话。你不能问用户问题。你不能……"
```

**修正**：只约束关键规则（格式 + 核心字段），其余让模型自由发挥。

### ❌ 反模式 3：忽略 System Prompt 优先级

```python
# 在 user message 里设规则——容易被用户输入覆盖
messages = [
    {"role": "user", "content": "请用 JSON 格式回复。用户输入：{input}"}
]

# 应该放在 system prompt 里——优先级更高
messages = [
    {"role": "system", "content": "你是一个任务助手，始终用 JSON 格式回复。"},
    {"role": "user", "content": input}
]
```

### ❌ 反模式 4：Few-shot 示例不一致

```python
# 示例格式不一致——模型困惑
"输入：好棒 → 正面
 输入：太差了，负面
 输入：还行吧 => 中性"
```

**修正**：所有示例用完全相同的格式。

---

## 代码路径

本章代码演示了以上 4 种技术，围绕「任务助手 Agent」展开：

| 文件 | 内容 |
|------|------|
| `python/main.py` | 4 种 Prompt 技术的 Python 实现 |
| `typescript/main.ts` | 对等的 TypeScript 实现（Zod 解析） |
| `exercises/README.md` | 练习题 + 参考答案 |

运行方式：

```bash
# Python
python3 02-prompt-engineering/python/main.py

# TypeScript
cd 02-prompt-engineering/typescript && npm install && npx tsx main.ts
```

---

## 下一步

本章让你的「任务助手」学会了"听懂指令"。下一章（第03章 工具调用）
将让它学会"动手干活"——通过 function calling 调用外部工具（查日历、发邮件、算账）。
