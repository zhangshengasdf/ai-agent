# 第07章 ReAct 模式（Reasoning + Acting）

> **「任务助手 Agent」获得了"思考力"**——从第04章的"隐式推理"（模型在黑盒里想），进化为"显式推理"
> （模型把思考过程写出来：**Thought → Action → Observation**）。
> 这是经典论文 *ReAct: Synergizing Reasoning and Acting in Language Models*（Yao et al., 2022）提出的核心范式，
> 几乎所有现代 Agent 框架的"推理轨迹"都源于此。

## TL;DR

> **30 秒速读**：ReAct 让模型在每一步行动前先写出思考过程（Thought → Action → Observation），分"显式"（文本解析）和"隐式"（tools API）两种实现方式。
> 
> **如果只记一件事**：显式 ReAct 用正则解析 Thought/Action 文本，可调试但脆弱；隐式 ReAct 用 tools API 的结构化 tool_calls，稳定但推理过程是黑盒。生产用隐式，教学/调试用显式。

---

## 本章目标

学完本章，你将理解：

1. **ReAct 是什么**：Reasoning + Acting 的合成词，让模型"先想清楚，再动手"
2. **Thought → Action → Observation 三段式**：每一步推理都显式地写在文本里
3. **显式 ReAct vs 隐式 ReAct**（本章核心洞察）：文本格式解析 vs tools API 内部推理
4. **为什么显式推理提升可靠性**：思考过程可见 → 减少"不假思索的幻觉"
5. **Prompt 模板设计**：Few-shot 示例 + 格式约束，让模型"听话"地输出结构化文本
6. **反模式**：模型不遵循格式、推理步骤过多增加成本

---

## 什么是 ReAct

**ReAct = Reasoning + Acting**。它解决的核心问题是：

> **模型在直接行动时容易"冲动"——没想清楚就调工具、调错工具、或者对工具结果理解错误。**

ReAct 的解法极其简单却有效：**强制模型在每一步行动之前，先写出它的"思考过程"**。

```
Thought: 我需要先查北京的温度，才能和上海比较。
Action: get_weather[北京]
Observation: 北京今天晴, 25°C, 湿度 40%, 东北风 2 级

Thought: 北京是 25°C。现在我需要查上海的温度。
Action: get_weather[上海]
Observation: 上海今天多云, 28°C, 湿度 65%, 东南风 3 级

Thought: 北京 25°C，上海 28°C。上海温度更高，高了 3°C。我现在知道答案了。
Final Answer: 上海温度更高（28°C > 北京 25°C），高 3°C。
```

三段式循环：

| 阶段 | 含义 | 谁产出 |
|------|------|--------|
| **Thought** | 推理：我现在知道什么、还缺什么、下一步该干嘛 | 模型 |
| **Action** | 行动：选择并调用一个工具 | 模型决策 → 代码执行 |
| **Observation** | 观察：工具返回的结果 | 代码执行后追加 |

这个循环持续进行，直到模型在 Thought 里判定"信息够了"，然后输出 `Final Answer`。

> 💡 **和第04章 Agent 循环的关系**：ReAct **不是**循环的替代品，而是循环里"推理"环节的**展开**。第04章用 tools API 让模型在黑盒里推理（隐式 ReAct），本章用文本格式让模型把推理写出来（显式 ReAct）。循环结构本身（`for step in range(MAX_STEPS)`）完全一样。

---

## 为什么显式推理提升可靠性

这是 ReAct 论文的核心实验结论。原因有三：

### 1. "想清楚再说"减少幻觉

当模型被强制写出 `Thought` 时，它实际上在做一个**自我审查**：

```
# 没有 Thought（隐式推理）→ 模型可能直接行动
模型 → get_weather["北京和上海"]  # ❌ 参数错误，工具不接受多城市

# 有 Thought（显式推理）→ 模型先想清楚再行动
Thought: 我需要分别查北京和上海的温度。先查北京。
Action: get_weather[北京]  # ✓ 参数正确
```

Thought 充当了"决策检查点"——模型在写下思考过程时，更容易发现自己的逻辑漏洞。

### 2. 推理轨迹可追溯

当 Agent 出错时，Thought 记录了它**为什么**这么做：

```
Thought: 用户问"东京天气"，我的工具列表里有 get_weather，它接受城市名。
Action: get_weather[东京]
Observation: 东京今天阴, 22°C
Thought: 东京是 22°C，比北京冷。  ← 这里推理出错了（用户没问北京）
```

通过回看 Thought，你能精确定位"推理在哪一步走偏了"，而不是面对一个黑盒猜半天。

### 3. 工具结果被"消化"而非"转发"

隐式推理时，模型有时只是**转述**工具结果（"北京是 25°C"）。显式推理强制模型在 Thought 里**消化**结果：

```
# 隐式（可能只是转发）
Observation: 上海 28°C
模型直接输出: "上海 28°C"  # 没有推理

# 显式（必须消化）
Observation: 上海 28°C
Thought: 上海 28°C 比北京 25°C 高，所以上海更热。  # 有推理
```

"消化"意味着模型把新信息和已有信息**关联**起来，而不是孤立地看待。

---

## 显式 ReAct vs 隐式 ReAct（本章核心洞察）

这是本章最重要的对比。理解它，你就理解了 Agent 推理的两种范式。

### 显式 ReAct（经典文本格式）

模型输出**纯文本**，你用正则或字符串解析提取 Thought 和 Action。

```
模型输出:
"Thought: 我需要先查北京的温度。\nAction: get_weather[北京]"

你的代码:
match = re.search(r'Thought:\s*(.*?)\nAction:\s*(\w+)\[(.*?)\]', text)
thought, tool, args = match.groups()
```

**特点**：
- ✅ 推理过程**完全可见**（Thought 是明文）
- ✅ **任何模型都能用**（只需纯文本补全能力，不需要 tools API 支持）
- ✅ 可调试性极高（看 Thought 排查逻辑错误）
- ⚠️ 模型**可能不遵循格式**（需要 Prompt 工程约束）
- ⚠️ Thought 占**额外 token**（增加成本和延迟）
- ⚠️ 正则解析**脆弱**（模型格式轻微偏差就会解析失败）

### 隐式 ReAct（现代 tools API）

模型在**内部**推理，直接输出结构化的 `tool_calls`，你看不到推理过程。

```
模型输出:
tool_calls: [{name: "get_weather", arguments: {city: "北京"}}]
content: null  # 可能没有文本推理

你的代码:
for tc in response.choices[0].message.tool_calls:
    result = TOOL_FUNCTIONS[tc.function.name](**json.loads(tc.function.arguments))
```

**特点**：
- ✅ **结构化输出**，无需正则解析（SDK 自动处理）
- ✅ **更稳定**（JSON Schema 约束，模型不易跑偏）
- ✅ **无 Thought 开销**（推理在模型内部，不占输出 token）
- ⚠️ 推理过程**不可见**（黑盒，调试困难）
- ⚠️ 需要**支持 tools API 的模型**（部分开源模型不支持）

### 对比表

| 维度 | 显式 ReAct（文本解析） | 隐式 ReAct（tools API） |
|------|----------------------|------------------------|
| **推理可见性** | ✅ Thought 明文可见 | ❌ 模型内部推理 |
| **工具调用格式** | 文本 `Action: name[args]` | 结构化 `tool_calls` JSON |
| **解析方式** | 正则 / 字符串解析 | SDK 自动解析 |
| **格式健壮性** | ⚠️ 模型可能不遵循 | ✅ JSON Schema 约束 |
| **可调试性** | ✅ 高（看 Thought 排查） | ⚠️ 低（黑盒） |
| **Token 成本** | ⚠️ Thought 占额外 token | ✅ 无 Thought 开销 |
| **模型兼容性** | ✅ 任何模型（纯文本） | ⚠️ 需支持 tools API |
| **适合场景** | 教学/调试/小模型 | 生产/大型应用 |

### 什么时候用哪个

**用显式 ReAct 当**：
- 教学（让学习者看清推理过程）
- 调试（Agent 出错时需要看 Thought 排查）
- 使用不支持 tools API 的模型（如某些开源模型）
- 需要审计推理过程（合规要求）

**用隐式 ReAct 当**：
- 生产环境（结构化更稳定）
- 成本敏感（无 Thought token 开销）
- 使用支持 tools API 的强大模型（GPT-4o、Claude 等）

> 💡 **现实世界**：现代框架（OpenAI Agents SDK、LangChain 等）默认用**隐式 ReAct**，因为结构化输出更稳定。但很多框架也提供"显示推理轨迹"的选项——本质就是在隐式 ReAct 基础上，把模型的内部推理也暴露出来（如 OpenAI 的 `reasoning` 字段）。理解显式 ReAct，你就能理解这些框架的底层逻辑。

---

## 显式 ReAct 的 Prompt 模板设计

显式 ReAct 的成败，80% 取决于 Prompt 设计。模型不会"天生"输出 `Thought: ... Action: ...` 格式——你必须**教它**。

### 核心要素：指令 + Few-shot 示例 + 格式约束

```
你是一个任务助手 Agent。请严格使用以下格式回答问题：

Thought: 你的推理过程（你现在知道什么、还缺什么、下一步该干嘛）
Action: 工具名[参数]
（等待 Observation）
Observation: 工具返回的结果

你可以多次重复 Thought/Action/Observation，直到信息足够。
最后用以下格式给出最终答案：

Thought: 信息已足够，我现在知道答案。
Final Answer: 你的最终回答

可用工具：
- get_weather[城市名]: 查询城市天气
- calculate[数学表达式]: 数学计算
- search_wiki[关键词]: 搜索百科

示例：
问题: 上海和深圳哪个温度更高？
Thought: 我需要分别查两个城市的温度。先查上海。
Action: get_weather[上海]
Observation: 上海今天多云, 28°C

Thought: 上海是 28°C。现在查深圳。
Action: get_weather[深圳]
Observation: 深圳今天小雨, 30°C

Thought: 上海 28°C，深圳 30°C。深圳温度更高。
Final Answer: 深圳温度更高（30°C > 上海 28°C）。

现在请回答：
问题: {用户的问题}
```

### 三个关键设计点

1. **格式指令要"死板"**：明确告诉模型用 `Thought:` / `Action:` / `Observation:` / `Final Answer:` 这些标签。不要说"可以用"，要说"必须用"。

2. **Few-shot 示例要"像样"**：给一个完整的示例（包含多步 Thought/Action/Observation），让模型"照葫芦画瓢"。示例的质量直接决定输出质量。

3. **可用工具要列出**：模型不知道有哪些工具可用——你在 Prompt 里列出工具名和参数格式，模型才知道 `Action: xxx[yyy]` 的 xxx 该填什么。

> ⚠️ **格式约束的反面教材**：如果你只说"请一步步思考"，模型可能输出自由文本（"首先我认为...然后..."），而不是结构化的 `Thought: ... Action: ...`。**结构化格式必须用 Prompt 强制约束**。

---

## 显式 ReAct 的解析逻辑

模型输出文本后，你需要**解析**出 Thought 和 Action。这是显式 ReAct 最脆弱的环节。

### 正则解析

```python
import re

text = "Thought: 我需要查北京温度。\nAction: get_weather[北京]"

match = re.search(
    r'Thought:\s*(.*?)\nAction:\s*(\w+)\[(.*?)\]',
    text,
    re.DOTALL,  # 让 . 匹配换行
)
if match:
    thought = match.group(1)  # "我需要查北京温度。"
    tool = match.group(2)     # "get_weather"
    args = match.group(3)     # "北京"
```

### 必须处理的三种情况

1. **模型输出 `Final Answer`**：检测到就终止循环，提取答案。
2. **模型遵循格式**：解析出 Thought + Action，执行工具。
3. **模型不遵循格式**（既没有 Final Answer 也没有 Action）：

```python
match = parse_react(text)
if "Final Answer:" in text:
    return extract_answer(text)
elif match:
    thought, tool, args = match.groups()
    result = execute_tool(tool, args)
    prompt += f"Observation: {result}\n"  # 追加到 prompt
else:
    # 模型格式错误 → 提醒它重新格式化
    prompt += "\n（格式错误，请用 Thought:/Action:/Final Answer: 格式）\n"
```

> 💡 **容错策略**：遇到格式错误，不要直接崩溃。把"格式提醒"追加到 prompt，让模型在下一次输出中自我纠正（这和第06章的"工具异常自我纠正"一脉相承）。

---

## 反模式（什么不该做）

### ❌ 不验证每步解析，假设模型一定遵循格式

```python
# 坏：假设正则一定匹配
match = re.search(r'Thought:\s*(.*?)\nAction:\s*(\w+)\[(.*?)\]', text)
thought, tool, args = match.groups()  # ❌ match 可能是 None！

# 正确：先检查
if match is None:
    if "Final Answer:" in text:
        return extract_answer(text)
    print("格式错误，模型未遵循 ReAct 格式")
    break
```

**后果**：`match.groups()` 在 `match=None` 时抛 `AttributeError`，整个 Agent 崩溃。

### ❌ 推理步骤过多（Thought 太长/太多）

```
# 坏：模型在每个 Thought 里写 200 字"内心戏"
Thought: 嗯，让我想想。用户问的是天气对比。天气对比是个很有趣的话题。
我记得气象学上对比温度需要...（省略 300 字）...所以我决定先查北京。
Action: get_weather[北京]
```

**后果**：
- **Token 成本爆炸**：Thought 占大量输出 token，每次循环烧钱
- **延迟增加**：模型生成 300 字 Thought 比生成 30 字慢 10 倍
- **干扰决策**：冗长的 Thought 有时让模型自己都"绕晕了"

**正确**：在 Prompt 里约束 "Thought 控制在 1-2 句话"。用 `max_tokens` 限制单步输出长度。

### ❌ 混用显式和隐式

```python
# 坏：给模型传了 tools 参数，又要求它输出 Thought/Action 文本格式
response = client.chat.completions.create(
    model=cfg.model,
    messages=messages,
    tools=tools,  # ❌ 这会让模型用 tool_calls（隐式）
)
# 同时 system prompt 说 "请用 Thought:/Action: 格式"  ← 矛盾！
```

**后果**：模型困惑，可能同时输出 `tool_calls` **和** `Thought: ... Action: ...` 文本，导致逻辑混乱。

**正确**：显式 ReAct **不传 `tools` 参数**——完全靠文本格式。隐式 ReAct **传 `tools` 参数**——不要求文本格式。**二选一，不要混用。**

### ❌ Observation 不追加到 prompt

```python
# 坏：执行了工具但没把结果反馈给模型
match = parse_react(text)
result = execute_tool(match)
# ❌ 忘记 prompt += f"Observation: {result}\n"
# 下一轮模型看不到结果，会重复调用同一个工具
```

**后果**：模型永远看不到工具结果，陷入无限重复调用。

**正确**：每一步的 Observation **必须**追加到 prompt，让模型在下一轮"看到"结果。这正是 ReAct 循环的核心。

## 常见错误

> 概念懂了，实际写代码还是会踩坑。这些是初学者最常犯的错误。

| 错误 | 症状 | 解决 |
|------|------|------|
| 正则写错，漏了 `re.DOTALL` | Thought 里有换行时只匹配到第一行，后面截断 | 加 `re.DOTALL` 标志，或用 `[\s\S]*?` 替代 `.*?` |
| 模型输出中文标点 `：` 而非英文 `:` | 正则 `r'Thought:\s*'` 匹配不到，整轮解析失败 | Prompt 里明确写"必须用英文冒号"，或正则同时匹配 `[：:]` |
| 没处理 `Final Answer` 的提取 | 模型给了答案但代码还在等 Action，循环空转到 max_steps | 先检测 `"Final Answer:" in text`，再走 Action 解析分支 |
| `max_tokens` 设太小 | Thought 被截断，Action 行没生成，解析失败 | 单步至少给 300-500 tokens，让模型有空间写完 Thought + Action |

---

## 完整流程图（显式 ReAct 解决"哪个城市更热"）

```
用户: 北京和上海哪个温度更高？

Step 1:
  模型输出: Thought: 我需要先查北京温度。
           Action: get_weather[北京]
  解析: tool=get_weather, args=北京
  执行: → "北京今天晴, 25°C"
  追加: prompt += "Observation: 北京今天晴, 25°C\n"

Step 2:
  模型输出: Thought: 北京 25°C。现在查上海。
           Action: get_weather[上海]
  解析: tool=get_weather, args=上海
  执行: → "上海今天多云, 28°C"
  追加: prompt += "Observation: 上海今天多云, 28°C\n"

Step 3:
  模型输出: Thought: 北京 25°C，上海 28°C。上海更高。
           Final Answer: 上海温度更高（28°C > 25°C），高 3°C。
  检测到 Final Answer → 终止，返回答案
```

对比隐式 ReAct（tools API）解决同一问题：

```
Step 1: tool_calls: [get_weather(北京)]  ← 推理不可见
Step 2: tool_calls: [get_weather(上海)]  ← 推理不可见
Step 3: content: "上海温度更高..."        ← 直接给答案
```

注意：隐式版**看不到**模型为什么决定先查北京——它在内部想好了，直接给你 `tool_calls`。

---

## 运行示例

```bash
# Python
cd ai-agent/07-react
python3 python/main.py

# TypeScript
cd ai-agent/07-react
npx tsx typescript/main.ts
```

代码会先用真实 API 尝试（占位符密钥会失败），然后**自动降级为离线 mock 演示**，100% 可靠地展示：

1. **显式 ReAct**（`OUT:explicit:step{N}:`）：Thought → Action → Observation 完整循环
2. **隐式 ReAct**（`OUT:implicit:step{N}:`）：tools API 的 tool_calls 序列
3. **对比输出**（`OUT:compare:`）：两种范式的并排对比表

---

## 兼容性注意

- **`.env` 是占位符密钥**（`OPENAI_API_KEY=sk-REPLACE-ME`）→ 真实 API 调用会 401。
  代码捕获错误并降级为离线 mock，依然展示完整的 ReAct 逻辑。
- **显式 ReAct 不依赖 tools API**——它只要求模型能做文本补全，所以**任何模型都能用**
  （包括不支持 function calling 的模型）。这是它的重要优势。
- **隐式 ReAct 需要支持 tools API 的模型**（GPT-4o、Claude、DeepSeek 等）。

---

## 下一步

本章你让「任务助手 Agent」获得了**显式思考力**——它会把推理过程写出来，让你看清每一步的"为什么"。

但 ReAct 是**线性推理**——一步步往下走。有些复杂任务需要**先规划全局，再分步执行**，甚至需要探索多条路径、回溯重来。

第08章「规划与 Tree-of-Thought」会解决这个问题：让 Agent 先生成一个"计划"（Plan），再按计划执行，遇到死胡同还能回溯。这让 Agent 能处理更复杂的多步任务。

> 💡 **ReAct 是基础**：几乎所有高级推理模式（Plan-and-Execute、Tree-of-Thought、Reflection）都建立在 ReAct 的"Thought → Action → Observation"之上。理解了 ReAct，后续章节会非常自然。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
