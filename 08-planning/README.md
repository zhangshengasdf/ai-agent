# 第08章 规划模式（Plan-and-Execute、Chain-of-Thought、Reflection）

> **「任务助手 Agent」获得了"战略眼光"**——从第07章的"边想边做"（ReAct 一步一步线性推进），
> 进化为"先规划全局、再分步执行、执行完还要反思改进"。
> 这是让 Agent 能处理复杂、多阶段任务的关键能力。

---

## 本章目标

学完本章，你将理解：

1. **复杂任务为何需要先规划**：减少返工、提高质量、降低成本
2. **Plan-and-Execute 模式**：先分解任务成步骤列表，再逐步执行，最后汇总
3. **Chain-of-Thought 深度推理**：让模型在规划阶段做充分的"思考"
4. **Reflection / Self-Critique**：让 Agent 审视自己的输出并改进
5. **何时该用规划**——简单任务别过度规划（反模式）
6. **反模式**：简单任务强加规划、规划后不验证

---

## 1. 复杂任务为何需要先规划

### 1.1 ReAct 的局限：边想边做容易"跑偏"

第07章的 ReAct 是**线性推理**——每一步只看到上一步的结果，然后决定下一步。这对简单任务（查天气、算算术）足够，但对复杂任务会出问题。

**例子：让 Agent 写一篇《AI Agent 调研报告》**

用 ReAct（无规划）：
```
Step 1: Thought: 先搜一下 AI Agent 是什么。Action: search[AI Agent 定义]
Step 2: Thought: 搜到了定义。再搜应用。Action: search[AI Agent 应用]
Step 3: Thought: 应用有了。搜框架。Action: search[AI Agent 框架]
Step 4: Thought: 框架有了。我想想还要搜什么... 哦，发展趋势。
        Action: search[AI Agent 趋势]
Step 5: Thought: 好像漏了挑战和风险。Action: search[AI Agent 挑战]
Step 6: Thought: 信息够了，开始写。Final Answer:（写报告）...
```

**问题**：
- **"想到哪搜到哪"**：每步的 Thought 都在临时决定下一步搜什么，缺乏全局视角
- **容易遗漏**：写到一半发现"漏了某个维度"，返工
- **难以并行**：步骤之间隐式依赖，无法预先批量检索
- **质量不稳定**：不同运行可能走完全不同的路径

### 1.2 规划的价值：先想清楚再动手

用 Plan-and-Execute（先规划）：
```
Phase 1 (Plan):
  一次性分解任务：
    1. 搜集 AI Agent 的定义与核心特征
    2. 搜集 AI Agent 的典型应用场景
    3. 搜集主流 AI Agent 框架
    4. 搜集 AI Agent 的发展趋势与挑战
    5. 综合以上信息撰写调研报告

Phase 2 (Execute):
  按计划逐步执行（每步可以并行检索，也可以串行）

Phase 3 (Synthesize):
  汇总所有步骤结果，生成最终报告
```

**好处**：
- **全局视野**：规划阶段一次性看清任务全貌，减少遗漏
- **可预测**：用户/开发者能审查计划，确认方向正确再执行
- **可并行**：步骤独立时可以并发执行（检索任务尤其适合）
- **质量更高**：综合阶段有完整信息，报告更全面

> 💡 **类比**：这就像写代码。ReAct 是"不设计就开写，遇到问题改设计"；Plan-and-Execute 是"先写设计文档，评审通过再编码"。后者返工率低得多。

---

## 2. Plan-and-Execute 模式

### 2.1 三阶段架构

Plan-and-Execute 把任务处理分成三个明确的阶段：

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Plan（规划）                                   │
│  输入：用户任务                                          │
│  输出：步骤列表 [step1, step2, ..., stepN]              │
│  实现：用 response_format=json_object 强制结构化输出     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 2: Execute（执行）                                │
│  输入：步骤列表                                          │
│  对每个步骤：调用工具 / 子查询 / 子 LLM                  │
│  输出：每步的执行结果 [result1, result2, ..., resultN]   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 3: Synthesize（汇总）                             │
│  输入：所有步骤结果                                      │
│  输出：综合的最终输出                                    │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Phase 1：用结构化输出生成计划

规划阶段的核心是**让模型输出一个结构化的步骤列表**，而不是自由文本。

```python
from pydantic import BaseModel

class Plan(BaseModel):
    """任务分解计划。"""
    steps: list[str]  # 有序的步骤列表

# 用 response_format 强制 JSON 输出
response = client.chat.completions.create(
    model=cfg.model,
    messages=[
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ],
    response_format={"type": "json_object"},  # 强制 JSON
)

# Pydantic 解析 + 校验
plan = Plan.model_validate_json(response.choices[0].message.content)
for i, step in enumerate(plan.steps, 1):
    print(f"步骤 {i}: {step}")
```

**关键点**：
- `response_format={"type": "json_object"}` 强制模型输出合法 JSON（第02章学过）
- Pydantic `Plan{steps: list[str]}` 做类型校验，保证拿到的是字符串列表
- `PLAN_SYSTEM_PROMPT` 约束步骤格式（如"每步要具体、可执行、3-6 步"）

### 2.3 Phase 2：逐步执行

执行阶段遍历步骤列表，对每个步骤做处理。每个步骤可以：
- 调用一个工具（如搜索、计算）
- 做一次子 LLM 调用（如"针对这个子问题写一段分析"）
- 组合多个操作

```python
results = []
for i, step in enumerate(plan.steps, 1):
    print(f"OUT:execute:step{i}: 执行: {step}")
    result = execute_step(step)  # 调工具或子查询
    results.append(result)
    print(f"OUT:execute:step{i}: 结果: {result[:80]}")
```

**与 ReAct 的区别**：
- ReAct 每步都问模型"下一步干什么"——动态决策
- Plan-and-Execute 步骤在规划阶段就定好了——执行阶段只是机械执行
- 这让执行阶段**可并行、可预测、可回放**

### 2.4 Phase 3：汇总

汇总阶段把所有步骤结果综合成最终输出。这通常是另一次 LLM 调用：

```python
synthesis_prompt = f"""
你是一个综合助手。用户任务：{task}
已经完成了以下步骤，收集到以下信息：
{format_results(results)}

请基于以上信息，完成用户的原始任务。
"""

final = client.chat.completions.create(
    model=cfg.model,
    messages=[{"role": "user", "content": synthesis_prompt}],
)
```

---

## 3. Chain-of-Thought：深度推理

### 3.1 CoT 是规划的地基

第02章我们学过 Chain-of-Thought——让模型"一步一步思考"。在 Plan-and-Execute 中，CoT 发生在**规划阶段**。

一个好的 `PLAN_SYSTEM_PROMPT` 会引导模型先做 CoT 再分解：

```
你是一个任务规划助手。用户会给你一个复杂任务，你需要：

1. 先理解任务的本质（这个任务到底要解决什么问题）
2. 思考完成任务需要哪些信息/操作（CoT 推理）
3. 把任务分解成 3-6 个有序的、具体的、可执行的步骤

输出 JSON 格式：{"steps": ["步骤1", "步骤2", ...]}

示例：
任务：写一篇 AI Agent 调研报告
思考：调研报告需要覆盖定义、应用、框架、趋势。每个维度单独检索，
最后综合。
步骤：
1. 检索 AI Agent 的定义与核心特征
2. 检索 AI Agent 的典型应用场景
3. 检索主流 AI Agent 开发框架
4. 检索 AI Agent 的发展趋势与挑战
5. 综合以上信息撰写调研报告
```

**注意**：虽然 prompt 里写了"思考"，但因为用了 `response_format=json_object`，
模型最终的输出只是 `{"steps": [...]}` JSON。CoT 发生在模型**内部**（推理过程被压缩进步骤质量里）。

### 3.2 CoT vs Plan-and-Execute 的关系

不要混淆：

| | Chain-of-Thought | Plan-and-Execute |
|---|---|---|
| **是什么** | 一种**推理引导**技术（Prompt 技巧） | 一种**任务分解**架构（流程设计） |
| **粒度** | 单次 LLM 调用内的"逐步思考" | 跨多次调用的"先规划再执行" |
| **输出** | 自由文本推理 | 结构化步骤列表 |
| **关系** | CoT 是 Plan 阶段的**地基**（帮模型想清楚） | Plan-and-Execute 是 CoT 的**工程化**（把思考变成可执行流程） |

一句话：**Plan-and-Execute 的规划阶段用 CoT 引导模型思考，但把思考结果固化为结构化步骤。**

---

## 4. Reflection / Self-Critique：让 Agent 审视自己

### 4.1 为什么需要反思

即使有规划，Agent 的第一次输出也可能不完美：
- 遗漏了重要信息
- 逻辑有漏洞
- 结构不清晰
- 准确性存疑

**Reflection** 的思路简单却强大：**让同一个 Agent 审视自己的输出，指出不足，然后改进。**

这模拟了人类的工作方式——写完初稿后"回头看一眼"，发现能改进的地方。

### 4.2 三轮反思流程

```
┌──────────────────────────────────────────────┐
│  Round 1: 生成初版                            │
│  输入：用户任务                               │
│  输出：draft（初版答案）                      │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  Round 2: 反思 / 自我批评                     │
│  输入：用户任务 + draft                       │
│  角色：你现在是个严格的审稿人                 │
│  输出：critique（指出 2-3 个不足）            │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│  Round 3: 改进                                │
│  输入：用户任务 + draft + critique            │
│  输出：revised（改进版答案）                  │
└──────────────────────────────────────────────┘
```

### 4.3 关键：反思 Prompt 的设计

反思阶段要用一个**不同角色的 Prompt**，强制模型从"批评者"视角看自己的输出：

```python
REFLECTION_PROMPT = """\
你是一个严格的审稿人。请审视以下初版答案，指出它的不足。

用户问题：{question}
初版答案：{draft}

请从以下维度批评：
1. 完整性：有没有遗漏重要信息？
2. 准确性：有没有事实错误或逻辑漏洞？
3. 结构：组织是否清晰？

只指出不足（2-3 点），不要给出完整改进版。
"""
```

**为什么有效**：
- 角色切换（从"作者"变"审稿人"）让模型用不同视角审视
- 显式列出批评维度（完整性/准确性/结构）避免泛泛而谈
- "只指出不足"防止模型直接重写（那样就跳过了诊断）

### 4.4 改进阶段

改进阶段把初版和反思一起喂给模型：

```python
REVISE_PROMPT = """\
用户问题：{question}
初版答案：{draft}
审稿意见：{critique}

请根据审稿意见改进初版答案，输出最终版。
"""
```

### 4.5 Reflection 的代价

Reflection 不是免费的：
- **延迟翻倍**：初版 + 反思 + 改进 = 3 次 LLM 调用（是单次的 3 倍延迟）
- **成本翻倍**：3 次调用的 token 总和
- **边际收益递减**：第 2 轮反思改进通常明显，第 3、4 轮收益很小

**实践建议**：对**质量敏感**的任务（写作、代码审查、重要决策）用 Reflection；对**速度敏感**的任务（简单问答、分类）不要用。

---

## 5. 何时该用规划（决策框架）

| 任务特征 | 是否规划 | 示例 |
|----------|----------|------|
| 单步任务（一次查询/计算） | ❌ 不规划 | "北京天气？"、"2+2=?" |
| 2-3 步，步骤隐式明确 | ❌ 不规划 | "北京和上海哪个更热？"（ReAct 够用） |
| 4+ 步，步骤需要设计 | ✅ Plan-and-Execute | "写调研报告"、"制定旅行计划" |
| 输出质量要求高 | ✅ + Reflection | "写重要邮件"、"生成代码" |
| 任务结构未知，需探索 | ⚠️ ReAct 更合适 | "调试这个 bug"（步骤未知） |

**经验法则**：
- 如果你能**一眼看出**任务需要哪些步骤 → 不规划（直接做或用 ReAct）
- 如果步骤需要**想一想才能确定** → 用 Plan-and-Execute
- 如果输出**质量比速度重要** → 加 Reflection

---

## 6. Plan-and-Execute vs ReAct vs Reflection 对比

| 维度 | ReAct（第07章） | Plan-and-Execute | Reflection |
|------|----------------|------------------|------------|
| **核心思想** | 边想边做 | 先规划再执行 | 做完反思改进 |
| **流程** | Thought→Action 循环 | Plan→Execute→Synthesize | Draft→Critique→Revise |
| **决策时机** | 每步动态决策 | 规划阶段一次决策 | 执行后回顾决策 |
| **适合任务** | 步骤未知、需探索 | 步骤明确、多阶段 | 质量敏感的生成 |
| **延迟** | 中（取决于步数） | 中高（规划+执行） | 高（3 次调用） |
| **可预测性** | 低（路径不确定） | 高（计划可审查） | 中 |
| **可并行** | 难（步骤间依赖） | 易（独立步骤可并发） | 不适用 |

**它们不是互斥的**——可以组合：
- **Plan-and-Execute + Reflection**：规划→执行→汇总→反思→改进（本章 Demo 2 体现）
- **ReAct + Plan**：用 ReAct 执行每个计划步骤（复杂任务的高级玩法）

---

## 7. 反模式（什么不该做）

### ❌ 反模式 1：简单任务强加规划

```python
# 坏：给"查天气"这种单步任务强加规划
task = "北京今天天气怎么样？"
plan = generate_plan(task)
# plan.steps = ["1. 查北京天气", "2. 总结"]  # 多此一举！
for step in plan.steps:
    execute(step)  # 纯增加延迟和成本
```

**后果**：
- **延迟增加**：规划本身要一次 LLM 调用，单步任务被拖慢
- **成本浪费**：为"查天气"这种任务多花一次 API 调用的钱
- **用户体验差**：简单问题等 5 秒才返回

**正确**：先用任务复杂度判断，简单任务直接做，复杂任务才规划。

### ❌ 反模式 2：规划后不验证（计划脱离实际）

```python
# 坏：生成了计划就直接执行，不检查计划是否合理
plan = generate_plan(task)
# 如果 plan.steps = ["1. 调用不存在的工具", "2. 查询需要权限的数据"]
for step in plan.steps:
    execute(step)  # 第一步就崩了，整个流程失败
```

**后果**：
- 计划里的步骤可能**不可执行**（调用了不存在的工具、访问了没权限的数据）
- 一步失败导致整个流程中断
- 用户看到"规划得很漂亮"但执行全失败，信任崩塌

**正确**：
- 执行前**验证计划**（步骤是否可执行、依赖是否满足）
- 每步执行有**错误处理**，失败时跳过或降级，不要让整盘崩
- 可以在计划生成后加一个"计划审查"步骤（类似 Reflection 审视计划）

### ❌ 反模式 3：规划粒度过粗或过细

```python
# 坏：粒度过粗（计划没分解，等于没规划）
plan.steps = ["1. 完成调研报告"]  # 只有一步，等于没规划

# 坏：粒度过细（每行代码都是一步，等于没抽象）
plan.steps = [
    "1. 打开浏览器",
    "2. 输入 URL",
    "3. 按回车",
    "4. 等待加载",
    "5. 读取内容",
    # ... 50 步 ...
]
```

**后果**：
- 过粗：失去规划的意义（还是要临场决策）
- 过细：规划阶段开销巨大，执行阶段变成机械转发

**正确**：每步应该是一个**有意义的子任务**（能独立产出结果），3-6 步为宜。

### ❌ 反模式 4：反思阶段不给具体批评维度

```python
# 坏：反思 prompt 太模糊
REFLECTION_PROMPT = "请看看这个答案好不好，哪里需要改进？"
# 模型可能回复："总体不错，可以再详细一点"  # 没用的废话
```

**后果**：模型泛泛而谈，反思流于形式，改进版和初版几乎一样。

**正确**：给出**具体的批评维度**（完整性/准确性/结构），强制模型聚焦。

### ❌ 反模式 5：无限反思（过度打磨）

```python
# 坏：不断反思直到"完美"
while not is_perfect(draft):
    critique = reflect(draft)
    draft = revise(draft, critique)
# 永远不会"完美"，陷入无限循环
```

**后果**：延迟无限增长，成本爆炸，边际收益趋零。

**正确**：**固定反思轮数**（通常 1 轮足够，最多 2 轮）。第 2 轮后的改进通常不值得额外成本。

---

## 8. 完整流程图（Plan-and-Execute + Reflection 写调研报告）

```
用户任务：写一篇 AI Agent 调研报告

Phase 1 (Plan):
  LLM 分解 → {"steps": [
    "1. 检索 AI Agent 定义",
    "2. 检索 AI Agent 应用场景",
    "3. 检索主流 AI Agent 框架",
    "4. 综合撰写报告"
  ]}

Phase 2 (Execute):
  Step 1: search_wiki("AI Agent 定义") → "AI Agent 是能感知..."
  Step 2: search_wiki("AI Agent 应用") → "应用于客服、编程..."
  Step 3: search_wiki("AI Agent 框架") → "LangChain、AutoGPT..."
  (每步结果累积到 results)

Phase 3 (Synthesize):
  LLM 汇总 → draft（初版报告）

Phase 4 (Reflection):
  LLM 审视 draft → critique（"缺少发展趋势"、"结构可优化"）

Phase 5 (Revise):
  LLM 根据 critique 改进 → final（最终报告）
```

---

## 运行示例

```bash
# Python
cd ai-agent/08-planning
python3 python/main.py

# TypeScript
cd ai-agent/08-planning
npx tsx typescript/main.ts
```

代码会先用真实 API 尝试（占位符密钥会失败），然后**自动降级为离线 mock 演示**，100% 可靠地展示：

1. **Plan-and-Execute**（`OUT:plan:` / `OUT:execute:step{N}:` / `OUT:synthesize:`）：完整的三阶段流程
2. **Reflection**（`OUT:reflect:draft:` / `OUT:reflect:critique:` / `OUT:reflect:revised:`）：三轮反思改进

---

## 兼容性注意

- **`.env` 是占位符密钥**（`OPENAI_API_KEY=sk-REPLACE-ME`）→ 真实 API 调用会 401。
  代码捕获错误并降级为离线 mock，依然展示完整的规划逻辑。
- **Plan-and-Execute 的规划阶段用 `response_format=json_object`**——所有 OpenAI 兼容提供商都支持（第02章验证过）。
- **离线 mock 设计**：预设步骤列表 + mock 执行结果 + 预设反思轨迹，不依赖真实 API。

---

## 下一步

本章你让「任务助手 Agent」获得了**战略眼光**——它会先规划全局、再分步执行、还能反思改进。

但这只是"自己跟自己较劲"。真正的复杂任务需要**多个 Agent 分工协作**——一个负责规划、一个负责执行、一个负责审查。

第10章「多 Agent 编排」会解决这个问题：让多个 Agent 像一个团队一样协作。本章的 Plan-and-Execute 和 Reflection 是多 Agent 协作的基础——理解了"一个 Agent 怎么规划和反思"，才能理解"多个 Agent 怎么分工和互审"。

> 💡 **规划是 Agent 的"大脑前额叶"**：人类的前额叶负责计划、决策、自我控制。Plan-and-Execute + Reflection 给 Agent 装上了一个"数字前额叶"。但要处理真正复杂的任务，还需要"多个大脑"协作——那是第10章的主题。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
