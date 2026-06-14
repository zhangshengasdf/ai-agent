# 第11章 练习 — 上下文工程

> 动手实践上下文压缩、子 Agent 隔离、Token 预算管理。每个练习都附参考答案。

---

## 练习 1（核心编程题）：实现一个"长任务执行器"

**场景**：构建一个长任务执行器，任务需要 **100+ 步工具调用**（如"爬取 100 个网页并提取标题"），
但整个过程中 Agent 的上下文必须始终控制在 token 预算内（如 2000 tokens）。

**要求**：
1. 用本章的 `ContextCompactor`（或自己实现一个等效的）管理上下文
2. 模拟一个 100 步的任务：每步"调用工具"产生一条包含假数据的消息（如 `"页面N的标题: XXX，内容摘要: YYY..."`）
3. 设 token 预算为 2000，keep_recent=6
4. 每步打印当前 token 数和是否触发了压缩
5. 最终验证：不管跑了多少步，上下文 token 数始终 ≤ 2000 + 某个合理余量

**Python 参考答案**：

```python
import json
from typing import Dict, List

Message = Dict[str, str]


def estimate_tokens(messages: List[Message]) -> int:
    """估算 messages 的 token 数（1 token ≈ 3 字符）。"""
    text = json.dumps(messages, ensure_ascii=False)
    return len(text) // 3


class LongTaskExecutor:
    """长任务执行器：100+ 步工具调用，上下文始终控制在预算内。"""

    def __init__(self, budget: int = 2000, keep_recent: int = 6) -> None:
        self.budget = budget
        self.keep_recent = keep_recent
        self._messages: List[Message] = []
        self._summary: str = ""
        self.compaction_count = 0

    def execute_step(self, step: int, tool_result: str) -> bool:
        """执行一步：追加工具结果，超预算时压缩。返回是否触发了压缩。"""
        self._messages.append({
            "role": "assistant",
            "content": f"第{step}步：调用工具完成。",
        })
        self._messages.append({
            "role": "tool",
            "content": tool_result,
        })

        if estimate_tokens(self._messages) > self.budget:
            self._compact()
            return True
        return False

    def _compact(self) -> None:
        """把旧消息摘要，保留最近 keep_recent 条。"""
        split = max(self.keep_recent, 1)
        old = self._messages[:-split]
        recent = self._messages[-split:]
        if not old:
            return

        # mock 摘要：提取步骤编号
        step_nums = []
        for m in old:
            if "第" in m.get("content", "") and "步" in m.get("content", ""):
                start = m["content"].find("第") + 1
                end = m["content"].find("步")
                step_nums.append(m["content"][start:end])

        range_str = f"步骤{step_nums[0]}-{step_nums[-1]}" if step_nums else "多步"
        new_summary = f"[已完成] {range_str}，共{len(step_nums)}步工具调用已执行。"

        self._summary = f"{self._summary}\n{new_summary}" if self._summary else new_summary
        self._messages = recent
        self.compaction_count += 1

    def get_context(self) -> List[Message]:
        """返回当前上下文（摘要 + 近期消息）。"""
        result: List[Message] = []
        if self._summary:
            result.append({"role": "system", "content": f"[进度摘要] {self._summary}"})
        result.extend(dict(m) for m in self._messages)
        return result

    def current_tokens(self) -> int:
        return estimate_tokens(self.get_context())


def run_long_task() -> None:
    """模拟 100 步长任务，验证上下文始终受控。"""
    executor = LongTaskExecutor(budget=2000, keep_recent=6)
    max_tokens_seen = 0

    for step in range(1, 101):
        # 模拟每步的工具结果（含假数据）
        tool_result = (
            f"页面{step}的标题: AI Agent教程第{step}章。"
            f"内容摘要: 本章讲解第{step}个核心概念，涉及工具调用、记忆系统等。"
            f"关键词: Agent, LLM, 工具, 记忆, 推理。"
        )
        compacted = executor.execute_step(step, tool_result)
        current = executor.current_tokens()
        max_tokens_seen = max(max_tokens_seen, current)

        if step % 20 == 0 or compacted:
            flag = " ⚡压缩!" if compacted else ""
            print(f"  步骤{step:3d}: {current} tokens (累计压缩{executor.compaction_count}次){flag}")

    print(f"\n最终: {executor.current_tokens()} tokens, 压缩{executor.compaction_count}次")
    print(f"过程中最大 token 数: {max_tokens_seen} (预算 {executor.budget})")
    print(f"✓ 100步任务完成，上下文始终 ≤ {max_tokens_seen} tokens")


run_long_task()
```

**TypeScript 参考答案**：

```typescript
interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

function estimateTokens(messages: Message[]): number {
  const text = JSON.stringify(messages);
  return Math.floor(text.length / 3);
}

class LongTaskExecutor {
  private messages: Message[] = [];
  private summary: string = "";
  compactionCount = 0;

  constructor(
    private readonly budget: number = 2000,
    private readonly keepRecent: number = 6,
  ) {}

  executeStep(step: number, toolResult: string): boolean {
    this.messages.push({ role: "assistant", content: `第${step}步：调用工具完成。` });
    this.messages.push({ role: "tool", content: toolResult });

    if (estimateTokens(this.messages) > this.budget) {
      this.compact();
      return true;
    }
    return false;
  }

  private compact(): void {
    const split = Math.max(this.keepRecent, 1);
    const old = this.messages.slice(0, -split);
    const recent = this.messages.slice(-split);
    if (old.length === 0) return;

    const stepNums: string[] = [];
    for (const m of old) {
      if (m.content.includes("第") && m.content.includes("步")) {
        const start = m.content.indexOf("第") + 1;
        const end = m.content.indexOf("步");
        stepNums.push(m.content.slice(start, end));
      }
    }

    const rangeStr = stepNums.length > 0
      ? `步骤${stepNums[0]}-${stepNums[stepNums.length - 1]}`
      : "多步";
    const newSummary = `[已完成] ${rangeStr}，共${stepNums.length}步工具调用已执行。`;

    this.summary = this.summary ? `${this.summary}\n${newSummary}` : newSummary;
    this.messages = recent;
    this.compactionCount++;
  }

  getContext(): Message[] {
    const result: Message[] = [];
    if (this.summary) {
      result.push({ role: "system", content: `[进度摘要] ${this.summary}` });
    }
    result.push(...this.messages.map((m) => ({ ...m })));
    return result;
  }

  currentTokens(): number {
    return estimateTokens(this.getContext());
  }
}

async function runLongTask(): Promise<void> {
  const executor = new LongTaskExecutor(2000, 6);
  let maxTokensSeen = 0;

  for (let step = 1; step <= 100; step++) {
    const toolResult =
      `页面${step}的标题: AI Agent教程第${step}章。` +
      `内容摘要: 本章讲解第${step}个核心概念，涉及工具调用、记忆系统等。` +
      `关键词: Agent, LLM, 工具, 记忆, 推理。`;
    const compacted = executor.executeStep(step, toolResult);
    const current = executor.currentTokens();
    maxTokensSeen = Math.max(maxTokensSeen, current);

    if (step % 20 === 0 || compacted) {
      const flag = compacted ? " ⚡压缩!" : "";
      console.log(`  步骤${String(step).padStart(3)}: ${current} tokens (累计压缩${executor.compactionCount}次)${flag}`);
    }
  }

  console.log(`\n最终: ${executor.currentTokens()} tokens, 压缩${executor.compactionCount}次`);
  console.log(`过程中最大 token 数: ${maxTokensSeen} (预算 ${executor.budget})`);
  console.log(`✓ 100步任务完成，上下文始终 ≤ ${maxTokensSeen} tokens`);
}

runLongTask();
```

**验证**：不管跑 100 步还是 1000 步，`max_tokens_seen` 应始终在 budget 附近（如 ≤ 2500），
不会无限增长。这证明了压缩机制有效控制了上下文规模。

> 💡 **进阶**：把 mock 摘要换成真实 LLM 调用（参考本章 `ContextCompactor._llm_summarize`），
> 摘要质量会大幅提升——LLM 能理解"步骤 1-20 完成了网页爬取"，而不只是提取编号。

---

## 练习 2（理解题）：对比三种上下文管理策略

运行 `python3 python/main.py`，观察 Demo 2（压缩）、Demo 3（子 Agent 隔离）、Demo 4（预算循环）的输出。

**问题**：
1. Demo 2 中，压缩触发后 token 数从多少降到了多少？摘要保留了什么信息？
2. Demo 3 中，主 Agent 上下文仅为子 Agent 总轨迹的百分之几？如果不隔离（把子 Agent 全量轨迹塞回主 Agent），主上下文会有多少 token？
3. Demo 4 中，budget=1500 时触发了几次压缩？如果 budget 改成 3000，会触发几次？为什么？

**参考答案**：

1. 压缩在约 1916 tokens 时触发（超过 2000 阈值前的累积），压缩后保留摘要 + 最近 6 条原文。摘要保留了关键词（如"Python"）和片段，旧的大段搜索结果被压缩掉。最终约 1949 tokens（因为新追加的消息又增加了，但旧消息已压缩）。

2. 主 Agent 上下文约为子轨迹的 11-12%（如 186 tokens vs 1572 tokens）。如果不隔离，把 3 个子 Agent 的全量轨迹（1572 tokens）全塞回主 Agent，主上下文会暴增到 ~1700+ tokens——丧失了隔离的全部意义。隔离节省了 ~1386 tokens。

3. budget=1500 时触发了 5 次压缩。如果改成 3000，阈值变为 2400，12 轮对话累积约 2017 tokens < 2400，**不会触发任何压缩**。这说明预算大小决定了压缩频率——预算越大，压缩越懒（成本低但上下文长）；预算越小，压缩越频繁（上下文短但 API 调用多）。

---

## 练习 3（编程题）：实现"笔记式记忆"

本章 README 第 6 节介绍了笔记式记忆：边执行边记要点。请实现一个 `NoteTakingAgent`，
每步执行工具后用一句话提炼要点，最终上下文只有笔记而非原始轨迹。

**要求**：
1. 模拟 5 步研究任务，每步产生一段 200 字的"工具结果"
2. 每步提取一句话笔记（mock 实现：取结果的前 30 字符 + `...`）
3. 最终上下文 = [system: 笔记列表] + [最近 2 步原文]，而非全部 5 步原文
4. 对比笔记式 vs 堆叠式的 token 数

**Python 参考答案**：

```python
import json
from typing import Dict, List

Message = Dict[str, str]


def estimate_tokens(messages: List[Message]) -> int:
    return len(json.dumps(messages, ensure_ascii=False)) // 3


def mock_extract_note(tool_result: str) -> str:
    """Mock 笔记提取：取前30字符模拟 LLM 提炼要点。"""
    return tool_result[:30].replace("\n", " ") + "..."


def run_note_taking_agent() -> None:
    """笔记式 Agent：每步记笔记，最终上下文只有笔记。"""
    steps = [
        ("搜索LangChain", "LangChain是主流的LLM应用框架，由Harrison Chase创建。支持工具调用、记忆、链式调用等核心功能。2023年后逐渐被LangGraph取代部分场景。"),
        ("搜索LangGraph", "LangGraph是LangChain团队推出的图式Agent编排框架。支持循环、条件分支、状态管理。适合复杂多Agent协作场景。"),
        ("搜索ReAct", "ReAct是Reasoning+Acting的缩写，由Yao等人2022年提出。让LLM交替输出推理和行动。是现代Agent的核心范式之一。"),
        ("搜索工具调用", "工具调用（Function Calling）是让LLM决定调用哪个函数。OpenAI 2023年6月正式支持。现代Agent的基础能力。"),
        ("搜索记忆系统", "记忆系统让Agent跨轮次保留信息。分短期(Buffer)、中期(Summary)、长期(Vector)。组合使用最常见。"),
    ]

    notes: List[str] = []
    raw_messages: List[Message] = [{"role": "system", "content": "你是研究助手。"}]

    for action, result in steps:
        note = mock_extract_note(result)
        notes.append(f"- [{action}] {note}")
        # 原始结果存进 raw_messages（模拟"不记笔记"的堆叠模式）
        raw_messages.append({"role": "tool", "content": result})

    # 笔记式上下文：只有笔记 + system
    note_messages: List[Message] = [
        {"role": "system", "content": "你是研究助手。\n已知要点:\n" + "\n".join(notes)}
    ]

    note_tokens = estimate_tokens(note_messages)
    raw_tokens = estimate_tokens(raw_messages)

    print("笔记式上下文:")
    print(f"  {note_tokens} tokens")
    for msg in note_messages:
        print(f"  [{msg['role']}] {msg['content'][:80]}...")

    print(f"\n堆叠式上下文（对比）:")
    print(f"  {raw_tokens} tokens ({len(raw_messages)} 条消息)")

    saving = raw_tokens - note_tokens
    ratio = note_tokens / max(raw_tokens, 1) * 100
    print(f"\n笔记式节省: {saving} tokens (仅为堆叠式的 {ratio:.1f}%)")
    print(f"✓ 笔记式用实时提炼代替事后压缩，上下文更精炼。")


run_note_taking_agent()
```

**TypeScript 参考答案**：

```typescript
interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

function estimateTokens(messages: Message[]): number {
  return Math.floor(JSON.stringify(messages).length / 3);
}

function mockExtractNote(toolResult: string): string {
  return toolResult.slice(0, 30).replace(/\n/g, " ") + "...";
}

function runNoteTakingAgent(): void {
  const steps: Array<[string, string]> = [
    ["搜索LangChain", "LangChain是主流的LLM应用框架，由Harrison Chase创建。支持工具调用、记忆、链式调用等核心功能。2023年后逐渐被LangGraph取代部分场景。"],
    ["搜索LangGraph", "LangGraph是LangChain团队推出的图式Agent编排框架。支持循环、条件分支、状态管理。适合复杂多Agent协作场景。"],
    ["搜索ReAct", "ReAct是Reasoning+Acting的缩写，由Yao等人2022年提出。让LLM交替输出推理和行动。是现代Agent的核心范式之一。"],
    ["搜索工具调用", "工具调用（Function Calling）是让LLM决定调用哪个函数。OpenAI 2023年6月正式支持。现代Agent的基础能力。"],
    ["搜索记忆系统", "记忆系统让Agent跨轮次保留信息。分短期(Buffer)、中期(Summary)、长期(Vector)。组合使用最常见。"],
  ];

  const notes: string[] = [];
  const rawMessages: Message[] = [{ role: "system", content: "你是研究助手。" }];

  for (const [action, result] of steps) {
    const note = mockExtractNote(result);
    notes.push(`- [${action}] ${note}`);
    rawMessages.push({ role: "tool", content: result });
  }

  const noteMessages: Message[] = [
    { role: "system", content: `你是研究助手。\n已知要点:\n${notes.join("\n")}` },
  ];

  const noteTokens = estimateTokens(noteMessages);
  const rawTokens = estimateTokens(rawMessages);

  console.log("笔记式上下文:");
  console.log(`  ${noteTokens} tokens`);
  for (const msg of noteMessages) {
    console.log(`  [${msg.role}] ${msg.content.slice(0, 80)}...`);
  }

  console.log(`\n堆叠式上下文（对比）:`);
  console.log(`  ${rawTokens} tokens (${rawMessages.length} 条消息)`);

  const saving = rawTokens - noteTokens;
  const ratio = (noteTokens / Math.max(rawTokens, 1)) * 100;
  console.log(`\n笔记式节省: ${saving} tokens (仅为堆叠式的 ${ratio.toFixed(1)}%)`);
  console.log(`✓ 笔记式用实时提炼代替事后压缩，上下文更精炼。`);
}

runNoteTakingAgent();
```

**验证**：笔记式 token 数应远小于堆叠式（通常 < 30%）。

---

## 练习 4（思考题）：何时用压缩，何时用子 Agent 隔离？

一个 Agent 要完成"阅读 10 篇论文，每篇提取 3 个要点，最后写综述"的任务。

**问题**：
1. 如果只用压缩（ContextCompactor），会遇到什么问题？
2. 如果用子 Agent 隔离（每篇论文派一个子 Agent），架构怎么设计？
3. 两者如何组合使用最优？

**参考答案**：

1. **只用压缩的问题**：10 篇论文 × 每篇 ~2000 tokens = 20000 tokens 的原始内容。即使压缩，摘要也会累积。而且压缩是"事后"的——模型在压缩前已经读了 20000 tokens，lost-in-the-middle 风险高。更关键的是，提取要点和写综述是**两种不同任务**，混在一个上下文里互相干扰。

2. **子 Agent 隔离架构**：

```
主 Agent（写综述）
  ├── 派子Agent1 → 读论文1 → 提取3要点 → 返回摘要
  ├── 派子Agent2 → 读论文2 → 提取3要点 → 返回摘要
  ├── ...
  └── 派子Agent10 → 读论文10 → 提取3要点 → 返回摘要
主 Agent 收到 10 段摘要 → 综合写综述
```

每个子 Agent 有独立上下文（读论文 + 提要点），完成后只返回 3 个要点的摘要。
主 Agent 上下文只有 10 段摘要（~1000 tokens），干净利落。

3. **组合最优**：
   - **子 Agent 内部**用压缩（论文很长，子 Agent 读论文时可能超限，需要压缩控制）
   - **主 Agent** 不需要压缩（只收 10 段摘要，天然在预算内）
   - 如果论文数量增到 100 篇，主 Agent 收 100 段摘要也可能超限——这时主 Agent 也需要压缩或分层子 Agent（每 10 篇一个"小组长"子 Agent 汇总，主 Agent 只收 10 个小组长的摘要）

> 💡 **核心洞察**：压缩解决"单条上下文太长"，隔离解决"多个独立子任务的轨迹互不干扰"。两者正交，组合使用。

---

## 练习 5（进阶思考）：上下文工程的成本权衡

本章提到压缩、隔离、笔记式记忆都会产生**额外 API 调用**（摘要、子 Agent 总结、笔记提炼）。

**问题**：
1. 压缩一次（调 LLM 摘要）的成本 vs 不压缩多传 token 的成本，何时前者更划算？
2. 子 Agent 隔离增加了 API 调用次数（每个子 Agent 独立调 LLM），为什么总体反而可能更省钱？
3. 如果 LLM 的上下文窗口从 128K 涨到 1M（如 Gemini），上下文工程还有必要吗？

**参考答案**：

1. **临界点计算**：假设摘要调用花费 `S` tokens（input + output），不压缩时每轮多传 `D` tokens。
   - 摘要一次的花费：`S`（固定）
   - 不压缩的额外花费：`D × 剩余轮数`（每轮都多传 `D`）
   - 当 `D × 剩余轮数 > S` 时，压缩更划算。
   - 例：摘要花 500 tokens，不压缩每轮多传 300 tokens，剩余 5 轮 → 1500 > 500，压缩省 1000 tokens。
   - **结论**：对话越长、剩余轮数越多，压缩越划算。短对话（<5 轮）不压缩可能更省钱（省了摘要调用）。

2. **子 Agent 隔离反而省钱的原因**：
   - **避免重复传长上下文**：如果不隔离，主 Agent 每轮都传子 Agent 的几千 token 轨迹。10 轮 = 10 × 几千。隔离后主 Agent 每轮只传几百 token 摘要，省下的远超子 Agent 的独立调用成本。
   - **并行化**：子 Agent 可以并行跑（asyncio/Promise.all），时间成本不叠加。
   - **精准上下文**：子 Agent 只看自己任务相关的上下文，不携带无关信息，输出质量更高（减少重试）。

3. **1M 窗口下仍有必要**，因为：
   - **成本仍然是 O(n²)**：即使窗口够大，每轮传 500K tokens 的成本是天文数字（gpt-4o 价格下 $1.25/轮）。
   - **Lost in the Middle 不消失**：论文显示即使在长窗口模型上，中间信息仍容易被忽略。窗口大 ≠ 检索质量好。
   - **延迟仍随长度增长**：1M tokens 的输入，首 token 延迟可能 10+ 秒。
   - **上下文工程的目标不是"不超窗口"，而是"用最少的 token 达到最好的效果"**。1M 窗口只是提高了上限，不改变"越长越贵越差"的本质。

> 💡 **结论**：上下文工程不是窗口小的权宜之计，而是 Agent 工程的永久课题。窗口再大，主动管理上下文永远比"全塞进去"更优。
