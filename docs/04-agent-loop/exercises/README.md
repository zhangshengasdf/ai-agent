# 第04章 练习 — Agent 循环

> 动手实践"多步循环"和"max_steps 防护"。每个练习都附参考答案。

---

## 练习 1：追踪 Agent 的决策链（理解题）

运行 `python3 python/main.py`（或 `npx tsx typescript/main.ts`），观察 Demo 1（查三城市天气）的输出。

**问题**：
1. Agent 共执行了几步？每步的 `action`（调用了什么工具）和 `observation`（工具返回了什么）分别是什么？
2. 哪一步是"终止条件 1"（模型自主决定停止）？模型在这一步做了什么？
3. 如果把任务改成"查 5 个城市天气"，你预期会增加几步？为什么？

**参考答案**：

1. 预期 4 步（取决于模型是否并行调用）：
   | Step | Action | Observation |
   |------|--------|-------------|
   | 1 | `get_weather("北京")` | 北京晴 25°C |
   | 2 | `get_weather("上海")` | 上海多云 28°C |
   | 3 | `get_weather("深圳")` | 深圳小雨 30°C |
   | 4 | （不调工具） | 最终回答：推荐北京… |

2. Step 4 是终止条件 1：`response.choices[0].message.tool_calls` 为空，模型直接返回 content（最终回答）。这就是"任务完成"的信号。

3. 增加 2 步（查第 4、5 个城市），总结仍是 1 步，总共约 6 步。如果模型支持并行工具调用，可能 2 步就查完 5 个城市（一次返回多个 tool_calls）+ 1 步总结 = 3 步。这就是并行工具调用的优势。

---

## 练习 2：调整 `max_steps` 观察"过早停止"

把 `MAX_STEPS = 10` 改成 `MAX_STEPS = 2`，重新运行 Demo 1。

**问题**：
1. Agent 在第 2 步发生了什么？输出里出现了哪个标记？
2. 最终回答是什么？为什么这不是一个好的回答？
3. 把 `MAX_STEPS` 改回 10，再改成 `1`，观察 Demo 1 还能正常完成吗？这说明了什么？

**参考答案**：

1. Agent 在第 2 步查完两个城市后，循环变量 `step` 超过 `MAX_STEPS=2`，触发终止条件 2，输出 `OUT:max_steps: ⚠️ 达到最大步数 2，强制停止！`。

2. 最终回答是 `"(已达到最大步数，可能需要更具体的指令或更好的工具)"`——这是个兜底文案，因为模型还没来得及总结就被强制停止了。这不是好回答，因为 `max_steps` 太小，任务还没完成。

3. `MAX_STEPS=1` 时，Demo 1 只能查 1 个城市就被停止，根本无法完成任务。这说明：**`max_steps` 必须 ≥ 任务实际需要的步数**。设太小会"砍断"Agent；设太大会失去防护意义。经验值：简单查询 5，中等任务 10，复杂研究 20-30。

---

## 练习 3：实现一个"步数计数器"（编程题）

修改 `agent_loop` 函数，让它返回一个**元组** `(answer, steps_used)`，其中 `steps_used` 是实际执行的步数。

**要求**：
- 正常完成（终止条件 1）时，`steps_used` = 实际步数
- 达到 `max_steps`（终止条件 2）时，`steps_used` = MAX_STEPS
- 在 main 里打印每个任务用了几步

**Python 参考答案**：

```python
def agent_loop(user_message: str) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": "你是任务助手 Agent..."},
        {"role": "user", "content": user_message},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=cfg.model, messages=messages, tools=tools,
        )
        assistant_msg = response.choices[0].message

        if not assistant_msg.tool_calls:
            # 终止条件 1：正常完成
            return assistant_msg.content or "(空回答)", step

        messages.append(assistant_msg.model_dump())
        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOL_FUNCTIONS[tc.function.name](**args)
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": str(result),
            })

    # 终止条件 2：达到 max_steps
    return "(已达到最大步数)", MAX_STEPS


# 在 main 里：
answer, steps = agent_loop("查北京、上海天气并推荐")
print(f"用了 {steps} 步，回答：{answer}")
```

**TypeScript 参考答案**：

```typescript
async function agentLoop(userMessage: string): Promise<[string, number]> {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: "你是任务助手 Agent..." },
    { role: "user", content: userMessage },
  ];

  for (let step = 1; step <= MAX_STEPS; step++) {
    const response = await client.chat.completions.create({
      model: cfg.model, messages, tools,
    });
    const assistantMsg = response.choices[0].message;

    if (!assistantMsg.tool_calls || assistantMsg.tool_calls.length === 0) {
      return [assistantMsg.content ?? "(空回答)", step];
    }

    messages.push(assistantMsg);
    for (const tc of assistantMsg.tool_calls) {
      if (tc.type !== "function") continue;
      const args = JSON.parse(tc.function.arguments);
      const result = TOOL_FUNCTIONS[tc.function.name](...Object.values(args));
      messages.push({ role: "tool", tool_call_id: tc.id, content: result });
    }
  }

  return ["(已达到最大步数)", MAX_STEPS];
}

// 调用：
const [answer, steps] = await agentLoop("查北京、上海天气并推荐");
console.log(`用了 ${steps} 步，回答：${answer}`);
```

**验证**：简单问题（如"你好"）应该返回 `steps=1`；多城市天气任务应该返回 `steps=3-5`。

---

## 练习 4（进阶）：设计一个会触发 max_steps 的真实场景

思考题：除了本章的"mock 无限循环模型"，真实场景中还有哪些情况会导致 Agent 触发 `max_steps`？

**提示**：从工具返回值、任务描述、模型能力三个角度想。

**参考答案**：

真实场景中触发 `max_steps` 的常见原因：

1. **工具返回值模糊/无用**
   - 工具总返回 `"需要更多信息"`、`"查询失败"`、`"无结果"`
   - 模型反复尝试，希望下次能得到有用数据 → 死循环

2. **任务描述不清或无解**
   - 用户问"帮我查世界上最好的城市"——没有客观答案，模型反复查
   - 用户问的工具无法回答的问题（如用天气工具查股票）

3. **模型能力不足**
   - 小模型（如 7B 以下）容易陷入重复调用，不会"主动停止"
   - 温度过高导致模型决策不稳定，反复改变主意

4. **工具描述误导**
   - 工具描述说"返回详细数据"，实际返回简短 mock → 模型以为"再调一次会更详细"

5. **缺少"停止信号"**
   - system prompt 没告诉模型"信息够了就直接回答"
   - 本章的 system prompt 特意加了"当信息足够回答时，直接给出最终回答"就是这个目的

**应对策略**：
- 工具返回明确结果（成功/失败都要清晰）
- system prompt 明确"停止条件"
- 选择能力足够的模型
- **永远设 `max_steps`**（这是最后防线）

---

## 练习 5（思考）：Agent 循环 vs 普通循环

对比以下两段伪代码，回答问题：

```python
# 代码 A：普通循环（固定次数）
for i in range(3):
    weather = get_weather(cities[i])
    results.append(weather)
answer = summarize(results)

# 代码 B：Agent 循环
for step in range(MAX_STEPS):
    response = llm(messages)
    if not response.tool_calls:
        return response.content
    execute_tools(response.tool_calls)
```

**问题**：
1. 代码 A 和代码 B 的本质区别是什么？
2. 哪种更适合"查 3 个固定城市的天气"？为什么？
3. 哪种更适合"帮我研究一个我完全不了解的话题"？为什么？
4. 代码 A 有没有"Agent"的属性？为什么？

**参考答案**：

1. **本质区别：谁做决策？**
   - 代码 A：**程序员**预先决定"查 3 次、按什么顺序、怎么总结"——流程是死的
   - 代码 B：**模型**实时决定"查什么、查几次、何时停"——流程是活的

2. **固定任务用代码 A**。查 3 个已知城市，流程确定，用普通循环更简单、更便宜、更可控。Agent 循环在这里是过度设计。

3. **开放性任务用代码 B**。研究未知话题时，你不知道要查几次、查什么——这需要模型根据中间结果动态决策。这正是 Agent 的用武之地。

4. **代码 A 没有"Agent"属性**。它只是一个普通的批处理脚本。Agent 的核心是**自主决策的连续性**——模型能根据观察结果调整下一步行动。代码 A 没有这个能力，每一步都是程序员预设的。

> 💡 **核心洞察**：不是所有任务都需要 Agent。**确定性任务用普通代码，开放性任务才用 Agent。** 滥用 Agent 会增加成本、降低可靠性、增加调试难度。
