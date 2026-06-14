# 第07章 练习 — ReAct 模式

> 动手实践"Thought → Action → Observation"显式推理，以及显式 vs 隐式 ReAct 的对比。每个练习都附参考答案。

---

## 练习 1：追踪显式 ReAct 的推理轨迹（理解题）

运行 `python3 python/main.py`（或 `npx tsx typescript/main.ts`），观察离线演示中显式 ReAct 的输出（`OUT:explicit:step{N}:` 标记）。

**问题**：
1. 显式 ReAct 共执行了几步？每步的 Thought、Action、Observation 分别是什么？
2. 哪一步触发了终止条件？模型在这一步输出了什么特殊标记？
3. 对比隐式 ReAct 的输出（`OUT:implicit:step{N}:`），两者步数相同吗？每步能看到的"信息量"有什么区别？

**参考答案**：

1. 显式 ReAct 共 3 步：
   | Step | Thought | Action | Observation |
   |------|---------|--------|-------------|
   | 1 | 需要分别查两城市温度，先查北京 | get_weather[北京] | 北京今天晴, 25°C |
   | 2 | 北京 25°C，现在查上海 | get_weather[上海] | 上海今天多云, 28°C |
   | 3 | 北京 25°C，上海 28°C，上海更高 | Final Answer | — |

2. Step 3 触发终止条件。模型输出了 `Final Answer:` 标记。`parse_react_output()` 检测到 "Final Answer:" 后返回 `{type: "final_answer"}`，循环终止。

3. 步数相同（都是 3 步）。但**信息量差别巨大**：
   - 显式版每步有 **Thought**（推理过程完全可见）：你能看到模型"为什么"先查北京、"为什么"判定上海更高。
   - 隐式版每步只有 **tool_calls**（推理不可见）：你只看到模型调了 get_weather(北京)，不知道它"为什么"选择这个顺序。
   - **核心洞察**：显式 ReAct 的价值不在于"更快"或"更准"，而在于**推理可追溯**——出错时你能看 Thought 定位问题。

---

## 练习 2：用 ReAct 解决一个规划任务（编程题）

修改 `main.py` / `main.ts`，用**显式 ReAct** 解决以下任务：

> **"制定一个 3 天的北京旅行计划：查北京天气、查北京景点、计算预算（住宿 300/晚 × 3 晚 + 餐饮 200/天 × 3 天）"**

**要求**：
1. 修改离线 mock 的预设输出（`mock_model_outputs` / `mockModelOutputs`），设计一个 3-4 步的 ReAct 轨迹
2. 每一步都要有合理的 Thought（说明推理过程）
3. 最终答案要包含天气、景点和预算的综合信息
4. 运行代码，验证输出格式正确

**Python 参考答案**：

```python
# 在 demo_explicit_react_offline() 中修改 mock_model_outputs：
mock_model_outputs = [
    # Step 1: 查北京天气
    "Thought: 制定旅行计划需要先了解目的地天气。先查北京天气。\n"
    "Action: get_weather[北京]",

    # Step 2: 查北京景点
    "Thought: 北京晴 25°C，适合旅行。现在查北京有哪些景点。\n"
    "Action: search_wiki[北京]",

    # Step 3: 计算预算
    "Thought: 天气好，景点有故宫长城。现在算预算：住宿 300*3 + 餐饮 200*3。\n"
    "Action: calculate[300*3+200*3]",

    # Step 4: 综合总结
    "Thought: 预算是 1500 元。天气、景点、预算都有了，可以给出计划。\n"
    "Final Answer: 北京 3 天旅行计划：天气晴 25°C 宜出行；"
    "推荐故宫、长城；预算 1500 元（住宿 900 + 餐饮 600）。",
]
```

**TypeScript 参考答案**：

```typescript
// 在 demoExplicitReactOffline() 中修改 mockModelOutputs：
const mockModelOutputs: string[] = [
  "Thought: 制定旅行计划需要先了解目的地天气。先查北京天气。\n" +
    "Action: get_weather[北京]",
  "Thought: 北京晴 25°C，适合旅行。现在查北京有哪些景点。\n" +
    "Action: search_wiki[北京]",
  "Thought: 天气好，景点有故宫长城。现在算预算：住宿 300*3 + 餐饮 200*3。\n" +
    "Action: calculate[300*3+200*3]",
  "Thought: 预算是 1500 元。天气、景点、预算都有了，可以给出计划。\n" +
    "Final Answer: 北京 3 天旅行计划：天气晴 25°C 宜出行；" +
    "推荐故宫、长城；预算 1500 元（住宿 900 + 餐饮 600）。",
];
```

**验证**：运行后应看到 4 步输出，每步都有 Thought（说明"为什么"选这个 Action），最后给出综合答案。注意 `calculate` 工具会真正执行 `300*3+200*3=1500`。

---

## 练习 3：处理模型格式错误（编程题）

显式 ReAct 的最大风险是模型不遵循格式。修改 `parse_react_output()` / `parseReactOutput()`，增加以下测试：

**要求**：
1. 构造一个**格式错误的模型输出**（既没有 `Action:` 也没有 `Final Answer:`）
2. 验证 `parse_react_output()` 正确返回 `parse_error` 类型
3. 在 `explicit_react_loop` 中，格式错误时应追加"格式提醒"到 prompt（已在代码中实现），思考：这个策略为什么能帮助模型自我纠正？

**Python 参考答案**：

```python
# 测试格式错误处理
def test_parse_error():
    bad_output = "我觉得应该先查天气，然后再做决定。"  # 没有 Thought:/Action:
    parsed = parse_react_output(bad_output)
    assert parsed["type"] == "parse_error", f"Expected parse_error, got {parsed['type']}"
    print("✓ 格式错误正确检测")

    # 也测试只有 Thought 没有 Action 的情况
    partial = "Thought: 我需要查天气。但我不确定用哪个工具。"
    parsed2 = parse_react_output(partial)
    assert parsed2["type"] == "parse_error"
    print("✓ 部分格式正确检测")

test_parse_error()
```

**为什么"格式提醒"能帮助自我纠正**：

当 `parse_react_output` 返回 `parse_error` 时，代码执行：
```python
prompt += f"\n（格式错误。请用 Thought:/Action:/Final Answer: 格式重新回答。）\n"
```

这把"你刚才格式错了，正确格式是 XXX"**反馈给模型**。模型在下一轮输出时，会看到这个提醒，从而调整自己的输出格式。这与第06章的"工具异常自我纠正"一脉相承——**错误不是终点，而是新的"观察"**，驱动 Agent 调整下一步。

---

## 练习 4（思考）：显式 vs 隐式 ReAct 的选择

一个电商公司要建一个客服 Agent，需要处理退货、查物流、推荐商品等任务。日均请求量 10 万次。

**问题**：
1. 你会选显式 ReAct 还是隐式 ReAct？为什么？
2. 在什么情况下，你会切换到另一种？
3. 如果客服回复出错需要排查，两种范式分别怎么定位问题？

**参考答案**：

1. **选隐式 ReAct（tools API）**。原因：
   - 10 万次/日 = 成本敏感，显式 ReAct 的 Thought 占额外输出 token，成本更高
   - 生产环境需要稳定性，tools API 的 JSON Schema 约束比正则解析更可靠
   - 现代 LLM（GPT-4o、Claude）都支持 tools API，兼容性不是问题
   - 结构化输出更容易接入下游系统（工单系统、CRM 等）

2. **切换到显式 ReAct 的情况**：
   - **合规审计要求**：金融/医疗行业要求记录 Agent 的完整推理过程
   - **调试困难**：隐式 ReAct 出错时无法看推理，切到显式版排查 Thought
   - **使用不支持 tools API 的模型**：某些开源模型（如老版本 Llama）不支持 function calling
   - **教学/演示**：需要让学习者看清推理过程时

3. **问题排查方式**：
   - **隐式 ReAct**：只能看 `tool_calls`（调了什么工具、传了什么参数）和最终 `content`。推理过程是黑盒，只能通过"输入→输出"对比来推测哪里出了问题。如果模型调了错误的工具，你不知道它"为什么"这么选。
   - **显式 ReAct**：可以看完整 `Thought` 轨迹。如果模型调了错误工具，你能看到它的推理过程（如"Thought: 用户说退货，但我不确定是退货还是换货，先试试 cancel_order"），精确定位逻辑偏差。

> 💡 **实践建议**：生产用隐式 ReAct（稳定+省钱），但保留一个"显式 ReAct 模式"开关用于调试和审计。很多框架（如 LangChain）的 `return_intermediate_steps` 选项就是这个思路。

---

## 练习 5（进阶）：实现一个混合 ReAct（带推理日志的隐式 ReAct）

思考题：现代框架（如 OpenAI 的 o1/o3 系列、Claude 的 extended thinking）提供了"推理日志"功能——模型在 tools API 基础上，额外返回它的推理过程。

**问题**：
1. 这种"带推理日志的隐式 ReAct"结合了显式和隐式的哪些优点？
2. 它和纯显式 ReAct 有什么本质区别？
3. 如果让你设计 API 接口，你会怎么定义这个"推理日志"字段？

**参考答案**：

1. **结合的优点**：
   - 隐式的优点：结构化 `tool_calls` 输出（稳定、易解析、不依赖正则）
   - 显式的优点：推理过程可见（可调试、可审计）
   - 即"**结构化的稳定性 + 推理的透明性**"——鱼和熊掌兼得

2. **本质区别**：
   - 纯显式 ReAct：推理（Thought）和行动（Action）**耦合在同一文本里**，你需要正则分离它们。模型如果不遵循格式，整个链条断裂。
   - 带推理日志的隐式 ReAct：推理和行动**分离在不同字段**——推理在 `reasoning` / `thinking` 字段（纯文本），行动在 `tool_calls` 字段（结构化 JSON）。两者独立，互不干扰。
   - 本质区别是**耦合 vs 解耦**：显式 ReAct 把推理和行动"塞"在一个文本块里（耦合），现代方案把它们"分"在不同字段（解耦）。

3. **API 接口设计**：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "reasoning": "用户问北京和上海哪个更热。我需要分别查两个城市的温度，然后比较。先查北京。",  // 推理日志
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"北京\"}"
        }
      }]
    }
  }]
}
```

关键设计点：
- `reasoning` 是**可选**字段（不是所有场景都需要推理日志，关掉可以省钱）
- `reasoning` 是**纯文本**（不需要结构化，它是给人类看的）
- `tool_calls` 保持**结构化**（给代码解析的）
- 两者**独立**（推理错了不影响 tool_calls 的结构，tool_calls 格式不对也不影响推理文本）

> 💡 **这正是行业演进方向**：从"显式 ReAct"（文本耦合）→ "隐式 ReAct"（结构化但黑盒）→ "带推理日志的隐式 ReAct"（结构化 + 透明）。理解这个演进，你就能理解所有现代 Agent 框架的设计逻辑。
