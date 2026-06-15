# AI Agent 开发速查表

> 本表汇总教程 17 章的核心 API 和模式，A4 打印友好。

---

## 1. LLM API 调用

```python
# OpenAI 兼容（Python）
from openai import OpenAI
client = OpenAI(api_key="sk-xxx")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是任务助手"},
        {"role": "user",   "content": "今天待办？"}
    ],
    temperature=0.7,
)
answer = response.choices[0].message.content
```

```typescript
// TypeScript
const client = new OpenAI({ apiKey: "sk-xxx" });
const response = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "今天待办？" }],
});
```

**角色速查**：`system`（定义人格）、`user`（用户输入）、`assistant`（模型回答）

**Temperature**：`0.0`=确定性（代码/分类）、`0.7`=创意、`1.0`=高随机

**流式响应**：`stream=True`，用 `chunk.choices[0].delta.content` 拼接

---

## 2. 工具调用（Function Calling）

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市当前天气",  # 越清晰越好
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
```

**4 步流程**：发送 tools → 收到 tool_calls → 执行函数 → 以 `role="tool"` 反馈结果

```python
TOOL_FUNCTIONS = {"get_weather": get_weather, "calculate": calculate}

for tc in response.choices[0].message.tool_calls:
    func = TOOL_FUNCTIONS[tc.function.name]
    args = json.loads(tc.function.arguments)
    result = func(**args)
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
```

**原则**：工具 3-5 个最佳，描述要具体，结果要截断后再反馈

---

## 3. Agent 循环

```python
MAX_STEPS = 10  # ⚠️ 必须有上限

def agent_loop(user_msg, tools, tool_funcs):
    messages = [{"role": "system", "content": "..."}, {"role": "user", "content": user_msg}]
    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(model=cfg.model, messages=messages, tools=tools)
        msg = response.choices[0].message
        if not msg.tool_calls:       # 终止条件 1：任务完成
            return msg.content
        messages.append(msg)
        for tc in msg.tool_calls:    # 执行工具
            result = tool_funcs[tc.function.name](**json.loads(tc.function.arguments))
            messages.append({"role": "tool", "content": str(result)})
    return "(达到最大步数)"           # 终止条件 2：保险丝
```

**max_steps 经验值**：简单 5，中等 10，复杂 20-30。永远不用 `float('inf')`

---

## 4. ReAct 模式

```
Thought: 我需要查北京温度。
Action: get_weather[北京]
Observation: 北京晴, 25°C
Thought: 信息足够。
Final Answer: 北京今天晴，25°C。
```

| 维度 | 显式（文本解析） | 隐式（tools API） |
|------|------------------|-------------------|
| 推理可见 | ✅ 明文 Thought | ❌ 黑盒 |
| 健壮性 | ⚠️ 可能不遵循格式 | ✅ JSON Schema |
| 成本 | ⚠️ Thought 占 token | ✅ 无开销 |
| 场景 | 教学/调试 | 生产 |

**Prompt 模板**：指令 + Few-shot 示例 + 格式约束（`Thought:`/`Action:`/`Final Answer:`）

**正则解析**：`re.search(r'Thought:\s*(.*?)\nAction:\s*(\w+)\[(.*?)\]', text, re.DOTALL)`

---

## 5. 错误处理

| 错误类型 | 可重试？ | 处理 |
|----------|----------|------|
| `APITimeoutError` | ✅ | 退避重试 |
| `RateLimitError` (429) | ✅ | 退避重试 |
| `AuthenticationError` (401) | ❌ | 立即退出 |
| `BadRequestError` (400) | ❌ | 立即退出 |
| 工具异常 | ⚠️ | 反馈给 Agent |

**指数退避**：

```python
for attempt in range(MAX_RETRIES := 3):
    try:
        return client.chat.completions.create(...)
    except (APITimeoutError, RateLimitError) as e:
        if attempt == MAX_RETRIES - 1: raise
        time.sleep(2 ** attempt)  # 1s, 2s, 4s
```

**工具异常自我纠正**：

```python
try:
    result = func(**args)
except Exception as e:
    result = f"[工具失败] {type(e).__name__}: {e}"  # 反馈给 Agent
messages.append({"role": "tool", "content": str(result)})
```

**幻觉工具检测**：先检查 `func_name in VALID_TOOLS`，不存在就告知正确列表

---

## 6. 记忆管理

**短期记忆**：`messages` 数组就是记忆，循环外初始化，循环内只 append

**滑动窗口**：保留最近 N 条历史

```python
messages = [messages[0]] + messages[-20:]  # 保留 system + 最近 20 条
```

**摘要压缩**：

```python
summary = client.chat.completions.create(
    model=cfg.model,
    messages=[{"role": "system", "content": "用 100 字总结对话"}, *messages],
).choices[0].message.content
messages = [{"role": "system", "content": f"摘要：{summary}"}]
```

**长期记忆**：用 embedding + 向量数据库存储/检索

---

## 7. 评估与可观测

**成本计算**：

```python
cost = usage.prompt_tokens * price["input"] / 1_000_000 \
     + usage.completion_tokens * price["output"] / 1_000_000
```

| 模型 | 输入 | 输出 | 100 轮约 |
|------|------|------|----------|
| gpt-4o-mini | $0.15/1M | $0.60/1M | ~$0.05 |
| gpt-4o | $2.50/1M | $10/1M | ~$1 |

**Tracing 结构**：记录每步的 `step`、`action`、`result`、`duration_ms`

**行为测试**：测试错误恢复、max_steps 生效、工具选择正确性

---

## 8. 反模式速查

| 反模式 | 正确做法 |
|--------|----------|
| 无 `max_steps` | 设 10-30 |
| 裸 `except` 吞错误 | 捕获具体异常，反馈给 Agent |
| 工具描述模糊 | 写清功能和场景 |
| 每步重建 messages | 循环外初始化，循环内 append |
| 不区分可重试/永久错误 | 用 `is_retryable()` 判断 |
| 混用显式/隐式 ReAct | 二选一 |
| 工具过多（>5） | 控制 3-5 个 |

---

## 9. 生产检查清单

- [ ] `max_steps` 设置（10-30）
- [ ] 单步超时（30s）
- [ ] 指数退避重试（max_retries=3）
- [ ] 区分可重试/永久错误
- [ ] 工具异常反馈（不崩溃）
- [ ] 幻觉工具名检测
- [ ] 输入/输出长度限制
- [ ] Token 用量监控
- [ ] 成本告警阈值

---

> 教程：`ai-agent/` | 17 章 + 4 项目 | Python + TypeScript
