# 第06章 练习 — 错误处理与容错

> 动手实践"重试、降级、自我纠正、护栏"。每个练习都附参考答案。

---

## 练习 1：故意制造一个会失败的工具（编程题）

这是本章的核心练习。修改 `main.py`（或 `main.ts`），添加一个**故意会失败的工具** `fetch_price`，让 Agent 遇到失败后学会换工具或放弃。

**要求**：
1. 添加工具 `fetch_price(product: str)`，它对 `"咖啡"` 抛 `RuntimeError("价格服务不可用")`，对其他商品返回 `f"{product}: ¥9.9"`
2. 把 `fetch_price` 注册到 `TOOL_FUNCTIONS` 和 `tools` 定义
3. 运行 Agent，问"帮我查咖啡的价格"
4. 观察 Agent 如何处理工具失败（自我纠正）

**Python 参考答案**：

```python
def fetch_price(product: str) -> str:
    """查询商品价格（mock）。对'咖啡'故意抛异常演示自我纠正。"""
    if product == "咖啡":
        raise RuntimeError("价格服务不可用（演示用）")
    return f"{product}: ¥9.9"


# 注册到 TOOL_FUNCTIONS
TOOL_FUNCTIONS["fetch_price"] = fetch_price

# 在 tools 列表添加定义
tools.append({
    "type": "function",
    "function": {
        "name": "fetch_price",
        "description": "查询商品价格",
        "parameters": {
            "type": "object",
            "properties": {"product": {"type": "string", "description": "商品名称"}},
            "required": ["product"],
        },
    },
})
```

**TypeScript 参考答案**：

```typescript
function fetchPrice(product: string): string {
  if (product === "咖啡") {
    throw new Error("价格服务不可用（演示用）");
  }
  return `${product}: ¥9.9`;
}

TOOL_FUNCTIONS["fetch_price"] = (product: string) => fetchPrice(product);

tools.push({
  type: "function",
  function: {
    name: "fetch_price",
    description: "查询商品价格",
    parameters: {
      type: "object",
      properties: { product: { type: "string", description: "商品名称" } },
      required: ["product"],
    },
  },
});
```

**验证**：运行 Agent 问"帮我查咖啡的价格"，预期输出包含：
- `⚠️ 工具异常，反馈给 Agent：RuntimeError: 价格服务不可用`
- Agent 看到 error 后，要么诚实回答"查不到"，要么换 `search_wiki` 查

**关键洞察**：如果工具失败后 Agent 直接崩溃，说明你的循环没有机制 2（工具异常反馈）。正确行为是 Agent 看到 error 信息后自我纠正。

---

## 练习 2：调整退避策略观察重试序列（理解题）

本章 Demo A 用 `backoff_scale=0.1`（Python）/ `100ms`（TS）演示退避序列。

**问题**：
1. Demo A 的退避序列是怎样的？每次等待多久？
2. 如果把 `backoff_scale` 改成 `1.0`（生产值），总等待时间是多少？为什么生产环境要等更久？
3. 如果把 `MAX_RETRIES` 从 3 改成 5，最坏情况下的总等待时间是多少（用 `backoff_scale=1.0`）？

**参考答案**：

1. 退避序列（`backoff_scale=0.1`）：
   | 尝试 | 等待时间 | 公式 |
   |------|----------|------|
   | 第 1 次失败 | 0.1s | `2^0 × 0.1` |
   | 第 2 次失败 | 0.2s | `2^1 × 0.1` |
   | 第 3 次 | 成功 | — |

2. `backoff_scale=1.0` 时：等待 1s + 2s = 3s（第 3 次成功）。生产环境要等更久，因为：
   - API 限流通常需要几秒到几十秒才恢复
   - 立刻重试会加重服务端负担，可能触发更严厉的封禁
   - 指数退避给服务端"喘息"时间

3. `MAX_RETRIES=5` + `backoff_scale=1.0`，最坏情况（5 次全失败）：
   `1 + 2 + 4 + 8 = 15s`（前 4 次失败各等待，第 5 次失败后放弃）
   这意味着一个请求最坏要卡 15 秒——用户可能已经失去耐心。所以 `MAX_RETRIES=3` 是经验平衡点。

---

## 练习 3：实现带抖动的退避（进阶编程题）

本章的退避是确定性的（`2 ** attempt`）。生产环境常加**随机抖动（jitter）**，避免多个客户端同时重试造成"惊群效应"。

**要求**：修改 `call_llm_with_retry`（或 `callLlmWithRetry`），在退避时间上加 `[0, wait)` 的随机量。

**Python 参考答案**：

```python
import random

def call_llm_with_retry(messages, *, tools_list=None, backoff_scale=1.0):
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(...)
        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            if not is_retryable(e):
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            base_wait = 2 ** attempt * backoff_scale
            jitter = random.uniform(0, base_wait)  # 抖动
            total_wait = base_wait + jitter
            print(f"等待 {total_wait:.1f}s（含抖动 {jitter:.1f}s）")
            time.sleep(total_wait)
```

**TypeScript 参考答案**：

```typescript
async function callLlmWithRetry(messages, toolsList?, backoffScaleMs = 1000) {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return await client.chat.completions.create(/* ... */);
    } catch (e) {
      if (!isRetryable(e)) throw e;
      if (attempt === MAX_RETRIES - 1) throw e;
      const baseWaitMs = Math.pow(2, attempt) * backoffScaleMs;
      const jitterMs = Math.random() * baseWaitMs;
      const totalMs = baseWaitMs + jitterMs;
      console.log(`等待 ${(totalMs / 1000).toFixed(1)}s（含抖动）`);
      await sleep(totalMs);
    }
  }
}
```

**为什么抖动重要**：假设 100 个客户端同时遇到限流，如果都用固定退避，它们会在同一时刻重试，再次触发限流。加了抖动后，重试时间分散开，降低"惊群"概率。

---

## 练习 4：识别反模式（思考题）

以下每段代码都有一个**反模式**，指出问题并给出修正。

**代码 A**：
```python
try:
    result = tool(**args)
except:
    pass  # 吞掉错误
```

**代码 B**：
```python
while True:
    try:
        return call_api()
    except Exception:
        continue
```

**代码 C**：
```python
func = TOOL_FUNCTIONS[func_name]  # 直接索引
result = func(**args)
```

**参考答案**：

**代码 A 的问题**：裸 `except` 吞掉所有错误，Agent 不知道出错了，静默失败。
**修正**：捕获具体异常，把错误反馈给 Agent（机制 2）：
```python
try:
    result = tool(**args)
except Exception as e:
    result = f"[工具失败] {type(e).__name__}: {e}"
messages.append({"role": "tool", "tool_call_id": tc_id, "content": result})
```

**代码 B 的问题**：无上限重试。遇到永久错误（如认证失败）会无限循环，浪费资源。
**修正**：加 `max_retries` 上限 + 只重试可重试错误（机制 1 + 4）：
```python
for attempt in range(MAX_RETRIES):
    try:
        return call_api()
    except (APITimeoutError, APIConnectionError, RateLimitError):
        if attempt == MAX_RETRIES - 1:
            raise
        time.sleep(2 ** attempt)
    except AuthenticationError:
        raise  # 永久错误，不重试
```

**代码 C 的问题**：没做幻觉工具名检测。如果 `func_name` 是模型编造的（不在 `TOOL_FUNCTIONS`），`TOOL_FUNCTIONS[func_name]` 直接 `KeyError` 崩溃。
**修正**：先检查是否在合法工具集合（机制 3）：
```python
if func_name not in VALID_TOOL_NAMES:
    result = f"工具 '{func_name}' 不存在，可用：{sorted(VALID_TOOL_NAMES)}"
else:
    result = TOOL_FUNCTIONS[func_name](**args)
```

---

## 练习 5（思考）：容错机制的取舍

本章讲了四大机制。但不是所有 Agent 都需要全部机制。

**问题**：
1. 一个"只查本地数据库"的内部 Agent（无外部 API 调用），需要指数退避重试吗？为什么？
2. 一个"调用 10 个不同外部 API"的研究 Agent，哪个机制最重要？为什么？
3. 护栏（输入/输出校验）和错误处理（重试/纠正）是什么关系？能互相替代吗？

**参考答案**：

1. **不需要退避重试**。本地数据库查询不会"限流"或"超时"（除非数据库本身挂了）。这种 Agent 更需要的是机制 2（工具异常反馈）和机制 3（幻觉检测），而不是机制 1（退避）。**容错机制要按需选择，不是越多越好。**

2. **机制 1（退避重试）最重要**。10 个外部 API，每个都可能超时/限流。没有退避重试，一个 API 抖动就让整个 Agent 失败。其次重要的是机制 2（工具异常反馈），让 Agent 在某个 API 挂掉时能换一个继续。

3. **护栏和错误处理是互补的，不能互相替代**：
   - **护栏**是"预防"——在输入/输出端拦截不合法内容，防止问题发生（如拦截超长输入、过滤注入攻击）
   - **错误处理**是"恢复"——在执行中出错时，让系统优雅恢复（如重试、换工具）
   
   类比：护栏是"安全带 + 安全帽"，错误处理是"急救箱 + 备用方案"。两者都需要——你不能因为戴了安全帽就不带急救箱，也不能因为带了急救箱就不戴安全帽。

> 💡 **核心洞察**：容错不是"把所有机制都加上"，而是"按失败模式对症下药"。先分析你的 Agent 会以哪些方式失败，再决定用哪些机制。盲目堆机制只会增加复杂度，不增加可靠性。
