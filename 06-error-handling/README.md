# 第06章 错误处理与容错（Error Handling & Resilience）

> **「任务助手 Agent」获得了"韧性"**——从第04章的"能跑就行"，进化为"出错也能优雅恢复"。
> 一个生产级 Agent 与玩具 Demo 的分水岭，不在于"功能多"，而在于**出错时会发生什么**。

---

## 本章目标

学完本章，你将理解：

1. **为什么错误处理要在早期教**：失败是系统设计问题，不是事后补丁
2. **四大容错机制**：指数退避重试、工具异常自我纠正、幻觉工具检测、错误分类
3. **降级策略**：工具失败时不是崩溃，而是反馈给 Agent 让它换思路
4. **护栏基础**：输入/输出校验，把"垃圾进垃圾出"挡在门外
5. **反模式**：裸 `except` 吞错误、无上限重试、不区分错误类型

---

## 为什么错误处理要在早期教

很多教程把"错误处理"放到最后一章，当作"高级技巧"。这是个昂贵的误解。

**Anthropic 的工程共识是：失败是系统设计问题，不是事后补丁。**

原因有三：

1. **Agent 比普通程序更容易出错**。Agent 由 LLM + 工具 + 网络 + 外部数据源组成，链条上任何一环都可能失败：API 限流、工具超时、模型幻觉一个不存在的工具名、JSON 解析失败……如果你等到上线才处理，每个错误都会变成用户投诉。

2. **早期不教，会养成"裸 except"习惯**。新手写 Agent 最常见的"反模式"就是 `try: ... except: pass`——把所有错误吞掉。结果 Agent 在生产环境静默失败，你连日志都没有。这种习惯一旦养成，很难纠正。

3. **容错机制是 Agent"可用性"的核心**。一个"功能 100 分但一出错就崩"的 Agent，实际可用性远低于"功能 70 分但出错能恢复"的 Agent。后者才是用户愿意用的。

> 💡 **心智模型**：把 Agent 想象成一个实习生。好实习生不是"从不犯错"（那不可能），而是"犯了错知道怎么补救"——重试一次、换个方法、或者诚实报告"我搞不定"。本章教你如何让 Agent 具备这种"补救能力"。

---

## 常见失败模式

在写容错代码之前，先认清 Agent 会以哪些方式失败。对症下药，才能写对 `except`。

| 失败模式 | 典型表现 | 根本原因 | 可重试？ |
|----------|----------|----------|----------|
| **API 超时** | `APITimeoutError` / 连接超时 | 网络抖动、服务端慢 | ✅ 是 |
| **API 限流** | `RateLimitError`（HTTP 429） | 短时间请求过多 | ✅ 是（等一会儿） |
| **连接错误** | `APIConnectionError` | DNS 失败、网络断 | ✅ 是 |
| **认证错误** | `AuthenticationError`（HTTP 401） | 密钥无效/过期 | ❌ 否（重试也没用） |
| **参数错误** | `BadRequestError`（HTTP 400） | 请求格式错、模型不支持 | ❌ 否（得改代码） |
| **工具异常** | 工具函数抛 `ValueError`/`KeyError` | 输入不合法、外部服务挂 | ⚠️ 看 Agent 决策 |
| **JSON 解析失败** | `json.JSONDecodeError` | 模型返回的 `arguments` 不是合法 JSON | ⚠️ 可重试（让模型重新生成） |
| **幻觉工具调用** | 模型调了不存在的工具名 | 模型"编造"了一个工具 | ⚠️ 告知后可纠正 |

**关键区分：可重试错误 vs 永久错误。**

- **可重试错误**（超时、限流、连接错误）：再试一次可能成功 → 重试 + 退避
- **永久错误**（认证、参数错误）：再试 100 次也一样 → 立即失败，别浪费时间

把这两类混为一谈，是新手最常见的错误：要么"什么错误都重试"（认证错误重试 3 次纯属浪费），要么"什么错误都不重试"（一次网络抖动就让 Agent 挂掉）。

---

## 机制 1：指数退避重试（Exponential Backoff）

面对可重试错误（超时、限流、连接错误），正确做法不是"立刻重试"，而是**等一会儿再试，每次等更久**。

### 为什么不能"立刻重试"

如果 API 限流是因为"你请求太频繁"，立刻重试只会让限流更严重。你需要**退让**，让服务端喘口气。

### 指数退避的公式

```
attempt 0: 失败 → 等 2^0 = 1 秒
attempt 1: 失败 → 等 2^1 = 2 秒
attempt 2: 失败 → 等 2^2 = 4 秒
attempt 3: 放弃（达到 max_retries=3）
```

每次等待时间**翻倍**——这就是"指数"的含义。它比"固定间隔"更智能：前几次快速重试（也许只是瞬时抖动），后面拉长间隔（给服务端恢复时间）。

### 实现代码骨架

```python
import time

MAX_RETRIES = 3

def call_llm_with_retry(messages, tools):
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=cfg.model, messages=messages, tools=tools,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            # 只重试"可重试错误"
            if attempt == MAX_RETRIES - 1:
                raise  # 最后一次也失败了，抛出去
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"第 {attempt+1} 次失败，{wait}s 后重试：{e}")
            time.sleep(wait)
```

> ⚠️ **只重试可重试错误**：`except` 子句必须精确列出可重试的异常类型。绝不能写裸 `except Exception`——那会把认证错误也重试，纯属浪费。

### 生产级增强（了解即可）

真实生产环境还会加：

- **抖动（jitter）**：在等待时间上加随机量（如 `wait + random()`），避免多个客户端同时重试造成"惊群"
- **上限封顶（cap）**：`min(wait, 60)` 防止等待时间无限增长
- **遵守 `Retry-After` 头**：HTTP 429 响应常带这个头，告诉你"多久后再试"

本章先用最朴素的 `2 ** attempt`，这些增强留到第15章讲。

---

## 机制 2：工具异常 + Agent 自我纠正

工具执行时抛异常，不应该让整个 Agent 崩溃。正确做法是：**把错误信息以 `role="tool"` 追加到 messages，让 Agent 自己决定怎么办**。

### 为什么"告诉 Agent 出错了"比"崩溃"好

Agent 是有推理能力的。如果你告诉它"`get_weather` 失败了：城市名拼写错误"，它能：

1. **换工具**：既然天气查不到，改用 `search_wiki` 查这个城市
2. **调整参数**：发现"Beijing"拼错了，改成"北京"再试
3. **诚实回答**：告诉用户"我查不到这个城市的天气"

这就是 **Agent 自我纠正（self-correction）**——错误不是终点，而是新的"观察"，驱动 Agent 调整下一步。

### 实现代码骨架

```python
for tc in assistant_msg.tool_calls:
    func_name = tc.function.name
    args = json.loads(tc.function.arguments)
    func = TOOL_FUNCTIONS.get(func_name)

    try:
        result = func(**args)
    except Exception as e:
        # ❌ 不要崩溃，也不要吞掉错误
        # ✅ 把错误"翻译"成 Agent 能理解的语言
        result = f"[工具执行失败] {func_name} 抛出异常：{type(e).__name__}: {e}"

    # 无论成功还是失败，都把结果反馈给 Agent
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": str(result),
    })
```

下一轮循环，Agent 会"看到"这条错误信息，然后重新决策。这比"工具一抛异常就整个崩溃"优雅得多。

> 💡 **错误信息要"可操作"**。不要只写 `"工具失败"`，要写 `"get_weather 抛出 ValueError：城市名 'xyz' 不在数据库中。可用的城市有：北京、上海、深圳"`。信息越具体，Agent 越容易纠正。

---

## 机制 3：幻觉工具名检测

LLM 有时会"编造"一个不存在的工具名——这在业内叫**幻觉（hallucination）**。

比如你只提供了 `get_weather`、`calculate`、`search_wiki` 三个工具，模型却调了 `get_stock_price`（你以为它知道，其实它在瞎编）。

### 检测 + 纠正

```python
VALID_TOOL_NAMES = set(TOOL_FUNCTIONS.keys())  # {"get_weather", "calculate", "search_wiki"}

for tc in assistant_msg.tool_calls:
    func_name = tc.function.name
    if func_name not in VALID_TOOL_NAMES:
        # 幻觉！告知 Agent 正确的工具列表
        result = (
            f"[错误] 工具 '{func_name}' 不存在。"
            f"可用的工具有：{', '.join(sorted(VALID_TOOL_NAMES))}。"
            f"请从上述列表中选择一个。"
        )
        messages.append({
            "role": "tool", "tool_call_id": tc.id, "content": result,
        })
        continue
    # ... 正常执行 ...
```

Agent 看到"这个工具不存在，这是正确的工具列表"后，通常会在下一步选一个合法的工具。这就是**用反馈而非崩溃来纠正幻觉**。

---

## 机制 4：区分可重试错误 vs 永久错误

这是本章的"元技能"——它决定了前三个机制如何组合。

### 分类逻辑

```python
from openai import (
    APITimeoutError, APIConnectionError, RateLimitError,
    AuthenticationError, BadRequestError, APIError,
)

def is_retryable(error: Exception) -> bool:
    """判断一个错误是否值得重试。"""
    if isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True   # 瞬时故障，再试可能成功
    if isinstance(error, APIError) and getattr(error, "status_code", 0) >= 500:
        return True   # 服务端 5xx 错误，通常瞬时
    return False      # 认证错误、参数错误等 → 永久错误

# 使用：
try:
    response = call_with_retry(...)  # 内部只重试 is_retryable 的
except AuthenticationError:
    raise SystemExit("密钥无效，请检查 .env")  # 永久错误，直接退出
except BadRequestError as e:
    raise SystemExit(f"请求格式错误：{e}")      # 永久错误，得改代码
```

### 错误分类表（快速参考）

| 错误类型 | 可重试？ | 处理方式 |
|----------|----------|----------|
| `APITimeoutError` | ✅ | 退避重试 |
| `APIConnectionError` | ✅ | 退避重试 |
| `RateLimitError`（429） | ✅ | 退避重试（等久点） |
| 服务端 `APIError`（5xx） | ✅ | 退避重试 |
| `AuthenticationError`（401） | ❌ | 立即退出，提示检查密钥 |
| `BadRequestError`（400） | ❌ | 立即退出，提示改请求 |
| 工具异常 | ⚠️ | 反馈给 Agent（机制 2） |
| `JSONDecodeError` | ⚠️ | 反馈给 Agent 让它重生成 |

---

## 降级策略（Graceful Degradation）

"降级"指：当首选方案失败时，退而求其次，而不是崩溃。

在 Agent 语境下，降级有三种层次：

### 层次 1：工具级降级（反馈给 Agent）

工具 A 失败 → 把错误告诉 Agent → Agent 改用工具 B。这是机制 2 的延伸——**Agent 自己做降级决策**。

```
Agent: 调 get_weather("火星")
工具:  [失败] 火星不在数据库
Agent: 那我改用 search_wiki("火星") 查点信息
工具:  火星是太阳系第四颗行星...
Agent: 我查不到火星的天气，但查到了百科信息：...
```

### 层次 2：API 级降级（换模型）

主模型（如 GPT-4o）API 挂了 → 切换到备用模型（如 deepseek-chat）。本章不深入，留到第15章可观测性讲。

### 层次 3：功能级降级（拒绝服务）

所有重试失败 → 返回"抱歉，服务暂时不可用"，而不是让用户对着一个转圈圈等到天荒地老。

> 💡 **降级的核心原则**：永远给用户一个"可接受的次优结果"，而不是"没有结果"。一个说"我现在查不到，但你可以试试 XXX"的 Agent，远好过一个卡死的 Agent。

---

## 护栏基础（Guardrails）

护栏指"在输入和输出两端做校验，把不合法的内容挡在门外"。本章只讲**基础**（输入/输出校验），深度安全护栏留到第17章。

### 输入护栏

在把用户输入交给 Agent 之前，先校验：

- **长度限制**：防止超长输入耗尽 token 预算
- **字符过滤**：过滤明显的注入攻击（如"忽略以上所有指令"）
- **敏感词检测**：拦截违法/有害内容

```python
def validate_input(user_message: str) -> str:
    if len(user_message) > 10_000:
        raise ValueError("输入过长（超过 10000 字符），请精简后重试")
    if "ignore previous instructions" in user_message.lower():
        raise ValueError("检测到疑似 prompt 注入，已拒绝")
    return user_message
```

### 输出护栏

在把 Agent 的回答返回给用户之前，先校验：

- **格式校验**：如果要求 JSON，检查是否真的是合法 JSON
- **内容校验**：过滤有害/不当内容
- **长度截断**：防止 Agent 啰嗦到 token 爆表

```python
def validate_output(answer: str) -> str:
    if len(answer) > 5000:
        return answer[:5000] + "\n\n（回答过长，已截断）"
    return answer
```

> ⚠️ **本章只讲基础**。真正的安全护栏（prompt 注入防御、越狱检测、PII 脱敏）是第17章的主题，涉及更复杂的攻防。这里只建立"校验"的意识。

---

## 反模式（什么不该做）

### ❌ 裸 `except` 吞错误

```python
# 坏：吞掉所有错误，连日志都没有
try:
    result = tool(**args)
except:
    result = None  # 然后呢？Agent 不知道出错了，继续往下走
```

**后果**：Agent 静默失败，调试时你完全不知道哪里错了。

**正确**：捕获**具体**异常类型，并把错误信息反馈给 Agent（机制 2）。

### ❌ 重试无上限

```python
# 坏：无限重试，万一永久错误就死循环
while True:
    try:
        return call_api()
    except Exception:
        continue  # 永远不放弃...但也永远不成功
```

**后果**：认证错误也会无限重试，浪费资源 + 可能触发更严厉的封禁。

**正确**：`max_retries=3`，最后一次失败就抛出去。

### ❌ 不区分可重试错误与永久错误

```python
# 坏：对 AuthenticationError 也重试 3 次，纯属浪费
try:
    call_api()
except Exception as e:
    for _ in range(3):
        try:
            call_api()
        except Exception:
            pass
```

**正确**：用 `is_retryable(error)` 判断，只重试瞬时错误。

### ❌ 工具异常直接崩溃

```python
# 坏：工具一抛异常，整个 Agent 循环崩溃
result = func(**args)  # 万一这里抛异常？
messages.append({"role": "tool", "content": result})
```

**正确**：`try/except` 包住工具执行，把错误反馈给 Agent（机制 2）。

### ❌ 幻觉工具名当"正常情况"

```python
# 坏：假设模型永远不会调错工具
func = TOOL_FUNCTIONS[func_name]  # KeyError 直接崩
```

**正确**：先检查 `func_name in VALID_TOOL_NAMES`，不存在就告知 Agent（机制 3）。

---

## 本章的四大机制如何组合

这四个机制不是孤立的，它们在 Agent 循环里协同工作：

```
agent_loop():
    for step in range(MAX_STEPS):
        try:
            # 机制 1 + 4：带退避的重试，只针对可重试错误
            response = call_llm_with_retry(messages, tools)
        except AuthenticationError:
            raise  # 永久错误，直接退出（机制 4）

        assistant_msg = response.choices[0].message
        if not assistant_msg.tool_calls:
            return validate_output(assistant_msg.content)  # 输出护栏

        messages.append(assistant_msg)
        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name
            # 机制 3：幻觉工具检测
            if func_name not in VALID_TOOL_NAMES:
                result = f"工具 '{func_name}' 不存在，可用：{...}"
            else:
                # 机制 2：工具异常 → 反馈给 Agent
                try:
                    result = TOOL_FUNCTIONS[func_name](**args)
                except Exception as e:
                    result = f"[工具失败] {e}"
            messages.append({"role": "tool", "content": result, ...})
```

每个机制负责一层防御，层层叠加，Agent 才能在出错时"软着陆"。

---

## 运行示例

```bash
# Python
cd ai-agent/06-error-handling
python3 python/main.py

# TypeScript
cd ai-agent/06-error-handling
npx tsx typescript/main.ts
```

代码会先用真实 API 尝试（占位符密钥会失败），然后**自动降级为离线 mock 演示**，100% 可靠地展示四大容错机制：

- **Demo A**：指数退避重试（mock 一个"前 2 次失败、第 3 次成功"的 API）
- **Demo B**：工具自我纠正（mock `get_weather` 第一次抛异常，Agent 换工具）
- **Demo C**：幻觉工具检测（mock 模型调不存在的工具，告知后纠正）
- **Demo D**：错误分类（展示可重试 vs 永久错误的判断逻辑）

输出用 `OUT:` 前缀标记，方便 grep 过滤 dotenvx 横幅。

---

## 兼容性注意

- **`.env` 是占位符密钥**（`OPENAI_API_KEY=sk-REPLACE-ME`）→ 真实 API 调用会 401。
  代码捕获 `AuthenticationError` 并降级为离线 mock，依然展示完整的容错逻辑。
- **Ollama 不支持 tools API**（返回 400）→ 同样降级为离线 mock。

---

## 下一步

本章你让「任务助手 Agent」获得了**韧性**——它能重试、能纠正、能区分错误、能优雅降级。

但到目前为止，我们的 Agent 都是"一次性"的——每次 `agent_loop` 调用，模型都从零开始，不记得上一次对话。

第07章「ReAct 推理」会让 Agent 学会**显式思考**——把推理过程（Thought → Action → Observation）写出来，而不是隐式地藏在 tools API 里。这让 Agent 的决策更透明、更可控，也为复杂任务（如多步研究）打下基础。

> 💡 **容错 + 推理是 Agent 的两条腿**：容错让它"走得更稳"（不出错能恢复），推理让它"走得更远"（能处理复杂任务）。缺一不可。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
