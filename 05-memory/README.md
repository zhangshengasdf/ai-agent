# 第05章 记忆系统（Memory Systems）

> **「任务助手 Agent」获得了"记忆力"**——第04章的 Agent 循环每次启动都"失忆"，
> 本章让它能在多轮对话中记住上下文、跨会话记住用户偏好、在海量知识中检索相关信息。

---

## 本章目标

学完本章，你将理解：

1. **LLM 为什么无状态**：每次 API 调用是独立的，模型本身没有"记忆"
2. **短期记忆（ConversationBuffer）**：完整保留对话历史，最简单最直接
3. **长期记忆（VectorMemory）**：用向量相似度检索，按相关性召回而非按时间
4. **摘要记忆（SummaryMemory）**：超阈值时压缩历史，平衡上下文长度与信息密度
5. **记忆的增删查操作**：所有记忆系统共有的 CRUD 接口
6. **上下文窗口管理与截断策略**：token 预算有限，如何取舍
7. **三大反模式**：全塞进上下文、把向量检索当完美召回、无截断策略

---

## 核心认知：LLM 是无状态的

这是理解所有记忆系统的前提。

> **LLM 的每次 API 调用是完全独立的**——模型本身不存储任何上一次对话的信息。

当你调用：

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "我叫小明"}],
)
```

下一次调用：

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "我叫什么？"}],
)
```

模型**完全不知道**你叫小明。它只会回答"我不知道你的名字"。

**所谓"对话连续性"，是你（开发者）在 `messages` 列表里把历史传回去实现的，不是模型自己记的。**

```python
# 这才是连续对话的正确方式——你负责把历史传回去
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "我叫小明"},
        {"role": "assistant", "content": "你好，小明！"},
        {"role": "user", "content": "我叫什么？"},  # 模型看历史才知道
    ],
)
```

> 💡 **记忆系统 = 你替模型管理"历史"的机制。** 模型永远只看当前传入的 `messages`，记忆系统决定"传哪些历史、怎么传、传多少"。

---

## 三种记忆模式总览

| 记忆类型 | 原理 | 何时用 | 成本 | 召回质量 |
|----------|------|--------|------|----------|
| **ConversationBuffer**（对话缓冲） | 完整保留所有消息，按时间顺序 | 短对话（<20 轮） | 低（纯内存） | 完美（全量） |
| **SummaryMemory**（摘要记忆） | 超阈值时让模型摘要压缩旧历史 | 中等对话（20-100 轮） | 中（调 LLM 摘要） | 中（有损压缩） |
| **VectorMemory**（向量记忆） | 把消息转向量，按相似度检索 | 长期记忆 / 跨会话 / 大知识库 | 高（embedding + 检索） | 高（按相关性） |

**一个真实的 Agent 通常组合使用**：当前对话用 Buffer，历史对话用 Vector，超长时用 Summary 压缩。

---

## 1. 短期记忆：ConversationBuffer

最简单也最常用：**把每条消息存进一个列表，下次调 API 时全量传回去。**

```python
class ConversationBuffer:
    def __init__(self):
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return list(self._messages)  # 返回副本，防外部修改

    def clear(self) -> None:
        self._messages.clear()
```

### 优点
- **实现极简**：一个列表搞定
- **无信息损失**：模型能看到完整历史，推理质量最高
- **延迟低**：无需额外 API 调用（不像摘要记忆要调 LLM 压缩）

### 致命缺点：上下文窗口有限
- GPT-4o-mini 上下文 128K tokens ≈ 约 50 页文本
- 一旦历史超出窗口，**API 会报错或静默截断**（取决于 SDK）
- **token 费用**：每轮都传完整历史，成本随对话长度**二次增长**（O(n²)）

> ⚠️ **一个 50 轮的对话，第 50 轮时你要传前 49 轮的全部 token。** 这是 Buffer 的天花板。

---

## 2. 摘要记忆：SummaryMemory

当 Buffer 太长时，**让模型把旧历史压缩成一段摘要**，只保留最近几轮原文 + 旧历史的摘要。

```
[旧摘要] + [最近 N 轮原文] → 传给 API
```

### 触发机制
- 设一个阈值（如 `max_messages=6`）
- 当消息数超过阈值，把**最早的一批**消息送去摘要
- 摘要结果作为一条 `system` 或 `assistant` 消息，替换掉那批原文

```python
class SummaryMemory:
    def __init__(self, max_messages: int = 6):
        self._max = max_messages
        self._messages: list[dict] = []
        self._summary: str = ""  # 累积的旧对话摘要

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._max:
            self._summarize_oldest()

    def _summarize_oldest(self) -> None:
        """把最早 2 条消息送去摘要，压缩进 self._summary。"""
        to_summarize = self._messages[:2]
        self._messages = self._messages[2:]
        # 实际调用 LLM 压缩（或离线 mock）
        chunk_text = "\n".join(m["content"] for m in to_summarize)
        self._summary += f"\n{self._llm_summarize(chunk_text)}"
```

### 优点
- **token 成本可控**：历史长度被"压平"，不再二次增长
- **保留语义**：摘要保留了关键信息（如"用户偏好 Python"），而非简单截断

### 缺点
- **有损压缩**：摘要可能丢失细节（如具体的数字、时间）
- **额外 API 调用**：每次摘要要调一次 LLM（延迟 + 成本）
- **摘要质量依赖模型**：弱模型摘要可能跑题

> 💡 **Buffer 与 Summary 的选择**：对话 <20 轮用 Buffer；20-100 轮用 Summary；>100 轮必须上 Vector。

---

## 3. 长期记忆：VectorMemory

当你需要**跨会话记住**用户偏好、或在海量知识库里**按相关性检索**时，按时间顺序的 Buffer/Summary 不够用了——你需要**语义检索**。

### 核心思想
把每条文本转成一个**向量**（embedding），存进向量库。查询时，把 query 也转向量，算它与所有存储向量的**余弦相似度**，返回最相似的 top_k 条。

```
文本 → embedding 模型 → 向量 → 存进库
查询 → embedding 模型 → 向量 → 算相似度 → 返回 top_k
```

### 余弦相似度（纯 Python 实现，不用 numpy）

本章刻意不接 Chroma/Pinecone 等重型向量库，用纯 Python 教学实现：

```python
import math

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """两向量的余弦相似度，范围 [-1, 1]。1 = 完全相同。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

### 教学用 Embedding：词频向量（不调真实 API）

真实 embedding 需要 API（如 OpenAI `text-embedding-3-small`）。本章用**词频向量**模拟——把文本分词，每个词对应向量的一个维度，值为出现次数。

```python
def simple_embedding(text: str) -> list[float]:
    """词频向量模拟 embedding（教学用，非真实语义）。"""
    words = text.lower().split()
    vec: dict[str, float] = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    return vec  # 用 dict 表示稀疏向量
```

**为什么词频向量能"近似"语义检索？** 因为语义相近的文本往往共享关键词。它不如真实 embedding（不懂同义词、不懂上下文），但**教学上足以演示检索流程**，且 100% 离线可运行。

> 📌 **真实项目**：把 `simple_embedding` 换成 `client.embeddings.create(...)` 即可，其余代码不变。本章的离线 mock 就是这个换法的预演。

### VectorMemory 接口

```python
class VectorMemory:
    def __init__(self):
        self._store: list[tuple[str, list[float]]] = []  # [(text, embedding)]

    def add(self, text: str) -> None:
        emb = simple_embedding(text)
        self._store.append((text, emb))

    def search(self, query: str, top_k: int = 3) -> list[str]:
        q_emb = simple_embedding(query)
        scored = [(text, cosine_similarity_dict(q_emb, emb))
                  for text, emb in self._store]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scored[:top_k]]
```

> ⚠️ 本章的"向量库"是内存列表，**不持久化到磁盘**。真实项目用 Chroma/Pinecone/pgvector——那是第09章和实战项目的事。

---

## 记忆的增删查操作（CRUD）

所有记忆系统都应实现这四个操作（本章三个实现各自覆盖一部分）：

| 操作 | 方法 | Buffer | Summary | Vector |
|------|------|--------|---------|--------|
| **增**（Create） | `add(role, content)` | ✓ | ✓ | ✓ |
| **查**（Read） | `get_messages()` / `search()` | ✓（全量） | ✓（摘要+近期） | ✓（top_k） |
| **删**（Delete） | `clear()` | ✓ | ✓ | ✓ |
| **改**（Update） | 较少用 | — | — | 可重 add |

**记忆不是只读的**——用户说"我改主意了，不用北京用上海"时，Agent 需要能更新或删除旧记忆。本章聚焦增/查，删用 `clear()`，改留作练习。

---

## 上下文窗口管理与截断策略

当记忆超出上下文窗口时，**必须有截断策略**，否则 API 报错或静默丢消息。

### 策略 1：FIFO 截断（最简单）
```python
def truncate(messages: list[dict], max_messages: int = 20) -> list[dict]:
    """保留 system + 最近 N 条。"""
    if messages[0]["role"] == "system":
        return [messages[0]] + messages[-(max_messages - 1):]
    return messages[-max_messages:]
```
**缺点**：丢掉最旧的消息，可能丢失关键信息（如用户名）。

### 策略 2：摘要压缩（SummaryMemory 做的）
把旧消息压缩成摘要，不直接丢。比 FIFO 好，但有 API 成本。

### 策略 3：向量检索召回（VectorMemory 做的）
不按时间，按相关性召回 top_k 条塞进上下文。适合知识库场景。

### 策略 4：滑窗 + 固定锚点
保留最近 N 条 + 几条"永不丢弃"的锚点（如 system prompt、用户名）。

```python
def smart_truncate(messages, recent=10, anchors=None):
    anchors = anchors or []
    # 永远保留 anchors（如用户偏好），加最近 recent 条
    keep = [m for m in messages if m["role"] == "system"]
    recent_msgs = [m for m in messages if m["role"] != "system"][-recent:]
    return keep + recent_msgs
```

> 💡 **没有万能策略**。对话场景用 Summary，知识库场景用 Vector，简单场景用 FIFO。组合使用最常见。

---

## 反模式（什么不该做）

### ❌ 把所有历史塞进上下文（上下文过载）

```python
# 坏：100 轮对话全传，token 成本爆炸 + 检索精度下降
messages = all_100_messages  # 可能 50K tokens，每轮都付费
response = client.chat.completions.create(messages=messages, ...)
```

**后果**：
1. **成本爆炸**：每轮传 50K tokens，100 轮 = 500 万 tokens 费用
2. **检索精度下降**：上下文越长，模型越容易"迷失在中间"（lost in the middle 现象），忽略关键信息
3. **延迟增大**：输入越长，首 token 延迟越高

**正确**：用 SummaryMemory 或 VectorMemory 控制上下文长度，只传"当前需要"的部分。

### ❌ 把向量检索当完美召回

```python
# 坏：以为向量检索 = 100% 召回相关内容
results = vector_memory.search("用户偏好", top_k=3)
# 直接用 results，不做验证
```

**后果**：
1. **召回不全**：向量检索基于语义相似度，相似 ≠ 相关。可能召回"语义像但意思不同"的内容
2. **漏掉精确匹配**：向量检索对关键词（如人名、产品号）不如全文检索精准
3. **embedding 漂移**：不同 embedding 模型的"相似"定义不同，跨模型比较会失真

**正确**：**混合检索**（向量 + 关键词），用向量召回候选集，再用精确匹配/重排序过滤。这是第09章 RAG 的核心。

### ❌ 无截断策略（指望 API 自己处理）

```python
# 坏：不管理长度，超了就超了
messages = grow_forever(...)  # 最终会超窗口
response = client.chat.completions.create(messages=messages, ...)
# 可能报错，也可能静默截断——你不知道丢了什么
```

**后果**：
- **硬报错**：超窗口时 API 返回 400（context_length_exceeded）
- **静默截断**：某些 SDK 自动截断最旧消息，你以为模型"记得"，其实丢了

**正确**：主动用上述策略 1-4 管理长度，**绝不让记忆无限增长**。

---

## 运行示例

```bash
# Python
cd ai-agent/05-memory
python3 python/main.py

# TypeScript
cd ai-agent/05-memory
npx tsx typescript/main.ts
```

输出用 `OUT:buffer:` / `OUT:summary:` / `OUT:vector:` 前缀标记三个 demo，方便 QA 脚本 grep 过滤 dotenvx 横幅。

### 离线可运行（关键设计）

本章三个 demo 的设计：
- **Demo 1 (Buffer)**：纯内存操作，不调 API，100% 离线
- **Demo 2 (Summary)**：先尝试真实 API 摘要，失败时降级为预设 mock 摘要，保证演示完整
- **Demo 3 (Vector)**：纯 Python 词频向量 + 余弦相似度，不调 embedding API，100% 离线

所以即使 `.env` 是占位符 `OPENAI_API_KEY=sk-REPLACE-ME`，三个 demo 都能完整跑通，exit 0。

---

## 与第04章的关系

第04章的 Agent 循环里，`messages` 列表在循环外初始化、循环内追加——那其实就是**最原始的 ConversationBuffer**。只是它没有"持久化"和"管理长度"的能力，每次 `agent_loop` 启动就重置。

本章给 `messages` 加上：
1. **独立成类**（ConversationBuffer）：可复用、可测试
2. **长度管理**（SummaryMemory）：超阈值自动压缩
3. **语义检索**（VectorMemory）：跨会话、跨任务检索

> 💡 **下一步**：把 SummaryMemory 或 VectorMemory 接到第04章的 `agent_loop` 里，Agent 就有了"跨会话记忆"。本章练习题就是这个。

---

## 下一步

本章你让「任务助手 Agent」获得了**记忆力**。但 Agent 跑起来后，**工具调用会失败、API 会超时、模型会返回垃圾**——第06章「错误处理与重试」解决这些生产级问题。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
