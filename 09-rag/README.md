# 第09章 RAG 检索（Retrieval-Augmented Generation）

> **「任务助手 Agent」获得了"查资料"的能力**——前几章的 Agent 只能用模型自带的知识回答，
> 本章让它能在**外部知识库**中检索相关信息，再基于检索结果生成准确的回答。
> 这是让 Agent 处理私有数据、最新信息的核心能力。

---

## 本章目标

学完本章，你将理解：

1. **Agent 为何需要外部知识**：模型训练数据有截止日期，无法访问私有数据
2. **Embedding 向量化原理**：文本 → 高维向量，语义相近 → 向量相近
3. **文档分块策略**：chunk_size + overlap，为什么太大太小都不好
4. **基础 RAG 流程**：检索 → 注入上下文 → 回答
5. **Agentic RAG**：Agent 自主决定"是否检索"和"检索什么"
6. **纯 RAG vs Agentic RAG**：什么时候该用哪个
7. **反模式**：分块不当、不做混合检索、塞满无关上下文

---

## 核心认知：Agent 为何需要外部知识

LLM 很强大，但它有**三个硬伤**，让它在很多实战场景中不可直接使用：

### 硬伤 1：训练数据有截止日期

GPT-4o 的训练数据截止到 2023 年 10 月。你问它"今天北京天气怎么样"、"最新的 Python 版本是什么"，
它会诚实地告诉你"我不知道"——或者更糟，**自信地编造一个过时的答案**（幻觉）。

### 硬伤 2：无法访问私有数据

你的公司有内部文档、产品手册、客户数据库、代码仓库。模型从未见过这些数据，你问它"我们公司的退货政策是什么"，它一无所知。

### 硬伤 3：知识是"压缩"在参数里的

模型的"知识"是训练时压缩进神经网络参数里的。这种压缩**不可靠**——模型会遗忘、混淆、张冠李戴。
你问"Python 的 GIL 是什么"，它可能答对，也可能把 GIL 和 GC（垃圾回收）搞混。

### RAG 的解决方案

> **RAG（检索增强生成）= 先从外部知识库检索相关文档，再让模型基于检索结果生成回答。**

```
用户提问 → 检索知识库 → 找到相关文档片段 → 注入到 prompt → 模型基于片段回答
```

这样模型不再依赖参数里的"压缩知识"，而是基于**真实的、可追溯的、可更新的**文档回答。

> 💡 **一句话**：RAG 让 Agent 的回答有了"出处"。不是凭记忆猜，而是查到原文再总结。

---

## Embedding 向量化原理

RAG 能工作的前提是**检索**——从知识库里找到与问题相关的片段。怎么做？用 **Embedding**。

### 什么是 Embedding

Embedding（嵌入）是把一段文本映射成一个**高维向量**（如 1536 维浮点数数组）的过程。

```
"Python 是一门编程语言" → [0.12, -0.34, 0.56, ..., 0.78]  (1536 个数)
"Python 语言简介"       → [0.11, -0.32, 0.55, ..., 0.79]  (与上一句很接近)
"今天天气晴朗"           → [-0.45, 0.23, -0.11, ..., 0.02] (与上两句差很远)
```

**核心性质：语义相近的文本，向量也相近。** 我们用**余弦相似度**（cosine similarity）衡量两个向量的接近程度。

### 余弦相似度

两个向量夹角的余弦值，范围 [-1, 1]：

- **1** = 方向完全相同（语义高度一致）
- **0** = 正交（语义无关）
- **-1** = 方向相反（语义对立，实践中罕见）

```python
import math

def cosine_similarity(a, b):
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

### 教学用 Embedding：词频向量

真实 Embedding 需要 API（如 OpenAI `text-embedding-3-small`，返回 1536 维向量）。
本章用**词频向量**模拟——把文本分词，每个词对应一个维度，值为出现次数。

```python
def simple_embedding(text: str) -> dict[str, float]:
    """词频向量模拟 embedding（教学用）。"""
    cleaned = text.lower()
    for ch in "，。！？,.!?;:\"'()[]{}（）【】":
        cleaned = cleaned.replace(ch, " ")
    words = cleaned.split()
    vec = {}
    for w in words:
        # 中文连续字符拆成单字（中文无空格分词）
        if len(w) > 1 and all("\u4e00" <= c <= "\u9fff" for c in w):
            for c in w:
                vec[c] = vec.get(c, 0.0) + 1.0
        else:
            vec[w] = vec.get(w, 0.0) + 1.0
    return vec
```

**为什么词频向量能近似语义检索？** 因为语义相近的文本往往共享关键词。它不懂同义词、不懂上下文，
但**教学上足以演示完整的 RAG 流程**，且 100% 离线可运行。

> 📌 **真实项目**：把 `simple_embedding` 换成 `client.embeddings.create(input=text, model="text-embedding-3-small")`，
> 返回真实语义向量，检索质量会大幅提升。本章代码结构不变。

### 中文分词陷阱（已在第05章踩坑并解决）

中文没有空格分隔，如果直接 `text.split()`，整句中文会变成**一个巨大的 token**，
导致 query 和文档**零词汇重叠**，余弦相似度全为 0。

**解决**：检测连续中文字符（`'\u4e00' <= c <= '\u9fff'`），拆成单字模拟基础分词。
这样 query "Python 有什么特点" 就能匹配到文档里含 "Python" 和 "特点" 的片段。

---

## 文档分块策略（Chunking）

### 为什么要分块

知识库里的文档通常很长（一本书几十万字）。直接把整篇文档 embedding 会导致：

1. **向量太粗糙**：一篇文章讲 5 个主题，embedding 混合了所有主题，检索时语义不精确
2. **上下文太长**：把整篇文章塞进 prompt，成本高、模型容易"迷失在中间"
3. **无法精确定位**：检索到"相关文章"但不知道"相关段落"在哪里

**分块 = 把长文档切成小的、语义集中的片段，每个片段单独 embedding。**

### 分块参数

两个核心参数：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `chunk_size` | 每块的最大长度（字符或 token） | 200-1000 |
| `overlap` | 相邻块之间的重叠长度 | chunk_size 的 10%-20% |

```
文档: [AAAA BBBB CCCC DDDD EEEE FFFF]
chunk_size=4, overlap=1:

块1: [AAAA]
块2:    [A BBBB]    ← 与块1重叠1个字符'A'
块3:       [BB CCCC]
块4:          [CC DDDD]
...
```

### 为什么要有重叠（overlap）

**重叠防止语义在边界处被切断。** 如果一个句子正好被切在两个块的边界，它的语义就分裂了，
检索时两个块都可能"部分匹配"但都不完整。重叠让边界句子同时出现在两个块里，提高召回率。

### 分块太大或太小的问题

**分块太小**（如 chunk_size=10）：
- 每块语义不完整，可能只有半个句子
- 块数量爆炸，检索成本上升
- 模型拿到的上下文碎片化，难以理解

**分块太大**（如 chunk_size=5000）：
- 每块语义太杂，embedding 不精确
- 能塞进上下文的块数减少，召回率下降
- 单块 token 多，成本高

> 💡 **经验值**：chunk_size=200-500 字符，overlap=50-100 字符，适合大部分中文文档。

### 按字符 vs 按段落

- **按字符**：强制每块不超过 N 字符，简单但可能切断句子
- **按段落**：以换行为分界，语义更完整，但段落长度不可控
- **按句子 + 字符上限**：先按句子分，累积到接近 chunk_size 就切一块（最常用）

本章用**按字符 + overlap 滑窗**实现，简单直观：

```python
def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # 步进 = chunk_size - overlap
    return chunks
```

---

## 基础 RAG 流程

基础 RAG 是一条**固定的管道**：无论问什么问题，都走"检索 → 注入 → 回答"三步。

```
用户提问
   │
   ▼
[1. 检索] query → embedding → 余弦相似度排序 → 取 top_k 个分块
   │
   ▼
[2. 注入] 把 top_k 分块拼成"背景知识"，塞进 system/user prompt
   │
   ▼
[3. 回答] 模型基于背景知识 + 问题，生成回答
```

### 1. 检索（Retrieve）

```python
# 构建 query 的 embedding
q_emb = simple_embedding(query)
# 与所有分块的 embedding 算相似度
scored = [(chunk, cosine_similarity(q_emb, emb)) for chunk, emb in index]
# 按相似度降序，取 top_k
scored.sort(key=lambda x: x[1], reverse=True)
top_chunks = scored[:top_k]
```

### 2. 注入（Augment）

把检索到的 top_k 个分块作为"背景知识"注入 prompt：

```python
context = "\n\n".join(f"[片段{i+1}] {chunk}" for i, (chunk, _) in enumerate(top_chunks))
prompt = f"请根据以下背景知识回答问题。\n\n背景知识：\n{context}\n\n问题：{query}"
```

### 3. 回答（Generate）

```python
response = client.chat.completions.create(
    model=cfg.model,
    messages=[
        {"role": "system", "content": "你是任务助手 Agent，基于背景知识回答问题。"},
        {"role": "user", "content": prompt},
    ],
)
answer = response.choices[0].message.content
```

### 基础 RAG 的问题

基础 RAG 是**无条件检索**——不管问什么，都去查知识库。这有两个问题：

1. **问"1+1=?"**：这是常识，模型自己就知道，但基础 RAG 还是去知识库里检索"1+1"，
   检索到的可能是无关片段，反而干扰回答。
2. **问"Python 的 GIL 是什么"**：可能知识库里没有直接答案，但 Agent 可以多次检索
   （先查"Python 特点"，再查"并发"），基础 RAG 只检索一次就放弃了。

这就引出了 **Agentic RAG**。

---

## Agentic RAG：让 Agent 自主决定检索

> **Agentic RAG = 把"检索"变成 Agent 的一个工具，让 Agent 自己决定是否检索、检索什么、检索几次。**

### 核心区别

| 维度 | 基础 RAG | Agentic RAG |
|------|----------|-------------|
| **谁决定检索？** | 固定管道（每次都检索） | **Agent 自主决定** |
| **检索次数** | 固定 1 次 | 0 次到 N 次（按需） |
| **检索 query** | = 用户原问题 | Agent 可以**改写 query**（如先查概念再查细节） |
| **处理常识问题** | 也检索（浪费/干扰） | **直接回答**（不检索） |
| **实现复杂度** | 低（管道） | 高（Agent 循环 + 工具） |
| **适合场景** | 简单 FAQ | 复杂知识问答、多跳推理 |

### 实现方式

把知识库检索封装成一个工具 `search_knowledge_base(query)`，注册到第04章的 Agent 循环里：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在知识库中检索与查询相关的文档片段。当用户问与知识库内容相关的问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词"}
                },
                "required": ["query"],
            },
        },
    }
]
```

Agent 循环中，模型会**根据问题自主决定**：

- 问"Python 有什么特点？" → 调用 `search_knowledge_base("Python 特点")` → 基于结果回答
- 问"1+1=?" → **不调用任何工具** → 直接回答 "2"

### Agentic RAG 的优势

1. **节省成本**：常识问题不检索，省掉无用的 embedding 计算和 API 调用
2. **多跳检索**：复杂问题可以多次检索，如"Agent 和 RAG 的关系" → 先查"Agent"再查"RAG"
3. **query 改写**：Agent 可以把模糊问题改写成精确检索词
4. **自我纠正**：第一次检索结果不理想，Agent 可以换关键词重新检索

---

## 纯 RAG vs Agentic RAG：什么时候用哪个

### 用基础 RAG 的场景

- **固定领域的 FAQ**：如产品文档问答，问题模式固定，每次都该查
- **严格的、可审计的管道**：要求每次回答都必须基于检索结果（合规场景）
- **简单场景**：不想引入 Agent 循环的复杂度

### 用 Agentic RAG 的场景

- **混合问答**：有些问题是常识（不需要检索），有些需要查资料
- **多跳推理**：一个问题需要综合多个文档片段
- **动态决策**：Agent 需要判断"检索结果够不够"，不够就再查
- **复杂知识库**：有多个子库，Agent 需要选择查哪个

> 💡 **趋势**：现代 RAG 系统越来越多地采用 Agentic RAG，因为 Agent 循环（第04章）已经成熟，
> 把检索作为工具的边际成本很低，但收益（灵活性、准确性）很大。

---

## 反模式（什么不该做）

### ❌ 分块太大或太小

```python
# 坏：分块太大，一个块 5000 字符，语义混杂
chunks = chunk_text(doc, chunk_size=5000, overlap=0)

# 坏：分块太小，一个块 10 字符，全是碎片
chunks = chunk_text(doc, chunk_size=10, overlap=0)
```

**后果**：太大 → embedding 不精确，检索召回率低；太小 → 上下文碎片化，模型无法理解。

**正确**：chunk_size=200-500 字符，overlap=50-100 字符，根据文档类型调优。

### ❌ 不做混合检索（只用向量检索）

```python
# 坏：只靠向量检索，认为语义相似就够了
results = vector_search(query, top_k=3)
```

**后果**：
- 向量检索对**精确关键词**（人名、产品号、错误码）不如全文检索精准
- 向量检索可能召回"语义像但意思不同"的内容（如"退货政策" vs "退款政策"）

**正确**：**混合检索**（向量 + 关键词/BM25），向量召回候选集，再用关键词匹配重排序。
本章只实现向量检索（教学用），生产系统必须加关键词检索。（Reranking、query expansion 等高级技术同理，只提及不讲透。）

### ❌ 把无关检索结果塞满上下文

```python
# 坏：无脑取 top_k，不管相似度多低都塞进去
top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
# 即使最高分只有 0.05（几乎无关），也塞 5 个进去
context = "\n".join(chunk for chunk, _ in top_chunks)
```

**后果**：
- **噪声淹没信号**：5 个无关片段塞进去，模型被干扰，回答质量反而下降
- **成本浪费**：不相关的 token 也要付费
- **"lost in the middle"**：上下文越长，模型越容易忽略中间的关键信息

**正确**：**设置相似度阈值**（如 score > 0.3），低于阈值的片段不塞进去。
如果所有片段都低于阈值，应该回答"知识库中没有相关信息"，而不是硬编。

```python
# 好：相似度阈值过滤
threshold = 0.3
filtered = [(c, s) for c, s in top_chunks if s >= threshold]
if not filtered:
    return "知识库中没有找到相关信息。"
```

> 💡 **本章练习题**就是实现这个相似度阈值过滤。

---

## 运行示例

```bash
# Python
cd ai-agent/09-rag
python3 python/main.py

# TypeScript
cd ai-agent/09-rag
npx tsx typescript/main.ts
```

输出用以下前缀标记各阶段，方便 QA 脚本 grep 过滤 dotenvx 横幅：

- `OUT:chunk:` — 分块结果
- `OUT:embed:` — Embedding 向量化
- `OUT:retrieve:` — 检索结果（top-k + 相似度分数）
- `OUT:answer:` — RAG 回答
- `OUT:agentic:` — Agentic RAG 决策过程

### 离线可运行（关键设计）

本章两个 demo 都**不依赖真实 API**：

- **基础 RAG**：embedding 用纯 Python 词频向量（复用第05章模式），完全不调 API；
  回答阶段 try 真实 API，失败降级 mock（基于检索片段拼接）
- **Agentic RAG**：预设 mock 决策序列（问 Python → 检索 → 回答；问 1+1 → 直接回答），
  不依赖真实 API

所以即使 `.env` 是占位符 `OPENAI_API_KEY=sk-REPLACE-ME`，两个 demo 都能完整跑通，exit 0。

---

## 与前几章的关系

| 章节 | 为本章提供了什么 |
|------|-----------------|
| 第05章 记忆系统 | VectorMemory 的 `simple_embedding` + `cosine_similarity` 模式（本章复用） |
| 第04章 Agent 循环 | Agent loop 骨架（Agentic RAG 把检索作为工具接入循环） |
| 第07章 ReAct | 显式推理（Agentic RAG 中 Agent 可以显式思考"是否需要检索"） |

> 💡 **下一步**：第10章「多 Agent 编编」会让多个 Agent 协作，其中一个常见分工就是
> "检索 Agent + 回答 Agent"——这正是 Agentic RAG 的多 Agent 演化形态。

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
- [知识库数据](./data/)
