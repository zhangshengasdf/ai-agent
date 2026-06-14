# 第09章 练习 — RAG 检索

> 以下练习基于本章代码（`python/main.py` / `typescript/main.ts`）。
> 建议先自己动手实现，再对照参考答案。

---

## 练习 1：相似度阈值过滤（核心练习）

**目标**：让 Agent 只检索**真正相关**的分块，而非无脑取 top-k。

本章的 `retrieve()` 函数总是返回 top_k 个分块，即使它们的相似度极低（如 0.05）。
这会把无关内容塞进上下文，干扰回答（README 反模式 #3）。

**要求**：

1. 修改 `retrieve()` 函数（或新增 `retrieve_with_threshold()`），增加一个 `min_score` 参数（默认 0.3）
2. 只返回 `score >= min_score` 的分块
3. 如果所有分块都低于阈值，返回空列表
4. 修改 `rag_answer()`：当检索结果为空时，回答"知识库中没有找到与该问题相关的信息。"

**验证**：

- 查询 "Python 有什么特点？" → 应返回相关分块（score > 0.3）
- 查询 "今天午饭吃什么？" → 应返回空列表（所有 score < 0.3）

### 参考答案（Python）

```python
def retrieve_with_threshold(
    query: str,
    index: List[Tuple[str, Embedding]],
    top_k: int = 3,
    min_score: float = 0.3,
) -> List[Tuple[str, float]]:
    """检索相关分块，过滤掉低于 min_score 的低质量结果。"""
    if not index:
        return []
    q_emb = simple_embedding(query)
    scored = [(chunk, cosine_similarity(q_emb, emb)) for chunk, emb in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    # 关键：过滤掉低于阈值的分块
    filtered = [(c, s) for c, s in scored[:top_k] if s >= min_score]
    return filtered


def rag_answer_with_threshold(query: str, index, top_k=3, min_score=0.3):
    results = retrieve_with_threshold(query, index, top_k, min_score)
    if not results:
        return f"知识库中没有找到与「{query}」相关的信息。"
    # ... 后续注入 + 回答逻辑与 rag_answer 一致
```

### 参考答案（TypeScript）

```typescript
function retrieveWithThreshold(
  query: string,
  index: IndexEntry[],
  topK = 3,
  minScore = 0.3,
): RetrieveResult[] {
  if (index.length === 0) return [];
  const qEmb = simpleEmbedding(query);
  const scored: RetrieveResult[] = index.map((entry) => ({
    chunk: entry.chunk,
    score: cosineSimilarity(qEmb, entry.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).filter((r) => r.score >= minScore);
}
```

**思考**：`min_score` 设多少合适？设太高（如 0.8）会漏掉相关内容，设太低（如 0.05）等于没过滤。
这取决于 embedding 模型——真实 embedding API（如 text-embedding-3-small）的相似度分布与词频向量不同，需要用验证集调优。

---

## 练习 2：换用真实 Embedding API

**目标**：把教学用的 `simple_embedding` 换成 OpenAI 真实 embedding，对比检索质量。

**要求**：

1. 新增 `real_embedding(text: str) -> list[float]` 函数，调用 `client.embeddings.create()`
2. 用 try/except 包裹（API 不可用时降级回 `simple_embedding`）
3. 修改 `build_index()` 和 `retrieve()`，用 `real_embedding` 替代 `simple_embedding`
4. 注意：真实 embedding 是 **密集向量**（1536 维 list[float]），余弦相似度需要改成 `zip` 版本而非 dict 版本

**验证**：

- 查询 "Python 有什么特点？" → 用真实 embedding 后，相关分块的 score 应该更高（如 0.7+）
- 查询 "Python 的并发模型" → 真实 embedding 能匹配到 GIL 相关段落（语义理解更强）

### 参考答案（Python）

```python
from typing import List as ListType
import os

# 缓存已计算的 embedding，避免重复调用 API
_embedding_cache: dict[str, list[float]] = {}


def real_embedding(text: str) -> list[float]:
    """用 OpenAI 真实 embedding API（失败降级会报错，因为返回类型不同）。"""
    if text in _embedding_cache:
        return _embedding_cache[text]
    try:
        resp = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        emb = resp.data[0].embedding
        _embedding_cache[text] = emb
        return emb
    except Exception:
        # API 不可用时，不能用 simple_embedding（返回 dict），这里直接抛错
        # 或者：把整个系统设计为 "要么全用真实，要么全用 mock"
        raise RuntimeError("Embedding API 不可用，请配置有效 API key")


def cosine_similarity_dense(a: list[float], b: list[float]) -> float:
    """密集向量的余弦相似度（真实 embedding 用）。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

**注意**：真实 embedding 返回 1536 维密集向量，与词频向量（稀疏 dict）不兼容。
切换时需要同时改 `build_index` 的存储类型和 `cosine_similarity` 的实现。

---

## 练习 3：多轮 Agentic RAG（多跳检索）

**目标**：让 Agentic RAG 能处理需要**多次检索**的复杂问题。

本章的 Agentic RAG 在 mock 模式下最多检索 1 次。真实场景中，Agent 可能需要多跳：

```
问题："Python 和 Agent 的关系是什么？"
Step 1: 检索 "Python" → 得到 Python 语言特点
Step 2: 检索 "AI Agent" → 得到 Agent 概念
Step 3: 综合两个结果回答
```

**要求**：

1. 修改离线 mock 的决策逻辑，支持多轮检索
2. 用关键词判断：如果问题包含多个知识库主题词（如同时有 "Python" 和 "Agent"），则分别检索
3. 每次检索后判断是否信息足够，不够就继续检索

### 参考答案（Python）

```python
def _agentic_rag_multi_hop(query: str, index) -> str:
    """多跳 mock：对包含多个主题的问题分别检索。"""
    topics = {
        "python": ["python", "编程"],
        "agent": ["agent", "智能体"],
        "llm": ["llm", "大模型", "大语言模型"],
    }

    # 检测问题涉及哪些主题
    query_lower = query.lower()
    relevant_topics = []
    for topic, keywords in topics.items():
        if any(kw in query_lower for kw in keywords):
            relevant_topics.append(topic)

    if not relevant_topics:
        return _direct_answer(query)  # 常识问题，直接回答

    # 对每个相关主题分别检索
    all_results = []
    for step, topic in enumerate(relevant_topics, 1):
        print(f"OUT:agentic:step{step}: 检索主题 '{topic}'")
        results = retrieve(topic, index, top_k=2)
        all_results.extend(results)

    # 综合所有结果回答
    context = "\n".join(chunk for chunk, _ in all_results)
    return _mock_answer(query, context)
```

---

## 练习 4：分块参数调优实验

**目标**：直观感受 chunk_size 和 overlap 对检索质量的影响。

**要求**：

1. 用不同的 `chunk_size`（50, 200, 500）和 `overlap`（0, 50, 100）组合构建索引
2. 对同一组查询（如 "Python 特点"、"Agent 概念"），对比不同参数下的 top-1 检索分数
3. 记录：哪种参数组合的检索分数最高？分块数量分别是多少？

**预期发现**：

- `chunk_size=50, overlap=0`：分块碎片化，检索分数低（语义不完整）
- `chunk_size=200, overlap=50`：分块适中，检索分数较高（推荐配置）
- `chunk_size=500, overlap=0`：分块太大，语义混杂，检索分数可能下降

### 参考答案（Python）

```python
def experiment_chunk_params(documents, query):
    """对比不同分块参数的检索效果。"""
    configs = [
        (50, 0),    # 碎片化
        (200, 50),  # 推荐
        (500, 0),   # 过大
    ]
    for chunk_size, overlap in configs:
        index = build_index(documents, chunk_size, overlap)
        results = retrieve(query, index, top_k=1)
        best_score = results[0][1] if results else 0
        print(
            f"chunk_size={chunk_size}, overlap={overlap}: "
            f"分块数={len(index)}, top-1 score={best_score:.3f}"
        )
```

---

## 练习 5：为 Agentic RAG 添加查询改写

**目标**：让 Agent 在检索前**改写 query**，提高召回率。

用户的问题可能很模糊（如"它怎么样？"），直接检索效果差。Agent 应该先改写：

```
用户："Python 的那个锁是什么？"
Agent 改写 query → "Python GIL 全局解释器锁"
检索 → 找到 GIL 相关段落
```

**要求**：

1. 新增一个工具 `rewrite_query(vague_query: str) -> str`，用 LLM（或 mock 关键词映射）改写模糊查询
2. 在 Agent 循环中，Agent 先调用 `rewrite_query`，再用改写后的 query 调用 `search_knowledge_base`
3. 离线 mock：用关键词映射模拟改写（如"锁" → "Python GIL 全局解释器锁"）

**思考**：这本质上是 **query expansion**（查询扩展）的一个简化版。生产系统会怎么做？

---

## 总结

| 练习 | 核心技能 | 难度 |
|------|----------|------|
| 1. 相似度阈值过滤 | 质量控制、反模式修复 | ★★☆ |
| 2. 真实 Embedding API | API 集成、密集向量 | ★★★ |
| 3. 多跳检索 | Agentic RAG 深化 | ★★★ |
| 4. 分块参数调优 | 实验思维、参数敏感性 | ★★☆ |
| 5. 查询改写 | Query expansion 入门 | ★★★ |

完成练习 1-3 后，你就掌握了 RAG 的核心工程能力：**检索质量控制 + 真实 API 集成 + Agentic 多跳**。
