# 练习题 — 深度研究助手

## 练习1：添加相似度阈值过滤

**目标**：给 `search_knowledge` 函数添加相似度阈值参数，当所有检索结果低于阈值时返回"未找到相关内容"。

**说明**：当前实现即使最高相似度只有 0.05（几乎无关），也会返回结果。添加阈值过滤可避免噪声干扰。

**要求**：
- 添加 `threshold` 参数（默认 0.1）
- 过滤掉低于阈值的结果
- 如果所有结果都被过滤，返回"知识库中未找到相关信息"

<details>
<summary>参考答案（Python）</summary>

```python
def search_knowledge(
    query: str, index: List[Tuple[str, Embedding]], top_k: int = 3, threshold: float = 0.1
) -> str:
    """RAG 检索：query embedding → cosine similarity → top-k，带阈值过滤。"""
    q_emb = simple_embedding(query)
    scored = [(chunk, cosine_similarity(q_emb, emb)) for chunk, emb in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    # 阈值过滤
    filtered = [(c, s) for c, s in scored[:top_k] if s >= threshold]
    if not filtered:
        return "（知识库中未找到相关信息）"
    parts = []
    for i, (chunk, score) in enumerate(filtered):
        parts.append(f"[片段{i+1}] (相似度:{score:.3f}) {chunk}")
    return "\n".join(parts)
```
</details>

<details>
<summary>参考答案（TypeScript）</summary>

```typescript
function searchKnowledge(
  query: string,
  index: IndexEntry[],
  topK = 3,
  threshold = 0.1,
): string {
  const qEmb = simpleEmbedding(query);
  const scored = index.map((entry) => ({
    chunk: entry.chunk,
    score: cosineSimilarity(qEmb, entry.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  const filtered = scored.slice(0, topK).filter((r) => r.score >= threshold);
  if (filtered.length === 0) {
    return "（知识库中未找到相关信息）";
  }
  return filtered
    .map(
      (r, i) => `[片段${i + 1}] (相似度:${r.score.toFixed(3)}) ${r.chunk}`,
    )
    .join("\n");
}
```
</details>

---

## 练习2：添加多轮研究循环

**目标**：在 `execute_research` 之后添加一个"检查-补充"阶段——如果笔记少于 3 条，自动追加一轮补充检索。

**说明**：Plan-and-Execute 的单次规划可能遗漏关键信息。添加一个自检循环可以让研究更完整。

**要求**：
- 执行完所有步骤后检查 `notes` 数量
- 如果 notes < 3，生成一个补充检索 query，再搜索一次
- 最多补充 2 轮，避免无限循环
- 在 trace 中记录补充轮次

<details>
<summary>参考答案（Python）</summary>

```python
def maybe补充检索(
    topic: str,
    index: List[Tuple[str, Embedding]],
    notes: List[str],
    tracer: Tracer,
    max_rounds: int = 2,
    min_notes: int = 3,
) -> None:
    """如果笔记不足，自动补充检索。"""
    for round_num in range(1, max_rounds + 1):
        if len(notes) >= min_notes:
            break
        # 用 topic 作为补充检索词
        t0 = time.time()
        result = search_knowledge(f"{topic} 补充", index)
        elapsed = (time.time() - t0) * 1000
        tracer.record(f"补充检索(第{round_num}轮)", elapsed)
        note = f"[补充检索-轮{round_num}] {result[:150]}"
        notes.append(note)
        print(f"OUT:step: 补充检索第{round_num}轮: {result[:60]}...")
```
</details>

<details>
<summary>参考答案（TypeScript）</summary>

```typescript
async function maybe补充检索(
  topic: string,
  index: IndexEntry[],
  notes: string[],
  tracer: Tracer,
  maxRounds = 2,
  minNotes = 3,
): Promise<void> {
  for (let round = 1; round <= maxRounds; round++) {
    if (notes.length >= minNotes) break;
    const t0 = performance.now();
    const result = searchKnowledge(`${topic} 补充`, index);
    const elapsed = performance.now() - t0;
    tracer.record(`补充检索(第${round}轮)`, elapsed);
    const note = `[补充检索-轮${round}] ${result.slice(0, 150)}`;
    notes.push(note);
    console.log(`OUT:step: 补充检索第${round}轮: ${result.slice(0, 60)}...`);
  }
}
```
</details>

---

## 练习3：实现 ReAct 推理模式

**目标**：将当前的 Plan-and-Execute 模式改为 ReAct（Reasoning + Acting）模式——每一步先"思考"再"行动"。

**说明**：Plan-and-Execute 是先规划再执行，ReAct 是边思考边行动。两者的核心区别在于：ReAct 的每一步都包含 Thought → Action → Observation 三元组。

**要求**：
- 每步输出格式：`Thought: ... → Action: ... → Observation: ...`
- Thought 阶段：LLM 分析当前信息，决定下一步行动
- Action 阶段：执行工具调用
- Observation 阶段：记录工具返回的结果
- 最多 5 步，防止无限循环
- 在 trace 中记录每个 ReAct 循环

<details>
<summary>参考答案（Python 核心逻辑）</summary>

```python
def react_research(
    topic: str,
    index: List[Tuple[str, Embedding]],
    notes: List[str],
    tracer: Tracer,
    max_steps: int = 5,
) -> None:
    """ReAct 模式：Thought → Action → Observation 循环。"""
    system = (
        "你是研究助手。每一步先输出 Thought（分析当前信息），"
        "再决定 Action（调用哪个工具），格式：\n"
        "Thought: ...\nAction: search_knowledge(query=\"...\")\n"
        "或 Action: finish(总结=\"...\")"
    )
    context = f"研究主题: {topic}\n已收集笔记: {len(notes)} 条"

    for step in range(1, max_steps + 1):
        t0 = time.time()
        resp = llm_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ], tracer=tracer, step_name=f"react_step_{step}")
        elapsed = (time.time() - t0) * 1000

        content = resp.get("content", "")
        print(f"\nOUT:step: ── ReAct 步骤 {step} ──")
        print(f"OUT:step: {content[:200]}")

        # 解析 Action
        if "finish" in content.lower():
            print("OUT:step: 研究完成")
            break
        if "search_knowledge" in content.lower():
            # 提取 query
            import re
            match = re.search(r'query[=:]\s*["\']?([^"\')\n]+)', content)
            query = match.group(1).strip() if match else topic
            result = search_knowledge(query, index)
            notes.append(f"[ReAct-{step}] {result[:150]}")
            context += f"\n\nObservation {step}: {result[:200]}"
```
</details>

<details>
<summary>参考答案（TypeScript 核心逻辑）</summary>

```typescript
async function reactResearch(
  topic: string,
  index: IndexEntry[],
  notes: string[],
  tracer: Tracer,
  maxSteps = 5,
): Promise<void> {
  const system =
    "你是研究助手。每一步先输出 Thought（分析当前信息），" +
    "再决定 Action（调用哪个工具），格式：\n" +
    'Thought: ...\nAction: search_knowledge(query="...")\n' +
    '或 Action: finish(总结="...")';
  let context = `研究主题: ${topic}\n已收集笔记: ${notes.length} 条`;

  for (let step = 1; step <= maxSteps; step++) {
    const resp = await llmChat(
      [
        { role: "system", content: system },
        { role: "user", content: context },
      ],
      undefined,
      tracer,
      `react_step_${step}`,
    );

    const content = resp.content;
    console.log(`\nOUT:step: ── ReAct 步骤 ${step} ──`);
    console.log(`OUT:step: ${content.slice(0, 200)}`);

    if (content.toLowerCase().includes("finish")) {
      console.log("OUT:step: 研究完成");
      break;
    }
    if (content.toLowerCase().includes("search_knowledge")) {
      const match = content.match(/query[=:]\s*["']?([^"')\n]+)/i);
      const query = match ? match[1].trim() : topic;
      const result = searchKnowledge(query, index);
      notes.push(`[ReAct-${step}] ${result.slice(0, 150)}`);
      context += `\n\nObservation ${step}: ${result.slice(0, 200)}`;
    }
  }
}
```
</details>
