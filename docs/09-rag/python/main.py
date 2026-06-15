"""第09章 RAG 检索（Retrieval-Augmented Generation）

本章实现两种 RAG：
  1. 基础 RAG pipeline：固定管道 检索→注入→回答（无条件检索）
  2. Agentic RAG：把检索作为工具，Agent 自主决定是否检索、检索什么

核心组件（复用第05章 VectorMemory 的模式）：
  - simple_embedding：纯 Python 词频向量（中文拆单字），不依赖 embedding API
  - cosine_similarity：纯 Python 余弦相似度（稀疏 dict 向量）
  - chunk_text：文档分块（chunk_size + overlap 滑窗）

离线设计：
  - 基础 RAG：embedding 纯 Python，回答阶段 try API 失败降级 mock
  - Agentic RAG：预设 mock 决策序列演示 Agent 自主检索（问 Python→检索，问 1+1→直接回答）
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

# 知识库目录（data/ 在章节根目录下）
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ════════════════════════════════════════════════════════════════════
# 1. Embedding + 相似度（复用第05章 VectorMemory 模式）
# ════════════════════════════════════════════════════════════════════

# 用稀疏向量（dict[词, 权重]）表示 embedding，纯 Python，不依赖 numpy。
Embedding = Dict[str, float]


def simple_embedding(text: str) -> Embedding:
    """词频向量模拟 embedding（教学用，非真实语义）。

    把文本分词（空格+标点分割），每个词一个维度，值为出现次数。
    对连续中文字符拆成单字（中文无空格分词，否则整句变一个 token）。
    真实项目换成 client.embeddings.create(...) 即可。
    """
    cleaned = text.lower()
    for ch in "，。！？,.!?;:\"'()[]{}（）【】\n\r\t#*-`>":
        cleaned = cleaned.replace(ch, " ")
    words = cleaned.split()

    vec: Embedding = {}
    for w in words:
        if len(w) > 1 and all("\u4e00" <= c <= "\u9fff" for c in w):
            # 连续中文字符 → 拆成单字，模拟基础分词
            for c in w:
                vec[c] = vec.get(c, 0.0) + 1.0
        else:
            vec[w] = vec.get(w, 0.0) + 1.0
    return vec


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """两个稀疏向量的余弦相似度，范围 [-1, 1]，1 = 完全相同。

    纯 Python 实现，不用 numpy。用 dict 键交集算点积。
    """
    dot = sum(a[w] * b.get(w, 0.0) for w in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ════════════════════════════════════════════════════════════════════
# 2. 文档分块（Chunking）
# ════════════════════════════════════════════════════════════════════


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """把长文本分成带重叠窗口的小块。

    Args:
        text: 原始文本。
        chunk_size: 每块最大字符数。
        overlap: 相邻块重叠的字符数（防止语义在边界被切断）。

    Returns:
        分块列表，每块 <= chunk_size 字符。
    """
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size  # 防止 overlap >= chunk_size 时死循环
        start += step
    return chunks


# ════════════════════════════════════════════════════════════════════
# 3. 知识库加载 + 索引构建
# ════════════════════════════════════════════════════════════════════


def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    """从 data/ 目录加载所有 .md/.txt 文档。

    Returns:
        [(filename, content)] 列表。
    """
    docs: List[Tuple[str, str]] = []
    if not data_dir.exists():
        return docs
    for path in sorted(data_dir.iterdir()):
        if path.suffix in (".md", ".txt"):
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def build_index(
    documents: List[Tuple[str, str]],
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[Tuple[str, Embedding]]:
    """把文档分块并向量化，构建内存索引。

    Args:
        documents: [(filename, content)] 列表。
        chunk_size: 分块大小。
        overlap: 分块重叠。

    Returns:
        [(chunk_text, embedding)] 内存索引。
    """
    index: List[Tuple[str, Embedding]] = []
    for filename, content in documents:
        chunks = chunk_text(content, chunk_size, overlap)
        for chunk in chunks:
            index.append((chunk, simple_embedding(chunk)))
    return index


def retrieve(
    query: str, index: List[Tuple[str, Embedding]], top_k: int = 3
) -> List[Tuple[str, float]]:
    """检索与 query 最相关的 top_k 个分块。

    Args:
        query: 查询文本。
        index: 内存索引 [(chunk, embedding)]。
        top_k: 返回的最大条数。

    Returns:
        [(chunk_text, similarity_score)] 按相似度降序。
    """
    if not index:
        return []
    q_emb = simple_embedding(query)
    scored = [(chunk, cosine_similarity(q_emb, emb)) for chunk, emb in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ════════════════════════════════════════════════════════════════════
# 4. 基础 RAG：检索 → 注入 → 回答
# ════════════════════════════════════════════════════════════════════


def rag_answer(query: str, index: List[Tuple[str, Embedding]], top_k: int = 3) -> str:
    """基础 RAG：检索 top_k 分块 → 注入 prompt → 模型回答。

    回答阶段先 try 真实 API，失败降级 mock（基于检索片段拼接）。
    """
    # 1. 检索
    results = retrieve(query, index, top_k=top_k)
    print(f"OUT:retrieve: 检索到 {len(results)} 个相关分块:")
    for i, (chunk, score) in enumerate(results, 1):
        preview = chunk[:60].replace("\n", " ")
        print(f"OUT:retrieve:   [{i}] score={score:.3f} | {preview}...")

    # 2. 注入上下文
    context_parts = []
    for i, (chunk, _) in enumerate(results, 1):
        context_parts.append(f"[片段{i}] {chunk}")
    context = "\n\n".join(context_parts)

    prompt = (
        f"请根据以下背景知识回答问题。\n\n"
        f"背景知识：\n{context}\n\n"
        f"问题：{query}"
    )

    # 3. 回答（try 真实 API，失败降级 mock）
    answer = _generate_answer(query, context, prompt)
    return answer


def _generate_answer(query: str, context: str, prompt: str) -> str:
    """调 LLM 生成回答，失败时降级为 mock 回答。"""
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是任务助手 Agent，请基于提供的背景知识回答用户问题。"
                    "如果背景知识中没有答案，请如实说明。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content or "(空回答)"
    except Exception as e:
        # 离线 mock：基于检索片段拼接一个教学用回答
        print(f"OUT:answer: [离线模式] API 不可用（{type(e).__name__}），降级 mock 回答。")
        return _mock_answer(query, context)


def _mock_answer(query: str, context: str) -> str:
    """离线 mock 回答：从检索片段中提取关键信息拼接。"""
    snippets = context.split("[片段")
    relevant = []
    for s in snippets[1:]:  # 跳过第一个（[片段 1] 之前的部分）
        # 提取片段文本的前 80 字符作为摘要
        lines = s.strip()
        if lines:
            # 去掉片段编号前缀 "] "
            idx = lines.find("]")
            if idx >= 0:
                lines = lines[idx + 1 :].strip()
            relevant.append(lines[:80])

    summary = " ".join(relevant)[:200]
    return f"[基于检索结果的 mock 回答] 关于「{query}」：根据知识库，{summary}..."


# ════════════════════════════════════════════════════════════════════
# 5. Agentic RAG：检索作为工具，Agent 自主决定
# ════════════════════════════════════════════════════════════════════

# 检索工具定义（JSON Schema，与第04章一致）
RAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在知识库中检索与查询相关的文档片段。"
            "当用户问与知识库内容（如 Python、AI Agent、LLM）相关的问题时使用。"
            "对于常识问题（如数学计算）不需要使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词，如 'Python 特点' 或 'Agent 概念'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

MAX_STEPS = 6


def search_knowledge_base(query: str, index: List[Tuple[str, Embedding]]) -> str:
    """检索工具：返回与 query 最相关的文档片段（供 Agent 使用）。"""
    results = retrieve(query, index, top_k=3)
    if not results:
        return "知识库为空，未找到相关信息。"
    parts = []
    for i, (chunk, score) in enumerate(results, 1):
        parts.append(f"[片段{i} 相关度={score:.3f}] {chunk}")
    return "\n\n".join(parts)


def agentic_rag(
    query: str, index: List[Tuple[str, Embedding]], use_mock: bool = False
) -> str:
    """Agentic RAG：Agent 自主决定是否检索、检索什么。

    Args:
        query: 用户问题。
        index: 知识库索引。
        use_mock: True 时用预设决策序列（离线演示），False 时调真实 API。

    Returns:
        Agent 的最终回答。
    """
    if use_mock:
        return _agentic_rag_mock(query, index)

    messages: List[Dict] = [
        {
            "role": "system",
            "content": (
                "你是任务助手 Agent。你有一个工具 search_knowledge_base 可以检索知识库。"
                "当用户问题与知识库内容（Python、AI Agent、LLM 等）相关时，调用工具检索。"
                "对于常识问题（如数学计算、打招呼），直接回答，不需要检索。"
                "检索到信息后，基于检索结果给出准确回答。"
            ),
        },
        {"role": "user", "content": query},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"OUT:agentic:step{step}: 思考中...")
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                tools=RAG_TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            print(
                f"OUT:agentic: [离线模式] API 不可用（{type(e).__name__}），"
                f"降级 mock 决策演示。"
            )
            return _agentic_rag_mock(query, index)

        assistant_msg = response.choices[0].message

        # 终止条件 1：模型不调工具 = 直接回答
        if not assistant_msg.tool_calls:
            answer = assistant_msg.content or "(空回答)"
            print(f"OUT:agentic:step{step}: ✓ Agent 决定直接回答（未调用检索工具）")
            print(f"OUT:agentic:step{step}: 回答: {answer[:120]}")
            return answer

        # Agent 决定调用检索工具
        messages.append(assistant_msg.model_dump())
        for tc in assistant_msg.tool_calls:
            if tc.type != "function":
                continue
            func_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            search_query = args.get("query", query)
            print(f"OUT:agentic:step{step}: → 调用 {func_name}(query='{search_query}')")
            result = search_knowledge_base(search_query, index)
            preview = result[:80].replace("\n", " ")
            print(f"OUT:agentic:step{step}: ← 检索结果: {preview}...")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )

    print(f"OUT:agentic: ⚠️ 达到最大步数 {MAX_STEPS}，停止。")
    return "(已达到最大步数)"


def _agentic_rag_mock(query: str, index: List[Tuple[str, Embedding]]) -> str:
    """离线 mock Agentic RAG：用预设决策序列演示 Agent 自主检索。

    决策逻辑（模拟一个聪明 Agent 的行为）：
      - 如果问题与知识库相关（含 Python/Agent/LLM 等关键词）→ 检索后回答
      - 如果是常识问题（如 1+1）→ 直接回答
    """
    print(f"OUT:agentic: [离线 mock] 用规则模拟 Agent 的检索决策。")

    # 模拟 Agent 的"判断"：问题是否与知识库相关
    kb_keywords = ["python", "agent", "llm", "大模型", "智能体", "语言模型", "编程"]
    query_lower = query.lower()
    needs_retrieval = any(kw in query_lower for kw in kb_keywords)

    if needs_retrieval:
        # Agent 决定检索
        print(f"OUT:agentic:step1: 思考中...")
        print(f"OUT:agentic:step1: Agent 判断：问题与知识库相关 → 调用检索工具")
        # 提取检索关键词（简化：用 query 本身）
        search_query = query
        print(f"OUT:agentic:step1: → search_knowledge_base(query='{search_query}')")
        result = search_knowledge_base(search_query, index)
        preview = result[:80].replace("\n", " ")
        print(f"OUT:agentic:step1: ← 检索结果: {preview}...")

        # Agent 基于检索结果回答
        print(f"OUT:agentic:step2: 思考中...")
        print(f"OUT:agentic:step2: Agent 判断：信息足够 → 基于检索结果回答（不再检索）")
        answer = _mock_answer(query, result)
        print(f"OUT:agentic:step2: 回答: {answer[:120]}")
        return answer
    else:
        # Agent 决定直接回答（不检索）
        print(f"OUT:agentic:step1: 思考中...")
        print(f"OUT:agentic:step1: Agent 判断：这是常识问题 → 不检索，直接回答")
        answer = _direct_answer(query)
        print(f"OUT:agentic:step1: 回答: {answer[:120]}")
        return answer


def _direct_answer(query: str) -> str:
    """对常识问题直接给出 mock 回答。"""
    # 简单数学检测：从 query 中提取数学表达式
    import re

    math_match = re.search(r"([\d\s+\-*/().]+)", query)
    if math_match:
        expr = math_match.group(1).strip()
        if expr and any(op in expr for op in "+-*/"):
            try:
                result = eval(expr)  # noqa: S307 — 受限字符集
                return f"{expr.strip()} = {result}"
            except Exception:
                pass
    if "你好" in query or "hello" in query.lower():
        return "你好！我是任务助手 Agent，有什么可以帮你的？"
    return f"这是常识问题，我直接回答：{query}"


# ════════════════════════════════════════════════════════════════════
# Demo 1: 基础 RAG pipeline
# ════════════════════════════════════════════════════════════════════


def demo_basic_rag() -> None:
    """演示基础 RAG：加载文档 → 分块 → 向量化 → 检索 → 回答。"""
    print(f"\n{'='*60}")
    print("Demo 1: 基础 RAG pipeline（检索 → 注入 → 回答）")
    print(f"{'='*60}")
    print("[说明] embedding 用纯 Python 词频向量；回答 try API，失败降级 mock。")

    # 1. 加载文档
    documents = load_documents(DATA_DIR)
    print(f"\nOUT:chunk: 加载 {len(documents)} 个文档:")
    for name, content in documents:
        print(f"OUT:chunk:   - {name} ({len(content)} 字符)")

    # 2. 分块 + 向量化
    index = build_index(documents, chunk_size=200, overlap=50)
    print(f"OUT:chunk: 分块完成（chunk_size=200, overlap=50）→ 共 {len(index)} 个分块")
    print(f"OUT:chunk: 前 3 个分块预览:")
    for i, (chunk, _) in enumerate(index[:3], 1):
        preview = chunk[:50].replace("\n", " ")
        print(f"OUT:chunk:   [{i}] {preview}...")

    # 3. Embedding 向量化验证
    sample_emb = simple_embedding("Python 编程语言")
    print(f"\nOUT:embed: simple_embedding('Python 编程语言') = {sample_emb}")
    a = simple_embedding("Python")
    b = simple_embedding("Python 语言")
    c = simple_embedding("天气")
    print(f"OUT:embed: cosine('Python', 'Python 语言') = {cosine_similarity(a, b):.3f} (应较高)")
    print(f"OUT:embed: cosine('Python', '天气') = {cosine_similarity(a, c):.3f} (应较低)")

    # 4. 检索 + 回答
    query = "Python 有什么特点？"
    print(f"\nOUT:retrieve: 查询: {query}")
    answer = rag_answer(query, index, top_k=3)
    print(f"\nOUT:answer: 回答: {answer[:200]}")

    print(f"\nOUT:answer: ✓ 基础 RAG 完成（固定管道：无条件检索 → 注入 → 回答）")


# ════════════════════════════════════════════════════════════════════
# Demo 2: Agentic RAG（Agent 自主决定是否检索）
# ════════════════════════════════════════════════════════════════════


def demo_agentic_rag() -> None:
    """演示 Agentic RAG：Agent 自主决定是否调用检索工具。"""
    print(f"\n{'='*60}")
    print("Demo 2: Agentic RAG（Agent 自主决定是否检索）")
    print(f"{'='*60}")
    print("[说明] 把检索作为工具，Agent 自主决定调用与否。")
    print("[说明] 先试真实 API，失败降级 mock 决策序列。")

    documents = load_documents(DATA_DIR)
    index = build_index(documents, chunk_size=200, overlap=50)

    # 场景 A：问文档相关问题 → Agent 应该检索
    print(f"\n--- 场景 A：问知识库相关问题 ---")
    query_a = "Python 语言有什么特点？"
    print(f"OUT:agentic: 问题: {query_a}")
    answer_a = agentic_rag(query_a, index, use_mock=False)
    print(f"OUT:agentic: ✓ Agent 对知识库相关问题进行了检索")

    # 场景 B：问常识问题 → Agent 应该直接回答（不检索）
    print(f"\n--- 场景 B：问常识问题 ---")
    query_b = "1+1 等于几？"
    print(f"OUT:agentic: 问题: {query_b}")
    answer_b = agentic_rag(query_b, index, use_mock=False)
    print(f"OUT:agentic: ✓ Agent 对常识问题直接回答（未检索）")

    # 场景 C：另一个知识库问题
    print(f"\n--- 场景 C：问 Agent 概念 ---")
    query_c = "什么是 AI Agent？"
    print(f"OUT:agentic: 问题: {query_c}")
    answer_c = agentic_rag(query_c, index, use_mock=False)
    print(f"OUT:agentic: ✓ Agent 根据问题性质做出了检索决策")

    print(f"\nOUT:agentic: 💡 对比：基础 RAG 对所有问题都检索；Agentic RAG 按需检索。")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print(f"[config] 章节主题: RAG 检索（基础 RAG + Agentic RAG）")
    print(f"[config] 知识库目录: {DATA_DIR}")

    # Demo 1: 基础 RAG
    demo_basic_rag()

    # Demo 2: Agentic RAG
    demo_agentic_rag()

    print(f"\n{'='*60}")
    print("所有演示完成！")
    print("💡 核心要点：")
    print("   - 基础 RAG：固定管道，无条件检索（简单但浪费）")
    print("   - Agentic RAG：Agent 自主决定检索（灵活但复杂）")
    print("   - Embedding 用纯 Python 词频向量模拟，真实项目换 embedding API")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
