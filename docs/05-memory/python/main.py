"""第05章 记忆系统（Memory Systems）

本章让 Agent 跨多轮对话"记住"上下文——LLM 本身无状态，每次 API 调用独立，
所谓的"记忆"是开发者替模型管理 messages 列表的机制。

三种记忆实现：
  1. ConversationBuffer: 完整保留所有对话（短期记忆，最简单）
  2. SummaryMemory: 超阈值时摘要压缩旧历史（中等对话）
  3. VectorMemory: 词频向量 + 余弦相似度检索（长期记忆，纯 Python，不依赖向量库）

离线设计：
  - Demo 1 (Buffer): 纯内存，100% 离线
  - Demo 2 (Summary): 先试真实 API 摘要，失败降级 mock 摘要
  - Demo 3 (Vector): 纯 Python 词频向量，不调 embedding API，100% 离线
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


# ════════════════════════════════════════════════════════════════════
# 1. ConversationBuffer — 完整保留所有对话（短期记忆）
# ════════════════════════════════════════════════════════════════════


class ConversationBuffer:
    """完整保留所有对话消息的短期记忆。

    最简单的记忆：一个列表，add 追加，get_messages 返回全量，clear 清空。
    优点：无信息损失，模型能看到完整历史，推理质量最高。
    缺点：上下文窗口有限（128K tokens），长对话会超限且成本二次增长。
    """

    def __init__(self) -> None:
        self._messages: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """追加一条消息（role: system/user/assistant）。"""
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """返回所有消息的副本（防外部修改内部状态）。"""
        return [dict(m) for m in self._messages]

    def count(self) -> int:
        """返回消息数量。"""
        return len(self._messages)

    def clear(self) -> None:
        """清空所有记忆。"""
        self._messages.clear()

    def to_prompt_messages(self) -> List[Dict[str, str]]:
        """转换为 API 期望的 messages 格式（本章直接就是该格式）。"""
        return self.get_messages()


# ════════════════════════════════════════════════════════════════════
# 2. SummaryMemory — 超阈值时摘要压缩（中等对话）
# ════════════════════════════════════════════════════════════════════


class SummaryMemory:
    """超阈值时把旧消息摘要压缩的记忆系统。

    当消息数超过 max_messages 时，把最早的一批送去 LLM 摘要，
    压缩成一段累积摘要（_summary），替换掉那批原文。
    最终传给 API 的是：[摘要 system msg] + [最近 N 条原文]。
    """

    def __init__(self, max_messages: int = 6, system_prompt: str = "") -> None:
        self._max = max_messages
        self._messages: List[Dict[str, str]] = []
        self._summary: str = ""
        self._system_prompt = system_prompt

    def add(self, role: str, content: str) -> None:
        """追加消息，超阈值时自动触发摘要。"""
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._max:
            self._summarize_oldest()

    def _summarize_oldest(self) -> None:
        """把最早 2 条消息送去摘要，压缩进 _summary。"""
        to_summarize = self._messages[:2]
        self._messages = self._messages[2:]

        chunk_text = "\n".join(f"[{m['role']}] {m['content']}" for m in to_summarize)

        # 先尝试真实 LLM 摘要，失败则用离线 mock
        new_summary = self._llm_summarize(chunk_text)
        if self._summary:
            self._summary = self._llm_summarize(self._summary + "\n" + new_summary)
        else:
            self._summary = new_summary

    def _llm_summarize(self, text: str) -> str:
        """调 LLM 摘要，失败时降级为离线 mock 摘要。"""
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {
                        "role": "system",
                        "content": "请用一句话（不超过50字）总结以下对话的要点：",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=100,
            )
            return resp.choices[0].message.content or "(摘要为空)"
        except Exception:
            # 离线 mock：提取关键词模拟摘要（教学用，非真实摘要）
            return self._mock_summarize(text)

    @staticmethod
    def _mock_summarize(text: str) -> str:
        """离线 mock 摘要：提取文本中的关键词模拟压缩。"""
        # 简单策略：取前 30 字符 + 关键词标记
        keywords = []
        for kw in ["小明", "北京", "上海", "Python", "天气", "偏好", "喜欢", "用户"]:
            if kw in text:
                keywords.append(kw)
        kw_str = "、".join(keywords) if keywords else "对话内容"
        snippet = text[:30].replace("\n", " ")
        return f"[摘要] 涉及{kw_str}。原文片段: {snippet}..."

    def get_messages(self) -> List[Dict[str, str]]:
        """返回 [摘要 system msg（如有）] + [最近 N 条原文]。"""
        result: List[Dict[str, str]] = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        if self._summary:
            result.append(
                {
                    "role": "system",
                    "content": f"[之前对话摘要] {self._summary}",
                }
            )
        result.extend(dict(m) for m in self._messages)
        return result

    def get_summary(self) -> str:
        """返回当前累积摘要（调试用）。"""
        return self._summary

    def count(self) -> int:
        """返回未压缩的原文消息数。"""
        return len(self._messages)

    def clear(self) -> None:
        """清空所有记忆和摘要。"""
        self._messages.clear()
        self._summary = ""


# ════════════════════════════════════════════════════════════════════
# 3. VectorMemory — 词频向量 + 余弦相似度检索（长期记忆）
# ════════════════════════════════════════════════════════════════════

# 用稀疏向量（dict[词, 权重]）表示 embedding，纯 Python，不依赖 numpy。
Embedding = Dict[str, float]


def simple_embedding(text: str) -> Embedding:
    """词频向量模拟 embedding（教学用，非真实语义）。

    把文本分词（简单空格+标点分割），每个词一个维度，值为出现次数。
    对连续的中文字符进一步拆成单字（中文无空格分词，否则无法匹配）。
    真实项目换成 client.embeddings.create(...) 即可。
    """
    # 简单分词：按空格和标点切分，转小写
    cleaned = text.lower()
    for ch in "，。！？,.!?;:\"'()[]{}（）【】":
        cleaned = cleaned.replace(ch, " ")
    words = cleaned.split()

    vec: Embedding = {}
    for w in words:
        if len(w) > 1 and all("\u4e00" <= c <= "\u9fff" for c in w):
            # 连续中文字符（无空格分隔）→ 拆成单字，模拟基础分词
            for c in w:
                vec[c] = vec.get(c, 0.0) + 1.0
        else:
            vec[w] = vec.get(w, 0.0) + 1.0
    return vec


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """两个稀疏向量的余弦相似度，范围 [-1, 1]，1 = 完全相同。

    纯 Python 实现，不用 numpy。用 dict 键交集算点积。
    """
    # 点积：只在共有的词上累加
    dot = sum(a[w] * b.get(w, 0.0) for w in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorMemory:
    """用词频向量 + 余弦相似度检索的长期记忆。

    存储 [(text, embedding)]，search(query) 返回最相似的 top_k 条文本。
    纯 Python 实现，不依赖 Chroma/Pinecone/numpy。
    """

    def __init__(self) -> None:
        self._store: List[Tuple[str, Embedding]] = []

    def add(self, text: str) -> None:
        """添加一条文本，自动计算 embedding 并存储。"""
        emb = simple_embedding(text)
        self._store.append((text, emb))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """检索与 query 最相似的 top_k 条文本，返回 [(text, score)]。

        Args:
            query: 查询文本。
            top_k: 返回的最大条数。

        Returns:
            [(text, similarity_score)] 按相似度降序。score ∈ [0, 1]。
        """
        q_emb = simple_embedding(query)
        scored = [(text, cosine_similarity(q_emb, emb)) for text, emb in self._store]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """返回存储的文本数量。"""
        return len(self._store)

    def clear(self) -> None:
        """清空所有存储。"""
        self._store.clear()


# ════════════════════════════════════════════════════════════════════
# Demo 1: ConversationBuffer — 多轮对话记忆
# ════════════════════════════════════════════════════════════════════


def demo_conversation_buffer() -> None:
    """演示 ConversationBuffer：跨多轮记住用户名和偏好（纯内存，不调 API）。"""
    print(f"\n{'='*60}")
    print("Demo 1: ConversationBuffer — 多轮对话记忆")
    print(f"{'='*60}")

    buffer = ConversationBuffer()

    # 模拟一段多轮对话
    buffer.add("system", "你是任务助手 Agent，会记住用户信息。")
    buffer.add("user", "你好，我叫小明。")
    buffer.add("assistant", "你好小明！有什么可以帮你的？")
    buffer.add("user", "我最喜欢用 Python 编程。")
    buffer.add("assistant", "记住了！Python 是一门优秀的语言。")

    print(f"OUT:buffer: 消息总数: {buffer.count()}")
    print(f"OUT:buffer: 记忆内容:")
    for i, msg in enumerate(buffer.get_messages(), 1):
        preview = msg["content"][:50]
        print(f"OUT:buffer:   [{i}] {msg['role']}: {preview}")

    # 演示"查"操作：能拿到完整历史
    history = buffer.get_messages()
    print(f"OUT:buffer: ✓ get_messages() 返回 {len(history)} 条（完整历史）")

    # 演示"删"操作
    buffer.clear()
    print(f"OUT:buffer: ✓ clear() 后消息数: {buffer.count()}")

    # 演示 buffer 可复用：新对话
    buffer.add("user", "新对话开始。")
    print(f"OUT:buffer: ✓ 新对话后消息数: {buffer.count()}（独立于旧对话）")

    print(f"OUT:buffer: 💡 Buffer 适合短对话（<20轮），长对话需 Summary 或 Vector。")


# ════════════════════════════════════════════════════════════════════
# Demo 2: SummaryMemory — 超阈值自动摘要压缩
# ════════════════════════════════════════════════════════════════════


def demo_summary_memory() -> None:
    """演示 SummaryMemory：消息超过阈值时自动摘要压缩旧历史。

    先尝试真实 API 摘要，失败降级 mock 摘要，保证演示完整。
    """
    print(f"\n{'='*60}")
    print("Demo 2: SummaryMemory — 超阈值自动摘要压缩")
    print(f"{'='*60}")
    print("[说明] 设 max_messages=6，超过时把最早 2 条送去摘要。")
    print("[说明] 先试真实 API 摘要，失败降级 mock 摘要（不依赖 API key）。")

    memory = SummaryMemory(max_messages=6, system_prompt="你是任务助手 Agent。")

    # 模拟一段会触发摘要的长对话（8 轮 > 阈值 6）
    conversation = [
        ("user", "你好，我叫小明，住在北京。"),
        ("assistant", "你好小明！北京是个好地方。"),
        ("user", "我喜欢用 Python 编程，特别是做数据分析。"),
        ("assistant", "Python 在数据分析领域很强大！"),
        ("user", "我最近在学机器学习，用 scikit-learn。"),
        ("assistant", "scikit-learn 是经典 ML 库，选择不错。"),
        ("user", "能推荐一个 Python 的可视化库吗？"),
        ("assistant", "推荐 matplotlib 和 seaborn，适合数据分析。"),
    ]

    print(f"\nOUT:summary: 逐条添加对话（共 {len(conversation)} 条，阈值 {memory._max}）:")
    for role, content in conversation:
        before_count = memory.count()
        memory.add(role, content)
        after_count = memory.count()
        summary_len = len(memory.get_summary())
        compressed = "触发了摘要！" if before_count >= memory._max and after_count < before_count else ""
        print(
            f"OUT:summary: +[{role}] {content[:30]}... "
            f"(原文数: {before_count}→{after_count}, 摘要长度: {summary_len}) {compressed}"
        )

    print(f"\nOUT:summary: 最终 get_messages() 返回:")
    final_msgs = memory.get_messages()
    for i, msg in enumerate(final_msgs, 1):
        preview = msg["content"][:60]
        print(f"OUT:summary:   [{i}] {msg['role']}: {preview}")

    print(f"\nOUT:summary: 累积摘要内容:")
    print(f"OUT:summary:   {memory.get_summary()[:100]}")
    print(f"OUT:summary: ✓ 旧历史被压缩进摘要，近期消息保留原文。")
    print(f"OUT:summary: 💡 Summary 平衡了上下文长度与信息密度。")


# ════════════════════════════════════════════════════════════════════
# Demo 3: VectorMemory — 语义检索（词频向量 + 余弦相似度）
# ════════════════════════════════════════════════════════════════════


def demo_vector_memory() -> None:
    """演示 VectorMemory：用词频向量 + 余弦相似度检索相关内容（100% 离线）。"""
    print(f"\n{'='*60}")
    print("Demo 3: VectorMemory — 语义检索（词频向量+余弦相似度）")
    print(f"{'='*60}")
    print("[说明] 用词频向量模拟 embedding，纯 Python，不调 embedding API。")

    vm = VectorMemory()

    # 添加一批"知识库"文本
    knowledge_base = [
        "小明喜欢用 Python 编程",
        "北京今天的天气是晴天 25 度",
        "机器学习是人工智能的分支",
        "Python 是 Guido 创建的编程语言",
        "上海今天下雨 30 度",
        "用户偏好用 Python 做数据分析",
    ]

    print(f"\nOUT:vector: 添加 {len(knowledge_base)} 条知识:")
    for text in knowledge_base:
        vm.add(text)
        print(f"OUT:vector:   + {text}")

    # 测试检索 1：与"编程"相关
    print(f"\nOUT:vector: 检索 1: query='Python 编程'")
    results = vm.search("Python 编程", top_k=3)
    for text, score in results:
        print(f"OUT:vector:   [{score:.3f}] {text}")

    # 测试检索 2：与"天气"相关
    print(f"\nOUT:vector: 检索 2: query='今天天气怎么样'")
    results = vm.search("今天天气怎么样", top_k=3)
    for text, score in results:
        print(f"OUT:vector:   [{score:.3f}] {text}")

    # 测试检索 3：完全无关的 query
    print(f"OUT:vector: 检索 3: query='音乐推荐'（无关查询）")
    results = vm.search("音乐推荐", top_k=3)
    for text, score in results:
        print(f"OUT:vector:   [{score:.3f}] {text}")

    print(f"\nOUT:vector: 余弦相似度验证:")
    a = simple_embedding("Python 编程")
    b = simple_embedding("Python 编程")
    print(f"OUT:vector:   cosine('Python 编程', 'Python 编程') = {cosine_similarity(a, b):.3f} (应为1.0)")
    c = simple_embedding("天气")
    d = simple_embedding("Python 编程")
    print(f"OUT:vector:   cosine('天气', 'Python 编程') = {cosine_similarity(c, d):.3f} (应较低)")
    print(f"OUT:vector: ✓ 词频向量能捕捉关键词重叠，近似语义检索。")
    print(f"OUT:vector: 💡 真实项目换成 embedding API，检索质量会大幅提升。")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print(f"[config] 章节主题: 记忆系统（ConversationBuffer + SummaryMemory + VectorMemory）")

    # Demo 1: ConversationBuffer（纯内存，不调 API）
    demo_conversation_buffer()

    # Demo 2: SummaryMemory（先试真实 API，失败降级 mock）
    demo_summary_memory()

    # Demo 3: VectorMemory（纯 Python 词频向量，100% 离线）
    demo_vector_memory()

    print(f"\n{'='*60}")
    print("所有演示完成！三种记忆系统均已展示。")
    print("💡 核心要点：LLM 无状态，记忆=你替模型管理 messages 的机制。")
    print("   - Buffer: 完整保留（短对话）")
    print("   - Summary: 超阈值摘要压缩（中等对话）")
    print("   - Vector: 语义检索召回（长期/跨会话）")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
