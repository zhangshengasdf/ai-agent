"""项目1 · 深度研究助手（Plan-and-Execute + RAG + 多工具）

综合实战：把 Agent 循环、规划、RAG 缝合成一个能"查资料→做笔记→出报告"的研究助手。

核心组件：
  - Plan-and-Execute：LLM 规划研究步骤 → 逐步执行
  - RAG：从 data/ 加载文档 → 分块 → 词频向量化 → 余弦相似度 top-k
  - 多工具：search_knowledge / write_note / get_summary
  - Trace + Cost：每步记录耗时/token/费用，最后打印 trace 树和总成本
  - 离线 Mock：API 不可用 → 预设计划 + 本地检索 + mock 报告，exit 0
"""

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

# 知识库目录（data/ 在项目根目录下）
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 模型价格（USD per 1K tokens，用于成本估算）
INPUT_PRICE = 0.00015
OUTPUT_PRICE = 0.0006


# ════════════════════════════════════════════════════════════════════
# 1. Embedding + 相似度（复用第09章 RAG 模式）
# ════════════════════════════════════════════════════════════════════

Embedding = Dict[str, float]


def simple_embedding(text: str) -> Embedding:
    """词频向量模拟 embedding（教学用）。中文连续字符拆成单字。"""
    cleaned = text.lower()
    for ch in "，。！？,.!?;:\"'()[]{}（）【】\n\r\t#*-`>":
        cleaned = cleaned.replace(ch, " ")
    words = cleaned.split()

    vec: Embedding = {}
    for w in words:
        if len(w) > 1 and all("\u4e00" <= c <= "\u9fff" for c in w):
            for c in w:
                vec[c] = vec.get(c, 0.0) + 1.0
        else:
            vec[w] = vec.get(w, 0.0) + 1.0
    return vec


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """两个稀疏向量的余弦相似度。"""
    dot = sum(a[w] * b.get(w, 0.0) for w in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """把长文本分成带重叠窗口的小块。"""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size
        start += step
    return chunks


# ════════════════════════════════════════════════════════════════════
# 2. 知识库加载 + 索引构建
# ════════════════════════════════════════════════════════════════════


def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    """从 data/ 目录加载所有 .md 文档。"""
    docs: List[Tuple[str, str]] = []
    if not data_dir.exists():
        return docs
    for p in sorted(data_dir.iterdir()):
        if p.suffix in (".md", ".txt"):
            docs.append((p.name, p.read_text(encoding="utf-8")))
    return docs


def build_index(
    documents: List[Tuple[str, str]], chunk_size: int = 200, overlap: int = 50
) -> List[Tuple[str, Embedding]]:
    """构建 (chunk_text, embedding) 索引。"""
    index: List[Tuple[str, Embedding]] = []
    for _filename, content in documents:
        for chunk in chunk_text(content, chunk_size, overlap):
            index.append((chunk, simple_embedding(chunk)))
    return index


# ════════════════════════════════════════════════════════════════════
# 3. Trace + Cost 追踪
# ════════════════════════════════════════════════════════════════════


@dataclass
class TraceStep:
    """单步追踪记录。"""
    name: str
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    detail: str = ""


@dataclass
class Tracer:
    """追踪所有步骤的耗时和成本。"""
    steps: List[TraceStep] = field(default_factory=list)

    def record(
        self,
        name: str,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        detail: str = "",
    ) -> None:
        cost = (input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE) / 1000
        self.steps.append(
            TraceStep(
                name=name,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                detail=detail,
            )
        )

    def print_tree(self) -> None:
        """打印 trace 树和总成本。"""
        print("\nOUT:trace: ── Trace Tree ──")
        total_cost = 0.0
        total_ms = 0.0
        for i, s in enumerate(self.steps):
            prefix = "  ├─" if i < len(self.steps) - 1 else "  └─"
            tokens_info = ""
            if s.input_tokens > 0 or s.output_tokens > 0:
                tokens_info = f" | tokens: {s.input_tokens}+{s.output_tokens}"
            cost_info = f" | cost: ${s.cost_usd:.6f}" if s.cost_usd > 0 else ""
            print(
                f"OUT:trace: {prefix} [{i+1}] {s.name} "
                f"({s.duration_ms:.0f}ms{tokens_info}{cost_info})"
            )
            if s.detail:
                detail_prefix = "  │  " if i < len(self.steps) - 1 else "     "
                print(f"OUT:trace: {detail_prefix}    {s.detail}")
            total_cost += s.cost_usd
            total_ms += s.duration_ms
        print(f"OUT:trace:")
        print(f"OUT:cost: 总耗时: {total_ms:.0f}ms | 总成本: ${total_cost:.6f}")


# ════════════════════════════════════════════════════════════════════
# 4. 工具定义
# ════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中检索与查询相关的文档片段。用于查找事实、定义、原理等信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "记录一条研究笔记。用于保存从检索或其他来源获得的关键信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "笔记内容"}
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "获取当前所有研究笔记的汇总。用于回顾已收集的信息。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

MOCK_PLAN_STEPS = [
    {
        "step": 1,
        "action": "search_knowledge",
        "query": "涌现能力 大语言模型",
        "goal": "了解 LLM 涌现能力的定义和例子",
    },
    {
        "step": 2,
        "action": "search_knowledge",
        "query": "RAG 检索增强生成 原理",
        "goal": "了解 RAG 的核心流程和优势",
    },
    {
        "step": 3,
        "action": "search_knowledge",
        "query": "多 Agent 系统 协作",
        "goal": "了解多 Agent 架构和协作模式",
    },
]


# ════════════════════════════════════════════════════════════════════
# 5. LLM 调用封装（带 try/catch 降级）
# ════════════════════════════════════════════════════════════════════


def llm_chat(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] | None = None,
    tracer: Tracer | None = None,
    step_name: str = "llm_call",
) -> Dict[str, Any]:
    """调用 LLM，失败时返回 mock 响应。"""
    t0 = time.time()
    try:
        kwargs: Dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        resp = client.chat.completions.create(**kwargs)
        elapsed = (time.time() - t0) * 1000
        msg = resp.choices[0].message
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        if tracer:
            tracer.record(step_name, elapsed, in_tok, out_tok)
        # 返回 dict 方便后续处理
        result: Dict[str, Any] = {"content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]
        return result
    except Exception:
        elapsed = (time.time() - t0) * 1000
        if tracer:
            tracer.record(step_name, elapsed, detail="(offline mock)")
        return {"content": "", "tool_calls": []}


# ════════════════════════════════════════════════════════════════════
# 6. 工具执行
# ════════════════════════════════════════════════════════════════════


def search_knowledge(
    query: str, index: List[Tuple[str, Embedding]], top_k: int = 3
) -> str:
    """RAG 检索：query embedding → cosine similarity → top-k。"""
    q_emb = simple_embedding(query)
    scored = [(chunk, cosine_similarity(q_emb, emb)) for chunk, emb in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    results = scored[:top_k]
    if not results or results[0][1] == 0:
        return "（未找到相关内容）"
    parts = []
    for i, (chunk, score) in enumerate(results):
        parts.append(f"[片段{i+1}] (相似度:{score:.3f}) {chunk}")
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════
# 7. Plan-and-Execute 主流程
# ════════════════════════════════════════════════════════════════════


def plan_research(
    topic: str, tracer: Tracer
) -> List[Dict[str, Any]]:
    """阶段1：让 LLM 规划研究步骤（结构化 JSON）。"""
    system = (
        "你是一个研究规划助手。给定研究主题，输出一个 JSON 数组，每个元素包含 "
        "step(编号)、action(工具名: search_knowledge/write_note/get_summary)、"
        "query(action 为 search_knowledge 时的检索词)、goal(这步的目标)。"
        "输出 3-5 步，只输出 JSON，不要其他文字。"
    )
    print(f"\nOUT:plan: ══ 研究规划 ══")
    print(f"OUT:plan: 主题: {topic}")

    resp = llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": f"研究主题: {topic}"}],
        tracer=tracer,
        step_name="plan",
    )

    # 尝试解析 JSON
    content = resp.get("content", "")
    try:
        # 提取 JSON（可能被 markdown 包裹）
        json_str = content
        if "```" in content:
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("[") or line.startswith("{"):
                    json_str = line
                    break
        steps = json.loads(json_str)
        if isinstance(steps, list) and len(steps) > 0:
            print(f"OUT:plan: LLM 生成了 {len(steps)} 个研究步骤")
            for s in steps:
                print(f"OUT:plan:   步骤{s.get('step', '?')}: {s.get('goal', s.get('query', ''))}")
            return steps
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # 降级：使用预设计划
    print(f"OUT:plan: (offline) 使用预设研究计划，共 {len(MOCK_PLAN_STEPS)} 步")
    for s in MOCK_PLAN_STEPS:
        print(f"OUT:plan:   步骤{s['step']}: {s['goal']}")
    return MOCK_PLAN_STEPS


def execute_research(
    steps: List[Dict[str, Any]],
    index: List[Tuple[str, Embedding]],
    notes: List[str],
    tracer: Tracer,
) -> None:
    """阶段2：逐步执行研究计划。"""
    print(f"\nOUT:step: ══ 逐步执行 ══")

    for step_def in steps:
        step_num = step_def.get("step", "?")
        action = step_def.get("action", "search_knowledge")
        query = step_def.get("query", "")
        goal = step_def.get("goal", "")

        print(f"\nOUT:step: ── 步骤 {step_num}: {goal} ──")

        if action == "search_knowledge":
            t0 = time.time()
            result = search_knowledge(query, index)
            elapsed = (time.time() - t0) * 1000
            tracer.record(f"search_knowledge({query})", elapsed)
            print(f"OUT:search: 查询: {query}")
            # 显示前 200 字符
            preview = result[:200] + ("..." if len(result) > 200 else "")
            print(f"OUT:search: 结果: {preview}")

            # 自动写笔记
            note_content = f"[{goal}] {result[:150]}"
            notes.append(note_content)
            print(f"OUT:note: 记录笔记: {note_content[:80]}...")

        elif action == "write_note":
            content = step_def.get("content", query)
            notes.append(content)
            print(f"OUT:note: 手动笔记: {content[:80]}")

        elif action == "get_summary":
            summary = "\n".join(f"  {i+1}. {n[:60]}" for i, n in enumerate(notes))
            print(f"OUT:note: 笔记汇总 ({len(notes)} 条):\n{summary}")


def generate_report(
    topic: str, notes: List[str], tracer: Tracer
) -> str:
    """阶段3：基于笔记生成研究报告。"""
    print(f"\nOUT:report: ══ 生成报告 ══")

    notes_text = "\n".join(f"- {n}" for n in notes)
    system = "你是一个研究报告撰写助手。根据以下研究笔记，撰写一份简洁的研究报告摘要。"
    user = f"研究主题: {topic}\n\n研究笔记:\n{notes_text}\n\n请输出报告摘要（200字以内）。"

    resp = llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tracer=tracer,
        step_name="report",
    )

    report = resp.get("content", "")
    if not report:
        # 离线 mock：用笔记拼接
        report = f"【{topic} - 研究报告摘要】\n\n"
        for i, note in enumerate(notes):
            report += f"{i+1}. {note[:80]}\n"
        report += "\n(离线模式：基于检索笔记自动生成)"
        print(f"OUT:report: (offline) 使用笔记拼接 mock 报告")
    else:
        print(f"OUT:report: LLM 生成了研究报告")

    print(f"OUT:report: {report[:300]}")
    return report


# ════════════════════════════════════════════════════════════════════
# 8. 主函数
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    topic = "大语言模型、RAG 与多 Agent 系统的融合趋势"
    tracer = Tracer()
    notes: List[str] = []

    # 加载知识库
    print("OUT: ══ 深度研究助手 ══")
    print(f"OUT: 研究主题: {topic}")

    documents = load_documents(DATA_DIR)
    print(f"OUT: 加载了 {len(documents)} 篇文档")

    t0 = time.time()
    index = build_index(documents)
    build_ms = (time.time() - t0) * 1000
    tracer.record("build_index", build_ms)
    print(f"OUT: 构建索引: {len(index)} 个文本块 ({build_ms:.0f}ms)")

    # 阶段1：规划
    steps = plan_research(topic, tracer)

    # 阶段2：执行
    execute_research(steps, index, notes, tracer)

    # 阶段3：报告
    generate_report(topic, notes, tracer)

    # 打印 trace 树
    tracer.print_tree()

    print("\nOUT: ══ 研究完成 ══")


if __name__ == "__main__":
    main()
