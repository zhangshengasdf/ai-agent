"""第11章 上下文工程（Context Engineering）

Prompt 工程的进化——当 Agent 跑几十步、调上百次工具时，"写好一个 prompt"不够了。
真正决定质量的是：你如何管理喂给模型的上下文。

本章实现上下文工程的三大支柱：
  1. ContextCompactor: 上下文压缩——超 token 阈值时把旧轨迹摘要成一条 system 消息
  2. SubAgent 隔离: 主 Agent 派子 Agent 干重活，只收回摘要（不看全量轨迹）
  3. TokenBudget: token 预算管理——每轮检查，接近上限时自动触发压缩

离线设计：
  - Token 估算用纯 Python（字符数 // 3），不依赖 tiktoken，100% 离线
  - 压缩/子 Agent：先试真实 API，失败降级 mock（预设文本），保证演示完整
  - 预算循环：纯本地模拟对话，100% 离线
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


# ════════════════════════════════════════════════════════════════════
# 通用类型 & Token 估算
# ════════════════════════════════════════════════════════════════════

Message = Dict[str, str]


def estimate_tokens(messages: List[Message]) -> int:
    """估算 messages 的 token 数（粗略，1 token ≈ 3 字符）。

    纯 Python 实现，不依赖 tiktoken。对教学足够（误差 ±20%），
    生产环境换成 tiktoken.encoding_for_model(model).encode(text)。
    """
    text = json.dumps(messages, ensure_ascii=False)
    return len(text) // 3


# ════════════════════════════════════════════════════════════════════
# 1. ContextCompactor — 上下文压缩
# ════════════════════════════════════════════════════════════════════


class ContextCompactor:
    """超 token 阈值时把旧轨迹摘要压缩的上下文管理器。

    当 estimate_tokens(messages) 超过 threshold 时，把最早的旧消息
    送去 LLM 摘要，压缩成一条累积摘要（summary），只保留最近 keep_recent 条原文。
    最终传给 API 的是：[摘要 system msg] + [最近 N 条原文]。
    """

    def __init__(self, threshold: int = 2000, keep_recent: int = 6) -> None:
        self.threshold = threshold  # token 阈值
        self.keep_recent = keep_recent  # 保留最近几条原文
        self._messages: List[Message] = []
        self._summary: str = ""

    def add(self, message: Message) -> bool:
        """追加消息，返回是否触发了压缩。"""
        self._messages.append(message)
        if estimate_tokens(self._messages) > self.threshold:
            self._compact()
            return True
        return False

    def _compact(self) -> None:
        """把旧消息摘要，保留最近 keep_recent 条原文。"""
        # 保留最后 keep_recent 条，其余送去摘要
        split = max(self.keep_recent, 1)
        old = self._messages[:-split]
        recent = self._messages[-split:]

        if not old:
            return  # 没有旧消息可压缩

        new_summary = self._llm_summarize(old)
        if self._summary:
            self._summary = f"{self._summary}\n{new_summary}"
        else:
            self._summary = new_summary
        self._messages = recent

    def _llm_summarize(self, messages: List[Message]) -> str:
        """调 LLM 摘要，失败时降级为离线 mock 摘要。"""
        try:
            text = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {
                        "role": "system",
                        "content": "请用一段话（不超过80字）总结以下对话的要点：",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content or "(摘要为空)"
        except Exception:
            return self._mock_summarize(messages)

    @staticmethod
    def _mock_summarize(messages: List[Message]) -> str:
        """离线 mock 摘要：提取关键词 + 片段模拟压缩结果。"""
        keywords: List[str] = []
        all_text = " ".join(m.get("content", "") for m in messages)
        for kw in [
            "Python", "Agent", "框架", "LangChain", "工具", "记忆",
            "小明", "北京", "天气", "研究", "压缩", "上下文",
        ]:
            if kw in all_text:
                keywords.append(kw)
        kw_str = "、".join(keywords[:5]) if keywords else "对话内容"
        snippet = all_text[:40].replace("\n", " ")
        return f"[摘要] 涉及{kw_str}。要点片段: {snippet}..."

    def get_messages(self) -> List[Message]:
        """返回 [摘要 system msg（如有）] + [最近 N 条原文]。"""
        result: List[Message] = []
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
        """返回原文消息数（不含摘要）。"""
        return len(self._messages)


# ════════════════════════════════════════════════════════════════════
# 2. SubAgent 隔离 — 主 Agent 派子 Agent，只收回摘要
# ════════════════════════════════════════════════════════════════════


class SubAgent:
    """模拟一个子 Agent：内部有多步轨迹，但只向外暴露摘要。

    真实场景中子 Agent 自己跑 agent_loop（第04章），有完整的工具调用轨迹。
    但主 Agent 不看这些轨迹——只看 SubAgent.get_summary() 返回的一句话摘要。
    这就是"隔离"：子 Agent 的几千 token 轨迹不进入主 Agent 上下文。
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._trace: List[Message] = []

    def run_research(self, topic: str) -> str:
        """模拟子 Agent 研究一个主题（多步轨迹），返回摘要。

        内部模拟 6 步：搜索→阅读→分析→补充搜索→整理→输出。
        这些步骤的完整轨迹存在 _trace 里，但主 Agent 看不到。
        """
        steps = [
            ("search", f"搜索 '{topic}' 的基础信息..."),
            ("read", f"阅读关于 {topic} 的 3 篇核心文章..."),
            ("analyze", f"分析 {topic} 的关键特征和应用场景..."),
            ("search", f"补充搜索 '{topic} 最新趋势 2025'..."),
            ("synthesize", f"整理 {topic} 的要点：定义、应用、趋势..."),
            ("output", f"输出 {topic} 的研究摘要。"),
        ]
        # 模拟每步的"原始结果"（真实场景是几千字的搜索结果）
        self._trace.append({"role": "system", "content": f"子Agent[{self.name}] 开始研究: {topic}"})
        for action, detail in steps:
            self._trace.append({"role": "assistant", "content": f"[{action}] {detail}"})
            # 模拟工具返回的大段原始数据
            self._trace.append(
                {"role": "tool", "content": f"{topic}相关数据: " + detail * 5}
            )

        # 生成摘要（先试真实 LLM，失败降级 mock）
        summary = self._summarize_for_parent(topic)
        return summary

    def _summarize_for_parent(self, topic: str) -> str:
        """生成给主 Agent 的摘要（1-2 句话）。"""
        try:
            trace_text = "\n".join(m["content"] for m in self._trace)
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"你刚完成对'{topic}'的研究。请用1-2句话总结核心发现（不超过60字）：",
                    },
                    {"role": "user", "content": trace_text},
                ],
                max_tokens=100,
            )
            return resp.choices[0].message.content or self._mock_summary(topic)
        except Exception:
            return self._mock_summary(topic)

    @staticmethod
    def _mock_summary(topic: str) -> str:
        """离线 mock 摘要（预设文本，模拟 LLM 压缩结果）。"""
        return f"{topic}的核心：它是当前Agent领域的关键技术，已有多款主流框架支持，2025年趋势是工具调用+记忆融合。"

    def trace_token_count(self) -> int:
        """返回子 Agent 完整轨迹的 token 数（用于对比展示）。"""
        return estimate_tokens(self._trace)

    def trace_step_count(self) -> int:
        """返回子 Agent 轨迹的消息条数。"""
        return len(self._trace)


def run_main_agent_with_subagents() -> Dict[str, int]:
    """模拟主 Agent 用子 Agent 隔离研究 3 个主题。

    对比：主 Agent 上下文只含 3 段摘要，远小于不隔离时的全量轨迹。
    """
    topics = ["LangChain 框架", "ReAct 推理模式", "向量记忆系统"]
    main_messages: List[Message] = [
        {"role": "system", "content": "你是研究助手 Agent。"},
        {"role": "user", "content": "研究这3个主题并汇总: " + ", ".join(topics)},
    ]

    subagent_total_tokens = 0

    for topic in topics:
        sub = SubAgent(name=f"researcher-{topic[:4]}")
        summary = sub.run_research(topic)
        trace_tokens = sub.trace_token_count()
        subagent_total_tokens += trace_tokens

        # 关键：主 Agent 只收摘要，不收完整轨迹
        main_messages.append(
            {
                "role": "assistant",
                "content": f"子Agent研究了'{topic}'，发现: {summary}",
            }
        )

    # 主 Agent 汇总
    main_messages.append(
        {
            "role": "assistant",
            "content": "三个主题研究完毕，共同点是都涉及Agent的核心能力。",
        }
    )

    main_tokens = estimate_tokens(main_messages)
    return {
        "main_tokens": main_tokens,
        "subagent_total_tokens": subagent_total_tokens,
        "main_msg_count": len(main_messages),
    }


# ════════════════════════════════════════════════════════════════════
# 3. TokenBudget — token 预算管理
# ════════════════════════════════════════════════════════════════════


class TokenBudget:
    """token 预算管理器：每轮检查，接近上限时自动触发压缩。

    模拟上下文窗口限制（如 4000 tokens），当用量超过 threshold_ratio
    （默认 80%）时触发 ContextCompactor 压缩。
    """

    def __init__(
        self,
        budget: int = 4000,
        threshold_ratio: float = 0.8,
        keep_recent: int = 6,
    ) -> None:
        self.budget = budget
        self.threshold_ratio = threshold_ratio
        self.threshold = int(budget * threshold_ratio)
        self._compactor = ContextCompactor(
            threshold=self.threshold, keep_recent=keep_recent
        )
        self.compaction_count = 0

    def add(self, message: Message) -> Dict[str, int]:
        """添加消息，返回当前 token 状态 + 是否压缩了。"""
        compacted = self._compactor.add(message)
        if compacted:
            self.compaction_count += 1
        tokens = self.current_tokens()
        return {
            "tokens": tokens,
            "budget": self.budget,
            "usage": tokens / self.budget,
            "compacted": 1 if compacted else 0,
        }

    def current_tokens(self) -> int:
        """返回当前上下文（含摘要）的 token 数。"""
        return estimate_tokens(self._compactor.get_messages())

    def get_messages(self) -> List[Message]:
        return self._compactor.get_messages()


# ════════════════════════════════════════════════════════════════════
# Demo 1: Token 估算（纯离线）
# ════════════════════════════════════════════════════════════════════


def demo_token_estimation() -> None:
    """演示 token 估算：对比不同长度消息的 token 数（纯离线）。"""
    print(f"\n{'='*60}")
    print("Demo 1: Token 估算（纯字符数 // 3，不依赖 tiktoken）")
    print(f"{'='*60}")

    test_cases: List[List[Message]] = [
        [{"role": "user", "content": "你好"}],
        [{"role": "user", "content": "请用Python写一个快速排序算法，并解释其时间复杂度。"}],
        [
            {"role": "system", "content": "你是助手。"},
            {"role": "user", "content": "研究AI Agent框架的趋势，包括LangChain、LangGraph等。"},
        ],
    ]

    print(f"OUT:token: 测试 {len(test_cases)} 组消息:")
    for i, msgs in enumerate(test_cases, 1):
        tokens = estimate_tokens(msgs)
        chars = len(json.dumps(msgs, ensure_ascii=False))
        preview = msgs[-1]["content"][:30]
        print(f"OUT:token:   [{i}] {tokens} tokens ({chars} chars) | {preview}...")

    # 批量增长演示
    print(f"\nOUT:token: 消息数增长 → token 增长（模拟对话累积）:")
    growing: List[Message] = [{"role": "system", "content": "你是任务助手 Agent。"}]
    for n in [1, 5, 10, 20, 50]:
        while len(growing) < n + 1:
            growing.append(
                {"role": "user", "content": f"第{len(growing)}轮对话：请帮我处理任务。"}
            )
            growing.append(
                {"role": "assistant", "content": f"好的，我来处理第{len(growing)}轮的任务。"}
            )
        tokens = estimate_tokens(growing)
        print(f"OUT:token:   {len(growing):3d} 条消息 → {tokens:5d} tokens")

    print(f"OUT:token: ✓ 纯字符估算，零依赖，可离线验证上下文规模。")
    print(f"OUT:token: 💡 生产环境换 tiktoken 可获得精确值（误差 <1%）。")


# ════════════════════════════════════════════════════════════════════
# Demo 2: ContextCompactor — 上下文压缩
# ════════════════════════════════════════════════════════════════════


def demo_context_compaction() -> None:
    """演示上下文压缩：模拟 10 轮对话，超阈值时触发压缩，对比前后 token。"""
    print(f"\n{'='*60}")
    print("Demo 2: ContextCompactor — 上下文压缩")
    print(f"{'='*60}")
    print("[说明] 设 threshold=2000 tokens，keep_recent=6。")
    print("[说明] 先试真实 API 摘要，失败降级 mock 摘要。")

    compactor = ContextCompactor(threshold=2000, keep_recent=6)

    # 模拟一段真实研究对话：每轮含大段工具结果（搜索返回），token 增长快
    big_result_1 = (
        "搜索结果: asyncio是Python 3.4引入的异步IO库。核心组件包括 event loop、"
        "coroutine、task、future。用 async def 定义协程函数，await 等待协程完成。"
        "与多线程相比，asyncio 单线程并发，无锁，适合IO密集场景。"
        "常见用法：aiohttp异步HTTP、aiofiles异步文件、asyncpg异步PG。"
    ) * 10  # 模拟真实搜索结果页（~800 tokens）
    big_result_2 = (
        "搜索结果: LangChain的Agent支持异步工具。用 @tool 装饰器可定义 async 工具，"
        "AgentExecutor 内部用 asyncio.gather 并发执行独立工具调用。"
        "LangGraph 进一步支持流式执行和中断恢复。"
        "注意：同步工具和异步工具混用时，框架会自动适配，但推荐统一用 async。"
    ) * 10  # 模拟真实搜索结果页（~800 tokens）
    big_result_3 = (
        "搜索结果: 向量记忆系统的 embedding API 调用是IO密集操作，应该用 async。"
        "OpenAI SDK 支持 async client：AsyncOpenAI。检索时用 await client.embeddings.create。"
        "批量 embedding 用 asyncio.gather 并发，比串行快 5-10 倍。"
        "上下文压缩同理：调LLM摘要用 async，不阻塞 event loop 上的其他工具。"
    ) * 10  # 模拟真实搜索结果页（~800 tokens）

    conversation = [
        ("user", "我想了解 Python 的异步编程，asyncio 怎么用？"),
        ("assistant", f"asyncio 是 Python 的异步IO库。我来查详细资料。\n{big_result_1}"),
        ("user", "能解释一下 event loop 吗？它和线程有什么区别？"),
        ("assistant", f"event loop 是核心。补充搜索结果:\n{big_result_2}"),
        ("user", "Agent 开发里怎么用异步？我听说 LangChain 支持异步工具。"),
        ("assistant", f"LangChain 支持。详细资料:\n{big_result_2}"),
        ("user", "那记忆系统呢？VectorMemory 的检索可以异步吗？"),
        ("assistant", f"可以异步。详细:\n{big_result_3}"),
        ("user", "上下文压缩也是这个原理吧？压缩时主 Agent 可以等。"),
        ("assistant", "对。压缩是IO密集操作（调LLM摘要），用 async 不阻塞其他工具。"),
    ]

    print(f"\nOUT:compact: 逐条添加对话（共 {len(conversation)} 条，阈值 {compactor.threshold} tokens）:")
    for role, content in conversation:
        before_tokens = estimate_tokens(compactor.get_messages())
        compacted = compactor.add({"role": role, "content": content})
        after_tokens = estimate_tokens(compactor.get_messages())
        flag = " ⚡触发了压缩!" if compacted else ""
        print(
            f"OUT:compact: +[{role:9s}] tokens: {before_tokens:4d}→{after_tokens:4d}"
            f" (原文{compactor.count()}条){flag}"
        )

    print(f"\nOUT:compact: 最终上下文（摘要 + 最近{compactor.keep_recent}条原文）:")
    final_msgs = compactor.get_messages()
    for i, msg in enumerate(final_msgs, 1):
        preview = msg["content"][:60]
        print(f"OUT:compact:   [{i}] {msg['role']}: {preview}")

    print(f"\nOUT:compact: 累积摘要:")
    print(f"OUT:compact:   {compactor.get_summary()[:120]}")
    final_tokens = estimate_tokens(final_msgs)
    print(f"OUT:compact: ✓ 最终 {final_tokens} tokens，旧轨迹被压缩进摘要。")
    print(f"OUT:compact: 💡 压缩把 token 从'线性增长'变成'有上限'，避免质量衰退。")


# ════════════════════════════════════════════════════════════════════
# Demo 3: SubAgent 隔离
# ════════════════════════════════════════════════════════════════════


def demo_subagent_isolation() -> None:
    """演示子 Agent 隔离：主 Agent 只收摘要，不看子 Agent 全量轨迹。"""
    print(f"\n{'='*60}")
    print("Demo 3: SubAgent 隔离 — 主 Agent 只收摘要")
    print(f"{'='*60}")
    print("[说明] 主 Agent 派 3 个子 Agent 研究，每个子 Agent 内部有 6 步轨迹。")
    print("[说明] 主 Agent 只收每子 Agent 的 1 句摘要。")

    # 先展示单个子 Agent 的内部轨迹
    sub = SubAgent(name="demo-researcher")
    summary = sub.run_research("LangChain 框架")
    print(f"\nOUT:subagent: 单个子 Agent 内部轨迹（主 Agent 看不到）:")
    print(f"OUT:subagent:   轨迹消息数: {sub.trace_step_count()} 条")
    print(f"OUT:subagent:   轨迹 token 数: {sub.trace_token_count()} tokens")
    print(f"OUT:subagent:   主 Agent 收到的摘要: {summary}")
    print(f"OUT:subagent:   摘要 token 数: {estimate_tokens([{'role':'assistant','content':summary}])}")

    # 完整的主 Agent + 3 个子 Agent
    print(f"\nOUT:subagent: 主 Agent 派 3 个子 Agent 研究（对比上下文大小）:")
    result = run_main_agent_with_subagents()
    print(f"OUT:subagent:   主 Agent 上下文: {result['main_tokens']} tokens ({result['main_msg_count']} 条消息)")
    print(f"OUT:subagent:   3个子Agent总轨迹: {result['subagent_total_tokens']} tokens（被隔离）")
    print(f"OUT:subagent:   隔离节省: {result['subagent_total_tokens'] - result['main_tokens']} tokens 不进主上下文")

    ratio = result["main_tokens"] / max(result["subagent_total_tokens"], 1) * 100
    print(f"OUT:subagent:   主上下文仅为子轨迹的 {ratio:.1f}%")
    print(f"OUT:subagent: ✓ 隔离让主 Agent 上下文保持干净，只看摘要不看原始搜索结果。")
    print(f"OUT:subagent: 💡 反模式：把子 Agent 全量轨迹塞回主 Agent = 丧失隔离意义。")


# ════════════════════════════════════════════════════════════════════
# Demo 4: TokenBudget — 完整的"对话→检查→压缩→继续"循环
# ════════════════════════════════════════════════════════════════════


def demo_token_budget() -> None:
    """演示 token 预算循环：模拟多轮对话，超 80% 自动压缩。"""
    print(f"\n{'='*60}")
    print("Demo 4: TokenBudget — 预算管理与自动压缩循环")
    print(f"{'='*60}")
    print("[说明] budget=1500 tokens（demo 缩小值，便于 12 轮内观察），超 80%（1200）触发压缩。")
    print("[说明] 模拟含大段解释的真实对话，观察 token 变化和自动压缩。")

    budget = TokenBudget(budget=1500, threshold_ratio=0.8, keep_recent=6)

    # 模拟含大段工具结果的真实对话（每轮 token 增长快，会触发压缩）
    detail_1 = (
        "详细回答: 工具调用是让 LLM 决定调用哪个函数。你定义工具的 JSON Schema，"
        "传给 API 的 tools 参数。模型分析用户意图后，返回结构化的 tool_calls 字段，"
        "包含函数名和参数。你执行该函数，把结果以 role=tool 追加到 messages，"
        "再调一次 API 让模型看结果。和普通函数调用的区别：调用决策由模型做，不是硬编码。"
    ) * 8  # 模拟含大段解释的真实回答（~900 tokens）
    detail_2 = (
        "详细回答: Agent 循环让模型多步推理。结构：for step in range(MAX_STEPS)，"
        "每步调 LLM→看有无 tool_calls→有则执行并追加结果→无则终止。"
        "为什么不能单次调用：复杂任务需要多步（查天气+查日历+综合判断），"
        "单次调用模型无法获得工具结果反馈。max_steps 防止无限循环。"
    ) * 8  # 模拟含大段解释的真实回答（~900 tokens）
    detail_3 = (
        "详细回答: ReAct=Reason+Act。显式版模型输出 Thought/Action/Observation 文本，"
        "你用正则解析。隐式版用 tools API，模型输出结构化 tool_calls。"
        "显式版推理过程可见可调试，但格式脆弱。隐式版结构稳定但推理黑盒。"
        "现代框架默认用隐式，但理解显式能看透底层。"
    ) * 8  # 模拟含大段解释的真实回答（~900 tokens）
    detail_4 = (
        "详细回答: 记忆系统选择：ConversationBuffer 完整保留，适合短对话（<20轮）。"
        "SummaryMemory 超阈值摘要压缩，适合中等对话（20-100轮）。"
        "VectorMemory 词频/embedding 向量+余弦相似度检索，适合长期/知识库。"
        "组合使用最常见：当前会话用 Buffer，用户画像用 Summary，知识库用 Vector。"
    ) * 8  # 模拟含大段解释的真实回答（~900 tokens）

    conversation = [
        ("user", "你好，我想学习 AI Agent 开发，从哪里开始？"),
        ("assistant", "建议从基础开始：先学 LLM API 调用，再学工具调用，最后学 Agent 循环。"),
        ("user", "工具调用是什么意思？和普通函数调用有什么区别？"),
        ("assistant", detail_1),
        ("user", "Agent 循环又是啥？为什么不能单次调用搞定？"),
        ("assistant", detail_2),
        ("user", "ReAct 推理是什么？和隐式工具调用有什么不同？"),
        ("assistant", detail_3),
        ("user", "记忆系统怎么选？Buffer、Summary、Vector 各适合什么？"),
        ("assistant", detail_4),
        ("user", "上下文工程又是什么？它和Prompt工程啥关系？"),
        ("assistant", "上下文工程是Prompt工程的进化：主动管理每次调用模型看到什么。"),
    ]

    print(f"\nOUT:budget: 预算循环演示（{len(conversation)} 轮对话）:")
    for i, (role, content) in enumerate(conversation, 1):
        status = budget.add({"role": role, "content": content})
        bar_filled = int(status["usage"] * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        flag = " ⚡压缩!" if status["compacted"] else ""
        print(
            f"OUT:budget: 轮{i:2d} [{role:9s}] {bar} {status['tokens']:4d}/{status['budget']} "
            f"({status['usage']:.0%}){flag}"
        )

    print(f"\nOUT:budget: 总压缩次数: {budget.compaction_count}")
    final = budget.get_messages()
    final_tokens = budget.current_tokens()
    print(f"OUT:budget: 最终上下文: {len(final)} 条消息, {final_tokens} tokens")
    print(f"OUT:budget: ✓ 预算循环让上下文始终在健康范围内，自动避免超限。")
    print(f"OUT:budget: 💡 这是所有长任务 Agent 的基础设施——没有它，Agent 跑久了必然衰退。")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    print(f"[config] provider={cfg.provider}, model={cfg.model}")
    print(f"[config] 章节主题: 上下文工程（压缩 + 子Agent隔离 + Token预算）")

    # Demo 1: Token 估算（纯离线）
    demo_token_estimation()

    # Demo 2: 上下文压缩（先试真实 API，失败降级 mock）
    demo_context_compaction()

    # Demo 3: 子 Agent 隔离（纯离线 mock 轨迹 + 先试 API 摘要）
    demo_subagent_isolation()

    # Demo 4: Token 预算循环（纯本地模拟）
    demo_token_budget()

    print(f"\n{'='*60}")
    print("所有演示完成！上下文工程三大支柱均已展示。")
    print("💡 核心要点：上下文是有限昂贵的资源，必须主动管理。")
    print("   - 压缩: 超阈值摘要旧轨迹（本章 ContextCompactor）")
    print("   - 隔离: 子 Agent 只回摘要（本章 SubAgent）")
    print("   - 预算: 超上限自动触发压缩（本章 TokenBudget）")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
