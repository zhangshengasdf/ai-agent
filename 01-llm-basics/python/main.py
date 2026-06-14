"""第01章 · LLM 基础 —— 与模型对话的第一步

演示内容：
1. 单轮对话 —— 用 system prompt 定义「任务助手 Agent」人格
2. Token 用量 —— 打印 prompt_tokens / completion_tokens
3. 温度对比 —— 同一问题用 temperature=0.0 和 1.0 各调一次
4. 流式输出 —— 逐 token 打印模型回答

所有输出以 "OUT:" 前缀标记，便于 QA 脚本过滤 tsx 的 dotenvx 横幅。
"""

import sys
from pathlib import Path

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI
from shared.config import get_config

# ── 初始化客户端 ────────────────────────────────────────────────────
cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

# ── 任务助手 Agent 的 system prompt ─────────────────────────────────
SYSTEM_PROMPT = (
    "你是一个「任务助手 Agent」——一个简洁、高效的任务管理助手。"
    "用户会问你关于待办、日程、任务优先级的问题。"
    "回答要简明扼要，直接给出建议，不要废话。"
    "如果用户的问题与任务管理无关，简短回答后提醒他你只擅长任务管理。"
)

USER_MESSAGE = "我今天有三个会要开，还有一个报告要写，怎么安排优先级？"


# ═══════════════════════════════════════════════════════════════════
# Demo 1: 单轮对话
# ═══════════════════════════════════════════════════════════════════
def demo_single_turn() -> None:
    """用 system prompt 定义人格，发送一条 user 消息，打印回复。"""
    print("=" * 60)
    print("OUT: [Demo 1] 单轮对话 —— 任务助手 Agent")
    print("=" * 60)

    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_MESSAGE},
        ],
    )

    answer = response.choices[0].message.content
    print(f"OUT: \n[用户] {USER_MESSAGE}")
    print(f"OUT: \n[任务助手] {answer}")

    # ── Demo 2: Token 用量 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("OUT: [Demo 2] Token 用量")
    print("=" * 60)
    usage = response.usage
    print(f"OUT: prompt_tokens     = {usage.prompt_tokens}")
    print(f"OUT: completion_tokens = {usage.completion_tokens}")
    print(f"OUT: total_tokens      = {usage.total_tokens}")


# ═══════════════════════════════════════════════════════════════════
# Demo 3: 温度对比
# ═══════════════════════════════════════════════════════════════════
def demo_temperature_comparison() -> None:
    """同一问题用 temperature=0.0 和 1.0 各调一次，打印对比。"""
    print("\n" + "=" * 60)
    print("OUT: [Demo 3] 温度对比 —— 同一问题，不同温度")
    print("=" * 60)

    question = "用一句话解释什么是 Agent。"

    for temp in [0.0, 1.0]:
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=temp,
        )
        answer = response.choices[0].message.content
        print(f"OUT: \n[temperature={temp}] {answer}")


# ═══════════════════════════════════════════════════════════════════
# Demo 4: 流式输出
# ═══════════════════════════════════════════════════════════════════
def demo_streaming() -> None:
    """用 stream=True 流式打印逐 token 输出。"""
    print("\n" + "=" * 60)
    print("OUT: [Demo 4] 流式输出 —— 逐 token 打印")
    print("=" * 60)
    print("OUT: \n[任务助手] ", end="", flush=True)

    stream = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "流式输出的好处是什么？用两句话回答。"},
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

    print()  # 换行


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"OUT: 提供商: {cfg.provider} | 模型: {cfg.model}")
    print()

    demo_single_turn()           # Demo 1 + 2
    demo_temperature_comparison() # Demo 3
    demo_streaming()             # Demo 4

    print("\nOUT: ✅ 所有演示完成！")
