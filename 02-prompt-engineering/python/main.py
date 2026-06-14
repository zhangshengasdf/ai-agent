"""第02章 Prompt 工程 — 4 种核心技术演示。

围绕「任务助手 Agent」展开：
1. 无 system vs 有 system — 同一问题，截然不同的回答
2. Few-shot 分类 — 用示例教模型做情感分类
3. Chain-of-Thought — 引导模型逐步推理
4. 结构化输出 — response_format + Pydantic 解析

运行：python3 02-prompt-engineering/python/main.py
"""

import json
import sys
from pathlib import Path

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI
from pydantic import BaseModel

from shared.config import get_config

# ── 初始化客户端 ───────────────────────────────────────────────────
cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

SEPARATOR = "=" * 60


# ── Pydantic 模型：用于场景 4 的结构化解析 ─────────────────────────
class TaskInfo(BaseModel):
    """任务助手返回的结构化任务信息。"""

    title: str
    priority: str  # "high" | "medium" | "low"
    description: str


# ═══════════════════════════════════════════════════════════════════
# 场景 1：无 System Prompt vs 有 System Prompt
# ═══════════════════════════════════════════════════════════════════
def demo_system_prompt() -> None:
    """演示 system prompt 对输出的影响。"""
    user_input = "我明天要开会，帮我准备一下"

    # ── 1a. 无 system prompt ───────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("场景 1：无 System Prompt")
    print(SEPARATOR)
    resp_no_sys = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": user_input}],
    )
    answer_no_sys = resp_no_sys.choices[0].message.content or ""
    print(f"OUT: {answer_no_sys[:200]}")

    # ── 1b. 有 system prompt（任务助手人格）────────────────────────
    print(f"\n{SEPARATOR}")
    print("场景 1：有 System Prompt（任务助手）")
    print(SEPARATOR)
    system_prompt = (
        "你是任务管理助手。用户提到任何事项，你都要提取为结构化任务。"
        "返回 JSON 格式：{\"title\": \"任务标题\", "
        "\"priority\": \"high|medium|low\", \"description\": \"任务描述\"}。"
        "优先级规则：紧急=high，重要=medium，其他=low。只返回 JSON，不要其他文字。"
    )
    resp_with_sys = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )
    answer_with_sys = resp_with_sys.choices[0].message.content or ""
    print(f"OUT: {answer_with_sys}")


# ═══════════════════════════════════════════════════════════════════
# 场景 2：Few-shot 情感分类
# ═══════════════════════════════════════════════════════════════════
def demo_few_shot() -> None:
    """用 3 个示例教模型做情感分类。"""
    print(f"\n{SEPARATOR}")
    print("场景 2：Few-shot 情感分类")
    print(SEPARATOR)

    few_shot_prompt = (
        "你是一个情感分类器。根据用户输入判断情感倾向，只回答分类结果。\n\n"
        "示例 1：\n"
        "输入：这家餐厅的菜太好吃了，下次还来！\n"
        "分类：正面\n\n"
        "示例 2：\n"
        "输入：等了一个小时才上菜，服务态度还很差。\n"
        "分类：负面\n\n"
        "示例 3：\n"
        "输入：餐厅在商场三楼，营业到晚上10点。\n"
        "分类：中性\n\n"
        "现在请分类：\n"
        "输入：{user_input}\n"
        "分类："
    )

    test_inputs = [
        "这个产品用起来太顺手了，强烈推荐！",
        "包装破损，客服还推卸责任。",
        "商品重量约 500 克，保质期 12 个月。",
    ]

    for inp in test_inputs:
        prompt = few_shot_prompt.format(user_input=inp)
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
        )
        label = (resp.choices[0].message.content or "").strip()
        print(f"OUT: 输入：{inp}")
        print(f"OUT: 分类：{label}")
        print()


# ═══════════════════════════════════════════════════════════════════
# 场景 3：Chain-of-Thought 推理
# ═══════════════════════════════════════════════════════════════════
def demo_chain_of_thought() -> None:
    """用 CoT 引导模型逐步推理一个数学/逻辑题。"""
    print(f"\n{SEPARATOR}")
    print("场景 3：Chain-of-Thought 推理")
    print(SEPARATOR)

    question = (
        "一个任务助手需要处理以下优先级排序问题：\n"
        "有 3 个任务：A（截止明天，预计 2 小时），B（截止下周，预计 30 分钟），"
        "C（截止今天下午，预计 4 小时）。\n"
        "如果今天是上午，且每天只有 4 小时处理任务，应该如何排序？"
    )

    # ── 直接回答 ───────────────────────────────────────────────────
    print("\n--- 直接回答（不引导 CoT）---")
    resp_direct = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": question + "\n请直接给出排序结果。"}],
    )
    print(f"OUT: {(resp_direct.choices[0].message.content or '')[:300]}")

    # ── CoT 引导 ───────────────────────────────────────────────────
    print("\n--- CoT 引导（请一步一步思考）---")
    resp_cot = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {
                "role": "user",
                "content": question + "\n请一步一步思考，分析每个任务的紧急程度和所需时间，然后给出排序。",
            }
        ],
    )
    print(f"OUT: {(resp_cot.choices[0].message.content or '')[:500]}")


# ═══════════════════════════════════════════════════════════════════
# 场景 4：结构化输出（response_format + Pydantic）
# ═══════════════════════════════════════════════════════════════════
def demo_structured_output() -> None:
    """用 response_format 强制 JSON 输出，再用 Pydantic 解析。"""
    print(f"\n{SEPARATOR}")
    print("场景 4：结构化输出（response_format + Pydantic）")
    print(SEPARATOR)

    system_prompt = (
        "你是任务管理助手。用户会描述一个任务，你需要提取为 JSON。\n"
        "JSON 格式：{\"title\": \"任务标题\", "
        "\"priority\": \"high|medium|low\", \"description\": \"任务描述\"}\n"
        "优先级规则：截止时间紧迫=high，重要但不急=medium，其他=low。\n"
        "只返回 JSON，不要其他文字。"
    )

    user_input = "下周三之前要提交项目报告，需要整理数据和写总结"

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )

    raw_json = resp.choices[0].message.content or "{}"
    print(f"OUT: 原始 JSON：{raw_json}")

    # Pydantic 解析 + 校验
    task = TaskInfo.model_validate_json(raw_json)
    print(f"OUT: title={task.title}, priority={task.priority}")
    print(f"OUT: description={task.description}")

    # 验证类型安全
    assert isinstance(task.title, str), "title 必须是字符串"
    assert task.priority in ("high", "medium", "low"), f"未知优先级: {task.priority}"
    print("OUT: Pydantic 解析成功，类型校验通过！")


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    print("第02章 Prompt 工程 — 4 种核心技术演示")
    print(f"提供商: {cfg.provider} | 模型: {cfg.model}")

    demo_system_prompt()
    demo_few_shot()
    demo_chain_of_thought()
    demo_structured_output()

    print(f"\n{SEPARATOR}")
    print("OUT: 全部场景演示完成！")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
