"""项目4 · 智能客服 Agent（记忆 + 工具 + 人机协作）

综合实战：构建一个带记忆、工具调用、情绪识别和安全防护的智能客服系统。

核心组件：
  - ConversationBuffer 记忆：跨轮记住用户信息（姓名、订单号）
  - 订单查询工具：读取 data/orders.json，按订单号查询状态
  - 情绪识别 + 转人工：检测不满关键词 → 触发 Handoff
  - 防 Prompt 注入：检测注入关键词，拒绝越权请求
  - 多轮对话：5 轮预设演示（自我介绍→查订单→追问→不满→转人工）
  - 离线 Mock：API 不可用 → 本地 fallback，exit 0
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 让章节代码能 import shared.config ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

# 订单数据路径
ORDERS_PATH = Path(__file__).resolve().parent.parent / "data" / "orders.json"


# ════════════════════════════════════════════════════════════════════
# 1. ConversationBuffer 记忆系统
# ════════════════════════════════════════════════════════════════════


@dataclass
class ConversationBuffer:
    """跨轮对话记忆：list of messages + add/get_messages/clear。"""

    messages: List[Dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        """添加一条消息到记忆。"""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """获取所有消息历史。"""
        return list(self.messages)

    def clear(self) -> None:
        """清空记忆。"""
        self.messages.clear()

    def extract_user_info(self) -> Dict[str, str]:
        """从对话历史中提取用户信息（姓名、订单号）。"""
        info: Dict[str, str] = {}
        for msg in self.messages:
            text = msg["content"]
            # 简单关键词提取
            for keyword in ["我叫", "我是", "我的名字是"]:
                if keyword in text:
                    after = text.split(keyword, 1)[1].strip()
                    name = ""
                    for ch in after:
                        if "\u4e00" <= ch <= "\u9fff":
                            name += ch
                        else:
                            break
                    if name:
                        info["name"] = name
            # 订单号提取
            if "ORD-" in text:
                idx = text.index("ORD-")
                order_id = text[idx : idx + 7]
                info["orderId"] = order_id
        return info


# ════════════════════════════════════════════════════════════════════
# 2. 订单查询工具
# ════════════════════════════════════════════════════════════════════


def load_orders() -> List[Dict[str, Any]]:
    """从 data/orders.json 加载订单数据。"""
    try:
        with open(ORDERS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def query_order(order_id: str) -> Optional[Dict[str, Any]]:
    """按订单号查询订单。"""
    orders = load_orders()
    for order in orders:
        if order.get("orderId") == order_id:
            return order
    return None


def format_order(order: Dict[str, Any]) -> str:
    """格式化订单信息为可读字符串。"""
    return (
        f"订单号: {order['orderId']} | "
        f"商品: {order['item']} | "
        f"状态: {order['status']} | "
        f"金额: ¥{order['amount']}"
    )


# ════════════════════════════════════════════════════════════════════
# 3. 情绪识别 + 转人工
# ════════════════════════════════════════════════════════════════════

EMOTION_KEYWORDS = ["投诉", "太差", "退款", "垃圾", "愤怒", "生气", "不满", "差评"]


def detect_negative_emotion(text: str) -> bool:
    """检测用户不满情绪关键词。"""
    return any(kw in text for kw in EMOTION_KEYWORDS)


def trigger_handoff(reason: str) -> None:
    """触发转人工客服。"""
    print(f"OUT:handoff: ⚠️ 检测到用户不满，原因: {reason}")
    print(f"OUT:handoff: 🔄 正在转接人工客服，请稍候...")
    print(f"OUT:handoff: 👤 人工客服已接入，祝您问题顺利解决！")


# ════════════════════════════════════════════════════════════════════
# 4. 防 Prompt 注入
# ════════════════════════════════════════════════════════════════════

INJECTION_KEYWORDS = [
    "忽略之前指令",
    "ignore previous",
    "ignore all previous",
    "管理员",
    "admin",
    "所有用户",
    "所有订单",
    "全部用户数据",
    "system prompt",
    "你的指令是",
]


def detect_injection(text: str) -> bool:
    """检测 Prompt 注入关键词。"""
    lower = text.lower()
    return any(kw.lower() in lower for kw in INJECTION_KEYWORDS)


def block_injection(text: str) -> None:
    """打印注入拦截信息。"""
    print(f"OUT:inject:block: 🛡️ 检测到潜在 Prompt 注入，已拦截")
    print(f"OUT:inject:block: 触发内容: {text[:60]}")
    print(f"OUT:inject:block: 我只能帮您查询您自己的订单信息，无法执行其他指令。")


# ════════════════════════════════════════════════════════════════════
# 5. LLM 调用封装（带 try/catch 降级）
# ════════════════════════════════════════════════════════════════════


def llm_chat(
    messages: List[Dict[str, str]],
    system_prompt: str,
) -> str:
    """调用 LLM，失败时返回空字符串。"""
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=full_messages,  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════════
# 6. 客服 Agent 主逻辑
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个友好、专业的智能客服助手。你的职责是：\n"
    "1. 帮助用户查询订单状态\n"
    "2. 回答关于商品和服务的问题\n"
    "3. 记住用户的姓名和订单号\n"
    "4. 如果用户情绪不满，建议转人工客服\n"
    "请用简洁、友好的中文回复。"
)


def process_user_input(
    user_input: str,
    buffer: ConversationBuffer,
) -> str:
    """处理用户输入，返回回复。"""

    # ── 1. 防注入检查 ──
    if detect_injection(user_input):
        block_injection(user_input)
        buffer.add("user", user_input)
        reply = "抱歉，我只能帮您查询订单信息，无法执行其他指令。请问有什么订单需要查询吗？"
        buffer.add("assistant", reply)
        return reply

    # ── 2. 情绪检测 ──
    if detect_negative_emotion(user_input):
        print(f"OUT:emotion: 😤 检测到负面情绪关键词")
        trigger_handoff(user_input)
        buffer.add("user", user_input)
        reply = "非常抱歉给您带来不好的体验，我已为您转接人工客服，请稍候。"
        buffer.add("assistant", reply)
        return reply

    # ── 3. 提取用户信息并打印 ──
    buffer.add("user", user_input)
    info = buffer.extract_user_info()
    if info:
        print(f"OUT:memory: 📝 记忆更新: {info}")

    # ── 4. 订单查询工具 ──
    order_id = info.get("orderId", "")
    if order_id:
        print(f"OUT:tool: 🔧 调用工具: query_order({order_id})")
        order = query_order(order_id)
        if order:
            order_str = format_order(order)
            print(f"OUT:tool: ✅ 查询结果: {order_str}")
        else:
            order_str = f"未找到订单 {order_id}"
            print(f"OUT:tool: ❌ {order_str}")
    else:
        order_str = ""

    # ── 5. 尝试 LLM 回复（离线 fallback） ──
    context = ""
    if info:
        context = f"已知用户信息: {info}\n"
    if order_str:
        context += f"订单查询结果: {order_str}\n"

    messages = buffer.get_messages()
    if context:
        messages = [{"role": "system", "content": context}] + messages

    llm_reply = llm_chat(messages, SYSTEM_PROMPT)

    if llm_reply:
        buffer.add("assistant", llm_reply)
        return llm_reply

    # ── 6. 离线 fallback 回复 ──
    fallback = generate_fallback_reply(user_input, info, order_str)
    buffer.add("assistant", fallback)
    return fallback


def generate_fallback_reply(
    user_input: str,
    info: Dict[str, str],
    order_str: str,
) -> str:
    """离线模式：根据上下文生成 fallback 回复。"""
    name = info.get("name", "")

    if order_str:
        greeting = f"{name}，" if name else ""
        return f"{greeting}查询到您的订单信息：{order_str}。请问还有什么需要帮助的吗？"

    if name:
        return f"{name}，您好！请问有什么可以帮您的？您可以提供订单号来查询订单状态。"

    # 通用回复
    lower = user_input.lower()
    if any(kw in lower for kw in ["你好", "您好", "hi", "hello"]):
        return "您好！欢迎联系智能客服，请问有什么可以帮您的？"

    if "订单" in lower:
        return "请提供您的订单号（如 ORD-001），我来帮您查询。"

    return "请问有什么可以帮您的？您可以提供订单号来查询订单状态。"


# ════════════════════════════════════════════════════════════════════
# 7. 多轮对话演示（5 轮预设）
# ════════════════════════════════════════════════════════════════════

DEMO_CONVERSATIONS = [
    "你好，我叫张三",
    "请帮我查一下订单 ORD-001 的状态",
    "这个订单什么时候能到？",
    "等了这么久还没到，太差了！我要退款！",
    "算了，我要投诉你们的服务！",
]


def run_demo() -> None:
    """运行 5 轮预设对话演示。"""
    print("OUT: ══ 智能客服 Agent ══")
    print(f"OUT: 模型: {cfg.model}")
    print(f"OUT: 提供商: {cfg.provider}")
    print()

    buffer = ConversationBuffer()

    for i, user_msg in enumerate(DEMO_CONVERSATIONS, 1):
        print(f"OUT: ── 第 {i} 轮 ──")
        print(f"OUT: 👤 用户: {user_msg}")

        reply = process_user_input(user_msg, buffer)
        print(f"OUT: 🤖 客服: {reply}")

        # 显示记忆状态
        msgs = buffer.get_messages()
        print(f"OUT:memory: 💾 记忆中消息数: {len(msgs)}")
        info = buffer.extract_user_info()
        if info:
            print(f"OUT:memory: 📝 已记住: {info}")
        print()

    # 最终记忆状态
    print("OUT: ══ 对话结束 ══")
    print(f"OUT:memory: 💾 最终记忆: {len(buffer.get_messages())} 条消息")
    final_info = buffer.extract_user_info()
    if final_info:
        print(f"OUT:memory: 📝 最终用户信息: {final_info}")


# ════════════════════════════════════════════════════════════════════
# 8. 主函数
# ════════════════════════════════════════════════════════════════════


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
