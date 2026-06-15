# 第05章 练习 — 记忆系统

> 动手实践三种记忆系统。每个练习都附参考答案。

---

## 练习 1：实现一个"记得用户偏好的客服"（核心编程题）

**场景**：构建一个客服 Agent，跨多轮对话记住用户的**姓名**和**偏好**（如喜欢的编程语言、所在城市），
后续对话能直接用这些信息，不需要用户每次重复。

**要求**：
1. 用 `ConversationBuffer` 存储对话
2. 实现一个 `extract_user_info(messages)` 函数，从历史消息里提取用户姓名和偏好
3. 模拟 3 轮对话，第 3 轮问"你还记得我叫什么吗？"，验证 Agent 能回答

**Python 参考答案**：

```python
import re
from typing import Dict, Optional
# 假设 ConversationBuffer 已从 main.py 导入


def extract_user_info(messages: list[dict]) -> Dict[str, str]:
    """从对话历史中提取用户姓名和偏好（简单关键词匹配）。"""
    info: Dict[str, str] = {}
    all_text = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    )

    # 提取姓名：匹配"我叫XX"
    name_match = re.search(r"我叫(\S+?)[，。,.!!\s]", all_text + " ")
    if name_match:
        info["name"] = name_match.group(1)

    # 提取偏好语言
    for lang in ["Python", "Java", "JavaScript", "Go", "Rust"]:
        if lang in all_text:
            info["language"] = lang
            break

    # 提取城市
    for city in ["北京", "上海", "深圳", "杭州", "广州"]:
        if city in all_text:
            info["city"] = city
            break

    return info


def simulate_customer_service() -> None:
    """模拟一个记得用户偏好的客服。"""
    buffer = ConversationBuffer()
    buffer.add("system", "你是客服 Agent，会记住用户的姓名和偏好。")

    # 第 1 轮：用户自报家门
    buffer.add("user", "你好，我叫小明，住在北京。")
    info = extract_user_info(buffer.get_messages())
    response = f"你好{info.get('name', '')}！很高兴为您服务。"
    buffer.add("assistant", response)
    print(f"轮1 系统提取: {info}")
    print(f"轮1 客服回复: {response}")

    # 第 2 轮：用户提偏好
    buffer.add("user", "我最喜欢用 Python 编程。")
    info = extract_user_info(buffer.get_messages())
    response = f"记住了，{info.get('name', '您')}喜欢 {info.get('language', '编程')}！"
    buffer.add("assistant", response)
    print(f"轮2 系统提取: {info}")
    print(f"轮2 客服回复: {response}")

    # 第 3 轮：验证记忆
    buffer.add("user", "你还记得我叫什么吗？")
    info = extract_user_info(buffer.get_messages())
    # Agent 用提取的信息回答（真实场景传给 LLM，这里直接模拟）
    name = info.get("name", "（未知）")
    city = info.get("city", "（未知）")
    lang = info.get("language", "（未知）")
    response = f"当然记得！你叫{name}，住在{city}，喜欢用{lang}。"
    print(f"轮3 系统提取: {info}")
    print(f"轮3 客服回复: {response}")


simulate_customer_service()
```

**TypeScript 参考答案**：

```typescript
// 假设 ConversationBuffer 已从 main.ts 导入

interface UserInfo {
  name?: string;
  language?: string;
  city?: string;
}

function extractUserInfo(messages: Message[]): UserInfo {
  const info: UserInfo = {};
  const allText = messages
    .filter((m) => m.role === "user")
    .map((m) => m.content)
    .join(" ");

  // 提取姓名：匹配"我叫XX"
  const nameMatch = allText.match(/我叫(\S+?)[，。,.!!\s]/);
  if (nameMatch) {
    info.name = nameMatch[1];
  }

  // 提取偏好语言
  for (const lang of ["Python", "Java", "JavaScript", "Go", "Rust"]) {
    if (allText.includes(lang)) {
      info.language = lang;
      break;
    }
  }

  // 提取城市
  for (const city of ["北京", "上海", "深圳", "杭州", "广州"]) {
    if (allText.includes(city)) {
      info.city = city;
      break;
    }
  }

  return info;
}

function simulateCustomerService(): void {
  const buffer = new ConversationBuffer();
  buffer.add("system", "你是客服 Agent，会记住用户的姓名和偏好。");

  buffer.add("user", "你好，我叫小明，住在北京。");
  let info = extractUserInfo(buffer.getMessages());
  let response = `你好${info.name ?? ""}！很高兴为您服务。`;
  buffer.add("assistant", response);
  console.log(`轮1 系统提取:`, info);
  console.log(`轮1 客服回复: ${response}`);

  buffer.add("user", "我最喜欢用 Python 编程。");
  info = extractUserInfo(buffer.getMessages());
  response = `记住了，${info.name ?? "您"}喜欢 ${info.language ?? "编程"}！`;
  buffer.add("assistant", response);
  console.log(`轮2 系统提取:`, info);
  console.log(`轮2 客服回复: ${response}`);

  buffer.add("user", "你还记得我叫什么吗？");
  info = extractUserInfo(buffer.getMessages());
  response = `当然记得！你叫${info.name ?? "（未知）}，` +
    `住在${info.city ?? "（未知）}，喜欢用${info.language ?? "（未知）}。`;
  console.log(`轮3 系统提取:`, info);
  console.log(`轮3 客服回复: ${response}`);
}

simulateCustomerService();
```

**验证**：第 3 轮的回复应包含"小明"、"北京"、"Python"。

> 💡 **进阶**：把 `extract_user_info` 的结果存进 `system` prompt（如"用户叫小明，住北京"），
> 每轮对话自动注入，这样即使用 `clear()` 清空对话历史，用户信息仍保留。

---

## 练习 2：对比 Buffer 和 Summary 的上下文长度（理解题）

运行 `python3 python/main.py`，观察 Demo 1（Buffer）和 Demo 2（Summary）的输出。

**问题**：
1. Demo 1 的 Buffer 最终有多少条消息？如果对话继续到 100 轮，会发生什么？
2. Demo 2 的 SummaryMemory 在添加第几条消息时触发了摘要？触发后原文数怎么变化？
3. 假设每条消息平均 20 tokens，Buffer 到 100 轮时累计传了多少 tokens？Summary 呢（假设摘要固定 50 tokens）？

**参考答案**：

1. Demo 1 的 Buffer 最终 6 条（5 条 add + system）。100 轮时 messages 会有 ~100 条，可能超出上下文窗口（128K tokens），且每轮都要传全部历史——成本 O(n²)。

2. 在添加第 7 条消息时（`len > max_messages=6`）触发摘要。触发后原文数从 7 降到 5（去掉最早 2 条）。每次超过阈值都会压缩，原文数始终 ≤ 6。

3. **Buffer**：第 100 轮时传前 99 轮 = 99 × 20 = 1980 tokens。但累计成本是 20+40+60+...+2000 = 20 × (1+2+...+100) = 20 × 5050 = **101,000 tokens**。

   **Summary**：每轮传"摘要(50) + 最近6条(120)" = 170 tokens。100 轮累计 ≈ 100 × 170 = **17,000 tokens**（加上摘要产生的额外 API 调用成本）。

   Summary 比 Buffer 节省约 **83%** 的 token，代价是有损压缩 + 额外摘要 API 调用。

---

## 练习 3：扩展 VectorMemory 的"删除"操作（编程题）

当前 `VectorMemory` 只有 `add` / `search` / `clear`，没有"删除单条"的能力。
请实现一个 `remove(text)` 方法，按文本内容删除一条记录。

**要求**：
- `remove(text)` 删除第一条与 `text` 完全匹配的记录
- 删除后 `count()` 减 1
- 不存在时打印提示，不报错

**Python 参考答案**：

```python
class VectorMemory:  # 扩展原有类
    # ... 原有方法不变 ...

    def remove(self, text: str) -> bool:
        """删除第一条与 text 完全匹配的记录。返回是否删除成功。"""
        for i, (stored_text, _) in enumerate(self._store):
            if stored_text == text:
                self._store.pop(i)
                return True
        return False


# 测试：
vm = VectorMemory()
vm.add("Python 很好")
vm.add("Java 也不错")
print(vm.count())          # 2
print(vm.remove("Python 很好"))  # True
print(vm.count())          # 1
print(vm.remove("不存在的"))    # False
```

**TypeScript 参考答案**：

```typescript
class VectorMemory {  // 扩展原有类
  // ... 原有方法不变 ...

  remove(text: string): boolean {
    const idx = this.store.findIndex((item) => item.text === text);
    if (idx >= 0) {
      this.store.splice(idx, 1);
      return true;
    }
    return false;
  }
}

// 测试：
const vm = new VectorMemory();
vm.add("Python 很好");
vm.add("Java 也不错");
console.log(vm.count());           // 2
console.log(vm.remove("Python 很好")); // true
console.log(vm.count());           // 1
console.log(vm.remove("不存在的"));    // false
```

**验证**：删除后 `search()` 不应再返回被删除的文本。

---

## 练习 4（进阶）：给 SummaryMemory 接入真实 LLM 摘要

当前 `_llm_summarize` 在 API 失败时降级为 mock 摘要。当有有效 API key 时，
让它调用真实 LLM 压缩对话。

**提示**：
- 参考第01章的 API 调用模式：`client.chat.completions.create(...)`
- system prompt 指示模型"用一句话总结对话要点"
- 设置 `max_tokens=100` 控制摘要长度
- 用 try/except 捕获失败，降级到 mock

**思考题**（不用写代码）：
1. 真实 LLM 摘要比 mock 摘好在哪？
2. 每次 add 都触发摘要会怎样？如何优化（如批量摘要）？
3. 摘要的"累积"会不会越来越长？如何限制？

**参考答案**：

1. 真实 LLM 摘要能**理解语义**——它知道"小明喜欢 Python"是用户偏好，"北京 25 度"是天气查询，
   会保留关键信息、丢弃寒暄。mock 只能取关键词和前 30 字符，丢失上下文关系。

2. 每次 add 都触发 = 每轮都调一次 LLM = **延迟翻倍 + 成本激增**。
   优化：**批量摘要**——积攒一批后再压缩（如每 5 轮摘要一次），而非每轮。
   本章的"超过阈值才摘要"就是这个思路。

3. 会。累积摘要是 `_summary += new_summary`，越来越长。
   限制方法：(a) 摘要超过 N 字时再做二次摘要压缩；(b) 滑窗只保留最近 M 轮的摘要；
   (c) 用向量检索替代累积摘要（长期记忆用 VectorMemory）。

---

## 练习 5（思考）：三种记忆何时组合使用？

一个真实的客服 Agent 同时服务多个用户，每个用户有：
- 当前会话的对话历史（当天）
- 跨会话的用户画像（姓名、偏好、历史工单）
- 产品知识库（FAQ、文档）

**问题**：这三个信息源分别该用哪种记忆？为什么？如何组合？

**参考答案**：

| 信息源 | 记忆类型 | 理由 |
|--------|----------|------|
| 当前会话历史 | ConversationBuffer | 短期、需要完整上下文、轮数少（<20）|
| 跨会话用户画像 | SummaryMemory | 中期、需要压缩（多会话累积太长）|
| 产品知识库 | VectorMemory | 海量、按需检索（FAQ 可能有上千条）|

**组合方式**（伪代码）：

```python
# 每次调用 API 时，拼装 messages：
messages = [
    {"role": "system", "content": f"用户画像: {user_profile_summary}"},  # SummaryMemory
    *buffer.get_messages(),  # ConversationBuffer（当前会话）
    {"role": "system", "content": f"相关知识: {knowledge_results}"},     # VectorMemory 检索结果
]
```

**关键洞察**：
- **Buffer 提供当前上下文**（这轮在聊什么）
- **Summary 提供用户记忆**（这个人是谁、喜欢什么）
- **Vector 提供知识检索**（产品文档里相关的部分）

三者各司其职，组合使用才能构建生产级 Agent。这正是第12章自造框架的记忆层设计。

> 💡 **反模式**：把所有信息塞进一个 Buffer——产品知识库 1000 条 + 用户画像 + 当前对话 = 上下文爆炸。
