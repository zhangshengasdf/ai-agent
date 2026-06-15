# 第10章 多 Agent 编排 — 练习与参考答案

> 这些练习帮助你巩固第10章的核心概念：Supervisor-Worker 模式、Agent Handoff、共享记忆。
> 建议先独立完成，再看参考答案。

---

## 练习 1：实现一个 Handoff 场景（客服 → 技术专家）⭐ 核心

### 题目

基于本章的 Handoff 模式，实现一个**销售客服 → 技术专家**的 Handoff 场景：

1. **销售客服 Agent**（system prompt："你是销售客服，负责产品咨询、报价、下单"）
2. **技术专家 Agent**（system prompt："你是技术专家，负责产品技术规格、兼容性、集成问题"）
3. **触发条件**：用户消息包含技术关键词（"规格"/"API"/"兼容"/"集成"/"SDK"）时触发 Handoff
4. **Handoff 流程**：
   - 销售客服先回答销售问题
   - 用户追问技术问题 → 检测关键词 → Handoff
   - 技术专家看到完整对话历史，继续回答

### 要求

- 实现关键词检测函数 `needs_handoff(message)`
- 实现 Handoff 函数：替换 system prompt，保留对话历史
- 模拟以下对话：
  ```
  用户：这个 API 服务多少钱？（销售问题）
  用户：它支持 Python SDK 吗？有 REST API 吗？（技术问题 → 触发 Handoff）
  ```
- 输出要能看出"哪个 Agent 在回答"和"Handoff 发生在哪一步"

### 提示

- 参考 `python/main.py` 的 `handoff_flow()` 或 `typescript/main.ts` 的 `handoffFlow()`
- TECH_KEYWORDS 改为销售场景的技术词
- 关键：Handoff 后必须换 system prompt，否则技术专家还以为自己是销售

---

### 参考答案（Python）

```python
"""练习 1 参考：销售客服 → 技术专家 Handoff"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

SALES_PROMPT = "你是销售客服，负责产品咨询、报价、下单。回答友好简洁。遇到技术问题转交技术专家。"
TECH_PROMPT = "你是技术专家，负责产品技术规格、兼容性、集成问题。你从销售客服接手对话，请基于上下文回答。"

# 技术关键词（销售场景）
TECH_KEYWORDS = ["规格", "API", "兼容", "集成", "SDK", "文档", "调用", "接口"]


def needs_handoff(message: str) -> bool:
    """检测是否需要转交给技术专家。"""
    return any(kw in message for kw in TECH_KEYWORDS)


def sales_handoff_demo():
    """销售客服 → 技术专家 Handoff 演示。"""
    conversation = [
        "这个 API 服务多少钱？按月付费还是按量？",  # 销售问题
        "它支持 Python SDK 吗？有 REST API 吗？兼容性怎么样？",  # 技术问题
    ]

    messages = [{"role": "system", "content": SALES_PROMPT}]
    current_agent = "SalesAgent"

    for user_msg in conversation:
        print(f"\n用户: {user_msg}")
        messages.append({"role": "user", "content": user_msg})

        if needs_handoff(user_msg):
            # Handoff：换 system prompt，保留对话历史
            messages = [
                {"role": "system", "content": TECH_PROMPT},
                *messages[1:],
            ]
            current_agent = "TechExpert"
            print(f"OUT:handoff: 触发 Handoff → {current_agent}")
            matched = [kw for kw in TECH_KEYWORDS if kw in user_msg]
            print(f"OUT:handoff: 命中关键词: {matched}")
        else:
            print(f"OUT:worker:SalesAgent: (销售客服处理)")

        # 调用当前 Agent 回复（try 真实 API，失败用 mock）
        try:
            response = client.chat.completions.create(
                model=cfg.model, messages=messages,
            )
            reply = response.choices[0].message.content or "(空)"
        except Exception:
            # 离线 mock 回复
            if current_agent == "TechExpert":
                reply = "支持 Python SDK 和 REST API，兼容主流框架。详见技术文档。"
            else:
                reply = "我们的 API 服务按量付费，详情我发你报价单。"
            print(f"OUT:worker:{current_agent}: [mock] {reply[:80]}")
            continue

        print(f"OUT:worker:{current_agent}: {reply[:80]}")
        messages.append({"role": "assistant", "content": reply})

    print(f"\nOUT:resolve: 最终由 {current_agent} 处理")


if __name__ == "__main__":
    sales_handoff_demo()
```

### 参考答案（TypeScript）

```typescript
import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

const SALES_PROMPT = "你是销售客服，负责产品咨询、报价、下单。回答友好简洁。遇到技术问题转交技术专家。";
const TECH_PROMPT = "你是技术专家，负责产品技术规格、兼容性、集成问题。你从销售客服接手对话。";

const TECH_KEYWORDS = ["规格", "API", "兼容", "集成", "SDK", "文档", "调用", "接口"];

function needsHandoff(message: string): boolean {
  return TECH_KEYWORDS.some((kw) => message.includes(kw));
}

interface ChatMessage { role: "system" | "user" | "assistant"; content: string; }

async function salesHandoffDemo(): Promise<void> {
  const conversation = [
    "这个 API 服务多少钱？按月付费还是按量？",
    "它支持 Python SDK 吗？有 REST API 吗？兼容性怎么样？",
  ];

  let messages: ChatMessage[] = [{ role: "system", content: SALES_PROMPT }];
  let currentAgent = "SalesAgent";

  for (const userMsg of conversation) {
    console.log(`\n用户: ${userMsg}`);
    messages.push({ role: "user", content: userMsg });

    if (needsHandoff(userMsg)) {
      messages = [{ role: "system", content: TECH_PROMPT }, ...messages.slice(1)];
      currentAgent = "TechExpert";
      console.log(`OUT:handoff: 触发 Handoff → ${currentAgent}`);
    } else {
      console.log(`OUT:worker:SalesAgent: (销售客服处理)`);
    }

    try {
      const resp = await client.chat.completions.create({
        model: cfg.model,
        messages: messages.map((m) => ({ role: m.role, content: m.content })),
      });
      const reply = resp.choices[0].message.content ?? "(空)";
      console.log(`OUT:worker:${currentAgent}: ${reply.slice(0, 80)}`);
      messages.push({ role: "assistant", content: reply });
    } catch {
      const mockReply = currentAgent === "TechExpert"
        ? "支持 Python SDK 和 REST API，兼容主流框架。"
        : "我们的 API 服务按量付费。";
      console.log(`OUT:worker:${currentAgent}: [mock] ${mockReply.slice(0, 80)}`);
      messages.push({ role: "assistant", content: mockReply });
    }
  }
  console.log(`\nOUT:resolve: 最终由 ${currentAgent} 处理`);
}

salesHandoffDemo().catch(console.error);
```

### 预期输出

```
用户: 这个 API 服务多少钱？按月付费还是按量？
OUT:worker:SalesAgent: (销售客服处理) / [mock] 我们的 API 服务按量付费...

用户: 它支持 Python SDK 吗？有 REST API 吗？兼容性怎么样？
OUT:handoff: 触发 Handoff → TechExpert
OUT:handoff: 命中关键词: ["API", "兼容", "SDK"]
OUT:worker:TechExpert: [mock] 支持 Python SDK 和 REST API，兼容主流框架...

OUT:resolve: 最终由 TechExpert 处理
```

---

## 练习 2：扩展 Supervisor-Worker，增加 Coder Worker

### 题目

本章的 Supervisor-Worker 有 Researcher 和 Writer 两个 Worker。请**增加一个 Coder Worker**，并设计一个会同时用到三个 Worker 的任务：

1. Coder 的 system prompt："你是程序员，负责写代码片段"
2. 设计任务："调研 Python 的异步编程并给出代码示例"（需要 Researcher 检索 + Coder 写码 + Writer 整理）
3. 实现 Supervisor 的分派逻辑：应该分解成 3+ 个 Assignment

### 要求

- 在 `WORKERS` 字典中注册 Coder
- 预设一个合理的 AssignmentPlan（3-4 个分派）
- 输出要能看到三个 Worker 依次执行

### 提示

- 参考 `demo_supervisor_worker_offline()` 的结构
- 分派顺序通常是：Researcher（调研）→ Coder（写码）→ Writer（整理成文）

---

### 参考答案（核心部分）

```python
# 注册 Coder Worker（已在 main.py 中）
CODER_PROMPT = "你是程序员，负责写代码片段。输出带语法标注的代码块，附简要说明。"
WORKERS["Coder"] = Worker("Coder", CODER_PROMPT, {})

# 离线 mock：三 Worker 协作
def demo_three_workers():
    task = "调研 Python 的异步编程并给出代码示例"
    mock_plan = AssignmentPlan(assignments=[
        Assignment(worker="Researcher", subtask="检索 Python 异步编程的概念和 asyncio 库"),
        Assignment(worker="Coder", subtask="基于调研结果写一个 async/await 代码示例"),
        Assignment(worker="Writer", subtask="把调研结果和代码示例整理成一篇文章"),
    ])
    
    print(f"任务: {task}")
    for i, a in enumerate(mock_plan.assignments, 1):
        print(f"OUT:supervisor:assignment{i}: → {a.worker}: {a.subtask}")
    
    # 执行（每个 Worker 的输出作为下一个的 context）
    context = ""
    for assignment in mock_plan.assignments:
        if assignment.worker == "Researcher":
            result = search_wiki(assignment.subtask)
        elif assignment.worker == "Coder":
            result = "```python\nimport asyncio\n\nasync def main():\n    await asyncio.sleep(1)\nasyncio.run(main())\n```"
        else:  # Writer
            result = f"# Python 异步编程\n\n## 概念\n{context[:50]}\n\n## 代码示例\n{result}"
        print(f"OUT:worker:{assignment.worker}: {result[:80]}")
        context += result + "\n"
```

---

## 练习 3：思考题 — 这些场景该用多 Agent 吗？

### 题目

判断以下场景**是否应该使用多 Agent**，并说明理由：

| 场景 | 是否用多 Agent？ | 理由 |
|------|-----------------|------|
| A. 翻译一篇 500 字的短文（中→英） | ? | ? |
| B. 写一份市场调研报告（需要检索 + 分析 + 写作） | ? | ? |
| C. 智能客服（退货咨询 + 技术支持 + 账单问题） | ? | ? |
| D. 计算 2+2 | ? | ? |
| E. 开发一个全栈应用（前端 + 后端 + 数据库 + 测试） | ? | ? |
| F. 总结一封邮件的要点 | ? | ? |

### 参考答案

| 场景 | 是否用多 Agent？ | 理由 |
|------|-----------------|------|
| A. 翻译短文 | ❌ 不用 | 单次 LLM 调用足够。多 Agent 是过度工程（反模式1） |
| B. 市场调研报告 | ✅ Supervisor-Worker | 任务可分解（检索+分析+写作），角色多样，适合分工 |
| C. 智能客服 | ✅ Handoff | 任务进行中发现需要不同专家（客服→技术→账单），适合 Handoff 路由 |
| D. 计算 2+2 | ❌ 不用 | 单步任务，一次工具调用搞定 |
| E. 全栈应用 | ✅ Supervisor-Worker | 工具过载（前端+后端+DB+测试工具远超15个），角色明确（前端/后端/DB工程师） |
| F. 总结邮件 | ❌ 不用 | 单次 LLM 调用足够 |

**核心原则**：能用单 Agent 解决的，别上多 Agent。只有工具过载/上下文过长/角色冲突/需要专家路由时才用。

---

## 练习 4：修复一个 Handoff 的 Bug

### 题目

下面的 Handoff 实现有一个**严重 bug**，导致技术专家"失忆"（不记得之前的对话）。找出并修复：

```python
def buggy_handoff():
    messages = [
        {"role": "system", "content": CUSTOMER_SERVICE_PROMPT},
        {"role": "user", "content": "我想退货"},
        {"role": "assistant", "content": "好的，请提供订单号"},
        {"role": "user", "content": "退货页面报了 500 错误"},
    ]
    
    # Handoff
    messages = [
        {"role": "system", "content": TECH_EXPERT_PROMPT},
    ]  # ← Bug 在这里
    
    response = client.chat.completions.create(model=cfg.model, messages=messages)
    print(response.choices[0].message.content)
```

### 问题

技术专家收到的 messages 只有 system prompt，**丢失了所有对话历史**。它会问"请问您需要什么帮助？"——完全不知道用户是来退货的。

### 修复

```python
def fixed_handoff():
    messages = [
        {"role": "system", "content": CUSTOMER_SERVICE_PROMPT},
        {"role": "user", "content": "我想退货"},
        {"role": "assistant", "content": "好的，请提供订单号"},
        {"role": "user", "content": "退货页面报了 500 错误"},
    ]
    
    # Handoff：换 system prompt，但保留 user/assistant 对话历史
    messages = [
        {"role": "system", "content": TECH_EXPERT_PROMPT},  # 新角色
        *messages[1:],  # ← 保留旧对话（去掉旧 system）
    ]
    
    response = client.chat.completions.create(model=cfg.model, messages=messages)
    # 技术专家现在能看到"用户想退货 + 遇到500错误"，给出针对性回答
    print(response.choices[0].message.content)
```

**关键**：`messages[1:]` 保留了 index 1 之后的所有消息（user/assistant），只替换了 index 0 的 system prompt。

---

## 练习 5：设计一个 Handoff 防无限循环机制（进阶）

### 题目

本章提到反模式 5："Handoff 无限链（踢皮球）"——Agent A → B → C → A 无限转交。

请设计一个机制，**限制 Handoff 次数**，超过限制就强制由当前 Agent 给出答案：

### 要求

- 设定最多 2 次 Handoff
- 记录 Handoff 历史
- 超过限制时，当前 Agent 必须回答（不能再 Handoff）

### 参考答案

```python
MAX_HANDOFFS = 2

def handoff_with_limit(conversation):
    messages = [{"role": "system", "content": CUSTOMER_SERVICE_PROMPT}]
    handoff_count = 0
    handoff_chain = ["CustomerService"]
    
    for user_msg in conversation:
        messages.append({"role": "user", "content": user_msg})
        
        if needs_handoff(user_msg):
            if handoff_count < MAX_HANDOFFS:
                # 允许 Handoff
                handoff_count += 1
                messages = [{"role": "system", "content": TECH_EXPERT_PROMPT}, *messages[1:]]
                handoff_chain.append("TechExpert")
                print(f"OUT:handoff: 第 {handoff_count} 次 Handoff: {handoff_chain[-2]} → {handoff_chain[-1]}")
            else:
                # 超过限制，强制当前 Agent 回答
                print(f"OUT:handoff: ⚠️ 达到最大 Handoff 次数 {MAX_HANDOFFS}，强制 {handoff_chain[-1]} 回答")
        
        # 当前 Agent 回复（不管是否 Handoff）
        reply = call_agent(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"OUT:worker:{handoff_chain[-1]}: {reply[:80]}")
    
    print(f"\nOUT:resolve: Handoff 链: {' → '.join(handoff_chain)} (共 {handoff_count} 次)")
```

---

## 学习检查清单

完成本章练习后，确认你能回答：

- [ ] 什么时候应该用多 Agent？什么时候不应该？（Anthropic 共识）
- [ ] Supervisor-Worker 和 Handoff 的核心区别是什么？
- [ ] Handoff 时为什么要替换 system prompt？不替换会怎样？
- [ ] 消息传递 vs 共享黑板 vs 共享对话历史，各自的优缺点？
- [ ] 为什么共享记忆需要并发控制？本章怎么避免这个问题的？
- [ ] 多 Agent 的 6 个反模式你能说出几个？

如果全部能答，恭喜你掌握了第10章的核心！
