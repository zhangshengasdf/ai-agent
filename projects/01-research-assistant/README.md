# 项目1 · 深度研究助手（Plan-and-Execute + RAG + 多工具）

> **综合实战**：把第04章 Agent 循环、第08章规划、第09章 RAG 缝合成一个能"查资料→做笔记→出报告"的研究助手。

---

## TL;DR

> **30 秒速读**：构建一个 Plan-and-Execute 架构的研究助手，LLM 先规划研究步骤再逐步执行，用 RAG 从本地知识库检索资料、用笔记工具记录发现、最后生成研究报告。
> 
> **如果只记一件事**：Plan-and-Execute 把复杂任务拆成结构化步骤，比让 Agent 自由发挥可靠得多。

---

## 你会学到什么

1. **Plan-and-Execute 架构**：给定研究主题 → LLM 规划研究步骤（结构化输出）→ 逐步执行
2. **RAG 知识检索**：从 `data/` 加载文档 → 分块 → 向量化 → 余弦相似度 top-k 检索
3. **多工具协作**：`search_knowledge` / `write_note` / `get_summary` 三种工具协同
4. **Trace + Cost 追踪**：每步记录耗时、token、费用，最后打印 trace 树和总成本
5. **离线 Mock**：API 不可用时自动降级为预设演示，确保 `exit 0`

---

## 架构概览

```
研究主题
   │
   ▼
┌──────────────┐
│   Planner    │  ← LLM 生成结构化研究计划（JSON steps）
└──────┬───────┘
       │
       ▼
┌──────────────┐   search_knowledge(query)
│   Executor   │──► RAG 检索 data/ 知识库
│  (逐 step)   │──► write_note(content)  记录发现
│              │──► get_summary()        查看笔记汇总
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Reporter   │  ← LLM 基于笔记生成研究报告
└──────────────┘
       │
       ▼
   Trace 树 + 总成本
```

---

## 常见错误

> 概念懂了，实际写代码还是会踩坑。

| 错误 | 症状 | 解决 |
|------|------|------|
| Planner 输出不是合法 JSON | `json.loads` 报错，整个流程崩掉 | 用 `response_format={"type": "json_object"}` + Pydantic 校验 + 重试 |
| RAG 检索返回空结果但不处理 | Agent 拿到空上下文瞎编答案 | 检索结果为空时返回明确提示，让 Agent 知道"没找到相关内容" |
| 笔记工具没做去重 | 同一条发现被记录三次，报告里重复段落 | `write_note` 前检查内容是否已存在，或用 set 去重 |
| 没追踪 token 和成本 | 研究任务跑了 20 步，月底账单才发现花了 $5 | 每步 LLM 调用后记录 token，最后打印总成本 |
| 知识库文档没分块直接塞 | 5000 字的文档一次性塞进 context，超出窗口 | 按段落或固定长度分块，每块 200-500 字，top-k 检索 |

---

## 运行方式

```bash
cd ai-agent/projects/01-research-assistant

# Python
python3 python/main.py

# TypeScript
npx tsx typescript/main.ts
```

输出前缀：`OUT:plan:` / `OUT:step:` / `OUT:search:` / `OUT:note:` / `OUT:report:` / `OUT:trace:` / `OUT:cost:`

---

## 离线设计

`.env` 中 API 密钥为占位符 `sk-REPLACE-ME` 时：
- **Planner**：try LLM 失败 → 使用预设 3 步研究计划
- **Executor**：检索用纯 Python/TS 词频向量（不调 API），笔记工具纯本地
- **Reporter**：try LLM 失败 → 用笔记拼接成 mock 报告

全程 **不依赖真实 API**，`exit 0`。

---

## 知识库

`data/` 目录下 3 篇 Markdown 文档：

| 文件 | 主题 |
|------|------|
| `llm-emergence.md` | 大语言模型的涌现能力 |
| `rag-principles.md` | RAG 检索增强生成原理 |
| `multi-agent.md` | 多 Agent 系统 |

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
- [知识库数据](./data/)
