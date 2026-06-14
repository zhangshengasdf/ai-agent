# 项目1 · 深度研究助手（Plan-and-Execute + RAG + 多工具）

> **综合实战**：把第04章 Agent 循环、第08章规划、第09章 RAG 缝合成一个能"查资料→做笔记→出报告"的研究助手。

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
