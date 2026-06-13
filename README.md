# AI Agent 进阶教程 — 从零到生产（Python + TypeScript）

> **一句话**：从最原始的 API 调用出发，自己从零造一个 mini Agent 框架，
> 再切换到现代框架——学完后，无论流行什么框架你都能快速上手，知识不过时。
> 全程 **Python + TypeScript 双语言并列**，**OpenAI 兼容接口可一键切换**提供商。

---

## 为什么有这个教程

市面上的 Agent 教程大多有两个毛病：

1. **"先工具后原理"**：上来就教你调 LangChain 的 `AgentExecutor`，一行代码跑起来很爽，
   但出 bug 时完全不知道哪里错了。框架一旦过时，你的知识也跟着报废。
2. **玩具示例**：清一色 "Hello World Agent"，看完不会写真实的 Agent。

本教程反其道而行：

### 🧭 先原理后工具

> **先看清原理，再选择工具。** 原理是不变的，工具是流动的。

我们的递进路径是：

```
原始 API 调用  →  自己从零造 mini 框架  →  用现代框架
   (Part 1-4)        (Part 5)              (实战项目)
```

- **Part 1–4** 用最朴素的 `client.chat.completions.create()` 手写每一个组件，
  让你亲眼看到 Agent 循环、工具调用、记忆、ReAct 推理是怎么运作的——没有魔法。
- **Part 5** 把这些组件组装成你自己的迷你框架（带 max_steps、工具注册表、可观测钩子等 6 大核心）。
- **实战项目** 再切换到现代框架（OpenAI Agents SDK / Pydantic AI / Mastra / Vercel AI SDK）。

这样无论明年流行什么新框架，你已经掌握了底层，几小时就能上手。

> ⚠️ **刻意不教 LangChain 全家桶**（含 LangGraph）。它抽象层太重、已过 peak relevance，
> 不适合做教学骨架。我们用更轻、更现代的 SDK 替代。

---

## 学习路径图

教程分 **6 部分 / 17 章 / 4 个实战项目**，每章独立文件夹、可单独学习运行：

| 部分 | 章节 | 主题 | 你将学会 |
|------|------|------|----------|
| **Part 1 · 基础** | 第01–03章 | LLM 基础 / Prompt 工程 / 工具调用 | tokens、上下文窗口、温度、system 角色、function calling |
| **Part 2 · 第一个 Agent** | 第04–06章 | Agent 循环 / 记忆系统 / 错误处理 | 把 LLM + 工具 + 循环缝合成一个能干活的 Agent |
| **Part 3 · 推理模式** | 第07–09章 | ReAct / 规划 / RAG 检索 | 让 Agent 会思考、会分步、会查资料 |
| **Part 4 · 多 Agent** | 第10–11章 | 多 Agent 编排 / 上下文工程 | 多个 Agent 协作，并管好它们的上下文预算 |
| **Part 5 · 从零造框架** | 第12–14章 | 架构设计 / 实现核心 / 高级特性 | 亲手造一个 mini Agent 框架（6 大核心组件） |
| **Part 6 · 生产化** | 第15–17章 | 评估测试 / 可观测调试 / 安全护栏 | 让 Agent 可测、可观测、安全地跑在生产 |
| **实战项目 ×4** | 项目1–4 | 深度研究助手 / 编程 Agent / 多 Agent 代码审查 / 智能客服 | 4 个可上线的完整项目 |

**关键路径**：基础 → 第一个 Agent → 推理 → 多 Agent → 从零造框架（顺序链）→ 生产化 → 实战。

每章都包含：**概念讲解（README）+ Python 代码 + TypeScript 代码 + 练习 + 反模式说明**。

---

## 🧑‍💻 贯穿全教程的统一示例：任务助手 Agent

为了让 22 章的知识连贯、不散，我们用 **同一个示例 Agent** 贯穿演进——
它叫 **「任务助手 Agent」**。

它会随着章节一步步长大：

| 阶段 | 任务助手会什么 | 出现章节 |
|------|----------------|----------|
| 🥚 雏形 | 只会单轮回答问题 | Part 1 |
| 🐣 能干活 | 会循环调用工具（查日历、建待办、算账） | Part 2 |
| 🐤 会思考 | 遇到复杂任务会 ReAct 推理、分步规划、查资料 | Part 3 |
| 🦅 会协作 | 多个助手分工合作（一个规划、一个执行、一个审查） | Part 4 |
| 🏗️ 有骨架 | 被重构成你自己框架里的一个实例 | Part 5 |
| 🚀 可生产 | 带评估、监控、护栏，安全上线 | Part 6 + 实战项目 |

这样做的好处：你每学一章，**任务助手就多一项能力**，前后对照极其清晰，
而不是每章换一个互不相干的玩具例子。

---

## 📦 环境准备

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| **Python** | ≥ 3.11 | `pip install python-dotenv openai` |
| **Node.js** | ≥ 20 | `npm install dotenv openai`（章节根目录会装） |
| **Git** | 任意 | 克隆本仓库 |

**选一个 LLM 提供商**（都兼容 OpenAI 接口，可随时切换）：

| 提供商 | 获取密钥 | 特点 |
|--------|----------|------|
| OpenAI | <https://platform.openai.com/api-keys> | 官方，质量高，需付费 |
| DeepSeek | <https://platform.deepseek.com/api_keys> | 国内可用，性价比高 |
| Qwen（通义千问） | <https://dashscope.console.aliyun.com/apiKey> | 阿里云，国内友好 |
| **Ollama** | <https://ollama.com> | **本地、免费、离线**，无需密钥 |

---

## 🚀 如何使用本教程

### 1. 克隆并配置

```bash
cd ai-agent/
cp .env.example .env          # 复制配置模板
# 用编辑器打开 .env，填入你选的提供商的 API 密钥
```

### 2. 切换提供商（核心特性 ⭐）

`.env` 里改 **一行** 就能切换全家：

```dotenv
# 想用 DeepSeek：
PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxx

# 想本地跑（免费）：
PROVIDER=ollama
# 先 ollama pull qwen2.5:7b
```

所有章节的代码会通过 [`shared/config`](./shared/README.md) 自动读到新配置，
**无需改任何代码**。这就是 OpenAI 兼容接口的力量。

### 3. 开始学习

从 [第01章 LLM 基础](./01-llm-basics/)（即将上线）开始，逐章推进。
每章文件夹内：

```bash
# Python
python XX-xxx/python/main.py

# TypeScript
npx tsx XX-xxx/typescript/main.ts
```

### 4. 自检配置

随时确认当前生效的提供商：

```bash
python ai-agent/shared/config.py      # Python
npx tsx ai-agent/shared/config.ts     # TypeScript
```

---

## 📁 仓库结构

```
ai-agent/
├── README.md            ← 你在这里（总导读）
├── .env.example         ← 配置模板（复制为 .env 后填写）
├── .gitignore
├── shared/              ← 配置中枢（所有章节共用）
│   ├── config.py        ← Python 配置助手 get_config()
│   ├── config.ts        ← TypeScript 配置助手 getConfig()
│   └── README.md        ← 给章节作者的说明
├── 01-llm-basics/       ← 第01章（T2+ 创建中）
│   ├── README.md        ← 概念讲解
│   ├── python/          ← 可运行 Python 代码
│   ├── typescript/      ← 可运行 TS 代码
│   └── exercises/       ← 练习与参考答案
├── 02-prompt-engineering/
├── ...                  ← 第03–17章
└── projects/            ← 4 个实战项目
```

> 章节目录正在陆续创建。本文件（脚手架）是地基，后续 22 章都依赖它。

---

## 🤝 贡献指南

- **每章自包含**：尽量减少跨章依赖，便于单独学习。
- **代码可复制粘贴**：不省略关键路径，不写 `...` 占位。
- **每章一个新概念**：避免认知过载。
- **永远通过 `shared/config` 初始化客户端**：绝不硬编码密钥或 base_url。
- **包含反模式说明**：明确告诉读者"什么不该做"。
- **关注成本与延迟**：避免教出 ¥5/查询、30 秒响应的 Agent。

---

## 许可证

MIT（见根目录 LICENSE，后续添加）。

**下一步**：配置好 `.env` 后，进入 [`01-llm-basics/`](./01-llm-basics/) 开始第一课。
快乐学习 🚀
