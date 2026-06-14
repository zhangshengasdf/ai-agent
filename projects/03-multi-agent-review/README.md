# 项目3 · 多 Agent 代码审查系统（Supervisor-Worker 协作）

> **综合实战**：Supervisor 接收代码 → 分派给 3 个专门 Reviewer Agent → 各自从安全/性能/风格维度审查 → 汇总排序报告。

---

## 你会学到什么

1. **Supervisor-Worker 架构**：一个协调者分派任务给多个专业 Agent，收集结果后汇总
2. **专业 Reviewer Agent**：每个 Reviewer 有专门的 system prompt 和审查逻辑
3. **规则 + LLM 混合审查**：优先用 LLM 深度审查，失败时降级为正则规则 mock
4. **严重程度排序**：Critical → Warning → Info 三级分类，同级别按行号排序
5. **离线 Mock**：预设含问题的代码片段，各 Reviewer 用正则规则审查，`exit 0`

---

## 架构概览

```
代码片段
   │
   ▼
┌──────────────┐
│  Supervisor  │  ← 协调者：分派任务、收集结果、汇总排序
└──┬───┬───┬───┘
   │   │   │
   ▼   ▼   ▼
┌──────┐ ┌──────┐ ┌──────┐
│安全   │ │性能   │ │风格   │  ← 3 个专业 Reviewer Agent
│审查   │ │审查   │ │审查   │     各有专门 system prompt
└──┬───┘ └──┬───┘ └──┬───┘
   │        │        │
   ▼        ▼        ▼
┌──────────────────────────┐
│     汇总报告（排序）      │  ← 按 Critical → Warning → Info 排序
└──────────────────────────┘
```

---

## 3 个 Reviewer Agent

| Reviewer | 职责 | 检测规则 |
|----------|------|----------|
| **Security** | 安全审查 | SQL 注入 (`execute.*%s`)、硬编码密码、XSS |
| **Performance** | 性能审查 | O(n²) 嵌套循环 (`for...for`)、不必要拷贝、线性查找 |
| **Style** | 风格审查 | 模糊命名 (`temp`/`data`)、缺少类型注解、缺少 docstring |

---

## 运行方式

```bash
cd ai-agent/projects/03-multi-agent-review

# Python
python3 python/main.py

# TypeScript
npx tsx typescript/main.ts
```

输出前缀：`OUT:supervisor:` / `OUT:reviewer:security:` / `OUT:reviewer:performance:` / `OUT:reviewer:style:` / `OUT:report:`

---

## 离线设计

`.env` 中 API 密钥为占位符 `sk-REPLACE-ME` 时：
- **Supervisor**：正常协调流程
- **各 Reviewer**：try LLM 失败 → 使用正则规则 mock 审查
- **汇总报告**：纯本地排序，不依赖 API

全程 **不依赖真实 API**，`exit 0`。

---

## 示例代码问题

内置示例代码包含以下问题（用于演示审查效果）：

| 问题 | 严重程度 | Reviewer |
|------|----------|----------|
| SQL 注入：`execute("...%s" % user_id)` | 🔴 Critical | Security |
| 硬编码密码：`password = "admin123"` | 🔴 Critical | Security |
| O(n²) 嵌套循环：双重 for 循环查找重复 | 🟡 Warning | Performance |
| 线性查找：`not in duplicates`（应用 set） | 🟡 Warning | Performance |
| 不必要拷贝：`result = temp` | 🔵 Info | Performance |
| 模糊命名：`temp`、`data`、`result` | 🔵 Info | Style |
| 缺少类型注解 | 🔵 Info | Style |
| 缺少 docstring | 🔵 Info | Style |

---

## 代码

- [Python 实现](./python/main.py)
- [TypeScript 实现](./typescript/main.ts)
- [练习题](./exercises/README.md)
