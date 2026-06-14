# 多 Agent 系统

## 为什么需要多 Agent

单个 Agent 在处理复杂任务时面临三个挑战：
1. **工具过载**：工具太多，模型选择困难
2. **上下文过长**：历史太长，检索精度下降
3. **角色冲突**：一个 Agent 同时扮演多个角色，容易混乱

## Supervisor-Worker 模式

最常见的多 Agent 架构：
- **Supervisor**：接收任务，分解后分派给专门的 Worker
- **Worker**：各司其职，完成后向 Supervisor 汇报

### 典型 Worker 角色
- Researcher：负责信息检索和整理
- Writer：负责内容撰写和润色
- Coder：负责代码生成和调试
- Reviewer：负责质量审查和反馈

## Agent Handoffs

任务转交机制：当一个 Agent 遇到超出能力范围的问题时，将任务连同上下文一起交给更合适的 Agent。

例如：客服 Agent 遇到技术问题 → 转交技术专家 Agent。

## 共享记忆

多个 Agent 之间需要共享信息：
- **消息传递**：Agent 直接发送结构化消息
- **共享记忆空间**：所有 Agent 可读写的公共存储
- **上下文隔离**：每个 Agent 只看到自己需要的信息

## 挑战

- 通信协议设计
- 死锁和活锁避免
- 共享状态一致性
- 成本控制（多次 LLM 调用）
