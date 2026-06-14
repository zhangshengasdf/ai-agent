# 第08章 练习 — 规划模式（Plan-and-Execute、Reflection）

> 动手实践"先规划再执行"和"反思改进"两种高级推理模式。每个练习都附参考答案。

---

## 练习 1：用 Plan-and-Execute 写一个"研究报告"任务（核心编程题）

修改 `main.py` / `main.ts` 的离线 mock，用 **Plan-and-Execute** 模式完成一个"研究报告"任务：

> **任务："撰写一份关于'大语言模型在医疗领域的应用'的研究报告"**

**要求**：
1. 修改 `demo_plan_and_execute_offline()` / `demoPlanAndExecuteOffline()` 中的 `mock_plan` / `mockPlan`，设计 4-5 个合理的步骤（覆盖：检索应用场景、检索优势、检索挑战/风险、撰写报告）
2. 扩展 `mock_search()` / `mockSearch()` 的知识库，增加"医疗"、"LLM 医疗"等关键词
3. 每个步骤执行后，结果应反映该步骤的检索主题
4. 最后的汇总阶段输出一份结构化的报告（标题 + 各小节）
5. 运行代码，验证输出包含 `OUT:plan:`、`OUT:execute:step{N}:`、`OUT:synthesize:` 标记

**Python 参考答案**：

```python
# 1. 扩展 mock_search 知识库（在 knowledge dict 中添加）：
def mock_search(query: str) -> str:
    knowledge = {
        # ... 原有内容 ...
        "医疗": "大语言模型在医疗领域应用于辅助诊断、病历摘要、医学文献分析、患者问答等场景。",
        "LLM 医疗": "大语言模型在医疗领域应用于辅助诊断、病历摘要、医学文献分析、患者问答等场景。",
        "优势": "大语言模型在医疗的优势包括：提升医生效率、快速处理海量文献、辅助初步筛查。",
        "挑战": "大语言模型在医疗的挑战包括：幻觉风险、隐私合规、缺乏可解释性、监管不确定性。",
    }
    # ... 原有匹配逻辑 ...


# 2. 修改 demo_plan_and_execute_offline 的预设计划：
def demo_plan_and_execute_offline() -> str:
    print("离线演示：Plan-and-Execute（规划→执行→汇总）")
    task = "撰写一份关于'大语言模型在医疗领域的应用'的研究报告"

    mock_plan = Plan(steps=[
        "检索大语言模型在医疗领域的典型应用场景",
        "检索大语言模型在医疗领域的主要优势",
        "检索大语言模型在医疗领域的挑战与风险",
        "综合以上信息撰写研究报告",
    ])
    # ... 后续执行流程不变 ...
```

**TypeScript 参考答案**：

```typescript
// 1. 扩展 mockSearch 知识库：
function mockSearch(query: string): string {
  const knowledge: Record<string, string> = {
    // ... 原有内容 ...
    医疗: "大语言模型在医疗领域应用于辅助诊断、病历摘要、医学文献分析、患者问答等场景。",
    优势: "大语言模型在医疗的优势包括：提升医生效率、快速处理海量文献、辅助初步筛查。",
    挑战: "大语言模型在医疗的挑战包括：幻觉风险、隐私合规、缺乏可解释性、监管不确定性。",
  };
  // ... 原有匹配逻辑 ...
}

// 2. 修改 demoPlanAndExecuteOffline 的预设计划：
function demoPlanAndExecuteOffline(): string {
  const task = "撰写一份关于'大语言模型在医疗领域的应用'的研究报告";
  const mockPlan: Plan = {
    steps: [
      "检索大语言模型在医疗领域的典型应用场景",
      "检索大语言模型在医疗领域的主要优势",
      "检索大语言模型在医疗领域的挑战与风险",
      "综合以上信息撰写研究报告",
    ],
  };
  // ... 后续执行流程不变 ...
}
```

**验证**：运行后应看到 4 个步骤，每步执行结果与步骤主题对应，最后汇总输出一份含 3 个小节的报告。注意 Plan-and-Execute 的核心——**规划阶段一次性分解，执行阶段机械检索，汇总阶段综合**。

---

## 练习 2：追踪 Plan-and-Execute 的三阶段流程（理解题）

运行 `python3 python/main.py`（或 `npx tsx typescript/main.ts`），观察离线演示中 Plan-and-Execute 的输出（`OUT:plan:` / `OUT:execute:step{N}:` / `OUT:synthesize:` 标记）。

**问题**：
1. Plan-and-Execute 共分为哪几个阶段？每个阶段的输入和输出是什么？
2. 规划阶段（Phase 1）用了什么技术保证步骤是结构化的？（提示：`response_format` + Pydantic/interface）
3. 如果用 ReAct（第07章）解决同一个"写调研报告"任务，会有什么不同？列出至少 2 点区别。

**参考答案**：

1. 三个阶段：
   | 阶段 | 输入 | 输出 | 实现技术 |
   |------|------|------|----------|
   | Phase 1 (Plan) | 用户任务 | 步骤列表 `Plan{steps}` | `response_format=json_object` + Pydantic/interface 解析 |
   | Phase 2 (Execute) | 步骤列表 | 每步的执行结果 `StepResult[]` | 遍历步骤，调用工具/子查询 |
   | Phase 3 (Synthesize) | 任务 + 所有步骤结果 | 最终综合输出 | LLM 汇总调用 |

2. **结构化输出技术**：
   - Python：`response_format={"type": "json_object"}` 强制模型输出合法 JSON + `Plan.model_validate_json(raw)` 用 Pydantic 解析校验（保证 `steps` 是 `list[str]`）
   - TypeScript：`response_format: { type: "json_object" }` 强制 JSON + `JSON.parse(raw) as Plan` 解析（加 `Array.isArray` 校验）
   - 如果不用 `response_format`，模型可能输出自由文本（如"第一步，我们需要..."），无法程序化解析

3. **与 ReAct 的区别**：
   - **决策时机**：ReAct 每步都问模型"下一步干什么"（动态决策）；Plan-and-Execute 在规划阶段一次性决定所有步骤（静态计划）
   - **可预测性**：ReAct 的路径不确定（不同运行可能走不同步骤）；Plan-and-Execute 的步骤在执行前就确定了（可审查、可回放）
   - **LLM 调用次数**：ReAct N 步 = N 次调用；Plan-and-Execute = 1（规划）+ N（执行）+ 1（汇总）次
   - **可并行**：ReAct 步骤间隐式依赖（难并行）；Plan-and-Execute 独立步骤可并发执行

---

## 练习 3：理解 Reflection 的价值与代价（思考题）

观察离线演示中 Reflection 的输出（`OUT:reflect:draft:` / `OUT:reflect:critique:` / `OUT:reflect:revised:` 标记）。

**问题**：
1. Reflection 经历了哪三轮？每轮的角色/视角有什么不同？
2. 对比初版（draft）和改进版（revised），反思带来了哪些具体改进？
3. Reflection 的"代价"是什么？什么场景下不该用 Reflection？

**参考答案**：

1. **三轮流程**：
   | 轮次 | 角色/视角 | 输入 | 输出 |
   |------|-----------|------|------|
   | Round 1 (Draft) | 作者（生成者） | 用户问题 | 初版答案 |
   | Round 2 (Critique) | **审稿人（批评者）** | 问题 + 初版 | 不足清单（2-3 点） |
   | Round 3 (Revise) | 作者（改进者） | 问题 + 初版 + 批评 | 改进版 |

   **关键**：Round 2 切换了角色（从"作者"变"审稿人"），这让模型用不同视角审视自己的输出。反思 Prompt 还显式列出批评维度（完整性/准确性/结构），避免泛泛而谈。

2. **具体改进**（从演示输出可见）：
   - 初版（40 字）：只说"能自主完成任务、调用工具、理解指令"——简略、无层次
   - 改进版（239 字）：
     - **补充了三要素**（感知、决策、行动）——完整性提升
     - **区分了 Agent 与 LLM 的区别**（循环、工具、自主性）——准确性提升
     - **结构化呈现**（编号列表 + 加粗关键词）——结构提升

3. **代价与禁忌**：
   - **代价**：延迟翻 3 倍（3 次 LLM 调用）、成本翻 3 倍
   - **不该用的场景**：
     - 简单问答（"今天天气？"）——Reflection 是浪费
     - 实时对话（用户在等）——3 倍延迟难以接受
     - 分类/提取任务（输出本来就短）——反思没东西可改
   - **该用的场景**：写作、代码生成、重要决策、报告撰写——质量比速度重要

---

## 练习 4：设计反思 Prompt（编程题）

反思 Prompt 的设计直接决定反思质量。修改 `REFLECTION_PROMPT`，为一个**代码审查场景**设计反思流程。

**要求**：
1. 场景：Agent 生成了一个 Python 函数代码，现在要反思这段代码的质量
2. 设计一个 `CODE_REFLECTION_PROMPT`，批评维度应包括：正确性、可读性、性能、安全性
3. 设计一个 `CODE_REVISE_PROMPT`，根据批评改进代码
4. （可选）写一个离线 mock，预设一段"有问题的代码" → "批评" → "改进代码"的轨迹

**Python 参考答案**：

```python
CODE_REFLECTION_PROMPT = """\
你是一个严格的代码审查专家。请审视以下 Python 代码，指出它的不足。

用户需求：{requirement}
代码：
```python
{code}
```

请从以下维度批评：
1. 正确性：有没有 bug 或边界情况没处理？
2. 可读性：命名、注释、结构是否清晰？
3. 性能：有没有不必要的开销？
4. 安全性：有没有注入风险、异常吞没等问题？

只指出不足（2-3 点），不要给出完整改进版。
"""

CODE_REVISE_PROMPT = """\
用户需求：{requirement}
原代码：
```python
{code}
```
审查意见：{critique}

请根据审查意见改进代码，输出最终版。保持简洁，只输出代码。
"""

# 离线 mock 示例
def demo_code_reflection_offline() -> None:
    print("离线演示：代码反思")
    requirement = "写一个函数，读取文件内容并返回"
    bad_code = (
        "def read_file(path):\n"
        "    f = open(path)\n"        # 问题1：没用 with，资源泄漏
        "    return f.read()"          # 问题2：没处理异常
    )
    critique = (
        "1. 正确性：没用 with 语句，异常时文件不会关闭（资源泄漏）。\n"
        "2. 安全性：没处理 FileNotFoundError，文件不存在时崩溃。\n"
        "3. 可读性：缺少类型注解和文档字符串。"
    )
    improved_code = (
        "def read_file(path: str) -> str:\n"
        '    """读取文件内容，文件不存在时返回空字符串。"""\n'
        "    try:\n"
        "        with open(path, encoding='utf-8') as f:\n"
        "            return f.read()\n"
        "    except FileNotFoundError:\n"
        "        return ''"
    )
    print(f"OUT:reflect: 初版代码:\n{bad_code}")
    print(f"OUT:reflect: 批评:\n{critique}")
    print(f"OUT:reflect: 改进版:\n{improved_code}")
```

**为什么这个 Prompt 设计有效**：
- **角色切换**（"代码审查专家"而非"代码作者"）让模型用批评视角看代码
- **四个维度**（正确性/可读性/性能/安全性）覆盖了代码审查的核心关注点
- **"只指出不足"** 防止模型直接重写（跳过诊断），确保改进有据可依

---

## 练习 5（进阶）：何时该用规划——决策框架（思考题）

对于以下 5 个任务，判断该用 ReAct、Plan-and-Execute、还是 Reflection（或组合），并说明理由。

| 任务 | 你的选择 | 理由 |
|------|----------|------|
| A. "北京到上海的高铁票价？" | ? | ? |
| B. "调试这段报错的代码" | ? | ? |
| C. "写一份季度销售分析报告" | ? | ? |
| D. "给客户写一封重要道歉邮件" | ? | ? |
| E. "规划一个 7 天日本旅行" | ? | ? |

**参考答案**：

| 任务 | 选择 | 理由 |
|------|------|------|
| A. 高铁票价 | **都不用（直接调用）** | 单步查询任务，一次工具调用即可。强加规划=多此一举（反模式1） |
| B. 调试报错代码 | **ReAct** | 步骤未知——需要"读错误→搜索→试修复→再读错误"的探索式循环。规划无法预先知道要查什么 |
| C. 季度销售分析报告 | **Plan-and-Execute** | 多阶段任务：拉数据→分析趋势→对比目标→撰写报告。步骤明确，适合先规划 |
| D. 重要道歉邮件 | **Reflection** | 质量敏感——一封糟糕的道歉邮件会激化矛盾。先生成初版，再反思语气/诚恳度/措辞，改进 |
| E. 7 天日本旅行规划 | **Plan-and-Execute + Reflection** | 组合：先用 Plan 分解（订机票/订酒店/排行程/查签证），再用 Reflect 审视行程合理性（会不会太赶？有没有遗漏？） |

**决策要点总结**：
- **步骤未知 → ReAct**（边探索边做）
- **步骤明确 + 多阶段 → Plan-and-Execute**（先规划再执行）
- **质量敏感 → Reflection**（做完反思改进）
- **复杂任务 → 组合使用**（Plan + Execute + Reflect）
- **简单任务 → 别过度规划**（直接做，省延迟省成本）

> 💡 **真实世界的组合**：很多生产级 Agent（如深度研究助手）用的是 **Plan-and-Execute + Reflection** 组合——先规划检索步骤，执行检索，汇总成初版报告，再用 Reflection 审视报告的完整性/准确性，最后改进。这正是第08章为第10章（多 Agent）和实战项目（深度研究助手）打下的基础。
