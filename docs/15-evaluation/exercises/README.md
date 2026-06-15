# 第15章 练习 — 评估与测试

> 动手实践行为测试、LLM-as-Judge 和回归测试套件。每个练习都附参考答案。

---

## 练习 1：判断行为测试的 PASS/FAIL（理解题）

运行 `python3 python/main.py`（或 `npx tsx typescript/main.ts`），观察 Demo 1（行为测试）的输出。

下面给出了 5 个测试场景。对每个场景，判断行为测试应该 PASS 还是 FAIL，
并说明理由（从"期望工具"和"实际工具"是否匹配的角度分析）。

| # | 场景 | 期望工具 | 实际工具调用 | PASS/FAIL？ |
|---|------|----------|-------------|-------------|
| A | "查北京天气" | `["get_weather"]` | `["get_weather"]` | ? |
| B | "查北京天气" | `["get_weather"]` | `[]`（Agent 直接猜了答案） | ? |
| C | "你好" | `[]` | `["get_weather"]`（Agent 闲聊还调了天气） | ? |
| D | "查北京上海天气并算温差" | `["get_weather", "calculate"]` | `["get_weather"]`（只查了一个城市） | ? |
| E | "查北京上海天气并算温差" | `["get_weather", "calculate"]` | `["get_weather", "get_weather", "calculate"]` | ? |

**问题**：
1. 哪些 PASS，哪些 FAIL？
2. 场景 B 的 Agent 输出可能包含正确答案（比如"北京今天晴"），但行为测试仍然 FAIL。为什么这是好事？
3. 场景 E 的实际调用比期望多了个 `get_weather`。这算 PASS 还是 FAIL？为什么？

**参考答案**：

| # | PASS/FAIL | 理由 |
|---|-----------|------|
| A | **PASS** | 期望的 `get_weather` 被调了，完全匹配 |
| B | **FAIL** | 期望调 `get_weather` 但实际没调。输出碰巧对不代表行为对 |
| C | **FAIL** | 期望不调工具（`[]`），但实际调了 `get_weather`。闲聊不应该调工具 |
| D | **FAIL** | 期望 `["get_weather", "calculate"]`，但缺少 `calculate`。Agent 可能只查了一个城市就停了 |
| E | **PASS** | 期望的 `get_weather` 和 `calculate` 都被调了。多一次 `get_weather` 是合理的（查了两个城市） |

1. PASS: A、E；FAIL: B、C、D。

2. 场景 B 的行为测试 FAIL 是好事，因为它抓住了**隐蔽 bug**：Agent 没调天气工具就直接猜答案。
   这种 Agent 在天气变化时会给出过时信息，或者在陌生城市上完全失效。只测输出发现不了这个问题，
   但行为测试能。**行为测试的核心价值就是区分"真做对了"和"碰巧蒙对了"**。

3. 场景 E 是 **PASS**。行为测试检查的是"期望的工具都被调了"（子集关系），
   不是"调的工具完全等于期望"。查两个城市需要调两次 `get_weather`，多出来的那次是正常行为。
   如果要求严格相等（不允许额外调用），会误杀很多合理场景。

---

## 练习 2：为"任务助手 Agent"编写行为测试用例（编程题）

假设你的任务助手 Agent 有以下 4 个工具：

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `get_weather` | 查天气 | `city: str` |
| `create_todo` | 创建待办 | `title: str, due: str` |
| `search_web` | 搜索网页 | `query: str` |
| `calculate` | 数学计算 | `expression: str` |

请编写 **5 个行为测试用例**，要求覆盖：

1. 单工具任务（Agent 只需要调一个工具）
2. 多工具串行任务（Agent 需要按顺序调多个工具）
3. 不需要工具的闲聊（Agent 不应该调任何工具）
4. 边界场景：模糊指令（Agent 需要自行判断该调什么工具）
5. 边界场景：信息不足（Agent 应该调工具获取信息，而不是瞎猜）

**要求**：
- 每个用例包含 `name`、`task`、`expected_tools`、`description`
- `expected_tools` 必须是可断言的（具体的工具名列表）
- 用 Python `@dataclass` 或 TypeScript `interface` 定义用例结构

**参考答案（Python 版）**：

```python
from dataclasses import dataclass

@dataclass
class BehaviorTestCase:
    name: str
    task: str
    expected_tools: list[str]
    description: str

test_cases = [
    # 1. 单工具任务
    BehaviorTestCase(
        name="single_tool_weather",
        task="查一下北京今天天气怎么样",
        expected_tools=["get_weather"],
        description="简单天气查询，Agent 应该只调 get_weather",
    ),

    # 2. 多工具串行任务
    BehaviorTestCase(
        name="multi_tool_weather_and_calc",
        task="查北京和上海的天气，然后算一下两地温差",
        expected_tools=["get_weather", "calculate"],
        description="需要先查两个城市天气，再用 calculate 算温差",
    ),

    # 3. 不需要工具的闲聊
    BehaviorTestCase(
        name="no_tool_chat",
        task="你好，给我讲个笑话吧",
        expected_tools=[],
        description="闲聊不应该调用任何工具",
    ),

    # 4. 边界：模糊指令
    BehaviorTestCase(
        name="ambiguous_needs_todo",
        task="提醒我明天下午三点开会",
        expected_tools=["create_todo"],
        description="模糊指令，但 Agent 应该判断出需要创建待办",
    ),

    # 5. 边界：信息不足
    BehaviorTestCase(
        name="unknown_info_needs_search",
        task="2024年诺贝尔物理学奖得主是谁",
        expected_tools=["search_web"],
        description="Agent 不可能凭记忆回答，应该调 search_web 而不是瞎猜",
    ),
]
```

**参考答案（TypeScript 版）**：

```typescript
interface BehaviorTestCase {
  name: string;
  task: string;
  expectedTools: string[];
  description: string;
}

const testCases: BehaviorTestCase[] = [
  {
    name: "single_tool_weather",
    task: "查一下北京今天天气怎么样",
    expectedTools: ["get_weather"],
    description: "简单天气查询，Agent 应该只调 get_weather",
  },
  {
    name: "multi_tool_weather_and_calc",
    task: "查北京和上海的天气，然后算一下两地温差",
    expectedTools: ["get_weather", "calculate"],
    description: "需要先查两个城市天气，再用 calculate 算温差",
  },
  {
    name: "no_tool_chat",
    task: "你好，给我讲个笑话吧",
    expectedTools: [],
    description: "闲聊不应该调用任何工具",
  },
  {
    name: "ambiguous_needs_todo",
    task: "提醒我明天下午三点开会",
    expectedTools: ["create_todo"],
    description: "模糊指令，但 Agent 应该判断出需要创建待办",
  },
  {
    name: "unknown_info_needs_search",
    task: "2024年诺贝尔物理学奖得主是谁",
    expectedTools: ["search_web"],
    description: "Agent 不可能凭记忆回答，应该调 search_web 而不是瞎猜",
  },
];
```

**关键收获**：好的行为测试用例要满足两个条件：(1) `expected_tools` 具体到工具名，能写 assert；
(2) 覆盖正常路径和边界路径。20 个精准用例比 200 个模糊用例有用得多。

---

## 练习 3：设计 LLM-as-Judge 的评分标准（Rubric）

你的任务助手 Agent 会帮用户写邮件。你需要用 LLM-as-Judge 评估邮件质量。

以下是一份"不合格"的 Judge prompt 和一份"合格"的 Judge prompt。
对比两者，回答问题。

**不合格版**：

```
请给这封邮件打分（1-5）：{output}
```

**合格版**：

```
你是一个严格的邮件质量评分员。请按以下标准对邮件打分（1-5 分）。

评分维度：
- 专业性（语气是否得体、格式是否规范）
- 完整性（是否覆盖了用户要求的所有要点）
- 简洁性（是否没有废话、长度是否合理）

评分标准：
- 5 分：三个维度都优秀，可以直接发送
- 4 分：有一个维度有小瑕疵，修改后可发送
- 3 分：有两个维度有明显问题，需要重写部分内容
- 2 分：大部分维度不合格，需要大幅修改
- 1 分：完全不可用

用户要求: {task}
邮件内容: {output}

请输出 JSON: {{"score": 1-5, "comment": "评语", "issues": ["问题1", "问题2"]}}
```

**问题**：
1. 不合格版会导致 Judge 产生哪些偏见？至少举出 2 个。
2. 合格版里"简洁性"维度的作用是什么？它解决了哪个已知偏见？
3. 为什么合格版要求 Judge 和被评估的 Agent 用不同的模型？
4. 请给这个 Judge prompt 再加一个维度，使评分更全面。写出你的维度名称和评分标准。

**参考答案**：

1. 不合格版会导致的偏见：
   - **冗长偏见**：没有"简洁性"标准，Judge 会默认更长的邮件更好，哪怕里面全是废话。
   - **标准漂移**：没有明确的分档定义，同一个邮件今天打 4 分明天打 3 分，Judge 凭感觉打分。
   - **自我偏爱**：如果 Judge 和 Agent 用同一个模型，Judge 会偏好和自己风格相似的输出。

2. "简洁性"维度直接对抗**冗长偏见**。有了这个维度，Judge 会被明确要求：
   "废话扣分"。一封 50 字说清楚的邮件比 500 字的废话得更高分。Rubric 的每个维度
   都是给 Judge 的一个"锚点"，防止它按自己的默认倾向打分。

3. 同一个模型评自己会产生**自我偏爱**偏见。比如 GPT-4o 评 GPT-4o 的输出，
   会倾向于给自己风格的回答更高分。用不同的模型做 Judge（比如用 Claude 评 GPT 的输出），
   能减少这种系统性偏差。生产中常见的组合是用便宜模型做 Judge（如 GPT-4o-mini），
   评估贵模型（如 GPT-4o）的输出。

4. 可以加的维度示例：

   **准确性**（邮件内容是否事实正确、数字是否准确）：
   - 5 分：所有事实和数字都正确
   - 4 分：有一处小错误（如日期差一天）
   - 3 分：有明显事实错误但不影响核心信息
   - 2 分：关键信息有误（如金额写错）
   - 1 分：大量事实错误，完全不可信

   这个维度补充了"专业性"和"完整性"无法覆盖的问题：一封写得再专业、再完整的邮件，
   如果里面的金额或日期是错的，也是废纸一张。

**关键收获**：Rubric 是 LLM-as-Judge 的灵魂。没有 rubric，Judge 按自己的默认倾向打分
（偏向长回答、偏向自己的风格）。每个 rubric 维度都是一个"校准锚点"，
把 Judge 的行为拉到你期望的方向上。

---

## 练习 4：构建回归测试套件并解读报告（综合题）

你给 Agent 换了一个新模型（从 GPT-4o 换成 DeepSeek），跑了一轮回归测试套件，
得到以下报告：

```
回归测试报告（共 8 个用例）
========================================
✓ PASS  weather_query          查北京天气 → 调了 get_weather
✓ PASS  no_tool_chat           你好 → 没调工具
✗ FAIL  weather_and_calc       查两地天气并算温差 → 只调了 get_weather，没调 calculate
✗ FAIL  todo_creation          提醒我开会 → 调了 search_web（应该调 create_todo）
✓ PASS  web_search             搜最新新闻 → 调了 search_web
✓ PASS  multi_weather          查三个城市天气 → 调了 3 次 get_weather
✗ FAIL  empty_input            (空字符串) → 调了 get_weather（应该不调工具）
✓ PASS  complex_research       研究量子计算最新进展 → 调了 search_web
========================================
总结: 5 PASS / 3 FAIL
```

**问题**：
1. 三个 FAIL 分别属于什么类型的行为退化？（选错工具、多余调用、漏掉调用、流程不完整）
2. 哪个 FAIL 最严重？为什么？
3. 你会怎么处理这次失败？直接回滚、改 prompt、还是继续上线？
4. 如果同一个测试跑 10 次，8 次 PASS 2 次 FAIL，你觉得应该怎么判断？这和传统单元测试有什么不同？

**参考答案**：

1. 三个 FAIL 的类型：
   - **weather_and_calc**：**流程不完整**（漏掉调用）。Agent 查了天气但没走到下一步调 calculate。
     新模型可能在查完天气后就直接给答案了，跳过了计算步骤。
   - **todo_creation**：**选错工具**。应该调 `create_todo` 但调了 `search_web`。
     新模型可能把"提醒我开会"理解成了"搜索开会相关的信息"。
   - **empty_input**：**多余调用**。空输入不应该触发任何工具调用，但 Agent 调了 `get_weather`。
     新模型可能对空输入的处理逻辑不同。

2. **todo_creation 最严重**。原因是：
   - 选错工具是**语义理解错误**（模型根本没理解用户意图），比流程不完整更难通过改 prompt 修复
   - 它会直接影响用户体验（用户要的是待办，Agent 给了一堆搜索结果）
   - 暗示新模型在"指令意图识别"上有系统性偏差，可能影响其他类似场景

3. 处理建议：
   - **不要直接回滚**，也不要直接上线。先诊断。
   - weather_and_calc 的问题可能通过改 prompt 修复（加一句"查完天气后如果有计算需求，
     必须调用 calculate 工具"）。
   - todo_creation 需要检查新模型对 `create_todo` 工具描述的理解，
     可能需要优化工具描述（加更多示例）。
   - empty_input 需要在 system prompt 里明确"如果用户输入为空或无意义，直接回复询问，不要调工具"。
   - **修完后重跑套件**。如果 3 个 FAIL 都修复了，可以上线。如果还有 1 个以上 FAIL，考虑回滚。

4. 8 次 PASS / 2 次 FAIL 应该**算 PASS（带标记）**。这是 Agent 回归测试和传统单元测试的核心区别：
   - **传统单元测试**：同样输入同样输出，100% 确定性。失败 1 次 = 有 bug。
   - **Agent 回归测试**：LLM 有随机性（即使 temperature=0 也有微小波动）。
     同一个测试跑 10 次，偶尔抽风 1-2 次是正常的。
   - **生产建议**：设一个容忍度阈值，比如 **80% 通过率**（10 次跑 8 次过就算过）。
     但如果一个测试从"10/10 过"变成"8/10 过"，虽然还在阈值内，也要标记为"稳定性下降"，
     纳入观察。`temperature=0` 可以降低随机性但不能完全消除。

**关键收获**：回归测试报告不只是 PASS/FAIL 计数。每个 FAIL 都要分类（选错工具、漏调用、多余调用），
判断严重程度，然后决定修复策略。LLM 的随机性意味着你要接受"偶尔失败"，
但这和"系统性退化"是两码事——套件的价值就是帮你区分这两者。

---

## 运行本章代码

```bash
# Python（行为测试 + LLM-as-Judge + 回归套件 三个 demo）
cd ai-agent/15-evaluation
pip install -r python/requirements.txt
python3 python/main.py

# TypeScript（对等实现）
cd ai-agent/15-evaluation
npx tsx typescript/main.ts
```

完成后，尝试上面的练习。练习 1 最简单（理解 PASS/FAIL），练习 4 最有挑战（综合分析回归报告）。
