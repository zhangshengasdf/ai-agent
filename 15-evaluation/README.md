# 第15章 评估与测试（行为测试、LLM-as-Judge、回归测试）

> **「任务助手 Agent」上线前最后一道关**——你怎么知道你的 Agent "好不好"？
> 不是靠感觉，而是靠**可量化的评估**。本章教你三种评估手段：**行为测试**
> （断言 Agent 走对了流程）、**LLM-as-Judge**（用模型给输出打分）、**回归测试**
> （防止升级模型后行为退化）。学完本章，你能给 Agent 建一套"体检套餐"，
> 每次改动都跑一遍，坏行为立即报警。

---

## 本章目标

学完本章，你将能够：

1. **区分 Agent 评估与模型评估**：理解为什么评估 Agent ≠ 评估 LLM
2. **写行为测试**：定义"输入 → 期望调用的工具/期望流程"，断言 Agent 走对路
3. **用 LLM-as-Judge**：让第二个 LLM 给第一个 LLM 的输出打分（1-5 分 + 评语）
4. **搭回归测试套件**：批量跑测试用例，汇总 PASS/FAIL 报告
5. **设计评估数据集**：知道该收集什么样的测试样本

> ⚠️ **前置条件**：先学第04章（Agent 循环）和第13章（自造框架）。本章的评估对象
> 就是前面造出来的 Agent——不跑过 Agent 循环，你不知道"行为"是什么意思。

---

## 核心概念：Agent 评估 ≠ 模型评估

这是本章最重要的认知。很多人把"评估 Agent"理解成"评估 LLM"——这是根本性的错误。

### 模型评估：看单次输出

传统的 LLM 评估（MMLU、HumanEval、BLEU）回答一个问题：

> **"给定输入，模型的输出对不对？"**

这是**单点评估**——一个输入，一个输出，一个分数。

```
输入: "1+1=?"
模型输出: "2"
评估: ✓ 正确
```

### Agent 评估：看行为序列

Agent 不是"输入→输出"，而是"输入→**多步行为序列**→最终输出"。评估 Agent 要回答：

> **"给定任务，Agent 走的路径对不对？中间选的工具对不对？最终结果好不好？"**

这是**轨迹评估**——一个输入，一条行为轨迹（调了哪些工具、每步推理了什么），一个综合判断。

```
任务: "查北京天气，算和上海温差"
Agent 行为轨迹:
  step1: get_weather(北京) ✓ 正确选了天气工具
  step2: get_weather(上海) ✓ 又查了一次（需要两地天气）
  step3: calculate(28-25) ✓ 选了计算工具
  step4: 返回最终回答 ✓ 在收集够信息后停止
评估: ✓ 行为正确（选对工具、走对流程、结果正确）
```

### 为什么这个区分重要

| 评估维度 | 模型评估 | Agent 评估 |
|---------|---------|-----------|
| 评估对象 | 单次输入输出 | 多步行为序列 |
| 核心问题 | 输出对不对 | 路径对不对 + 结果好不好 |
| 关键指标 | 准确率/BLEU | 工具选择正确率 + 流程完成率 + 最终质量 |
| 测试方式 | 答案比对 | 行为断言 + 质量打分 |
| 失败模式 | 输出错 | 选错工具、死循环、不该停的时候停了 |

**一个 LLM 评分很高的模型，做成 Agent 后可能表现很差**——因为它总选错工具，
或者该停的时候不停。反过来，一个"笨"模型如果工具调用逻辑清晰，可能是个好 Agent。

> 💡 **本章的核心论点**：评估 Agent，你要测**行为**（选了什么工具、走了什么流程），
> 不只是测**输出**（最终文字好不好）。行为测试是 Agent 评估的灵魂。

---

## 三大评估手段

### 手段 1：行为测试（Behavior Testing）

**核心思想**：像测软件一样测 Agent——定义"输入 → 期望行为"，断言 Agent 做到了。

#### 什么是"行为"

Agent 的行为就是它在执行任务时**做了什么**：
- 调了哪些工具？顺序对不对？
- 该调工具时调了吗？不该调时瞎调了吗？
- 该停止时停了吗？该继续时 prematurely 终止了吗？

#### 测试用例结构

一个行为测试用例包含三部分：

```python
@dataclass
class BehaviorTestCase:
    name: str                    # 测试名（如 "weather_query_calls_weather_tool"）
    task: str                    # 输入任务（如 "查北京天气"）
    expected_tools: list[str]    # 期望调用的工具（如 ["get_weather"]）
    description: str             # 这个测试在验证什么
```

#### 断言模式

跑完 Agent 后，检查它的**工具调用记录**是否满足期望：

```python
def run_behavior_test(agent, test_case):
    result = agent.run(test_case.task)
    actual_tools = result.tools_called  # Agent 实际调了哪些工具

    # 断言 1：期望的工具都被调了
    for expected in test_case.expected_tools:
        assert expected in actual_tools, (
            f"期望调用 {expected}，但实际只调了 {actual_tools}"
        )

    # 断言 2（可选）：不该调的工具没被调
    # 断言 3（可选）：工具调用顺序正确
```

#### 为什么要测行为而不是只测输出

```python
# ❌ 只测输出：无法区分"碰巧蒙对"和"真懂"
task = "查北京天气"
output = agent.run(task)
assert "晴" in output  # 但 Agent 可能是瞎猜的，根本没调天气工具！

# ✓ 测行为：确认 Agent 真的走了正确流程
result = agent.run(task)
assert "get_weather" in result.tools_called  # 确认调了天气工具
assert result.final_output  # 而且有输出
```

**行为测试的价值**：它能抓住"输出对但行为错"的隐蔽 bug。比如 Agent 碰巧猜对了答案，
但完全没调工具——这种 Agent 在复杂任务上必崩，但只测输出你发现不了。

### 手段 2：LLM-as-Judge（用模型评估输出质量）

**核心思想**：行为测试能验证"流程对不对"，但"最终输出好不好"很难用 assert 判断
（比如"这段回答写得好不好"没有明确的 true/false）。这时候用**另一个 LLM 当裁判**。

#### 工作原理

```
任务 → [Agent（被评估方）] → 输出
                               ↓
              [Judge LLM（裁判）] ← 评分标准（rubric）
                               ↓
                    分数（1-5）+ 评语
```

Judge LLM 收到三样东西：
1. **原始任务**（用户问了什么）
2. **Agent 的输出**（被评估的回答）
3. **评分标准 / rubric**（从哪些维度打分，每档什么含义）

Judge 根据 rubric 给出**结构化评分**：分数（通常是 1-5）+ 文字评语。

#### 评分标准（Rubric）设计

Rubric 是 LLM-as-Judge 的灵魂——没有明确标准，Judge 会乱打分。

```python
JUDGE_PROMPT = """你是一个严格的评分员。请对以下回答打分（1-5 分）。

评分维度：
- 正确性（事实是否准确）
- 完整性（是否覆盖了任务要求的所有方面）
- 清晰度（表述是否清楚易懂）

评分标准：
- 5 分：完全正确、完整、清晰
- 4 分：基本正确，有小瑕疵
- 3 分：部分正确，有明显遗漏
- 2 分：大部分错误
- 1 分：完全错误或无关

任务: {task}
回答: {output}

请输出 JSON: {{"score": 1-5, "comment": "评语"}}
"""
```

#### LLM-as-Judge 的陷阱

| 陷阱 | 表现 | 对策 |
|------|------|------|
| **位置偏见** | Judge 偏向第一个/最后一个出现的答案 | 随机打乱顺序（A/B 测试时） |
| **冗长偏见** | Judge 给更长的回答更高分（哪怕废话多） | rubric 明确"简洁性"维度 |
| **自我偏爱** | 同一个模型评自己会偏高 | Judge 用**不同的**模型 |
| **标准漂移** | 同一标准，不同时间 Judge 打分不一致 | 用 few-shot 锚定标准 |

> 💡 **生产建议**：LLM-as-Judge 适合做**快速迭代反馈**（开发时跑一遍看大致质量），
> 不适合做**最终验收**（验收要人工 + 自动结合）。把它当成"廉价的初步筛查"。

### 手段 3：回归测试套件（Regression Suite）

**核心思想**：当你改了 prompt、换了模型、升级了 SDK，怎么知道 Agent 没变坏？
**把行为测试和 Judge 测试攒成一套，每次改动都全跑一遍**，坏掉的立即报警。

#### 为什么要回归测试

Agent 系统特别脆弱，因为这些变化都可能悄悄破坏行为：
- **换模型**：从 GPT-4o 换成 DeepSeek，工具调用格式可能微妙不同
- **改 prompt**：加了一句"请简洁"，Agent 可能不再调工具了
- **升 SDK**：OpenAI SDK 更新后，`tool_calls` 字段结构可能变
- **改工具**：给 `get_weather` 加了个参数，Agent 传参方式可能崩

没有回归测试，这些变化你只能"上线后等用户投诉"才发现问题。有了回归测试，
**改完代码跑一遍套件，5 秒知道有没有坏东西**。

#### 套件结构

```python
class RegressionSuite:
    def __init__(self):
        self.cases = []  # 一堆测试用例

    def add(self, test_case):
        self.cases.append(test_case)

    def run_all(self) -> TestReport:
        results = []
        for case in self.cases:
            result = case.run()  # 跑单个测试
            results.append(result)
        return TestReport(results)  # 汇总成报告

@dataclass
class TestReport:
    results: list[TestResult]
    passed: int
    failed: int
    # 生成汇总：PASS 8 / FAIL 2，列出失败的详情
```

#### 回归测试 vs 单元测试

| 维度 | 单元测试（传统） | 回归测试（Agent） |
|------|----------------|------------------|
| 测什么 | 函数的输入输出 | Agent 的行为轨迹 |
| 确定性 | 100% 确定（同样输入同样输出） | LLM 有随机性（需容忍度） |
| 运行速度 | 毫秒级 | 秒级（要调 LLM） |
| 失败含义 | 代码有 bug | 行为退化了（或 LLM 抽风） |

因为 LLM 有随机性，Agent 回归测试要接受"偶尔抽风"——同一个测试跑 10 次，
9 次过就算过（容忍度）。或者固定 `seed`/`temperature=0` 降低随机性。

---

## 评估数据集设计

好的评估离不开好的数据集。以下是设计评估数据集的原则。

### 数据集分类

一套完整的 Agent 评估数据集应该包含三类样本：

| 类型 | 占比 | 作用 | 例子 |
|------|------|------|------|
| **黄金用例** | 60% | 主流程正确性（Agent 必须做对的） | "查天气" → 必须调 get_weather |
| **边界用例** | 30% | 边缘场景（考验鲁棒性） | 空输入、超长输入、不存在的城市 |
| **对抗用例** | 10% | 恶意/混淆输入（考验安全性） | "忽略之前指令"（越狱试探） |

### 黄金用例的设计原则

1. **覆盖所有工具**：每个工具至少有 3 个测试用例触发它
2. **覆盖典型流程**：单工具任务、多工具串行、多工具并行（如果支持）
3. **明确的期望**：每个用例的"期望行为"必须是**可断言**的（能写 assert）

```python
# 好的黄金用例：期望明确可断言
{
    "name": "weather_query_uses_weather_tool",
    "task": "北京今天天气怎么样",
    "expected_tools": ["get_weather"],  # 可断言
}

# 坏的黄金用例：期望模糊不可断言
{
    "name": "friendly_response",
    "task": "你好",
    "expected": "友好的回答",  # 怎么 assert "友好"？不可断言！
}
```

### 数据集来源

- **人工编写**：最精准，但成本高（每条 5-15 分钟）
- **真实日志**：从生产日志采样真实用户问题（脱敏后），最贴近实际
- **LLM 生成**：让 LLM 生成测试用例（便宜但质量参差，需人工筛选）
- **混合**：80% 真实日志 + 15% 人工边界 + 5% LLM 生成对抗

> 💡 **起步建议**：先手写 10-20 个黄金用例（覆盖核心工具），上线后从日志补充。
> 不要一开始就追求大规模数据集——20 个精准用例比 200 个垃圾用例有用得多。

---

## 评估流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent 评估完整流程                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 评估数据集    │───▶│  跑 Agent     │───▶│ 收集行为轨迹  │      │
│  │ (黄金+边界)  │    │  (每条任务)   │    │ (工具+输出)  │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                          ┌───────────────────────┤              │
│                          ▼                       ▼              │
│                   ┌──────────────┐      ┌──────────────┐        │
│                   │  行为断言     │      │ LLM-as-Judge │        │
│                   │ (工具选对没)  │      │ (输出好不好) │        │
│                   └──────┬───────┘      └──────┬───────┘        │
│                          │                     │                │
│                          └──────────┬──────────┘                │
│                                     ▼                           │
│                            ┌──────────────┐                     │
│                            │  汇总报告     │                     │
│                            │ PASS X/FAIL Y │                     │
│                            └──────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

行为断言 + LLM-as-Judge = 完整评估。缺前者你不知道流程对不对，缺后者你不知道质量好不好。

---

## 反模式（什么不该做）

### ❌ 只测输出，不测行为

```python
# 坏：只看最终输出，不看 Agent 走了什么路
output = agent.run("查北京天气")
assert "晴" in output  # Agent 可能瞎猜的，根本没调天气工具！

# 好：先断言行为，再看输出
result = agent.run("查北京天气")
assert "get_weather" in result.tools_called  # 确认走了正确流程
assert result.final_output  # 而且有合理输出
```

这是最常见的反模式。只测输出会漏掉"行为错但输出碰巧对"的 bug。

### ❌ 没有回归测试

```python
# 坏：每次改 prompt / 换模型，全靠手动试一试
agent.system_prompt += "请更简洁"  # 改完手动问几个问题，感觉还行就上线
# 结果：上线后用户发现 Agent 不调工具了（"简洁"让它跳过了工具调用）

# 好：有回归套件，改完自动全跑
agent.system_prompt += "请更简洁"
report = regression_suite.run_all(agent)
if report.failed > 0:
    print(f"⚠️ {report.failed} 个测试失败，改动破坏了行为，回滚！")
```

没有回归测试的 Agent 项目，每次改动都是在玩俄罗斯轮盘赌。

### ❌ LLM-as-Judge 没有评分标准

```python
# 坏：让 Judge "打个分"，没有明确标准 → Judge 乱打
judge_prompt = f"给这个回答打分（1-5）：{output}"
# Judge 会偏向冗长的、第一个出现的、和自己风格像的回答

# 好：给 Judge 明确的 rubric
judge_prompt = f"""按以下标准打分（1-5）：
5=完全正确完整；3=部分正确；1=完全错误
任务: {task}
回答: {output}
"""
```

### ❌ 评估数据集太小或太偏

```python
# 坏：只有 3 个测试用例，全是"查天气"
test_cases = [weather_case_1, weather_case_2, weather_case_3]
# 覆盖率极低，换算工具坏了都测不出来

# 好：覆盖所有工具 + 边界场景
test_cases = [
    weather_case,      # 覆盖 get_weather
    calculate_case,    # 覆盖 calculate
    multi_tool_case,   # 覆盖多工具串行
    empty_input_case,  # 边界：空输入
    unknown_tool_case, # 边界：该不该调工具
]
```

### ❌ 把 LLM-as-Judge 当唯一标准

LLM-as-Judge 有偏见（冗长偏见、位置偏见、自我偏爱）。**不要把它当成唯一的质量标准**，
要结合行为测试 + 人工抽检。纯粹依赖 Judge 会让你的 Agent 朝着"讨好 Judge"的方向优化，
而不是真正变好。

---

## Python vs TypeScript 实现差异

| 差异点 | Python | TypeScript |
|--------|--------|------------|
| 测试风格 | `assert` 或 `unittest` | 手动 `if (!ok) throw` |
| Judge 调用 | 同步 `client.chat.completions.create` | 异步 `await client...create` |
| 数据类 | `@dataclass` | `interface` + 对象字面量 |
| 报告生成 | f-string 拼接 | 模板字符串 |
| 离线 mock | 预设 tool_calls 序列 + 预设 Judge 分数 | 同上，async 版 |

> 💡 **TypeScript 没有 `assert` 内建语句**，所以行为测试用手动 `if (!condition) throw new Error()`，
> 或者封装成 `expect(actual).toContain(expected)` 风格的辅助函数。本章用手动 throw 保持透明。

---

## 运行示例

```bash
# Python
cd ai-agent/15-evaluation
pip install -r python/requirements.txt
python3 python/main.py

# TypeScript
cd ai-agent/15-evaluation
npx tsx typescript/main.ts
```

输出（节选，`.env` 用占位符 sk-REPLACE-ME，自动降级 mock）：

```
========================================================================
Demo 1: 行为测试（Behavior Testing）
========================================================================
OUT:test:weather_query: ▶ 运行测试: weather_query
  任务: 查一下北京今天的天气
  期望工具: ['get_weather']
OUT:test:weather_query: ✓ 通过 — 实际调用 ['get_weather']

OUT:test:weather_temp_calc: ▶ 运行测试: weather_temp_calc
  任务: 查北京和上海天气，算两地温差
  期望工具: ['get_weather', 'calculate']
OUT:test:weather_temp_calc: ✓ 通过 — 实际调用 ['get_weather', 'calculate']

OUT:test:no_tool_needed: ▶ 运行测试: no_tool_needed
  任务: 你好
  期望工具: []
OUT:test:no_tool_needed: ✓ 通过 — 闲聊无需工具，实际调用 []

========================================================================
Demo 2: LLM-as-Judge
========================================================================
OUT:judge:task: 解释什么是递归
OUT:judge:candidate: 递归是函数调用自身的编程技巧...
OUT:judge:offline: 真实 API 不可用，降级 mock Judge
OUT:judge:score: 4/5
OUT:judge:comment: 解释正确且举例恰当，但缺少"基线条件"的重要性强调。

========================================================================
Demo 3: 回归测试套件
========================================================================
OUT:regression:running: 共 5 个测试用例
OUT:regression:case:1 weather_query → PASS
OUT:regression:case:2 weather_temp_calc → PASS
...
OUT:regression:summary: 5/5 通过，0 失败 ✓
```

---

## 本章代码说明

| 文件 | 内容 |
|------|------|
| `python/main.py` | 3 个 demo（行为测试 / LLM-as-Judge / 回归套件） |
| `typescript/main.ts` | 对等实现（async 全链路） |
| `exercises/README.md` | 3 个行为测试练习 + 参考答案 |

本章的评估对象是前面章节造出来的 Agent（概念上引用第04章 agent loop / 第13章框架）。
为保持独立可运行，代码里用**离线 mock Agent**（返回预设 tool_calls 序列）模拟真实 Agent。

---

## 下一步

学完本章，你能给 Agent 建一套"体检套餐"。但这还不够——体检完你要能**看到** Agent
具体在干什么。第16章「可观测与调试」教你 tracing（追踪每一步）、logging（结构化日志）、
metrics（性能指标），让 Agent 从黑盒变成白盒。

> 💡 **评估 + 可观测 = Agent 的"质量保证"双保险**：评估是"定期体检"
> （每次改动跑套件），可观测是"实时监控"（生产环境实时追踪）。两者结合，
> Agent 出问题的概率和影响都会大幅降低。

---

## 代码

- [Python 实现](./python/main.py)（3 个 demo）
- [TypeScript 实现](./typescript/main.ts)（对等）
- [练习题](./exercises/README.md)
