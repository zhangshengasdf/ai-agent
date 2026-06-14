# 第14章 练习 — 从零造框架高级特性

> 这些练习帮你把第14章的 3 个高级特性（流式、结构化、工具校验）内化成肌肉记忆。
> 每个练习都附参考答案，但**先自己写一遍**再看答案——挫败感是学习的必经之路。

---

## 练习 1：给自造框架加 LLM 响应缓存（中等难度）

### 题目

在第13章的 `DefaultLLMClient` 基础上加一个**响应缓存**：相同 `(model, messages, tools)`
的请求第二次直接返回缓存结果，不调 API。

**要求**：
1. 用 `dict` 做内存缓存（key 是请求指纹，value 是归一化响应）
2. 加缓存命中率统计（`hit_count` / `miss_count`）
3. 加 `cache_clear()` 方法
4. 写一个 demo 演示：第一次 miss（调 API），第二次 hit（返回缓存），打印命中率

**提示**：
- key 的计算：`json.dumps({"model": ..., "messages": ..., "tools": ...}, sort_keys=True)`
  → `hashlib.sha256(json_str.encode()).hexdigest()`
- 第13章 `chat_with_retry` 已经有"重试 + mock 降级"，缓存应该加在最外层（重试之前）

### 参考答案

```python
import hashlib
import json
from typing import Any


class CachedLLMClient:
    """带响应缓存的 LLM 客户端（装饰第13章的 DefaultLLMClient）。"""

    def __init__(self, inner_client: Any) -> None:
        self._inner = inner_client
        self._cache: dict[str, dict[str, Any]] = {}
        self.hit_count = 0
        self.miss_count = 0

    def _cache_key(
        self, messages: list[dict], tools: list[dict] | None, **kwargs: Any
    ) -> str:
        payload = json.dumps(
            {"messages": messages, "tools": tools, "extra": kwargs},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def chat_with_retry(
        self, messages: list[dict], max_retries: int = 3, **kwargs: Any
    ) -> dict[str, Any]:
        key = self._cache_key(messages, kwargs.get("tools"), **kwargs)
        if key in self._cache:
            self.hit_count += 1
            print(f"OUT:cache:hit: 命中缓存（hit={self.hit_count}, miss={self.miss_count}）")
            return self._cache[key]

        self.miss_count += 1
        print(f"OUT:cache:miss: 缓存未命中，调 API（hit={self.hit_count}, miss={self.miss_count}）")
        response = self._inner.chat_with_retry(messages, max_retries=max_retries, **kwargs)
        self._cache[key] = response
        return response

    def cache_clear(self) -> None:
        self._cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


# ── 演示 ──
if __name__ == "__main__":
    # 假装 InnerClient 是第13章的 DefaultLLMClient（这里 mock）
    class MockInner:
        def __init__(self):
            self.call_count = 0

        def chat_with_retry(self, messages, max_retries=3, **kwargs):
            self.call_count += 1
            return {"content": f"回答 {self.call_count}", "tool_calls": None}

    inner = MockInner()
    cached = CachedLLMClient(inner)

    messages = [{"role": "user", "content": "你好"}]
    r1 = cached.chat_with_retry(messages)  # miss
    r2 = cached.chat_with_retry(messages)  # hit（相同 messages）
    r3 = cached.chat_with_retry(messages)  # hit

    messages_diff = [{"role": "user", "content": "再见"}]
    r4 = cached.chat_with_retry(messages_diff)  # miss（不同 messages）

    print(f"\n命中率: {cached.hit_rate:.2%}")  # 50%（2 hit / 4 total）
    print(f"inner 调用次数: {inner.call_count}")  # 2（r1, r4 真调了）
```

**学习要点**：
- 缓存是"装饰器模式"的典型应用——不修改原 LLMClient，包一层
- key 必须包含所有影响响应的字段（messages、tools、temperature 等）
- 生产场景要加 TTL（过期时间）和大小限制，否则内存爆炸

---

## 练习 2：结构化输出的"软校验"模式（中等难度）

### 题目

本章 Demo 2 的结构化输出是"硬校验"——校验失败就重试。但有些场景下，
LLM 输出的"近似正确"的 JSON 也可以用（比如 `priority: "4"` 是字符串但能转数字）。

**要求**：
1. 写一个 `soft_validate_task_summary(raw)` 函数，能自动修复常见小错：
   - 数字字段的字符串形式 → 转 number（`"4"` → `4`）
   - difficulty 大小写不敏感（`"Medium"` / `"MEDIUM"` → `"medium"`）
   - 缺 `estimated_hours` 字段时，默认填 `1.0`
2. 修复后用 `TaskSummary.model_validate` 再校验一次
3. 写 demo 演示 3 种修复（字符串数字、大小写、缺字段）

### 参考答案

```python
from pydantic import BaseModel, Field, ValidationError


class TaskSummary(BaseModel):
    name: str
    description: str
    difficulty: str  # easy|medium|hard
    priority: int = Field(ge=1, le=5)
    estimated_hours: float = Field(gt=0)


def soft_validate_task_summary(raw: dict):
    """软校验：先修复常见小错，再用 Pydantic 严格校验。"""
    fixed = dict(raw)

    # 修复 1：数字字段的字符串形式 → 转 number
    for num_field in ("priority", "estimated_hours"):
        val = fixed.get(num_field)
        if isinstance(val, str):
            try:
                fixed[num_field] = int(val) if num_field == "priority" else float(val)
            except ValueError:
                pass  # 转不了就留给 Pydantic 报错

    # 修复 2：difficulty 大小写不敏感
    diff = fixed.get("difficulty")
    if isinstance(diff, str):
        fixed["difficulty"] = diff.lower()

    # 修复 3：缺 estimated_hours 默认 1.0
    if "estimated_hours" not in fixed:
        fixed["estimated_hours"] = 1.0

    # 严格校验
    return TaskSummary.model_validate(fixed)


# ── 演示 ──
if __name__ == "__main__":
    test_cases = [
        # 字符串数字
        {"name": "A", "description": "desc", "difficulty": "medium",
         "priority": "4", "estimated_hours": "8.5"},
        # 大小写
        {"name": "B", "description": "desc", "difficulty": "MEDIUM",
         "priority": 3, "estimated_hours": 2.0},
        # 缺字段
        {"name": "C", "description": "desc", "difficulty": "easy",
         "priority": 2},
    ]

    for i, raw in enumerate(test_cases, 1):
        try:
            result = soft_validate_task_summary(raw)
            print(f"案例 {i}: ✓ {result.model_dump()}")
        except ValidationError as ve:
            print(f"案例 {i}: ✗ 仍校验失败 {ve.errors()}")
```

**学习要点**：
- "软校验"提升 LLM 输出的可用率（减少不必要的重试）
- 但有边界：不能修复语义错误（比如 priority=10 超范围还是得报错）
- 生产实践：先软校验，软校验失败再触发"硬重试"（反馈给 LLM）

---

## 练习 3：流式输出 + 结构化输出的组合（较难）

### 题目

本章 Demo 1（流式）和 Demo 2（结构化）是分开的。真实场景里，
你可能想要**"流式输出 + 最终结构化"**——用户边看流式文字，最后拿到一个 JSON 对象。

**要求**：
1. 写一个 `stream_then_parse(client, model, prompt, schema_class)` 函数：
   - 用 `stream=True` 逐块收集完整文本（边收边打印）
   - 流结束后，尝试 `schema_class.model_validate_json(full_text)`
   - 校验失败 → 重试整个流程（最多 3 次）
2. 写离线 mock 演示

**提示**：
- 真实 API 通常不会"边流式边 JSON"——而是流式完成后整体校验
- 进阶版：用 OpenAI 的 `response_format={"type": "json_schema", ...}` 模式（仅 OpenAI 支持）

### 参考答案

```python
import json
import re
import time
from typing import Any, Type
from pydantic import BaseModel, ValidationError


def stream_then_parse(
    client: Any,
    model: str,
    prompt: str,
    schema_class: Type[BaseModel],
    max_retries: int = 3,
) -> BaseModel:
    """流式收集 → 结构化校验 → 失败重试。"""
    messages = [
        {"role": "system", "content": f"输出符合此 schema 的 JSON: {schema_class.model_json_schema()}"},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, max_retries + 1):
        print(f"OUT:combo:attempt: 第 {attempt}/{max_retries} 次...")
        full_text = ""

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                response_format={"type": "json_object"},
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    print(f"OUT:combo:chunk: {delta}", end="", flush=True)
                    full_text += delta
            print()
        except Exception as e:
            print(f"OUT:combo:offline: API 失败（{type(e).__name__}），降级 mock")
            full_text = '{"name": "任务", "difficulty": "medium", "priority": 3}'

        try:
            result = schema_class.model_validate_json(full_text)
            print(f"OUT:combo:result: ✓ {result.model_dump()}")
            return result
        except ValidationError as ve:
            print(f"OUT:combo:retry: ✗ 校验失败 {ve.errors()[:1]}")
            messages.append({"role": "assistant", "content": full_text})
            messages.append({"role": "user", "content": f"修正: {ve.json()}"})

    raise RuntimeError("重试耗尽")


# 离线 mock 演示
def stream_then_parse_mock():
    """离线 mock：演示"流式 + 结构化"组合。"""
    class TaskSummary(BaseModel):
        name: str
        difficulty: str
        priority: int

    # 模拟 LLM 的流式输出（字符块）
    full_json = '{"name": "实现登录", "difficulty": "medium", "priority": 4}'
    chunks = re.findall(r".{1,5}", full_json)
    print("OUT:combo:mock: 流式输出 ↓")
    collected = ""
    for chunk in chunks:
        time.sleep(0.05)
        print(f"OUT:combo:chunk: {chunk}", end="", flush=True)
        collected += chunk
    print()

    result = TaskSummary.model_validate_json(collected)
    print(f"OUT:combo:result: ✓ 流式完成后校验通过 → {result.model_dump()}")
    return result


if __name__ == "__main__":
    stream_then_parse_mock()
```

**学习要点**：
- "流式 + 结构化"是真实生产场景（如 ChatGPT 的 function calling 流式模式）
- 流式过程中 JSON 是不完整的，必须等流结束才能整体 `JSON.parse`
- 进阶：用 [partial JSON parser](https://github.com/pydantic/pydantic-core) 解析不完整 JSON（流式中途就能预览部分字段）

---

## 练习 4（进阶）：工具参数校验 + 自动修复（难）

### 题目

本章 Demo 3 的工具校验是"硬失败"——类型不对就报错。但有些错误可以自动修复：
- `{"city": 123}` → `{"city": "123"}`（int → str 强转）
- `{"city": "北京 "}` → `{"city": "北京"}`（自动 trim）

**要求**：
1. 扩展 `validate_tool_args`，加一个 `auto_fix=True` 参数
2. 自动修复规则：string 字段接受任意类型 → `str(value).strip()`；number 字段接受字符串数字
3. 返回 `(fixed_args, errors)` 元组（修复后的 args + 剩余错误）
4. demo 演示：3 个可修复 + 1 个不可修复（必填字段缺失）

### 参考答案（要点）

```python
def validate_and_fix_args(args, schema, auto_fix=True):
    errors = []
    fixed = dict(args)

    # 1. 必填字段（无法自动修复）
    for field in schema.get("required", []):
        if field not in fixed:
            errors.append(f"缺少必填字段: '{field}'（无法自动修复）")

    # 2. 类型检查 + 自动修复
    for field, value in list(fixed.items()):
        if field not in schema.get("properties", {}):
            continue
        expected = schema["properties"][field].get("type")
        if expected == "string" and not isinstance(value, str):
            if auto_fix:
                fixed[field] = str(value).strip()
            else:
                errors.append(f"'{field}' 期望 string")
        elif expected in ("integer", "number") and isinstance(value, str):
            if auto_fix:
                try:
                    fixed[field] = int(value) if expected == "integer" else float(value)
                except ValueError:
                    errors.append(f"'{field}' 无法转为 {expected}")
            else:
                errors.append(f"'{field}' 期望 {expected}")

    return fixed, errors


# demo:
schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["city"],
}

# 可修复
fixed, errs = validate_and_fix_args({"city": 12345, "count": "5"}, schema)
# → fixed = {"city": "12345", "count": 5}, errs = []

# 不可修复（缺 city）
fixed, errs = validate_and_fix_args({"count": 5}, schema)
# → errs = ["缺少必填字段: 'city'（无法自动修复）"]
```

**学习要点**：
- 自动修复要保守——只修"明显无损"的转换（int↔str、trim）
- 不要自动修复枚举值（`"Medium"` → `"medium"` 是软校验的活，不是工具校验的活）
- 修复后必须再过一次 schema 校验（防御性编程）

---

## 练习 5（思考题）：何时该把自造框架切换成现代框架？

### 题目（开放讨论，无标准答案）

你刚用第12-14章的 mini 框架做了一个内部工具 Agent。现在产品要把它推给外部用户，
需求清单如下：

1. 需要 **流式输出**（用户等不了 5 秒空白）
2. 需要 **结构化输出**（前端要解析 JSON）
3. 需要 **多语言**（英文、日文、中文）
4. 需要 **tracing**（出 bug 要能追溯哪一步错了）
5. 需要 **A/B 测试**（对比两个 prompt 的效果）
6. 团队有 **3 个工程师**，未来扩到 8 个
7. 需求还会**持续演进**（未来要加 RAG、多 Agent）

**问题**：
1. 你会继续扩展 mini 框架，还是切到 OpenAI Agents SDK / Pydantic AI？为什么？
2. 如果切，选哪个框架？给出 3 个理由。
3. 如果继续用 mini 框架，列出需要自己实现的特性清单 + 估时。

### 参考思路（不是"答案"）

**应该切到现代框架**。理由：

- 需求清单里 5/7 项（流式、结构化、tracing、A/B、多语言）现代框架已内置
- 团队 3→8 人意味着需要"通用语言"——框架是行业语言，mini 框架只有你懂
- "持续演进"是关键信号——自造框架的演进成本会指数增长

**框架选择**（取决于语言栈）：
- Python + 想要 Pydantic 生态 → **Pydantic AI**
- Python + 想要 OpenAI 官方支持 → **OpenAI Agents SDK**
- TypeScript + 前端友好 → **Vercel AI SDK**
- TypeScript + Agent 编排 → **Mastra**

**继续自造的成本估算**（劝退用）：
- 流式：2 天（本章 Demo 1）
- 结构化 + 重试：3 天（本章 Demo 2）
- 工具校验：1 天（本章 Demo 3）
- tracing：5-10 天（第16章内容）
- A/B 框架：3-5 天（第15章内容）
- 多语言 i18n：2-3 天
- 文档 + onboarding：5+ 天
- **总计：20-30 天**，而切框架 + 学习 = 3-5 天

**核心洞察**：自造框架的价值在"学原理"，不在"省钱"。学完原理，就用框架——
你的价值在"懂原理所以能用好框架"，不在"维护一个简陋版 LangChain"。

---

## 总结

这 5 个练习覆盖了本章 3 个特性的深化：
- 练习 1：缓存（性能优化，框架常见特性）
- 练习 2：软校验（鲁棒性提升，生产实践）
- 练习 3：流式 + 结构化组合（真实场景）
- 练习 4：自动修复（防御性编程）
- 练习 5：架构决策（最重要的"软实力"）

做完这些，你已经不只是"会用"高级特性，而是"会设计"高级特性。下一步：第15章
评估与测试——让 Agent 的"好坏"变成可量化的指标。
