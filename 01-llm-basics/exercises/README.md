# 第01章 · 练习

---

## 练习 1：观察温度对输出的影响

### 任务

修改 `main.py`（或 `main.ts`），用 **5 个不同温度值**（0.0, 0.3, 0.7, 1.0, 1.5）调用同一个问题，打印每次的回答。

### 要求

1. 问题用："用一个比喻解释什么是大语言模型。"
2. 温度用列表 `[0.0, 0.3, 0.7, 1.0, 1.5]` 循环调用
3. 每次打印格式：`[temperature=X] 回答内容`
4. 观察并记录：哪个温度的回答最"安全"？哪个最"有创意"？

### 思考题

- temperature=1.5 超过了 1.0，会发生什么？（提示：大多数 API 会接受，但输出可能变得不连贯）
- 如果你要做一个"事实问答"应用，应该用什么温度？为什么？

---

## 练习 2：用 tokenizer 计算 token 数

### 任务

安装 token 计算库，给定一段文本，计算它的 token 数量。

### Python 版

```bash
pip3 install --user --break-system-packages tiktoken
```

```python
import tiktoken

# 用和模型匹配的编码器
enc = tiktoken.encoding_for_model("gpt-4o-mini")

texts = [
    "Hello world",
    "你好世界",
    "def hello(): print('hi')",
    "我今天有三个会要开，还有一个报告要写，怎么安排优先级？",
]

for text in texts:
    tokens = enc.encode(text)
    print(f"'{text}' → {len(tokens)} tokens")
    print(f"  token IDs: {tokens}")
    print(f"  逐 token 解码: {[enc.decode([t]) for t in tokens]}")
    print()
```

### TypeScript 版

```bash
npm install gpt-tokenizer
```

```typescript
import { encode } from "gpt-tokenizer";

const texts = [
  "Hello world",
  "你好世界",
  "def hello(): print('hi')",
  "我今天有三个会要开，还有一个报告要写，怎么安排优先级？",
];

for (const text of texts) {
  const tokens = encode(text);
  console.log(`'${text}' → ${tokens.length} tokens`);
  console.log(`  token IDs: ${tokens}`);
  console.log();
}
```

### 思考题

- 为什么中文比英文"贵"？（同样内容，中文 token 数更多）
- 如果你的 Agent 每次对话预算 4000 tokens，你能传多少轮对话历史？

---

## 参考答案

### 练习 1 参考代码（Python）

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

question = "用一个比喻解释什么是大语言模型。"

for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": "你是一个擅长用比喻解释概念的助手。"},
            {"role": "user", "content": question},
        ],
        temperature=temp,
    )
    answer = response.choices[0].message.content
    print(f"[temperature={temp}] {answer}")
    print()
```

### 练习 2 参考答案要点

- **英文 token 效率高**：GPT 系列的 tokenizer 训练数据以英文为主，英文词汇在词表中有更多专用 token。
- **中文需要更多 token**：一个汉字可能被拆成 2-3 个 token（UTF-8 字节级别的编码）。
- **4000 token 预算**：大约能传 10-15 轮简短对话（每轮 ~200-300 tokens）。长对话必须用摘要/截断策略（第05章讲）。
