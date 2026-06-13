# `shared/` — 配置中枢

这个目录是**整个教程的配置中枢**。所有章节都通过这里的 `config` 助手初始化客户端，
**绝不**在自己的代码里硬编码 API 密钥或 base_url。

## 文件

| 文件 | 作用 |
|------|------|
| `config.py` | Python 配置助手：`from shared.config import get_config` |
| `config.ts` | TypeScript 配置助手：`import { getConfig } from "../shared/config"` |

## 一行切换提供商

在 `ai-agent/.env` 里改这一行，全教程所有章节自动改道：

```dotenv
PROVIDER=openai     # openai | deepseek | qwen | ollama
```

四个提供商都兼容 OpenAI Chat Completions API，所以只需 `openai` SDK，差别只在 `base_url` 和 `model`。
`get_config()` / `getConfig()` 会把这些差异封装好交给你。

## 章节作者必读（写给后续 22 章）

### ✅ 永远这样初始化客户端

**Python**（章节位于 `ai-agent/XX-xxx/python/main.py`）：

```python
import sys
from pathlib import Path

# 让章节代码能 import shared.config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI
from shared.config import get_config

cfg = get_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

resp = client.chat.completions.create(
    model=cfg.model,
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

**TypeScript**（章节位于 `ai-agent/XX-xxx/typescript/main.ts`）：

```typescript
import OpenAI from "openai";
import { getConfig } from "../../shared/config";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

const resp = await client.chat.completions.create({
  model: cfg.model,
  messages: [{ role: "user", content: "你好" }],
});
console.log(resp.choices[0].message.content);
```

### ❌ 绝对不要这样

```python
# ❌ 硬编码密钥 —— 会被 git 提交，造成安全事故
client = OpenAI(api_key="sk-xxxxx")

# ❌ 硬编码 base_url —— 切换提供商时全部失效
client = OpenAI(base_url="https://api.openai.com/v1", api_key=os.getenv("X"))

# ❌ 自己读环境变量 —— 绕过了统一校验，密钥缺失时报晦涩的连接错误
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

## 提供商配置表（权威来源）

| provider | base_url | 密钥环境变量 | 默认模型 |
|----------|----------|--------------|----------|
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `qwen-plus` |
| `ollama` | `http://localhost:11434/v1` | `OLLAMA_API_KEY`（本地无需真密钥） | `qwen2.5:7b` |

> 想覆盖某个提供商的默认模型？在 `.env` 里设 `<PROVIDER>_MODEL`，例如 `OPENAI_MODEL=gpt-4o`。

## 设计原则

1. **密钥缺失 → 友好报错**：`get_config()` 在密钥为空时打印中文指引并退出（Python）或抛清晰错误（TS），
   绝不把空密钥传给 SDK——那是新手最常见的"看不懂的连接错误"来源。
2. **路径无关**：助手从 `shared/config.*` 的位置反推 `ai-agent/.env`，
   所以无论你从哪个章节目录运行代码，都能读到同一份配置。
3. **最小依赖**：只用 `openai` SDK + `python-dotenv`/`dotenv`，不引入任何 Agent 框架。
   原理阶段保持透明，这是"先原理后工具"教学哲学的体现。

## 自检

```bash
# Python
python ai-agent/shared/config.py

# TypeScript
npx tsx ai-agent/shared/config.ts
```

两者都应打印当前的 `provider / base_url / model`（密钥会脱敏显示）。
