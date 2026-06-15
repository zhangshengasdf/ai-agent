# 第03章 练习题

## 练习 1：新增一个自定义工具

### 任务

给「任务助手」新增一个 `get_stock_price` 工具，查询股票价格（mock 数据）。

### 要求

1. 定义工具的 JSON Schema（name、description、parameters）
2. 实现 `get_stock_price(symbol: str) -> str` 函数，返回 mock 股价
3. 把它注册到 `TOOL_FUNCTIONS` 映射中
4. 测试：让模型回答"苹果公司的股价是多少？"

### 参考答案

**Python**:

```python
# 1. 工具函数实现
def get_stock_price(symbol: str) -> str:
    """查询股票的当前价格（mock 数据）。"""
    mock_stocks = {
        "AAPL": "苹果 (AAPL): $192.35, +1.23%",
        "GOOGL": "谷歌 (GOOGL): $176.88, -0.45%",
        "MSFT": "微软 (MSFT): $454.27, +0.89%",
        "TSLA": "特斯拉 (TSLA): $248.50, +3.12%",
        "BABA": "阿里巴巴 (BABA): $89.62, -1.05%",
    }
    upper = symbol.upper()
    return mock_stocks.get(upper, f"未找到股票代码 '{symbol}' 的行情数据")

# 2. JSON Schema 定义
stock_tool = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "查询股票的当前价格，输入股票代码如 AAPL、GOOGL",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码，如'AAPL'代表苹果公司",
                }
            },
            "required": ["symbol"],
        },
    },
}

# 3. 注册到工具列表和映射
tools.append(stock_tool)
TOOL_FUNCTIONS["get_stock_price"] = get_stock_price

# 4. 测试
run_tool_flow("苹果公司的股价是多少？")
```

**TypeScript**:

```typescript
// 1. 工具函数实现
function getStockPrice(symbol: string): string {
  const mockStocks: Record<string, string> = {
    AAPL: "苹果 (AAPL): $192.35, +1.23%",
    GOOGL: "谷歌 (GOOGL): $176.88, -0.45%",
    MSFT: "微软 (MSFT): $454.27, +0.89%",
    TSLA: "特斯拉 (TSLA): $248.50, +3.12%",
    BABA: "阿里巴巴 (BABA): $89.62, -1.05%",
  };
  return mockStocks[symbol.toUpperCase()] ?? `未找到股票代码 '${symbol}' 的行情数据`;
}

// 2. 注册
tools.push({
  type: "function",
  function: {
    name: "get_stock_price",
    description: "查询股票的当前价格，输入股票代码如 AAPL、GOOGL",
    parameters: {
      type: "object",
      properties: {
        symbol: {
          type: "string",
          description: "股票代码，如'AAPL'代表苹果公司",
        },
      },
      required: ["symbol"],
    },
  },
});
TOOL_FUNCTIONS["get_stock_price"] = (symbol: string) => getStockPrice(symbol);
```

---

## 练习 2：思考题——如果模型返回不存在的工具名怎么办？

### 问题

模型有时候会"幻觉"出一个不存在的工具名（比如 `fetch_url`），而你只注册了 `get_weather`、`calculate`、`search_wiki`。这时候代码会怎样？

### 思考

1. 当前代码中，`TOOL_FUNCTIONS.get(func_name)` 返回 `None`，我们会返回 `"错误：未知工具"` 给模型。这样做对吗？
2. 更好的做法是什么？要不要把错误信息告诉模型，让它重试？
3. 要不要在 Step 1 就过滤掉不存在的 tool_call？

### 参考思路

当前的实现已经是一种合理的防御：

```python
func = TOOL_FUNCTIONS.get(func_name)
if func is None:
    result = f"错误：未知工具 '{func_name}'"
```

把错误信息以 `role="tool"` 反馈给模型，模型有机会在 Step 4 中做出合理回应（比如道歉并用文字回答）。

但这只是**单轮**的处理方式。在第04章（Agent 循环）和第06章（错误处理）中，我们会学习更完善的策略：

- **重试**：让模型重新选择工具
- **回退**：直接用文字回答，不用工具
- **日志记录**：记录异常的 tool_call 用于后续分析

> 预告：第06章「错误处理」会系统讲解这些策略。

---

## 练习 3（进阶）：改进 calculate 工具

### 任务

当前的 `calculate` 工具用 `eval()` 实现，虽然限制了字符集，但在生产环境中仍然不安全。请用更安全的方式重写它。

### 提示

- Python: 可以用 `ast.literal_eval()`（但不支持运算符），或者手动解析表达式
- 或者用第三方库 `simpleeval`（`pip install simpleeval`）
- TypeScript: 可以用 `Function` 构造函数代替 `eval`，或者用 `mathjs` 库

### 参考答案（Python，使用 ast 模块）

```python
import ast
import operator

def calculate_safe(expression: str) -> str:
    """安全的数学计算，使用 ast 解析而非 eval。"""
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        else:
            raise ValueError(f"不支持的表达式: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"计算错误：{e}"
```

---

## 总结

通过这些练习，你应该加深了对工具调用的理解：

1. **新增工具**：JSON Schema 定义 + 函数实现 + 注册映射，三步完成
2. **错误处理**：模型可能返回不存在的工具，需要防御性编程
3. **安全性**：工具函数要限制输入范围，避免注入攻击

下一步：第04章「Agent 循环」——让任务助手能**反复调用工具**直到完成复杂任务。
