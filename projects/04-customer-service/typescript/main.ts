/**
 * 项目4 · 智能客服 Agent（记忆 + 工具 + 人机协作）
 *
 * 综合实战：构建一个带记忆、工具调用、情绪识别和安全防护的智能客服系统。
 *
 * 核心组件：
 *   - ConversationBuffer 记忆：跨轮记住用户信息（姓名、订单号）
 *   - 订单查询工具：读取 data/orders.json，按订单号查询状态
 *   - 情绪识别 + 转人工：检测不满关键词 → 触发 Handoff
 *   - 防 Prompt 注入：检测注入关键词，拒绝越权请求
 *   - 多轮对话：5 轮预设演示（自我介绍→查订单→追问→不满→转人工）
 *   - 离线 Mock：API 不可用 → 本地 fallback，exit 0
 */

import OpenAI from "openai";
import { getConfig } from "../../../shared/config";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const cfg = getConfig();
const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });

// 订单数据路径
const ORDERS_PATH = resolve(__dirname, "..", "data", "orders.json");

// ════════════════════════════════════════════════════════════════════
// 1. ConversationBuffer 记忆系统
// ════════════════════════════════════════════════════════════════════

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface UserInfo {
  name?: string;
  orderId?: string;
}

class ConversationBuffer {
  private messages: Message[] = [];

  add(role: Message["role"], content: string): void {
    this.messages.push({ role, content });
  }

  getMessages(): Message[] {
    return [...this.messages];
  }

  clear(): void {
    this.messages = [];
  }

  extractUserInfo(): UserInfo {
    const info: UserInfo = {};
    for (const msg of this.messages) {
      const text = msg.content;
      // 姓名提取
      for (const keyword of ["我叫", "我是", "我的名字是"]) {
        if (text.includes(keyword)) {
          const after = text.split(keyword)[1]?.trim() ?? "";
          let name = "";
          for (const ch of after) {
            if (/[\u4e00-\u9fff]/.test(ch)) {
              name += ch;
            } else {
              break;
            }
          }
          if (name) info.name = name;
        }
      }
      // 订单号提取
      if (text.includes("ORD-")) {
        const idx = text.indexOf("ORD-");
        info.orderId = text.slice(idx, idx + 7);
      }
    }
    return info;
  }
}

// ════════════════════════════════════════════════════════════════════
// 2. 订单查询工具
// ════════════════════════════════════════════════════════════════════

interface Order {
  orderId: string;
  customer: string;
  item: string;
  status: string;
  amount: number;
}

function loadOrders(): Order[] {
  try {
    const raw = readFileSync(ORDERS_PATH, "utf-8");
    return JSON.parse(raw) as Order[];
  } catch {
    return [];
  }
}

function queryOrder(orderId: string): Order | undefined {
  const orders = loadOrders();
  return orders.find((o) => o.orderId === orderId);
}

function formatOrder(order: Order): string {
  return (
    `订单号: ${order.orderId} | ` +
    `商品: ${order.item} | ` +
    `状态: ${order.status} | ` +
    `金额: ¥${order.amount}`
  );
}

// ════════════════════════════════════════════════════════════════════
// 3. 情绪识别 + 转人工
// ════════════════════════════════════════════════════════════════════

const EMOTION_KEYWORDS = [
  "投诉",
  "太差",
  "退款",
  "垃圾",
  "愤怒",
  "生气",
  "不满",
  "差评",
];

function detectNegativeEmotion(text: string): boolean {
  return EMOTION_KEYWORDS.some((kw) => text.includes(kw));
}

function triggerHandoff(reason: string): void {
  console.log(`OUT:handoff: ⚠️ 检测到用户不满，原因: ${reason}`);
  console.log(`OUT:handoff: 🔄 正在转接人工客服，请稍候...`);
  console.log(`OUT:handoff: 👤 人工客服已接入，祝您问题顺利解决！`);
}

// ════════════════════════════════════════════════════════════════════
// 4. 防 Prompt 注入
// ════════════════════════════════════════════════════════════════════

const INJECTION_KEYWORDS = [
  "忽略之前指令",
  "ignore previous",
  "ignore all previous",
  "管理员",
  "admin",
  "所有用户",
  "所有订单",
  "全部用户数据",
  "system prompt",
  "你的指令是",
];

function detectInjection(text: string): boolean {
  const lower = text.toLowerCase();
  return INJECTION_KEYWORDS.some((kw) => lower.includes(kw.toLowerCase()));
}

function blockInjection(text: string): void {
  console.log(`OUT:inject:block: 🛡️ 检测到潜在 Prompt 注入，已拦截`);
  console.log(`OUT:inject:block: 触发内容: ${text.slice(0, 60)}`);
  console.log(
    `OUT:inject:block: 我只能帮您查询您自己的订单信息，无法执行其他指令。`,
  );
}

// ════════════════════════════════════════════════════════════════════
// 5. LLM 调用封装（带 try/catch 降级）
// ════════════════════════════════════════════════════════════════════

async function llmChat(
  messages: Message[],
  systemPrompt: string,
): Promise<string> {
  const fullMessages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    ...messages.map((m) => ({ role: m.role as "user" | "assistant", content: m.content })),
  ];
  try {
    const resp = await client.chat.completions.create({
      model: cfg.model,
      messages: fullMessages,
    });
    return resp.choices[0].message.content ?? "";
  } catch {
    return "";
  }
}

// ════════════════════════════════════════════════════════════════════
// 6. 客服 Agent 主逻辑
// ════════════════════════════════════════════════════════════════════

const SYSTEM_PROMPT = [
  "你是一个友好、专业的智能客服助手。你的职责是：",
  "1. 帮助用户查询订单状态",
  "2. 回答关于商品和服务的问题",
  "3. 记住用户的姓名和订单号",
  "4. 如果用户情绪不满，建议转人工客服",
  "请用简洁、友好的中文回复。",
].join("\n");

async function processUserInput(
  userInput: string,
  buffer: ConversationBuffer,
): Promise<string> {
  // ── 1. 防注入检查 ──
  if (detectInjection(userInput)) {
    blockInjection(userInput);
    buffer.add("user", userInput);
    const reply =
      "抱歉，我只能帮您查询订单信息，无法执行其他指令。请问有什么订单需要查询吗？";
    buffer.add("assistant", reply);
    return reply;
  }

  // ── 2. 情绪检测 ──
  if (detectNegativeEmotion(userInput)) {
    console.log(`OUT:emotion: 😤 检测到负面情绪关键词`);
    triggerHandoff(userInput);
    buffer.add("user", userInput);
    const reply =
      "非常抱歉给您带来不好的体验，我已为您转接人工客服，请稍候。";
    buffer.add("assistant", reply);
    return reply;
  }

  // ── 3. 提取用户信息并打印 ──
  buffer.add("user", userInput);
  const info = buffer.extractUserInfo();
  if (info.name || info.orderId) {
    console.log(`OUT:memory: 📝 记忆更新: ${JSON.stringify(info)}`);
  }

  // ── 4. 订单查询工具 ──
  let orderStr = "";
  if (info.orderId) {
    console.log(`OUT:tool: 🔧 调用工具: query_order(${info.orderId})`);
    const order = queryOrder(info.orderId);
    if (order) {
      orderStr = formatOrder(order);
      console.log(`OUT:tool: ✅ 查询结果: ${orderStr}`);
    } else {
      orderStr = `未找到订单 ${info.orderId}`;
      console.log(`OUT:tool: ❌ ${orderStr}`);
    }
  }

  // ── 5. 尝试 LLM 回复（离线 fallback） ──
  const messages = buffer.getMessages();
  const llmReply = await llmChat(messages, SYSTEM_PROMPT);

  if (llmReply) {
    buffer.add("assistant", llmReply);
    return llmReply;
  }

  // ── 6. 离线 fallback 回复 ──
  const fallback = generateFallbackReply(userInput, info, orderStr);
  buffer.add("assistant", fallback);
  return fallback;
}

function generateFallbackReply(
  userInput: string,
  info: UserInfo,
  orderStr: string,
): string {
  const name = info.name ?? "";

  if (orderStr) {
    const greeting = name ? `${name}，` : "";
    return `${greeting}查询到您的订单信息：${orderStr}。请问还有什么需要帮助的吗？`;
  }

  if (name) {
    return `${name}，您好！请问有什么可以帮您的？您可以提供订单号来查询订单状态。`;
  }

  const lower = userInput.toLowerCase();
  if (["你好", "您好", "hi", "hello"].some((kw) => lower.includes(kw))) {
    return "您好！欢迎联系智能客服，请问有什么可以帮您的？";
  }

  if (userInput.includes("订单")) {
    return "请提供您的订单号（如 ORD-001），我来帮您查询。";
  }

  return "请问有什么可以帮您的？您可以提供订单号来查询订单状态。";
}

// ════════════════════════════════════════════════════════════════════
// 7. 多轮对话演示（5 轮预设）
// ════════════════════════════════════════════════════════════════════

const DEMO_CONVERSATIONS = [
  "你好，我叫张三",
  "请帮我查一下订单 ORD-001 的状态",
  "这个订单什么时候能到？",
  "等了这么久还没到，太差了！我要退款！",
  "算了，我要投诉你们的服务！",
];

async function runDemo(): Promise<void> {
  console.log("OUT: ══ 智能客服 Agent ══");
  console.log(`OUT: 模型: ${cfg.model}`);
  console.log(`OUT: 提供商: ${cfg.provider}`);
  console.log();

  const buffer = new ConversationBuffer();

  for (let i = 0; i < DEMO_CONVERSATIONS.length; i++) {
    const userMsg = DEMO_CONVERSATIONS[i];
    console.log(`OUT: ── 第 ${i + 1} 轮 ──`);
    console.log(`OUT: 👤 用户: ${userMsg}`);

    const reply = await processUserInput(userMsg, buffer);
    console.log(`OUT: 🤖 客服: ${reply}`);

    // 显示记忆状态
    const msgs = buffer.getMessages();
    console.log(`OUT:memory: 💾 记忆中消息数: ${msgs.length}`);
    const info = buffer.extractUserInfo();
    if (info.name || info.orderId) {
      console.log(`OUT:memory: 📝 已记住: ${JSON.stringify(info)}`);
    }
    console.log();
  }

  // 最终记忆状态
  console.log("OUT: ══ 对话结束 ══");
  console.log(
    `OUT:memory: 💾 最终记忆: ${buffer.getMessages().length} 条消息`,
  );
  const finalInfo = buffer.extractUserInfo();
  if (finalInfo.name || finalInfo.orderId) {
    console.log(
      `OUT:memory: 📝 最终用户信息: ${JSON.stringify(finalInfo)}`,
    );
  }
}

// ════════════════════════════════════════════════════════════════════
// 8. 主函数
// ════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  await runDemo();
}

main();
