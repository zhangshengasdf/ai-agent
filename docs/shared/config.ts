/**
 * Shared configuration helper for the AI Agent tutorial.
 *
 * Every chapter imports `getConfig()` from here — it is the SINGLE source of
 * truth for which provider / baseURL / apiKey / model the tutorial uses.
 *
 * Switching providers is a one-line change in `.env`:
 *
 *     PROVIDER=deepseek   // or openai | qwen | ollama
 *
 * …and every chapter reroutes automatically. All four providers speak the
 * OpenAI-compatible Chat Completions API, so we only need the `openai` SDK.
 *
 * Usage from any chapter:
 *
 *     import { getConfig } from "../shared/config";
 *     import OpenAI from "openai";
 *
 *     const cfg = getConfig();
 *     const client = new OpenAI({ baseURL: cfg.baseUrl, apiKey: cfg.apiKey });
 *     // ... use cfg.model for chat completions
 */

import { config as loadEnv } from "dotenv";
import { resolve } from "node:path";

// ──────────────────────────────────────────────────────────────────────
// Load .env from the tutorial root (ai-agent/.env), regardless of cwd.
// This file lives at ai-agent/shared/config.ts, so the root is parent dir.
// ──────────────────────────────────────────────────────────────────────
loadEnv({ path: resolve(__dirname, "..", ".env") });

/** Resolved provider configuration handed to `new OpenAI(...)`. */
export interface Config {
  /** Active provider name ("openai" | "deepseek" | "qwen" | "ollama"). */
  provider: string;
  /** OpenAI-compatible API base URL for this provider. */
  baseUrl: string;
  /** The actual secret key read from the environment. */
  apiKey: string;
  /** Default chat model id for this provider. */
  model: string;
}

/** Static description of a provider's connection details. */
interface ProviderSpec {
  readonly baseUrl: string;
  readonly apiKeyEnv: string;
  readonly defaultModel: string;
}

// ──────────────────────────────────────────────────────────────────────
// Provider registry.
//
// Each entry says:
//   baseUrl        — where to send OpenAI-compatible requests
//   apiKeyEnv      — the NAME of the env var that holds the secret
//   defaultModel   — the model used unless overridden by <PROVIDER>_MODEL
// ──────────────────────────────────────────────────────────────────────
export const PROVIDERS: Readonly<Record<string, ProviderSpec>> = {
  openai: {
    baseUrl: "https://api.openai.com/v1",
    apiKeyEnv: "OPENAI_API_KEY",
    defaultModel: "gpt-4o-mini",
  },
  deepseek: {
    baseUrl: "https://api.deepseek.com/v1",
    apiKeyEnv: "DEEPSEEK_API_KEY",
    defaultModel: "deepseek-chat",
  },
  qwen: {
    baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    apiKeyEnv: "DASHSCOPE_API_KEY",
    defaultModel: "qwen-plus",
  },
  ollama: {
    baseUrl: "http://localhost:11434/v1",
    apiKeyEnv: "OLLAMA_API_KEY",
    defaultModel: "qwen2.5:7b",
  },
};

const DEFAULT_PROVIDER = "openai";
// Ollama runs locally and needs no real secret — use a stable dummy so the
// OpenAI SDK (which rejects empty keys) is happy.
const OLLAMA_DUMMY_KEY = "ollama";

/**
 * Resolve and validate the active provider configuration.
 *
 * @param provider Optional override. When omitted (the usual case) the
 *   provider is read from the `PROVIDER` env var, falling back to `"openai"`.
 * @returns A populated {@link Config}.
 * @throws A learner-readable `Error` if the provider is unknown or its API
 *   key is missing/empty. We throw (rather than pass an empty key to the SDK)
 *   because silent empty keys are the #1 source of confusing connection
 *   errors for beginners. Catch it and show `.message` to the user.
 */
export function getConfig(provider?: string): Config {
  const name = (provider ?? process.env.PROVIDER ?? DEFAULT_PROVIDER)
    .trim()
    .toLowerCase();

  // --- Unknown provider -------------------------------------------------
  const spec = PROVIDERS[name];
  if (!spec) {
    const available = Object.keys(PROVIDERS).sort().join(", ");
    throw new Error(
      `未知提供商 '${name}'。支持的提供商: ${available}。` +
        `请在 .env 中设置 PROVIDER 为上述之一。`,
    );
  }

  const baseUrl = spec.baseUrl;
  const apiKeyEnv = spec.apiKeyEnv;
  let apiKey = (process.env[apiKeyEnv] ?? "").trim();

  // --- Model: allow per-provider override via <PROVIDER>_MODEL ---------
  const overrideEnv = `${name.toUpperCase()}_MODEL`;
  const model =
    (process.env[overrideEnv] ?? "").trim() || spec.defaultModel;

  // --- Ollama: fill in a dummy key if none provided --------------------
  if (name === "ollama" && !apiKey) {
    apiKey = OLLAMA_DUMMY_KEY;
  }

  // --- Missing/empty API key → friendly error --------------------------
  if (!apiKey) {
    throw new Error(
      `未设置环境变量 '${apiKeyEnv}'。当前 PROVIDER=${name}，` +
        `需要该提供商的 API 密钥。请编辑 .env，填入你的 ${apiKeyEnv}，` +
        `或将 PROVIDER 改为另一个已配置的提供商。` +
        `(本地 Ollama 无需密钥，设 PROVIDER=ollama 即可。)`,
    );
  }

  return { provider: name, baseUrl, apiKey, model };
}

/** Return a shallow copy of the provider registry (handy for tooling/docs). */
export function listProviders(): Record<string, ProviderSpec> {
  return { ...PROVIDERS };
}

// ──────────────────────────────────────────────────────────────────────
// `npx tsx shared/config.ts` prints the resolved config — a quick smoke test.
// ──────────────────────────────────────────────────────────────────────
if (require.main === module) {
  try {
    const cfg = getConfig();
    console.log(`provider = ${cfg.provider}`);
    console.log(`base_url = ${cfg.baseUrl}`);
    console.log(`model    = ${cfg.model}`);
    console.log(
      `api_key  = ${cfg.apiKey.length > 4 ? "***" + cfg.apiKey.slice(-4) : "(set)"}`,
    );
  } catch (err) {
    console.error((err as Error).message);
    process.exit(1);
  }
}
