"""Shared configuration helper for the AI Agent tutorial.

Every chapter imports ``get_config()`` from here — it is the SINGLE source of
truth for which provider / base_url / api_key / model the tutorial uses.

Switching providers is a one-line change in ``.env``::

    PROVIDER=deepseek   # or openai | qwen | ollama

…and every chapter reroutes automatically. All four providers speak the
OpenAI-compatible Chat Completions API, so we only need the ``openai`` SDK.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────
# Load .env from the tutorial root (ai-agent/.env), regardless of cwd.
# This file lives at ai-agent/shared/config.py, so the root is parent.parent.
# ──────────────────────────────────────────────────────────────────────
TUTORIAL_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = TUTORIAL_ROOT / ".env"
load_dotenv(DOTENV_PATH)


@dataclass(frozen=True)
class Config:
    """Resolved provider configuration handed to ``openai.OpenAI(...)``.

    Attributes:
        provider:   The active provider name (``"openai"`` | ``"deepseek"``
                    | ``"qwen"`` | ``"ollama"``).
        base_url:   OpenAI-compatible API base URL for this provider.
        api_key:    The actual secret key read from the environment.
        model:      Default chat model id for this provider.
    """

    provider: str
    base_url: str
    api_key: str
    model: str


# ──────────────────────────────────────────────────────────────────────
# Provider registry.
#
# Each entry says:
#   base_url         — where to send OpenAI-compatible requests
#   api_key_env      — the NAME of the env var that holds the secret
#   default_model    — the model used unless overridden by <PROVIDER>_MODEL
# ──────────────────────────────────────────────────────────────────────
PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "default_model": "qwen2.5:7b",
    },
}

DEFAULT_PROVIDER = "openai"
# Ollama runs locally and needs no real secret — use a stable dummy so the
# OpenAI SDK (which rejects empty keys) is happy.
_OLLAMA_DUMMY_KEY = "ollama"


def get_config(provider: str | None = None) -> Config:
    """Resolve and validate the active provider configuration.

    Args:
        provider: Optional override. When ``None`` (the usual case) the
            provider is read from the ``PROVIDER`` env var, falling back to
            ``"openai"``.

    Returns:
        A populated :class:`Config`.

    Exits the process (``sys.exit(1)``) with a friendly, learner-readable
    message if the requested provider is unknown or its API key is missing.
    We deliberately exit instead of raising so learners see guidance, not a
    raw stack trace — silent empty keys passed to the SDK are the #1 source
    of confusing connection errors for beginners.
    """
    name = (provider or os.getenv("PROVIDER", DEFAULT_PROVIDER)).strip().lower()

    # --- Unknown provider -------------------------------------------------
    if name not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS))
        print(
            f"\n[config] 未知提供商 '{name}'。\n"
            f"          支持的提供商: {available}\n"
            f"          请在 {DOTENV_PATH} 中设置 PROVIDER 为上述之一。",
            file=sys.stderr,
        )
        sys.exit(1)

    spec = PROVIDERS[name]
    base_url = spec["base_url"]
    api_key_env = spec["api_key_env"]
    api_key = os.getenv(api_key_env, "").strip()

    # --- Model: allow per-provider override via <PROVIDER>_MODEL ---------
    override_env = f"{name.upper()}_MODEL"
    model = os.getenv(override_env, "").strip() or spec["default_model"]

    # --- Ollama: fill in a dummy key if none provided --------------------
    if name == "ollama" and not api_key:
        api_key = _OLLAMA_DUMMY_KEY

    # --- Missing/empty API key → friendly error --------------------------
    if not api_key:
        print(
            f"\n[config] 未设置环境变量 '{api_key_env}'。\n"
            f"          当前 PROVIDER={name}，需要该提供商的 API 密钥。\n"
            f"          请编辑 {DOTENV_PATH}，填入你的 {api_key_env}，\n"
            f"          或将 PROVIDER 改为另一个已配置的提供商。\n"
            f"          (本地 Ollama 无需密钥，设 PROVIDER=ollama 即可。)",
            file=sys.stderr,
        )
        sys.exit(1)

    return Config(provider=name, base_url=base_url, api_key=api_key, model=model)


def list_providers() -> Dict[str, Dict[str, str]]:
    """Return a copy of the provider registry (handy for tooling/docs)."""
    return {name: dict(spec) for name, spec in PROVIDERS.items()}


# ──────────────────────────────────────────────────────────────────────
# `python shared/config.py` prints the resolved config — a quick smoke test.
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    print(f"provider = {cfg.provider}")
    print(f"base_url = {cfg.base_url}")
    print(f"model    = {cfg.model}")
    print(f"api_key  = {'***' + cfg.api_key[-4:] if len(cfg.api_key) > 4 else '(set)'}")
