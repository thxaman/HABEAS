"""HABEAS configuration: environment-backed immutable settings.

Values come from environment variables, optionally loaded from the repo-root
.env file. Secrets are held in SecretStr so they never leak into repr/logs.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

# Hard-coded demo-safe graph-name prefix (PLAN.md §6). Destructive reset is
# only ever allowed for graphs whose name starts with this prefix.
DEMO_GRAPH_PREFIX = "HABEAS_demo"

MIN_QUERY_TIMEOUT_MS = 100
MAX_QUERY_TIMEOUT_MS = 60_000

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseModel):
    """Immutable runtime settings. Default is the safe, offline demo config."""

    model_config = {"frozen": True}

    falkordb_host: str = "localhost"
    falkordb_port: int = Field(default=6379, ge=1, le=65535)
    graph_name: str = "HABEAS_demo_local"
    # Default False: simulated actions require explicit opt-in (PLAN.md §9).
    habeas_demo_mode: bool = False
    # falkordb-py timeout is expressed in milliseconds (verified in Phase 0).
    falkordb_query_timeout_ms: int = Field(
        default=10_000, ge=MIN_QUERY_TIMEOUT_MS, le=MAX_QUERY_TIMEOUT_MS
    )
    llm_mode: Literal["fake", "real"] = "fake"
    # Real-LLM provider. Groq exposes an OpenAI-compatible chat API, so the
    # same langchain-openai client is used with Groq's endpoint and an
    # explicit GROQ_API_KEY; OPENAI_API_KEY is never repurposed for Groq.
    llm_provider: Literal["openai", "groq"] = "openai"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = ""
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = ""
    # Documented Groq OpenAI-compatible endpoint (override only for proxies).
    groq_base_url: str = "https://api.groq.com/openai/v1"
    tool_budget: int = Field(default=12, ge=1, le=100)
    # Phase 5: write-capable action tools are opt-in. Even when enabled they
    # only ever operate on a demo-safe graph (see is_sandbox).
    habeas_enable_actions: bool = False

    @property
    def is_demo_graph(self) -> bool:
        return self.graph_name.startswith(DEMO_GRAPH_PREFIX)

    @property
    def is_sandbox(self) -> bool:
        """True only for the explicitly configured disposable demo graph."""
        return self.habeas_demo_mode and self.is_demo_graph


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {raw!r}"
        ) from exc


def load_settings() -> Settings:
    """Build Settings from the environment. Existing env vars win over .env."""
    load_dotenv(_ENV_PATH)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    # LLM_MODEL is a provider-neutral model name; the provider-specific
    # variables win when both are set. Never used as a credential.
    generic_model = os.getenv("LLM_MODEL", "")
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        # Deterministic inference when LLM_PROVIDER is unset: follow the only
        # configured credential, otherwise keep the historical OpenAI default.
        provider = "groq" if (groq_key and not openai_key) else "openai"
    return Settings(
        falkordb_host=os.getenv("FALKORDB_HOST", "localhost"),
        falkordb_port=_env_int("FALKORDB_PORT", 6379),
        graph_name=os.getenv("GRAPH_NAME", "HABEAS_demo_local"),
        habeas_demo_mode=_env_bool("HABEAS_DEMO_MODE", False),
        falkordb_query_timeout_ms=_env_int("FALKORDB_QUERY_TIMEOUT_MS", 10_000),
        llm_mode=os.getenv("LLM_MODE", "fake"),
        llm_provider=provider,
        openai_api_key=openai_key,
        openai_model=os.getenv("OPENAI_MODEL") or generic_model,
        groq_api_key=groq_key,
        groq_model=os.getenv("GROQ_MODEL") or generic_model,
        groq_base_url=os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
        tool_budget=_env_int("TOOL_BUDGET", 12),
        habeas_enable_actions=_env_bool("HABEAS_ENABLE_ACTIONS", False),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Tests may call get_settings.cache_clear()."""
    return load_settings()
