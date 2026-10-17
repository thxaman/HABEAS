"""Phase 7 provider integration tests: FakeLLM preserved, OpenAI preserved,
Groq added as an OpenAI-compatible provider over the live MCP tool interface.

The live smoke test is skipped unless GROQ_API_KEY/GROQ_MODEL are configured
(via environment or the repo .env); the offline suite never requires
credentials.
"""
from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from src.agent.llm import (
    FakeLLM,
    LLMConfigurationError,
    LLMProviderError,
    RealLLM,
    make_llm,
)
from src.agent.loop import run_investigation
from src.agent.state import (
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_EVIDENCE_SUFFICIENT,
    TERMINATION_NO_DISCRIMINATING_QUERY,
    TERMINATION_UNRECOVERABLE_ERROR,
)
from src.config import Settings, load_settings

GROQ_TEST_KEY = "gsk-test-credential-not-real"
OPENAI_TEST_KEY = "sk-test-credential-not-real"


def _live_settings() -> Settings:
    """Settings as configured locally; the live test uses the real .env key."""
    return load_settings()


def _groq_configured(settings: Settings) -> bool:
    return bool(settings.groq_api_key.get_secret_value()) and bool(settings.groq_model)


def _net(settings: Settings) -> str:
    return settings.graph_name + "_test_net"


@pytest.fixture(scope="module")
def trace_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("traces")


# ---- provider selection and credential handling -----------------------------


def test_fake_mode_remains_the_default_without_credentials(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "fake")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    llm = make_llm(Settings())
    assert isinstance(llm, FakeLLM)
    assert llm.mode == "fake"


def test_provider_is_explicit_and_validated(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert load_settings().llm_provider == "groq"
    monkeypatch.setenv("LLM_PROVIDER", "banana")
    with pytest.raises(ValidationError):
        load_settings()


def test_groq_mode_without_key_raises_clear_error():
    bare = Settings(llm_mode="real", llm_provider="groq", groq_api_key="",
                    groq_model="openai/gpt-oss-120b")
    with pytest.raises(LLMConfigurationError) as excinfo:
        RealLLM(bare)
    message = str(excinfo.value)
    assert "GROQ_API_KEY" in message
    assert "fake" in message
    assert "OPENAI_API_KEY" not in message  # never repurposes the OpenAI credential


def test_groq_mode_without_model_raises_clear_error():
    bad = Settings(llm_mode="real", llm_provider="groq", groq_api_key=GROQ_TEST_KEY,
                   groq_model="")
    with pytest.raises(LLMConfigurationError) as excinfo:
        RealLLM(bad)
    assert "GROQ_MODEL" in str(excinfo.value)


def test_groq_errors_never_leak_the_secret():
    bad = Settings(llm_mode="real", llm_provider="groq", groq_api_key=GROQ_TEST_KEY,
                   groq_model="")
    with pytest.raises(LLMConfigurationError) as excinfo:
        RealLLM(bad)
    assert GROQ_TEST_KEY not in str(excinfo.value)
    assert GROQ_TEST_KEY not in repr(bad)


def test_openai_integration_is_preserved():
    ok = Settings(llm_mode="real", llm_provider="openai",
                  openai_api_key=OPENAI_TEST_KEY, openai_model="gpt-4.1-mini")
    llm = RealLLM(ok)
    assert llm.provider == "openai"
    assert llm.model_name == "gpt-4.1-mini"
    client = llm._chat()
    assert client.model_name == "gpt-4.1-mini"
    # No Groq endpoint is injected for the OpenAI provider.
    assert not getattr(client, "openai_api_base", None)


def test_groq_uses_the_openai_compatible_endpoint_with_explicit_credential():
    settings = Settings(
        llm_mode="real", llm_provider="groq", groq_api_key=GROQ_TEST_KEY,
        groq_model="openai/gpt-oss-120b",
    )
    llm = RealLLM(settings)
    assert llm.provider == "groq"
    assert llm.model_name == "openai/gpt-oss-120b"
    client = llm._chat()
    assert client.model_name == "openai/gpt-oss-120b"
    base = getattr(client, "openai_api_base", None) or getattr(client, "base_url", None)
    assert base == "https://api.groq.com/openai/v1"
    key = getattr(client, "openai_api_key", None)
    if hasattr(key, "get_secret_value"):
        key = key.get_secret_value()
    assert key == GROQ_TEST_KEY


def test_make_llm_provider_override_never_touches_other_credentials():
    settings = Settings(
        llm_mode="real", llm_provider="openai", openai_api_key=OPENAI_TEST_KEY,
        openai_model="gpt-4.1-mini", groq_api_key=GROQ_TEST_KEY,
        groq_model="openai/gpt-oss-120b",
    )
    llm = make_llm(settings, provider="groq")
    assert isinstance(llm, RealLLM)
    assert llm.provider == "groq"
    assert llm.model_name == "openai/gpt-oss-120b"
    with pytest.raises(LLMConfigurationError):
        make_llm(settings, provider="banana")


def test_groq_credential_is_used_over_an_absent_openai_credential(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", GROQ_TEST_KEY)
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    s = load_settings()
    assert s.llm_provider == "groq"
    assert s.groq_model == "openai/gpt-oss-120b"


def test_llm_model_env_is_a_shared_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    s = load_settings()
    assert s.groq_model == "llama-3.3-70b-versatile"


# ---- provider failure is a recorded outcome, not a crash --------------------


def test_real_client_is_rebuilt_per_event_loop():
    """The provider client must not carry a transport across asyncio.run() loops.

    Verified cause (Phase 7 real-smoke): langchain-openai reuses a process-wide
    async client whose httpx pool is bound to the loop of first use, so a
    second asyncio.run() fails every provider call with "Event loop is closed"
    even when a fresh ChatOpenAI is constructed. The fix gives each loop its
    own client AND an explicit httpx.AsyncClient created inside that loop.
    """
    ok = Settings(llm_mode="real", llm_provider="openai",
                  openai_api_key=OPENAI_TEST_KEY, openai_model="gpt-4.1-mini")
    llm = RealLLM(ok)

    async def use():
        client = llm._chat()
        return client, llm._http_client

    first_client, first_http = asyncio.run(use())
    second_client, second_http = asyncio.run(use())
    # a new asyncio.run() => a new loop => rebuilt client and transport
    assert second_client is not first_client
    assert second_http is not first_http
    # the client is handed an explicit in-loop transport (the actual fix)
    assert first_http is not None and second_http is not None
    assert first_client.http_async_client is first_http
    # ...and the client is stable within a single loop
    async def reuse():
        return llm._chat() is llm._chat()

    assert asyncio.run(reuse()) is True


def test_provider_failure_terminates_with_evidence_preserved(
    settings, test_net, trace_dir
):
    class ProviderDown:
        mode = "real"

        async def decide(self, *, state, available_tools, stage):
            raise LLMProviderError("simulated provider outage")

    result = asyncio.run(run_investigation(
        "ALRT-CYCLE", settings=settings, llm=ProviderDown(),
        graph_name=_net(settings), trace_dir=trace_dir, case_id="CASE-PROVFAIL",
    ))
    state = result.state
    # The triage call succeeded before the provider died; its evidence survives.
    assert state.evidence and state.evidence[0].tool == "alert_context"
    assert state.termination_reason == TERMINATION_UNRECOVERABLE_ERROR
    assert any("provider failure" in e.lower() for e in state.errors)
    assert result.verdict == "insufficient_evidence"


# ---- bounded live smoke over the real MCP server (opt-in by credentials) ----


@pytest.mark.skipif(
    not _groq_configured(_live_settings()),
    reason="GROQ_API_KEY/GROQ_MODEL not configured; live smoke test skipped",
)
def test_groq_live_smoke_tool_call_round_trip(settings, test_net, trace_dir):
    """One bounded Groq investigation over the real MCP stdio server.

    Covers: a live model selecting an investigation tool from the MCP-derived
    schemas, the tool result being processed into an evidence record, and the
    loop terminating through its normal bounds. Without credentials the test
    skips; the offline tests above keep working with no key.
    """
    live = _live_settings()
    bounded = Settings(**{**settings.model_dump(), "tool_budget": 4})
    llm = RealLLM(live, timeout_s=90.0)

    result = asyncio.run(run_investigation(
        "ALRT-CYCLE",
        settings=bounded,
        graph_name=_net(bounded),
        llm=llm,
        trace_dir=trace_dir,
        case_id="CASE-GROQ-LIVE",
        max_steps=12,
    ))
    state = result.state

    assert state.tool_calls_used >= 1, (
        "the live model did not select any investigation tool; errors: "
        f"{state.errors}"
    )
    assert not any("provider failure" in e.lower() for e in state.errors)
    assert state.termination_reason in {
        TERMINATION_EVIDENCE_SUFFICIENT,
        TERMINATION_NO_DISCRIMINATING_QUERY,
        TERMINATION_BUDGET_EXHAUSTED,
    }

    ok_records = [r for r in state.evidence if r.ok]
    assert ok_records, "no tool result was processed into an evidence record"
    record = ok_records[0]
    assert isinstance(record.tool, str) and record.tool
    assert record.interpretation.strip(), "tool result interpretation missing"
    assert record.query_ids, "the real MCP envelope carried no query evidence"
    assert all(h.startswith("sha256:") for h in record.rows_hash)
