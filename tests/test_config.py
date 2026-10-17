"""Settings parsing, defaults, and safety properties (no database needed)."""
import pytest
from pydantic import ValidationError

from src.config import DEMO_GRAPH_PREFIX, Settings, load_settings


def test_defaults_are_safe_and_demo_named():
    s = Settings()
    assert s.graph_name.startswith(DEMO_GRAPH_PREFIX)
    assert s.is_demo_graph
    assert s.habeas_demo_mode is False  # simulated actions need explicit opt-in
    assert s.llm_mode == "fake"
    assert s.falkordb_query_timeout_ms == 10_000  # milliseconds
    assert s.tool_budget == 12


def test_non_demo_graph_name_is_not_demo_safe():
    assert not Settings(graph_name="customer_prod").is_demo_graph


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("HABEAS_DEMO_MODE", "true")
    monkeypatch.setenv("GRAPH_NAME", "HABEAS_demo_local")
    monkeypatch.setenv("TOOL_BUDGET", "9")
    s = load_settings()
    assert s.habeas_demo_mode is True
    assert s.is_demo_graph
    assert s.tool_budget == 9


def test_invalid_llm_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "banana")
    with pytest.raises(ValidationError):
        load_settings()


def test_timeout_must_be_a_bounded_positive_millisecond_value(monkeypatch):
    monkeypatch.setenv("FALKORDB_QUERY_TIMEOUT_MS", "5")
    with pytest.raises(ValidationError):
        load_settings()


def test_non_integer_timeout_is_rejected_with_clear_error(monkeypatch):
    monkeypatch.setenv("FALKORDB_QUERY_TIMEOUT_MS", "10 seconds")
    with pytest.raises(ValueError, match="must be an integer"):
        load_settings()


def test_api_key_never_appears_in_repr_or_str(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    s = load_settings()
    assert "sk-super-secret-value" not in repr(s)
    assert "sk-super-secret-value" not in str(s)
