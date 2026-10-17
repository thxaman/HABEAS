"""FalkorDB Cloud connection tests (config, TLS, cache, redaction, MCP env).

These pin the cloud-deployment contract:

- ``FALKORDB_USERNAME`` / ``FALKORDB_PASSWORD`` (SecretStr) / ``FALKORDB_TLS``
  are real settings; empty/off reproduces the local Docker default exactly;
- the DB client cache is keyed by ALL connection settings, so two different
  credentials (or TLS on/off) can never share a client;
- credential material never appears in exception messages or repr — failures
  still name their cause;
- the MCP server subprocess receives the RESOLVED connection settings through
  its environment (so Streamlit Community Cloud secrets reach the child), and
  the UI promotes ``st.secrets`` keys into the environment when absent.

DB-free except one smoke test that uses the local graph fixture.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.ui import backend
from src.agent.mcp_client import server_parameters
from src.db import classify_failure, clear_client_cache, get_client
from src import db as db_module

CLOUD = Settings(
    falkordb_host="demo.falke.app",
    falkordb_port=6379,
    falkordb_username="default",
    falkordb_password=SecretStr("pw-4test"),
    falkordb_tls=True,
    graph_name="HABEAS_demo_showcase_s42",
)


@pytest.fixture(autouse=True)
def _clean_client_cache():
    clear_client_cache()
    yield
    clear_client_cache()


@pytest.fixture()
def fake_falkordb(monkeypatch):
    calls: list[dict] = []

    def factory(**kwargs):
        calls.append(kwargs)
        return object()  # a fresh client per construction

    monkeypatch.setattr(db_module, "FalkorDB", factory)
    return calls


# ------------------------------------------------------------- settings


def test_defaults_are_the_local_docker_connection():
    s = Settings()
    assert s.falkordb_host == "localhost"
    assert s.falkordb_port == 6379
    assert s.falkordb_username == ""
    assert s.falkordb_password.get_secret_value() == ""
    assert s.falkordb_tls is False


def test_load_settings_reads_the_cloud_variables(monkeypatch):
    monkeypatch.setenv("FALKORDB_HOST", "demo.falke.app")
    monkeypatch.setenv("FALKORDB_USERNAME", "default")
    monkeypatch.setenv("FALKORDB_PASSWORD", "pw-4test")
    monkeypatch.setenv("FALKORDB_TLS", "true")
    from src.config import load_settings

    s = load_settings()
    assert s.falkordb_host == "demo.falke.app"
    assert s.falkordb_username == "default"
    assert s.falkordb_password.get_secret_value() == "pw-4test"
    assert s.falkordb_tls is True


def test_password_is_a_secret_in_every_textual_form():
    s = CLOUD
    assert "pw-4test" not in repr(s)
    assert "pw-4test" not in str(s)
    assert "**********" in repr(s.falkordb_password)


# ---------------------------------------------------------- client + cache


def test_local_default_connects_without_credentials_or_tls(fake_falkordb):
    get_client(Settings())
    assert len(fake_falkordb) == 1
    kwargs = fake_falkordb[0]
    assert kwargs["host"] == "localhost" and kwargs["port"] == 6379
    assert kwargs["username"] is None and kwargs["password"] is None
    assert kwargs["ssl"] is False


def test_cloud_settings_pass_username_password_and_tls(fake_falkordb):
    get_client(CLOUD)
    kwargs = fake_falkordb[0]
    assert kwargs["host"] == "demo.falke.app"
    assert kwargs["username"] == "default"
    assert kwargs["password"] == "pw-4test"
    assert kwargs["ssl"] is True


def test_client_cache_is_keyed_by_every_connection_setting(fake_falkordb):
    a = get_client(CLOUD)
    same = get_client(
        Settings(
            falkordb_host="demo.falke.app",
            falkordb_username="default",
            falkordb_password=SecretStr("pw-4test"),
            falkordb_tls=True,
        )
    )
    assert same is a and len(fake_falkordb) == 1  # identical config: reused

    only_password = get_client(
        Settings(
            falkordb_host="demo.falke.app",
            falkordb_username="default",
            falkordb_password=SecretStr("different"),
            falkordb_tls=True,
        )
    )
    assert only_password is not a

    only_tls = get_client(
        Settings(
            falkordb_host="demo.falke.app",
            falkordb_username="default",
            falkordb_password=SecretStr("pw-4test"),
            falkordb_tls=False,
        )
    )
    assert only_tls is not a

    only_user = get_client(
        Settings(
            falkordb_host="demo.falke.app",
            falkordb_username="other",
            falkordb_password=SecretStr("pw-4test"),
            falkordb_tls=True,
        )
    )
    assert only_user is not a
    assert len(fake_falkordb) == 4  # one construction per distinct config


# ------------------------------------------------------- secret redaction


def test_classified_failure_never_carries_the_password(monkeypatch):
    monkeypatch.setattr(db_module, "get_settings", lambda: CLOUD)
    qe = classify_failure(RuntimeError("WRONGPASS for pw-4test on socket"))
    assert "pw-4test" not in str(qe)
    assert "WRONGPASS" in str(qe)  # the CAUSE survives, the secret does not
    assert qe.kind in ("connection", "syntax", "unknown")


def test_constructor_failure_is_scrubbed_before_propagating(monkeypatch):
    def exploding(**_kwargs):
        raise RuntimeError("could not connect to demo.falke.app with pw-4test")

    monkeypatch.setattr(db_module, "FalkorDB", exploding)
    with pytest.raises(RuntimeError) as excinfo:
        get_client(CLOUD)
    message = str(excinfo.value)
    assert "pw-4test" not in message
    assert "demo.falke.app" in message  # the cause stays legible


def test_redact_secrets_handles_empty_and_unset_passwords():
    assert classify_failure(RuntimeError("plain failure")).args == (
        "plain failure",)
    assert db_module.redact_secrets("nothing to hide", Settings()) == (
        "nothing to hide")


# ------------------------------------------------------- MCP child process


def test_mcp_subprocess_inherits_resolved_cloud_connection():
    params = server_parameters(
        CLOUD, graph_name="HABEAS_demo_showcase_s42"
    )
    env = params.env
    assert env["FALKORDB_HOST"] == "demo.falke.app"
    assert env["FALKORDB_USERNAME"] == "default"
    assert env["FALKORDB_PASSWORD"] == "pw-4test"  # transport only, never logged
    assert env["FALKORDB_TLS"] == "true"
    assert env["GRAPH_NAME"] == "HABEAS_demo_showcase_s42"


def test_mcp_subprocess_local_default_omits_the_password():
    env = server_parameters(Settings()).env
    assert "FALKORDB_PASSWORD" not in env
    assert env["FALKORDB_TLS"] == "false"
    assert env["FALKORDB_USERNAME"] == ""


# --------------------------------------------- Streamlit secrets promotion


@pytest.fixture()
def _saved_env():
    keys = backend._STREAMLIT_SECRET_KEYS
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield saved
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _install_streamlit_stub(monkeypatch, secrets: dict) -> None:
    stub = types.ModuleType("streamlit")
    stub.secrets = secrets  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit", stub)


def test_streamlit_secrets_are_promoted_into_the_environment(
    monkeypatch, _saved_env
):
    _install_streamlit_stub(monkeypatch, {
        "FALKORDB_HOST": "demo.falke.app",
        "FALKORDB_PORT": "6379",
        "FALKORDB_USERNAME": "default",
        "FALKORDB_PASSWORD": "pw-4test",
        "FALKORDB_TLS": "true",
        "GRAPH_NAME": "",  # empty secrets are skipped
    })
    promoted = backend.load_streamlit_secrets_into_env()
    assert set(promoted) == {
        "FALKORDB_HOST", "FALKORDB_PORT", "FALKORDB_USERNAME",
        "FALKORDB_PASSWORD", "FALKORDB_TLS",
    }
    assert os.environ["FALKORDB_HOST"] == "demo.falke.app"
    assert os.environ["FALKORDB_TLS"] == "true"
    assert "GRAPH_NAME" not in os.environ
    # idempotent: a second call promotes nothing new
    assert backend.load_streamlit_secrets_into_env() == []


def test_environment_variables_win_over_streamlit_secrets(
    monkeypatch, _saved_env
):
    monkeypatch.setenv("FALKORDB_HOST", "env-wins.example")
    _install_streamlit_stub(monkeypatch, {
        "FALKORDB_HOST": "demo.falke.app",
        "FALKORDB_PASSWORD": "pw-4test",
    })
    promoted = backend.load_streamlit_secrets_into_env()
    assert "FALKORDB_HOST" not in promoted
    assert os.environ["FALKORDB_HOST"] == "env-wins.example"
    assert os.environ["FALKORDB_PASSWORD"] == "pw-4test"


def test_streamlit_secrets_bridge_is_a_noop_without_streamlit(monkeypatch,
                                                              _saved_env):
    monkeypatch.setitem(sys.modules, "streamlit", None)  # import fails
    assert backend.load_streamlit_secrets_into_env() == []


# --------------------------------------------------- resolved demo graph


def test_ui_demo_graph_name_is_unchanged():
    """The UI's disposable demo graph stays HABEAS_demo_showcase_s42."""
    settings = backend.ui_settings()
    assert settings.graph_name == "HABEAS_demo_showcase_s42"
    assert settings.is_sandbox is True


@pytest.mark.usefixtures("graph")
def test_local_client_still_connects(settings):
    """Regression: the default Docker path is untouched by the cloud work."""
    graph = db_module.get_graph(settings)
    columns, rows = db_module.normalize_rows(
        db_module.run_query(graph, "RETURN 1 AS ok", read_only=True)
    )
    assert rows == [[1]]
