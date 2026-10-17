"""Phase 3 integration tests: the real MCP server over real stdio.

These tests launch ``python -m src.tools.server`` as a subprocess, speak the
MCP protocol through ``mcp.client.stdio``, discover tools, and call them. They
assert on the returned evidence envelopes, argument validation, read-only
behavior, ground-truth isolation, stdout purity, clear startup failures, and
child-process cleanup.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from src.agent.mcp_client import (
    MCPClientError,
    call_tool_sync,
    list_tools_sync,
    open_session,
    server_parameters,
)
from src.db import run_query
from src.tools import query_catalog as qc

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_SOURCE = REPO_ROOT / "src" / "tools" / "server.py"
CLIENT_SOURCE = REPO_ROOT / "src" / "agent" / "mcp_client.py"

EXPECTED_TOOLS = {
    "alert_context",
    "entity_profile", "trace_funds", "detect_cycle", "shared_identifiers",
    "fan_in_out", "hub_scores", "prior_cases", "active_path_exists",
    # Phase 8.5: the analyst UI loads its inbox and renders the graph from
    # these two bounded, read-only tools rather than a hard-coded list.
    "alert_inbox", "alert_neighborhood",
}
CORE_THREE = ["entity_profile", "trace_funds", "detect_cycle"]

# Valid calls for every advertised tool, against the controlled test graph.
VALID_CALLS: dict[str, dict[str, Any]] = {
    "alert_context": {"alert_id": "ALRT-CYCLE"},
    "entity_profile": {"entity_id": "TNI1", "window_days": 365},
    "trace_funds": {"account_id": "TNS1", "direction": "out", "hops": 2},
    "detect_cycle": {"account_id": "TNC1", "max_len": 4},
    "shared_identifiers": {"customer_a_id": "TCU_S1", "customer_b_id": "TCU_S2"},
    "fan_in_out": {"account_id": "TNI1", "window_days": 365},
    "hub_scores": {"window_days": 365, "limit": 10},
    "prior_cases": {"entity_id": "TNI1"},
    "active_path_exists": {
        "source_account_id": "TNS1", "target_account_id": "TNHUB", "max_hops": 3,
    },
    "alert_inbox": {"limit": 50},
    "alert_neighborhood": {"account_id": "TNC1", "hops": 2},
}


def _net_name(settings) -> str:
    return settings.graph_name + "_test_net"


async def _list_with_params(params):
    async with open_session(params=params) as session:
        return await session.list_tools()


# ---- single protocol session shared by the module --------------------------


async def _call_all(settings, graph_name: str) -> dict[str, Any]:
    """Call every tool once inside ONE protocol session (also proves reuse)."""
    results: dict[str, Any] = {}
    async with open_session(settings, graph_name=graph_name) as session:
        listed = await session.list_tools()
        results["_names"] = sorted(t.name for t in listed.tools)
        results["_schemas"] = {t.name: t.inputSchema for t in listed.tools}
        for name, args in VALID_CALLS.items():
            raw = await session.call_tool(name, args)
            text = "".join(
                p.text for p in raw.content if getattr(p, "type", None) == "text"
            )
            results[name] = {"isError": bool(raw.isError), "text": text}
    return results


@pytest.fixture(scope="module")
def mcp_calls(settings, test_net):
    """All tool results collected over a single real MCP stdio session."""
    return asyncio.run(_call_all(settings, _net_name(settings)))


def _envelope(call: dict[str, Any]) -> dict[str, Any]:
    return json.loads(call["text"])


# ---- discovery -------------------------------------------------------------


def test_server_advertises_exactly_the_phase2_read_tools(mcp_calls):
    assert set(mcp_calls["_names"]) == EXPECTED_TOOLS, mcp_calls["_names"]


def test_no_write_cypher_or_mutation_tool_is_exposed(mcp_calls):
    forbidden_fragments = (
        "cypher", "query", "write", "delete", "drop", "merge", "create",
        "freeze", "apply", "mutate",
    )
    for name in mcp_calls["_names"]:
        lowered = name.lower()
        assert not any(f in lowered for f in forbidden_fragments), name


def test_tool_schemas_declare_typed_parameters(mcp_calls):
    schemas = mcp_calls["_schemas"]
    assert schemas["entity_profile"]["properties"]["entity_id"]["type"] == "string"
    assert schemas["entity_profile"]["required"] == ["entity_id"]
    assert schemas["trace_funds"]["properties"]["hops"]["type"] == "integer"
    assert schemas["trace_funds"]["properties"]["min_amount"]["type"] == "number"
    assert schemas["detect_cycle"]["properties"]["max_len"]["type"] == "integer"
    for name, schema in schemas.items():
        assert schema.get("type") == "object", name


# ---- calls and evidence preservation ---------------------------------------


def test_core_three_tools_return_ok_envelopes(mcp_calls):
    for name in CORE_THREE:
        call = mcp_calls[name]
        assert call["isError"] is False, name
        env = _envelope(call)
        assert env["ok"] is True, name
        assert env["tool"] == name
        assert env["interpretation"].strip(), name
        assert env["query_evidence"], name


def test_all_eight_tools_succeed_over_the_protocol(mcp_calls):
    for name in EXPECTED_TOOLS:
        env = _envelope(mcp_calls[name])
        assert env["ok"] is True, f"{name}: {env.get('error')}"


def test_evidence_envelope_fields_survive_the_mcp_boundary(mcp_calls):
    required = {
        "query_id", "cypher", "params", "interpolated", "columns", "rows",
        "rows_hash", "truncated", "row_count", "duration_ms",
    }
    for name in EXPECTED_TOOLS:
        env = _envelope(mcp_calls[name])
        for ev in env["query_evidence"]:
            assert required <= set(ev), f"{name} missing {required - set(ev)}"
            assert ev["query_id"] in qc._QUERY_TEMPLATES, f"{name}: unknown query id"
            assert ev["rows_hash"].startswith("sha256:")
            assert isinstance(ev["rows"], list)
            for token in qc._INTERPOLATION_TOKENS:
                assert token not in ev["cypher"], f"{name}: unbound interpolation"


def test_mcp_results_match_in_process_results_exactly(mcp_calls, test_net):
    """No evidence may be discarded or altered by the MCP layer."""
    from src.tools.reads import entity_profile, trace_funds

    def hashes(envelope: dict[str, Any]) -> dict[str, str]:
        return {ev["query_id"]: ev["rows_hash"] for ev in envelope["query_evidence"]}

    in_process = {
        "entity_profile": entity_profile(test_net, "TNI1", window_days=365).model_dump(),
        "trace_funds": trace_funds(test_net, "TNS1", "out", hops=2).model_dump(),
    }
    for name, env in in_process.items():
        assert hashes(env) == hashes(_envelope(mcp_calls[name])), name


def test_interpretations_are_identical_across_the_boundary(mcp_calls, test_net):
    from src.tools.reads import fan_in_out

    local = fan_in_out(test_net, "TNI1", window_days=365).model_dump()
    assert local["interpretation"] == _envelope(mcp_calls["fan_in_out"])["interpretation"]


# ---- validation before querying --------------------------------------------


@pytest.mark.parametrize(
    "tool,args",
    [
        ("trace_funds", {"account_id": "TNS1", "hops": 9}),
        ("trace_funds", {"account_id": "TNS1", "hops": 0}),
        ("trace_funds", {"account_id": "TNS1", "direction": "sideways"}),
        ("trace_funds", {"account_id": "bad id!", "hops": 1}),
        ("detect_cycle", {"account_id": "TNC1", "max_len": 6}),
        ("detect_cycle", {"account_id": "TNC1", "max_len": 1}),
        ("entity_profile", {"entity_id": "TNI1", "window_days": 0}),
        ("entity_profile", {"entity_id": "TNI1", "limit": 10_000}),
        ("fan_in_out", {"account_id": "TNI1", "min_senders": 0}),
        ("hub_scores", {"limit": 99}),
        ("alert_context", {"alert_id": "bad id!"}),
        ("alert_context", {"alert_id": ""}),
        (
            "active_path_exists",
            {"source_account_id": "TNS1", "target_account_id": "TNHUB", "max_hops": 7},
        ),
    ],
)
def test_invalid_arguments_are_rejected_before_any_query(settings, test_net, tool, args):
    result = call_tool_sync(tool, args, settings=settings, graph_name=_net_name(settings))
    assert result.ok is False, f"{tool} accepted invalid args {args}"
    assert result.envelope is not None
    assert result.envelope["error"]["kind"] == "validation"
    # nothing reached the graph: a rejected call must carry no query evidence
    assert result.query_evidence == []


# ---- read-only behavior ----------------------------------------------------


def _counts(graph) -> tuple[int, int]:
    row = run_query(
        graph,
        "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() "
        "RETURN nodes AS nodes, count(r) AS rels",
        read_only=True,
    ).result_set[0]
    return int(row[0]), int(row[1])


def test_mcp_tool_calls_leave_graph_counts_unchanged(settings, test_net):
    before = _counts(test_net)
    for name, args in VALID_CALLS.items():
        result = call_tool_sync(
            name, args, settings=settings, graph_name=_net_name(settings)
        )
        assert result.ok is True, name
    assert _counts(test_net) == before, "an MCP tool mutated the test graph"


# ---- ground-truth isolation ------------------------------------------------


def test_no_ground_truth_reaches_the_client(mcp_calls):
    forbidden = (
        "ground_truth", "is_fraud", "pattern_type", "expected_account_members",
        "tempting_false_positive", "planted",
    )
    for name in EXPECTED_TOOLS:
        lowered = mcp_calls[name]["text"].lower()
        for marker in forbidden:
            assert marker not in lowered, f"{name} leaked {marker!r} through MCP"
    for name, schema in mcp_calls["_schemas"].items():
        blob = json.dumps(schema).lower()
        for marker in forbidden:
            assert marker not in blob, f"{name} schema mentions {marker!r}"


def test_server_exposes_no_cypher_argument_and_no_manifest_reference(mcp_calls):
    for name, schema in mcp_calls["_schemas"].items():
        props = set((schema.get("properties") or {}).keys())
        assert "cypher" not in props and "query" not in props, name
    source = SERVER_SOURCE.read_text(encoding="utf-8")
    assert "ground_truth" not in source


# ---- stdout purity ---------------------------------------------------------


def test_server_and_client_modules_never_write_to_stdout():
    for path in (SERVER_SOURCE, CLIENT_SOURCE):
        text = path.read_text(encoding="utf-8")
        assert "sys.stdout" not in text, f"{path.name} touches sys.stdout"
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("print("), f"{path.name} prints: {stripped}"


def test_a_full_session_succeeds_which_stray_stdout_would_corrupt(settings, test_net):
    """A round trip only succeeds if stdout carried nothing but JSON-RPC."""
    tools = list_tools_sync(settings, graph_name=_net_name(settings))
    assert len(tools) == len(EXPECTED_TOOLS)


# ---- startup failure reporting and process cleanup -------------------------


def _server_process_count() -> int | None:
    """Count live ``src.tools.server`` child processes, or None if unavailable."""
    if os.name != "nt":
        try:
            out = subprocess.run(
                ["pgrep", "-f", "src.tools.server"], capture_output=True, text=True
            )
            return len([ln for ln in out.stdout.split() if ln.strip()])
        except Exception:
            return None
    try:
        out = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -like '*src.tools.server*' } | "
                "Measure-Object | Select-Object -ExpandProperty Count",
            ],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip() or "0")
    except Exception:
        return None


def test_startup_failure_is_reported_clearly(settings):
    broken = server_parameters(settings, args=["-m", "src.tools.does_not_exist"])
    with pytest.raises(MCPClientError) as excinfo:
        asyncio.run(_list_with_params(broken))
    message = str(excinfo.value)
    assert "HABEAS MCP server" in message
    assert "src.tools.does_not_exist" in message


def test_child_that_exits_immediately_is_reported(settings):
    params = server_parameters(
        settings,
        command=sys.executable,
        args=["-c", "import sys; sys.exit(7)"],
    )
    with pytest.raises(MCPClientError):
        asyncio.run(_list_with_params(params))


def test_no_server_process_is_left_running_after_success_or_failure(settings, test_net):
    baseline = _server_process_count()
    if baseline is None:
        pytest.skip("process enumeration unavailable on this platform")

    list_tools_sync(settings, graph_name=_net_name(settings))  # normal session
    try:  # a session that fails during startup
        asyncio.run(_list_with_params(
            server_parameters(settings, args=["-m", "src.tools.does_not_exist"])
        ))
    except MCPClientError:
        pass

    after = _server_process_count()
    assert after is not None
    assert after <= baseline, f"leaked MCP server processes: {after} > {baseline}"


def test_raw_stdio_session_against_the_server(settings, test_net):
    """Explicit raw-protocol test: stdio_client + ClientSession, no helpers."""

    async def run() -> tuple[list[str], dict[str, Any]]:
        params = server_parameters(settings, graph_name=_net_name(settings))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = sorted(t.name for t in listed.tools)
                raw = await session.call_tool(
                    "detect_cycle", {"account_id": "TNC1", "max_len": 4}
                )
                text = "".join(
                    p.text for p in raw.content if getattr(p, "type", None) == "text"
                )
                return names, json.loads(text)

    names, envelope = asyncio.run(run())
    assert set(names) == EXPECTED_TOOLS
    assert envelope["ok"] is True
    assert envelope["result"]["temporally_corroborated_count"] == 1