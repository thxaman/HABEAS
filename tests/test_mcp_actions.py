"""Phase 5 MCP integration: the sandbox action tools over real stdio.

Two boundaries are exercised here:

1. **Registration is opt-in.** With actions disabled (the default) the server
   advertises exactly the Phase 2 read tools; with the sandbox opt-in it also
   advertises the narrow action tools. Neither set exposes a Cypher argument,
   a generic write, or a graph-name argument.
2. **Eligibility is re-derived server-side.** Calling ``freeze_account`` over
   MCP for a case that has no persisted evidence is DECLINED with structured
   reasons and no mutation — a caller cannot authorize a freeze by asserting
   a verdict.
"""
from __future__ import annotations

from pathlib import Path

from src.agent.mcp_client import call_tool_sync, list_tools_sync
from src.db import run_query

REPO_ROOT = Path(__file__).resolve().parents[1]

READ_TOOLS = {
    "alert_context",
    "entity_profile", "trace_funds", "detect_cycle", "shared_identifiers",
    "fan_in_out", "hub_scores", "prior_cases", "active_path_exists",
    # Phase 8.5: the UI inbox/neighbourhood reads are read-only tools and
    # must never be confused with the write-capable action set.
    "alert_inbox", "alert_neighborhood",
}
ACTION_TOOLS = {
    "create_case", "cite_evidence", "flag_entity", "close_case",
    "simulate_freeze", "freeze_account", "verify_freeze",
}


def _call(settings, tool, arguments, graph_name):
    return call_tool_sync(tool, arguments, settings=settings,
                          graph_name=graph_name)


def _status(graph, account_id):
    rows = run_query(
        graph, "MATCH (a:Account {id: $id}) RETURN a.status",
        {"id": account_id}, read_only=True,
    ).result_set
    return rows[0][0] if rows else None


def test_disabled_actions_are_not_registered(settings, test_net):
    disabled = settings.model_copy(update={"habeas_enable_actions": False})
    names = {t["name"] for t in list_tools_sync(
        disabled, graph_name=disabled.graph_name + "_test_net")}
    assert names == READ_TOOLS


def test_enabled_actions_are_registered_on_the_sandbox(sandbox_settings, action_net):
    names = {t["name"] for t in list_tools_sync(
        sandbox_settings, graph_name=sandbox_settings.graph_name)}
    assert names == READ_TOOLS | ACTION_TOOLS


def test_action_tools_expose_no_cypher_graph_or_write_anything_argument(
    sandbox_settings, action_net
):
    for tool in list_tools_sync(sandbox_settings,
                                graph_name=sandbox_settings.graph_name):
        props = set((tool["input_schema"].get("properties") or {}))
        for forbidden in ("cypher", "query", "graph", "graph_name", "read_only",
                          "allowed", "force"):
            assert forbidden not in props, f"{tool['name']} exposes {forbidden!r}"


def test_action_modules_never_reference_ground_truth():
    for name in ("actions.py", "server.py"):
        text = (REPO_ROOT / "src" / "tools" / name).read_text(encoding="utf-8")
        assert "ground_truth" not in text, name


def test_unapproved_citation_is_a_structured_error_not_a_mutation(
    sandbox_settings, action_net
):
    graph = action_net
    gname = sandbox_settings.graph_name
    created = _call(sandbox_settings, "create_case",
                    {"alert_id": "ALRT-CYCLE", "entity_ids": ["TNC1"]}, gname)
    assert created.ok is True
    case_id = created.envelope["result"]["case_id"]

    bad = _call(sandbox_settings, "cite_evidence", {
        "case_id": case_id,
        "citations": [{"evidence_id": "EV-X", "claim": "invented",
                       "query_id": "MATCH_ALL_DELETE", "stance": "supporting"}],
    }, gname)
    assert bad.ok is False
    assert bad.envelope["error"]["kind"] == "validation"
    stored = run_query(graph, "MATCH (e:Evidence) RETURN count(e)", read_only=True)
    assert int(stored.result_set[0][0]) == 0


def test_simulate_freeze_over_mcp_is_read_only(sandbox_settings, action_net):
    graph = action_net
    gname = sandbox_settings.graph_name
    created = _call(sandbox_settings, "create_case",
                    {"alert_id": "ALRT-CYCLE", "entity_ids": ["TNC1"]}, gname)
    case_id = created.envelope["result"]["case_id"]
    before_edges = run_query(
        graph, "MATCH ()-[r:TRANSFERRED_TO]->() WHERE r.active = true "
               "RETURN count(r)", read_only=True,
    ).result_set[0][0]

    simulated = _call(sandbox_settings, "simulate_freeze",
                      {"case_id": case_id, "account_id": "TNC1"}, gname)
    assert simulated.ok is True
    metrics = simulated.envelope["result"]["metrics"]
    assert metrics["simulation_complete"] is True
    assert metrics["metric_semantics"]["direct_outgoing_volume"]
    after_edges = run_query(
        graph, "MATCH ()-[r:TRANSFERRED_TO]->() WHERE r.active = true "
               "RETURN count(r)", read_only=True,
    ).result_set[0][0]
    assert after_edges == before_edges, "simulation must not deactivate anything"
    assert _status(graph, "TNC1") == "active"


def test_declined_freeze_without_evidence_through_mcp(sandbox_settings, action_net):
    graph = action_net
    gname = sandbox_settings.graph_name
    created = _call(sandbox_settings, "create_case",
                    {"alert_id": "ALRT-CYCLE", "entity_ids": ["TNC1"]}, gname)
    assert created.ok is True
    case_id = created.envelope["result"]["case_id"]
    assert case_id == "CASE-ALRT-CYCLE"

    frozen = _call(sandbox_settings, "freeze_account",
                   {"case_id": case_id, "account_id": "TNC1",
                    "reason": "attempt without any persisted evidence"}, gname)
    assert frozen.ok is True, "declining is a successful, structured outcome"
    result = frozen.envelope["result"]
    assert result["action_status"] == "declined"
    assert result["frozen"] is False
    assert result["mutation_performed"] is False
    assert any("simulation" in reason
               for reason in result["policy"]["blocking_reasons"])

    assert _status(graph, "TNC1") == "active"
    actions = run_query(graph, "MATCH (a:Action) RETURN count(a)", read_only=True)
    assert int(actions.result_set[0][0]) == 0


def test_mcp_cannot_assert_a_verdict_into_eligibility(sandbox_settings, action_net):
    """An injected 'allowed'/'verdict' argument cannot authorize a freeze."""
    graph = action_net
    gname = sandbox_settings.graph_name
    created = _call(sandbox_settings, "create_case",
                    {"alert_id": "ALRT-CYCLE", "entity_ids": ["TNC1"]}, gname)
    case_id = created.envelope["result"]["case_id"]
    raw = _call(sandbox_settings, "freeze_account", {
        "case_id": case_id, "account_id": "TNC1", "reason": "try to bypass",
        "allowed": True, "verdict": "suspicion_supported_for_review",
    }, gname)
    assert raw.ok is True
    result = raw.envelope["result"]
    assert result["action_status"] == "declined", (
        "an injected authorization flag must not create eligibility"
    )
    assert result["policy"]["allowed"] is False
    assert result["mutation_performed"] is False
    assert _status(graph, "TNC1") == "active"
