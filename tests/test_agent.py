"""Phase 4 tests: the investigation agent over the real MCP stdio server.

Deterministic (FakeLLM) scenarios A-G are exercised end-to-end through the
actual MCP server, plus the `alert_context` prerequisite tool and the
`CaseState` / analysis layer. Tests assert on tool selection, evidence
updates, hypothesis changes and termination — not only on final text.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.agent import analysis
from src.agent.llm import (
    FakeLLM,
    FinalDecision,
    LLMConfigurationError,
    RealLLM,
    ScriptedLLM,
    ToolDecision,
    make_llm,
)
from src.agent.loop import run_investigation
from src.agent.mcp_client import ToolCallResult
from src.agent.state import (
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_UNRECOVERABLE_ERROR,
    VERDICT_INSUFFICIENT,
    VERDICT_LIKELY_BENIGN,
    VERDICT_SUSPICIOUS,
    CaseState,
    Hypothesis,
)
from src.db import run_query
from src.tools.reads import alert_context

REPO_ROOT = Path(__file__).resolve().parents[1]


def _net(settings) -> str:
    return settings.graph_name + "_test_net"


def _run(alert: str, settings, graph_name: str, **kwargs):
    return asyncio.run(
        run_investigation(alert, settings=settings, graph_name=graph_name, **kwargs)
    )


@pytest.fixture(scope="module")
def trace_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("traces")


# ---- preflight prerequisite: alert_context ---------------------------------


def test_entity_profile_cannot_resolve_an_alert_id(test_net):
    from src.tools.reads import entity_profile

    env = entity_profile(test_net, "ALRT-CYCLE")
    assert env.ok is True
    assert env.result["found"] is False, "prerequisite is unnecessary if this ever passes"


def test_alert_context_resolves_alert_to_entities_and_transaction(test_net):
    env = alert_context(test_net, "ALRT-CYCLE")
    assert env.ok is True
    result = env.result
    assert result["found"] is True
    assert result["trigger_transaction"]["txn_id"] == "TNT000"
    assert result["sender_account_id"] == "TNC1"
    assert result["receiver_account_id"] == "TNC2"
    assert set(result["entity_ids"]) == {"TNC1", "TNC2"}
    assert "not a finding of wrongdoing" in env.interpretation
    assert [e.query_id for e in env.query_evidence] == ["alert_context_v1"]


def test_alert_context_missing_alert_is_a_clear_not_found(test_net):
    env = alert_context(test_net, "ALRT-NOPE")
    assert env.ok is True
    assert env.result["found"] is False
    assert env.result["entity_ids"] == []
    assert "No Alert" in env.interpretation


def test_alert_context_rejects_unsafe_ids(test_net):
    env = alert_context(test_net, "bad id!")
    assert env.ok is False and env.error.kind == "validation"


# ---- scenario A: planted, temporally corroborated cycle --------------------


def test_scenario_a_corroborated_cycle(settings, test_net, trace_dir):
    result = _run("ALRT-CYCLE", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-A")
    state = result.state
    assert state.tools_called()[:5] == [
        "alert_context", "entity_profile", "detect_cycle", "fan_in_out", "trace_funds",
    ]
    assert any(
        r.tool == "trace_funds" and r.derived.get("observed_leg_events", 0) > 0
        for r in state.evidence
    )
    cycle_evidence = next(r for r in state.evidence if r.tool == "detect_cycle")
    assert cycle_evidence.derived["corroborated_count"] == 1
    assert result.verdict == VERDICT_SUSPICIOUS
    assert set(state.signal_families) == {"circulation", "rapid_forwarding"}
    assert "not a determination" in state.rationale or "offence" in state.rationale
    circ = state.hypothesis("chronologically_corroborated_circulation")
    assert circ.status == "supported" and circ.supporting_evidence
    assert state.errors == []


# ---- scenario B: benign high-fan-in salary receiver ------------------------


def test_scenario_b_benign_fan_in_is_not_declared_fraud(settings, test_net, trace_dir):
    result = _run("ALRT-BENIGN", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-B")
    state = result.state
    assert result.verdict != VERDICT_SUSPICIOUS
    assert result.verdict == VERDICT_LIKELY_BENIGN
    assert "fan_in_out" in state.tools_called()
    fan = next(r for r in state.evidence if r.tool == "fan_in_out")
    assert fan.derived["distinct_senders"] >= 3
    assert fan.derived["out_amount"] == 0.0
    assert not analysis.fan_in_is_concentrated(fan.derived)
    assert state.signal_families == []
    benign = state.hypothesis("benign_high_volume_or_shared_infrastructure")
    assert benign.status == "supported"
    assert "no outgoing volume" in state.rationale


# ---- scenario C: accidental structural cycle -------------------------------


def test_scenario_c_structural_cycle_is_not_claimed_as_corroborated(
    settings, test_net, trace_dir
):
    result = _run("ALRT-STRUCT", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-C")
    state = result.state
    cycle = next(r for r in state.evidence if r.tool == "detect_cycle")
    assert cycle.derived["cycle_count"] >= 1
    assert cycle.derived["corroborated_count"] == 0, "the ring must not be corroborated"
    assert cycle.derived["structural_only_count"] >= 1
    circ = state.hypothesis("chronologically_corroborated_circulation")
    assert circ.status == "contradicted"
    assert circ.uncertainties, "the structural/corroborated distinction must be recorded"
    assert result.verdict != VERDICT_SUSPICIOUS
    assert "structural" in state.rationale.lower()


# ---- scenario D: insufficient / contradictory evidence ---------------------


def test_scenario_d_insufficient_evidence_is_qualified(settings, test_net, trace_dir):
    result = _run("ALRT-THIN", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-D")
    state = result.state
    assert result.verdict == VERDICT_INSUFFICIENT
    assert state.signal_families == []
    assert state.limitations, "an inconclusive result must state its limits"
    assert state.claims
    unresolved = state.hypothesis("insufficient_evidence_to_distinguish")
    assert unresolved.status == "supported"
    assert unresolved.uncertainties


def test_scenario_d2_contradictory_signals_do_not_force_a_fraud_verdict(
    settings, test_net, trace_dir
):
    """A suspicious-looking signal alone must not produce a fraud verdict."""
    scripted = ScriptedLLM([
        ToolDecision(tool="hub_scores", arguments={"window_days": 365, "limit": 10},
                     purpose="look for an outlying receiver"),
        FinalDecision(reason="no further discriminating query"),
    ])
    result = _run("ALRT-THIN", settings, _net(settings), llm=scripted,
                  trace_dir=trace_dir, case_id="CASE-D2")
    state = result.state
    assert result.verdict == VERDICT_INSUFFICIENT
    hub = next((r for r in state.evidence if r.tool == "hub_scores"), None)
    assert hub is not None and hub.ok
    assert hub.derived["method"] == "weighted_in_degree"
    assert state.signal_families == []
    assert all("fraud" not in c.claim.lower() for c in state.claims)


# ---- scenario E: tool-budget exhaustion ------------------------------------


def test_scenario_e_budget_exhaustion_terminates_safely(settings, test_net, trace_dir):
    result = _run("ALRT-CYCLE", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-E", max_steps=2)
    state = result.state
    assert state.tool_calls_used <= state.tool_budget
    assert state.termination_reason in {
        TERMINATION_BUDGET_EXHAUSTED, "no_further_discriminating_query",
    }
    assert state.evidence, "evidence must survive early termination"
    assert state.investigation_steps
    trace_path = Path(state.trace_path)
    assert trace_path.exists()
    records = [
        json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
    ]
    assert len([r for r in records if r.get("event") == "tool_call"]) == state.tool_calls_used


def test_scenario_e2_exhausted_budget_stops_tool_calls(settings, test_net, trace_dir):
    from src.config import Settings

    tight = Settings(**{**settings.model_dump(), "tool_budget": 1})
    result = _run("ALRT-CYCLE", tight, _net(tight), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-E2")
    state = result.state
    # the budget is enforced in code: triage used the single allowed call
    assert state.tool_calls_used == 1
    assert state.termination_reason == TERMINATION_BUDGET_EXHAUSTED
    assert len(state.evidence) == 1
    assert any("budget" in lim for lim in state.limitations)


# ---- scenario F: MCP failure / malformed tool output -----------------------


class FailingClient:
    """Raises on one tool; otherwise returns minimal valid envelopes."""

    def __init__(self, fail_on: str = "trace_funds") -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []

    async def list_tools(self):
        return [
            {"name": n, "description": "",
             "input_schema": {"type": "object", "properties": {}}}
            for n in ("alert_context", "entity_profile", "detect_cycle",
                      "fan_in_out", "trace_funds", "hub_scores")
        ]

    async def call_tool(self, name: str, arguments):
        self.calls.append(name)
        if name == self.fail_on:
            raise RuntimeError("simulated MCP transport failure")
        if name == "alert_context":
            envelope = {
                "ok": True, "tool": name, "duration_ms": 1.0,
                "query_evidence": [{
                    "query_id": "alert_context_v1", "params": {}, "rows": [],
                    "columns": [], "rows_hash": "sha256:x", "row_count": 1,
                    "truncated": False,
                }],
                "result": {
                    "found": True, "alert_id": "ALRT-CYCLE", "reason": "r",
                    "trigger_transaction": {"txn_id": "TNT000"},
                    "sender_account_id": "TNC1", "receiver_account_id": "TNC2",
                    "concerned_account_id": "TNC1",
                    "entity_ids": ["TNC1", "TNC2"],
                },
                "interpretation": "resolved",
            }
        else:
            envelope = {"ok": True, "tool": name, "duration_ms": 1.0,
                        "query_evidence": [], "result": {}, "interpretation": ""}
        return ToolCallResult(tool=name, ok=True, envelope=envelope,
                              text=json.dumps(envelope), is_error=False)


def test_scenario_f_transport_failure_is_recorded_not_invented(
    settings, test_net, trace_dir
):
    client = FailingClient(fail_on="trace_funds")
    result = asyncio.run(run_investigation(
        "ALRT-CYCLE", settings=settings, llm=FakeLLM(), client=client,
        trace_dir=trace_dir, case_id="CASE-F",
    ))
    state = result.state
    failed = [r for r in state.evidence if not r.ok]
    assert failed, "the failure must be recorded as evidence"
    assert failed[0].tool == "trace_funds"
    assert "simulated MCP transport failure" in (failed[0].error or "")
    assert state.termination_reason == TERMINATION_UNRECOVERABLE_ERROR
    assert state.errors
    # no graph evidence may be invented for the failed call
    assert failed[0].rows_hash == [] and failed[0].query_ids == []
    assert result.verdict == VERDICT_INSUFFICIENT
    assert any("failed" in lim for lim in state.limitations)


def test_scenario_f2_malformed_tool_output_is_flagged(settings, test_net, trace_dir):
    class MalformedClient(FailingClient):
        async def call_tool(self, name, arguments):
            self.calls.append(name)
            return ToolCallResult(tool=name, ok=False, envelope=None, text="not json",
                                  is_error=True, error="returned non-JSON content")

    result = asyncio.run(run_investigation(
        "ALRT-CYCLE", settings=settings, llm=FakeLLM(), client=MalformedClient(),
        trace_dir=trace_dir, case_id="CASE-F2",
    ))
    state = result.state
    assert not state.evidence[0].ok
    assert state.errors
    assert result.verdict == VERDICT_INSUFFICIENT


# ---- scenario G: repeated investigation determinism ------------------------


def test_scenario_g_repeated_investigation_is_deterministic(settings, test_net, trace_dir):
    first = _run("ALRT-CYCLE", settings, _net(settings), llm=FakeLLM(),
                 trace_dir=trace_dir, case_id="CASE-G1")
    second = _run("ALRT-CYCLE", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-G2")
    assert first.tool_sequence == second.tool_sequence
    assert first.verdict == second.verdict
    assert first.state.signal_families == second.state.signal_families
    assert first.state.rationale == second.state.rationale
    assert first.termination_reason == second.termination_reason
    # No case-memory savings are claimed anywhere before Phase 5 exists.
    assert "reus" not in first.state.rationale.lower()
    assert "reus" not in json.dumps(first.state.model_dump()).lower()


# ---- state model and analysis units ---------------------------------------


def test_case_state_serializes_without_secrets_or_ground_truth(
    settings, test_net, trace_dir
):
    result = _run("ALRT-CYCLE", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-SER")
    payload = result.state.model_dump_json()
    for forbidden in ("ground_truth", "is_fraud", "pattern_type", "sk-",
                      "OPENAI_API_KEY"):
        assert forbidden not in payload
    restored = CaseState.model_validate_json(payload)
    assert restored.alert_id == result.state.alert_id
    assert len(restored.evidence) == len(result.state.evidence)


def test_evidence_records_carry_query_ids_hashes_and_params(settings, test_net, trace_dir):
    result = _run("ALRT-FANIN", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-EV")
    ok_records = [r for r in result.state.evidence if r.ok]
    assert ok_records
    for record in ok_records:
        assert record.query_ids
        assert all(h.startswith("sha256:") for h in record.rows_hash)
        assert len(record.rows_hash) == len(record.query_ids)
        assert record.interpretation.strip()


def test_tool_budget_counts_failed_calls_too(settings, test_net, trace_dir):
    client = FailingClient(fail_on="trace_funds")
    result = asyncio.run(run_investigation(
        "ALRT-CYCLE", settings=settings, llm=FakeLLM(), client=client,
        trace_dir=trace_dir, case_id="CASE-COUNT",
    ))
    assert result.state.tool_calls_used == len(client.calls)


def test_trace_records_are_complete_and_secret_free(settings, test_net, trace_dir):
    result = _run("ALRT-FANIN", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=trace_dir, case_id="CASE-TRACE")
    records = [
        json.loads(line)
        for line in Path(result.state.trace_path).read_text().splitlines() if line.strip()
    ]
    tool_calls = [r for r in records if r["event"] == "tool_call"]
    assert len(tool_calls) == result.state.tool_calls_used
    for record in tool_calls:
        for key in ("ts", "alert_id", "stage", "tool", "arguments", "ok",
                    "query_ids", "evidence_hashes", "row_counts", "truncated",
                    "duration_ms", "transition"):
            assert key in record, key
        assert "api_key" not in json.dumps(record).lower()
    stages = [r["stage"] for r in records if r["event"] == "stage"]
    for stage in ("TRIAGE", "CHALLENGE", "DECIDE", "REPORT"):
        assert stage in stages


def test_trace_write_failure_is_visible_not_fabricated(settings, test_net, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    result = _run("ALRT-CYCLE", settings, _net(settings), llm=FakeLLM(),
                  trace_dir=blocked, case_id="CASE-TRACEFAIL")
    assert any("trace write failed" in e for e in result.state.errors)


# ---- LLM configuration ----------------------------------------------------


def test_fake_mode_is_the_default_and_needs_no_credentials(settings):
    assert make_llm(settings).mode == "fake"


def test_real_mode_without_credentials_raises_clear_error(settings):
    from src.config import Settings

    bare = Settings(**{
        **settings.model_dump(), "llm_mode": "real", "llm_provider": "openai",
        "openai_api_key": "", "groq_api_key": "",
    })
    with pytest.raises(LLMConfigurationError) as excinfo:
        RealLLM(bare)
    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "fake" in message  # points at the offline alternative


def test_real_mode_error_never_leaks_a_secret(settings):
    from src.config import Settings

    secret = "sk-this-must-not-appear-in-errors"
    bad = Settings(**{
        **settings.model_dump(), "llm_mode": "real", "llm_provider": "openai",
        "openai_api_key": secret, "openai_model": "",
    })
    with pytest.raises(LLMConfigurationError) as excinfo:
        RealLLM(bad)
    assert secret not in str(excinfo.value)


def test_unknown_llm_mode_is_rejected(settings):
    with pytest.raises(LLMConfigurationError):
        make_llm(settings, mode="banana")


def test_system_prompt_states_the_evidence_discipline():
    from src.agent.prompts import SYSTEM_PROMPT

    for phrase in ("not to confirm", "untrusted", "suspicious_activity",
                   "likely_benign", "insufficient_evidence"):
        assert phrase in SYSTEM_PROMPT


def test_agent_modules_never_import_graph_reads_directly():
    """The agent must reach the graph only through MCP."""
    for name in ("loop.py", "llm.py", "analysis.py", "prompts.py", "state.py",
                 "trace.py"):
        text = (REPO_ROOT / "src" / "agent" / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                for banned in ("src.tools.reads", "src.tools.algorithms",
                               "src.seed", "get_graph", "src.db"):
                    assert banned not in stripped, f"{name} imports {banned}"


# ---- read-only guarantee ---------------------------------------------------


def test_investigations_leave_graph_counts_unchanged(settings, test_net, trace_dir):
    def counts():
        row = run_query(
            test_net,
            "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() "
            "RETURN nodes AS nodes, count(r) AS rels",
            read_only=True,
        ).result_set[0]
        return int(row[0]), int(row[1])

    before = counts()
    for alert in ("ALRT-CYCLE", "ALRT-BENIGN", "ALRT-STRUCT"):
        _run(alert, settings, _net(settings), llm=FakeLLM(),
             trace_dir=trace_dir, case_id=f"CASE-RO-{alert}")
    assert counts() == before, "Phase 4 must not mutate the graph"


def test_hypothesis_model_fields_are_present():
    h = Hypothesis(name="x")
    for field in ("name", "status", "supporting_evidence", "contradicting_evidence",
                  "uncertainties", "rationale"):
        assert field in h.model_dump()