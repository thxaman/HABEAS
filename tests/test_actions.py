"""Phase 5 tests: case memory, simulation, and the guarded sandbox freeze.

Destructive cases run against the controlled `action_net` graph, which the
fixture rebuilds before and after each test, so nothing depends on the shared
graph's mutable state.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pytest

from src.agent import analysis
from src.agent.llm import FakeLLM
from src.agent.policy import (
    MIN_EVIDENCE_RECORDS,
    VERDICT_INSUFFICIENT_EVIDENCE,
    VERDICT_LIKELY_BENIGN,
    VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
    VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION,
    canonical_verdict,
    escalation_verdict,
    evaluate_freeze_eligibility,
)
from src.agent.workflow import run_case_workflow
from src.db import run_query
from src.tools import query_catalog as qc
from src.tools.actions import (
    cite_evidence,
    close_case,
    create_case,
    flag_entity,
    freeze_account,
    simulate_freeze,
    verify_freeze,
)
from src.tools.reads import prior_cases

REPO_ROOT = Path(__file__).resolve().parents[1]

CYCLE_ALERT = "ALRT-CYCLE"
BENIGN_ALERT = "ALRT-BENIGN"
STRUCT_ALERT = "ALRT-STRUCT"
FANIN_ALERT = "ALRT-FANIN"
CYCLE_CASE = f"CASE-{CYCLE_ALERT}"
# Corrected-policy (v2) qualifying scenario: a slow ring plus one genuinely
# distinct rapid pass-through on TNQ1 — the only fixture scenario where two
# target-relevant families with disjoint observations support a freeze.
QUAL_ALERT = "ALRT-QUAL"
QUAL_CASE = f"CASE-{QUAL_ALERT}"


@pytest.fixture(scope="module")
def trace_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("traces5")


def _workflow(alert, settings, trace_dir, **kwargs):
    return asyncio.run(run_case_workflow(
        alert, settings=settings, llm=FakeLLM(), trace_dir=trace_dir, **kwargs
    ))


def _result_of(result, key):
    return (result.envelopes.get(key) or {}).get("result") or {}


def _account_status(graph, account_id):
    row = run_query(graph, "MATCH (a:Account {id: $id}) RETURN a.status",
                    {"id": account_id}, read_only=True).result_set
    return row[0][0] if row else None


def _active_edges(graph):
    return int(run_query(graph, "MATCH ()-[r:TRANSFERRED_TO]->() WHERE r.active = true "
                                "RETURN count(r)", read_only=True).result_set[0][0])


def _prepare_case(settings, graph, alert, trace_dir):
    """Investigate + record case + cite evidence, without attempting a freeze."""
    return _workflow(alert, settings, trace_dir, freeze=False)


# ---- canonical verdict vocabulary -----------------------------------------


def test_phase4_verdicts_map_onto_the_canonical_vocabulary():
    assert canonical_verdict("suspicious_activity") == VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION
    assert canonical_verdict("likely_benign") == VERDICT_LIKELY_BENIGN
    assert canonical_verdict("insufficient_evidence") == VERDICT_INSUFFICIENT_EVIDENCE
    assert canonical_verdict(None) == "incomplete_investigation"
    # a provisional suspicion is NOT a supported finding
    assert canonical_verdict("suspicious_activity") != VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW


def test_escalation_requires_evidence_not_a_provisional_label(
    sandbox_settings, action_net, trace_dir
):
    result = _workflow(CYCLE_ALERT, sandbox_settings, trace_dir, freeze=False)
    state = result.state
    assert state.provisional_verdict == "suspicious_activity"
    assert escalation_verdict(state, state.provisional_verdict) == \
        VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW

    starved = state.model_copy(deep=True)
    starved.evidence = [r for r in starved.evidence if r.tool == "detect_cycle"]
    assert escalation_verdict(starved, "suspicious_activity") == \
        VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION


# ---- A: supported suspicious case with a permitted sandbox freeze ----------


def test_scenario_a_supported_case_is_frozen_and_verified(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _workflow(QUAL_ALERT, sandbox_settings, trace_dir)

    assert result.canonical_verdict == VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW
    assert result.frozen is True

    sim = _result_of(result, "simulate_freeze")["metrics"]
    assert sim["simulation_complete"] is True
    assert sim["active_edges_disabled"] == 4
    assert sim["downstream_accounts_before"] == 4
    assert sim["newly_unreachable_count"] == 4
    assert sim["direct_outgoing_volume"] == 3880.0
    assert "NOT money that would be frozen" in \
        sim["metric_semantics"]["direct_outgoing_volume"]
    assert "funds actually unavailable (no accounting model exists)" in sim["excluded"]

    freeze = _result_of(result, "freeze_account")
    assert freeze["action_status"] == "applied"
    assert freeze["mutation_performed"] is True
    assert freeze["policy"]["allowed"] is True
    assert freeze["policy"]["policy_version"] == 2
    assert freeze["policy"]["qualifying_evidence_ids"]
    assert freeze["policy"]["distinct_provenance_families"] == [
        "circulation", "rapid_forwarding",
    ]
    pair = freeze["policy"]["provenance"]["pairs"][0]
    assert pair["pair"] == ["circulation", "rapid_forwarding"]
    assert pair["classification"] == "distinct_observations"
    assert pair["shared_count"] == 0

    # the persisted action carries the exact, case-scoped qualifying evidence
    # references (the records that back the two counted families plus the
    # target's alert context)
    assert freeze["evidence_ids"] == [
        f"{QUAL_CASE}:EV-0001", f"{QUAL_CASE}:EV-0003",
        f"{QUAL_CASE}:EV-0005", f"{QUAL_CASE}:EV-0006",
    ]

    assert _account_status(graph, "TNQ1") == "frozen"
    verify = _result_of(result, "verify_freeze")
    assert verify["verified"] is True
    assert verify["checks"]["account_is_frozen"] is True
    assert verify["checks"]["action_record_present"] is True
    assert verify["checks"]["routing_edge_count_matches_simulation"] is True
    assert verify["simulation_expected_active_edges"] == verify["active_edges_total"]
    assert verify["active_incident_edges"] == 0


# ---- B: benign high-fan-in receiver is NOT frozen --------------------------


def test_scenario_b_benign_receiver_is_flagged_but_not_frozen(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _workflow(BENIGN_ALERT, sandbox_settings, trace_dir)
    state = result.state

    # A naive fan-in heuristic WOULD have raised this account...
    fan = next(r for r in state.evidence if r.tool == "fan_in_out")
    assert fan.derived["distinct_senders"] >= 3
    # ...but the discriminating test shows no forwarding concentration.
    assert not analysis.fan_in_is_concentrated(fan.derived)
    assert state.signal_families == []

    assert result.canonical_verdict == VERDICT_LIKELY_BENIGN
    assert result.frozen is False
    assert result.freeze_declined is True
    freeze = _result_of(result, "freeze_account")
    assert freeze["frozen"] is False
    assert freeze["mutation_performed"] is False
    assert any("verdict" in r for r in freeze["policy"]["blocking_reasons"])
    assert _account_status(graph, "TNBEN") == "active"


# ---- C: insufficient evidence cannot be frozen -----------------------------


def test_scenario_c_insufficient_evidence_is_never_frozen(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _workflow(STRUCT_ALERT, sandbox_settings, trace_dir)
    assert result.canonical_verdict == VERDICT_INSUFFICIENT_EVIDENCE
    assert result.frozen is False
    freeze = _result_of(result, "freeze_account")
    assert freeze["mutation_performed"] is False
    assert freeze["frozen"] is False
    assert _account_status(graph, "TNN1") == "active"
    assert "structural" in result.state.rationale.lower()


# ---- D: missing / wrong-case / wrong-account / stale simulation ------------


def test_scenario_d_missing_simulation_blocks_the_freeze(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    out = freeze_account(
        graph, CYCLE_CASE, "TNC1", "no simulation exists",
        eligibility_inputs={"case_state": result.state}, settings=sandbox_settings,
    )
    assert out.ok is True
    assert out.result["mutation_performed"] is False
    assert out.result["action_status"] == "declined"
    assert "no simulation is recorded" in out.interpretation
    assert _account_status(graph, "TNC1") == "active"


def test_scenario_d_wrong_account_simulation_blocks_the_freeze(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    sim = simulate_freeze(graph, CYCLE_CASE, "TNC3", settings=sandbox_settings)
    assert sim.ok is True
    out = freeze_account(
        graph, CYCLE_CASE, "TNC1", "wrong-account simulation",
        eligibility_inputs={"case_state": result.state}, settings=sandbox_settings,
    )
    assert out.result["mutation_performed"] is False
    assert "no simulation is recorded" in out.interpretation
    assert _account_status(graph, "TNC1") == "active"


def test_scenario_d_wrong_case_simulation_blocks_the_freeze(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    create_case(graph, FANIN_ALERT, entity_ids=["TNI1"], settings=sandbox_settings)
    simulate_freeze(graph, CYCLE_CASE, "TNC1", settings=sandbox_settings)
    out = freeze_account(
        graph, f"CASE-{FANIN_ALERT}", "TNC1", "wrong-case simulation",
        eligibility_inputs={"case_state": result.state}, settings=sandbox_settings,
    )
    assert out.result["mutation_performed"] is False
    assert _account_status(graph, "TNC1") == "active"


def test_scenario_d_stale_simulation_blocks_the_freeze(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    sim = simulate_freeze(graph, CYCLE_CASE, "TNC1", settings=sandbox_settings)
    assert sim.ok is True and sim.result["metrics"]["simulation_complete"] is True

    # a material change to the frozen candidate's OWN routing state after the
    # simulation (removing incident edges changes the fingerprint)
    run_query(
        graph, qc.get_template("deactivate_incident_edges_v1"),
        {"account_id": "TNC1", "now": "2026-02-02T00:00:00Z", "case_id": "TAMPER"},
        read_only=False,
    )
    out = freeze_account(
        graph, CYCLE_CASE, "TNC1", "stale simulation",
        eligibility_inputs={"case_state": result.state}, settings=sandbox_settings,
    )
    assert out.result["mutation_performed"] is False
    assert "stale" in out.interpretation
    assert _account_status(graph, "TNC1") == "active"


def test_scenario_d_incomplete_simulation_blocks_the_freeze(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    sim = simulate_freeze(graph, CYCLE_CASE, "TNC1", topology_limit=1,
                          settings=sandbox_settings)
    assert sim.ok is True
    assert sim.result["metrics"]["simulation_complete"] is False
    assert sim.result["metrics"]["truncated_queries"]

    out = freeze_account(
        graph, CYCLE_CASE, "TNC1", "truncated snapshot",
        eligibility_inputs={"case_state": result.state}, settings=sandbox_settings,
    )
    assert out.ok is True
    assert out.result["mutation_performed"] is False
    assert "incomplete" in out.interpretation
    # the persisted simulation says it is incomplete, so nothing may mutate
    persisted = run_query(
        graph, qc.get_template("simulation_lookup_v1"),
        {"simulation_id": sim.result["simulation_id"]}, read_only=True,
    ).result_set[0][0]
    assert persisted["status"] == "incomplete"
    assert _account_status(graph, "TNC1") == "active"


# ---- E: idempotency --------------------------------------------------------


def test_scenario_e_repeated_case_creation_is_idempotent(sandbox_settings, action_net):
    graph = action_net
    first = create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"],
                        settings=sandbox_settings)
    second = create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"],
                         settings=sandbox_settings)
    assert first.result["created"] is True
    assert second.result["created"] is False
    assert first.result["case_id"] == second.result["case_id"] == CYCLE_CASE
    assert first.result["created_at"] == second.result["created_at"]
    count = run_query(graph, "MATCH (c:Case {id: $id}) RETURN count(c)",
                      {"id": CYCLE_CASE}, read_only=True).result_set[0][0]
    assert int(count) == 1


def test_scenario_e_repeated_freeze_does_not_mutate_twice(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    first = _workflow(QUAL_ALERT, sandbox_settings, trace_dir)
    assert first.frozen is True
    edges_after_first = _active_edges(graph)

    second = freeze_account(
        graph, QUAL_CASE, "TNQ1", "repeat attempt",
        eligibility_inputs={"case_state": first.state}, settings=sandbox_settings,
    )
    assert second.result["action_status"] == "already_applied"
    assert second.result["mutation_performed"] is False
    assert _active_edges(graph) == edges_after_first
    actions = run_query(
        graph, "MATCH (c:Case {id: $id})-[:HAS_ACTION]->(a:Action) RETURN count(a)",
        {"id": QUAL_CASE}, read_only=True,
    ).result_set[0][0]
    assert int(actions) == 1


# ---- F: contradictions and unresolved hypotheses block the action ----------


def _synthetic_state(evidence_records, primary=("TNC1",)):
    from src.agent.state import CaseState

    state = CaseState(alert_id="SYNTH", case_id="CASE-SYNTH",
                      primary_entity_ids=list(primary), evidence=evidence_records)
    state.hypotheses = analysis.recompute_hypotheses(state)
    return state


def _evidence(evidence_id, tool, derived, ok=True):
    from src.agent.state import EvidenceRecord

    return EvidenceRecord(
        evidence_id=evidence_id, tool=tool, stage="test", ok=ok, derived=derived,
        query_ids=["detect_closed_paths_v1"], rows_hash=["sha256:x"],
        row_count=[1], truncated=[False],
    )


def _rapid_trace(evidence_id="EV-3", truncated=False):
    from src.agent.state import EvidenceRecord

    return EvidenceRecord(
        evidence_id=evidence_id, tool="trace_funds", stage="test", ok=True,
        derived={"account_id": "TNC1", "rapid_forward_count": 1, "leg_events": [
            {"src": "TNC4", "dst": "TNC1", "ts": "2026-02-01T00:00:00Z", "amount": 100.0},
            {"src": "TNC1", "dst": "TNC2", "ts": "2026-02-01T00:30:00Z", "amount": 90.0},
        ]},
        query_ids=["trace_paths_out_v1"], rows_hash=["sha256:x"],
        row_count=[100 if truncated else 2], truncated=[truncated],
    )


def _cycle_evidence(evidence_id="EV-1"):
    return _evidence(evidence_id, "detect_cycle",
                     {"cycle_count": 1, "corroborated_count": 1,
                      "structural_only_count": 0,
                      "cycles": [{"members": ["TNC1", "TNC2", "TNC3", "TNC4"],
                                  "temporally_corroborated": True}]})


def _concentrated_fan_in(evidence_id="EV-2"):
    return _evidence(evidence_id, "fan_in_out",
                     {"distinct_senders": 5, "distinct_receivers": 1,
                      "in_amount": 100.0, "out_amount": 90.0, "out_in_ratio": 0.9,
                      "top_receiver": "TNC2", "top_receiver_share": 1.0})


def _evaluate(state, **overrides):
    kwargs = dict(
        verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW, target_account_id="TNC1",
        simulation={"simulation_complete": True}, simulation_stale=False,
        account_exists=True, account_already_frozen=False, sandbox_enabled=True,
        graph_is_demo=True, actions_enabled=True,
    )
    kwargs.update(overrides)
    return evaluate_freeze_eligibility(state, **kwargs)


def test_scenario_f_failed_query_blocks_the_action():
    state = _synthetic_state([
        _cycle_evidence(), _concentrated_fan_in(),
        _evidence("EV-3", "trace_funds", {}, ok=False),
    ])
    decision = _evaluate(state)
    assert decision.allowed is False
    assert any("failed" in r for r in decision.blocking_reasons)


def test_scenario_f_material_truncation_blocks_the_action():
    state = _synthetic_state([_cycle_evidence(), _concentrated_fan_in(),
                              _rapid_trace(truncated=True)])
    decision = _evaluate(state)
    assert any("truncated" in r for r in decision.blocking_reasons)


def test_scenario_f_target_must_be_supported_by_the_evidence():
    state = _synthetic_state([_cycle_evidence(), _concentrated_fan_in(),
                              _rapid_trace()])
    decision = _evaluate(state, target_account_id="TNC9")
    assert decision.allowed is False
    assert any("not the entity the evidence supports" in r
               for r in decision.blocking_reasons)


def test_scenario_f_unresolved_benign_hypothesis_is_reported():
    """When the benign explanation is not ruled out the policy says so."""
    state = _synthetic_state([_cycle_evidence("EV-1")])
    state.hypotheses = analysis.recompute_hypotheses(state)
    decision = _evaluate(state)
    benign = state.hypothesis("benign_high_volume_or_shared_infrastructure")
    assert benign is not None
    if benign.status != "contradicted":
        assert decision.allowed is False
        assert any("benign explanation" in r for r in decision.blocking_reasons)


def test_scenario_f_policy_cannot_be_authorized_from_outside():
    parameters = inspect.signature(evaluate_freeze_eligibility).parameters
    assert "allowed" not in parameters
    assert "verdict" in parameters


# ---- G: evidence citation integrity ---------------------------------------


def test_scenario_g_citation_requires_an_approved_query_id(sandbox_settings, action_net):
    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)
    bad = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-0001", "claim": "invented",
          "query_id": "MATCH_ALL_WRITE", "stance": "supporting"}],
        settings=sandbox_settings,
    )
    assert bad.ok is False
    assert bad.error.kind == "validation"
    assert "unapproved query id" in bad.error.message
    stored = run_query(graph, "MATCH (e:Evidence) RETURN count(e)", read_only=True)
    assert int(stored.result_set[0][0]) == 0, "nothing may be persisted on rejection"


def test_scenario_g_citation_rejects_empty_claim_and_bad_stance(
    sandbox_settings, action_net
):
    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)
    assert cite_evidence(graph, CYCLE_CASE, [], settings=sandbox_settings).ok is False
    bad_claim = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-1", "claim": "  ", "query_id": "detect_closed_paths_v1"}],
        settings=sandbox_settings,
    )
    assert bad_claim.ok is False and "claim" in bad_claim.error.message
    bad_stance = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-1", "claim": "c", "query_id": "detect_closed_paths_v1",
          "stance": "definitely_guilty"}],
        settings=sandbox_settings,
    )
    assert bad_stance.ok is False and "stance" in bad_stance.error.message


def test_scenario_g_evidence_is_persisted_with_reproducible_metadata(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    cited = _result_of(result, "cite_evidence")
    assert cited["cited_count"] >= MIN_EVIDENCE_RECORDS
    assert sum(cited["stance_counts"].values()) == cited["cited_count"]
    assert set(cited["stance_counts"]) == {"supporting", "contradicting", "unresolved"}

    rows = run_query(
        graph, qc.get_template("case_evidence_v1"),
        {"case_id": CYCLE_CASE, "limit": 100}, read_only=True,
    ).result_set
    assert rows
    columns = ["evidence_id", "claim", "stance", "query_id", "source_tool",
               "params_json", "derived_json", "result_hash", "row_count",
               "truncated", "interpretation", "evidence_kind", "captured_at"]
    for row in rows:
        item = dict(zip(columns, row))
        assert item["query_id"] in qc._QUERY_TEMPLATES
        assert item["result_hash"].startswith("sha256:")
        assert item["evidence_kind"] in {"immutable_history", "current_state",
                                        "captured_snapshot"}
        json.loads(item["params_json"])


def test_scenario_g_no_tool_path_executes_model_supplied_cypher():
    for func in (create_case, cite_evidence, flag_entity, close_case,
                 simulate_freeze, freeze_account, verify_freeze):
        parameters = inspect.signature(func).parameters
        assert "cypher" not in parameters, func.__name__
        assert "query" not in parameters, func.__name__
    assert qc.WRITE_QUERY_IDS
    for query_id in qc.WRITE_QUERY_IDS:
        assert "$" in qc.get_template(query_id)


def test_scenario_g_citation_hash_must_match_the_normalized_rows(
    sandbox_settings, action_net
):
    from src.tools.envelopes import rows_hash

    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)
    columns = ["community"]
    rows = [["ring-1"]]
    good_hash = rows_hash(columns, rows)

    mismatched = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-H1", "claim": "mismatch",
          "query_id": "detect_closed_paths_v1",
          "columns": columns, "rows": rows, "result_hash": "sha256:" + "0" * 64}],
        settings=sandbox_settings,
    )
    assert mismatched.ok is False and mismatched.error.kind == "validation"
    assert "does not match its normalized rows" in mismatched.error.message

    malformed = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-H2", "claim": "bad hash",
          "query_id": "detect_closed_paths_v1", "result_hash": "not-a-hash"}],
        settings=sandbox_settings,
    )
    assert malformed.ok is False and "malformed result_hash" in malformed.error.message

    without_columns = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-H3", "claim": "rows without columns",
          "query_id": "detect_closed_paths_v1", "rows": rows}],
        settings=sandbox_settings,
    )
    assert without_columns.ok is False
    assert "without columns" in without_columns.error.message

    # nothing may have been persisted by any rejected citation
    count = run_query(graph, "MATCH (e:Evidence) RETURN count(e)", read_only=True)
    assert int(count.result_set[0][0]) == 0

    ok = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-H4", "claim": "consistent",
          "query_id": "detect_closed_paths_v1",
          "columns": columns, "rows": rows, "result_hash": good_hash}],
        settings=sandbox_settings,
    )
    assert ok.ok is True
    assert ok.result["citations"][0]["result_hash"] == good_hash


def test_scenario_g_recitation_versions_instead_of_overwriting(
    sandbox_settings, action_net
):
    from src.tools.envelopes import rows_hash

    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)
    columns = ["n"]
    original_rows = [["first-observation"]]
    original_hash = rows_hash(columns, original_rows)
    first = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-V1", "claim": "snapshot one",
          "query_id": "detect_closed_paths_v1",
          "columns": columns, "rows": original_rows,
          "result_hash": original_hash}],
        settings=sandbox_settings,
    )
    assert first.ok is True

    changed_rows = [["second-observation"]]
    second = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-V1", "claim": "snapshot two",
          "query_id": "detect_closed_paths_v1",
          "columns": columns, "rows": changed_rows}],
        settings=sandbox_settings,
    )
    assert second.ok is True
    stored = second.result["citations"][0]
    assert stored["versioned"] is True
    assert stored["evidence_id"] == f"{CYCLE_CASE}:EV-V1-v2"

    # the ORIGINAL snapshot is still present, unchanged
    original = run_query(
        graph, qc.get_template("evidence_lookup_v1"),
        {"evidence_id": f"{CYCLE_CASE}:EV-V1"}, read_only=True,
    ).result_set
    assert original and original[0][0] == original_hash
    original_claim = run_query(
        graph, "MATCH (e:Evidence {id: $id}) RETURN e.claim",
        {"id": f"{CYCLE_CASE}:EV-V1"}, read_only=True,
    ).result_set
    assert original_claim[0][0] == "snapshot one"


def test_scenario_g_hash_only_snapshot_is_accepted(sandbox_settings, action_net):
    """The workflow cites hash-only snapshots (rows omitted); that is allowed."""
    from src.tools.envelopes import rows_hash

    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)
    real_hash = rows_hash(["src", "dst"], [["TNC1", "TNC2"]])
    out = cite_evidence(
        graph, CYCLE_CASE,
        [{"evidence_id": "EV-HASHONLY", "claim": "hash-only snapshot",
          "query_id": "trace_paths_out_v1", "rows": [], "columns": [],
          "result_hash": real_hash, "derived": {"path_count": 1}}],
        settings=sandbox_settings,
    )
    assert out.ok is True
    assert out.result["citations"][0]["result_hash"] == real_hash


# ---- H: persistent case memory --------------------------------------------

CYCLE2_ALERT = "ALRT-CYCLE2"
CYCLE2_CASE = f"CASE-{CYCLE2_ALERT}"


def test_scenario_h_prior_cases_discovers_the_new_case(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    first = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    assert first.case_id == CYCLE_CASE

    prior = prior_cases(graph, "TNC1")
    assert prior.ok is True
    assert prior.result["case_count"] >= 1
    case = next(c for c in prior.result["cases"] if c["case_id"] == CYCLE_CASE)
    assert case["verdict"] == VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW
    assert case["updated_at"]

    # a repeat investigation re-derives the same case id, pulls the earlier
    # case as a LEAD, and still re-runs its current-state checks
    second = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    assert second.case_id == CYCLE_CASE
    assert second.tool_sequence[:2] == first.tool_sequence[:2]
    assert second.tool_sequence[2] == "prior_cases"
    assert second.tool_sequence[3:] == first.tool_sequence[2:]
    assert second.state.prior_case_leads == [CYCLE_CASE]
    assert "detect_cycle" in second.tool_sequence, "current state must be rechecked"
    assert _result_of(second, "create_case")["created"] is False
    # a same-alert repeat must not link the case to itself
    assert _result_of(second, "create_case")["reused_case_ids"] == []


def test_scenario_h_cross_alert_reuse_links_but_does_not_inherit_verdicts(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    first = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    assert first.case_id == CYCLE_CASE

    second = _prepare_case(sandbox_settings, graph, CYCLE2_ALERT, trace_dir)
    assert second.case_id == CYCLE2_CASE
    assert second.state.prior_case_leads == [CYCLE_CASE]

    created = _result_of(second, "create_case")
    assert created["reused_case_ids"] == [CYCLE_CASE]
    interpretation = (second.envelopes["create_case"] or {}).get("interpretation", "")
    assert "not settled truth" in interpretation

    linked = run_query(
        graph,
        "MATCH (c:Case {id: $new})-[:REUSES]->(o:Case) "
        "RETURN o.id AS id, o.verdict AS verdict",
        {"new": CYCLE2_CASE}, read_only=True,
    ).result_set
    assert [r[0] for r in linked] == [CYCLE_CASE]
    # the earlier case keeps its own record; the new case is a separate node
    assert linked[0][1] == VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW
    # the verdict was re-derived from fresh evidence, not inherited
    assert second.state.provisional_verdict == "suspicious_activity"
    assert "detect_cycle" in second.tool_sequence


def test_scenario_h_reuse_references_are_validated_before_any_write(
    sandbox_settings, action_net
):
    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)

    missing = create_case(
        graph, FANIN_ALERT, reuses_case_ids=["CASE-DOES-NOT-EXIST"],
        settings=sandbox_settings,
    )
    assert missing.ok is False and missing.error.kind == "validation"
    assert "does not exist" in missing.error.message
    # no half-written case may remain
    count = run_query(
        graph, "MATCH (c:Case {id: $id}) RETURN count(c)",
        {"id": f"CASE-{FANIN_ALERT}"}, read_only=True,
    ).result_set[0][0]
    assert int(count) == 0

    self_ref = create_case(
        graph, CYCLE_ALERT, reuses_case_ids=[CYCLE_CASE], settings=sandbox_settings,
    )
    assert self_ref.ok is False
    assert "cannot reuse itself" in self_ref.error.message


def test_scenario_h_flag_entity_accepts_accounts_and_customers(
    sandbox_settings, action_net
):
    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)

    account = flag_entity(graph, CYCLE_CASE, "TNC1", "account under review",
                          settings=sandbox_settings)
    assert account.ok is True
    assert account.result["kind"] == "Account"

    customer = flag_entity(graph, CYCLE_CASE, "TCU_C1", "customer under review",
                           risk="high", settings=sandbox_settings)
    assert customer.ok is True
    assert customer.result["kind"] == "Customer"
    stored = run_query(
        graph, "MATCH (n:Customer {id: $id}) RETURN n.flagged_for_review",
        {"id": "TCU_C1"}, read_only=True,
    ).result_set[0][0]
    assert stored is True

    missing = flag_entity(graph, CYCLE_CASE, "TN-NOPE", "x",
                          settings=sandbox_settings)
    assert missing.ok is False and "does not exist" in missing.error.message


def test_scenario_h_close_case_requires_evidence_for_suspicion(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    # BENIGN_ALERT produces a likely-benign, single-record story: it must never
    # be closable as a supported suspicion.
    result = _prepare_case(sandbox_settings, graph, BENIGN_ALERT, trace_dir)
    case_id = result.case_id
    refused = close_case(
        graph, case_id, verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
        summary="unbacked claim", settings=sandbox_settings,
    )
    assert refused.ok is False
    assert "does not satisfy the escalation conditions" in refused.error.message
    status = run_query(
        graph, "MATCH (c:Case {id: $id}) RETURN c.status",
        {"id": case_id}, read_only=True,
    ).result_set[0][0]
    assert status == "open", "a refused close must not touch the case"

    # an unsupported suspicion claim is still allowed to close as its honest
    # verdict
    allowed = close_case(
        graph, case_id, verdict=VERDICT_LIKELY_BENIGN,
        summary="no forwarding concentration observed", settings=sandbox_settings,
    )
    assert allowed.ok is True
    assert allowed.result["status"] == "closed"


def test_scenario_h_close_case_accepts_evidence_backed_suspicion(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    out = close_case(
        graph, result.case_id, verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
        summary="two independent signal families corroborated", settings=sandbox_settings,
    )
    assert out.ok is True
    assert out.result["verdict"] == VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW

    row = run_query(
        graph, "MATCH (c:Case {id: $id}) RETURN c.status, c.verdict",
        {"id": result.case_id}, read_only=True,
    ).result_set[0]
    assert row[0] == "closed" and row[1] == VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW


def test_scenario_h_prior_verdict_is_not_treated_as_verified_truth(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    prior = prior_cases(graph, "TNC1")
    case = prior.result["cases"][0]
    assert "not settled truth" in prior.interpretation or "re-check" in prior.interpretation
    assert case["status"] == "open"


# ---- I: ground-truth isolation --------------------------------------------


def test_scenario_i_action_layer_never_touches_ground_truth():
    for name in ("actions.py", "server.py"):
        text = (REPO_ROOT / "src" / "tools" / name).read_text(encoding="utf-8")
        assert "ground_truth" not in text, name
    for name in ("policy.py", "workflow.py"):
        text = (REPO_ROOT / "src" / "agent" / name).read_text(encoding="utf-8")
        assert "ground_truth" not in text, name
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "src.seed" not in stripped, f"{name}: {stripped}"


def test_scenario_i_persisted_nodes_carry_no_ground_truth(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    rows = run_query(
        graph,
        "MATCH (n) WHERE n:Case OR n:Evidence OR n:Simulation OR n:Action "
        "UNWIND keys(n) AS k RETURN DISTINCT k",
        read_only=True,
    ).result_set
    keys = {r[0] for r in rows}
    for forbidden in ("is_fraud", "pattern_type", "expected_account_members",
                      "ground_truth", "label", "family"):
        assert forbidden not in keys


def test_scenario_i_case_state_has_no_manifest_access(
    sandbox_settings, action_net, trace_dir
):
    result = _prepare_case(sandbox_settings, action_net, CYCLE_ALERT, trace_dir)
    payload = result.state.model_dump_json().lower()
    for marker in ("ground_truth", "is_fraud", "pattern_type",
                   "expected_account_members"):
        assert marker not in payload


# ---- J: post-action verification and unrelated topology --------------------


def test_scenario_j_verified_freeze_preserves_unrelated_topology(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    accounts_before = {
        r[0]: r[1] for r in run_query(
            graph, "MATCH (a:Account) RETURN a.id, a.status", read_only=True
        ).result_set
    }
    txns_before = int(run_query(graph, "MATCH (t:Transaction) RETURN count(t)",
                                read_only=True).result_set[0][0])
    edges_before = int(run_query(graph, "MATCH ()-[r:TRANSFERRED_TO]->() RETURN count(r)",
                                 read_only=True).result_set[0][0])
    active_before = _active_edges(graph)

    result = _workflow(QUAL_ALERT, sandbox_settings, trace_dir)
    assert result.frozen is True
    assert _result_of(result, "verify_freeze")["verified"] is True

    # the frozen account changed; every OTHER account is untouched
    accounts_after = {
        r[0]: r[1] for r in run_query(
            graph, "MATCH (a:Account) RETURN a.id, a.status", read_only=True
        ).result_set
    }
    changed = {k for k in accounts_before if accounts_before[k] != accounts_after.get(k)}
    assert changed == {"TNQ1"}
    assert accounts_after["TNQ1"] == "frozen"

    # historical data preserved: transactions and edges still exist
    assert int(run_query(graph, "MATCH (t:Transaction) RETURN count(t)",
                         read_only=True).result_set[0][0]) == txns_before
    assert int(run_query(graph, "MATCH ()-[r:TRANSFERRED_TO]->() RETURN count(r)",
                         read_only=True).result_set[0][0]) == edges_before
    # exactly the incident edges were deactivated, not deleted
    assert _active_edges(graph) == active_before - 4


def test_scenario_j_verification_notices_post_action_routing_drift(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _workflow(QUAL_ALERT, sandbox_settings, trace_dir)
    assert result.frozen is True
    assert verify_freeze(graph, QUAL_CASE, "TNQ1",
                         settings=sandbox_settings).result["verified"] is True

    # reactivate one deactivated edge behind the action's back (test tampering)
    run_query(
        graph,
        "MATCH ()-[r:TRANSFERRED_TO {active: false}]->() "
        "WITH r LIMIT 1 SET r.active = true",
        read_only=False,
    )
    drifted = verify_freeze(graph, QUAL_CASE, "TNQ1", settings=sandbox_settings)
    assert drifted.result["checks"]["routing_edge_count_matches_simulation"] is False
    assert drifted.result["verified"] is False
    assert "FAILED" in drifted.interpretation


# ---- K: a failed action changes nothing ------------------------------------


def test_scenario_k_nonexistent_account_is_declined_without_mutation(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, CYCLE_ALERT, trace_dir)
    simulate_freeze(graph, CYCLE_CASE, "TNC1", settings=sandbox_settings)
    before = _active_edges(graph)
    out = freeze_account(
        graph, CYCLE_CASE, "TNC-does-not-exist", "unknown account",
        eligibility_inputs={"case_state": result.state}, settings=sandbox_settings,
    )
    assert out.ok is True
    assert out.result["mutation_performed"] is False
    assert any("does not exist" in r for r in out.result["policy"]["blocking_reasons"])
    assert _active_edges(graph) == before
    assert int(run_query(graph, "MATCH (a:Action) RETURN count(a)",
                         read_only=True).result_set[0][0]) == 0


def test_scenario_k_non_sandbox_graph_is_refused():
    from src.config import Settings, load_settings

    settings = load_settings()
    unsafe = Settings(**{
        **settings.model_dump(),
        "graph_name": "customer_production",
        "habeas_demo_mode": False,
        "habeas_enable_actions": True,
    })
    out = freeze_account(
        object(), "CASE-X", "ACC-1", "attempt",
        eligibility_inputs={}, settings=unsafe,
    )
    assert out.ok is False
    assert out.error.kind == "validation"
    assert "not the disposable sandbox" in out.error.message


def test_scenario_k_disabled_actions_are_refused(settings, action_net):
    graph = action_net
    disabled = settings.model_copy(update={"habeas_enable_actions": False})
    out = create_case(graph, CYCLE_ALERT, settings=disabled)
    assert out.ok is False
    assert "disabled" in out.error.message


def test_scenario_k_missing_case_state_is_refused(sandbox_settings, action_net):
    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)
    out = freeze_account(graph, CYCLE_CASE, "TNC1", "no evidence supplied",
                         eligibility_inputs={}, settings=sandbox_settings)
    assert out.ok is False
    assert "case_state is required" in out.error.message


# ---- L: duplicate / concurrent attempts -----------------------------------


def test_scenario_l_duplicate_action_attempts_apply_once(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    first = _workflow(QUAL_ALERT, sandbox_settings, trace_dir)
    assert first.frozen is True
    outcomes = [
        freeze_account(graph, QUAL_CASE, "TNQ1", f"duplicate {i}",
                       eligibility_inputs={"case_state": first.state},
                       settings=sandbox_settings).result["action_status"]
        for i in range(3)
    ]
    assert set(outcomes) == {"already_applied"}
    assert int(run_query(graph, "MATCH (a:Action) RETURN count(a)",
                         read_only=True).result_set[0][0]) == 1
    assert _account_status(graph, "TNQ1") == "frozen"
    # no OTHER account was frozen: only TNQ1 plus the control account TNFZ,
    # which the fixture seeds as frozen on purpose
    frozen_now = {
        r[0] for r in run_query(
            graph, "MATCH (a:Account) WHERE a.status = 'frozen' RETURN a.id",
            read_only=True,
        ).result_set
    }
    assert frozen_now == {"TNQ1", "TNFZ"}


def test_scenario_l_concurrent_freeze_attempts_stay_consistent(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _prepare_case(sandbox_settings, graph, QUAL_ALERT, trace_dir)
    simulate_freeze(graph, QUAL_CASE, "TNQ1", settings=sandbox_settings)

    async def attempt(i: int):
        return await asyncio.to_thread(
            freeze_account, graph, QUAL_CASE, "TNQ1", f"concurrent {i}",
            eligibility_inputs={"case_state": result.state}, settings=sandbox_settings,
            evidence_ids=[], requested_by="test",
        )

    async def run():
        return await asyncio.gather(*(attempt(i) for i in range(3)))

    outs = asyncio.run(run())
    statuses = [o.result["action_status"] for o in outs]
    assert all(o.ok for o in outs)
    assert "declined" not in statuses
    assert statuses.count("applied") <= 1
    assert int(run_query(graph, "MATCH (a:Action) RETURN count(a)",
                         read_only=True).result_set[0][0]) == 1
    assert _account_status(graph, "TNQ1") == "frozen"
    assert verify_freeze(graph, QUAL_CASE, "TNQ1",
                         settings=sandbox_settings).result["verified"] is True