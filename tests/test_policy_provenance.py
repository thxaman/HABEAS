"""Corrective Phase 5 tests: provenance-aware action eligibility (policy v2).

Covers the corrective-pass acceptance cases (PLAN.md, "Corrective Phase 5
pass"):

A. overlapping circulation + rapid families share transactions and count once;
B. identical observations are deduplicated, never double-counted;
C. partial overlap is handled by the documented rule;
D. genuinely distinct, target-relevant families can qualify;
E. a signal that belongs only to a sibling account cannot authorize freezing
   the proposed target;
F. target-specific signals are still rejected when other conditions fail;
G. missing ids / failed queries / truncation / unresolved hypotheses are not
   bypassed by provenance counting;
H. the policy is independent of LLM confidence and of ground truth;
I. an eligible sandbox action persists its exact evidence ids and its policy
   version (hardened action record);
J. an ineligible action — and forged, malformed or cross-case evidence
   references — performs no mutation;
K. the historical CASE-ALR-0001 record and the Phase 6 report stay intact;
L. the verifier distinguishes an authorized post-action change from tampering.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agent import analysis, policy
from src.agent.llm import FakeLLM
from src.agent.state import CaseState, EvidenceRecord, H_BENIGN, Hypothesis
from src.agent.workflow import run_case_workflow
from src.db import run_query
from src.report import build_report
from src.report.verifier import (
    STATUS_CHANGED,
    STATUS_MISMATCH,
    load_case_records,
    verify_case,
)
from src.tools.actions import freeze_account, simulate_freeze

TARGET = "ACC-TGT"      # the proposed freeze target
SIBLING = "ACC-SIB"     # a related but different account

_RING = ["ACC-TGT", "ACC-R2", "ACC-R3", "ACC-R4"]
_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---- synthetic builders -----------------------------------------------------


def _ts(minutes: int) -> str:
    return (_BASE + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _leg(src: str, dst: str, txn: str, minutes: int, amount: float = 600.0) -> dict:
    return {"src": src, "dst": dst, "txn_id": txn, "ts": _ts(minutes),
            "amount": amount}


def _ring_events(rounds: int = 3, gap: int = 120) -> list[dict]:
    events = []
    t = 0
    for r in range(rounds):
        for j in range(4):
            t += gap
            events.append({
                "src": _RING[j], "dst": _RING[(j + 1) % 4],
                "txn_id": f"C{r}{j}", "ts": _ts(t), "amount": 500.0 + j,
            })
    return events


def _cycle_record(evidence_id: str = "EV-C", rounds: int = 3) -> EvidenceRecord:
    events = _ring_events(rounds)
    return EvidenceRecord(
        evidence_id=evidence_id,
        tool="detect_cycle",
        stage="TEST",
        ok=True,
        query_ids=["detect_closed_paths_v1"],
        truncated=[False],
        derived={
            "corroborated_count": 1,
            "cycles": [{
                "members": list(_RING),
                "temporally_corroborated": True,
                "event_count": len(events),
                "legs": [{"src": e["src"], "dst": e["dst"]} for e in events],
                "events": events,
            }],
        },
    )


def _rapid_legs(prefix: str = "R", start: int = 600, gap: int = 30,
                amount: float = 600.0) -> list[dict]:
    return [
        _leg("ACC-FEED", TARGET, f"{prefix}-IN", start, amount),
        _leg(TARGET, "ACC-EXIT", f"{prefix}-OUT", start + gap, amount * 0.95),
    ]


def _trace_record(evidence_id: str = "EV-T", legs: list[dict] | None = None
                  ) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        tool="trace_funds",
        stage="TEST",
        ok=True,
        query_ids=["trace_paths_out_v1"],
        truncated=[False],
        derived={"account_id": TARGET, "direction": "out",
                 "leg_events": legs if legs is not None else _rapid_legs()},
    )


def _alert_context_record(evidence_id: str = "EV-A",
                          entities: tuple[str, ...] = (TARGET, "ACC-OTHER")
                          ) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id, tool="alert_context", stage="TEST", ok=True,
        derived={"found": True, "alert_id": "ALRT-TEST",
                 "entity_ids": list(entities)},
    )


def _fan_record(evidence_id: str = "EV-F", account: str = TARGET
                ) -> EvidenceRecord:
    """Aggregate-only family: rollups without individual transaction ids."""
    return EvidenceRecord(
        evidence_id=evidence_id, tool="fan_in_out", stage="TEST", ok=True,
        derived={
            "account_id": account,
            "distinct_senders": 4, "distinct_receivers": 1,
            "in_amount": 4000.0, "out_amount": 3600.0,
            "out_in_ratio": 0.9, "top_receiver": "ACC-HUB",
            "top_receiver_share": 1.0, "direction": "in",
        },
    )


def _benign_ruled_out() -> list[Hypothesis]:
    return [Hypothesis(name=H_BENIGN, status="contradicted",
                       contradicting_evidence=["EV-C"])]


def _state(*records: EvidenceRecord, primary: tuple[str, ...] = (TARGET,),
           hypotheses: list[Hypothesis] | None = None) -> CaseState:
    return CaseState(
        alert_id="ALRT-TEST",
        primary_entity_ids=list(primary),
        hypotheses=hypotheses if hypotheses is not None else _benign_ruled_out(),
        evidence=list(records),
    )


def _basic_eligible_state(rapid_legs: list[dict] | None = None,
                          extra_records: list[EvidenceRecord] | None = None,
                          primary: tuple[str, ...] = (TARGET,),
                          hypotheses: list[Hypothesis] | None = None) -> CaseState:
    records = [
        _cycle_record("EV-C"),
        _trace_record("EV-T", legs=rapid_legs),
        _alert_context_record("EV-A"),
    ]
    records.extend(extra_records or [])
    return _state(*records, primary=primary, hypotheses=hypotheses)


def _decision(state: CaseState, target: str = TARGET, **overrides):
    kwargs = {
        "verdict": policy.VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
        "target_account_id": target,
        "simulation": {"id": "SIM-TEST", "simulation_complete": True},
        "simulation_stale": False,
        "account_exists": True,
        "account_already_frozen": False,
        "sandbox_enabled": True,
        "graph_is_demo": True,
        "actions_enabled": True,
    }
    kwargs.update(overrides)
    return policy.evaluate_freeze_eligibility(state, **kwargs)


# ---- A–H: the eligibility policy itself (pure, no database) -----------------


def test_a_overlapping_families_count_once():
    """Rapid legs drawn from ring transactions: one counted family."""
    overlap = [
        _leg("ACC-R4", TARGET, "C13", 600, 600.0),    # ring txn R4->TGT
        _leg(TARGET, "ACC-R2", "C20", 630, 590.0),    # ring txn TGT->R2
    ]
    state = _basic_eligible_state(rapid_legs=overlap)
    decision = _decision(state)

    assert set(decision.signal_families) == {"circulation", "rapid_forwarding"}
    assert decision.target_relevant_families == ["circulation", "rapid_forwarding"]
    assert decision.checks["enough_signal_families"] is True  # names alone agree
    # ...but the observations are the same transactions: ONE counted family.
    assert decision.distinct_provenance_families == ["circulation"]
    assert decision.checks["enough_distinct_provenance_families"] is False
    assert decision.allowed is False
    pair = decision.provenance["pairs"][0]
    assert pair["classification"] == analysis.OVERLAP_SHARED
    assert set(pair["shared_txn_ids"]) == {"C13", "C20"}
    blocking = "; ".join(decision.blocking_reasons)
    assert "distinct-provenance" in blocking
    assert "shared transaction id" in blocking


def test_b_identical_observations_are_deduplicated_not_double_counted():
    """Identical re-observations dedupe; an identical set counts once."""
    legs = _rapid_legs("R", start=600)
    cycle = EvidenceRecord(
        evidence_id="EV-C", tool="detect_cycle", stage="TEST", ok=True,
        derived={"corroborated_count": 1, "cycles": [{
            "members": ["ACC-FEED", TARGET],
            "temporally_corroborated": True,
            "event_count": 2,
            "legs": [{"src": "ACC-FEED", "dst": TARGET},
                     {"src": TARGET, "dst": "ACC-FEED"}],
            "events": [
                {"src": "ACC-FEED", "dst": TARGET, "txn_id": "R-IN",
                 "ts": _ts(600), "amount": 600.0},
                {"src": TARGET, "dst": "ACC-FEED", "txn_id": "R-OUT",
                 "ts": _ts(630), "amount": 590.0},
            ],
        }]},
    )
    state = _state(
        _trace_record("EV-T1", legs=legs),
        _trace_record("EV-T2", legs=[dict(leg) for leg in legs]),  # re-observed
        cycle,
        _alert_context_record(),
    )
    decision = _decision(state)

    # one rapid family, and its observation set is the deduplicated pair
    assert decision.signal_families == ["circulation", "rapid_forwarding"]
    assert decision.provenance["sources"]["rapid_forwarding"]["txn_ids"] == [
        "R-IN", "R-OUT"]
    pair = decision.provenance["pairs"][0]
    assert pair["classification"] == analysis.OVERLAP_SHARED
    assert pair["shared_count"] == 2
    assert decision.distinct_provenance_families == ["circulation"]
    assert decision.allowed is False


def test_c_partial_overlap_is_handled_by_the_documented_rule():
    """Some shared transaction ids: the pair contributes one counted family."""
    partial = [
        _leg("ACC-R4", TARGET, "C13", 1000, 600.0),     # ring txn
        _leg(TARGET, "ACC-EXIT", "R-OUT", 1030, 590.0),  # not a ring txn
    ]
    state = _basic_eligible_state(rapid_legs=partial)
    decision = _decision(state)

    pair = decision.provenance["pairs"][0]
    assert pair["classification"] == analysis.OVERLAP_PARTIAL
    assert pair["shared_txn_ids"] == ["C13"]
    assert pair["shared_count"] == 1
    assert decision.distinct_provenance_families == ["circulation"]
    assert decision.allowed is False
    assert any("overlap" in reason for reason in decision.blocking_reasons)


def test_d_distinct_target_relevant_families_can_qualify():
    """Two genuinely distinct, target-implicating families authorize the freeze."""
    state = _basic_eligible_state()
    decision = _decision(state)

    assert decision.allowed is True
    assert decision.distinct_provenance_families == [
        "circulation", "rapid_forwarding"]
    assert decision.checks["enough_distinct_provenance_families"] is True
    assert decision.checks["signal_families_implicate_target"] is True
    assert decision.qualifying_evidence_ids == ["EV-C", "EV-T", "EV-A"]
    pair = decision.provenance["pairs"][0]
    assert pair["classification"] == analysis.OVERLAP_DISTINCT
    assert pair["shared_count"] == 0
    assert decision.policy_version == policy.POLICY_VERSION == 2
    assert "distinct-provenance signal families" in decision.summary()


def test_e_sibling_only_signal_cannot_authorize_the_target():
    """A family whose observations concern a sibling account is excluded."""
    sibling_legs = [
        _leg("ACC-F2", SIBLING, "S-IN", 1200, 500.0),
        _leg(SIBLING, "ACC-EXIT2", "S-OUT", 1230, 490.0),
    ]
    # only the sibling's pass-through was observed; nothing crosses the target
    state = _state(
        _cycle_record("EV-C"),
        _trace_record("EV-TS", legs=sibling_legs),
        _alert_context_record(),
        primary=(TARGET, SIBLING),
    )
    decision = _decision(state)

    assert set(decision.signal_families) == {"circulation", "rapid_forwarding"}
    assert decision.target_relevant_families == ["circulation"]
    assert decision.checks["signal_families_implicate_target"] is True
    assert decision.distinct_provenance_families == ["circulation"]
    assert decision.allowed is False
    excluded = {item["family"]: item["reason"]
                for item in decision.provenance["excluded"]}
    assert "rapid_forwarding" in excluded
    assert SIBLING in excluded["rapid_forwarding"]
    blocking = "; ".join(decision.blocking_reasons)
    assert "was not counted" in blocking
    assert SIBLING in blocking


def test_f_target_relevant_families_do_not_bypass_other_conditions():
    """Provenance counting never overrides the remaining safeguards."""
    # simulation incomplete
    blocked = _decision(
        _basic_eligible_state(),
        simulation={"id": "SIM-TEST", "simulation_complete": False},
    )
    assert blocked.checks["enough_distinct_provenance_families"] is True
    assert blocked.allowed is False
    assert any("incomplete" in reason for reason in blocked.blocking_reasons)

    # unresolved benign hypothesis
    unresolved = _decision(_basic_eligible_state(
        hypotheses=[Hypothesis(name=H_BENIGN, status="open")],
    ))
    assert unresolved.allowed is False
    assert unresolved.checks["benign_explanation_ruled_out"] is False
    assert any("benign explanation" in reason
               for reason in unresolved.blocking_reasons)


def test_g_provenance_counting_cannot_bypass_safeguards():
    # (1) a failed query leaves an evidence gap
    failed = EvidenceRecord(evidence_id="EV-X", tool="detect_cycle",
                            stage="TEST", ok=False,
                            interpretation="query timed out")
    blocked = _decision(_basic_eligible_state(extra_records=[failed]))
    assert blocked.allowed is False
    assert blocked.checks["no_failed_queries"] is False
    assert any("failed" in reason for reason in blocked.blocking_reasons)

    # (2) material truncation blocks even though two distinct families count
    truncated = _cycle_record("EV-C2")
    truncated.truncated = [True]
    gated = _state(truncated, _trace_record("EV-T"), _alert_context_record())
    blocked = _decision(gated)
    assert blocked.checks["no_material_truncation"] is False
    assert len(blocked.distinct_provenance_families) == 2
    assert blocked.allowed is False
    assert any("truncated" in reason for reason in blocked.blocking_reasons)

    # (3) verdict outside the review gate
    blocked = _decision(_basic_eligible_state(),
                        verdict=policy.VERDICT_LIKELY_BENIGN)
    assert blocked.checks["verdict_supports_review"] is False
    assert blocked.allowed is False

    # (4) an aggregate-only family can never be the second counted family,
    #     even when it implicates the target
    fan = _fan_record("EV-F")
    blocked = _decision(_state(
        _cycle_record("EV-C"), fan, _alert_context_record()))
    assert "fan_in_out" in blocked.signal_families
    assert "fan_in_out" in blocked.target_relevant_families
    assert blocked.distinct_provenance_families == ["circulation"]
    assert blocked.allowed is False
    blocking = "; ".join(blocked.blocking_reasons)
    assert "no transaction-level observations" in blocking
    assert "independence cannot be established" in blocking


def test_h_policy_is_independent_of_llm_confidence_and_ground_truth():
    policy_source = inspect.getsource(policy)
    analysis_source = inspect.getsource(analysis)
    # the policy can never read the manifest or any LLM surface
    assert "ground_truth" not in policy_source
    assert "ground_truth" not in analysis_source
    assert "src.agent.llm" not in policy_source
    assert ".confidence" not in policy_source

    # narrative confidence text changes nothing
    state = _basic_eligible_state()
    first = _decision(state)
    for record in state.evidence:
        record.interpretation = ("LLM confidence: 0.99 - definitely fraud, "
                                 "ground truth says suspicious")
    second = _decision(state)
    assert first.model_dump() == second.model_dump()
    assert second.allowed is True
    # determinism: same evidence, same decision
    assert _decision(state).model_dump() == second.model_dump()


# ---- I–L: persisted actions, mutations, history, verification ---------------


@pytest.fixture(scope="module")
def trace_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("traces_policy_v2")


def _workflow(alert: str, settings, trace_dir, **kwargs):
    return asyncio.run(run_case_workflow(
        alert, settings=settings, llm=FakeLLM(), trace_dir=trace_dir, **kwargs
    ))


def _account_status(graph, account_id):
    row = run_query(graph, "MATCH (a:Account {id: $id}) RETURN a.status",
                    {"id": account_id}, read_only=True).result_set
    return row[0][0] if row else None


def _active_edges(graph):
    return int(run_query(graph, "MATCH ()-[r:TRANSFERRED_TO]->() "
                                "WHERE r.active = true RETURN count(r)",
                         read_only=True).result_set[0][0])


def test_i_eligible_action_persists_exact_evidence_and_policy_version(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _workflow("ALRT-QUAL", sandbox_settings, trace_dir)
    assert result.frozen is True

    row = run_query(
        graph, "MATCH (a:Action {id: $id}) RETURN properties(a)",
        {"id": "ACT-CASE-ALRT-QUAL-TNQ1"}, read_only=True,
    ).result_set
    assert row, "the applied action record must exist"
    props = dict(row[0][0])

    assert props["case_id"] == "CASE-ALRT-QUAL"
    assert props["account_id"] == "TNQ1"
    assert props["status"] == "applied"

    evidence_ids = json.loads(props["evidence_ids_json"])
    assert evidence_ids == [
        "CASE-ALRT-QUAL:EV-0001", "CASE-ALRT-QUAL:EV-0003",
        "CASE-ALRT-QUAL:EV-0005", "CASE-ALRT-QUAL:EV-0006",
    ]
    stored = run_query(
        graph,
        "MATCH (c:Case {id: $c})-[:HAS_EVIDENCE]->(e:Evidence) "
        "RETURN collect(e.id)",
        {"c": "CASE-ALRT-QUAL"}, read_only=True,
    ).result_set[0][0]
    assert set(evidence_ids) <= set(stored), (
        "every persisted evidence reference must exist as case evidence"
    )

    recorded_policy = json.loads(props["policy_json"])
    assert recorded_policy["policy_version"] == 2
    assert recorded_policy["allowed"] is True
    assert recorded_policy["distinct_provenance_families"] == [
        "circulation", "rapid_forwarding"]
    assert recorded_policy["provenance"]["pairs"][0]["classification"] == \
        analysis.OVERLAP_DISTINCT

    verification = json.loads(props["verification_json"])
    assert verification["verified"] is True
    assert verification["checks"]["action_record_present"] is True


def test_j_ineligible_actions_and_forged_references_change_nothing(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net

    # (1) overlapping-evidence case: the freeze is declined, nothing written
    declined = _workflow("ALRT-CYCLE", sandbox_settings, trace_dir)
    assert declined.freeze_declined is True
    assert declined.frozen is False
    assert int(run_query(graph, "MATCH (a:Action) RETURN count(a)",
                         read_only=True).result_set[0][0]) == 0
    assert _account_status(graph, "TNC1") == "active"

    # (2) forged / malformed / cross-case evidence references on an otherwise
    #     eligible case: rejected fail-closed, no mutation
    prepared = _workflow("ALRT-QUAL", sandbox_settings, trace_dir, freeze=False)
    simulate_freeze(graph, "CASE-ALRT-QUAL", "TNQ1", settings=sandbox_settings)
    edges_before = _active_edges(graph)
    attempts = [
        ("CASE-ALRT-QUAL:EV-9999", "does not back the qualifying evidence"),
        ("not a valid id!!", "malformed"),
        ("CASE-OTHER:EV-0001", "not linked to case"),
    ]
    for bad_id, marker in attempts:
        out = freeze_account(
            graph, "CASE-ALRT-QUAL", "TNQ1", "forged references",
            eligibility_inputs={"case_state": prepared.state},
            settings=sandbox_settings, evidence_ids=[bad_id],
        )
        assert out.ok is False, f"forged reference {bad_id!r} was accepted"
        assert marker in out.error.message
    assert _active_edges(graph) == edges_before
    assert _account_status(graph, "TNQ1") == "active"
    assert int(run_query(graph, "MATCH (a:Action) RETURN count(a)",
                         read_only=True).result_set[0][0]) == 0


def test_l_verifier_distinguishes_authorized_change_from_tampering(
    sandbox_settings, action_net, trace_dir
):
    graph = action_net
    result = _workflow("ALRT-QUAL", sandbox_settings, trace_dir)
    assert result.frozen is True
    case_id = result.case_id

    # the ring query legitimately changed: attributed to the recorded action
    claims = {c.query_id: c for c in verify_case(graph, case_id).claims}
    assert claims["detect_closed_paths_v1"].status == STATUS_CHANGED
    assert "ACT-CASE-ALRT-QUAL-TNQ1" in claims["detect_closed_paths_v1"].reason

    # tampering with immutable history is a mismatch, never the action
    records = load_case_records(graph, case_id)
    alert_record = next(r for r in records.evidence
                        if r["query_id"] == "alert_context_v1")
    run_query(
        graph, "MATCH (e:Evidence {id: $id}) SET e.result_hash = $hash",
        {"id": alert_record["evidence_id"], "hash": "sha256:" + "0" * 64},
        read_only=False,
    )
    tampered = {c.evidence_id: c for c in verify_case(graph, case_id).claims}[
        alert_record["evidence_id"]]
    assert tampered.status == STATUS_MISMATCH
    assert "consistent with the recorded, authorized action" not in tampered.reason


def test_legacy_policy_versions_render_a_historical_note_not_a_redecision(
    sandbox_settings, action_net, trace_dir
):
    """A v1-era authorization is rendered as recorded, never as a v2 outcome.

    The applied action is preserved verbatim; only its recorded policy_version
    is set to 1 to simulate a legacy record (the live v1 record itself lives in
    reports/audit/, see test_k).
    """
    graph = action_net
    result = _workflow("ALRT-QUAL", sandbox_settings, trace_dir)
    assert result.frozen is True

    row = run_query(graph, "MATCH (a:Action) RETURN a.id, a.policy_json",
                    read_only=True).result_set[0]
    action_id, raw = row[0], row[1]
    recorded = json.loads(raw)
    recorded["policy_version"] = 1
    run_query(
        graph, "MATCH (a:Action {id: $id}) SET a.policy_json = $p",
        {"id": action_id, "p": json.dumps(recorded)}, read_only=False,
    )

    artifact = build_report(graph, result.case_id, state=result.state)
    markdown = artifact.markdown
    assert artifact.payload["intervention"]["policy_source"] == (
        "persisted Action policy_json (historical authorization)")
    assert "Recorded policy version: 1 (current policy: 2)." in markdown
    assert "preserved exactly as recorded" in markdown
    assert "re-presented as a current-policy outcome" in markdown
    # the report must not claim the current policy produced this decision
    assert "Eligibility outcome" in markdown
    assert "allowed = True" in markdown


def test_k_historical_alr0001_record_and_reports_stay_intact():
    repo = Path(__file__).resolve().parents[1]

    report = (repo / "reports" / "CASE-ALR-0001.md").read_text(encoding="utf-8")
    assert "- Disposition: freeze_applied" in report
    assert "ACT-CASE-ALR-0001-ACC-0387" in report
    assert "6 shared transaction id(s)" in report
    # the historical report predates policy-version rendering: it must not have
    # been regenerated by the corrected builder
    assert "Recorded policy version" not in report

    audit = json.loads(
        (repo / "reports" / "audit" / "CASE-ALR-0001-action-audit.json")
        .read_text(encoding="utf-8"))
    assert audit["action_id"] == "ACT-CASE-ALR-0001-ACC-0387"
    assert audit["action_node_properties"]["status"] == "applied"
    parsed = audit["action_json_fields_parsed"]
    assert parsed["policy"]["policy_version"] == 1  # recorded verbatim
    findings = audit["review_findings"]
    assert findings["record_unchanged"] is True
    assert findings["policy_version_at_authorization"] == 1
    assert findings["current_policy_version"] == policy.POLICY_VERSION
    assert findings["could_authorize_under_v2"] is False
    overlap = findings["overlap_finding"]
    assert overlap["classification"] == "shared_observations"
    assert overlap["shared_count"] == 6
    assert set(overlap["shared_txn_ids"]) == {
        "TXN-001524", "TXN-001525", "TXN-001528",
        "TXN-001529", "TXN-001532", "TXN-001533",
    }
    assert overlap["families"] == ["circulation", "rapid_forwarding"]

    md = (repo / "reports" / "audit" / "CASE-ALR-0001-review-findings.md") \
        .read_text(encoding="utf-8")
    assert "not a re-decision" in md
    assert "policy v1" in md
    assert "could not authorize a freeze under policy" in md
