"""Phase 6.1 claim-level evidence verification (corrective pass) — adversarial
tests A–I (test J is the full suite).

These tests pin the corrective-pass semantics end to end over the isolated
test network, with real tool envelopes persisted through the real write layer:

- (A) a fan-in/fan-out claim is only ``reproduced_current_state`` when its
  complete source manifest (window anchor + senders + receivers) is verified
  AND the recorded derived aggregates are recomputed and match;
- (B) rapid-forwarding verification rests on the actual per-leg transaction
  event source, not only the path query;
- (C) circulation claims rest on the cycle topology query plus the separate
  temporal-corroboration event source;
- (D) tampering a secondary source is detected even though the first query /
  window anchor replays untouched; the claim is never reported verified from
  a subset;
- (E) a missing required source is refused at write time and is a mismatch at
  verify time; a pre-6.1 anchor-only legacy citation is never reproduced;
- (F) valid but irrelevant approved queries cannot establish a claim;
- (G) authorized post-action changes are attributed to the recorded action,
  while unexplained graph changes are mismatches;
- (H) absence of evidence is never reported as a contradiction, and the
  insufficiency hypothesis is contradicted only under the same
  distinct-provenance counting rule the action policy uses;
- (I) historical report artifacts are preserved byte-for-byte and corrected
  reports carry the claim-level source coverage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.agent import analysis
from src.agent import policy as action_policy
from src.agent.state import (
    H_BENIGN,
    H_INSUFFICIENT,
    CaseState,
    EvidenceRecord,
)
from src.agent.workflow import build_citations
from src.db import run_query
from src.report import verifier as V
from src.report.builder import build_report, write_report
from src.tools import query_catalog as qc
from src.tools.actions import (
    cite_evidence,
    create_case,
    freeze_account,
    simulate_freeze,
)
from src.tools.algorithms import detect_cycle
from src.tools.envelopes import run_evidence
from src.tools.reads import entity_profile, fan_in_out, trace_funds

pytestmark = pytest.mark.usefixtures("action_net")

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

# Historical artifacts (Phase 6 v2 reports, the Phase 5 corrective outputs and
# the action audit trail) are never rewritten by this pass; the corrected
# outputs go to a NEW directory.
PINNED_HISTORY = {
    "CASE-ALR-0001.md":
        "9a951b5f5865c0457f35022e9f2621d7cd9d30d57cccd500bbc590e297cb8af4",
    "CASE-ALR-0004.md":
        "156d4eaee8d3ba4ffeae14dce50d6129938b120bea21f95c951b45763d30d490",
    "corrective/CASE-ALR-0001.md":
        "d6859b2ed42cc2b80864716b3cabfbd7994063ccaf8e2970a28fb37167904949",
    "corrective/CASE-ALR-0004.md":
        "3f9e7773c937ae9f15dd2b76ff446e75024a7016566f92627897b8e80e9e43a6",
    "corrective/CASE-ALR-0007.md":
        "0910f8616ccfdf153ed259838c18cd2851eec299c426d2033d747cc3da3236f2",
    "corrective/CASE-ALR-0021.md":
        "5ef41e07478fd9fdcc9d2251e4848f4e7278e9c21fe26d4422337e50614728b5",
    "audit/CASE-ALR-0001-action-audit.json":
        "e9b8a35c9271352c95e2c30dab03ca682bd3b38d6f2149e85dc88251db65a967",
    "audit/CASE-ALR-0001-review-findings.md":
        "d74eea6aa27b813ceb16d432ecce7bc26437059a630dae65dcfdb01c4cdd3eb2",
}

ZERO_HASH = "sha256:" + "0" * 64


# ---- helpers -----------------------------------------------------------------


def _env_record(tool, envelope, target, evidence_id="EV-0001"):
    """One EvidenceRecord in the investigation loop's shape, from a REAL
    tool envelope (query ids, params, hashes, truncation, bounds, derived)."""
    data = envelope.model_dump()
    queries = data.get("query_evidence") or []
    deriver = analysis.DERIVERS.get(tool)
    return EvidenceRecord(
        evidence_id=evidence_id,
        tool=tool,
        stage="test",
        ok=bool(data.get("ok")),
        interpretation=data.get("interpretation") or "",
        query_ids=[q.get("query_id", "") for q in queries],
        params=[dict(q.get("params") or {}) for q in queries],
        rows_hash=[q.get("rows_hash", "") for q in queries],
        row_count=[int(q.get("row_count") or 0) for q in queries],
        truncated=[bool(q.get("truncated")) for q in queries],
        bounds=[dict(q.get("bound") or {}) for q in queries],
        entity_ids=[target],
        derived=(deriver(data, target) if deriver else {}),
    )


def _state(tool_envs, target, alert="ALRT-CLV"):
    records = [
        _env_record(tool, envelope, target, f"EV-{index:04d}")
        for index, (tool, envelope) in enumerate(tool_envs, start=1)
    ]
    return CaseState(
        alert_id=alert, primary_entity_ids=[target], evidence=records
    )


def _cite_state(graph, settings, state, target, alert, verdict=None):
    created = create_case(
        graph, alert, verdict=verdict, entity_ids=[target], settings=settings
    )
    assert created.ok, created.error
    case_id = f"CASE-{alert}"
    citations = build_citations(state, target)
    assert citations, "the state must yield material citations"
    out = cite_evidence(graph, case_id, citations, settings=settings)
    assert out.ok, out.error
    return case_id, out


def _claims(graph, case_id):
    verification = V.verify_case(graph, case_id)
    return {
        claim.evidence_id.split(":")[-1]: claim
        for claim in verification.claims
    }, verification


def _claim(graph, case_id, local_id):
    claims, _ = _claims(graph, case_id)
    assert local_id in claims, sorted(claims)
    return claims[local_id]


def _record(graph, case_id, local_id):
    records = V.load_case_records(graph, case_id)
    for record in records.evidence:
        if str(record.get("evidence_id", "")).split(":")[-1] == local_id:
            return record
    raise AssertionError(f"no persisted evidence {local_id!r} on {case_id}")


def _set_props(graph, case_id, local_id, **props):
    run_query(
        graph,
        "MATCH (e:Evidence {id: $id}) SET e += $props",
        {"id": f"{case_id}:{local_id}", "props": props},
        read_only=False,
    )


def _edit_sources(graph, case_id, local_id, mutate):
    record = _record(graph, case_id, local_id)
    sources = json.loads(record["sources_json"])
    mutate(sources)
    _set_props(graph, case_id, local_id, sources_json=json.dumps(sources))


# ---- A: fan-in/fan-out needs all sources AND the derived result --------------


def test_a_fan_in_out_requires_both_sources_and_the_derived_result(
    action_net, sandbox_settings
):
    graph = action_net
    envelope = fan_in_out(graph, "TNI1")
    assert envelope.ok, envelope.error
    state = _state([("fan_in_out", envelope)] * 3, "TNI1")
    case_id, out = _cite_state(graph, sandbox_settings, state, "TNI1", "ALRT-CVA")

    cited = out.result["citations"][0]
    assert cited["source_count"] == 3
    assert cited["source_query_ids"] == [
        "txn_max_ts_v1", "fan_in_senders_v1", "fan_out_receivers_v1",
    ]
    assert cited["required_source_queries"] == [
        "fan_in_senders_v1", "fan_out_receivers_v1",
    ]

    clean = _claim(graph, case_id, "EV-0001")
    assert clean.status == V.STATUS_REPRODUCED
    assert clean.compound is True
    assert clean.source_count == 3
    assert [s.query_id for s in clean.sources] == [
        "txn_max_ts_v1", "fan_in_senders_v1", "fan_out_receivers_v1",
    ]
    assert all(s.status == V.STATUS_REPRODUCED for s in clean.sources)
    assert clean.checked_source_queries == [
        "txn_max_ts_v1", "fan_in_senders_v1", "fan_out_receivers_v1",
    ]
    checks = {c.name: c for c in clean.checks}
    assert checks["derived_consistency"].ok is True

    # (1) a manifest that lost the receiver source can never be reported as a
    # fully reproduced fan-in/fan-out claim, even when the declared count and
    # the required list are edited to look consistent.
    _edit_sources(graph, case_id, "EV-0001", lambda sources: sources.pop(2))
    _set_props(
        graph, case_id, "EV-0001",
        source_count=2,
        required_source_queries_json=json.dumps(["fan_in_senders_v1"]),
    )
    dropped = _claim(graph, case_id, "EV-0001")
    assert dropped.status == V.STATUS_MISMATCH
    assert "fan_out_receivers_v1" in dropped.reason
    assert dropped.sources == []
    assert dropped.status != V.STATUS_REPRODUCED

    # (2) tampering the recorded derived aggregates is a mismatch
    derived = json.loads(_record(graph, case_id, "EV-0002")["derived_json"])
    derived["top_receiver_share"] = 0.01
    _set_props(
        graph, case_id, "EV-0002", derived_json=json.dumps(derived)
    )
    bad_derived = _claim(graph, case_id, "EV-0002")
    assert bad_derived.status == V.STATUS_MISMATCH
    assert "derived" in bad_derived.reason

    # (3) a full capture whose derived result is missing cannot be labeled
    # fully reproduced from the source rows alone
    _set_props(graph, case_id, "EV-0003", derived_json=json.dumps({}))
    no_derived = _claim(graph, case_id, "EV-0003")
    assert no_derived.status == V.STATUS_MISMATCH
    assert "derived" in no_derived.reason


# ---- B: rapid forwarding rests on the actual per-leg events ------------------


def test_b_rapid_forwarding_is_verified_from_actual_event_evidence(
    action_net, sandbox_settings
):
    graph = action_net
    out_env = trace_funds(graph, "TNQ1", "out", hops=2)
    in_env = trace_funds(graph, "TNQ1", "in", hops=2)
    assert out_env.ok and in_env.ok, (out_env.error, in_env.error)

    state = _state(
        [("trace_funds", out_env), ("trace_funds", in_env)], "TNQ1"
    )
    rapid = analysis.rapid_forwarding_for_account(state, "TNQ1")
    assert rapid["rapid_forward_count"] >= 1, (
        "the fixture must contain a genuine rapid pass-through on TNQ1"
    )

    case_id, out = _cite_state(
        graph, sandbox_settings, state, "TNQ1", "ALRT-CVB"
    )
    cited_out = out.result["citations"][0]
    assert "trace_paths_out_v1" in cited_out["source_query_ids"]
    assert "txn_events_for_legs_v1" in cited_out["source_query_ids"]
    assert cited_out["required_source_queries"] == [
        "trace_paths_out_v1", "txn_events_for_legs_v1",
    ]

    clean_out = _claim(graph, case_id, "EV-0001")
    assert clean_out.status == V.STATUS_REPRODUCED
    assert [s.query_id for s in clean_out.sources] == [
        "txn_max_ts_v1", "trace_paths_out_v1", "txn_events_for_legs_v1",
    ]

    # The rapid-forwarding claim rests on the per-leg transaction events: when
    # that source no longer matches, the claim is refused, even though the
    # path query and the window anchor still reproduce.
    def tamper_events(sources):
        for entry in sources:
            if entry["query_id"] == "txn_events_for_legs_v1":
                entry["result_hash"] = ZERO_HASH

    _edit_sources(graph, case_id, "EV-0001", tamper_events)
    broken = _claim(graph, case_id, "EV-0001")
    assert broken.status == V.STATUS_MISMATCH
    by_qid = {s.query_id: s.status for s in broken.sources}
    assert by_qid["txn_events_for_legs_v1"] == V.STATUS_MISMATCH
    assert by_qid["trace_paths_out_v1"] == V.STATUS_REPRODUCED
    assert by_qid["txn_max_ts_v1"] == V.STATUS_REPRODUCED

    # The untouched inbound-trace record on the same case is unaffected.
    clean_in = _claim(graph, case_id, "EV-0002")
    assert clean_in.status == V.STATUS_REPRODUCED


# ---- C: circulation needs topology + temporal corroboration ------------------


def test_c_cycle_claims_rest_on_topology_and_temporal_event_sources(
    action_net, sandbox_settings
):
    graph = action_net
    envelope = detect_cycle(graph, "TNC1")
    assert envelope.ok, envelope.error
    assert envelope.result["temporally_corroborated_count"] >= 1

    state = _state([("detect_cycle", envelope)], "TNC1")
    case_id, out = _cite_state(
        graph, sandbox_settings, state, "TNC1", "ALRT-CVC"
    )
    cited = out.result["citations"][0]
    assert cited["source_query_ids"] == [
        "detect_closed_paths_v1", "txn_max_ts_v1", "txn_events_for_legs_v1",
    ]
    assert cited["required_source_queries"] == [
        "detect_closed_paths_v1", "txn_events_for_legs_v1",
    ]

    clean = _claim(graph, case_id, "EV-0001")
    assert clean.status == V.STATUS_REPRODUCED
    assert [s.query_id for s in clean.sources] == [
        "detect_closed_paths_v1", "txn_max_ts_v1", "txn_events_for_legs_v1",
    ]
    assert {s.role for s in clean.sources} == {"window_anchor", "claim_source"}

    # Temporal corroboration is a separate source; losing it breaks the claim.
    def tamper_events(sources):
        for entry in sources:
            if entry["query_id"] == "txn_events_for_legs_v1":
                entry["result_hash"] = "sha256:" + "1" * 64

    _edit_sources(graph, case_id, "EV-0001", tamper_events)
    broken = _claim(graph, case_id, "EV-0001")
    assert broken.status == V.STATUS_MISMATCH
    assert "txn_events_for_legs_v1" in broken.mismatched_source_queries


# ---- D: secondary-source tampering under an unchanged first query ------------


def test_d_secondary_source_tampering_is_detected_from_any_subset(
    action_net, sandbox_settings
):
    graph = action_net
    envelope = fan_in_out(graph, "TNI1")
    state = _state([("fan_in_out", envelope)], "TNI1")
    case_id, _ = _cite_state(
        graph, sandbox_settings, state, "TNI1", "ALRT-CVD"
    )
    anchor_hash = _record(graph, case_id, "EV-0001")["result_hash"]

    def tamper_senders(sources):
        sources[1]["result_hash"] = "sha256:" + "2" * 64

    _edit_sources(graph, case_id, "EV-0001", tamper_senders)

    claim = _claim(graph, case_id, "EV-0001")
    assert claim.status == V.STATUS_MISMATCH
    # The window anchor (the record's first query) replays untouched...
    by_qid = {s.query_id: s.status for s in claim.sources}
    assert by_qid["txn_max_ts_v1"] == V.STATUS_REPRODUCED
    assert by_qid["fan_out_receivers_v1"] == V.STATUS_REPRODUCED
    # ...the edited secondary source is the mismatch, and the claim is never
    # reported as fully verified from the reproducing subset.
    assert by_qid["fan_in_senders_v1"] == V.STATUS_MISMATCH
    assert claim.mismatched_source_queries == ["fan_in_senders_v1"]
    assert claim.checked_source_queries == [
        "txn_max_ts_v1", "fan_out_receivers_v1",
    ]
    # Only the manifest entry was edited; the top-level mirror is untouched.
    assert _record(graph, case_id, "EV-0001")["result_hash"] == anchor_hash


# ---- E: missing required sources ---------------------------------------------


def test_e_missing_required_source_is_refused_at_write_and_is_mismatch_at_verify(
    action_net, sandbox_settings
):
    graph = action_net
    envelope = fan_in_out(graph, "TNI1")
    state = _state([("fan_in_out", envelope)], "TNI1")
    case_id, _ = _cite_state(
        graph, sandbox_settings, state, "TNI1", "ALRT-CVE"
    )

    # (1) write time: a citation that declares a required query it does not
    # carry is never persisted.
    citations = build_citations(state, "TNI1")
    bad = dict(citations[0])
    bad["evidence_id"] = "EV-BAD"
    bad["required_source_queries"] = [
        "fan_in_senders_v1", "fan_out_receivers_v1",
        "hub_weighted_in_degree_v1",
    ]
    refused = cite_evidence(graph, case_id, [bad], settings=sandbox_settings)
    assert refused.ok is False
    assert "missing from its source manifest" in refused.error.message
    claims, _ = _claims(graph, case_id)
    assert "EV-BAD" not in claims

    # (2) verify time: removing a required source record from the persisted
    # manifest is a mismatch (the record is never silently downgraded).
    _edit_sources(graph, case_id, "EV-0001", lambda sources: sources.pop(2))
    _set_props(graph, case_id, "EV-0001", source_count=2)
    broken = _claim(graph, case_id, "EV-0001")
    assert broken.status == V.STATUS_MISMATCH
    assert "fan_out_receivers_v1" in broken.reason
    assert "missing from the source manifest" in broken.reason
    assert broken.sources == []


def test_e2_legacy_anchor_only_citations_are_never_reproduced(
    action_net, sandbox_settings
):
    graph = action_net
    alert = "ALRT-CVE2"
    case_id = f"CASE-{alert}"
    created = create_case(graph, alert, entity_ids=["TNI1"], settings=sandbox_settings)
    assert created.ok, created.error

    anchor = run_evidence(
        graph, "txn_max_ts_v1", qc.get_template("txn_max_ts_v1"), {},
        row_cap=10,
    )
    cited = cite_evidence(graph, case_id, [{
        "evidence_id": "EV-LEGACY",
        "claim": "legacy pre-6.1 anchor-only capture",
        "stance": "supporting",
        "query_id": "txn_max_ts_v1",
        "params": anchor.params,
        "rows": anchor.rows,
        "columns": anchor.columns,
        "result_hash": anchor.rows_hash,
        "row_count": anchor.row_count,
        "truncated": anchor.truncated,
        "evidence_kind": "captured_snapshot",
        "source_tool": "fan_in_out",
    }], settings=sandbox_settings)
    assert cited.ok, cited.error

    # Emulate the pre-6.1 legacy shape: the manifest properties were never
    # written, so only the first query is recorded.
    run_query(
        graph,
        "MATCH (e:Evidence {id: $id}) REMOVE e.sources_json, e.source_count, "
        "e.required_source_queries_json",
        {"id": f"{case_id}:EV-LEGACY"},
        read_only=False,
    )

    claim = _claim(graph, case_id, "EV-LEGACY")
    assert claim.compound is False
    # The anchor replays, but a replay of the window anchor alone can never
    # carry a fan-in/fan-out claim.
    assert claim.status == V.STATUS_UNVERIFIABLE
    assert "pre-6.1 legacy record" in claim.reason
    checks = {c.name: c for c in claim.checks}
    assert checks["source_coverage"].ok is False


# ---- F: valid but irrelevant queries cannot establish a claim ----------------


def test_f_valid_but_irrelevant_queries_cannot_establish_the_claim(
    action_net, sandbox_settings
):
    graph = action_net
    fan_env = fan_in_out(graph, "TNI1")
    anchor = next(
        q for q in fan_env.query_evidence if q.query_id == "txn_max_ts_v1"
    )
    profile_env = entity_profile(graph, "TCU_I1")
    lookup = next(
        q for q in profile_env.query_evidence
        if q.query_id == "entity_profile_lookup_v1"
    )

    alert = "ALRT-CVF"
    case_id = f"CASE-{alert}"
    created = create_case(graph, alert, entity_ids=["TNI1"], settings=sandbox_settings)
    assert created.ok, created.error
    cited = cite_evidence(graph, case_id, [{
        "evidence_id": "EV-IRRELEVANT",
        "claim": "coordinated fan-in/fan-out cash-out concentration on TNI1",
        "stance": "supporting",
        "query_id": anchor.query_id,
        "params": anchor.params,
        "interpolation": {},
        "result_hash": anchor.rows_hash,
        "row_count": anchor.row_count,
        "truncated": anchor.truncated,
        "source_tool": "fan_in_out",
        "sources": [
            {
                "query_id": anchor.query_id,
                "params": anchor.params,
                "interpolation": {},
                "result_hash": anchor.rows_hash,
                "row_count": anchor.row_count,
                "truncated": anchor.truncated,
            },
            {
                "query_id": lookup.query_id,
                "params": lookup.params,
                "interpolation": {},
                "result_hash": lookup.rows_hash,
                "row_count": lookup.row_count,
                "truncated": lookup.truncated,
            },
        ],
        "required_source_queries": ["entity_profile_lookup_v1"],
    }], settings=sandbox_settings)
    assert cited.ok, cited.error

    claim = _claim(graph, case_id, "EV-IRRELEVANT")
    # Both recorded queries are approved and replayable, but neither of the
    # fan-in/fan-out tool's claim-bearing sources is present: the claim is
    # refused, not reported as verified from valid-looking stand-ins.
    assert claim.status == V.STATUS_MISMATCH
    assert "fan_in_senders_v1" in claim.reason
    assert "fan_out_receivers_v1" in claim.reason
    assert claim.sources == []


# ---- G: authorized vs unexplained post-action changes ------------------------


def test_g_authorized_and_unexplained_post_action_changes_are_distinguished(
    action_net, sandbox_settings
):
    graph = action_net

    # (1) An unexplained graph change: the replay differs and NO recorded
    # action explains it -> mismatch.
    cycle_env = detect_cycle(graph, "TNC1")
    state = _state([("detect_cycle", cycle_env)], "TNC1", alert="ALRT-CVG")
    case_id, _ = _cite_state(
        graph, sandbox_settings, state, "TNC1", "ALRT-CVG"
    )
    clean = _claim(graph, case_id, "EV-0001")
    assert clean.status == V.STATUS_REPRODUCED

    run_query(
        graph,
        "MATCH (:Account {id: 'TNC4'})-[r:TRANSFERRED_TO]->"
        "(:Account {id: 'TNC1'}) SET r.active = false",
        read_only=False,
    )
    unexplainable = _claim(graph, case_id, "EV-0001")
    assert unexplainable.status == V.STATUS_MISMATCH
    assert "no recorded action explains" in unexplainable.reason

    # (2) An authorized sandbox freeze on a different case: the difference is
    # attributed to the recorded action (changed_after_action), with the
    # unchanged historical sources listed as reproduced.
    envs = [
        ("detect_cycle", detect_cycle(graph, "TNQ1")),
        ("trace_funds", trace_funds(graph, "TNQ1", "out", hops=2)),
        ("trace_funds", trace_funds(graph, "TNQ1", "in", hops=2)),
    ]
    freeze_state = _state(envs, "TNQ1", alert="ALRT-CVQ")
    freeze_state.provisional_verdict = "suspicious_activity"
    freeze_state.hypotheses = analysis.recompute_hypotheses(freeze_state)
    # The recorded case verdict must be the ESCALATED canonical outcome (the
    # same value the real workflow records); the freeze policy never escalates
    # on its own say-so.
    escalated = action_policy.escalation_verdict(
        freeze_state, action_policy.VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION
    )
    assert escalated == action_policy.VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW
    case2, _ = _cite_state(
        graph, sandbox_settings, freeze_state, "TNQ1", "ALRT-CVQ",
        verdict=escalated,
    )

    sim = simulate_freeze(graph, case2, "TNQ1", settings=sandbox_settings)
    assert sim.ok, sim.error
    assert sim.result["metrics"]["simulation_complete"] is True
    freeze = freeze_account(
        graph, case2, "TNQ1", "claim-level verification test freeze",
        eligibility_inputs={"case_state": freeze_state},
        settings=sandbox_settings,
    )
    assert freeze.ok, freeze.error
    assert freeze.result["action_status"] == "applied", freeze.result.get("policy")

    authorized = _claim(graph, case2, "EV-0001")
    assert authorized.status == V.STATUS_CHANGED
    assert "ACT-CASE-ALRT-CVQ-TNQ1" in authorized.reason
    by_qid = {s.query_id: s.status for s in authorized.sources}
    assert by_qid["detect_closed_paths_v1"] == V.STATUS_CHANGED
    assert by_qid["txn_max_ts_v1"] == V.STATUS_REPRODUCED
    assert by_qid["txn_events_for_legs_v1"] == V.STATUS_REPRODUCED

    trace_claim = _claim(graph, case2, "EV-0002")
    assert trace_claim.status == V.STATUS_CHANGED
    assert "ACT-CASE-ALRT-CVQ-TNQ1" in trace_claim.reason

    # The pre-action observation is never called false; the distinction is
    # explicit in the reason text.
    assert "the replay differs from the recorded result and the difference is " \
        "consistent with the recorded, authorized action" in authorized.reason


# ---- H: hypothesis semantics (report-level reasoning) ------------------------


def test_h_absence_is_never_reported_as_a_contradiction(action_net):
    graph = action_net

    # (a) Nothing beyond the alert context was examined: the benign
    # explanation stays OPEN — no contradiction is recorded from absence.
    bare = CaseState(
        alert_id="ALRT-H1", primary_entity_ids=["TNC1"],
        evidence=[EvidenceRecord(
            evidence_id="EV-1", tool="alert_context", stage="test", ok=True,
        )],
    )
    hypotheses = {h.name: h for h in analysis.recompute_hypotheses(bare)}
    benign = hypotheses[H_BENIGN]
    assert benign.status == "open"
    assert "open rather than contradicted" in benign.rationale
    insufficient = hypotheses[H_INSUFFICIENT]
    assert insufficient.status == "supported"  # "cannot distinguish"
    assert "does not distinguish" in insufficient.rationale

    # (b) A positive benign discriminator supports the benign reading (and the
    # rationale names it — never merely the absence of suspicion).
    fan_env = fan_in_out(graph, "TNBEN")
    state = _state([("fan_in_out", fan_env)], "TNBEN", alert="ALRT-H2")
    hypotheses = {h.name: h for h in analysis.recompute_hypotheses(state)}
    benign = hypotheses[H_BENIGN]
    assert benign.status == "supported"
    assert "no outgoing volume" in benign.rationale
    assert hypotheses[H_INSUFFICIENT].status == "supported"


def test_h_overlapping_labels_count_once_and_cite_positive_conflicts(action_net):
    graph = action_net

    # TNC ring: circulation and rapid-forwarding observations come from the
    # SAME ring transactions. Under the policy v2 counting rule they are one
    # distinct-provenance family, so the insufficiency hypothesis must NOT be
    # contradicted by the label count.
    tnc = _state([
        ("detect_cycle", detect_cycle(graph, "TNC1")),
        ("trace_funds", trace_funds(graph, "TNC1", "in", hops=2)),
        ("trace_funds", trace_funds(graph, "TNC1", "out", hops=2)),
    ], "TNC1", alert="ALRT-H3")
    assessment = analysis.signal_family_assessment(tnc, "TNC1")
    assert sorted(assessment["detected"]) == [
        "circulation", "rapid_forwarding",
    ]
    assert assessment["distinct_provenance_families"] == ["circulation"]

    hypotheses = {h.name: h for h in analysis.recompute_hypotheses(tnc)}
    benign = hypotheses[H_BENIGN]
    assert benign.status == "contradicted"
    # The contradiction cites the POSITIVE conflicting families...
    assert "circulation" in benign.rationale
    assert "rapid_forwarding" in benign.rationale
    # ...and never rests on an absent benign discriminator.
    assert "no positive benign discriminator observed yet" not in benign.rationale
    insufficient = hypotheses[H_INSUFFICIENT]
    assert insufficient.status == "supported"
    assert "overlapping" in insufficient.rationale

    # TNQ qualifying ring: the rapid pass-through is genuinely distinct, so
    # TWO distinct-provenance families contradict the insufficiency hypothesis.
    tnq = _state([
        ("detect_cycle", detect_cycle(graph, "TNQ1")),
        ("trace_funds", trace_funds(graph, "TNQ1", "in", hops=2)),
        ("trace_funds", trace_funds(graph, "TNQ1", "out", hops=2)),
    ], "TNQ1", alert="ALRT-H4")
    assessment = analysis.signal_family_assessment(tnq, "TNQ1")
    assert assessment["distinct_provenance_families"] == [
        "circulation", "rapid_forwarding",
    ]
    hypotheses = {h.name: h for h in analysis.recompute_hypotheses(tnq)}
    assert hypotheses[H_INSUFFICIENT].status == "contradicted"
    assert "distinct" in hypotheses[H_INSUFFICIENT].rationale


# ---- I: corrected reports preserve history and carry claim coverage ----------


def test_i_historical_reports_are_preserved_and_corrected_reports_carry_coverage(
    action_net, sandbox_settings, tmp_path
):
    for relative, digest in PINNED_HISTORY.items():
        path = REPORTS_DIR / relative
        assert path.is_file(), f"historical artifact missing: {relative}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
            f"historical artifact was modified: {relative}"
        )

    graph = action_net
    envelope = fan_in_out(graph, "TNI1")
    state = _state([("fan_in_out", envelope)], "TNI1", alert="ALRT-CVI")
    case_id, _ = _cite_state(
        graph, sandbox_settings, state, "TNI1", "ALRT-CVI"
    )

    artifact = build_report(graph, case_id)
    assert artifact.report_version == "phase6-v3"
    row = artifact.payload["evidence_ledger"][0]
    assert row["sub_queries_of_call"] == [
        "txn_max_ts_v1", "fan_in_senders_v1", "fan_out_receivers_v1",
    ]
    assert row["required_source_queries"] == [
        "fan_in_senders_v1", "fan_out_receivers_v1",
    ]
    assert row["source_count"] == 3
    assert row["verification_status"] == V.STATUS_REPRODUCED

    markdown = artifact.markdown
    assert "## 5. Evidence ledger" in markdown
    # Every source query of the invocation is listed, not just the first.
    assert "`fan_in_senders_v1`" in markdown
    assert "`fan_out_receivers_v1`" in markdown
    # Section 10 verifies each source individually and states per-claim
    # coverage (which supporting queries were checked / unavailable).
    assert "Source-query verification" in markdown
    assert artifact.payload["verification"]["references"], (
        "the hypothesis-backed material claims must be mapped to evidence"
    )
    assert "Per-claim source-query coverage" in markdown
    assert "checked: `fan_in_senders_v1`, `fan_out_receivers_v1`" in markdown

    out_path = write_report(artifact, tmp_path)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == artifact.markdown

    # Writing the corrected report left every historical artifact untouched.
    for relative, digest in PINNED_HISTORY.items():
        path = REPORTS_DIR / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
            f"historical artifact was modified: {relative}"
        )
