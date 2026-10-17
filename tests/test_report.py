"""Phase 6 adversarial tests: auditable reports + independent verifier (PLAN.md §11).

These tests exercise the report builder and the LLM-free verifier against
isolated sandbox graphs. Heavy scenarios (a real freeze, a real declined
action) run once per module on dedicated graphs; tamper/mismatch scenarios use
the function-scoped ``action_net`` fixture so the shared test graph is rebuilt
around each destructive test.

Requirement coverage: A report completeness; B freeze pre/post distinction;
C declined action + account stays active; D corrupted row/hash detected;
E nonexistent evidence id; F unknown query id / template drift / invalid
params / arbitrary Cypher rejected; G non-first-row tamper; H unexpected
discrepancy not confused with an authorized change; I historical window replay
(no silent re-anchoring); J truncated/failed queries never unqualified;
K untested hypothesis not shown disproven; L deterministic doubling +
post-generation change detection; M Phase 0-5 safety intact.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re

import pytest

from conftest import _build_net
from src.agent.llm import FakeLLM
from src.agent.state import (
    CaseState,
    EvidenceRecord,
    H_RAPID_MOVEMENT,
    Hypothesis,
)
from src.agent.workflow import build_citations, run_case_workflow
from src.config import DEMO_GRAPH_PREFIX, Settings
from src.db import get_graph, run_query
from src.report import build_report, write_report
from src.report import builder as builder_mod
from src.report import verifier as verifier_mod
from src.report.verifier import (
    STATUS_CHANGED,
    STATUS_MISMATCH,
    STATUS_REPRODUCED,
    STATUS_UNVERIFIABLE,
    STATUS_VERIFIED_RECORDED,
    load_case_records,
    verify_case,
)
from src.seed import schema
from src.tools import query_catalog as qc
from src.tools.actions import cite_evidence, create_case
from src.tools.envelopes import run_evidence

REPORT_NET_SUFFIX = "_report_net"
DECLINED_NET_SUFFIX = "_report_net_declined"
OVERLAP_NET_SUFFIX = "_report_net_overlap"

CYCLE_ALERT = "ALRT-CYCLE"
CYCLE_CASE = "CASE-ALRT-CYCLE"
BENIGN_ALERT = "ALRT-BENIGN"
BENIGN_CASE = "CASE-ALRT-BENIGN"
# Corrected-policy (v2) qualifying scenario: slow ring + distinct rapid
# pass-through on TNQ1.
QUAL_ALERT = "ALRT-QUAL"
QUAL_CASE = "CASE-ALRT-QUAL"

ALL_SECTION_HEADERS = [
    "## 1. Case summary",
    "## 2. Executive findings",
    "## 3. Competing hypotheses",
    "## 4. Investigation timeline",
    "## 5. Evidence ledger",
    "## 6. Transaction paths",
    "## 7. Intervention assessment",
    "## 8. Post-action verification",
    "## 9. Prior-case context",
    "## 10. Independent verifier results",
]


def _dedicated_settings(settings, suffix: str) -> Settings:
    return Settings(**{
        **settings.model_dump(),
        "graph_name": settings.graph_name + suffix,
        "habeas_demo_mode": True,
        "habeas_enable_actions": True,
    })


def _fresh_graph(settings, suffix: str):
    name = settings.graph_name + suffix
    assert name.startswith(DEMO_GRAPH_PREFIX), "report graphs must stay demo-safe"
    try:
        graph = get_graph(settings, name=name)
        run_query(graph, "RETURN 1 AS ok", read_only=False)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"FalkorDB not reachable ({type(exc).__name__}: {exc})")
    _build_net(graph)
    return graph


@pytest.fixture(scope="module")
def trace_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("traces6")


@pytest.fixture(scope="module")
def frozen_case(settings):
    """ALRT-QUAL on a dedicated graph: investigation + successful freeze.

    The QUAL scenario is the corrected-policy qualifying case: a slow ring
    (circulation) plus one genuinely distinct rapid pass-through on TNQ1, so
    the recorded policy authorizes the freeze and the report shows an
    authorized post-action change, not tamper.
    """
    dedicated = _dedicated_settings(settings, REPORT_NET_SUFFIX)
    graph = _fresh_graph(settings, REPORT_NET_SUFFIX)
    result = asyncio.run(run_case_workflow(QUAL_ALERT, settings=dedicated, llm=FakeLLM()))
    assert result.frozen is True, "the fixture requires a successful freeze"
    artifact = build_report(
        graph, result.case_id, state=result.state, envelopes=result.envelopes
    )
    return {
        "graph": graph, "settings": dedicated, "result": result,
        "artifact": artifact,
    }


@pytest.fixture(scope="module")
def overlap_case(settings):
    """ALRT-CYCLE on a dedicated graph: overlapping families, declined freeze.

    Circulation and rapid-forwarding observations are drawn from the same ring
    transactions, so under policy v2 they count as one distinct-provenance
    family and the freeze is declined even though the escalation verdict still
    supports review.
    """
    dedicated = _dedicated_settings(settings, OVERLAP_NET_SUFFIX)
    graph = _fresh_graph(settings, OVERLAP_NET_SUFFIX)
    result = asyncio.run(run_case_workflow(CYCLE_ALERT, settings=dedicated, llm=FakeLLM()))
    assert result.freeze_declined is True, "the fixture requires a declined freeze"
    artifact = build_report(
        graph, result.case_id, state=result.state, envelopes=result.envelopes
    )
    return {
        "graph": graph, "settings": dedicated, "result": result,
        "artifact": artifact,
    }


@pytest.fixture(scope="module")
def declined_case(settings):
    """ALRT-BENIGN on a dedicated graph: investigation + declined freeze."""
    dedicated = _dedicated_settings(settings, DECLINED_NET_SUFFIX)
    graph = _fresh_graph(settings, DECLINED_NET_SUFFIX)
    result = asyncio.run(run_case_workflow(BENIGN_ALERT, settings=dedicated, llm=FakeLLM()))
    assert result.freeze_declined is True, "the fixture requires a declined freeze"
    artifact = build_report(
        graph, result.case_id, state=result.state, envelopes=result.envelopes
    )
    return {
        "graph": graph, "settings": dedicated, "result": result,
        "artifact": artifact,
    }


def _case_with_citation(graph, settings, case_alert=CYCLE_ALERT, local_id="EV-1"):
    """Create a case and cite one real alert_context observation."""
    create_case(graph, case_alert, entity_ids=["TNC1"], settings=settings)
    case_id = f"CASE-{case_alert}"
    evidence = run_evidence(
        graph, "alert_context_v1", qc.get_template("alert_context_v1"),
        {"alert_id": case_alert}, row_cap=10,
    )
    assert evidence.truncated is False
    out = cite_evidence(graph, case_id, [{
        "evidence_id": local_id,
        "claim": "alert context captured for the report tests",
        "stance": "supporting",
        "query_id": "alert_context_v1",
        "params": evidence.params,
        "rows": evidence.rows,
        "columns": evidence.columns,
        "result_hash": evidence.rows_hash,
        "row_count": evidence.row_count,
        "truncated": evidence.truncated,
        "evidence_kind": "captured_snapshot",
        "source_tool": "alert_context",
    }], settings=settings)
    assert out.ok is True, out
    return case_id, f"{case_id}:{local_id}"


def _set_evidence_props(graph, evidence_id, **props):
    run_query(
        graph,
        "MATCH (e:Evidence {id: $id}) SET e += $props",
        {"id": evidence_id, "props": props},
        read_only=False,
    )


def _graph_counters(graph):
    nodes = run_query(graph, "MATCH (n) RETURN count(n)", read_only=True).result_set[0][0]
    rels = run_query(graph, "MATCH ()-[r]->() RETURN count(r)", read_only=True).result_set[0][0]
    return int(nodes), int(rels)


# ---- A: report completeness -------------------------------------------------


def test_a_frozen_report_has_all_sections_and_a_fully_linked_ledger(frozen_case):
    artifact = frozen_case["artifact"]
    markdown = artifact.markdown

    for header in ALL_SECTION_HEADERS:
        assert header in markdown, f"missing section {header!r}"

    assert artifact.report_version.startswith("phase6")
    assert re.match(r"^sha256:[0-9a-f]{64}$", artifact.fingerprint)
    assert re.search(r"Generated: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", markdown)

    ledger = artifact.payload["evidence_ledger"]
    assert ledger, "the ledger must not be empty for a real investigation"
    assert len(artifact.verification.claims) == len(ledger)
    # Every ledger id is the persisted, case-scoped id and appears in prose.
    for row in ledger:
        assert row["evidence_id"].startswith(f"{frozen_case['result'].case_id}:")
        assert f"`{row['evidence_id']}`" in markdown
    # The verification summary accounts for every claim exactly once.
    assert sum(artifact.verification.summary.values()) == len(ledger)
    # Ledger verification cells are the vocabulary strings, not dataclass reprs.
    statuses = {row["verification_status"] for row in ledger}
    assert statuses <= {STATUS_VERIFIED_RECORDED, STATUS_REPRODUCED, STATUS_CHANGED,
                        STATUS_MISMATCH, STATUS_UNVERIFIABLE}
    assert "ClaimVerification(" not in markdown
    # Material claims point at evidence; hypotheses reference real ids.
    for ref in artifact.verification.references:
        for evidence_id in ref.evidence_ids:
            assert evidence_id.startswith(f"{frozen_case['result'].case_id}:")


def test_a_report_writes_to_disk_byte_identically(frozen_case, tmp_path):
    artifact = frozen_case["artifact"]
    path = write_report(artifact, tmp_path)
    assert path.name == f"{artifact.case_id}.md"
    assert path.read_text(encoding="utf-8") == artifact.markdown


# ---- B: freeze pre/post distinction -----------------------------------------


def test_b_frozen_report_keeps_pre_and_post_action_distinct(frozen_case):
    artifact = frozen_case["artifact"]
    markdown = artifact.markdown

    assert artifact.payload["disposition"] == "freeze_applied"

    # The pre-action ring observation no longer reproduces, and the difference
    # is attributed to the recorded action — never called false.
    cycle_claims = [c for c in artifact.verification.claims
                    if c.query_id == "detect_closed_paths_v1"]
    assert len(cycle_claims) == 1
    assert cycle_claims[0].status == STATUS_CHANGED
    assert "ACT-CASE-ALRT-QUAL-TNQ1" in cycle_claims[0].reason
    assert cycle_claims[0].replay["recorded_hash"] != cycle_claims[0].replay["replayed_hash"]

    # Post-action verification is a separate, current-state section.
    post = artifact.payload["post_action"]
    assert post["post_action"] and post["post_action"][0]["verified"] is True

    before = artifact.payload["intervention"]["action"]["before_state"]
    after = artifact.payload["intervention"]["action"]["after_state"]
    assert before["status"] == "active"
    assert after["status"] == "frozen"

    assert "never a sign that the original observation was false" in markdown
    # This case was authorized under policy v2 by two families whose
    # observations are genuinely distinct: the report states the distinct
    # provenance explicitly, and the unqualified independence warning must NOT
    # appear (it belongs to overlapping/unavailable cases only).
    assert "independent signal families" in markdown
    assert "independence is not established" not in markdown
    assert "supported by distinct underlying transaction events (no shared ids)" \
        in markdown
    # The recorded policy version is surfaced so audits can distinguish what
    # was authorized under which policy.
    assert "Recorded policy version: 2 (current policy: 2)." in markdown
    # No claim that money moved, was blocked or was frozen.
    lowered = markdown.lower()
    for banned in ("funds were frozen", "money was frozen", "frozen funds",
                   "funds are frozen", "balance was frozen"):
        assert banned not in lowered
    assert "does not move, block, seize or refund any money" in lowered
    assert "simulated graph mutation" in lowered


# ---- C: declined action ------------------------------------------------------


def test_c_declined_report_lists_blocking_reasons_and_account_stays_active(declined_case):
    artifact = declined_case["artifact"]
    graph = declined_case["graph"]
    markdown = artifact.markdown

    assert artifact.payload["disposition"] == "freeze_declined"
    intervention = artifact.payload["intervention"]
    assert intervention["action"] is None
    assert intervention["blocking_reasons"], "blocking reasons must be recorded"
    assert any("verdict" in reason for reason in intervention["blocking_reasons"])
    assert intervention["eligibility"]["allowed"] is False

    # The declined account is independently re-read as active.
    statuses = {row["entity_id"]: row["status"] for row in
                artifact.payload["post_action"]["current_state"]}
    assert statuses.get("TNBEN") == "active"
    row = run_query(graph, "MATCH (a:Account {id: 'TNBEN'}) RETURN a.status",
                    read_only=True).result_set
    assert row[0][0] == "active"

    assert artifact.payload["post_action"]["post_action"] == []
    assert "No action was applied" in markdown
    for reason in intervention["blocking_reasons"]:
        assert reason in markdown or reason.replace("'", "'") in markdown
    bad = [c for c in artifact.verification.claims if c.status == STATUS_MISMATCH]
    assert bad == [], f"declined report must not carry mismatches: {bad}"


# ---- D/G: corrupted snapshots and rows --------------------------------------


def test_d_tampered_result_hash_is_detected(action_net, sandbox_settings):
    graph = action_net
    case_id, evidence_id = _case_with_citation(graph, sandbox_settings)

    clean = verify_case(graph, case_id)
    assert [c.status for c in clean.claims] == [STATUS_REPRODUCED]

    _set_evidence_props(graph, evidence_id, result_hash="sha256:" + "0" * 64)
    tampered = verify_case(graph, case_id)
    claim = tampered.claims[0]
    assert claim.status == STATUS_MISMATCH
    assert "do not match the declared result hash" in claim.reason


def test_g_tampering_a_non_first_row_is_detected(action_net, sandbox_settings):
    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNI1"], settings=sandbox_settings)
    case_id = CYCLE_CASE
    evidence = run_evidence(
        graph, "fan_in_senders_v1", qc.get_template("fan_in_senders_v1"),
        {"account_id": "TNI1", "cutoff": "2026-01-01T00:00:00Z", "limit": 50},
        row_cap=50,
    )
    assert evidence.row_count >= 3, "need a genuinely multi-row citation"
    out = cite_evidence(graph, case_id, [{
        "evidence_id": "EV-ROWS",
        "claim": "sender rollup for tamper coverage",
        "stance": "supporting",
        "query_id": "fan_in_senders_v1",
        "params": evidence.params,
        "rows": evidence.rows,
        "columns": evidence.columns,
        "result_hash": evidence.rows_hash,
        "row_count": evidence.row_count,
        "truncated": evidence.truncated,
        "evidence_kind": "captured_snapshot",
        "source_tool": "fan_in_out",
    }], settings=sandbox_settings)
    assert out.ok is True
    evidence_id = f"{case_id}:EV-ROWS"
    assert verify_case(graph, case_id).claims[0].status == STATUS_REPRODUCED

    rows = json.loads(json.dumps(evidence.rows))
    rows[-1][2] = float(rows[-1][2]) + 1.0  # only the LAST row changes
    _set_evidence_props(graph, evidence_id, rows_json=json.dumps(rows))
    claim = verify_case(graph, case_id).claims[0]
    assert claim.status == STATUS_MISMATCH
    assert "do not match the declared result hash" in claim.reason


# ---- E: missing evidence -----------------------------------------------------


def test_e_references_to_nonexistent_evidence_ids_are_unverifiable(action_net, sandbox_settings):
    graph = action_net
    case_id, _ = _case_with_citation(graph, sandbox_settings)

    verification = verify_case(graph, case_id, claim_refs=[{
        "claim": "a claim whose evidence does not exist",
        "source": "test",
        "evidence_ids": [f"{case_id}:EV-9999"],
    }])
    ref = verification.references[0]
    assert ref.status == STATUS_UNVERIFIABLE
    assert ref.missing_evidence_ids == [f"{case_id}:EV-9999"]
    assert "missing evidence ids" in ref.reason

    # The same guarantee through the report builder: a hypothesis referencing
    # an absent record cannot render as verified.
    crafted = CaseState(
        alert_id=CYCLE_ALERT,
        primary_entity_ids=["TNC1"],
        hypotheses=[Hypothesis(
            name=H_RAPID_MOVEMENT, status="open",
            supporting_evidence=["EV-9999"],
        )],
    )
    artifact = build_report(graph, case_id, state=crafted, generated_at="2026-01-01T00:00:00Z")
    refs = [r for r in artifact.verification.references
            if "EV-9999" in r.evidence_ids or "EV-9999" in r.missing_evidence_ids]
    assert refs and refs[0].status == STATUS_UNVERIFIABLE
    assert "EV-9999" in artifact.markdown


# ---- F: hostile records ------------------------------------------------------


def test_f_hostile_records_are_rejected_without_execution(action_net, sandbox_settings):
    graph = action_net
    case_id, evidence_id = _case_with_citation(graph, sandbox_settings)

    # (1) template drift: the stored fingerprint no longer matches the catalog
    _set_evidence_props(graph, evidence_id, template_sha256="sha256:" + "0" * 64)
    drifted = verify_case(graph, case_id).claims[0]
    assert drifted.status == STATUS_MISMATCH
    assert "template definition" in drifted.reason and "changed after capture" in drifted.reason

    # (2) invalid params: a key the approved template does not use
    _set_evidence_props(
        graph, evidence_id,
        template_sha256=qc.template_sha256("alert_context_v1"),
        params_json=json.dumps({"alert_id": "ALRT-CYCLE", "evil": True}),
    )
    bad_params = verify_case(graph, case_id).claims[0]
    assert bad_params.status == STATUS_MISMATCH
    assert "not used by alert_context_v1" in bad_params.reason

    # (3) invalid param VALUE: fails the documented validator
    _set_evidence_props(
        graph, evidence_id,
        params_json=json.dumps({"alert_id": "not a valid id!!"}),
    )
    bad_value = verify_case(graph, case_id).claims[0]
    assert bad_value.status == STATUS_MISMATCH

    # (4) unknown query id + arbitrary Cypher stored on the record: rejected,
    # never executed. The node count is unchanged afterwards.
    nodes_before, rels_before = _graph_counters(graph)
    run_query(
        graph,
        "MATCH (c:Case {id: $case}) MERGE (e:Evidence {id: $eid}) "
        "SET e.claim = 'hostile record', e.stance = 'supporting', "
        "e.query_id = 'evil_freeform_v1', e.params_json = '{}', "
        "e.rows_json = '[]', e.columns_json = '[]', e.derived_json = '{}', "
        "e.result_hash = $hash, e.row_count = 0, e.truncated = false, "
        "e.interpretation = 'hostile', e.evidence_kind = 'captured_snapshot', "
        "e.source_tool = 'test', e.captured_at = '2026-01-01T00:00:00Z', "
        "e.case_id = $case, e.cypher = 'MATCH (n) DETACH DELETE n' "
        "MERGE (c)-[:HAS_EVIDENCE]->(e) RETURN e.id AS id",
        {"case": case_id, "eid": f"{case_id}:EV-EVIL",
         "hash": "sha256:" + "1" * 64},
        read_only=False,
    )
    hostile = {c.evidence_id: c for c in verify_case(graph, case_id).claims}
    claim = hostile[f"{case_id}:EV-EVIL"]
    assert claim.status == STATUS_MISMATCH
    assert "unknown query id" in claim.reason
    assert "no execution attempted" in " ".join(chk.detail for chk in claim.checks)
    nodes_after, rels_after = _graph_counters(graph)
    # +1 node and +1 HAS_EVIDENCE relationship from the hostile write itself;
    # the embedded DETACH DELETE text was never executed.
    assert (nodes_after, rels_after) == (nodes_before + 1, rels_before + 1)

    # (5) the write layer refuses unapproved query ids outright
    refused = cite_evidence(graph, case_id, [{
        "claim": "attempted citation", "stance": "supporting",
        "query_id": "evil_cypher_v2", "result_hash": "sha256:" + "2" * 64,
    }], settings=sandbox_settings)
    assert refused.ok is False
    assert "unapproved query id" in (refused.error.message if refused.error else "")

    # Static guarantee: the verifier never executes raw Cypher.
    verifier_source = inspect.getsource(verifier_mod)
    assert "run_query(" not in verifier_source
    assert "src.agent.llm" not in verifier_source


# ---- H: unexpected discrepancy vs authorized change --------------------------


def test_h_unexpected_tamper_is_not_confused_with_the_recorded_action(
    action_net, sandbox_settings, trace_dir
):
    graph = action_net
    result = asyncio.run(run_case_workflow(
        QUAL_ALERT, settings=sandbox_settings, llm=FakeLLM(), trace_dir=trace_dir,
    ))
    assert result.frozen is True
    case_id = result.case_id

    # The authorized action explains the ring query's change...
    claims = {c.query_id: c for c in verify_case(graph, case_id).claims}
    assert claims["detect_closed_paths_v1"].status == STATUS_CHANGED

    # ...but an impossible change in immutable history is a mismatch and must
    # not be attributed to the action.
    record = load_case_records(graph, case_id)
    alert_record = next(r for r in record.evidence
                        if r["query_id"] == "alert_context_v1")
    _set_evidence_props(
        graph, alert_record["evidence_id"],
        result_hash="sha256:" + "3" * 64,
    )
    claims = {c.evidence_id: c for c in verify_case(graph, case_id).claims}
    tampered = claims[alert_record["evidence_id"]]
    assert tampered.status == STATUS_MISMATCH
    # The mismatch is never attributed to the authorized action.
    assert "consistent with the recorded, authorized action" not in tampered.reason


# ---- I: historical window replay ---------------------------------------------


def test_i_replay_uses_the_recorded_window_not_the_current_latest_timestamp(
    action_net, sandbox_settings
):
    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNC1"], settings=sandbox_settings)
    case_id = CYCLE_CASE

    from src.tools.reads import trace_funds

    envelope = trace_funds(graph, "TNC1", "out", hops=2, limit=50, window_days=90)
    assert envelope.ok is True
    trace_query = next(
        q for q in envelope.query_evidence if q.query_id == "trace_paths_out_v1"
    )
    assert trace_query.bound == {"<<hops>>": 2}
    cutoff = trace_query.params["cutoff"]

    out = cite_evidence(graph, case_id, [{
        "evidence_id": "EV-WINDOW",
        "claim": "bounded outbound trace at investigation time",
        "stance": "supporting",
        "query_id": "trace_paths_out_v1",
        "params": trace_query.params,
        "interpolation": trace_query.bound,
        "rows": trace_query.rows,
        "columns": trace_query.columns,
        "result_hash": trace_query.rows_hash,
        "row_count": trace_query.row_count,
        "truncated": trace_query.truncated,
        "evidence_kind": "captured_snapshot",
        "source_tool": "trace_funds",
    }], settings=sandbox_settings)
    assert out.ok is True
    evidence_id = f"{case_id}:EV-WINDOW"

    # Immediately after capture the exact bounded query reproduces.
    claim = verify_case(graph, case_id).claims[0]
    assert claim.status == STATUS_REPRODUCED
    assert claim.replay["params"]["cutoff"] == cutoff
    assert claim.replay["interpolation"] == {"<<hops>>": 2}

    # The graph legitimately changes: a brand-new active route appears AFTER
    # the recorded observation. Replay must still use the ORIGINAL cutoff and
    # interpolation (never re-anchor to the new latest timestamp), and the
    # difference must be shown as a change, not hidden.
    schema.insert_relationships(
        graph, "TRANSFERRED_TO",
        [{"src": "TNC1", "dst": "TNHUB",
          "props": {"count": 1, "total_amount": 500000.0,
                    "last_ts": "2026-06-01T00:00:00Z", "active": True}}],
        src_label="Account", dst_label="Account",
    )
    changed = verify_case(graph, case_id).claims[0]
    assert changed.status == STATUS_MISMATCH
    assert "no recorded action explains" in changed.reason
    assert changed.replay["params"]["cutoff"] == cutoff, (
        "replay silently re-anchored to a new window"
    )
    assert changed.replay["interpolation"] == {"<<hops>>": 2}
    assert changed.replay["recorded_hash"] != changed.replay["replayed_hash"]


# ---- J: truncated and failed queries -----------------------------------------


def test_j_truncated_evidence_is_qualified_never_fully_reproduced(action_net, sandbox_settings):
    graph = action_net
    create_case(graph, CYCLE_ALERT, entity_ids=["TNI1"], settings=sandbox_settings)
    case_id = CYCLE_CASE
    cutoff = "2026-01-01T00:00:00Z"

    truncated = run_evidence(
        graph, "fan_in_senders_v1", qc.get_template("fan_in_senders_v1"),
        {"account_id": "TNI1", "cutoff": cutoff, "limit": 2}, row_cap=2,
    )
    assert truncated.truncated is True and truncated.row_count == 2

    out = cite_evidence(graph, case_id, [
        {
            "evidence_id": "EV-TRUNC-ROWS",
            "claim": "truncated sender rollup with preserved rows",
            "stance": "supporting",
            "query_id": "fan_in_senders_v1",
            "params": truncated.params,
            "rows": truncated.rows,
            "columns": truncated.columns,
            "result_hash": truncated.rows_hash,
            "row_count": truncated.row_count,
            "truncated": True,
            "evidence_kind": "captured_snapshot",
            "source_tool": "fan_in_out",
        },
        {
            "evidence_id": "EV-TRUNC-HASH",
            "claim": "truncated sender rollup without preserved rows",
            "stance": "supporting",
            "query_id": "fan_in_senders_v1",
            "params": truncated.params,
            "result_hash": truncated.rows_hash,
            "row_count": truncated.row_count,
            "truncated": True,
            "evidence_kind": "captured_snapshot",
            "source_tool": "fan_in_out",
        },
    ], settings=sandbox_settings)
    assert out.ok is True

    claims = {c.evidence_id.split(":")[-1]: c for c in verify_case(graph, case_id).claims}
    assert claims["EV-TRUNC-ROWS"].status == STATUS_VERIFIED_RECORDED
    assert "truncated" in claims["EV-TRUNC-ROWS"].reason
    assert claims["EV-TRUNC-HASH"].status == STATUS_VERIFIED_RECORDED
    assert "truncated" in claims["EV-TRUNC-HASH"].reason
    for claim in claims.values():
        assert claim.status != STATUS_REPRODUCED, (
            "a truncated observation must never be presented as fully reproduced"
        )

    # Failed tool calls never become citations in the first place.
    failed_state = CaseState(
        alert_id=CYCLE_ALERT,
        primary_entity_ids=["TNC1"],
        evidence=[
            EvidenceRecord(evidence_id="EV-1", tool="alert_context",
                           stage="START", ok=False, error="boom"),
            EvidenceRecord(
                evidence_id="EV-2", tool="alert_context", stage="START", ok=True,
                query_ids=["alert_context_v1"], params=[{}], rows_hash=[""],
                row_count=[0], truncated=[False],
            ),
        ],
    )
    citations = build_citations(failed_state, None)
    assert [c["evidence_id"] for c in citations] == ["EV-2"]


# ---- K: untested hypotheses ---------------------------------------------------


def test_k_untested_hypothesis_is_not_presented_as_disproven(action_net, sandbox_settings):
    graph = action_net
    case_id, _ = _case_with_citation(graph, sandbox_settings)
    crafted = CaseState(
        alert_id=CYCLE_ALERT,
        primary_entity_ids=["TNC1"],
        hypotheses=[Hypothesis(
            name=H_RAPID_MOVEMENT,
            status="open",
            supporting_evidence=[],
            contradicting_evidence=[],
            uncertainties=["the timing question was never examined"],
            rationale="",
        )],
    )
    artifact = build_report(graph, case_id, state=crafted,
                            generated_at="2026-01-01T00:00:00Z")
    hypothesis = artifact.payload["hypotheses"][0]
    assert hypothesis["status"] == "open"
    assert "remains untested" in hypothesis["note"]
    assert "does not treat it as disproven" in hypothesis["note"]
    assert "remains untested" in artifact.markdown
    assert "the timing question was never examined" in artifact.markdown
    # The assessment is not filtered through the mismatch vocabulary.
    refs = [r for r in artifact.verification.references
            if r.claim.startswith("Hypothesis 'rapid_movement")]
    assert refs and refs[0].status == STATUS_UNVERIFIABLE  # no evidence claimed


# ---- L: determinism and post-generation change detection -----------------------


def test_l_reports_are_deterministic_and_detect_post_generation_changes(
    action_net, sandbox_settings
):
    graph = action_net
    case_id, evidence_id = _case_with_citation(graph, sandbox_settings)

    first = build_report(graph, case_id, generated_at="2026-07-01T00:00:00Z")
    second = build_report(graph, case_id, generated_at="2026-07-01T00:00:00Z")
    assert first.markdown == second.markdown
    assert first.fingerprint == second.fingerprint

    # The generation timestamp itself is excluded from the content fingerprint.
    later = build_report(graph, case_id, generated_at="2026-07-02T00:00:00Z")
    assert later.fingerprint == first.fingerprint
    assert later.markdown != first.markdown

    # A post-generation change to a cited record is detected: the rebuilt
    # report's content fingerprint no longer matches the original report, so
    # a reviewer can see the record was edited after generation.
    _set_evidence_props(graph, evidence_id, claim="EDITED AFTER GENERATION")
    third = build_report(graph, case_id, generated_at="2026-07-01T00:00:00Z")
    assert third.fingerprint != first.fingerprint
    assert "EDITED AFTER GENERATION" in third.markdown
    # Integrity-level changes are additionally flagged by the verifier itself
    # (see the tamper tests); a prose edit changes content, not integrity.
    assert third.verification.claims[0].status == STATUS_REPRODUCED


# ---- M: Phase 0-5 safety intact -----------------------------------------------


def test_m_report_and_verifier_never_mutate_the_graph(action_net, sandbox_settings):
    graph = action_net
    case_id, _ = _case_with_citation(graph, sandbox_settings)
    before = _graph_counters(graph)
    status_before = run_query(
        graph, "MATCH (a:Account) RETURN a.id, a.status ORDER BY a.id", read_only=True,
    ).result_set
    build_report(graph, case_id, generated_at="2026-01-01T00:00:00Z")
    verify_case(graph, case_id)
    after = _graph_counters(graph)
    status_after = run_query(
        graph, "MATCH (a:Account) RETURN a.id, a.status ORDER BY a.id", read_only=True,
    ).result_set
    assert before == after
    assert status_before == status_after


def test_m_report_modules_do_not_write_and_catalog_never_deletes():
    for module in (builder_mod, verifier_mod):
        source = inspect.getsource(module)
        assert "run_query(" not in source, "report modules must not execute raw queries"
        assert "DELETE" not in source
    for query_id, template in qc._QUERY_TEMPLATES.items():
        assert "DELETE" not in template.upper(), (
            f"catalog template {query_id} contains a delete; agent tools may never delete"
        )
    # Reads stay read-only: only the Phase 5 write templates are non-reads.
    write_ids = set(qc.WRITE_QUERY_IDS)
    assert write_ids, "the write catalog must be explicit"
    assert "evidence_upsert_v1" in write_ids
    assert "detect_closed_paths_v1" not in write_ids


def test_m_corrected_policy_counts_only_distinct_provenance_families(
    frozen_case, overlap_case
):
    """Policy v2 is provenance-aware: overlapping families count once.

    The overlap fixture (ALRT-CYCLE) still escalates for review, but its two
    families are drawn from the same ring transactions, so the freeze is
    declined and the report records the overlap with the shared ids. The
    frozen fixture (ALRT-QUAL) has two genuinely distinct, target-relevant
    families, so the same policy authorizes its freeze. Any future change to
    this rule must be deliberate and separately tested.
    """
    # --- overlapping case: declined; overlap recorded with shared ids ------
    overlap = overlap_case["artifact"]
    provenance = overlap.payload["provenance"]
    pair = next(p for p in provenance["pairs"]
                if set(p["pair"]) == {"circulation", "rapid_forwarding"})
    assert pair["classification"] in ("shared_observations", "partially_shared")
    assert pair["shared_count"] > 0
    # The shared observations are ring transactions, not independent evidence.
    ring_txns = {f"TNT{i:03d}" for i in range(8)}
    assert set(pair["shared_txn_ids"]) <= ring_txns

    eligibility = overlap.payload["intervention"]["eligibility"]
    assert set(eligibility["signal_families"]) == {"circulation", "rapid_forwarding"}
    assert eligibility["distinct_provenance_families"] == ["circulation"]
    assert eligibility["allowed"] is False
    assert any("distinct-provenance" in reason
               for reason in eligibility["blocking_reasons"])

    caveats = overlap.payload["executive_findings"]["provenance_caveats"]
    assert any("not independent corroboration" in caveat for caveat in caveats)
    assert "not independent corroboration" in overlap.markdown

    # --- qualifying case: two distinct target-relevant families, allowed ---
    frozen = frozen_case["artifact"]
    eligibility = frozen.payload["intervention"]["eligibility"]
    assert set(eligibility["signal_families"]) == {"circulation", "rapid_forwarding"}
    assert set(eligibility["distinct_provenance_families"]) == {
        "circulation", "rapid_forwarding"}
    assert eligibility["allowed"] is True


def test_m_verifier_replay_classification_is_complete():
    """Every approved READ query is classified exactly once for replay; write
    templates are never replayed. New catalog entries cannot slip through
    unchecked."""
    reads = set(qc._QUERY_TEMPLATES) - set(qc.WRITE_QUERY_IDS)
    buckets = (
        set(verifier_mod._IMMUTABLE_QUERIES)
        | set(verifier_mod._ACTION_SENSITIVE_QUERIES)
        | set(verifier_mod._MUTABLE_METADATA_QUERIES)
    )
    assert reads == buckets, (
        f"unclassified reads: {reads - buckets}; stale classifications: {buckets - reads}"
    )
    assert not buckets & set(qc.WRITE_QUERY_IDS)


def test_m_param_and_interpolation_contracts_cover_the_catalog():
    for query_id in qc._QUERY_TEMPLATES:
        for name in qc.param_names(query_id):
            assert name in qc._PARAM_VALIDATORS, (
                f"no validator for ${name} of {query_id}"
            )
        for token in qc.template_tokens(query_id):
            assert token in qc._INTERPOLATION_VALIDATORS, (
                f"no validator for {token} of {query_id}"
            )
