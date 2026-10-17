"""Canonical verdicts and the deterministic action-eligibility policy (PLAN.md §9).

Two things live here, both ordinary Python:

1. **One canonical verdict vocabulary** replacing the looser Phase 4 strings.
   A provisional suspicion is NOT a fraud finding, and only a
   ``suspicion_supported_for_review`` verdict may ever be escalated to a
   sandbox action.

2. **The action-eligibility policy.** The LLM may *propose* an intervention;
   this module alone decides whether it is permitted. It weighs evidence
   quality, independent signal families, contradictions, material unresolved
   hypotheses, failed queries, truncation, and whether the proposed target is
   actually the entity the evidence supports.

Nothing here consults the ground-truth manifest, and no LLM confidence value
is ever an input.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---- canonical verdicts ----------------------------------------------------

VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION = "suspicious_activity"
VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW = "suspicion_supported_for_review"
VERDICT_LIKELY_BENIGN = "likely_benign"
VERDICT_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
VERDICT_INCOMPLETE_INVESTIGATION = "incomplete_investigation"

CanonicalVerdict = Literal[
    "suspicious_activity",
    "suspicion_supported_for_review",
    "likely_benign",
    "insufficient_evidence",
    "incomplete_investigation",
]

# Phase 4 compatibility mapping (documented in PLAN.md §9).
PHASE4_VERDICT_MIGRATION = {
    "suspicious_activity": VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION,
    "likely_benign": VERDICT_LIKELY_BENIGN,
    "insufficient_evidence": VERDICT_INSUFFICIENT_EVIDENCE,
}

MIN_EVIDENCE_RECORDS = 3
MIN_SIGNAL_FAMILIES = 2

POLICY_VERSION = 2
"""Eligibility policy version, persisted with every authorization.

- v1 counted signal families by name only.
- v2 is provenance-aware: only families whose observations implicate the
  proposed target count, and two counted families must have disjoint
  underlying observations (``analysis.distinct_provenance_families``).

v2 applies to new authorizations only. Records written under v1 keep the
version they were recorded with and are never re-evaluated, re-labelled or
recalculated as if v2 had produced them.
"""

# Queries whose truncation would undermine a freeze justification. Contextual
# queries (e.g. hub_scores) are recorded as caveats instead of blockers.
MATERIAL_QUERIES = frozenset({
    "entity_profile_lookup_v1", "entity_profile_flows_v1",
    "detect_closed_paths_v1", "trace_paths_out_v1", "trace_paths_in_v1",
    "fan_in_senders_v1", "fan_out_receivers_v1", "txn_events_for_legs_v1",
})


def canonical_verdict(provisional: str | None) -> str:
    """Map any verdict string onto the canonical vocabulary."""
    if provisional is None:
        return VERDICT_INCOMPLETE_INVESTIGATION
    return PHASE4_VERDICT_MIGRATION.get(provisional, provisional)


# ---- eligibility -----------------------------------------------------------


class EligibilityDecision(BaseModel):
    """Outcome of the deterministic action-eligibility policy."""

    action: str
    allowed: bool
    verdict: str
    policy_version: int = POLICY_VERSION
    blocking_reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    signal_families: list[str] = Field(default_factory=list)
    target_relevant_families: list[str] = Field(default_factory=list)
    distinct_provenance_families: list[str] = Field(default_factory=list)
    qualifying_evidence_ids: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        if self.allowed:
            return (
                f"{self.action} is permitted for this case: "
                f"{len(self.qualifying_evidence_ids)} qualifying evidence records "
                f"across {len(self.distinct_provenance_families)} "
                f"distinct-provenance signal families "
                f"({', '.join(self.distinct_provenance_families)})."
            )
        return (
            f"{self.action} is NOT permitted: "
            + "; ".join(self.blocking_reasons or ["no qualifying basis"])
        )


def _provenance_findings(assessment: dict[str, Any]) -> list[str]:
    """Deterministic sentences explaining families that could not be counted."""
    from src.agent import analysis

    findings: list[str] = []
    for item in assessment.get("excluded") or []:
        findings.append(f"family {item['family']} was not counted: {item['reason']}")
    relevant = set(assessment.get("target_relevant") or [])
    for name in sorted(relevant):
        source = (assessment.get("sources") or {}).get(name) or {}
        if not source.get("available"):
            findings.append(
                f"family {name} has no transaction-level observations "
                f"({source.get('basis')}), so it can never be the second family "
                "of a pair with distinct provenance"
            )
    for pair in assessment.get("pairs") or []:
        if set(pair["pair"]).isdisjoint(relevant):
            continue
        classification = pair["classification"]
        names = " and ".join(pair["pair"])
        ids = ", ".join(pair["shared_txn_ids"][:6])
        if classification == analysis.OVERLAP_SHARED:
            findings.append(
                f"{names}: one family's observations are contained in the "
                f"other's ({pair['shared_count']} shared transaction id(s): {ids})"
            )
        elif classification == analysis.OVERLAP_PARTIAL:
            findings.append(
                f"{names}: their observations overlap "
                f"({pair['shared_count']} shared transaction id(s): {ids})"
            )
        elif classification == analysis.OVERLAP_UNAVAILABLE:
            findings.append(
                f"{names}: at least one family's observations cannot be "
                "resolved to individual transactions, so independence cannot "
                "be established"
            )
    return findings


def evaluate_freeze_eligibility(
    state,
    *,
    verdict: str,
    target_account_id: str,
    simulation: dict[str, Any] | None,
    simulation_stale: bool,
    account_exists: bool,
    account_already_frozen: bool,
    sandbox_enabled: bool,
    graph_is_demo: bool,
    actions_enabled: bool,
) -> EligibilityDecision:
    """Decide whether a sandbox freeze of ``target_account_id`` is permitted.

    Policy v2: counted evidence must (a) implicate the proposed target and
    (b) come from families whose underlying observations are genuinely
    distinct — different signal names derived from the same transactions
    count once. Every check is independent and reported, so an operator can
    see exactly which condition failed, which families overlapped, and by how
    much. Any failure means NO mutation is performed. LLM confidence is never
    an input; this function only sees recorded evidence and policy inputs.
    """
    from src.agent import analysis
    from src.agent.state import H_BENIGN

    assessment = analysis.signal_family_assessment(state, target_account_id)
    all_families = list(assessment["detected"])
    families = all_families
    relevant_families = list(assessment["target_relevant"])
    counted_families = list(assessment["distinct_provenance_families"])
    ok_records = [r for r in state.evidence if r.ok]
    failed_records = [r for r in state.evidence if not r.ok]

    material_truncated = sorted({
        qid
        for record in ok_records
        for qid, truncated in zip(record.query_ids, record.truncated)
        if truncated and qid in MATERIAL_QUERIES
    })
    contextual_truncated = sorted({
        qid
        for record in ok_records
        for qid, truncated in zip(record.query_ids, record.truncated)
        if truncated and qid not in MATERIAL_QUERIES
    })

    benign = state.hypothesis(H_BENIGN)
    benign_ruled_out = bool(benign and benign.status == "contradicted")

    # The evidence must actually point at the proposed target.
    target_in_evidence = target_account_id in set(state.primary_entity_ids) or any(
        target_account_id in (record.entity_ids or []) for record in ok_records
    )

    checks = {
        "actions_enabled": actions_enabled,
        "sandbox_graph": sandbox_enabled and graph_is_demo,
        "verdict_supports_review": verdict == VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
        "enough_evidence_records": len(ok_records) >= MIN_EVIDENCE_RECORDS,
        "enough_signal_families": len(families) >= MIN_SIGNAL_FAMILIES,
        "signal_families_implicate_target": bool(relevant_families),
        "enough_distinct_provenance_families": len(counted_families) >= MIN_SIGNAL_FAMILIES,
        "benign_explanation_ruled_out": benign_ruled_out,
        "no_failed_queries": not failed_records,
        "no_material_truncation": not material_truncated,
        "target_supported_by_evidence": target_in_evidence,
        "account_exists": account_exists,
        "simulation_present": simulation is not None,
        "simulation_complete": bool(simulation and simulation.get("simulation_complete")),
        "simulation_fresh": simulation is not None and not simulation_stale,
        "account_not_already_frozen": not account_already_frozen,
    }

    blocking: list[str] = []
    if not checks["actions_enabled"]:
        blocking.append("action tools are not enabled in this configuration")
    if not checks["sandbox_graph"]:
        blocking.append(
            "the configured graph is not the disposable sandbox "
            "(HABEAS_DEMO_MODE + demo-safe graph name required)"
        )
    if not checks["verdict_supports_review"]:
        blocking.append(
            f"canonical verdict is {verdict!r}; only "
            f"{VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW!r} may be actioned"
        )
    if not checks["enough_evidence_records"]:
        blocking.append(
            f"only {len(ok_records)} successful evidence records; "
            f"{MIN_EVIDENCE_RECORDS} required"
        )
    if not checks["enough_signal_families"]:
        blocking.append(
            f"only {len(families)} signal family/families were detected; "
            f"{MIN_SIGNAL_FAMILIES} required"
        )
    if not checks["signal_families_implicate_target"]:
        if families:
            blocking.append(
                f"none of the detected signal families implicate the proposed "
                f"target {target_account_id}: "
                + "; ".join(
                    f"{item['family']}: {item['reason']}"
                    for item in assessment["excluded"]
                )
            )
        else:
            blocking.append(
                "no signal family was detected, so none can implicate the "
                f"proposed target {target_account_id}"
            )
    if not checks["enough_distinct_provenance_families"] and relevant_families:
        findings = _provenance_findings(assessment)
        blocking.append(
            f"only {len(counted_families)} distinct-provenance signal "
            f"family/families implicate {target_account_id}; "
            f"{MIN_SIGNAL_FAMILIES} required"
            + (f" ({'; '.join(findings)})" if findings else "")
        )
    if not checks["benign_explanation_ruled_out"]:
        status = benign.status if benign else "absent"
        blocking.append(f"the benign explanation is not ruled out (status: {status})")
    if not checks["no_failed_queries"]:
        blocking.append(
            f"{len(failed_records)} investigation quer(y/ies) failed, leaving an "
            "evidence gap"
        )
    if not checks["no_material_truncation"]:
        blocking.append(f"material query results were truncated: {material_truncated}")
    if not checks["target_supported_by_evidence"]:
        blocking.append(
            f"account {target_account_id} is not the entity the evidence supports"
        )
    if not checks["account_exists"]:
        blocking.append(f"account {target_account_id} does not exist")
    if not checks["simulation_present"]:
        blocking.append("no simulation is recorded for this case and account")
    elif not checks["simulation_complete"]:
        blocking.append("the recorded simulation is incomplete")
    elif not checks["simulation_fresh"]:
        blocking.append(
            "the simulation is stale: relevant graph topology changed since it ran"
        )
    if not checks["account_not_already_frozen"]:
        blocking.append(f"account {target_account_id} is already frozen")

    caveats: list[str] = []
    if contextual_truncated:
        caveats.append(
            f"contextual query results were truncated ({contextual_truncated}); "
            "the ranking is not exhaustive"
        )
    if (
        assessment["overlapping_pairs"]
        or len(counted_families) < len(relevant_families)
    ):
        caveats.append(
            "signal families can overlap: the same transfers can produce both "
            "circulation and rapid-forwarding evidence, so family counts are not "
            "independent statistical tests"
        )

    # Qualifying evidence: records that back a target-relevant family, plus
    # the target's own context records. A record that does not implicate the
    # proposed target — or that does not support any counted family — never
    # qualifies.
    family_backing = set(assessment["target_backing_evidence_ids"])
    qualifying: list[str] = []
    for record in ok_records:
        include = False
        if record.tool == "alert_context":
            derived = record.derived or {}
            include = target_account_id in (derived.get("entity_ids") or []) or (
                target_account_id in (record.entity_ids or [])
            )
        elif record.tool == "entity_profile":
            params = (record.params[0] if record.params else {}) or {}
            include = target_account_id in (record.entity_ids or []) or (
                target_account_id in {str(v) for v in params.values()}
            )
        else:
            include = record.evidence_id in family_backing
        if include and record.evidence_id not in qualifying:
            qualifying.append(record.evidence_id)

    return EligibilityDecision(
        action="freeze_account",
        allowed=all(checks.values()),
        verdict=verdict,
        blocking_reasons=blocking,
        caveats=caveats,
        signal_families=families,
        target_relevant_families=relevant_families,
        distinct_provenance_families=counted_families,
        qualifying_evidence_ids=qualifying,
        checks=checks,
        provenance=assessment,
    )


def escalation_verdict(state, provisional: str | None) -> str:
    """Canonical verdict, escalating a provisional suspicion only on evidence.

    A Phase 4 ``suspicious_activity`` result is provisional. It becomes
    ``suspicion_supported_for_review`` only when the deterministic evidence
    conditions for escalation hold — never on an LLM's say-so.
    """
    canonical = canonical_verdict(provisional)
    if canonical != VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION:
        return canonical

    from src.agent import analysis
    from src.agent.state import H_BENIGN

    families = analysis.detect_signal_families(state)
    ok_records = [r for r in state.evidence if r.ok]
    failed = [r for r in state.evidence if not r.ok]
    material_truncated = any(
        truncated and qid in MATERIAL_QUERIES
        for record in ok_records
        for qid, truncated in zip(record.query_ids, record.truncated)
    )
    benign = state.hypothesis(H_BENIGN)

    if (
        len(families) >= MIN_SIGNAL_FAMILIES
        and len(ok_records) >= MIN_EVIDENCE_RECORDS
        and not failed
        and not material_truncated
        and benign is not None
        and benign.status == "contradicted"
    ):
        return VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW
    return canonical