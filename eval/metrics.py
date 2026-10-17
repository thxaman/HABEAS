"""Deterministic evaluation metrics (PLAN.md §12, Phase 7 spec).

Everything here is pure Python over per-alert result records produced by
``eval.harness`` (and reproducible from ``eval/results.json`` alone). No
wall-clock value is part of any metric; timings live in a separate section.

Outcome taxonomy — one canonical tier per evaluated alert:

``supported_for_review``
    canonical verdict ``suspicion_supported_for_review``: the deterministic
    policy escalated the investigation for human/action review.
``suspicious_unconfirmed``
    canonical verdict ``suspicious_activity``: a signal was detected, but the
    escalation conditions were not met. Deliberately counted as *neither* a
    positive finding nor benign.
``benign``
    canonical verdict ``likely_benign``.
``abstention``
    ``insufficient_evidence`` / ``incomplete_investigation`` / missing verdict
    without an unrecoverable failure: an explicit "not enough evidence".
``failed``
    the run failed (harness exception, or an investigation that terminated on
    an unrecoverable error such as a provider outage). Failed runs stay in
    every denominator and are never silently re-classified.

Four quantities are deliberately kept apart (never conflated):

1. *pattern detected*  — expected signal families observed in evidence;
2. *verdict supported for review* — the ``supported_for_review`` tier;
3. *freeze-eligible*   — Policy v2 allowed an action (recorded per attempt);
4. *frozen*            — the sandbox mutation actually applied.

Positive/benign/abstention mapping used by recall and FP rate:

- positive (strict)  = ``supported_for_review``;
- positive (lenient) = ``supported_for_review`` ∪ ``suspicious_unconfirmed``
  (reported alongside, never as the headline);
- benign outcome     = ``benign``;
- abstentions        = ``abstention`` (visible in the confusion matrix);
- failures           = ``failed`` (visible; counted in completion metrics).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.agent.state import (
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_EVIDENCE_SUFFICIENT,
    TERMINATION_NO_DISCRIMINATING_QUERY,
    TERMINATION_UNRECOVERABLE_ERROR,
)
from src.agent.policy import (
    VERDICT_INCOMPLETE_INVESTIGATION,
    VERDICT_INSUFFICIENT_EVIDENCE,
    VERDICT_LIKELY_BENIGN,
    VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
    VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION,
)
from src.seed.patterns import (
    SIGNAL_CIRCULAR_TRANSFER,
    SIGNAL_FAN_IN_CONCENTRATION,
    SIGNAL_RAPID_IN_OUT,
    SIGNAL_SINGLE_FORWARDING_DESTINATION,
    SIGNAL_TEMPORAL_CONSISTENCY,
)

OUTCOME_SUPPORTED = "supported_for_review"
OUTCOME_UNCONFIRMED = "suspicious_unconfirmed"
OUTCOME_BENIGN = "benign"
OUTCOME_ABSTENTION = "abstention"
OUTCOME_FAILED = "failed"

OUTCOME_TIERS = (
    OUTCOME_SUPPORTED,
    OUTCOME_UNCONFIRMED,
    OUTCOME_BENIGN,
    OUTCOME_ABSTENTION,
    OUTCOME_FAILED,
)

# Planted signal id -> the independent signal family (agent vocabulary) it
# should surface as. Shared-device/identifier signals are association
# evidence, not a pattern family, and are therefore excluded from the
# pattern-detection denominator (documented, not silently dropped).
SIGNAL_TO_FAMILY = {
    SIGNAL_CIRCULAR_TRANSFER: "circulation",
    SIGNAL_TEMPORAL_CONSISTENCY: "circulation",
    SIGNAL_FAN_IN_CONCENTRATION: "fan_in_out",
    SIGNAL_SINGLE_FORWARDING_DESTINATION: "fan_in_out",
    SIGNAL_RAPID_IN_OUT: "rapid_forwarding",
}

_COMPLETED_TERMINATIONS = {TERMINATION_EVIDENCE_SUFFICIENT, TERMINATION_NO_DISCRIMINATING_QUERY}

_VERIFIER_PASS_STATUSES = (
    "reproduced_current_state",
    "verified_against_recorded_evidence",
    "changed_after_action",
)


def outcome_tier(record: dict[str, Any]) -> str:
    """Map one alert record to its canonical outcome tier (see module doc)."""
    if record.get("status") != "ok":
        return OUTCOME_FAILED
    if record.get("termination_reason") == TERMINATION_UNRECOVERABLE_ERROR:
        return OUTCOME_FAILED
    canonical = record.get("canonical_verdict")
    if canonical == VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW:
        return OUTCOME_SUPPORTED
    if canonical == VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION:
        return OUTCOME_UNCONFIRMED
    if canonical == VERDICT_LIKELY_BENIGN:
        return OUTCOME_BENIGN
    if canonical in (VERDICT_INSUFFICIENT_EVIDENCE, VERDICT_INCOMPLETE_INVESTIGATION):
        return OUTCOME_ABSTENTION
    return OUTCOME_ABSTENTION


def metric(numerator: int, denominator: int) -> dict[str, Any]:
    """Headline metric shape: numerator, denominator, rate, display."""
    value: dict[str, Any] = {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": (round(numerator / denominator, 4) if denominator else None),
        "sample_size": int(denominator),
    }
    value["display"] = (
        f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"
        if denominator
        else f"{numerator}/0 (n/a)"
    )
    return value


def describe(values: list[float]) -> dict[str, Any]:
    """Sample statistics for one distribution (descriptive, not a metric)."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None, "sum": 0}
    ordered = sorted(values)
    n = len(ordered)
    # nearest-rank 90th percentile, clamped to the last sample
    p90_index = max(0, min(n - 1, math.ceil(0.9 * n) - 1))
    return {
        "n": n,
        "mean": round(sum(ordered) / n, 3),
        "median": round(ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2, 3),
        "p90": round(ordered[p90_index], 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "sum": round(sum(ordered), 3),
    }


def expected_signal_families(record: dict[str, Any]) -> list[str]:
    families = {
        SIGNAL_TO_FAMILY[signal]
        for signal in record.get("gt_signals") or []
        if signal in SIGNAL_TO_FAMILY
    }
    return sorted(families)


def completed(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "ok"
        and record.get("termination_reason") in _COMPLETED_TERMINATIONS
    )


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute every headline metric from the per-alert records.

    The per-record ``gt_*`` fields come from the validated manifest, so the
    denominators are manifest-derived; a bounded subset naturally scores only
    the alerts that were evaluated (clearly labelled by the runner).
    """
    tiers = {r["alert_id"]: outcome_tier(r) for r in records}
    suspicious = [r for r in records if r.get("gt_label") == "suspicious"]
    benign = [r for r in records if r.get("gt_label") == "benign"]

    tp_strict = sum(tiers[r["alert_id"]] == OUTCOME_SUPPORTED for r in suspicious)
    tp_lenient = sum(
        tiers[r["alert_id"]] in (OUTCOME_SUPPORTED, OUTCOME_UNCONFIRMED) for r in suspicious
    )
    fp_strict = sum(tiers[r["alert_id"]] == OUTCOME_SUPPORTED for r in benign)
    fp_lenient = sum(
        tiers[r["alert_id"]] in (OUTCOME_SUPPORTED, OUTCOME_UNCONFIRMED) for r in benign
    )
    abstain = sum(tiers[r["alert_id"]] == OUTCOME_ABSTENTION for r in records)
    failed = sum(tiers[r["alert_id"]] == OUTCOME_FAILED for r in records)

    # ---- confusion matrix (inconclusive and failed outcomes stay visible) --
    confusing_rows: dict[str, dict[str, int]] = {}
    for label, group in (("suspicious", suspicious), ("benign", benign)):
        row = {tier: 0 for tier in OUTCOME_TIERS}
        for record in group:
            row[tiers[record["alert_id"]]] += 1
        row["total"] = len(group)
        confusing_rows[label] = row

    # ---- completion / termination ------------------------------------------
    completed_count = sum(completed(r) for r in records)
    budget_exhausted = sum(
        r.get("termination_reason") == TERMINATION_BUDGET_EXHAUSTED for r in records
    )
    early_termination = sum(
        r.get("termination_reason") == TERMINATION_UNRECOVERABLE_ERROR
        or r.get("status") != "ok"
        for r in records
    )

    # ---- pattern-family coverage (denominator from the manifest) -----------
    family_names = sorted({r["gt_family"] for r in suspicious})
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in suspicious:
        by_family.setdefault(record["gt_family"], []).append(record)
    family_detail: dict[str, Any] = {}
    fully_covered = 0
    any_detected = 0
    for family in family_names:
        group = by_family[family]
        detected = [r["alert_id"] for r in group if tiers[r["alert_id"]] == OUTCOME_SUPPORTED]
        detected_lenient = [
            r["alert_id"]
            for r in group
            if tiers[r["alert_id"]] in (OUTCOME_SUPPORTED, OUTCOME_UNCONFIRMED)
        ]
        missed = [r for r in group if r["alert_id"] not in detected]
        if detected_lenient:
            any_detected += 1
        if len(detected) == len(group):
            fully_covered += 1
        family_detail[family] = {
            "suspicious_total": len(group),
            "detected": detected,
            "missed": [r["alert_id"] for r in missed],
            "detected_lenient": detected_lenient,
            "detection_rate": metric(len(detected), len(group)),
            "detection_rate_lenient": metric(len(detected_lenient), len(group)),
        }

    # ---- action / freeze outcomes (kept separate from verdicts) ------------
    attempted = [r for r in records if (r.get("action") or {}).get("attempted")]
    eligible = [r for r in attempted if (r["action"] or {}).get("eligible") is True]
    applied = [r for r in records if (r.get("action") or {}).get("status") == "applied"]
    declined = [r for r in attempted if (r["action"] or {}).get("status") == "declined"]
    already = [r for r in attempted if (r["action"] or {}).get("status") == "already_applied"]
    in_progress = [r for r in attempted if (r["action"] or {}).get("status") == "in_progress"]
    action_failures = [
        r for r in attempted if (r["action"] or {}).get("status") in (None, "error")
    ]
    blocking_checks: Counter[str] = Counter()
    blocking_reasons: Counter[str] = Counter()
    for record in attempted:
        action = record.get("action") or {}
        for check in action.get("failing_checks") or []:
            blocking_checks[str(check)] += 1
        for reason in action.get("blocking_reasons") or []:
            blocking_reasons[str(reason)] += 1

    # ---- verifier integrity ------------------------------------------------
    verifier_totals: Counter[str] = Counter()
    verified_cases = 0
    verifier_errors = 0
    for record in records:
        info = record.get("verifier") or {}
        if info.get("status") == "ran":
            verified_cases += 1
            for status, count in (info.get("summary") or {}).items():
                verifier_totals[str(status)] += int(count)
        elif info.get("status") == "error":
            verifier_errors += 1
    verifier_checked = sum(verifier_totals.values())
    verifier_pass = sum(verifier_totals.get(s, 0) for s in _VERIFIER_PASS_STATUSES)

    # ---- pattern detection (separate from verdicts, documented above) ------
    detection_denominator = 0
    detection_hits = 0
    for record in suspicious:
        expected = expected_signal_families(record)
        if not expected:
            continue
        detection_denominator += 1
        observed = set(record.get("signal_families") or [])
        if set(expected) <= observed:
            detection_hits += 1

    # ---- tool calls (deterministic, part of the scorecard) ------------------
    call_values = [float(r.get("tool_calls") or 0) for r in records]
    suspicious_calls = [float(r.get("tool_calls") or 0) for r in suspicious]
    benign_calls = [float(r.get("tool_calls") or 0) for r in benign]

    return {
        "counts": {
            "evaluated": len(records),
            "suspicious": len(suspicious),
            "benign": len(benign),
            "families": len(family_names),
        },
        "recall": {
            "strict": metric(tp_strict, len(suspicious)),
            "lenient": metric(tp_lenient, len(suspicious)),
            "definition": (
                "strict positive = suspicion_supported_for_review; lenient adds "
                "suspicious_activity (calculated and reported, never a headline)"
            ),
        },
        "false_positive_rate": {
            "strict": metric(fp_strict, len(benign)),
            "lenient": metric(fp_lenient, len(benign)),
        },
        "inconclusive_rate": {
            "overall": metric(abstain, len(records)),
            "suspicious": metric(
                sum(tiers[r["alert_id"]] == OUTCOME_ABSTENTION for r in suspicious),
                len(suspicious),
            ),
            "benign": metric(
                sum(tiers[r["alert_id"]] == OUTCOME_ABSTENTION for r in benign),
                len(benign),
            ),
        },
        "failure_rate": metric(failed, len(records)),
        "completion": {
            "completed": metric(completed_count, len(records)),
            "tool_budget_exhausted": metric(budget_exhausted, len(records)),
            "early_termination": metric(early_termination, len(records)),
        },
        "confusion": {"columns": list(OUTCOME_TIERS), "rows": confusing_rows},
        "pattern_coverage": {
            "strict": metric(fully_covered, len(family_names)),
            "lenient": metric(any_detected, len(family_names)),
            "by_family": family_detail,
            "definition": (
                "strict = planted families whose every suspicious alert reached "
                "supported_for_review; lenient = families with at least one "
                "detected alert; denominator = distinct planted families"
            ),
        },
        "pattern_detection": {
            "expected_families_observed": metric(detection_hits, detection_denominator),
            "definition": (
                "suspicious alerts whose planted signal families were all observed "
                "in the recorded signal families; shared-device/identifier signals "
                "are association evidence and excluded from this denominator"
            ),
        },
        "action": {
            "attempted": metric(len(attempted), len(records)),
            "eligible_for_freeze": metric(len(eligible), len(attempted)),
            "applied": metric(len(applied), len(attempted)),
            "declined": metric(len(declined), len(attempted)),
            "already_applied": metric(len(already), len(attempted)),
            "in_progress": metric(len(in_progress), len(attempted)),
            "failed": metric(len(action_failures), len(attempted)),
            "blocking_check_counts": dict(sorted(blocking_checks.items())),
            "blocking_reason_counts": dict(blocking_reasons.most_common(20)),
            "definition": (
                "eligible = Policy v2 recorded allowed=true on the freeze attempt; "
                "applied = the sandbox mutation was performed; declined/already/"
                "failed are separate outcomes. Eligibility is NOT propagation: a "
                "declined case can be correctly declined"
            ),
        },
        "verifier": {
            "cases_verified": verified_cases,
            "cases_with_error": verifier_errors,
            "claims_checked": metric(verifier_pass, verifier_checked),
            "status_counts": {status: verifier_totals.get(status, 0) for status in _VERIFIER_PASS_STATUSES}
            | {status: verifier_totals.get(status, 0) for status in ("mismatch", "unverifiable")},
            "definition": (
                "pass = reproduced_current_state | verified_against_recorded_evidence | "
                "changed_after_action (the last requires an action record on the "
                "case); mismatch and unverifiable are failures"
            ),
        },
        "tool_calls": {
            "total": int(sum(call_values)),
            "per_alert": {r["alert_id"]: int(r.get("tool_calls") or 0) for r in records},
            "all": describe(call_values),
            "suspicious": describe(suspicious_calls),
            "benign": describe(benign_calls),
        },
    }


def pattern_diagnostics(
    records: list[dict[str, Any]], denominators: dict[str, Any]
) -> dict[str, Any]:
    """Per-planted-family diagnostics, reproducible from ``results.json``.

    For every family in the validated manifest: positives, detections, misses
    (with representative reasons and unresolved hypotheses), the evidence
    tools used, target-specific / provenance-qualified evidence counts, Policy
    v2 eligibility, and — where applicable — benign alerts that share pattern
    accounts and were flagged.
    """
    tiers = {r["alert_id"]: outcome_tier(r) for r in records}
    by_id = {r["alert_id"]: r for r in records}
    benign_records = [r for r in records if r.get("gt_label") == "benign"]
    flagged_benign = [
        r for r in benign_records
        if tiers[r["alert_id"]] in (OUTCOME_SUPPORTED, OUTCOME_UNCONFIRMED)
    ]

    families: dict[str, Any] = {}
    not_evaluated: list[str] = []
    for family, alert_ids in sorted(denominators.get("suspicious_by_family", {}).items()):
        group = [by_id[alert_id] for alert_id in alert_ids if alert_id in by_id]
        if not group:
            not_evaluated.append(family)
            continue
        detected = [r["alert_id"] for r in group if tiers[r["alert_id"]] == OUTCOME_SUPPORTED]
        detected_lenient = [
            r["alert_id"] for r in group
            if tiers[r["alert_id"]] in (OUTCOME_SUPPORTED, OUTCOME_UNCONFIRMED)
        ]
        tools_used: Counter = Counter()
        target_relevant = 0
        distinct_provenance = 0
        eligible = 0
        unresolved: dict[str, list[str]] = {}
        reasons: dict[str, list[str]] = {}
        for record in group:
            tools_used.update(record.get("tool_sequence") or [])
            action = record.get("action") or {}
            if action.get("target_relevant_families"):
                target_relevant += 1
            if len(action.get("distinct_provenance_families") or []) >= 2:
                distinct_provenance += 1
            if action.get("eligible") is True:
                eligible += 1
            if record["alert_id"] not in detected_lenient:
                unresolved[record["alert_id"]] = [
                    h.get("name") for h in record.get("hypotheses") or []
                    if h.get("status") in ("open", "insufficient")
                ]
                why = []
                if record.get("status") != "ok":
                    why.append(record.get("error") or "harness error")
                why.extend(record.get("errors") or [])
                why.extend((record.get("limitations") or [])[:2])
                if not why:
                    why.append(
                        f"verdict {record.get('canonical_verdict')!r} after "
                        f"{record.get('tool_calls')} tool call(s); termination "
                        f"{record.get('termination_reason')!r}"
                    )
                reasons[record["alert_id"]] = why[:3]
        member_accounts = set(denominators.get("pattern_account_members", {}).get(family, []))
        fp_same_members = [
            r["alert_id"] for r in flagged_benign
            if r.get("gt_account_id") in member_accounts
        ]
        families[family] = {
            "suspicious_total": len(group),
            "detected": detected,
            "detected_lenient": detected_lenient,
            "missed": [r["alert_id"] for r in group if r["alert_id"] not in detected],
            "detection_rate": metric(len(detected), len(group)),
            "evidence_tools_used": dict(tools_used.most_common()),
            "target_relevant_evidence": metric(target_relevant, len(group)),
            "distinct_provenance_evidence": metric(distinct_provenance, len(group)),
            "freeze_eligible": metric(eligible, len(group)),
            "unresolved_hypotheses_by_miss": unresolved,
            "failure_reasons_by_miss": reasons,
            "benign_false_positives_sharing_pattern_accounts": fp_same_members,
        }

    return {
        "by_family": families,
        "families_not_evaluated": not_evaluated,
        "benign_false_positives": [r["alert_id"] for r in flagged_benign],
    }


def checklist_failures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Missed suspicious alerts, false positives, failures, unresolved items.

    Reproducible from results.json; the Markdown scorecard and the tests use
    this same function so the report can never drift from the data.
    """
    items: list[dict[str, Any]] = []
    for record in records:
        tier = outcome_tier(record)
        label = record.get("gt_label")
        reasons: list[str] = []
        if record.get("status") != "ok":
            reasons.append(record.get("error") or "harness error")
        reasons.extend(record.get("errors") or [])
        reasons.extend(record.get("limitations") or [])
        if tier == OUTCOME_FAILED and not reasons:
            reasons.append("investigation terminated on an unrecoverable error")
        if label == "suspicious" and tier not in (OUTCOME_SUPPORTED, OUTCOME_UNCONFIRMED):
            items.append({
                "kind": "missed_suspicious_alert",
                "alert_id": record["alert_id"],
                "gt_family": record.get("gt_family"),
                "outcome_tier": tier,
                "canonical_verdict": record.get("canonical_verdict"),
                "termination_reason": record.get("termination_reason"),
                "unresolved_hypotheses": [
                    h.get("name") for h in record.get("hypotheses") or []
                    if h.get("status") in ("open", "insufficient")
                ],
                "reasons": reasons[:3],
            })
        if label == "benign" and tier in (OUTCOME_SUPPORTED, OUTCOME_UNCONFIRMED):
            items.append({
                "kind": "false_positive_on_benign",
                "alert_id": record["alert_id"],
                "outcome_tier": tier,
                "canonical_verdict": record.get("canonical_verdict"),
                "reasons": reasons[:3],
            })
        if tier == OUTCOME_FAILED or (record.get("errors") or []) or record.get("status") != "ok":
            items.append({
                "kind": "run_problem",
                "alert_id": record["alert_id"],
                "status": record.get("status"),
                "termination_reason": record.get("termination_reason"),
                "truncated_evidence": record.get("truncated_evidence") or [],
                "errors": record.get("errors") or [],
                "harness_error": record.get("error"),
            })
        elif record.get("truncated_evidence"):
            items.append({
                "kind": "truncation_notice",
                "alert_id": record["alert_id"],
                "truncated_evidence": record["truncated_evidence"],
            })
    return items
