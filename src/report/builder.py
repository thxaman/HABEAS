"""Phase 6 deterministic report builder (PLAN.md §11) — no LLM.

The report explains what HABEAS investigated, which graph evidence supported
or contradicted each hypothesis, what intervention was considered, why the
action was permitted or declined, what the graph looks like afterwards, and
which claims could be independently reproduced by ``src.report.verifier``.

Everything material is referenced: prose statements point at persisted
evidence ids (or are explicitly attributed to case/action audit history).
The builder never invents timestamps, values, explanations or results: missing
data is reported as missing, and limitations (budget exhaustion, truncation,
overlaps between signal families, unreplayable snapshots) are carried into the
report instead of being smoothed over.

Phase 6.1 (claim-level coverage): each evidence record's persisted source
manifest is rendered with its per-source verification outcome, so a claim is
never presented as fully verified from a subset of the queries that produced
it, and the report states which supporting queries were checked and which
remain unavailable.

Reports are risk indicators for human review. They never assert criminality,
and they never claim that money was frozen: the only freeze in this system is a
simulated graph mutation that changes an account's status flag and deactivates
routing relationships.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.agent import analysis
from src.agent.policy import POLICY_VERSION
from src.agent.state import (
    ALL_HYPOTHESES,
    H_BENIGN,
    H_CIRCULATION,
    H_FAN_IN_OUT,
    H_INSUFFICIENT,
    H_RAPID_MOVEMENT,
    H_SHARED_IDENTITY,
    TERMINATION_BUDGET_EXHAUSTED,
)
from src.report.verifier import (
    ALL_STATUSES,
    STATUS_REPRODUCED,
    STATUS_UNVERIFIABLE,
    STATUS_VERIFIED_RECORDED,
    CaseRecords,
    CaseVerification,
    load_case_records,
    verify_case,
)

# phase6-v2: §7 additionally records the policy version that authorized or
# declined the intervention, so historical v1 authorizations are never
# re-presented as current-policy outcomes.
# phase6-v3 (corrective Phase 6.1): every citation carries a complete
# per-sub-query source manifest; §5/§10 render the per-source verification
# (which supporting queries were checked, which remain unavailable) and no
# claim is presented as fully verified from a subset of its sources.
REPORT_VERSION = "phase6-v3"

_HYPOTHESIS_TITLES = {
    H_CIRCULATION: "Chronologically corroborated circulation",
    H_FAN_IN_OUT: "Coordinated fan-in / fan-out cash-out",
    H_RAPID_MOVEMENT: "Rapid movement of incoming funds",
    H_SHARED_IDENTITY: "Shared identifier link warranting review",
    H_BENIGN: "Benign high-volume or shared-infrastructure explanation",
    H_INSUFFICIENT: "Evidence insufficient to distinguish explanations",
}

_FAMILY_TITLES = {
    "circulation": "circulation",
    "fan_in_out": "fan-in/fan-out",
    "rapid_forwarding": "rapid forwarding",
}

_OVERLAP_SENTENCES = {
    "distinct_observations": (
        "supported by distinct underlying transaction events (no shared ids)"
    ),
    "partially_shared": (
        "supported by partially overlapping underlying transaction events"
    ),
    "shared_observations": (
        "supported by the same underlying transaction events (one family's "
        "observations are contained in the other's)"
    ),
    "provenance_unavailable": (
        "not resolvable to individual transactions for at least one family, so "
        "their independence cannot be established"
    ),
}

_METRIC_LABELS = {
    "simulation_complete": "Simulation completed with full coverage",
    "active_edges_total": "Active routing edges graph-wide",
    "active_edges_disabled": "Active routing edges the freeze deactivates",
    "downstream_accounts_before": "Unique accounts reachable before (bounded model)",
    "downstream_accounts_after": "Unique accounts reachable after (bounded model)",
    "newly_unreachable_accounts": "Accounts that would become unreachable",
    "direct_outgoing_volume": (
        "Historical outgoing transaction volume in the window (synthetic units; "
        "not an amount that was frozen or moved by this action)"
    ),
    "direct_outgoing_transaction_count": "Historical outgoing transactions in the window",
    "disabled_edges": "Routing edges this freeze would deactivate",
    "newly_unreachable_count": "Count of accounts that would become unreachable",
    "truncated_queries": "Bounded queries that hit their row cap during simulation",
    "excluded": "Model exclusions (what these metrics deliberately do not represent)",
    "limits": "Simulation bounds (hops, window, topology cap)",
    "metric_semantics": "Metric definitions",
    "window": "Observation window used for the simulation",
    "account_id": "Account the simulation concerns",
    "account_status": "Account status at simulation time",
    "hops": "Hop bound used",
}

_DISCLAIMER = (
    "HABEAS operates on synthetic data only. This report describes risk "
    "indicators for human review; it is not a determination of criminality or "
    "wrongdoing. The freeze mechanism in this system performs a simulated "
    "graph mutation only: it sets an account's status to frozen and "
    "deactivates its routing relationships. It does not move, block, seize or "
    "refund any money."
)


# ---- small deterministic helpers ---------------------------------------------


def _md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def _metric_value(value: Any) -> str:
    """Render a simulation metric value; nested structures as canonical JSON."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return str(value)


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sort_evidence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (str(r.get("captured_at") or ""), str(r.get("evidence_id") or "")),
    )


def _json_field(record: Mapping[str, Any], key: str, fallback: Any) -> Any:
    raw = record.get(key)
    if raw in (None, ""):
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _evidence_id_map(case_id: str, state, records: CaseRecords) -> dict[str, str]:
    """Map investigation-time evidence ids (``EV-XXXX``) to persisted ids.

    ``cite_evidence`` stores each record under a case-scoped id
    (``CASE-...:EV-XXXX``) while the investigation state, policy clauses and
    hypothesis references use the local id. Every report reference must name
    the persisted id, so references are resolved through this map.
    """
    persisted = {str(r.get("evidence_id")) for r in records.evidence}
    mapping: dict[str, str] = {}
    if state is not None:
        for record in state.evidence:
            local = record.evidence_id
            scoped = f"{case_id}:{local}"
            if scoped in persisted:
                mapping[local] = scoped
            elif local in persisted:
                mapping[local] = local
    return mapping


def _resolve(evidence_id: str, id_map: Mapping[str, str]) -> str:
    return id_map.get(evidence_id, evidence_id)


@dataclass
class ReportArtifact:
    case_id: str
    generated_at: str
    report_version: str
    markdown: str
    payload: dict[str, Any]
    fingerprint: str
    verification: CaseVerification


# ---- section builders ----------------------------------------------------------


def _claim_references(state, id_map: Mapping[str, str]) -> list[dict[str, Any]]:
    """Material claims mapped to the evidence ids that back them."""
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    if state is not None:
        for hypothesis in state.hypotheses:
            ids = list(dict.fromkeys(
                _resolve(eid, id_map)
                for eid in hypothesis.supporting_evidence + hypothesis.contradicting_evidence
            ))
            key = (hypothesis.name, tuple(ids))
            if key in seen:
                continue
            seen.add(key)
            refs.append({
                "claim": (
                    f"Hypothesis '{hypothesis.name}' assessed as "
                    f"{hypothesis.status}"
                ),
                "source": "hypothesis_assessment",
                "evidence_ids": ids,
            })
        for claim in state.claims:
            ids = list(dict.fromkeys(_resolve(eid, id_map) for eid in claim.evidence_ids))
            key = (claim.claim, tuple(ids))
            if key in seen:
                continue
            seen.add(key)
            refs.append({
                "claim": claim.claim,
                "source": "investigation_claim",
                "evidence_ids": ids,
            })
    return refs


def _qualify_family_language(text: str, provenance: dict[str, Any] | None) -> str:
    """Never carry an unqualified 'independent signal families' phrase into a
    report when the provenance audit shows overlap or unproven independence.

    The policy/analysis wording is designed for the investigation loop; the
    report is additionally constrained by the provenance audit (PLAN.md §11).
    """
    if not text or not provenance:
        return text
    if not (provenance.get("overlapping_pairs")
            or provenance.get("unavailable_pairs")):
        return text
    return text.replace(
        "independent signal families",
        "signal families whose evidential independence is not established "
        "(see provenance caveats)",
    )


def _executive_findings(state, records: CaseRecords, provenance: dict[str, Any] | None) -> dict[str, Any]:
    hypotheses = list(state.hypotheses) if state is not None else []
    supported = [h for h in hypotheses if h.status == "supported"]
    leading = supported[0] if supported else None
    if leading is None and hypotheses:
        order = {name: i for i, name in enumerate(ALL_HYPOTHESES)}
        leading = sorted(
            hypotheses,
            key=lambda h: (order.get(h.name, 99), h.name),
        )[0]
    caveats: list[str] = []
    if provenance:
        for pair in provenance.get("pairs", []):
            if pair["classification"] in ("shared_observations", "partially_shared"):
                caveats.append(
                    f"Signal families {pair['pair'][0]} and {pair['pair'][1]} are "
                    f"{_OVERLAP_SENTENCES[pair['classification']]} "
                    f"({pair['shared_count']} shared transaction id(s)); they are "
                    "feature diversity over the same activity, not independent "
                    "corroboration."
                )
            elif pair["classification"] == "provenance_unavailable":
                caveats.append(
                    f"Signal families {pair['pair'][0]} and {pair['pair'][1]} are "
                    f"{_OVERLAP_SENTENCES['provenance_unavailable']}; independence "
                    "is not claimed."
                )
    findings: dict[str, Any] = {
        "supported_hypothesis_count": len(supported),
        "leading_hypothesis": (
            {
                "name": leading.name,
                "title": _HYPOTHESIS_TITLES.get(leading.name, leading.name),
                "status": leading.status,
                "supporting_evidence_ids": list(leading.supporting_evidence),
                "contradicting_evidence_ids": list(leading.contradicting_evidence),
                "rationale": _qualify_family_language(leading.rationale, provenance),
            }
            if leading is not None
            else None
        ),
        "verdict": records.case.get("verdict") if records.case else None,
        "verdict_rationale": _qualify_family_language(
            state.rationale if state is not None else "", provenance
        ),
        "provenance_caveats": caveats,
        "qualification": (
            "The findings below are prioritisation indicators for human review. "
            "They are derived from deterministic Python analysis of the recorded "
            "tool evidence; they are not a finding of wrongdoing."
        ),
    }
    return findings


def _hypothesis_section(
    state, verification: CaseVerification, id_map: Mapping[str, str]
) -> list[dict[str, Any]]:
    if state is None:
        return []
    status_by_id = {c.evidence_id: c for c in verification.claims}
    out: list[dict[str, Any]] = []
    for hypothesis in state.hypotheses:
        def links(ids: list[str]) -> list[dict[str, Any]]:
            return [
                {
                    "evidence_id": _resolve(eid, id_map),
                    "verification_status": (
                        status_by_id[_resolve(eid, id_map)].status
                        if _resolve(eid, id_map) in status_by_id
                        else STATUS_UNVERIFIABLE
                    ),
                }
                for eid in dict.fromkeys(ids)
            ]
        out.append({
            "name": hypothesis.name,
            "title": _HYPOTHESIS_TITLES.get(hypothesis.name, hypothesis.name),
            "status": hypothesis.status,
            "supporting": links(hypothesis.supporting_evidence),
            "contradicting": links(hypothesis.contradicting_evidence),
            "uncertainties": list(hypothesis.uncertainties),
            "rationale": hypothesis.rationale,
            "note": (
                "This hypothesis remains untested to a conclusion; the report "
                "does not treat it as disproven."
                if hypothesis.status in ("open",)
                else ""
            ),
        })
    return out


def _timeline_section(state, records: CaseRecords) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if state is not None and state.investigation_steps:
        evidence_iter = iter(_sort_evidence(records.evidence))
        current = next(evidence_iter, None)
        for step in state.investigation_steps:
            entry: dict[str, Any] = {
                "index": step.index,
                "stage": step.stage,
                "tool": step.tool,
                "purpose": step.purpose,
                "arguments": _truncate(json.dumps(step.arguments, sort_keys=True), 300),
                "result_summary": step.result_summary,
                "assessment_update": step.hypothesis_update,
                "ok": step.ok,
                "evidence_id": None,
                "captured_at": None,
            }
            if step.tool and current is not None and current.get("source_tool") == step.tool:
                entry["evidence_id"] = current.get("evidence_id")
                entry["captured_at"] = current.get("captured_at")
                current = next(evidence_iter, None)
            elif step.tool:
                entry["evidence_id"] = None
            entries.append(entry)
        return {"source": "investigation_step_trace", "entries": entries}
    for record in _sort_evidence(records.evidence):
        entries.append({
            "index": None,
            "stage": "persisted",
            "tool": record.get("source_tool"),
            "purpose": "",
            "arguments": "",
            "result_summary": _truncate(str(record.get("interpretation") or ""), 300),
            "assessment_update": "",
            "ok": True,
            "evidence_id": record.get("evidence_id"),
            "captured_at": record.get("captured_at"),
        })
    return {
        "source": "persisted_evidence_records"
        + ("" if records.evidence else "; no evidence records"),
        "entries": entries,
    }


def _evidence_ledger(
    records: CaseRecords, state, verification: CaseVerification,
    id_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    claim_by_id = {c.evidence_id: c for c in verification.claims}
    state_record_by_id = {}
    if state is not None:
        state_record_by_id = {
            _resolve(r.evidence_id, id_map): r for r in state.evidence
        }
    rows: list[dict[str, Any]] = []
    for record in _sort_evidence(records.evidence):
        evidence_id = str(record.get("evidence_id"))
        source_record = state_record_by_id.get(evidence_id)
        claim = claim_by_id.get(evidence_id)
        # Phase 6.1: the complete per-source manifest with its verification
        # outcome. The persisted record's own manifest (when present) is the
        # ground truth; the in-memory investigation record is a fallback for
        # records that were never verified (which cannot normally happen).
        sources: list[dict[str, Any]] = []
        if claim is not None and claim.sources:
            for source in claim.sources:
                sources.append({
                    "query_id": source.query_id,
                    "role": source.role,
                    "required": source.required,
                    "status": source.status,
                    "reason": source.reason,
                    "replay": source.replay,
                })
        elif source_record is not None:
            recorded_sources = _json_field(
                record, "sources_json", None
            )
            if isinstance(recorded_sources, list):
                for entry in recorded_sources:
                    if isinstance(entry, dict) and entry.get("query_id"):
                        sources.append({
                            "query_id": entry.get("query_id"),
                            "role": entry.get("role") or "claim_source",
                            "required": entry.get("query_id") != "txn_max_ts_v1",
                            "status": STATUS_UNVERIFIABLE,
                            "reason": "not independently verified",
                            "replay": None,
                        })
        if not sources:
            primary_qid = record.get("query_id")
            if primary_qid:
                sources = [{
                    "query_id": primary_qid,
                    "role": "claim_source",
                    "required": True,
                    "status": (
                        claim.status if claim is not None else STATUS_UNVERIFIABLE
                    ),
                    "reason": claim.reason if claim is not None else "not independently verified",
                    "replay": claim.replay if claim is not None else None,
                }]
        compound = bool(claim.compound) if claim is not None else False
        if compound:
            outcomes = ", ".join(
                f"`{s['query_id']}` ({s['status']})" for s in sources
            )
            coverage_note = (
                f"{len(sources)} recorded source quer"
                f"{'y' if len(sources) == 1 else 'ies'}, each verified "
                f"individually: {outcomes}"
            )
        else:
            coverage_note = (
                "legacy citation: only the cited query id was recorded; "
                "sub-query coverage was not captured (pre-6.1 record)"
            )
        rows.append({
            "evidence_id": evidence_id,
            "claim": record.get("claim"),
            "stance": record.get("stance"),
            "query_id": record.get("query_id"),
            "source_tool": record.get("source_tool"),
            "params": _json_field(record, "params_json", {}),
            "interpolation": _json_field(record, "interpolation_json", {}),
            "result_hash": record.get("result_hash"),
            "row_count": record.get("row_count"),
            "truncated": bool(record.get("truncated")),
            "evidence_kind": record.get("evidence_kind"),
            "captured_at": record.get("captured_at"),
            "interpretation": record.get("interpretation"),
            "snapshot_rows_preserved": bool(
                _json_field(record, "rows_json", [])
            ),
            "sources": sources,
            "source_count": (
                claim.source_count if claim is not None else len(sources)
            ),
            "compound": compound,
            "required_source_queries": (
                list(claim.required_source_queries) if claim is not None else []
            ),
            "sub_queries_of_call": [s["query_id"] for s in sources],
            "source_coverage_note": coverage_note,
            "verification_status": (
                claim.status if claim is not None else STATUS_UNVERIFIABLE
            ),
            "verification_reason": claim.reason if claim is not None else "",
        })
    return rows


def _transaction_paths(state, id_map: Mapping[str, str]) -> dict[str, Any]:
    structural: list[dict[str, Any]] = []
    corroborated: list[dict[str, Any]] = []
    leg_events: list[dict[str, Any]] = []
    rapid: list[dict[str, Any]] = []
    if state is not None:
        for record in state.evidence:
            record_eid = _resolve(record.evidence_id, id_map)
            if record.tool == "detect_cycle" and record.ok:
                for cycle in record.derived.get("cycles", []):
                    entry = {
                        "members": list(cycle.get("members") or []),
                        "length": cycle.get("length"),
                        "legs": list(cycle.get("legs") or []),
                        "evidence_id": record_eid,
                        "completed": bool(cycle.get("temporally_corroborated")),
                        "event_count": cycle.get("event_count"),
                        "events": [
                            {
                                "txn_id": e.get("txn_id"),
                                "src": e.get("src"),
                                "dst": e.get("dst"),
                                "ts": e.get("ts"),
                                "amount": e.get("amount"),
                            }
                            for e in (cycle.get("events") or [])[:24]
                        ],
                    }
                    (corroborated if entry["completed"] else structural).append(entry)
            if record.tool == "trace_funds" and record.ok:
                for leg in record.derived.get("leg_events") or []:
                    leg_events.append({
                        "src": leg.get("src"), "dst": leg.get("dst"),
                        "ts": leg.get("ts"), "amount": leg.get("amount"),
                        "txn_id": leg.get("txn_id"),
                        "evidence_id": record_eid,
                        "direction": record.derived.get("direction"),
                    })
        # Rapid in/out timing is judged over the UNION of an account's trace
        # records (an outbound trace alone never contains the inbound leg),
        # exactly as the signal analysis does; per-record matches would miss it.
        for account_id in state.primary_entity_ids or []:
            agg = analysis.rapid_forwarding_for_account(state, account_id)
            agg_ids = [
                _resolve(eid, id_map) for eid in (agg.get("evidence_ids") or [])
            ]
            for match in agg.get("matches") or []:
                rapid.append({
                    "account_id": account_id,
                    "in_ts": match.get("in_ts"), "out_ts": match.get("out_ts"),
                    "gap_minutes": match.get("gap_minutes"),
                    "in_amount": match.get("in_amount"),
                    "out_amount": match.get("out_amount"),
                    "in_txn_id": match.get("in_txn_id"),
                    "out_txn_id": match.get("out_txn_id"),
                    "in_src": match.get("in_src"),
                    "out_dst": match.get("out_dst"),
                    "evidence_id": ", ".join(agg_ids) if agg_ids else None,
                })
        rapid = sorted(
            rapid,
            key=lambda m: (str(m.get("account_id")), str(m.get("in_ts")),
                           str(m.get("out_ts"))),
        )
    deleted = [e["txn_id"] for e in leg_events if e.get("txn_id") is None]
    notes = [
        "Structural connectivity (a ring in the aggregate topology) is "
        "distinguished from temporally corroborated circulation (timestamped "
        "transactions chaining around the ring in order): only the latter is "
        "reported as circulation.",
        "Trace paths show currently-active bounded routes between accounts; the "
        "timestamped leg events beside them are the transaction-level evidence "
        "behind the rapid-forwarding signal.",
    ]
    if deleted:
        notes.append(
            "Some leg events carried no transaction id in the tool result; they "
            "are shown without one rather than assumed."
        )
    return {
        "structural_cycles": structural,
        "corroborated_cycles": corroborated,
        "observed_leg_events": leg_events,
        "rapid_matches": rapid,
        "notes": notes,
    }


def _intervention_section(
    records: CaseRecords, envelopes: Mapping[str, Any] | None
) -> dict[str, Any]:
    simulation = records.simulation
    sim_metrics = (
        _json_field(simulation, "metrics_json", {}) if simulation else {}
    )
    freeze_env = (envelopes or {}).get("freeze_account") or {}
    freeze_result = (freeze_env.get("result") or {}) if freeze_env else {}
    applied_actions = [
        a for a in records.actions
        if a.get("action") == "freeze_account" and a.get("status") == "applied"
    ]
    policy = freeze_result.get("policy")
    policy_source = "recorded freeze_account result"
    if policy is None and applied_actions:
        policy = _json_field(applied_actions[0], "policy_json", {}) or None
        policy_source = "persisted Action policy_json (historical authorization)"
    if freeze_result:
        action_status = freeze_result.get("action_status")
    elif applied_actions:
        action_status = "applied"
    else:
        action_status = None

    section: dict[str, Any] = {
        "target_account_id": (
            (freeze_result or {}).get("account_id")
            or (applied_actions[0].get("account_id") if applied_actions else None)
            or (simulation.get("account_id") if simulation else None)
        ),
        "simulation": None,
        "simulation_metrics": [],
        "eligibility": None,
        "policy_source": policy_source if policy else None,
        "action": None,
        "blocking_reasons": [],
        "declined": False,
        "notes": [],
    }
    if simulation:
        section["simulation"] = {
            "simulation_id": simulation.get("id"),
            "account_id": simulation.get("account_id"),
            "window_days": simulation.get("window_days"),
            "hops": simulation.get("hops"),
            "status": simulation.get("status"),
            "simulation_complete": bool(simulation.get("simulation_complete")),
            "created_at": simulation.get("created_at"),
            "graph_fingerprint": simulation.get("graph_fingerprint"),
        }
        section["simulation_metrics"] = [
            {
                "metric": key,
                "label": _METRIC_LABELS.get(key, key),
                "value": value,
            }
            for key, value in sorted(sim_metrics.items())
        ]
        section["notes"].append(
            "Simulation metrics count routing edges and historical transaction "
            "volumes in the synthetic graph; they are not monetary amounts moved "
            "or frozen by this action."
        )
    if policy:
        section["eligibility"] = {
            "allowed": bool(policy.get("allowed")),
            "verdict": policy.get("verdict"),
            "signal_families": list(policy.get("signal_families") or []),
            "target_relevant_families": list(
                policy.get("target_relevant_families") or []
            ),
            "distinct_provenance_families": list(
                policy.get("distinct_provenance_families") or []
            ),
            "qualifying_evidence_ids": list(policy.get("qualifying_evidence_ids") or []),
            "blocking_reasons": list(policy.get("blocking_reasons") or []),
            "caveats": list(policy.get("caveats") or []),
            "checks": policy.get("checks") or {},
            "policy_version": policy.get("policy_version"),
        }
        section["blocking_reasons"] = list(policy.get("blocking_reasons") or [])
    if action_status == "declined":
        section["declined"] = True
        section["notes"].append(
            "The freeze was declined by the deterministic eligibility policy; "
            "no graph mutation was performed for this case."
        )
    elif action_status in ("applied", "already_applied"):
        action = applied_actions[0] if applied_actions else None
        if action:
            section["action"] = {
                "action_id": action.get("id"),
                "action": action.get("action"),
                "account_id": action.get("account_id"),
                "status": action.get("status"),
                "reason": action.get("reason"),
                "requested_by": action.get("requested_by"),
                "simulation_id": action.get("simulation_id"),
                "evidence_ids": _json_field(action, "evidence_ids_json", []),
                "created_at": action.get("created_at"),
                "before_state": _json_field(action, "before_state_json", {}),
                "after_state": _json_field(action, "after_state_json", {}),
            }
        section["notes"].append(
            "The recorded action sets the account status to frozen and "
            "deactivates its routing relationships in the synthetic graph. No "
            "funds are moved or blocked by this system."
        )
    elif simulation is not None:
        section["notes"].append(
            "No freeze action was recorded for this case. The eligibility "
            "outcome shown above, if any, is taken from the recorded action "
            "audit trail, not re-derived."
        )
    return section


def _post_action_section(verification: CaseVerification) -> dict[str, Any]:
    notes: list[str] = []
    if verification.post_action:
        for entry in verification.post_action:
            if entry.get("verified"):
                notes.append(
                    f"Post-action checks for {entry.get('account_id')} pass: the "
                    "recorded action is present, the account is frozen, no active "
                    "incident routing edges remain, and the graph-wide active edge "
                    "count matches the pre-action simulation."
                )
            else:
                notes.append(
                    f"Post-action checks for {entry.get('account_id')} do not all "
                    "pass; the structured checks below show exactly which ones."
                )
    else:
        notes.append(
            "No applied action exists for this case, so there is no post-action "
            "mutation to verify. The current-state table above is an independent "
            "re-read of the graph."
        )
    notes.append(
        "The pre-action routing paths and the post-action graph state are kept "
        "distinct: a difference after an authorized freeze is the recorded action "
        "taking effect, never a sign that the original observation was false."
    )
    return {"post_action": verification.post_action, "current_state": verification.current_state, "notes": notes}


def _prior_cases_section(
    state, records: CaseRecords, id_map: Mapping[str, str]
) -> dict[str, Any]:
    leads: list[dict[str, Any]] = []
    prior_evidence_ids: list[str] = []
    if state is not None:
        for record in state.evidence:
            if record.tool == "prior_cases" and record.ok:
                record_eid = _resolve(record.evidence_id, id_map)
                prior_evidence_ids.append(record_eid)
                for case in record.derived.get("cases") or []:
                    leads.append({**case, "evidence_id": record_eid})
        if not leads and state.prior_case_leads:
            leads = [{"case_id": cid} for cid in state.prior_case_leads]
    for reuse in records.reuses:
        if not any(lead.get("case_id") == reuse.get("older_case_id") for lead in leads):
            leads.append({
                "case_id": reuse.get("older_case_id"),
                "status": reuse.get("status"),
                "verdict": reuse.get("verdict"),
                "updated_at": reuse.get("updated_at"),
                "evidence_id": None,
                "source": "persisted REUSES link",
            })
    notes = [
        "Prior cases are historical context only. Their verdicts are never "
        "inherited: this case re-derived its own verdict from the evidence "
        "collected during this investigation.",
        "There is no preserved ground truth for prior-case facts inside the "
        "agent-visible graph; facts marked as leads are re-checked rather than "
        "trusted.",
    ]
    revalidated: list[str] = []
    if state is not None:
        revalidated = sorted({
            record.tool for record in state.evidence
            if record.ok and record.tool in
            ("entity_profile", "detect_cycle", "fan_in_out", "trace_funds",
             "hub_scores", "shared_identifiers")
        })
    return {
        "leads": leads,
        "prior_cases_evidence_ids": prior_evidence_ids,
        "revalidated_tools": revalidated,
        "notes": notes,
    }


def _limitations(state, records: CaseRecords, verification: CaseVerification) -> list[str]:
    limitations: list[str] = []
    if state is not None:
        limitations.extend(state.limitations)
        if state.termination_reason == TERMINATION_BUDGET_EXHAUSTED:
            limitations.append(
                f"The investigation exhausted its tool budget "
                f"({state.tool_calls_used}/{state.tool_budget} calls); hypotheses "
                "that were not reached remain untested."
            )
    truncated = [
        r.get("evidence_id") for r in records.evidence if r.get("truncated")
    ]
    if truncated:
        limitations.append(
            f"Evidence records {sorted(truncated)} were truncated at capture; "
            "claims resting on them cannot be treated as complete."
        )
    hash_only = [
        r.get("evidence_id") for r in records.evidence
        if not _json_field(r, "rows_json", [])
    ]
    if hash_only:
        limitations.append(
            f"{len(hash_only)} citation(s) preserve only the result hash, not the "
            "rows. Their integrity is verified by exact replay where the query is "
            "replayable; otherwise the evidence is reported as unverifiable."
        )
    legacy_no_fp = [
        r.get("evidence_id") for r in records.evidence if not r.get("template_sha256")
    ]
    if legacy_no_fp:
        limitations.append(
            f"Citation(s) {sorted(legacy_no_fp)} carry no template fingerprint, "
            "so catalog drift could not be checked for them."
        )
    legacy_citations = [
        c.evidence_id for c in verification.claims if not c.compound
    ]
    if legacy_citations:
        limitations.append(
            f"Citation(s) {sorted(legacy_citations)} predate the Phase 6.1 "
            "compound source manifest: only the first cited query id was "
            "recorded, so sub-query coverage cannot be verified for them and "
            "they keep their pre-6.1 single-source semantics."
        )
    unavailable_sources = sorted({
        s.query_id for c in verification.claims for s in c.sources
        if s.status == STATUS_UNVERIFIABLE
    })
    if unavailable_sources:
        limitations.append(
            "Source queries that could not be independently checked: "
            f"{unavailable_sources}. Section 10 lists, per claim, which "
            "supporting queries were checked and which remain unavailable."
        )
    unqualified = [
        c for c in verification.claims
        if c.status not in (STATUS_REPRODUCED, STATUS_VERIFIED_RECORDED)
    ]
    if unqualified:
        limitations.append(
            "Not every evidence claim could be independently reproduced: "
            + ", ".join(f"{c.evidence_id}={c.status}" for c in unqualified)
            + ". See section 10 for the per-claim reasons."
        )
    limitations.append(
        "The report is reproducible from the same structured case record; the "
        "per-source manifests are integrity checks, not cryptographic "
        "immutability. If a cited record (or any recorded source of it) changes "
        "after generation, the next verification run detects the discrepancy."
    )
    return limitations


# ---- payload + markdown ---------------------------------------------------------


def _build_payload(
    case_id: str,
    records: CaseRecords,
    state,
    envelopes: Mapping[str, Any] | None,
    verification: CaseVerification,
    id_map: Mapping[str, str],
) -> dict[str, Any]:
    provenance = analysis.signal_provenance(state) if state is not None else None
    payload: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "case_id": case_id,
        "disclaimer": _DISCLAIMER,
        "case": {
            "case_id": case_id,
            "alert_id": records.case.get("alert_id"),
            "title": records.case.get("title"),
            "status": records.case.get("status"),
            "verdict": records.case.get("verdict"),
            "severity": records.case.get("severity"),
            "summary": _qualify_family_language(
                records.case.get("summary") or "", provenance
            ),
            "created_at": records.case.get("created_at"),
            "updated_at": records.case.get("updated_at"),
            "about_entities": list(records.about_entities),
        },
        "executive_findings": _executive_findings(state, records, provenance),
        "hypotheses": _hypothesis_section(state, verification, id_map),
        "provenance": provenance,
        "timeline": _timeline_section(state, records),
        "evidence_ledger": _evidence_ledger(records, state, verification, id_map),
        "transaction_paths": _transaction_paths(state, id_map),
        "intervention": _intervention_section(records, envelopes),
        "post_action": _post_action_section(verification),
        "prior_cases": _prior_cases_section(state, records, id_map),
        "verification": {
            "claims": [
                {
                    "evidence_id": c.evidence_id,
                    "claim": c.claim,
                    "query_id": c.query_id,
                    "source_tool": c.source_tool,
                    "status": c.status,
                    "reason": c.reason,
                    "replay": c.replay,
                    "compound": c.compound,
                    "source_count": c.source_count,
                    "required_source_queries": list(c.required_source_queries),
                    "checked_source_queries": list(c.checked_source_queries),
                    "unavailable_source_queries": list(c.unavailable_source_queries),
                    "mismatched_source_queries": list(c.mismatched_source_queries),
                    "sources": [
                        {
                            "query_id": s.query_id,
                            "role": s.role,
                            "required": s.required,
                            "status": s.status,
                            "reason": s.reason,
                            "replay": s.replay,
                        }
                        for s in c.sources
                    ],
                }
                for c in sorted(verification.claims, key=lambda c: c.evidence_id)
            ],
            "summary": dict(verification.summary),
            "references": [
                {
                    "claim": r.claim,
                    "source": r.source,
                    "evidence_ids": r.evidence_ids,
                    "missing_evidence_ids": r.missing_evidence_ids,
                    "per_evidence": r.per_evidence,
                    "checked_source_queries": list(r.checked_source_queries),
                    "unavailable_source_queries": list(r.unavailable_source_queries),
                    "mismatched_source_queries": list(r.mismatched_source_queries),
                    "status": r.status,
                    "reason": r.reason,
                }
                for r in verification.references
            ],
            "reference_summary": dict(verification.reference_summary),
            "notes": list(verification.notes),
        },
    }
    disposition = "no_recorded_action"
    intervention = payload["intervention"]
    if intervention.get("action"):
        disposition = "freeze_applied"
    elif intervention.get("declined"):
        disposition = "freeze_declined"
    payload["disposition"] = disposition
    payload["limitations"] = _limitations(state, records, verification)
    return payload


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _render_markdown(payload: dict[str, Any], generated_at: str) -> str:
    case = payload["case"]
    lines: list[str] = []
    add = lines.append

    add(f"# HABEAS investigation report — {payload['case_id']}")
    add("")
    add(f"Report version: {payload['report_version']}  ")
    add(f"Generated: {generated_at}  ")
    add(f"Content fingerprint: {_fingerprint(payload)}")
    add("")
    add(f"> {payload['disclaimer']}")
    add("")

    # 1. case summary
    add("## 1. Case summary")
    add("")
    add(f"- Case: {case.get('case_id') or payload['case_id']}")
    add(f"- Alert: {_md_cell(case.get('alert_id'))}")
    add(f"- Title: {_md_cell(case.get('title'))}")
    add(f"- Status: {_md_cell(case.get('status'))} | Canonical verdict: "
        f"{_md_cell(case.get('verdict'))} | Severity: {_md_cell(case.get('severity'))}")
    add(f"- Investigated entities: {', '.join(case.get('about_entities') or []) or 'none recorded'}")
    add(f"- Investigation window: created {_md_cell(case.get('created_at'))}, "
        f"last updated {_md_cell(case.get('updated_at'))}")
    add(f"- Disposition: {payload['disposition']}")
    if case.get("summary"):
        add(f"- Case summary: {_md_cell(case.get('summary'))}")
    add("")

    # 2. executive findings
    findings = payload["executive_findings"]
    add("## 2. Executive findings")
    add("")
    leading = findings.get("leading_hypothesis")
    if leading:
        add(f"Leading hypothesis: **{leading['title']}** "
            f"(`{leading['name']}`, status: {leading['status']}).")
        if leading.get("supporting_evidence_ids"):
            add("")
            add("Supported by evidence: "
                + ", ".join(f"`{eid}`" for eid in leading["supporting_evidence_ids"]))
        if leading.get("rationale"):
            add("")
            add(leading["rationale"])
    else:
        add("No hypothesis reached a supported status; the evidence below is "
            "insufficient to single out a leading explanation.")
    add("")
    add(f"Canonical verdict: **{_md_cell(findings.get('verdict'))}** "
        "(source: case audit history).")
    if findings.get("verdict_rationale"):
        add("")
        add(findings["verdict_rationale"])
    add("")
    add(findings["qualification"])
    for caveat in findings.get("provenance_caveats") or []:
        add("")
        add(f"Provenance caveat: {caveat}")
    add("")

    # 3. competing hypotheses
    add("## 3. Competing hypotheses")
    add("")
    if not payload["hypotheses"]:
        add("The investigation state was not supplied and hypotheses are not "
            "reconstructible from the persisted evidence alone; see the evidence "
            "ledger (section 5) for what was captured.")
    for hypothesis in payload["hypotheses"]:
        add(f"### {hypothesis['title']} (`{hypothesis['name']}`)")
        add("")
        add(f"Status: **{hypothesis['status']}**. {hypothesis.get('note') or ''}".strip())
        if hypothesis.get("rationale"):
            add("")
            add(hypothesis["rationale"])
        add("")
        add("| Stance | Evidence | Verification |")
        add("|---|---|---|")
        for entry in hypothesis.get("supporting") or []:
            add(f"| supporting | `{entry['evidence_id']}` | {entry['verification_status']} |")
        for entry in hypothesis.get("contradicting") or []:
            add(f"| contradicting | `{entry['evidence_id']}` | {entry['verification_status']} |")
        if not (hypothesis.get("supporting") or hypothesis.get("contradicting")):
            add("| — | no evidence references recorded | — |")
        if hypothesis.get("uncertainties"):
            add("")
            add("Unresolved questions / uncertainties:")
            for item in hypothesis["uncertainties"]:
                add(f"- {_md_cell(item)}")
        add("")
    provenance = payload.get("provenance")
    if provenance:
        add("### Signal provenance (evidence independence)")
        add("")
        if not provenance.get("families_detected"):
            add("No signal family was detected in this investigation.")
        for pair in provenance.get("pairs") or []:
            a, b = pair["pair"]
            sentence = _OVERLAP_SENTENCES[pair["classification"]]
            add(f"- `{a}` vs `{b}`: {sentence}"
                + (f" ({pair['shared_count']} shared transaction id(s))"
                   if pair["classification"] == "shared_observations" else "")
                + ("." if not sentence.endswith(".") else ""))
        families = provenance.get("families") or {}
        for name, source in sorted(families.items()):
            accounts = ", ".join(source.get("accounts") or []) or "—"
            add(f"- `{name}`: accounts {accounts}; basis: {source.get('basis')}"
                + (f"; note: {source.get('note')}" if source.get("note") else ""))
        add("")

    # 4. investigation timeline
    timeline = payload["timeline"]
    add("## 4. Investigation timeline")
    add("")
    add(f"Source: {timeline['source']}.")
    add("")
    add("| # | Stage | Tool | Why selected | Result (summary) | Assessment update | Evidence |")
    add("|---|---|---|---|---|---|---|")
    for entry in timeline["entries"]:
        add("| " + " | ".join([
            _md_cell(entry.get("index") if entry.get("index") is not None else "—"),
            _md_cell(entry.get("stage")),
            _md_cell(entry.get("tool") or "—"),
            _md_cell(_truncate(entry.get("purpose") or "", 120) or "—"),
            _md_cell(_truncate(entry.get("result_summary") or "", 160) or "—"),
            _md_cell(_truncate(entry.get("assessment_update") or "", 120) or "—"),
            f"`{entry['evidence_id']}`" if entry.get("evidence_id") else "—",
        ]) + " |")
    add("")

    # 5. evidence ledger
    add("## 5. Evidence ledger")
    add("")
    add("Each citation lists every source query recorded for its tool "
        "invocation; the per-source verification outcomes are in section 10.")
    add("")
    add("| Evidence | Claim | Source queries | Entities (params) | Hash | Rows | Trunc | Kind | Captured | Verification |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in payload["evidence_ledger"]:
        params = _truncate(json.dumps(row.get("params") or {}, sort_keys=True), 80)
        query_ids = [s["query_id"] for s in row.get("sources") or []]
        query_cell = ", ".join(f"`{q}`" for q in query_ids) or (
            f"`{row['query_id']}`" if row.get("query_id") else "—")
        add("| " + " | ".join([
            f"`{row['evidence_id']}`",
            _md_cell(_truncate(row.get("claim") or "", 140)),
            _md_cell(query_cell),
            _md_cell(params),
            _md_cell(_truncate(str(row.get("result_hash") or ""), 24)) + "…",
            _md_cell(row.get("row_count")),
            "yes" if row.get("truncated") else "no",
            _md_cell(row.get("evidence_kind")),
            _md_cell(row.get("captured_at")),
            _md_cell(row.get("verification_status")),
        ]) + " |")
    add("")
    add("Per-evidence notes:")
    for row in payload["evidence_ledger"]:
        add(f"- `{row['evidence_id']}`: {_md_cell(row.get('interpretation') or '')} "
            f"({row.get('source_coverage_note')})")
    add("")

    # 6. transaction paths
    paths = payload["transaction_paths"]
    add("## 6. Transaction paths")
    add("")
    for note in paths["notes"]:
        add(f"- {note}")
    add("")
    if paths["corroborated_cycles"]:
        add("### Temporally corroborated circulation")
        add("")
        for cycle in paths["corroborated_cycles"]:
            add(f"- Ring {' → '.join(cycle['members'])} → {cycle['members'][0]} "
                f"(evidence `{cycle['evidence_id']}`, "
                f"{cycle['event_count']} timestamped event(s)):")
            for event in cycle["events"]:
                add(f"  - {event['ts']}: {event['src']} → {event['dst']} "
                    f"amount {event['amount']} (txn `{event['txn_id']}`)")
            if cycle["event_count"] and cycle["event_count"] > len(cycle["events"]):
                add(f"  - … {cycle['event_count'] - len(cycle['events'])} more event(s) "
                    "recorded in the evidence record")
        add("")
    if paths["structural_cycles"]:
        add("### Structural cycles (aggregate topology only)")
        add("")
        for cycle in paths["structural_cycles"]:
            add(f"- Ring {' → '.join(cycle['members'])} → {cycle['members'][0]} "
                f"(evidence `{cycle['evidence_id']}`): the aggregate topology is "
                "closed, but timestamps do not chain around it; this is structural "
                "connectivity, not observed circulation.")
        add("")
    if paths["rapid_matches"]:
        add("### Rapid in/out movement of incoming funds")
        add("")
        add("| Account | In (ts / txn) | Out (ts / txn) | Gap (min) | In → Out amount | Evidence |")
        add("|---|---|---|---|---|---|")
        for match in paths["rapid_matches"]:
            add("| " + " | ".join([
                _md_cell(match.get("account_id")),
                f"{_md_cell(match.get('in_ts'))} / `{_md_cell(match.get('in_txn_id'))}`",
                f"{_md_cell(match.get('out_ts'))} / `{_md_cell(match.get('out_txn_id'))}`",
                _md_cell(match.get("gap_minutes")),
                f"{_md_cell(match.get('in_amount'))} → {_md_cell(match.get('out_amount'))}",
                f"`{match['evidence_id']}`",
            ]) + " |")
        add("")
    if not (paths["corroborated_cycles"] or paths["structural_cycles"]
            or paths["rapid_matches"]):
        add("No path-level evidence was recorded for this case.")
        add("")

    # 7. intervention assessment
    intervention = payload["intervention"]
    add("## 7. Intervention assessment")
    add("")
    if intervention.get("target_account_id"):
        add(f"Account assessed for intervention: `{intervention['target_account_id']}`.")
        add("")
    simulation = intervention.get("simulation")
    if simulation:
        add(f"Pre-action simulation `{simulation['simulation_id']}` "
            f"(window {simulation['window_days']} days, {simulation['hops']} hops, "
            f"status {simulation['status']}, complete: "
            f"{'yes' if simulation['simulation_complete'] else 'no'}).")
        add("")
        add("| Metric | Meaning | Value |")
        add("|---|---|---|")
        for metric in intervention["simulation_metrics"]:
            add(f"| `{metric['metric']}` | {_md_cell(metric['label'])} | "
                f"{_md_cell(_metric_value(metric['value']))} |")
        add("")
    eligibility = intervention.get("eligibility")
    if eligibility:
        add(f"Eligibility outcome (source: {intervention.get('policy_source')}): "
            f"allowed = {eligibility['allowed']}.")
        if eligibility.get("signal_families"):
            add("")
            add("Signal families recorded by the policy: "
                + ", ".join(f"`{f}`" for f in eligibility["signal_families"]))
        if eligibility.get("distinct_provenance_families"):
            add("")
            add("Distinct-provenance families counted for eligibility: "
                + ", ".join(
                    f"`{f}`" for f in eligibility["distinct_provenance_families"]))
        if eligibility.get("policy_version") is not None:
            version = eligibility["policy_version"]
            add("")
            add(f"Recorded policy version: {version} "
                f"(current policy: {POLICY_VERSION}).")
            if version != POLICY_VERSION:
                add("")
                add(
                    "This authorization was recorded under an earlier policy "
                    f"(version {version}) and is preserved exactly as recorded. "
                    f"The current policy (version {POLICY_VERSION}) is not applied "
                    "retroactively, and this historical authorization is never "
                    "re-presented as a current-policy outcome."
                )
        if eligibility.get("qualifying_evidence_ids"):
            add("")
            add("Qualifying evidence: "
                + ", ".join(f"`{e}`" for e in eligibility["qualifying_evidence_ids"]))
        if eligibility.get("blocking_reasons"):
            add("")
            add("Blocking reasons:")
            for reason in eligibility["blocking_reasons"]:
                add(f"- {_md_cell(reason)}")
        if eligibility.get("caveats"):
            add("")
            add("Policy caveats:")
            for caveat in eligibility["caveats"]:
                add(f"- {_md_cell(caveat)}")
        add("")
    action = intervention.get("action")
    if action:
        add(f"Recorded action `{action['action_id']}` ({action['action']} on "
            f"`{action['account_id']}`, status {action['status']}).")
        add("")
        add(f"- Reason: {_md_cell(action.get('reason'))}")
        add(f"- Requested by: {_md_cell(action.get('requested_by'))}; "
            f"created: {_md_cell(action.get('created_at'))}")
        add(f"- Simulation: `{action.get('simulation_id')}`; evidence ids: "
            + (", ".join(f"`{e}`" for e in action.get("evidence_ids") or []) or "—"))
        add(f"- Before: {_md_cell(json.dumps(action.get('before_state') or {}, sort_keys=True))}")
        add(f"- After: {_md_cell(json.dumps(action.get('after_state') or {}, sort_keys=True))}")
        add("")
    elif intervention.get("declined"):
        add("No action was applied: the eligibility policy declined the freeze "
            "with the blocking reasons above, and the account's state was left "
            "unchanged at that time.")
        add("")
    for note in intervention.get("notes") or []:
        add(f"- {note}")
    add("")

    # 8. post-action verification
    post = payload["post_action"]
    add("## 8. Post-action verification")
    add("")
    add("Current graph state (independently re-read):")
    add("")
    add("| Entity | Kind | Status | Check |")
    add("|---|---|---|---|")
    if post["current_state"]:
        for entry in post["current_state"]:
            add(f"| `{entry.get('entity_id')}` | {_md_cell(entry.get('kind'))} | "
                f"{_md_cell(entry.get('status'))} | {_md_cell(entry.get('check'))} |")
    else:
        add("| — | — | — | no entities recorded |")
    add("")
    for entry in post["post_action"]:
        add(f"Recorded action `{entry.get('action_id')}` on `{entry.get('account_id')}`: "
            f"verified = {entry.get('verified')}.")
        add("")
        add("| Check | Result |")
        add("|---|---|")
        for name, ok in sorted((entry.get("checks") or {}).items()):
            add(f"| `{name}` | {ok} |")
        add("")
    for note in post["notes"]:
        add(f"- {note}")
    add("")

    # 9. prior-case context
    prior = payload["prior_cases"]
    add("## 9. Prior-case context")
    add("")
    if prior["leads"]:
        add("| Prior case | Verdict | Status | Source |")
        add("|---|---|---|---|")
        for lead in prior["leads"]:
            source = (f"evidence `{lead['evidence_id']}`"
                      if lead.get("evidence_id") else lead.get("source") or "audit history")
            add(f"| `{lead.get('case_id')}` | {_md_cell(lead.get('verdict'))} | "
                f"{_md_cell(lead.get('status'))} | {_md_cell(source)} |")
    else:
        add("No prior-case leads were recorded or found at investigation time.")
    add("")
    if prior["revalidated_tools"]:
        add("Independently re-collected during this investigation (never inherited "
            "from prior cases): " + ", ".join(f"`{t}`" for t in prior["revalidated_tools"]))
        add("")
    for note in prior["notes"]:
        add(f"- {note}")
    add("")

    # 10. independent verifier results
    verification = payload["verification"]
    add("## 10. Independent verifier results")
    add("")
    add("Deterministic, LLM-free verification of the persisted evidence records "
        "(query-catalog resolution, parameter contracts, full-row hash recompute, "
        "and exact read-only replay with the original window where possible). "
        "Every recorded source query of a citation is verified individually; a "
        "record is only marked `reproduced_current_state` when ALL of its "
        "required sources replay-match.")
    add("")
    add("| Evidence | Primary query | Sources | Status | Reason |")
    add("|---|---|---|---|---|")
    for claim in verification["claims"]:
        primary = f"`{claim['query_id']}`" if claim.get("query_id") else "—"
        source_count = claim.get("source_count") or len(
            claim.get("sources") or []
        )
        add(f"| `{claim['evidence_id']}` | {primary} | {source_count} | "
            f"{claim['status']} | {_md_cell(_truncate(claim['reason'], 260))} |")
    add("")
    add("Summary: " + ", ".join(
        f"{status}={verification['summary'].get(status, 0)}"
        for status in ALL_STATUSES
    ))
    add("")
    source_rows = [
        (claim, source)
        for claim in verification["claims"]
        for source in claim.get("sources") or []
    ]
    if source_rows:
        add("Source-query verification (each recorded supporting query; "
            "`not replayed` means no snapshot and no replay could check it):")
        add("")
        add("| Evidence | Source query | Role | Required | Replay | Status | Reason |")
        add("|---|---|---|---|---|---|---|")
        for claim, source in source_rows:
            replay = source.get("replay") or {}
            replay_cell = (
                "replayed" if replay.get("replayed_hash") is not None
                else "not replayed"
            )
            add("| " + " | ".join([
                f"`{claim['evidence_id']}`",
                f"`{source['query_id']}`",
                _md_cell(source.get("role")),
                "yes" if source.get("required") else "no",
                replay_cell,
                _md_cell(source.get("status")),
                _md_cell(_truncate(source.get("reason") or "", 220)),
            ]) + " |")
        add("")
    if verification["references"]:
        add("Material claims vs evidence:")
        add("")
        add("| Claim | Source | Referenced evidence | Status |")
        add("|---|---|---|---|")
        for ref in verification["references"]:
            refs = ", ".join(f"`{e}`" for e in ref["evidence_ids"]) or "—"
            add(f"| {_md_cell(_truncate(ref['claim'], 120))} | {_md_cell(ref['source'])} | "
                f"{refs} | {ref['status']} |")
        add("")
        add("Per-claim source-query coverage (which supporting queries were "
            "independently checked and which remain unavailable):")
        add("")
        for ref in verification["references"]:
            checked = ref.get("checked_source_queries") or []
            unavailable = ref.get("unavailable_source_queries") or []
            mismatched = ref.get("mismatched_source_queries") or []
            parts: list[str] = []
            if checked:
                parts.append(
                    "checked: " + ", ".join(f"`{q}`" for q in checked)
                )
            if mismatched:
                parts.append(
                    "inconsistent: " + ", ".join(f"`{q}`" for q in mismatched)
                )
            if unavailable:
                parts.append(
                    "unavailable: " + ", ".join(f"`{q}`" for q in unavailable)
                )
            if not parts:
                parts.append("no source queries recorded")
            add(f"- {_md_cell(_truncate(ref['claim'], 160))}: "
                + "; ".join(parts) + ".")
        add("")
        add("Claim summary: " + ", ".join(
            f"{status}={verification['reference_summary'].get(status, 0)}"
            for status in ALL_STATUSES
        ))
        add("")
    for note in verification["notes"]:
        add(f"- {_md_cell(note)}")
    add("")

    add("## Limitations")
    add("")
    for limitation in payload["limitations"]:
        add(f"- {_md_cell(limitation)}")
    add("")
    add(f"_Report fingerprint: {_fingerprint(payload)} (content excluding the "
        "generation timestamp). Regenerating this report from the same case "
        "record reproduces it._")
    add("")
    return "\n".join(lines)


# ---- public API ------------------------------------------------------------------


def build_report(
    graph,
    case_id: str,
    *,
    state=None,
    envelopes: Mapping[str, Any] | None = None,
    settings=None,
    generated_at: str | None = None,
) -> ReportArtifact:
    """Build the Markdown report and structured payload for one persisted case.

    ``state`` (a :class:`~src.agent.state.CaseState`) enriches the hypotheses,
    timeline, paths and provenance sections; when omitted, the builder
    reconstructs what the persisted evidence supports and says so. ``envelopes``
    may carry the workflow's recorded tool envelopes so the intervention
    section can quote the ORIGINAL eligibility decision. The builder never
    mutates the graph.
    """
    records = load_case_records(graph, case_id)
    if records.case is None:
        raise ValueError(f"case {case_id} does not exist; nothing to report")

    if state is None:
        try:
            from src.tools.actions import reconstruct_case_state

            state = reconstruct_case_state(graph, case_id)
        except Exception:
            state = None

    id_map = _evidence_id_map(case_id, state, records)
    claim_refs = _claim_references(state, id_map)
    verification = verify_case(
        graph, case_id, claim_refs=claim_refs, records=records
    )
    payload = _build_payload(case_id, records, state, envelopes, verification, id_map)
    timestamp = generated_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    markdown = _render_markdown(payload, timestamp)
    return ReportArtifact(
        case_id=case_id,
        generated_at=timestamp,
        report_version=REPORT_VERSION,
        markdown=markdown,
        payload=payload,
        fingerprint=_fingerprint(payload),
        verification=verification,
    )


def write_report(artifact: ReportArtifact, reports_dir: str | Path = "reports") -> Path:
    """Write ``reports/<case_id>.md`` and return the path."""
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{artifact.case_id}.md"
    path.write_text(artifact.markdown, encoding="utf-8")
    return path
