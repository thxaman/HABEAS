"""Deterministic evidence analysis for the investigation agent (PLAN.md §9).

Everything in this module is ordinary Python over tool envelopes. The LLM
chooses which tool to call; it never decides what the evidence *means*. That
separation is what stops a plausible-sounding model from manufacturing a
finding, and it makes the agent's conclusions reproducible in tests.

Rapid-movement detection is derived from the timestamped leg events that
`trace_funds` already returns, so no extra tool or LLM judgement is needed.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from src.agent.state import (
    FAMILY_CIRCULATION,
    FAMILY_FAN_IN_OUT,
    FAMILY_RAPID_FORWARDING,
    VERDICT_INSUFFICIENT,
    VERDICT_LIKELY_BENIGN,
    VERDICT_SUSPICIOUS,
)

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Documented, tested thresholds. These are prioritisation thresholds for
# review, not statistical cut-offs and not a finding of wrongdoing.
RAPID_MAX_GAP_MINUTES = 90
RAPID_MIN_FORWARD_RATIO = 0.8
FAN_IN_MIN_SENDERS = 3
FAN_IN_MIN_OUT_IN_RATIO = 0.7
FAN_IN_MIN_TOP_SHARE = 0.7


# ---- per-tool derivations --------------------------------------------------


def derive_alert_context(envelope: dict[str, Any]) -> dict[str, Any]:
    result = envelope.get("result") or {}
    return {
        "found": bool(result.get("found")),
        "entity_ids": list(result.get("entity_ids") or []),
        "sender": result.get("sender_account_id"),
        "receiver": result.get("receiver_account_id"),
        "concerned": result.get("concerned_account_id"),
        "reason": result.get("reason"),
        "trigger_txn": result.get("trigger_transaction") or {},
    }


def derive_profile(envelope: dict[str, Any]) -> dict[str, Any]:
    result = envelope.get("result") or {}
    return {
        "found": bool(result.get("found")),
        "kind": result.get("kind"),
        "status": (result.get("properties") or {}).get("status"),
        "incoming": result.get("incoming") or {},
        "outgoing": result.get("outgoing") or {},
        "owner_customers": list(result.get("owner_customers") or []),
        "prior_case_count": result.get("prior_case_count"),
    }


def derive_cycle(envelope: dict[str, Any]) -> dict[str, Any]:
    result = envelope.get("result") or {}
    cycles = result.get("cycles") or []
    return {
        "cycle_count": int(result.get("cycle_count") or 0),
        "corroborated_count": int(result.get("temporally_corroborated_count") or 0),
        "structural_only_count": sum(
            1 for c in cycles if not c.get("temporally_corroborated")
        ),
        "cycles": [
            {
                "members": list(c.get("members") or []),
                "length": c.get("length"),
                "temporally_corroborated": bool(c.get("temporally_corroborated")),
                "event_count": c.get("event_count"),
                # Phase 6 provenance/report: retain the directed legs and the
                # timestamped events (with transaction ids) that the tool
                # already returned, so signal support can be traced back to
                # individual transactions without re-querying.
                "legs": [dict(leg) for leg in (c.get("legs") or [])],
                "events": [dict(evt) for evt in (c.get("events") or [])],
            }
            for c in cycles
        ],
    }


def derive_fan_in_out(envelope: dict[str, Any]) -> dict[str, Any]:
    result = envelope.get("result") or {}
    return {
        # Retained so the family's target relevance can be checked without
        # re-deriving it from parameters (policy v2).
        "account_id": result.get("account_id"),
        "distinct_senders": int(result.get("distinct_senders") or 0),
        "distinct_receivers": int(result.get("distinct_receivers") or 0),
        "in_amount": float(result.get("in_amount") or 0.0),
        "out_amount": float(result.get("out_amount") or 0.0),
        "out_in_ratio": result.get("out_in_ratio"),
        "top_receiver": result.get("top_receiver"),
        "top_receiver_share": result.get("top_receiver_share"),
    }


def derive_shared_identifiers(envelope: dict[str, Any]) -> dict[str, Any]:
    result = envelope.get("result") or {}
    return {
        "shared_devices": list(result.get("shared_devices") or []),
        "shared_identifiers": list(result.get("shared_identifiers") or []),
        "kinds": list(result.get("shared_identifier_kinds") or []),
    }


def derive_trace(envelope: dict[str, Any], account_id: str) -> dict[str, Any]:
    result = envelope.get("result") or {}
    leg_events = list(result.get("leg_events") or [])
    rapid = derive_rapid_from_events(leg_events, account_id)
    return {
        "account_id": account_id,
        "direction": result.get("direction"),
        "path_count": int(result.get("path_count") or 0),
        "observed_leg_events": len(leg_events),
        "truncated": bool(result.get("truncated")),
        # retained (bounded) so inbound and outbound legs from separate trace
        # calls can be combined before judging in/out timing
        "leg_events": leg_events[:200],
        "rapid_forward_count": rapid["rapid_forward_count"],
        "rapid_matches": rapid["matches"],
    }


def derive_hub_scores(envelope: dict[str, Any]) -> dict[str, Any]:
    result = envelope.get("result") or {}
    return {
        "method": result.get("method"),
        "hubs": [
            {"account_id": h.get("account_id"),
             "weighted_in_amount": h.get("weighted_in_amount")}
            for h in (result.get("hubs") or [])
        ],
    }


def derive_prior_cases(envelope: dict[str, Any]) -> dict[str, Any]:
    result = envelope.get("result") or {}
    cases = [
        {
            "case_id": case.get("case_id"),
            "status": case.get("status"),
            "verdict": case.get("verdict"),
            "severity": case.get("severity"),
            "updated_at": case.get("updated_at"),
        }
        for case in (result.get("cases") or [])
    ]
    return {"case_count": int(result.get("case_count") or 0), "cases": cases}


def derive_rapid_from_events(
    events: list[dict[str, Any]], account_id: str
) -> dict[str, Any]:
    """Onward movement of incoming funds, from timestamps the tools returned."""
    incoming = sorted(
        (e for e in events if e.get("dst") == account_id), key=lambda e: e["ts"]
    )
    outgoing = sorted(
        (e for e in events if e.get("src") == account_id), key=lambda e: e["ts"]
    )
    matches: list[dict[str, Any]] = []
    for in_evt in incoming:
        in_dt = datetime.strptime(in_evt["ts"], _TS_FMT)
        for out_evt in outgoing:
            gap = (
                datetime.strptime(out_evt["ts"], _TS_FMT) - in_dt
            ).total_seconds() / 60
            if gap < 0:
                continue
            if gap > RAPID_MAX_GAP_MINUTES:
                break
            in_amount = float(in_evt.get("amount") or 0)
            out_amount = float(out_evt.get("amount") or 0)
            if in_amount > 0 and out_amount >= RAPID_MIN_FORWARD_RATIO * in_amount:
                matches.append({
                    "in_ts": in_evt["ts"],
                    "out_ts": out_evt["ts"],
                    "gap_minutes": round(gap, 1),
                    "in_amount": in_amount,
                    "out_amount": out_amount,
                    # Phase 6 provenance: the exact transactions behind a match,
                    # so shared underlying observations across families can be
                    # detected instead of inferred from signal names.
                    "in_txn_id": in_evt.get("txn_id"),
                    "out_txn_id": out_evt.get("txn_id"),
                    "in_src": in_evt.get("src"),
                    "out_dst": out_evt.get("dst"),
                })
                break
    return {
        "account_id": account_id,
        "rapid_forward_count": len(matches),
        "matches": matches[:10],
    }


def derive_rapid_movement(envelope: dict[str, Any], account_id: str) -> dict[str, Any]:
    events = (envelope.get("result") or {}).get("leg_events") or []
    return derive_rapid_from_events(events, account_id)


def rapid_forwarding_for_account(state, account_id: str) -> dict[str, Any]:
    """Combine every successful trace's leg events before judging timing.

    An outbound trace alone never contains the inbound leg, and vice versa, so
    in/out timing must be evaluated over the union of what the tools returned.
    """
    events: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    for record in state.evidence:
        if record.tool == "trace_funds" and record.ok:
            legs = record.derived.get("leg_events") or []
            if legs:
                events.extend(legs)
                evidence_ids.append(record.evidence_id)
    result = derive_rapid_from_events(events, account_id)
    result["evidence_ids"] = evidence_ids
    return result


DERIVERS = {
    "alert_context": lambda env, _: derive_alert_context(env),
    "entity_profile": lambda env, _: derive_profile(env),
    "detect_cycle": lambda env, _: derive_cycle(env),
    "fan_in_out": lambda env, _: derive_fan_in_out(env),
    "shared_identifiers": lambda env, _: derive_shared_identifiers(env),
    "trace_funds": lambda env, acct: derive_trace(env, acct),
    "hub_scores": lambda env, _: derive_hub_scores(env),
    "prior_cases": lambda env, _: derive_prior_cases(env),
}


# ---- hypothesis / verdict reasoning ----------------------------------------


def fan_in_is_concentrated(fan: dict[str, Any]) -> bool:
    """Coordinated cash-out signature: many senders, mostly forwarded onward."""
    share = fan.get("top_receiver_share")
    ratio = fan.get("out_in_ratio")
    return (
        fan["distinct_senders"] >= FAN_IN_MIN_SENDERS
        and fan["out_amount"] > 0
        and ratio is not None and ratio >= FAN_IN_MIN_OUT_IN_RATIO
        and share is not None and share >= FAN_IN_MIN_TOP_SHARE
    )


def detect_signal_families(state) -> dict[str, dict[str, Any]]:
    """Map each *independent* signal family to the evidence that supports it.

    A family is only claimed when a real tool result supports it; absence of
    evidence never counts as evidence of absence.
    """
    families: dict[str, dict[str, Any]] = {}

    corroborated: list[dict[str, Any]] = []
    for record in state.evidence:
        if record.tool == "detect_cycle" and record.ok:
            for cycle in record.derived.get("cycles", []):
                if cycle.get("temporally_corroborated"):
                    corroborated.append({**cycle, "evidence_id": record.evidence_id})
    if corroborated:
        families[FAMILY_CIRCULATION] = {"cycles": corroborated}

    concentrated: list[dict[str, Any]] = []
    for record in state.evidence:
        if (
            record.tool == "fan_in_out"
            and record.ok
            and fan_in_is_concentrated(record.derived)
        ):
            concentrated.append({**record.derived, "evidence_id": record.evidence_id})
    if concentrated:
        families[FAMILY_FAN_IN_OUT] = {"accounts": concentrated}

    rapid: list[dict[str, Any]] = []
    for account_id in state.primary_entity_ids:
        agg = rapid_forwarding_for_account(state, account_id)
        if agg["rapid_forward_count"] > 0:
            rapid.append({**agg, "evidence_id": (agg["evidence_ids"] or [""])[-1]})
    if rapid:
        families[FAMILY_RAPID_FORWARDING] = {"accounts": rapid}

    return families


# ---- Phase 6: signal provenance (evidence independence) --------------------
#
# Different signal NAMES do not establish independent evidential support. A
# corroborated cycle and a rapid-forwarding match on the same ring can both
# derive from the very same transaction events; counting them as two
# independent families would overstate the evidence. These helpers trace each
# family to the underlying transaction ids (where the tools return them) so
# the report can state what is actually shared and classify each pair.

_OVERLAP_DISTINCT = "distinct_observations"
_OVERLAP_PARTIAL = "partially_shared"
_OVERLAP_SHARED = "shared_observations"
_OVERLAP_UNAVAILABLE = "provenance_unavailable"

# Public aliases: the action-eligibility policy (v2) consumes these values, and
# they are part of the persisted provenance audit.
OVERLAP_DISTINCT = _OVERLAP_DISTINCT
OVERLAP_PARTIAL = _OVERLAP_PARTIAL
OVERLAP_SHARED = _OVERLAP_SHARED
OVERLAP_UNAVAILABLE = _OVERLAP_UNAVAILABLE


def circulation_observations(state) -> dict[str, Any]:
    """Transaction ids supporting the circulation family (corroborated cycles)."""
    txn_ids: set[str] = set()
    accounts: set[str] = set()
    evidence_ids: list[str] = []
    for record in state.evidence:
        if record.tool != "detect_cycle" or not record.ok:
            continue
        found = False
        for cycle in record.derived.get("cycles", []):
            if not cycle.get("temporally_corroborated"):
                continue
            for evt in cycle.get("events") or []:
                txn_id = evt.get("txn_id")
                if txn_id:
                    txn_ids.add(txn_id)
                    found = True
                for account in (evt.get("src"), evt.get("dst")):
                    if account:
                        accounts.add(account)
        if found and record.evidence_id not in evidence_ids:
            evidence_ids.append(record.evidence_id)
    return {
        "available": bool(txn_ids),
        "txn_ids": sorted(txn_ids),
        "accounts": sorted(accounts),
        "evidence_ids": evidence_ids,
        "basis": "txn_events_for_legs_v1 over corroborated ring legs",
    }


def rapid_observations(state) -> dict[str, Any]:
    """Transaction ids supporting the rapid-forwarding family, per account."""
    by_account: dict[str, list[str]] = {}
    evidence_ids: list[str] = []
    for account_id in state.primary_entity_ids:
        agg = rapid_forwarding_for_account(state, account_id)
        ids: set[str] = set()
        for match in agg.get("matches") or []:
            for key in ("in_txn_id", "out_txn_id"):
                if match.get(key):
                    ids.add(match[key])
        if ids:
            by_account[account_id] = sorted(ids)
        for eid in agg.get("evidence_ids") or []:
            if eid and eid not in evidence_ids:
                evidence_ids.append(eid)
    all_ids = sorted({txn for ids in by_account.values() for txn in ids})
    return {
        "available": bool(all_ids),
        "txn_ids": all_ids,
        "by_account": by_account,
        "accounts": sorted(by_account),
        "evidence_ids": evidence_ids,
        "basis": "trace_funds leg events (in->out pairs within 90 min at >=80% of amount)",
    }


def classify_observation_overlap(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Classify two families' underlying observation sets.

    - ``distinct_observations``: no shared transaction ids;
    - ``partially_shared``: some transaction ids shared, some unique to each;
    - ``shared_observations``: one set is fully contained in the other;
    - ``provenance_unavailable``: at least one family's observations cannot be
      resolved to individual transactions (e.g. aggregate-only queries), so
      independence cannot be established and must not be claimed.
    """
    if not a.get("available") or not b.get("available"):
        return _OVERLAP_UNAVAILABLE
    set_a = set(a.get("txn_ids") or [])
    set_b = set(b.get("txn_ids") or [])
    if not (set_a & set_b):
        return _OVERLAP_DISTINCT
    if set_a <= set_b or set_b <= set_a:
        return _OVERLAP_SHARED
    return _OVERLAP_PARTIAL


def fan_in_out_observations(state) -> dict[str, Any]:
    """Account attribution for the fan-in/out family (aggregate rollups).

    The underlying queries return per-counterparty rollups, not individual
    transaction ids, so observation-level provenance stays unavailable while
    the accounts the family concerns remain explicit — which is what the
    target-relevance check needs.
    """
    families = detect_signal_families(state)
    accounts: set[str] = set()
    evidence_ids: list[str] = []
    for entry in families.get(FAMILY_FAN_IN_OUT, {}).get("accounts") or []:
        account = entry.get("account_id")
        if account:
            accounts.add(str(account))
        evidence_id = entry.get("evidence_id")
        if evidence_id and evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
    return {
        "available": False,
        "txn_ids": None,
        "accounts": sorted(accounts),
        "evidence_ids": evidence_ids,
        "basis": "fan_in_senders_v1 / fan_out_receivers_v1 (aggregate rollups)",
        "note": (
            "aggregate queries return per-counterparty rollups, not individual "
            "transaction ids; evidence independence for this family cannot be "
            "established from observations"
        ),
    }


def family_observation_sources(state) -> dict[str, dict[str, Any]]:
    """Per-family observations: txn ids where resolvable, accounts, evidence ids."""
    return {
        FAMILY_CIRCULATION: circulation_observations(state),
        FAMILY_RAPID_FORWARDING: rapid_observations(state),
        FAMILY_FAN_IN_OUT: fan_in_out_observations(state),
    }


def signal_provenance(state) -> dict[str, Any]:
    """Trace detected signal families to their underlying observations.

    Pure function over the investigation state. Reports and review use it to
    distinguish feature diversity on the same transactions from genuinely
    independent evidential support; the target-scoped assessment below is what
    the action-eligibility policy (v2) counts.
    """
    families = detect_signal_families(state)
    sources = family_observation_sources(state)

    detected = sorted(families)
    pairs: list[dict[str, Any]] = []
    for i, name_a in enumerate(detected):
        for name_b in detected[i + 1:]:
            a, b = sources[name_a], sources[name_b]
            classification = classify_observation_overlap(a, b)
            shared = sorted(set(a.get("txn_ids") or []) & set(b.get("txn_ids") or []))
            pairs.append({
                "pair": [name_a, name_b],
                "classification": classification,
                "shared_txn_ids": shared,
                "shared_count": len(shared),
                "unique_txn_ids": {
                    name_a: sorted(set(a.get("txn_ids") or []) - set(b.get("txn_ids") or [])),
                    name_b: sorted(set(b.get("txn_ids") or []) - set(a.get("txn_ids") or [])),
                },
            })

    return {
        "families": {name: sources[name] for name in detected},
        "families_detected": detected,
        "pairs": pairs,
        "distinct_pairs": [p["pair"] for p in pairs if p["classification"] == _OVERLAP_DISTINCT],
        "overlapping_pairs": [
            p["pair"] for p in pairs
            if p["classification"] in (_OVERLAP_PARTIAL, _OVERLAP_SHARED)
        ],
        "unavailable_pairs": [
            p["pair"] for p in pairs if p["classification"] == _OVERLAP_UNAVAILABLE
        ],
    }


def distinct_provenance_families(
    names: Iterable[str], sources: dict[str, dict[str, Any]]
) -> list[str]:
    """Largest subset of ``names`` whose members are pairwise distinct in provenance.

    Documented overlap rule (action policy v2): two families count as two
    distinct provenance families only when their observation sets share ZERO
    transaction ids. Any overlap — partial or contained — makes the pair
    contribute at most one counted family; a family whose observations cannot
    be resolved to individual transactions (``provenance_unavailable``) can
    never be the second family of a pair. Ties are broken deterministically:
    names are sorted, and the first qualifying subset in
    ``itertools.combinations`` order is returned.
    """
    from itertools import combinations

    ordered = sorted(names)
    for size in range(len(ordered), 0, -1):
        for combo in combinations(ordered, size):
            if all(
                classify_observation_overlap(sources[a], sources[b]) == OVERLAP_DISTINCT
                for a, b in combinations(combo, 2)
            ):
                return list(combo)
    return []


def signal_family_assessment(state, target_account_id: str) -> dict[str, Any]:
    """Provenance-aware, target-scoped signal-family assessment (policy v2).

    Deterministic Python over the recorded evidence. A detected family only
    counts when its observations implicate the proposed action target, and the
    number of *distinct-provenance* families is the size of the largest
    pairwise-disjoint subset of those target-relevant families (see
    ``distinct_provenance_families``). The assessment decides nothing itself:
    the policy turns it into checks and blocking reasons, and the action
    record persists it for audit.
    """
    families = detect_signal_families(state)
    sources = family_observation_sources(state)
    detected = sorted(families)

    relevant: list[str] = []
    excluded: list[dict[str, str]] = []
    for name in detected:
        accounts = [str(a) for a in (sources[name].get("accounts") or [])]
        if not accounts:
            excluded.append({
                "family": name,
                "reason": "its observations could not be attributed to any account",
            })
        elif target_account_id not in accounts:
            shown = ", ".join(accounts[:5]) + ("..." if len(accounts) > 5 else "")
            excluded.append({
                "family": name,
                "reason": (
                    f"its observations concern {shown} but not the proposed "
                    f"target {target_account_id}"
                ),
            })
        else:
            relevant.append(name)

    counted = distinct_provenance_families(relevant, sources)

    pairs: list[dict[str, Any]] = []
    for i, name_a in enumerate(detected):
        for name_b in detected[i + 1:]:
            a, b = sources[name_a], sources[name_b]
            shared = sorted(set(a.get("txn_ids") or []) & set(b.get("txn_ids") or []))
            pairs.append({
                "pair": [name_a, name_b],
                "classification": classify_observation_overlap(a, b),
                "shared_txn_ids": shared,
                "shared_count": len(shared),
            })

    backing_ids: list[str] = []
    for name in relevant:
        for evidence_id in sources[name].get("evidence_ids") or []:
            if evidence_id not in backing_ids:
                backing_ids.append(evidence_id)

    return {
        "target_account_id": target_account_id,
        "detected": detected,
        "target_relevant": relevant,
        "excluded": excluded,
        "distinct_provenance_families": counted,
        "pairs": pairs,
        "overlapping_pairs": [
            p["pair"] for p in pairs
            if p["classification"] in (OVERLAP_PARTIAL, OVERLAP_SHARED)
        ],
        "sources": {
            name: {
                "available": bool(sources[name].get("available")),
                "basis": sources[name].get("basis"),
                "accounts": [str(a) for a in (sources[name].get("accounts") or [])],
                "evidence_ids": list(sources[name].get("evidence_ids") or []),
                "txn_ids": sources[name].get("txn_ids"),
            }
            for name in detected
        },
        "target_backing_evidence_ids": backing_ids,
        "rule": (
            "two signal families count as two distinct provenance families only "
            "when both implicate the proposed target and their observation sets "
            "share no transaction ids; overlapping or unresolvable families "
            "contribute at most one counted family"
        ),
    }


def benign_discriminators(state) -> list[str]:
    """Observations that positively support a benign reading (not mere absence)."""
    found: list[str] = []
    for record in state.evidence:
        if not record.ok:
            continue
        if record.tool == "fan_in_out":
            fan = record.derived
            if fan["distinct_senders"] >= FAN_IN_MIN_SENDERS and fan["out_amount"] == 0:
                found.append(
                    f"{fan['distinct_senders']} senders but no outgoing volume in the "
                    "window (no cash-out behaviour observed)"
                )
            elif (
                fan["distinct_senders"] >= FAN_IN_MIN_SENDERS
                and not fan_in_is_concentrated(fan)
            ):
                found.append(
                    "high sender count without forwarding concentration "
                    f"(out/in ratio {fan['out_in_ratio']}, top receiver share "
                    f"{fan['top_receiver_share']})"
                )
        if record.tool == "detect_cycle" and record.ok:
            if record.derived.get("structural_only_count") and not record.derived.get(
                "corroborated_count"
            ):
                found.append(
                    "cycle detected structurally but not corroborated by transaction "
                    "timestamps — no evidence funds circulated"
                )
        if record.tool == "shared_identifiers" and record.ok:
            if record.derived.get("shared_devices") or record.derived.get(
                "shared_identifiers"
            ):
                found.append(
                    "shared device/identifier infrastructure present, which is an "
                    "association signal rather than evidence of fraud"
                )
    return found


def unresolved_questions(state) -> list[str]:
    """Questions no successful evidence answers — never reported as 'no risk'."""
    asked = {r.tool for r in state.evidence if r.ok}
    questions: list[str] = []
    if "entity_profile" not in asked:
        questions.append("the target account profile was never obtained")
    if "detect_cycle" not in asked:
        questions.append("no cycle check was performed")
    if "fan_in_out" not in asked:
        questions.append("no fan-in/fan-out check was performed")
    if "trace_funds" not in asked:
        questions.append("fund routing beyond direct counterparties was not traced")
    if "shared_identifiers" not in asked:
        questions.append("shared infrastructure between counterparties was not examined")
    if any(not r.ok for r in state.evidence):
        questions.append("one or more tool calls failed, leaving evidence gaps")
    if any(any(r.truncated) for r in state.evidence if r.ok):
        questions.append(
            "one or more tool results were truncated, so counts may be incomplete"
        )
    return questions


def compute_verdict(state) -> tuple[str, str, list[str], list[str]]:
    """Return (verdict, rationale, signal_families, limitations). Deterministic."""
    families = detect_signal_families(state)
    family_names = sorted(families)
    benign = benign_discriminators(state)
    limitations: list[str] = []

    corroborated_cycles = families.get(FAMILY_CIRCULATION, {}).get("cycles", [])
    structural_only = any(
        r.tool == "detect_cycle"
        and r.ok
        and r.derived.get("structural_only_count")
        and not r.derived.get("corroborated_count")
        for r in state.evidence
    )

    if len(family_names) >= 2:
        verdict = VERDICT_SUSPICIOUS
        rationale = (
            f"{len(family_names)} independent signal families support review: "
            f"{', '.join(family_names)}. "
        )
        if corroborated_cycles:
            rationale += (
                f"{len(corroborated_cycles)} directed cycle(s) had transaction "
                "timestamps that advance around the ring, so circulation is "
                "corroborated rather than merely structural. "
            )
        rationale += (
            "Circular movement and rapid forwarding are risk indicators for human "
            "review; on their own they are not a determination that any person "
            "committed an offence."
        )
        limitations.append(
            "Signal families can overlap: the same underlying transfers can produce "
            "both a corroborated cycle and a rapid-forwarding signal, so the family "
            "count is not an independent statistical test."
        )
    elif len(family_names) == 1:
        verdict = VERDICT_INSUFFICIENT
        rationale = (
            f"Only one signal family ({family_names[0]}) is supported by evidence. "
            "A single signal is not enough to characterise the activity, so no "
            "suspicious-activity verdict is returned."
        )
        if structural_only:
            rationale += (
                " The target does sit on a directed ring in the aggregate transfer "
                "topology, but the transaction timestamps do NOT advance around it, "
                "so that ring is structural only and is not evidence that funds "
                "circulated."
            )
    elif structural_only:
        verdict = VERDICT_INSUFFICIENT
        rationale = (
            "The target sits on a directed ring in the aggregate transfer topology, "
            "but the transaction timestamps do NOT advance around it. That is a "
            "structural cycle only and it is not evidence that funds circulated, so "
            "the investigation is inconclusive."
        )
    elif benign:
        verdict = VERDICT_LIKELY_BENIGN
        rationale = (
            "No supported signal family was found, and the evidence contains "
            "positive benign discriminators: " + "; ".join(benign) + "."
        )
    else:
        verdict = VERDICT_INSUFFICIENT
        rationale = (
            "No supported signal family was found, but the available evidence is "
            "also too thin to describe the activity as benign."
        )

    limitations.extend(unresolved_questions(state))
    if not state.evidence:
        limitations.append("no evidence was collected at all")
    if state.termination_reason == "tool_budget_exhausted":
        limitations.append(
            "the tool budget was exhausted before the investigation finished, so the "
            "conclusion is provisional and incomplete"
        )
    return verdict, rationale, family_names, limitations


def summarize_evidence(record) -> str:
    """One-line, deterministic summary of an evidence record."""
    if not record.ok:
        return f"{record.tool} failed: {record.error or 'unknown error'}"
    d = record.derived
    if record.tool == "alert_context":
        return (
            f"alert resolved: {d.get('reason')} on txn "
            f"{(d.get('trigger_txn') or {}).get('txn_id')}; entities {d.get('entity_ids')}"
        )
    if record.tool == "entity_profile":
        return (
            f"profile: status={d.get('status')} incoming="
            f"{(d.get('incoming') or {}).get('total_amount')} outgoing="
            f"{(d.get('outgoing') or {}).get('total_amount')} prior_cases="
            f"{d.get('prior_case_count')}"
        )
    if record.tool == "detect_cycle":
        return (
            f"cycles={d.get('cycle_count')} corroborated={d.get('corroborated_count')} "
            f"structural_only={d.get('structural_only_count')}"
        )
    if record.tool == "fan_in_out":
        return (
            f"senders={d.get('distinct_senders')} receivers={d.get('distinct_receivers')} "
            f"out_in={d.get('out_in_ratio')} top_share={d.get('top_receiver_share')}"
        )
    if record.tool == "trace_funds":
        return (
            f"paths={d.get('path_count')} leg_events={d.get('observed_leg_events')} "
            f"rapid_forward={d.get('rapid_forward_count')}"
        )
    if record.tool == "shared_identifiers":
        return (
            f"shared_devices={len(d.get('shared_devices') or [])} "
            f"shared_identifiers={len(d.get('shared_identifiers') or [])}"
        )
    if record.tool == "hub_scores":
        return f"hub ranking returned {len(d.get('hubs') or [])} accounts (weighted in-degree)"
    if record.tool == "prior_cases":
        return f"prior cases={d.get('case_count')}"
    return record.interpretation[:120]


def summarize_all(records: Iterable) -> str:
    return "; ".join(summarize_evidence(r) for r in records)


# ---- hypothesis and claim recomputation ------------------------------------


def recompute_hypotheses(state) -> list[Any]:
    """Rebuild every hypothesis from the evidence collected so far.

    Statuses are derived, never asserted: a hypothesis is only "supported"
    when a real tool result bears on it, "contradicted" when a real result
    undercuts it, and "insufficient"/"open" when the discriminating test has
    not been run. Absence of evidence is never recorded as evidence of absence.
    """
    from src.agent.state import (
        H_BENIGN,
        H_CIRCULATION,
        H_FAN_IN_OUT,
        H_INSUFFICIENT,
        H_RAPID_MOVEMENT,
        H_SHARED_IDENTITY,
        Hypothesis,
    )

    families = detect_signal_families(state)
    benign = benign_discriminators(state)
    hypotheses: list[Hypothesis] = []

    def cycle_records():
        return [r for r in state.evidence if r.tool == "detect_cycle" and r.ok]

    # circulation
    circ = Hypothesis(name=H_CIRCULATION)
    for record in cycle_records():
        d = record.derived
        if d.get("corroborated_count"):
            circ.supporting_evidence.append(record.evidence_id)
        elif d.get("cycle_count"):
            circ.contradicting_evidence.append(record.evidence_id)
            circ.uncertainties.append(
                "a ring exists in the aggregate topology but the transaction "
                "timestamps do not corroborate circulation"
            )
        else:
            circ.contradicting_evidence.append(record.evidence_id)
    if not cycle_records():
        circ.status = "insufficient"
        circ.uncertainties.append("no cycle check was completed")
    elif circ.supporting_evidence:
        circ.status = "supported"
        circ.rationale = (
            "timestamped transactions advance around the ring, so circulation is "
            "corroborated rather than structural"
        )
    else:
        circ.status = "contradicted"
        circ.rationale = "no corroborated circulation found for the targets examined"
    hypotheses.append(circ)

    # coordinated fan-in / cash-out
    fan = Hypothesis(name=H_FAN_IN_OUT)
    fan_records = [r for r in state.evidence if r.tool == "fan_in_out" and r.ok]
    for record in fan_records:
        if fan_in_is_concentrated(record.derived):
            fan.supporting_evidence.append(record.evidence_id)
        else:
            fan.contradicting_evidence.append(record.evidence_id)
    if not fan_records:
        fan.status = "insufficient"
        fan.uncertainties.append("no fan-in/fan-out check was completed")
    else:
        fan.status = "supported" if fan.supporting_evidence else "contradicted"
        fan.rationale = (
            "incoming volume from many senders is forwarded to a single destination"
            if fan.supporting_evidence
            else "high sender counts were NOT accompanied by forwarding concentration"
        )
    hypotheses.append(fan)

    # rapid movement
    rapid = Hypothesis(name=H_RAPID_MOVEMENT)
    trace_records = [r for r in state.evidence if r.tool == "trace_funds" and r.ok]
    rapid_hits = [
        rapid_forwarding_for_account(state, account_id)
        for account_id in state.primary_entity_ids
    ]
    rapid_supported = [hit for hit in rapid_hits if hit["rapid_forward_count"] > 0]
    for hit in rapid_supported:
        rapid.supporting_evidence.extend(hit["evidence_ids"])
    for record in trace_records:
        if not any(record.evidence_id in hit["evidence_ids"] for hit in rapid_supported):
            rapid.contradicting_evidence.append(record.evidence_id)
    if not trace_records:
        rapid.status = "insufficient"
        rapid.uncertainties.append("onward routing and timing were not traced")
    else:
        rapid.status = "supported" if rapid.supporting_evidence else "contradicted"
        rapid.rationale = (
            f"incoming funds were moved onward within {RAPID_MAX_GAP_MINUTES} minutes "
            f"at >= {RAPID_MIN_FORWARD_RATIO:.0%} of the inbound amount"
            if rapid.supporting_evidence
            else "no inbound-to-outbound legs matched the rapid-forwarding criteria"
        )
    hypotheses.append(rapid)

    # shared identity link
    shared = Hypothesis(name=H_SHARED_IDENTITY)
    shared_records = [
        r for r in state.evidence if r.tool == "shared_identifiers" and r.ok
    ]
    for record in shared_records:
        d = record.derived
        if d.get("shared_devices") or d.get("shared_identifiers"):
            shared.supporting_evidence.append(record.evidence_id)
        else:
            shared.contradicting_evidence.append(record.evidence_id)
    if not shared_records:
        shared.status = "open"
        shared.uncertainties.append(
            "shared infrastructure between the counterparties was not examined"
        )
    else:
        shared.status = "supported" if shared.supporting_evidence else "contradicted"
        shared.rationale = (
            "the counterparties share synthetic device/identifier infrastructure — an "
            "association signal that still requires a flow-based explanation"
            if shared.supporting_evidence
            else "no shared device or identifier was found between the counterparties"
        )
    hypotheses.append(shared)

    # benign explanation
    benign_h = Hypothesis(name=H_BENIGN)
    # Rapid-forwarding evidence must be judged on the AGGREGATED trace view:
    # a single-direction trace never contains both legs, so per-record absence
    # of rapid movement would wrongly "support" the benign reading.
    aggregated_rapid = any(
        rapid_forwarding_for_account(state, account_id)["rapid_forward_count"] > 0
        for account_id in state.primary_entity_ids
    )
    for record in state.evidence:
        if not record.ok:
            continue
        if (
            record.tool == "fan_in_out"
            and record.derived["distinct_senders"] >= FAN_IN_MIN_SENDERS
            and not fan_in_is_concentrated(record.derived)
        ):
            benign_h.supporting_evidence.append(record.evidence_id)
        if record.tool == "detect_cycle" and record.derived.get("structural_only_count"):
            benign_h.supporting_evidence.append(record.evidence_id)
        if record.tool == "trace_funds" and not aggregated_rapid:
            benign_h.supporting_evidence.append(record.evidence_id)
    contradicting_ids: set[str] = set()
    family_citations: list[str] = []
    for name in sorted(families):
        group_citations: list[str] = []
        for group in families[name].values():
            for entry in group if isinstance(group, list) else []:
                if isinstance(entry, dict) and entry.get("evidence_id"):
                    contradicting_ids.add(entry["evidence_id"])
                    if entry["evidence_id"] not in group_citations:
                        group_citations.append(entry["evidence_id"])
        if group_citations:
            family_citations.append(f"{name} ({', '.join(group_citations)})")
    benign_h.contradicting_evidence = sorted(contradicting_ids)
    benign_h.status = (
        "contradicted" if benign_h.contradicting_evidence
        else ("supported" if benign_h.supporting_evidence else "open")
    )
    # Phase 6.1: a benign contradiction must rest on POSITIVE conflicting
    # evidence — signal families backed by real tool results — and the
    # rationale cites it. The absence of a benign discriminator is never
    # reported as the reason the benign reading was contradicted.
    if benign_h.contradicting_evidence:
        benign_h.rationale = (
            "positive conflicting observations from signal families backed by "
            "evidence undercut a benign reading: " + "; ".join(family_citations)
        )
        if benign:
            benign_h.rationale += (
                "; counterbalanced by positive benign discriminators: "
                + "; ".join(benign)
            )
    elif benign_h.supporting_evidence:
        benign_h.rationale = (
            "; ".join(benign) if benign
            else "positive benign discriminators were observed in the "
            "collected evidence"
        )
    else:
        benign_h.rationale = (
            "no positive benign discriminator was observed, and no positive "
            "signal family contradicts a benign reading yet; the benign "
            "explanation remains open rather than contradicted"
        )
    hypotheses.append(benign_h)

    # insufficient evidence
    insufficient = Hypothesis(name=H_INSUFFICIENT)
    insufficient.uncertainties.extend(unresolved_questions(state))
    # Phase 6.1: the insufficiency hypothesis is contradicted only when the
    # SAME distinct-provenance counting rule the action policy uses (policy
    # v2) yields two or more target-relevant, mutually independent families.
    # Two signal labels over one shared observation set do not distinguish
    # the competing explanations.
    target = state.primary_entity_ids[0] if state.primary_entity_ids else None
    assessment = signal_family_assessment(state, target) if target else None
    counted = list((assessment or {}).get("distinct_provenance_families") or [])
    detected = sorted(families)
    if len(counted) >= 2:
        insufficient.status = "contradicted"
        insufficient.rationale = (
            f"{len(counted)} target-relevant signal families with distinct "
            f"provenance are supported by evidence ({', '.join(counted)}), so "
            "the evidence distinguishes a concrete risk pattern from the "
            "alternatives"
        )
    elif not state.evidence:
        insufficient.status = "supported"
        insufficient.rationale = "no evidence was collected"
    else:
        insufficient.status = "supported"
        overlap_pairs = [
            p for p in (assessment or {}).get("overlapping_pairs") or []
        ]
        if target is None and detected:
            insufficient.rationale = (
                "signal labels were detected but no target account was "
                "attributed to the alert, so the signals cannot be scoped to "
                "the account under review; the available evidence does not "
                "distinguish the competing explanations"
            )
        elif len(detected) >= 2 and overlap_pairs:
            pretty = "; ".join("+".join(p) for p in overlap_pairs)
            insufficient.rationale = (
                f"{len(detected)} signal labels were detected but the counting "
                "rule does not treat them as independent evidence "
                f"(overlapping observations: {pretty}); the available evidence "
                "does not distinguish the competing explanations"
            )
        elif detected and not counted:
            insufficient.rationale = (
                "signal labels were detected but none implicates the reviewed "
                "target with resolvable transaction provenance; the available "
                "evidence does not distinguish the competing explanations"
            )
        else:
            insufficient.rationale = (
                "the available evidence does not distinguish the competing "
                "explanations"
            )
    hypotheses.append(insufficient)
    return hypotheses


def recompute_claims(state) -> list[Any]:
    """Record which factual claims are supported, contradicted, or unverified."""
    from src.agent.state import ClaimStatus, H_CIRCULATION, H_FAN_IN_OUT, H_RAPID_MOVEMENT

    claims: list[ClaimStatus] = []
    ctx = next((r for r in state.evidence if r.tool == "alert_context" and r.ok), None)
    if ctx and ctx.derived.get("found"):
        txn = (ctx.derived.get("trigger_txn") or {}).get("txn_id")
        claims.append(ClaimStatus(
            claim=f"alert {state.alert_id} was triggered by transaction {txn}",
            status="supported", evidence_ids=[ctx.evidence_id],
        ))
    else:
        claims.append(ClaimStatus(
            claim=f"alert {state.alert_id} was resolved to a trigger transaction",
            status="unverified",
        ))

    for hypothesis_name, text in (
        (H_CIRCULATION, "the target participates in a temporally corroborated cycle"),
        (H_FAN_IN_OUT, "incoming volume is forwarded to a single destination"),
        (H_RAPID_MOVEMENT, "incoming funds are moved onward within 90 minutes"),
        ("shared_identifier_link_warranting_review",
         "the counterparties share device or identifier infrastructure"),
    ):
        record = state.hypothesis(hypothesis_name)
        if record is None:
            continue
        if record.supporting_evidence:
            status, ids = "supported", list(record.supporting_evidence)
        elif record.contradicting_evidence:
            status, ids = "contradicted", list(record.contradicting_evidence)
        else:
            status, ids = "unverified", []
        claims.append(ClaimStatus(claim=text, status=status, evidence_ids=ids))
    return claims

# --------------------------------------------------------------- evidence stance

# Tool -> stance mapping. Deterministic and deliberately conservative: a
# contextual tool never counts as supporting evidence, and a failed call is
# always recorded as unresolved rather than silently dropped.
# (Phase 8.6: moved here from workflow.py so the loop can label an evidence
# record the moment it is appended, without a circular import.)
_CONTEXTUAL_TOOLS = {"hub_scores", "prior_cases"}


def evidence_stance(record) -> str:
    """Classify one evidence record as supporting / contradicting / unresolved."""
    if not record.ok:
        return "unresolved"
    tool = record.tool
    derived = record.derived or {}
    if tool in _CONTEXTUAL_TOOLS:
        return "unresolved"
    if tool in ("alert_context", "entity_profile"):
        return "supporting"
    if tool == "detect_cycle":
        return "supporting" if derived.get("corroborated_count") else "contradicting"
    if tool == "fan_in_out":
        return "supporting" if fan_in_is_concentrated(derived) else "contradicting"
    if tool == "trace_funds":
        return "unresolved"  # resolved by the caller using the aggregated view
    if tool == "shared_identifiers":
        return (
            "supporting"
            if derived.get("shared_devices") or derived.get("shared_identifiers")
            else "contradicting"
        )
    return "unresolved"


def evidence_nodes_and_edges(record) -> tuple[list[str], list[list[str]]]:
    """Accounts and directed legs a tool actually returned (real rows only).

    Used by the UI to highlight newly implicated graph elements while an
    investigation is still running. Nothing is inferred: only accounts the
    tool returned (``entity_ids``) and legs present in the derived payload
    (cycle legs/events, traced leg events).
    """
    nodes: list[str] = []
    for value in (record.entity_ids or []):
        if value and value not in nodes:
            nodes.append(value)
    edges: list[list[str]] = []
    derived = record.derived or {}
    for cycle in derived.get("cycles") or []:
        for leg in cycle.get("legs") or []:
            pair = [leg.get("src"), leg.get("dst")]
            if all(pair) and pair not in edges:
                edges.append(pair)
    for evt in derived.get("leg_events") or []:
        pair = [evt.get("src"), evt.get("dst")]
        if all(pair) and pair not in edges:
            edges.append(pair)
    for pair in edges:
        for value in pair:
            if value not in nodes:
                nodes.append(value)
    return nodes, edges
