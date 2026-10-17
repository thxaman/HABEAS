"""Phase 2 graph algorithms (PLAN.md §8) — plain Python, no LLM, no MCP.

`detect_cycle` finds bounded directed cycles and separates **structural**
cycles from **temporally corroborated** ones. A structural cycle is never
presented as proof that funds circulated.

Path direction is always reconstructed from Edge endpoints via the shared
`reads.reconstruct_path_accounts` helper — `nodes(path)` order is unreliable
for closed cycles on this server (probe P0-17).

The temporal check here is a pure function local to the tool layer, keeping
agent-facing tools independent of the trusted seed package. It mirrors the
fixture detector in `src/seed/patterns.py` deliberately: seed code must not
be imported by agent-facing code (ground-truth isolation, PLAN.md §5).
"""
from __future__ import annotations

import time
from typing import Any, Sequence

from src.tools import query_catalog as qc
from src.tools.envelopes import (
    ROW_LIMIT_DEFAULT,
    ToolEnvelope,
    ToolValidationError,
    error_envelope,
    make_envelope,
    run_evidence,
)
from src.tools.reads import (
    _edge_from_normalized,
    _events_for_legs,
    _node_from_normalized,
    _window,
    reconstruct_path_accounts,
)

_MAX_CORROBORATION_CYCLES = 10  # cap event lookups per detect_cycle call

# A cycle is only "temporally corroborated" if its ring traffic completes at
# least this many full chronological laps. One lap is too easy for random
# noise to produce by chance, so it is reported as structural only.
MIN_CORROBORATION_ROUNDS = 2


# ---- pure algorithm helpers (unit-testable without a database) -------------


def canonical_cycle(members: Sequence[str]) -> tuple[str, ...]:
    """Rotate a cycle so its smallest id leads; drop any closing repeat.

    Rotation canonicalization means the same cycle reached from different
    starting accounts is reported once (PLAN.md §8).
    """
    seq = list(members)
    if len(seq) > 1 and seq[0] == seq[-1]:
        seq = seq[:-1]  # tolerate a path-shape input that closes on itself
    i = seq.index(min(seq))
    return tuple(seq[i:] + seq[:i])


def cycle_temporally_consistent(
    events: Sequence[tuple[str, str, str, float]],
    ring_order: Sequence[str] | None = None,
) -> bool:
    """True iff the timestamped events support *chronological circulation*.

    ``events`` are (src, dst, ts, amount) tuples; ``ring_order`` is the
    directed cycle's account order (as reconstructed from Edge endpoints).

    The check is deliberately stronger than "the legs form a loop": it maps
    every event to its position in the ring, then requires the ts-sorted
    events to advance exactly one leg at a time around the ring for at least
    MIN_CORROBORATION_ROUNDS complete rounds. A single pass is not enough —
    random noise readily yields one plausible-looking lap, while genuine
    repeated circulation shows the ring traversed more than once. Rings whose
    traffic is duplicated, off-ring, out of order, or only a single lap
    return False and must be reported as **structural only** (PLAN.md §6/§8).

    With ``ring_order=None`` it degrades to a plain chaining test.
    """
    if len(events) < 2:
        return False
    ordered = sorted(events, key=lambda e: e[2])
    if ring_order is None:
        return all(
            ordered[i][1] == ordered[(i + 1) % len(ordered)][0]
            for i in range(len(ordered))
        )
    n = len(ring_order)
    if n < 2:
        return False
    leg_index = {(ring_order[i], ring_order[(i + 1) % n]): i for i in range(n)}
    seq: list[int] = []
    for src, dst, _, _ in ordered:
        idx = leg_index.get((src, dst))
        if idx is None:
            return False  # an event that is not one of the ring's legs
        seq.append(idx)
    if len(seq) % n != 0:
        return False  # does not complete whole rounds around the ring
    if len(seq) // n < MIN_CORROBORATION_ROUNDS:
        return False  # a single lap is too weak to corroborate circulation
    return all((seq[i + 1] - seq[i]) % n == 1 for i in range(len(seq) - 1))


def _accounts_exist(graph, account_id: str) -> bool:
    ev = run_evidence(
        graph, "entity_profile_lookup_v1",
        qc.get_template("entity_profile_lookup_v1"),
        {"entity_id": account_id},
    )
    return bool(ev.rows) and ev.rows[0][0] == "Account"


# ---------------------------------------------------------------- detect_cycle


def detect_cycle(
    graph,
    account_id: str,
    *,
    max_len: int = 4,
    window_days: int = 90,
    limit: int = ROW_LIMIT_DEFAULT,
) -> ToolEnvelope:
    t0 = time.perf_counter()
    try:
        account_id = qc.validate_entity_id(account_id, "account_id")
        max_len = qc.validate_cycle_len(max_len)
        window_days = qc.validate_window_days(window_days)
        limit = qc.validate_limit(limit)

        cypher = qc.bind_hop_bound("detect_closed_paths_v1", "<<max_len>>", max_len)
        paths_ev = run_evidence(
            graph, "detect_closed_paths_v1", cypher,
            {"account_id": account_id, "limit": limit},
            interpolated=["max_len"], bound={"<<max_len>>": max_len}, row_cap=limit,
        )
        evidence = [paths_ev]

        if not paths_ev.rows:
            exists = _accounts_exist(graph, account_id)
            result = {
                "account_id": account_id,
                "max_len": max_len,
                "cycle_count": 0,
                "temporally_corroborated_count": 0,
                "cycles": [],
                "account_exists": exists,
                "truncated": paths_ev.truncated,
            }
            interpretation = (
                f"No directed cycle of at most {max_len} accounts was found "
                + (f"through {account_id}" if exists
                   else f"(account {account_id} does not exist)")
                + " in the active transfer topology. Absence of a cycle within "
                f"{max_len} hops is not evidence of legitimate activity."
            )
            return make_envelope("detect_cycle", evidence, result, interpretation, t0)

        # Deduplicate by rotation-canonical membership; keep one representative.
        cycles: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in paths_ev.rows:
            rels, nodes = row[0], row[1]
            raw_rels = [_edge_from_normalized(r) for r in rels]
            raw_nodes = [_node_from_normalized(n) for n in nodes]
            sequence, legs = reconstruct_path_accounts(raw_nodes, raw_rels)
            key = canonical_cycle(sequence)
            if len(key) < 2:
                continue
            cycles.setdefault(key, {"members": list(key), "legs": legs})

        window, window_ev = _window(graph, window_days)
        evidence.extend(window_ev)

        cycle_results = []
        for key, data in cycles.items():
            legs = data["legs"]
            events: list[dict[str, Any]] = []
            if legs and len(cycle_results) < _MAX_CORROBORATION_CYCLES:
                ev_rows, events_ev = _events_for_legs(
                    graph, legs, window["cutoff"], limit
                )
                evidence.append(events_ev)
                events = ev_rows
            event_tuples = [(e["src"], e["dst"], e["ts"], e["amount"]) for e in events]
            corroborated = cycle_temporally_consistent(event_tuples, list(key))
            cycle_results.append({
                "members": data["members"],
                "length": len(key),
                "legs": [{"src": s, "dst": d} for s, d in legs],
                "structurally_closed": True,
                "temporally_corroborated": corroborated,
                "event_count": len(events),
                "events": events if corroborated else events[:8],
            })

        cycle_results.sort(
            key=lambda c: (-int(c["temporally_corroborated"]), c["length"])
        )
        corroborated_count = sum(
            1 for c in cycle_results if c["temporally_corroborated"]
        )
        result = {
            "account_id": account_id,
            "max_len": max_len,
            "window": window,
            "cycle_count": len(cycle_results),
            "temporally_corroborated_count": corroborated_count,
            "cycles": cycle_results,
            "truncated": paths_ev.truncated,
        }
        interpretation = (
            f"Found {len(cycle_results)} distinct directed cycle(s) of length "
            f"2..{max_len} through {account_id} in the active transfer topology; "
            f"{corroborated_count} of them are temporally corroborated "
            "(timestamped transactions chain around the ring). "
        )
        if corroborated_count:
            interpretation += (
                "A temporally corroborated cycle indicates funds could have "
                "circulated in order, but circular movement alone does not "
                "establish wrongdoing."
            )
        else:
            interpretation += (
                "Only structural cycles were found: the aggregate topology forms a "
                "ring, but the transaction timestamps do NOT chain around it, so "
                "this is not evidence that the same funds circulated."
            )
        if paths_ev.truncated:
            interpretation += " Path rows hit the limit; the cycle set may be incomplete."
        return make_envelope("detect_cycle", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("detect_cycle", exc, t0)


# ---------------------------------------------------------------- hub_scores


def hub_scores(graph, *, window_days: int = 30, limit: int = 10) -> ToolEnvelope:
    t0 = time.perf_counter()
    try:
        window_days = qc.validate_window_days(window_days)
        if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= 50):
            raise ToolValidationError(f"limit must be in [1, 50], got {limit!r}")

        window, evidence = _window(graph, window_days)
        ev = run_evidence(
            graph, "hub_weighted_in_degree_v1",
            qc.get_template("hub_weighted_in_degree_v1"),
            {"cutoff": window["cutoff"], "limit": limit}, row_cap=limit,
        )
        evidence.append(ev)

        hubs = [
            {
                "account_id": r[0],
                "weighted_in_amount": round(float(r[1]), 2),
                "distinct_senders": int(r[2]),
                "in_event_count": int(r[3]),
            }
            for r in ev.rows
        ]
        result = {
            "method": "weighted_in_degree",
            "description": "weighted in-degree: sum of active incoming TRANSFERRED_TO total_amount in window",
            "window": window,
            "hubs": hubs,
            "truncated": ev.truncated,
        }
        if hubs:
            top = hubs[0]
            interpretation = (
                f"Ranked accounts by weighted in-degree (sum of active incoming "
                f"transfer volume in the {window_days}-day window ending "
                f"{window['as_of'] or 'no data'}). Top account {top['account_id']} "
                f"has weighted in-amount {top['weighted_in_amount']:.2f} from "
                f"{top['distinct_senders']} distinct sender(s). This is a "
                "prioritization metric, NOT PageRank, and a high score is a lead "
                "for review — high legitimate volume also scores highly and is not "
                "evidence of misconduct."
            )
        else:
            interpretation = (
                "No active incoming transfers were found in the window, so no "
                "weighted in-degree ranking could be produced."
            )
        return make_envelope("hub_scores", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("hub_scores", exc, t0)