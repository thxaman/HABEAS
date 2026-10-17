"""Planted pattern families, the ground-truth manifest model, and the
deterministic detectors used by fixture tests and the evaluation harness.

The manifest NEVER lives on the graph (PLAN.md §5/§6): it is written to
src/seed/ground_truth.json, which is gitignored and must never be referenced
by the agent, its prompts, or the MCP tool layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# --- pattern families and signal ids (stable code constants) ---------------

FAMILY_CYCLE = "cycle"
FAMILY_CYCLE_RAPID = "cycle_rapid"
FAMILY_FAN_IN_OUT = "fan_in_out"
FAMILY_RAPID_SHARED = "rapid_shared"
FAMILY_BENIGN = "benign"

FamilyName = Literal["cycle", "cycle_rapid", "fan_in_out", "rapid_shared", "benign"]

SIGNAL_CIRCULAR_TRANSFER = "circular_transfer"
SIGNAL_TEMPORAL_CONSISTENCY = "temporal_consistency"
SIGNAL_FAN_IN_CONCENTRATION = "fan_in_concentration"
SIGNAL_SINGLE_FORWARDING_DESTINATION = "single_forwarding_destination"
SIGNAL_RAPID_IN_OUT = "rapid_in_out"
SIGNAL_SHARED_DEVICE = "shared_device"
SIGNAL_SHARED_IDENTIFIER = "shared_identifier"

# --- ground-truth manifest model -------------------------------------------


class AlertTruth(BaseModel):
    alert_id: str
    family: FamilyName
    label: Literal["suspicious", "benign"]
    trigger_transaction_id: str
    account_id: str
    expected_account_members: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)


class PatternTruth(BaseModel):
    family: FamilyName
    account_members: list[str]
    customer_members: list[str] = Field(default_factory=list)
    # Explicit ring membership for cycle-family patterns, so tests and the
    # evaluation harness can distinguish the circulating accounts from the
    # feeder/exit accounts that only participate in a rapid pass-through.
    ring_members: list[str] = Field(default_factory=list)
    temporally_consistent: bool | None = None
    notes: str = ""


class GroundTruth(BaseModel):
    schema_version: int = 1
    seed: int
    graph_name: str
    counts: dict[str, int]
    alerts: list[AlertTruth]
    patterns: list[PatternTruth]
    # Accounts/devices deliberately designed so a naive single-signal
    # detector would flag them (PLAN.md §6: noise needs a tempting trap).
    tempting_false_positives: list[str] = Field(default_factory=list)


# --- deterministic detectors (pure Python, no graph access) ----------------

# One underlying transaction event: (src_account, dst_account, ts_iso, amount)
Event = tuple[str, str, str, float]


def canonical_cycle(path: list[str]) -> tuple[str, ...]:
    """Rotate a cycle [a, b, c, d] so its smallest id comes first."""
    i = path.index(min(path))
    return tuple(path[i:] + path[:i])


def find_cycles(
    adjacency: dict[str, list[str]], max_len: int = 4
) -> list[tuple[str, ...]]:
    """All directed simple cycles of length 2..max_len (bounded depth).

    Canonicalized by rotation so the same cycle is reported once (PLAN.md §8).
    """
    if max_len < 2:
        return []
    found: set[tuple[str, ...]] = set()

    def dfs(start: str, node: str, path: list[str]) -> None:
        for nxt in adjacency.get(node, ()):
            if nxt == start and len(path) >= 2:
                found.add(canonical_cycle(path))
            elif nxt not in path and len(path) < max_len:
                dfs(start, nxt, path + [nxt])

    for start in adjacency:
        dfs(start, start, [start])
    return sorted(found)


def cycle_temporally_consistent(events: list[Event]) -> bool:
    """True iff the cycle's events, sorted by ts, chain around the ring.

    A structural cycle whose timestamps do NOT circulate chronologically
    must return False (PLAN.md §6/§8: distinguish structure from flow).
    """
    if len(events) < 2:
        return False
    ordered = sorted(events, key=lambda e: e[2])
    for i, event in enumerate(ordered):
        nxt = ordered[(i + 1) % len(ordered)]
        if event[1] != nxt[0]:
            return False
    return True


def fan_in_out(events: list[Event], account_id: str) -> dict:
    """Fan-in/fan-out profile for one account over the given events.

    Event count and amount volume stay separate; every ratio handles a zero
    denominator explicitly (PLAN.md §7/§8).
    """
    incoming = [e for e in events if e[1] == account_id]
    outgoing = [e for e in events if e[0] == account_id]
    in_amount = sum(e[3] for e in incoming)
    out_amount = sum(e[3] for e in outgoing)
    by_receiver: dict[str, float] = {}
    for e in outgoing:
        by_receiver[e[1]] = by_receiver.get(e[1], 0.0) + e[3]
    top_receiver = max(by_receiver, key=by_receiver.get) if by_receiver else None
    top_share = (
        round(by_receiver[top_receiver] / out_amount, 4)
        if top_receiver and out_amount > 0
        else None
    )
    return {
        "distinct_senders": len({e[0] for e in incoming}),
        "distinct_receivers": len({e[1] for e in outgoing}),
        "in_amount": round(in_amount, 2),
        "out_amount": round(out_amount, 2),
        "out_in_ratio": round(out_amount / in_amount, 4) if in_amount > 0 else None,
        "top_receiver": top_receiver,
        "top_receiver_share": top_share,
        "event_count": len(incoming) + len(outgoing),
    }


def rapid_in_out(
    events: list[Event],
    account_id: str,
    max_gap_minutes: int = 90,
    min_forward_ratio: float = 0.8,
) -> list[dict]:
    """Inbound transactions forwarded within a short interval.

    Returns matches: {"in_txn_ts", "out_ts", "gap_minutes", "in_amount",
    "out_amount", "forward_ratio"}. Empty list means no rapid movement.
    """
    fmt = "%Y-%m-%dT%H:%M:%SZ"

    def parse(ts: str) -> datetime:
        return datetime.strptime(ts, fmt)

    inbound = sorted((e for e in events if e[1] == account_id), key=lambda e: e[2])
    outbound = sorted((e for e in events if e[0] == account_id), key=lambda e: e[2])
    matches: list[dict] = []
    for in_evt in inbound:
        in_dt = parse(in_evt[2])
        for out_evt in outbound:
            gap = (parse(out_evt[2]) - in_dt).total_seconds() / 60.0
            if gap < 0:
                continue
            if gap > max_gap_minutes:
                break
            if out_evt[3] >= min_forward_ratio * in_evt[3]:
                matches.append({
                    "in_txn_ts": in_evt[2],
                    "out_ts": out_evt[2],
                    "gap_minutes": round(gap, 1),
                    "in_amount": in_evt[3],
                    "out_amount": out_evt[3],
                    "forward_ratio": round(out_evt[3] / in_evt[3], 4)
                    if in_evt[3] > 0
                    else None,
                })
                break
    return matches


def weighted_in_degree(edges: list[dict]) -> list[dict]:
    """Hub metric from active TRANSFERRED_TO aggregate edges (PLAN.md §8).

    This is weighted in-degree over the derived projection, NOT PageRank.
    edges: [{"src", "dst", "total_amount", "count"}].
    Returns per-account: weighted_in_amount, distinct_senders, in_edge_count.
    """
    per_account: dict[str, dict] = {}
    for edge in edges:
        acc = per_account.setdefault(
            edge["dst"],
            {"account_id": edge["dst"], "weighted_in_amount": 0.0,
             "distinct_senders": set(), "in_edge_count": 0},
        )
        acc["weighted_in_amount"] += edge["total_amount"]
        acc["distinct_senders"].add(edge["src"])
        acc["in_edge_count"] += edge["count"]
    result = []
    for acc in per_account.values():
        result.append({
            "account_id": acc["account_id"],
            "weighted_in_amount": round(acc["weighted_in_amount"], 2),
            "distinct_senders": len(acc["distinct_senders"]),
            "in_edge_count": acc["in_edge_count"],
        })
    return sorted(result, key=lambda r: -r["weighted_in_amount"])
