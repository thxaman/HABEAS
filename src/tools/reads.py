"""Phase 2 read tools (PLAN.md §7) — plain Python over the approved catalog.

Every function takes the graph explicitly, validates its inputs, executes
only approved read-only templates via ``envelopes.run_evidence``, and returns
a :class:`ToolEnvelope` with a deterministic, non-empty interpretation.
Interpretations describe what was observed — they never assert misconduct or
legal conclusions, and structural topology is never presented as proof of
chronological fund movement.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from src.tools import query_catalog as qc
from src.tools.envelopes import (
    ROW_LIMIT_DEFAULT,
    QueryEvidence,
    ToolEnvelope,
    error_envelope,
    make_envelope,
    run_evidence,
)

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
# Cap on how many path legs get event-level corroboration per tool call.
_MAX_EVENT_LEGS = 40


def _window(graph, window_days: int) -> tuple[dict[str, Any], list[QueryEvidence]]:
    """Anchor the observation window at the graph's latest transaction ts.

    The demo data lives in a fixed synthetic period, so "recent" is measured
    from the newest event in the graph, not the wall clock.
    """
    ev = run_evidence(graph, "txn_max_ts_v1", qc.get_template("txn_max_ts_v1"), {})
    max_ts = ev.rows[0][0] if ev.rows else None
    if max_ts:
        cutoff = (
            datetime.strptime(max_ts, _TS_FMT) - timedelta(days=window_days)
        ).strftime(_TS_FMT)
    else:
        cutoff = ""  # no transactions at all: nothing can match the window
    info = {"as_of": max_ts, "window_days": window_days, "cutoff": cutoff}
    return info, [ev]


def reconstruct_path_accounts(nodes: list, rels: list) -> tuple[list[str], list[tuple[str, str]]]:
    """Rebuild the visited account sequence from Edge endpoints alone.

    List order is NOT trusted. For closed self-returning paths this server has
    been observed returning `nodes(path)` and `relationships(path)` in
    inconsistent orders (one forward, the other reversed — and which is which
    varies between queries; see docs/falkordb-capabilities.md P0-17). What IS
    faithful is each Edge's own stored endpoints.

    So we collect the directed legs, then walk them from the path's start
    (a node with no incoming leg, or any node for a closed cycle) to recover
    the true traversal order and direction.
    """
    id_by_internal = {n.id: (n.properties or {}).get("id") for n in nodes}
    directed: list[tuple[str, str]] = []
    for edge in rels:
        src = id_by_internal.get(edge.src_node)
        dst = id_by_internal.get(edge.dest_node)
        if src is None or dst is None:
            continue  # endpoint not among the returned path nodes
        directed.append((src, dst))
    if not directed:
        return [], []

    outgoing: dict[str, str] = {}
    in_degree: dict[str, int] = {}
    for src, dst in directed:
        outgoing[src] = dst
        in_degree[dst] = in_degree.get(dst, 0) + 1

    starts = [node for node in outgoing if in_degree.get(node, 0) == 0]
    start = starts[0] if len(starts) == 1 else directed[0][0]

    sequence = [start]
    legs: list[tuple[str, str]] = []
    current = start
    for _ in range(len(directed)):
        if current not in outgoing:
            break
        nxt = outgoing[current]
        legs.append((current, nxt))
        sequence.append(nxt)
        current = nxt
        if current == start:
            break
    return sequence, legs


def _leg_props(edge) -> dict[str, Any]:
    props = dict(edge.properties or {})
    return {
        "count": int(props.get("count", 0)),
        "total_amount": float(props.get("total_amount", 0.0)),
        "last_ts": props.get("last_ts"),
        "active": bool(props.get("active", True)),
    }


def _events_for_legs(
    graph, legs: list[tuple[str, str]], cutoff: str, limit: int
) -> tuple[list[dict[str, Any]], QueryEvidence]:
    params = {
        "legs": [{"src": s, "dst": d} for s, d in legs],
        "cutoff": cutoff,
        "limit": limit,
    }
    ev = run_evidence(
        graph, "txn_events_for_legs_v1",
        qc.get_template("txn_events_for_legs_v1"), params, row_cap=limit,
    )
    events = [
        {"src": r[0], "dst": r[1], "txn_id": r[2], "ts": r[3], "amount": float(r[4])}
        for r in ev.rows
    ]
    return events, ev


# ---------------------------------------------------------------- entity_profile


def entity_profile(
    graph, entity_id: str, *, window_days: int = 30, limit: int = ROW_LIMIT_DEFAULT
) -> ToolEnvelope:
    t0 = time.perf_counter()
    try:
        entity_id = qc.validate_entity_id(entity_id, "entity_id")
        window_days = qc.validate_window_days(window_days)
        limit = qc.validate_limit(limit)

        window, evidence = _window(graph, window_days)
        lookup = run_evidence(
            graph, "entity_profile_lookup_v1",
            qc.get_template("entity_profile_lookup_v1"),
            {"entity_id": entity_id},
        )
        evidence.append(lookup)

        if not lookup.rows:
            return make_envelope(
                "entity_profile", evidence,
                {"found": False, "entity_id": entity_id},
                f"No Account or Customer with id {entity_id!r} exists in the graph; "
                "nothing was profiled.",
                t0,
            )

        kind, props = lookup.rows[0][0], dict(lookup.rows[0][1] or {})
        flows = run_evidence(
            graph, "entity_profile_flows_v1",
            qc.get_template("entity_profile_flows_v1"),
            {"entity_id": entity_id, "cutoff": window["cutoff"], "limit": limit},
            row_cap=limit,
        )
        evidence.append(flows)
        incoming = [r for r in flows.rows if r[1] == entity_id]
        outgoing = [r for r in flows.rows if r[0] == entity_id]

        result: dict[str, Any] = {
            "found": True,
            "entity_id": entity_id,
            "kind": kind,
            "properties": props,
            "window": window,
            "incoming": {
                "txn_count": int(sum(r[2] for r in incoming)),
                "total_amount": round(sum(r[3] for r in incoming), 2),
                "distinct_counterparties": len({r[0] for r in incoming}),
            },
            "outgoing": {
                "txn_count": int(sum(r[2] for r in outgoing)),
                "total_amount": round(sum(r[3] for r in outgoing), 2),
                "distinct_counterparties": len({r[1] for r in outgoing}),
            },
        }

        if kind == "Account":
            owner = run_evidence(
                graph, "entity_profile_owner_v1",
                qc.get_template("entity_profile_owner_v1"),
                {"entity_id": entity_id, "limit": 5}, row_cap=5,
            )
            evidence.append(owner)
            result["owner_customers"] = [r[0] for r in owner.rows]
        else:
            owned = run_evidence(
                graph, "entity_profile_owned_accounts_v1",
                qc.get_template("entity_profile_owned_accounts_v1"),
                {"entity_id": entity_id, "limit": 10}, row_cap=10,
            )
            evidence.append(owned)
            result["owned_accounts"] = [r[0] for r in owned.rows]

        prior = run_evidence(
            graph, "prior_cases_v1", qc.get_template("prior_cases_v1"),
            {"entity_id": entity_id, "limit": limit}, row_cap=limit,
        )
        evidence.append(prior)
        result["prior_case_count"] = len(prior.rows)

        status = props.get("status", "n/a")
        interpretation = (
            f"{kind} {entity_id} (status: {status}) — in the {window_days}-day window "
            f"ending {window['as_of'] or 'no data'}, it received "
            f"{result['incoming']['txn_count']} transactions totaling "
            f"{result['incoming']['total_amount']:.2f} from "
            f"{result['incoming']['distinct_counterparties']} counterparties and sent "
            f"{result['outgoing']['txn_count']} transactions totaling "
            f"{result['outgoing']['total_amount']:.2f} to "
            f"{result['outgoing']['distinct_counterparties']} counterparties; "
            f"{result['prior_case_count']} prior cases reference this entity. "
            "Flow figures are historical transaction activity, not a determination."
        )
        if flows.truncated:
            interpretation += " Counterparty rows hit the limit; counts may be truncated."
        return make_envelope("entity_profile", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("entity_profile", exc, t0)


# ---------------------------------------------------------------- trace_funds


def trace_funds(
    graph,
    account_id: str,
    direction: str = "out",
    *,
    hops: int = 3,
    min_amount: float = 0,
    window_days: int = 30,
    limit: int = ROW_LIMIT_DEFAULT,
) -> ToolEnvelope:
    t0 = time.perf_counter()
    try:
        account_id = qc.validate_entity_id(account_id, "account_id")
        direction = qc.validate_direction(direction)
        hops = qc.validate_hop_count(hops)
        min_amount = qc.validate_amount(min_amount)
        window_days = qc.validate_window_days(window_days)
        limit = qc.validate_limit(limit)

        window, evidence = _window(graph, window_days)
        query_id = "trace_paths_out_v1" if direction == "out" else "trace_paths_in_v1"
        cypher = qc.bind_hop_bound(query_id, "<<hops>>", hops)
        paths_ev = run_evidence(
            graph, query_id, cypher,
            {
                "account_id": account_id, "min_amount": min_amount,
                "cutoff": window["cutoff"], "limit": limit,
            },
            interpolated=["hops"], bound={"<<hops>>": hops}, row_cap=limit,
        )
        evidence.append(paths_ev)

        paths: list[dict[str, Any]] = []
        leg_set: list[tuple[str, str]] = []
        for row in paths_ev.rows:
            rels, nodes = row[0], row[1]
            raw_rels = [_edge_from_normalized(r) for r in rels]
            raw_nodes = [_node_from_normalized(n) for n in nodes]
            sequence, legs = reconstruct_path_accounts(raw_nodes, raw_rels)
            paths.append({
                "accounts": sequence,
                "hop_count": len(legs),
                "legs": [
                    {"src": s, "dst": d, **_leg_props(e)}
                    for (s, d), e in zip(legs, raw_rels)
                ],
            })
            for leg in legs:
                if leg not in leg_set:
                    leg_set.append(leg)

        leg_events: list[dict[str, Any]] = []
        events_note = ""
        if leg_set:
            corroboration_legs = leg_set[:_MAX_EVENT_LEGS]
            if len(leg_set) > _MAX_EVENT_LEGS:
                events_note = (
                    f" Event-level corroboration capped at the first "
                    f"{_MAX_EVENT_LEGS} distinct legs."
                )
            leg_events, events_ev = _events_for_legs(
                graph, corroboration_legs, window["cutoff"], limit
            )
            evidence.append(events_ev)

        result = {
            "account_id": account_id,
            "direction": direction,
            "hops": hops,
            "min_amount": min_amount,
            "window": window,
            "path_count": len(paths),
            "paths": paths,
            "leg_events": leg_events,
            "truncated": paths_ev.truncated,
        }
        farthest = max((p["hop_count"] for p in paths), default=0)
        interpretation = (
            f"Traced {direction}-going active transfer topology from {account_id} "
            f"over at most {hops} hops (edges with total >= {min_amount:.2f} and "
            f"activity in the {window_days}-day window ending "
            f"{window['as_of'] or 'no data'}): {len(paths)} bounded paths, reaching "
            f"at most {farthest} hops away. Paths follow the derived "
            f"TRANSFERRED_TO projection; they show structural connectivity, not "
            f"chronological movement of one sum of money.{events_note}"
        )
        if paths_ev.truncated:
            interpretation += " Path rows hit the limit; this is not an exhaustive set."
        return make_envelope("trace_funds", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("trace_funds", exc, t0)


class _NormalizedEdge:
    """Driver-shaped shim so reconstruction works on normalized rows too."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.id = data.get("id")
        self.src_node = data.get("src")
        self.dest_node = data.get("dst")
        self.relation = data.get("relation")
        self.properties = data.get("properties") or {}


class _NormalizedNode:
    def __init__(self, data: dict[str, Any]) -> None:
        self.id = data.get("id")
        self.labels = data.get("labels") or []
        self.properties = data.get("properties") or {}


def _edge_from_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return _NormalizedEdge(value)
    return value  # already a driver Edge


def _node_from_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return _NormalizedNode(value)
    return value  # already a driver Node


# ---------------------------------------------------------------- alert_context


def alert_context(graph, alert_id: str) -> ToolEnvelope:
    """Resolve an Alert id into its trigger transaction and involved accounts.

    Added in Phase 4 as a narrowly scoped prerequisite: `entity_profile`
    handles Account/Customer ids only, so the investigation agent needs an
    evidence-based way to turn an alert into its target entities. It returns
    only graph data — never the ground-truth manifest.
    """
    t0 = time.perf_counter()
    try:
        alert_id = qc.validate_entity_id(alert_id, "alert_id")
        ev = run_evidence(
            graph, "alert_context_v1", qc.get_template("alert_context_v1"),
            {"alert_id": alert_id},
        )
        if not ev.rows:
            return make_envelope(
                "alert_context", [ev],
                {"found": False, "alert_id": alert_id, "entity_ids": []},
                f"No Alert with id {alert_id!r} exists in the graph; the alert "
                "could not be resolved to any entity.",
                t0,
            )
        row = dict(zip(ev.columns, ev.rows[0]))
        entity_ids = [
            value for value in (
                row.get("concerned_account_id"),
                row.get("sender_account_id"),
                row.get("receiver_account_id"),
            ) if value
        ]
        # preserve order, drop duplicates
        entity_ids = list(dict.fromkeys(entity_ids))
        result = {
            "found": True,
            "alert_id": row["alert_id"],
            "reason": row.get("reason"),
            "created_at": row.get("created_at"),
            "triage_status": row.get("triage_status"),
            "trigger_transaction": {
                "txn_id": row.get("txn_id"),
                "ts": row.get("ts"),
                "amount": row.get("amount"),
            },
            "sender_account_id": row.get("sender_account_id"),
            "receiver_account_id": row.get("receiver_account_id"),
            "concerned_account_id": row.get("concerned_account_id"),
            "entity_ids": entity_ids,
        }
        interpretation = (
            f"Alert {row['alert_id']} (reason: {row.get('reason') or 'unspecified'}, "
            f"triage: {row.get('triage_status') or 'unknown'}) was triggered by "
            f"transaction {row.get('txn_id')} of {row.get('amount')} at "
            f"{row.get('ts')}, moving funds from {row.get('sender_account_id')} to "
            f"{row.get('receiver_account_id')}; the alert is linked to account(s) "
            f"{', '.join(entity_ids) or 'none'}. An alert is a trigger for review, "
            "not a finding of wrongdoing."
        )
        return make_envelope("alert_context", [ev], result, interpretation, t0)
    except Exception as exc:
        return error_envelope("alert_context", exc, t0)


# ------------------------------------------------------------ shared_identifiers


def shared_identifiers(
    graph,
    customer_a_id: str,
    customer_b_id: str,
    *,
    limit: int = ROW_LIMIT_DEFAULT,
) -> ToolEnvelope:
    t0 = time.perf_counter()
    try:
        customer_a_id = qc.validate_entity_id(customer_a_id, "customer_a_id")
        customer_b_id = qc.validate_entity_id(customer_b_id, "customer_b_id")
        limit = qc.validate_limit(limit)

        devices = run_evidence(
            graph, "shared_devices_v1", qc.get_template("shared_devices_v1"),
            {"customer_a_id": customer_a_id, "customer_b_id": customer_b_id,
             "limit": limit},
            row_cap=limit,
        )
        idents = run_evidence(
            graph, "shared_identifiers_v1", qc.get_template("shared_identifiers_v1"),
            {"customer_a_id": customer_a_id, "customer_b_id": customer_b_id,
             "limit": limit},
            row_cap=limit,
        )
        evidence = [devices, idents]
        shared_devices = [
            {"device_id": r[0], "first_seen": r[1], "last_seen": r[2]}
            for r in devices.rows
        ]
        shared_idents = [
            {"identifier_id": r[0], "kind": r[1]} for r in idents.rows
        ]
        kinds = sorted({i["kind"] for i in shared_idents})
        result = {
            "customer_a_id": customer_a_id,
            "customer_b_id": customer_b_id,
            "shared_devices": shared_devices,
            "shared_identifiers": shared_idents,
            "shared_identifier_kinds": kinds,
        }
        interpretation = (
            f"Customers {customer_a_id} and {customer_b_id} share "
            f"{len(shared_devices)} device(s) and {len(shared_idents)} "
            f"identifier(s)"
            + (f" of kind(s) {', '.join(kinds)}" if kinds else "")
            + ". Shared infrastructure is an association signal only; on its own it "
            "is not evidence of fraud."
        )
        return make_envelope("shared_identifiers", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("shared_identifiers", exc, t0)


# ---------------------------------------------------------------- fan_in_out


def fan_in_out(
    graph,
    account_id: str,
    *,
    window_days: int = 30,
    min_senders: int = 3,
    limit: int = ROW_LIMIT_DEFAULT,
) -> ToolEnvelope:
    t0 = time.perf_counter()
    try:
        account_id = qc.validate_entity_id(account_id, "account_id")
        window_days = qc.validate_window_days(window_days)
        min_senders = qc.validate_min_senders(min_senders)
        limit = qc.validate_limit(limit)

        window, evidence = _window(graph, window_days)
        senders = run_evidence(
            graph, "fan_in_senders_v1", qc.get_template("fan_in_senders_v1"),
            {"account_id": account_id, "cutoff": window["cutoff"], "limit": limit},
            row_cap=limit,
        )
        receivers = run_evidence(
            graph, "fan_out_receivers_v1", qc.get_template("fan_out_receivers_v1"),
            {"account_id": account_id, "cutoff": window["cutoff"], "limit": limit},
            row_cap=limit,
        )
        evidence += [senders, receivers]

        in_amount = round(sum(r[2] for r in senders.rows), 2)
        out_amount = round(sum(r[2] for r in receivers.rows), 2)
        top_receiver = receivers.rows[0][0] if receivers.rows else None
        top_share = (
            round(receivers.rows[0][2] / out_amount, 4)
            if top_receiver and out_amount > 0 else None
        )
        result = {
            "account_id": account_id,
            "window": window,
            "distinct_senders": len(senders.rows),
            "distinct_receivers": len(receivers.rows),
            "in_txn_count": int(sum(r[1] for r in senders.rows)),
            "in_amount": in_amount,
            "out_txn_count": int(sum(r[1] for r in receivers.rows)),
            "out_amount": out_amount,
            "out_in_ratio": round(out_amount / in_amount, 4) if in_amount > 0 else None,
            "top_receiver": top_receiver,
            "top_receiver_share": top_share,
            "min_senders_threshold": min_senders,
            "meets_min_senders": len(senders.rows) >= min_senders,
            "truncated": senders.truncated or receivers.truncated,
        }
        ratio_text = (
            f"{result['out_in_ratio']}" if result["out_in_ratio"] is not None
            else "undefined (no incoming amount in window)"
        )
        interpretation = (
            f"In the {window_days}-day window ending {window['as_of'] or 'no data'}, "
            f"account {account_id} received from {result['distinct_senders']} distinct "
            f"sender(s) (incoming {in_amount:.2f}) and sent to "
            f"{result['distinct_receivers']} distinct receiver(s) (outgoing "
            f"{out_amount:.2f}); out/in ratio {ratio_text}"
            + (
                f"; largest single receiver {top_receiver} takes "
                f"{top_share:.0%} of outgoing volume"
                if top_share is not None else ""
            )
            + f". Sender count {'meets' if result['meets_min_senders'] else 'is below'} "
            f"the min_senders={min_senders} threshold. These are observed flows "
            "only; coordinated fan-in/fan-out would still need corroborating "
            "signals before any conclusion."
        )
        if result["truncated"]:
            interpretation += " Rows hit the limit; counts may be truncated."
        return make_envelope("fan_in_out", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("fan_in_out", exc, t0)


# ---------------------------------------------------------------- prior_cases


def prior_cases(
    graph, entity_id: str, *, limit: int = ROW_LIMIT_DEFAULT
) -> ToolEnvelope:
    t0 = time.perf_counter()
    try:
        entity_id = qc.validate_entity_id(entity_id, "entity_id")
        limit = qc.validate_limit(limit)
        ev = run_evidence(
            graph, "prior_cases_v1", qc.get_template("prior_cases_v1"),
            {"entity_id": entity_id, "limit": limit}, row_cap=limit,
        )
        cases = [dict(zip(["case_id", "status", "verdict", "severity", "updated_at"],
                          [r[0], r[1], r[2], r[3], r[4]])) for r in ev.rows]
        result = {"entity_id": entity_id, "case_count": len(cases), "cases": cases}
        if cases:
            listing = ", ".join(
                f"{c['case_id']} ({c['status'] or 'unknown status'})" for c in cases
            )
            interpretation = (
                f"{len(cases)} prior case(s) reference entity {entity_id}: {listing}. "
                "Prior verdicts are leads to re-check, not settled truth."
            )
        else:
            interpretation = (
                f"No prior cases reference entity {entity_id}; this entity has no "
                "recorded investigation history to reuse."
            )
        return make_envelope("prior_cases", [ev], result, interpretation, t0)
    except Exception as exc:
        return error_envelope("prior_cases", exc, t0)


# ----------------------------------------------------------- active_path_exists


def active_path_exists(
    graph,
    source_account_id: str,
    target_account_id: str,
    *,
    max_hops: int = 5,
) -> ToolEnvelope:
    t0 = time.perf_counter()
    try:
        source = qc.validate_entity_id(source_account_id, "source_account_id")
        target = qc.validate_entity_id(target_account_id, "target_account_id")
        max_hops = qc.validate_hop_count(max_hops, "max_hops")
        cypher = qc.bind_hop_bound("active_path_min_hops_v1", "<<max_hops>>", max_hops)
        ev = run_evidence(
            graph, "active_path_min_hops_v1", cypher,
            {"source_account_id": source, "target_account_id": target},
            interpolated=["max_hops"], bound={"<<max_hops>>": max_hops},
        )
        exists = bool(ev.rows)
        hops = int(ev.rows[0][0]) if exists else None
        result = {
            "source_account_id": source,
            "target_account_id": target,
            "max_hops": max_hops,
            "exists": exists,
            "shortest_hops": hops,
        }
        if exists:
            interpretation = (
                f"An active transfer route exists from {source} to {target}: the "
                f"shortest bounded route within {max_hops} hops uses {hops} hop(s) "
                "over active TRANSFERRED_TO edges and non-frozen accounts. This is "
                "current routing state, not a transaction history."
            )
        else:
            interpretation = (
                f"No active transfer route from {source} to {target} exists within "
                f"{max_hops} hop(s) over active edges and non-frozen accounts. "
                "Historical transactions may still exist between them."
            )
        return make_envelope("active_path_exists", [ev], result, interpretation, t0)
    except Exception as exc:
        return error_envelope("active_path_exists", exc, t0)


# ------------------------------------------------------- UI: alert inbox


def alert_inbox(graph, *, limit: int = ROW_LIMIT_DEFAULT) -> ToolEnvelope:
    """Bounded, read-only listing of every alert with its investigation state.

    Phase 8.5: the analyst UI loads its inbox from here instead of a
    hard-coded list. Each row carries the alert's concerned account, trigger
    transaction, and — where an earlier investigation already ran on this
    graph — the persisted case verdict and latest action status.
    """
    t0 = time.perf_counter()
    try:
        limit = qc.validate_limit(limit)
        ev = run_evidence(
            graph, "alert_inbox_v1", qc.get_template("alert_inbox_v1"),
            {"limit": limit}, row_cap=limit,
        )
        rows = [dict(zip(ev.columns, r)) for r in ev.rows]
        alerts = [
            {
                "alert_id": row.get("alert_id"),
                "reason": row.get("reason"),
                "created_at": row.get("created_at"),
                "triage_status": row.get("triage_status"),
                "concerned_account_id": row.get("concerned_account_id"),
                "trigger_transaction": {
                    "txn_id": row.get("txn_id"),
                    "ts": row.get("ts"),
                    "amount": row.get("amount"),
                },
                "case_id": row.get("case_id"),
                "case_verdict": row.get("case_verdict"),
                "case_status": row.get("case_status"),
                "action_id": row.get("action_id"),
                "action_status": row.get("action_status"),
            }
            for row in rows
        ]
        investigated = sum(1 for a in alerts if a["case_id"])
        frozen = sum(1 for a in alerts if a["action_status"] == "applied")
        result = {
            "alerts": alerts,
            "alert_count": len(alerts),
            "already_investigated": investigated,
            "actions_applied": frozen,
        }
        interpretation = (
            f"{len(alerts)} alert(s) in the graph; {investigated} already have a "
            f"persisted case and {frozen} have an applied action. Inbox rows are "
            "graph records — an alert is a trigger for review, not a finding."
        )
        return make_envelope("alert_inbox", [ev], result, interpretation, t0)
    except Exception as exc:
        return error_envelope("alert_inbox", exc, t0)


# ------------------------------------------------- UI: neighbourhood subgraph


def alert_neighborhood(
    graph,
    account_id: str,
    *,
    hops: int = 2,
    window_days: int = 365,
    limit: int = ROW_LIMIT_DEFAULT,
) -> ToolEnvelope:
    """Bounded read-only neighbourhood of one account for graph rendering.

    Returns the focal account, the accounts reachable in either direction
    within ``hops`` (1..5, a validated integer), and every TRANSFERRED_TO edge
    incident to that set — with its aggregate count/volume, last timestamp and
    the ``active`` flag, so the UI can show active versus deactivated routing.
    Nothing is invented here: the UI draws exactly these rows.
    """
    t0 = time.perf_counter()
    try:
        account_id = qc.validate_entity_id(account_id, "account_id")
        hops = qc.validate_hop_count(hops, "hops")
        limit = qc.validate_limit(limit)

        out_ev = run_evidence(
            graph, "neighborhood_out_accounts_v1",
            qc.bind_hop_bound("neighborhood_out_accounts_v1", "<<hops>>", hops),
            {"account_id": account_id, "limit": limit},
            interpolated=["hops"], bound={"<<hops>>": hops}, row_cap=limit,
        )
        in_ev = run_evidence(
            graph, "neighborhood_in_accounts_v1",
            qc.bind_hop_bound("neighborhood_in_accounts_v1", "<<hops>>", hops),
            {"account_id": account_id, "limit": limit},
            interpolated=["hops"], bound={"<<hops>>": hops}, row_cap=limit,
        )
        ids = [account_id]
        for ev in (out_ev, in_ev):
            for row in ev.rows:
                value = row[0]
                if value and value not in ids:
                    ids.append(value)

        edge_ev = run_evidence(
            graph, "neighborhood_edges_v1",
            qc.get_template("neighborhood_edges_v1"),
            {"account_ids": ids, "limit": limit}, row_cap=limit,
        )
        edges = [
            {
                "src": r[0], "dst": r[1], "count": int(r[2] or 0),
                "total_amount": float(r[3] or 0.0), "last_ts": r[4],
                "active": bool(r[5]) if r[5] is not None else None,
            }
            for r in edge_ev.rows
        ]
        # Underlying transaction events for the busiest legs (bounded), so the
        # UI can show real transfer evidence the moment an alert is selected —
        # still read-only and still from an approved template.
        legs = [(e["src"], e["dst"]) for e in edges][:_MAX_EVENT_LEGS]
        window, window_evs = _window(graph, window_days)
        events: list[dict[str, Any]] = []
        event_evs: list[QueryEvidence] = []
        if legs:
            event_limit = min(limit * 2, 500)
            events, ev_ev = _events_for_legs(
                graph, legs, window["cutoff"], event_limit
            )
            event_evs.append(ev_ev)

        truncated = bool(
            out_ev.truncated or in_ev.truncated or edge_ev.truncated
            or any(ev.truncated for ev in event_evs)
        )
        result = {
            "focal_account_id": account_id,
            "hops": hops,
            "accounts": ids,
            "account_count": len(ids),
            "edges": edges,
            "edge_count": len(edges),
            "transactions": events,
            "transaction_count": len(events),
            "window": window,
            "truncated": truncated,
        }
        interpretation = (
            f"Bounded {hops}-hop neighbourhood of {account_id}: {len(ids)} "
            f"account(s) and {len(edges)} TRANSFERRED_TO edge(s); "
            f"{sum(1 for e in edges if e['active'])} active. Structural "
            "connectivity only — edge aggregates are not chronological flows."
        )
        if truncated:
            interpretation += (
                " One or more queries hit their row cap, so the neighbourhood "
                "shown may be incomplete."
            )
        return make_envelope(
            "alert_neighborhood",
            [out_ev, in_ev, edge_ev, *window_evs, *event_evs],
            result, interpretation, t0,
        )
    except Exception as exc:
        return error_envelope("alert_neighborhood", exc, t0)
