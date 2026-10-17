"""Pure presentation helpers for the HABEAS UI (Phase 8.5).

No Streamlit import, no I/O, no re-derivation of backend conclusions: these
functions only *format* values that a real result object already produced.
The two substantive transformations are:

- ``graph_from_evidence`` — builds the account-graph view strictly from the
  evidence the investigation recorded (cycle legs/events from ``detect_cycle``,
  leg events from ``trace_funds``). No synthetic nodes or edges are ever
  invented, structural connectivity stays visually distinct from chronological
  circulation, and truncation is carried through as a first-class flag;
- ``to_dot`` — bounded DOT rendering (node/edge caps, honest cap notice) for
  ``st.graphviz_chart``, which renders it client-side without a new dependency.
"""
from __future__ import annotations

import csv
import html
import io
import re
import time
from typing import Any

from src.agent.policy import (
    VERDICT_INCOMPLETE_INVESTIGATION,
    VERDICT_INSUFFICIENT_EVIDENCE,
    VERDICT_LIKELY_BENIGN,
    VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
    VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION,
)

# Tone vocabulary shared with app.py's CSS (and asserted by tests): every
# status maps to exactly one tone so supported / contradicted / open /
# inconclusive stay visually consistent everywhere.
TONES = ("positive", "watch", "ok", "abstain", "error", "muted")

_VERDICT_CHIPS = {
    VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW: ("Supported for review", "positive"),
    VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION: ("Suspicious — unconfirmed", "watch"),
    VERDICT_LIKELY_BENIGN: ("Likely benign", "ok"),
    VERDICT_INSUFFICIENT_EVIDENCE: ("Insufficient evidence", "abstain"),
    VERDICT_INCOMPLETE_INVESTIGATION: ("Incomplete investigation", "abstain"),
}

_ACTION_CHIPS = {
    "applied": ("Freeze applied (sandbox)", "positive"),
    "declined": ("Declined by policy", "abstain"),
    "already_applied": ("Already applied earlier", "muted"),
    "in_progress": ("In progress elsewhere", "watch"),
}

_HYPOTHESIS_CHIPS = {
    "supported": ("Supported", "positive"),
    "contradicted": ("Contradicted", "error"),
    "weakened": ("Weakened", "watch"),
    "open": ("Open / untested", "muted"),
    "insufficient": ("Insufficient evidence", "abstain"),
}

_STANCE_CHIPS = {
    "supporting": ("Supports", "positive"),
    "contradicting": ("Contradicts", "error"),
    "unresolved": ("Unresolved", "muted"),
}

# Verification status → (short label, tone). Mismatch is the only failure.
_VERIFICATION_CHIPS = {
    "reproduced_current_state": ("Reproduced", "positive"),
    "verified_against_recorded_evidence": ("Verified vs record", "positive"),
    "changed_after_action": ("Changed after action", "watch"),
    "mismatch": ("Mismatch", "error"),
    "unverifiable": ("Unverifiable", "abstain"),
    "skipped": ("Skipped", "muted"),
}


def verdict_chip(verdict: str | None) -> tuple[str, str]:
    if verdict in _VERDICT_CHIPS:
        return _VERDICT_CHIPS[verdict]
    return ("No verdict", "abstain")


def action_chip(action_status: str | None) -> tuple[str, str]:
    if action_status in _ACTION_CHIPS:
        return _ACTION_CHIPS[action_status]
    if not action_status:
        return ("No action attempted", "muted")
    return (str(action_status), "watch")


def hypothesis_chip(status: str) -> tuple[str, str]:
    return _HYPOTHESIS_CHIPS.get(status, (status or "unknown", "muted"))


def stance_chip(stance: str) -> tuple[str, str]:
    return _STANCE_CHIPS.get(stance, (stance or "unknown", "muted"))


def verification_chip(status: str) -> tuple[str, str]:
    return _VERIFICATION_CHIPS.get(status, (status or "unknown", "watch"))


def run_status_chip(snapshot: dict[str, Any]) -> tuple[str, str]:
    """Top-level run state: green only for a real, completed investigation."""
    if snapshot.get("status") == "configuration_error":
        return ("Provider not configured", "error")
    if snapshot.get("status") == "failed":
        return ("Investigation failed", "error")
    if snapshot.get("status") != "ok":
        return ("No investigation run yet", "muted")
    if not snapshot.get("completed"):
        return ("Investigation incomplete", "watch")
    return ("Investigation completed", "positive")


def verification_counts(summary: dict[str, Any] | None) -> dict[str, int]:
    """Normalized claim-verification counts (missing statuses count as 0)."""
    summary = summary or {}
    return {
        "reproduced": int(summary.get("reproduced_current_state", 0) or 0)
        + int(summary.get("verified_against_recorded_evidence", 0) or 0),
        "changed_after_action": int(summary.get("changed_after_action", 0) or 0),
        "mismatch": int(summary.get("mismatch", 0) or 0),
        "unverifiable": int(summary.get("unverifiable", 0) or 0),
    }


_VERIFICATION_STATUS_ORDER = (
    "reproduced_current_state",
    "verified_against_recorded_evidence",
    "changed_after_action",
    "mismatch",
    "unverifiable",
    "skipped",
)


def verification_status_rows(
    summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Per-status verification counts from a report summary, in fixed order.

    Only the statuses the verifier actually recorded are returned. Unknown
    extra summary keys are ignored here rather than being mislabelled.
    """
    summary = summary or {}
    rows: list[dict[str, Any]] = []
    for status in _VERIFICATION_STATUS_ORDER:
        if status in summary:
            rows.append({"status": status, "count": int(summary.get(status) or 0)})
    return rows


# ------------------------------------------------------- connection status


def connection_chip(status: dict[str, Any] | None) -> tuple[str, str]:
    """Visible graph-connection state, derived from a recorded read result.

    Three states only: ``connected`` (a recent approved read succeeded),
    ``unavailable`` (the read failed) and ``not_checked`` (no check has been
    recorded). There is no constant "connected" label in the UI.
    """
    state = (status or {}).get("state")
    if state == "connected":
        return ("FalkorDB connected", "ok")
    if state == "unavailable":
        return ("FalkorDB unavailable", "error")
    return ("FalkorDB not checked", "muted")


#: Age after which a successful check is labelled stale (still shown as the
#: last known state, never silently refreshed or presented as current).
CONNECTION_STALE_AFTER_S = 300


def connection_age(
    status: dict[str, Any] | None, now: float | None = None
) -> str | None:
    """Human age of the last connection check, or ``None`` when never checked."""
    checked_at = (status or {}).get("checked_at")
    if not checked_at:
        return None
    try:
        seconds = max(0, int((now if now is not None else time.time())
                             - float(checked_at)))
    except (TypeError, ValueError):
        return None
    if seconds < 60:
        return f"{seconds}s ago"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s ago" if sec else f"{minutes}m ago"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m ago" if minutes else f"{hours}h ago"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h ago" if hours else f"{days}d ago"


def connection_is_stale(
    status: dict[str, Any] | None,
    now: float | None = None,
    *,
    max_age_s: int = CONNECTION_STALE_AFTER_S,
) -> bool:
    """True when the last check is older than the documented stale threshold."""
    checked_at = (status or {}).get("checked_at")
    if not checked_at:
        return False
    try:
        return ((now if now is not None else time.time()) - float(checked_at)) > max_age_s
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------- graph view


MAX_GRAPH_NODES = 40
MAX_GRAPH_EDGES = 80
MAX_TXN_PER_EDGE = 2


def _add_edge(
    edges: dict[tuple[str, str], dict[str, Any]],
    src: str,
    dst: str,
    *,
    kind: str,
    corroborated: bool | None = None,
    txn_id: str | None = None,
    ts: str | None = None,
) -> None:
    key = (src, dst)
    edge = edges.setdefault(
        key,
        {
            "src": src, "dst": dst, "kind": kind,
            "corroborated": corroborated, "txn_ids": [], "timestamps": [],
        },
    )
    if kind == "cycle":
        # Corroboration is the stronger recorded fact and is sticky: if any
        # recorded cycle leg/event for this pair was chronologically
        # corroborated, the edge is corroborated; otherwise it stays
        # structural-only.
        if corroborated:
            edge["corroborated"] = True
        elif edge["corroborated"] is None:
            edge["corroborated"] = False
    if txn_id and txn_id not in edge["txn_ids"] and len(edge["txn_ids"]) < 6:
        edge["txn_ids"].append(txn_id)
    if ts and ts not in edge["timestamps"]:
        edge["timestamps"].append(ts)


def graph_from_evidence(
    evidence_entries: list[dict[str, Any]],
    primary_entities: list[str] | None = None,
) -> dict[str, Any]:
    """Build the graph view from recorded evidence only.

    Nodes/edges come from ``detect_cycle`` derived cycles (legs + timestamped
    events, with the cycle's corroboration flag) and ``trace_funds`` derived
    leg events. Structural-only cycle legs are kept but flagged, so structure
    is never presented as chronological circulation.
    """
    primary = {e for e in (primary_entities or []) if e}
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    truncated = False
    notes: list[str] = []

    def touch(account: str, kind: str = "account") -> None:
        if not account:
            return
        node = nodes.setdefault(account, {"id": account, "kind": "account"})
        if account in primary:
            node["kind"] = "primary"

    for entry in evidence_entries or []:
        derived = entry.get("derived") or {}
        if any(entry.get("truncated") or []):
            truncated = True
        if derived.get("truncated"):
            truncated = True

        for cycle in derived.get("cycles") or []:
            corroborated = bool(cycle.get("temporally_corroborated"))
            for leg in cycle.get("legs") or []:
                src, dst = leg.get("src"), leg.get("dst")
                touch(src), touch(dst)
                _add_edge(edges, src, dst, kind="cycle", corroborated=corroborated)
            for evt in cycle.get("events") or []:
                touch(evt.get("src")), touch(evt.get("dst"))
                _add_edge(
                    edges, evt.get("src"), evt.get("dst"), kind="cycle",
                    corroborated=corroborated,
                    txn_id=evt.get("txn_id"), ts=evt.get("ts"),
                )

        for evt in derived.get("leg_events") or []:
            touch(evt.get("src")), touch(evt.get("dst"))
            _add_edge(
                edges, evt.get("src"), evt.get("dst"), kind="traced",
                txn_id=evt.get("txn_id"), ts=evt.get("ts"),
            )

    edge_list = sorted(edges.values(), key=lambda e: (e["src"], e["dst"]))
    capped = len(edge_list) > MAX_GRAPH_EDGES or len(nodes) > MAX_GRAPH_NODES
    if capped:
        notes.append(
            f"graph view capped at {MAX_GRAPH_NODES} nodes / {MAX_GRAPH_EDGES} "
            "edges (bounded rendering; the evidence ledger is authoritative)"
        )
        edge_list = edge_list[:MAX_GRAPH_EDGES]
        kept = {e["src"] for e in edge_list} | {e["dst"] for e in edge_list}
        nodes = {k: v for k, v in nodes.items() if k in kept}
    structural_only = [e for e in edge_list if e["kind"] == "cycle" and not e["corroborated"]]
    if structural_only:
        notes.append(
            f"{len(structural_only)} cycle leg(s) shown dashed: structural "
            "connectivity only, NOT chronological circulation"
        )
    if truncated:
        notes.append(
            "some underlying query results were truncated; the graph may be "
            "incomplete (see the evidence inspector)"
        )
    return {
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "edges": edge_list,
        "truncated": truncated,
        "capped": capped,
        "notes": notes,
        "has_content": bool(edge_list),
    }


_DOT_COLORS = {
    "bg": "#0f1b2d", "node": "#1c2b45", "node_border": "#3f5f8a",
    "primary": "#14b8a6", "text": "#dbe7f4",
    "cycle_ok": "#14b8a6",       # teal: chronologically corroborated leg
    "cycle_struct": "#64748b",   # slate dashed: structure only
    "traced": "#38bdf8",         # cyan dotted: traced transfer leg
}


def to_dot(graph: dict[str, Any]) -> str:
    """Render the bounded graph view as a DOT string (dark theme)."""
    c = _DOT_COLORS
    lines = [
        "digraph HABEAS {",
        '  bgcolor="transparent";',
        "  rankdir=LR;",
        f'  node [shape=box style="rounded,filled" fillcolor="{c["node"]}" '
        f'color="{c["node_border"]}" fontcolor="{c["text"]}" '
        'fontsize=11 margin="0.12,0.08"];',
        f'  edge [fontcolor="{c["text"]}" fontsize=9];',
    ]
    for node in graph.get("nodes") or []:
        nid = node["id"]
        if node.get("kind") == "primary":
            lines.append(
                f'  "{nid}" [color="{c["primary"]}" penwidth=2 '
                f'fontcolor="{c["primary"]}"];'
            )
        else:
            lines.append(f'  "{nid}";')
    for edge in graph.get("edges") or []:
        attrs: list[str] = []
        if edge["kind"] == "cycle":
            if edge.get("corroborated"):
                attrs += [f'color="{c["cycle_ok"]}"', 'penwidth=2',
                          'label="circulated"']
            else:
                attrs += [
                    f'color="{c["cycle_struct"]}"', 'style=dashed',
                    'label="structure only"',
                ]
        else:
            txns = edge.get("txn_ids") or []
            stamps = edge.get("timestamps") or []
            label_parts = []
            if txns:
                shown = ", ".join(txns[:MAX_TXN_PER_EDGE])
                more = len(txns) - MAX_TXN_PER_EDGE
                label_parts.append(shown + (f" +{more}" if more > 0 else ""))
            if stamps:
                label_parts.append(str(stamps[0]))
            attrs += [
                f'color="{c["traced"]}"', 'style=dotted',
                f'label="{ " · ".join(label_parts) }"' if label_parts else "",
            ]
        attr_text = " ".join(a for a in attrs if a)
        lines.append(
            f'  "{edge["src"]}" -> "{edge["dst"]}"'
            + (f" [{attr_text}]" if attr_text else "") + ";"
        )
    lines.append("}")
    return "\n".join(lines)


# ------------------------------------------------------- evidence inspector


# ------------------------------------------------- evidence <-> verification

# Report/audit ids are ``<case_id>:<local_id>`` and gain a ``-v2``/``-v3``
# suffix when a citation is re-recorded; investigation ids are the bare
# ``<local_id>``. Phase 8.6 matches on the normalized local id ONLY — the
# original ids stay untouched in the evidence records, the report and the
# audit trail (this function is a UI-side lookup key, nothing else).
_CASE_PREFIX = ":"
_RECITATION_SUFFIX = re.compile(r"-v\d+$")


def normalize_evidence_id(value: Any) -> str:
    """UI-side matching key for an evidence id (never stored, never displayed)."""
    if value is None:
        return ""
    text = str(value).strip()
    if _CASE_PREFIX in text:
        text = text.rsplit(_CASE_PREFIX, 1)[1]
    return _RECITATION_SUFFIX.sub("", text)


# Most severe first: when several assessments map to one evidence record we
# surface the worst outcome rather than the friendliest one, and say so.
_VERIFICATION_SEVERITY = (
    "mismatch",
    "unverifiable",
    "unknown",
    "changed_after_action",
    "verified_against_recorded_evidence",
    "reproduced_current_state",
)
NOT_CHECKED = "not_checked"


def _severity_rank(status: str) -> int:
    try:
        return _VERIFICATION_SEVERITY.index(status)
    except ValueError:
        return len(_VERIFICATION_SEVERITY)  # unknown statuses sort last


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text and text not in out:
            out.append(text)
    return out


def _match_claims(
    verification_claims: list[dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Group verification assessments by normalized evidence id."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for claim in verification_claims or []:
        key = claim.get("local_evidence_id") or normalize_evidence_id(
            claim.get("evidence_id")
        )
        if not key:
            continue
        grouped.setdefault(key, []).append(claim)
    return grouped


def evidence_inspector_rows(
    evidence_entries: list[dict[str, Any]],
    verification_claims: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One row per evidence record: queries, stance, and verification outcome.

    The verification column is resolved by normalized id. An evidence record
    with no matching assessment stays ``not_checked`` — it is never silently
    reported as verified. When several assessments match one record (e.g. a
    re-citation), the most severe outcome is surfaced with its own reason and
    every original verification id is retained.
    """
    grouped = _match_claims(verification_claims)
    rows: list[dict[str, Any]] = []
    for entry in evidence_entries or []:
        query_ids = list(entry.get("query_ids") or [])
        row_counts = list(entry.get("row_count") or [])
        key = normalize_evidence_id(entry.get("evidence_id"))
        matches = grouped.get(key) or []

        if not matches:
            checked = False
            status = NOT_CHECKED
            details = (
                "no verification assessment recorded for this evidence id "
                f"(looked up as {key or 'n/a'})"
            )
            verification_ids: list[str] = []
            checked_sources: list[str] = []
            mismatched_sources: list[str] = []
        else:
            checked = True
            ranked = sorted(
                matches,
                key=lambda c: _severity_rank(str(c.get("status") or "unknown")),
            )
            chosen = ranked[0]
            status = str(chosen.get("status") or "unknown")
            details = str(chosen.get("reason") or "")
            if len(matches) > 1:
                extra = ", ".join(sorted({
                    str(c.get("status") or "unknown") for c in matches
                }))
                suffix = (
                    f"{len(matches)} assessments matched this record "
                    f"({extra}); showing the most severe"
                )
                details = f"{details} · {suffix}" if details else suffix
            verification_ids = _dedupe([c.get("evidence_id") for c in matches])
            checked_sources = _dedupe(
                [q for c in matches for q in (c.get("checked_source_queries") or [])]
            )
            mismatched_sources = _dedupe(
                [q for c in matches for q in (c.get("mismatched_source_queries") or [])]
            )

        rows.append(
            {
                "evidence_id": entry.get("evidence_id"),
                "tool": entry.get("tool"),
                "ok": bool(entry.get("ok")),
                "stance": entry.get("stance"),
                "query_ids": query_ids,
                "query_count": len(query_ids),
                "multi_query": len(query_ids) > 1,
                "total_rows": sum(int(r or 0) for r in row_counts),
                "truncated_any": any(entry.get("truncated") or []),
                # `checked` means an assessment EXISTS / was attempted — it is
                # NOT a statement that the evidence proved true.
                "checked": checked,
                "verification_status": status,
                "verification_reason": (matches[0].get("reason") or "") if matches else "",
                "verification_details": details,
                "verification_ids": verification_ids,
                "verification_count": len(matches),
                "checked_source_queries": checked_sources,
                "mismatched_source_queries": mismatched_sources,
                "interpretation": entry.get("interpretation") or "",
            }
        )
    return rows


# ------------------------------------------------------------- evidence CSV

#: Exact CSV header, in order. Values match the on-screen evidence table.
EVIDENCE_CSV_COLUMNS = (
    "evidence",
    "tool",
    "stance",
    "queries",
    "query_ids",
    "rows",
    "truncated",
    "checked",
    "verification_status",
    "verification_details",
    "checked_source_queries",
    "mismatched_source_queries",
)

#: One consistent serialization for every list-valued CSV field.
CSV_LIST_SEPARATOR = "; "


def _csv_list(values: list[Any] | None) -> str:
    return CSV_LIST_SEPARATOR.join(str(v) for v in (values or []))


def evidence_csv(rows: list[dict[str, Any]]) -> str:
    """Render the evidence rows as CSV text (explicit download control).

    ``checked`` is "yes" when a verification assessment was recorded or
    attempted for the record, "no" otherwise; the OUTCOME always lives in
    ``verification_status`` and is never converted to a passing value.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(EVIDENCE_CSV_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows or []:
        writer.writerow(
            {
                "evidence": row.get("evidence_id") or "",
                "tool": row.get("tool") or "",
                "stance": row.get("stance") or "",
                "queries": row.get("query_count", 0),
                "query_ids": _csv_list(row.get("query_ids")),
                "rows": row.get("total_rows", 0),
                "truncated": "yes" if row.get("truncated_any") else "no",
                "checked": "yes" if row.get("checked") else "no",
                "verification_status": row.get("verification_status") or NOT_CHECKED,
                "verification_details": row.get("verification_details") or "",
                "checked_source_queries": _csv_list(row.get("checked_source_queries")),
                "mismatched_source_queries": _csv_list(
                    row.get("mismatched_source_queries")
                ),
            }
        )
    return buffer.getvalue()


# ------------------------------------------------------------------ escaping


def escape_html(value: Any) -> str:
    """Escape dynamic text placed into HTML rendered with unsafe_allow_html."""
    return html.escape("" if value is None else str(value), quote=True)


# ------------------------------------------------------- live (partial) views


def live_view_from_events(events: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Progressive view built strictly from the events emitted so far.

    Returns only what the recorded events actually contain: the evidence
    records added (with their own ids, queries, stance and returned
    nodes/edges), the latest hypothesis assessment, the policy gate result,
    the simulation, the action outcome and the verification result — each only
    if the workflow has already reached it. Nothing is invented or predicted.
    """
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    nodes: list[str] = []
    edges: list[list[str]] = []
    tool_calls = 0
    hypotheses: list[dict[str, Any]] = []
    families: list[str] = []
    policy: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    simulation: dict[str, Any] | None = None
    errors: list[str] = []

    for event in events or []:
        kind = event.get("type")
        meta = event.get("meta") or {}
        if kind == "tool" and event.get("phase") == "completed":
            tool_calls += 1
        elif kind == "evidence":
            evidence_id = meta.get("evidence_id")
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                evidence.append(
                    {
                        "evidence_id": evidence_id,
                        "tool": meta.get("tool") or event.get("tool"),
                        "stance": meta.get("stance"),
                        "query_ids": list(meta.get("query_ids") or []),
                        "row_count": list(meta.get("row_count") or []),
                        "truncated": list(meta.get("truncated") or []),
                        "derived": {},
                        # the nodes/edges THIS record's tool call returned, so
                        # the record can be highlighted in the graph
                        "nodes": [n for n in (meta.get("nodes") or []) if n],
                        "edges": [
                            list(edge) for edge in (meta.get("edges") or [])
                            if isinstance(edge, (list, tuple)) and len(edge) == 2
                        ],
                        "ok": bool(event.get("ok")),
                        "interpretation": event.get("detail") or "",
                    }
                )
            for node in meta.get("nodes") or []:
                if node and node not in nodes:
                    nodes.append(node)
            for edge in meta.get("edges") or []:
                if (isinstance(edge, (list, tuple)) and len(edge) == 2
                        and list(edge) not in edges):
                    edges.append(list(edge))
                    for value in edge:
                        if value and value not in nodes:
                            nodes.append(value)
        elif kind == "hypotheses":
            if meta.get("hypotheses"):
                hypotheses = list(meta["hypotheses"])
            if meta.get("signal_families"):
                families = list(meta["signal_families"])
        elif kind == "policy":
            policy = dict(meta)
        elif kind == "action":
            if event.get("stage") == "SIMULATE":
                simulation = dict(meta)
            elif event.get("stage") == "ACT":
                action = dict(meta)
        elif kind == "verification":
            verification = dict(meta)
        elif kind == "error":
            errors.append(event.get("detail") or event.get("title") or "error")

    return {
        "evidence": evidence,
        "evidence_count": len(evidence),
        "nodes": nodes,
        "edges": edges,
        "tool_calls": tool_calls,
        "hypotheses": hypotheses,
        "signal_families": families,
        "policy": policy,
        "simulation": simulation,
        "action": action,
        "verification": verification,
        "errors": errors,
        "event_count": len(events or []),
    }


# ------------------------------------------------- evidence cross-links


def evidence_for_entity(
    evidence_entries: list[dict[str, Any]] | None, account_id: str
) -> list[str]:
    """Evidence ids whose record mentions this account (ids only)."""
    out: list[str] = []
    for entry in evidence_entries or []:
        mentioned = set(entry.get("entity_ids") or [])
        derived = entry.get("derived") or {}
        for cycle in derived.get("cycles") or []:
            mentioned.update(cycle.get("members") or [])
            for leg in cycle.get("legs") or []:
                mentioned.update(v for v in (leg.get("src"), leg.get("dst")) if v)
            for evt in cycle.get("events") or []:
                mentioned.update(v for v in (evt.get("src"), evt.get("dst")) if v)
        for evt in derived.get("leg_events") or []:
            mentioned.update(v for v in (evt.get("src"), evt.get("dst")) if v)
        if derived.get("account_id"):
            mentioned.add(derived["account_id"])
        if account_id in mentioned:
            evidence_id = entry.get("evidence_id")
            if evidence_id and evidence_id not in out:
                out.append(evidence_id)
    return out


def evidence_for_edge(
    evidence_entries: list[dict[str, Any]] | None, src: str, dst: str
) -> list[str]:
    """Evidence ids that cover this directed transfer leg (ids only)."""
    out: list[str] = []
    for entry in evidence_entries or []:
        derived = entry.get("derived") or {}
        covered = any(
            (leg.get("src"), leg.get("dst")) == (src, dst)
            for cycle in derived.get("cycles") or []
            for leg in cycle.get("legs") or []
        ) or any(
            (evt.get("src"), evt.get("dst")) == (src, dst)
            for evt in derived.get("leg_events") or []
        )
        evidence_id = entry.get("evidence_id")
        if covered and evidence_id and evidence_id not in out:
            out.append(evidence_id)
    return out


def evidence_graph_elements(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Accounts and directed legs an evidence record itself reports.

    Sources, in order: recorded cycle members/legs/events, traced leg events,
    a derived focal account, the live event's per-record ``nodes``/``edges``
    (when the workflow has only emitted progress so far) and the record's
    ``entity_ids``. The result is exactly what the record contains — when it
    contains no usable mapping the ``mapped`` flag is False and the UI must say
    so instead of inventing a relationship.
    """
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []

    def add_node(value: Any) -> None:
        if value and str(value) not in nodes:
            nodes.append(str(value))

    def add_edge(src: Any, dst: Any) -> None:
        if not src or not dst:
            return
        pair = (str(src), str(dst))
        add_node(pair[0])
        add_node(pair[1])
        if pair not in edges:
            edges.append(pair)

    entry = entry or {}
    derived = entry.get("derived") or {}
    for cycle in derived.get("cycles") or []:
        for member in cycle.get("members") or []:
            add_node(member)
        for leg in cycle.get("legs") or []:
            add_edge(leg.get("src"), leg.get("dst"))
        for event in cycle.get("events") or []:
            add_edge(event.get("src"), event.get("dst"))
    for event in derived.get("leg_events") or []:
        add_edge(event.get("src"), event.get("dst"))
    add_node(derived.get("account_id"))

    # Live progress records carry the nodes/edges returned by that one tool
    # call; use them only for this record, never as a graph-wide guess.
    for node in entry.get("nodes") or []:
        add_node(node)
    for edge in entry.get("edges") or []:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            add_edge(edge[0], edge[1])
    for entity in entry.get("entity_ids") or []:
        add_node(entity)

    return {
        "nodes": nodes,
        "edges": [list(pair) for pair in edges],
        "mapped": bool(nodes or edges),
    }


# ------------------------------------------------------- workflow progress

#: The seven steps shown to the analyst, in order.
WORKFLOW_STEPS = (
    ("alert", "Alert"),
    ("investigation", "Network investigation"),
    ("evidence", "Evidence and hypotheses"),
    ("policy", "Policy decision"),
    ("simulation", "Impact simulation"),
    ("action", "Action outcome"),
    ("verification", "Independent verification"),
)


def workflow_progress(
    events: list[dict[str, Any]] | None = None,
    *,
    alert_selected: bool = False,
) -> list[dict[str, Any]]:
    """Which workflow steps the REAL run has actually reached.

    Derived only from recorded events (never a timer or an animation): a step
    is ``done`` when its event type has been observed, ``current`` when the
    preceding steps are done and the run is still going, and ``pending``
    otherwise. With no events and no selection nothing is marked.
    """
    reached: set[str] = set()
    if alert_selected:
        reached.add("alert")
    for event in events or []:
        kind = event.get("type")
        stage = event.get("stage")
        if kind in ("tool", "evidence", "hypotheses"):
            reached.add("investigation")
        if kind in ("evidence", "hypotheses"):
            reached.add("evidence")
        if kind == "policy":
            reached.add("policy")
        if kind == "action" and stage == "SIMULATE":
            reached.add("simulation")
        if kind == "action" and stage == "ACT":
            reached.add("action")
        if kind == "verification":
            reached.add("verification")

    ordered = [key for key, _ in WORKFLOW_STEPS]
    last_done = max(
        (ordered.index(k) for k in reached if k in ordered), default=-1
    )
    steps: list[dict[str, Any]] = []
    for index, (key, label) in enumerate(WORKFLOW_STEPS):
        # Progress is monotonic: reaching a later stage implies every earlier
        # stage already happened (a run that has produced evidence has also
        # selected the alert and investigated the network).
        if index <= last_done:
            state = "done"
        elif index == last_done + 1:
            state = "current"
        else:
            state = "pending"
        steps.append({"key": key, "label": label, "state": state})
    return steps


# ------------------------------------------------------------- run summary


def summarize_run(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compact metric-card view of one run (safe on failed snapshots)."""
    action = snapshot.get("action") or {}
    return {
        "status": snapshot.get("status"),
        "verdict": snapshot.get("canonical_verdict"),
        "tool_calls": snapshot.get("tool_calls") or 0,
        "tool_budget": snapshot.get("tool_budget"),
        "evidence_count": len(snapshot.get("evidence") or []),
        "signal_families": list(snapshot.get("signal_families") or []),
        "distinct_provenance_families": list(
            action.get("distinct_provenance_families") or []
        ),
        "completed": bool(snapshot.get("completed")),
        "termination_reason": snapshot.get("termination_reason"),
        "errors": list(snapshot.get("errors") or []),
        "truncated_evidence": list(snapshot.get("truncated_evidence") or []),
        "hypotheses_open": sum(
            1 for h in snapshot.get("hypotheses") or []
            if h.get("status") in ("open", "insufficient")
        ),
    }


def is_stale(run: dict[str, Any] | None, selected_alert: str) -> bool:
    """True when a stored run belongs to a different alert than selected."""
    if not run:
        return False
    return run.get("alert_id") != selected_alert


def guided_result_summary(view: dict[str, Any], *, status: str) -> dict[str, Any]:
    """Actual result facts for one guided-scenario card.

    Every value is copied from the current run's recorded policy/action
    envelope (or the live policy event that preceded it). Nothing is inferred
    from the alert id and no outcome is predicted.
    """
    view = view or {}
    policy = dict(view.get("policy") or {})
    action = view.get("action") or {}
    allowed = policy.get("allowed")
    if allowed is None:
        allowed = action.get("allowed")
    families = (policy.get("distinct_provenance_families")
                or action.get("distinct_provenance_families") or [])
    reasons = (policy.get("blocking_reasons")
               or action.get("blocking_reasons") or [])
    return {
        "alert_id": view.get("alert_id"),
        "status": status,
        "completed": bool(view.get("completed")),
        "verdict": view.get("canonical_verdict"),
        "action_status": action.get("action_status"),
        "allowed": allowed,
        "blocking_reasons": list(reasons),
        "qualifying_families": list(families),
        "termination_reason": view.get("termination_reason"),
        "error": view.get("error"),
    }


# ------------------------------------------------- neighbourhood + evidence


def merge_graphs(
    neighborhood: dict[str, Any],
    evidence_graph: dict[str, Any],
    *,
    focal: str | None = None,
) -> dict[str, Any]:
    """Combine the MCP neighbourhood rows with the investigation's evidence graph.

    The neighbourhood contributes the accounts and the real aggregate edges
    (with their active flag); the investigation contributes the cycle legs and
    their corroboration flags. Neither side invents anything, and evidence-only
    nodes are kept so an investigated path is always visible.
    """
    accounts: list[str] = []
    for value in (neighborhood.get("accounts") or []):
        if value and value not in accounts:
            accounts.append(value)
    for node in (evidence_graph.get("nodes") or []):
        value = node.get("id")
        if value and value not in accounts:
            accounts.append(value)

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in (neighborhood.get("edges") or []):
        key = (edge.get("src"), edge.get("dst"))
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "src": edge.get("src"), "dst": edge.get("dst"),
            "count": edge.get("count"), "total_amount": edge.get("total_amount"),
            "last_ts": edge.get("last_ts"), "active": edge.get("active"),
            "source": "neighborhood",
        })

    cycle_edges: list[dict[str, Any]] = []
    for edge in (evidence_graph.get("edges") or []):
        if edge.get("kind") != "cycle":
            continue
        cycle_edges.append({
            "src": edge.get("src"), "dst": edge.get("dst"),
            "corroborated": bool(edge.get("corroborated")),
        })
        key = (edge.get("src"), edge.get("dst"))
        if key not in seen:
            seen.add(key)
            edges.append({
                "src": edge.get("src"), "dst": edge.get("dst"),
                "count": len(edge.get("txn_ids") or []),
                "total_amount": None,
                "last_ts": (edge.get("timestamps") or [None])[0],
                "active": None, "source": "evidence",
            })

    notes: list[str] = []
    if neighborhood.get("truncated") or evidence_graph.get("truncated"):
        notes.append("one or more queries were truncated — this view may be "
                     "incomplete")
    if neighborhood.get("interpretation"):
        notes.append(neighborhood["interpretation"])
    notes.extend(evidence_graph.get("notes") or [])
    return {
        "accounts": accounts,
        "edges": edges,
        "cycle_edges": cycle_edges,
        "focal": focal or neighborhood.get("focal_account_id"),
        "notes": [n for n in notes if n],
        "truncated": bool(neighborhood.get("truncated")
                          or evidence_graph.get("truncated")),
    }


# ------------------------------------------------- evaluation artifact view


def eval_metric_rows(artifact: dict[str, Any] | None) -> list[dict[str, str]]:
    """Rows for the optional evaluation-artifact view.

    Returns nothing unless the artifact is present AND identified by seed,
    mode and dataset. The values are aggregate metrics recorded by the offline
    evaluation run; they are never presented as the current investigation's
    result.
    """
    artifact = artifact or {}
    if not artifact.get("ok") or not artifact.get("identified"):
        return []

    def add(label: str, value: Any) -> None:
        if value not in (None, ""):
            rows.append({"metric": label, "value": str(value)})

    rows: list[dict[str, str]] = []
    counts = artifact.get("counts") or {}
    add("Alerts evaluated", counts.get("evaluated"))
    recall = artifact.get("recall") or {}
    add("Recall — strict", recall.get("strict"))
    add("Recall — lenient", recall.get("lenient"))
    fpr = artifact.get("false_positive_rate") or {}
    add("False-positive rate — strict", fpr.get("strict"))
    coverage = artifact.get("pattern_coverage") or {}
    add("Pattern coverage — strict", coverage.get("strict"))
    verifier = artifact.get("verifier") or {}
    add("Verifier cases checked", verifier.get("cases_verified"))
    add("Verifier cases with error", verifier.get("cases_with_error"))
    add("Verifier claims checked", verifier.get("claims_checked"))
    status_counts = verifier.get("status_counts") or {}
    if status_counts:
        add(
            "Verification statuses",
            " · ".join(f"{key} {value}" for key, value in status_counts.items()),
        )
    tool_calls = artifact.get("tool_calls") or {}
    add("Tool calls — mean per alert", tool_calls.get("mean"))
    add("Tool calls — median per alert", tool_calls.get("median"))
    return rows
