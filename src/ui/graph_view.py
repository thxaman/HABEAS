"""Plotly graph rendering for the HABEAS UI (Phase 8.5).

Turns the neighbourhood/evidence rows returned by the backend into a directed
account graph. Every node and edge here comes from a real query result — the
builder never invents a node or an edge to fill a gap.

Design decisions:

- **layout** uses ``networkx.spring_layout`` with a fixed seed, so the same
  graph always renders in the same place (deterministic for a demo, and
  testable);
- **selection**: nodes and edges are separate Plotly traces carrying a
  ``customdata`` key. Streamlit's ``on_select="rerun"`` returns the clicked
  point; :func:`selected_from_event` maps it back to its real record;
- **encoding** never relies on colour alone — active and deactivated edges
  differ in dash pattern *and* label, corroborated circulation is solid with a
  text label, structural-only cycles are dashed and labelled "structure only".

No Streamlit import here: the plotly figure and the selection parser are pure
and unit-tested directly.
"""
from __future__ import annotations

from typing import Any

import networkx as nx

_COLORS = {
    "bg": "#0b1424",
    "focal": "#14b8a6",        # teal — the alerted account
    "account": "#2f4a70",      # slate blue
    "primary": "#38bdf8",      # cyan — investigated entities
    "corroborated": "#2dd4bf",  # teal solid
    "structural": "#94a3b8",    # slate dashed
    "active": "#38bdf8",        # cyan solid
    "inactive": "#f59e0b",      # amber dashed
    "live": "#fbbf24",          # amber overlay: implicated during the run
    "focus": "#a78bfa",         # violet overlay: highlighted evidence record
    "font": "#dbe7f4",
}


def _layout(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, tuple[float, float]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((s, d) for s, d in edges)
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}
    # seed fixed so a given graph always lays out identically
    return nx.spring_layout(graph, seed=7, k=None, iterations=60)


def build_figure(
    *,
    accounts: list[str],
    edges: list[dict[str, Any]],
    focal_account_id: str | None = None,
    primary_entities: list[str] | None = None,
    cycle_edges: list[dict[str, Any]] | None = None,
    transactions: list[dict[str, Any]] | None = None,
    live_nodes: list[str] | None = None,
    live_edges: list[list[str]] | None = None,
    focus_nodes: list[str] | None = None,
    focus_edges: list[list[str]] | None = None,
    height: int = 470,
):
    """Build the Plotly figure. Returns ``None`` for an empty graph."""
    import plotly.graph_objects as go

    accounts = [a for a in dict.fromkeys(accounts or []) if a]
    edges = list(edges or [])
    if not accounts:
        return None
    primary = set(primary_entities or [])
    fig = go.Figure()

    edge_keys = [(e["src"], e["dst"]) for e in edges if e.get("src") and e.get("dst")]
    pos = _layout(accounts, edge_keys)

    # ---- edges -----------------------------------------------------------
    # Grouped into three visually distinct traces so the legend explains the
    # semantics (and so the meaning survives without colour).
    corroborated = {
        (c["src"], c["dst"]) for c in (cycle_edges or [])
        if c.get("corroborated") and c.get("src") and c.get("dst")
    }
    structural = {
        (c["src"], c["dst"]) for c in (cycle_edges or [])
        if not c.get("corroborated") and c.get("src") and c.get("dst")
    }

    # Per-segment transaction counts, so an edge hover can quote what the
    # queries actually returned (never a synthesised event).
    txn_counts: dict[tuple[str, str], int] = {}
    for txn in (transactions or []):
        key = (txn.get("src"), txn.get("dst"))
        txn_counts[key] = txn_counts.get(key, 0) + 1

    def edge_hover(edge: dict[str, Any], classification: str) -> str:
        src_id, dst_id = edge.get("src"), edge.get("dst")
        lines = [f"{src_id} → {dst_id}", classification]
        if edge.get("count") is not None:
            lines.append(f"{int(edge['count'])} transfer(s) aggregated")
        if edge.get("total_amount") is not None:
            lines.append(f"{float(edge['total_amount']):,.2f} total volume")
        if edge.get("last_ts"):
            lines.append(f"last transfer {edge['last_ts']}")
        known = txn_counts.get((src_id, dst_id))
        if known:
            lines.append(f"{known} transaction event(s) returned by the query")
        return "<br>".join(lines)

    def add_edges(trace_edges, name, color, dash, classification,
                  show_legend=True):
        """One grouped trace whose per-point metadata matches ITS OWN segment.

        ``xs``/``ys`` carry three entries per segment (start, end, break); the
        hover text and customdata arrays are built with exactly the same shape
        so a click resolves to the segment that was actually clicked — the
        earlier implementation copied segment 0's metadata onto every point.
        """
        xs: list[float | None] = []
        ys: list[float | None] = []
        texts: list[str] = []
        custom: list[list[Any]] = []
        for edge in trace_edges:
            src_id, dst_id = edge.get("src"), edge.get("dst")
            if src_id not in pos or dst_id not in pos:
                continue
            x0, y0 = pos[src_id]
            x1, y1 = pos[dst_id]
            hover = edge_hover(edge, classification)
            identity = ["edge", src_id, dst_id]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            # identical length to xs/ys, one entry per point of THIS segment
            texts += [hover, hover, hover]
            custom += [identity, identity, identity]
        if not xs:
            return
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line={"color": color, "width": 2.4, "dash": dash},
                name=name, legendgroup=name, showlegend=show_legend,
                hoverinfo="text", text=texts, customdata=custom,
            )
        )

    _CIRCULATION = "corroborated circulation (chronological)"
    _STRUCTURAL = "structural connectivity only (not circulation)"

    add_edges(
        [e for e in edges if e.get("active") is not False
         and (e["src"], e["dst"]) not in corroborated
         and (e["src"], e["dst"]) not in structural],
        name="active routing (TRANSFERRED_TO)", color=_COLORS["active"],
        dash="solid", classification="active routing (current topology)",
    )
    add_edges(
        [e for e in edges if e.get("active") is False],
        name="deactivated routing", color=_COLORS["inactive"], dash="dash",
        classification="deactivated routing (no longer active)",
    )
    add_edges(
        [{"src": s, "dst": d} for (s, d) in sorted(corroborated)],
        name="corroborated circulation", color=_COLORS["corroborated"],
        dash="solid", classification=_CIRCULATION,
    )
    add_edges(
        [{"src": s, "dst": d} for (s, d) in sorted(structural)],
        name="structural cycle only", color=_COLORS["structural"], dash="dot",
        classification=_STRUCTURAL,
    )

    # ---- live overlay (elements the run actually implicated so far) -------
    # Only ever built from accounts/legs a tool returned during this run, so
    # it marks real progress rather than animating historical transfers.
    live_edge_list = [
        e for e in (live_edges or []) if e and len(e) == 2
        and e[0] in pos and e[1] in pos
    ]
    if live_edge_list:
        lx: list[float | None] = []
        ly: list[float | None] = []
        ltext: list[str] = []
        for src_id, dst_id in live_edge_list:
            x0, y0 = pos[src_id]
            x1, y1 = pos[dst_id]
            label = f"{src_id} → {dst_id}<br>implicated by a tool result in this run"
            lx += [x0, x1, None]
            ly += [y0, y1, None]
            ltext += [label, label, label]
        fig.add_trace(
            go.Scatter(
                x=lx, y=ly, mode="lines",
                line={"color": _COLORS["live"], "width": 4.5, "dash": "dash"},
                name="implicated during this run", legendgroup="live",
                showlegend=True, hoverinfo="text", text=ltext,
            )
        )
    overlay_nodes = [n for n in (live_nodes or []) if n in pos]
    if overlay_nodes:
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in overlay_nodes],
                y=[pos[n][1] for n in overlay_nodes],
                mode="markers",
                marker={"size": 30, "color": "rgba(0,0,0,0)",
                        "line": {"width": 2.5, "color": _COLORS["live"]}},
                name="implicated during this run", legendgroup="live",
                showlegend=False,
                hovertext=[f"account {n}<br>implicated during this run"
                           for n in overlay_nodes],
                hoverinfo="text",
            )
        )

    # ---- focus overlay: one selected evidence record's own elements --------
    # Built only from accounts/legs THAT record reported; a record with no
    # usable mapping highlights nothing (the caller says so in words).
    focus_edge_list = [
        e for e in (focus_edges or []) if e and len(e) == 2
        and e[0] in pos and e[1] in pos
    ]
    if focus_edge_list:
        fx: list[float | None] = []
        fy: list[float | None] = []
        ftext: list[str] = []
        fcustom: list[list[Any]] = []
        for src_id, dst_id in focus_edge_list:
            x0, y0 = pos[src_id]
            x1, y1 = pos[dst_id]
            label = f"{src_id} → {dst_id}<br>reported by the highlighted evidence record"
            identity = ["edge", src_id, dst_id]
            fx += [x0, x1, None]
            fy += [y0, y1, None]
            ftext += [label, label, label]
            fcustom += [identity, identity, identity]
        fig.add_trace(
            go.Scatter(
                x=fx, y=fy, mode="lines",
                line={"color": _COLORS["focus"], "width": 4.5, "dash": "solid"},
                name="highlighted evidence", legendgroup="focus",
                showlegend=True, hoverinfo="text", text=ftext,
                customdata=fcustom,
            )
        )
    focus_node_list = [n for n in (focus_nodes or []) if n in pos]
    if focus_node_list:
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in focus_node_list],
                y=[pos[n][1] for n in focus_node_list],
                mode="markers",
                marker={"size": 32, "color": "rgba(0,0,0,0)",
                        "line": {"width": 2.8, "color": _COLORS["focus"]}},
                name="highlighted evidence", legendgroup="focus",
                showlegend=False,
                customdata=[[["node", n]] for n in focus_node_list],
                hovertext=[f"account {n}<br>mentioned by the highlighted evidence record"
                           for n in focus_node_list],
                hoverinfo="text",
            )
        )

    # ---- nodes -----------------------------------------------------------
    def bucket(account: str) -> str:
        if account == focal_account_id:
            return "focal"
        if account in primary:
            return "primary"
        return "account"

    for key, name, color in (
        ("focal", "alerted account", _COLORS["focal"]),
        ("primary", "implicated entity", _COLORS["primary"]),
        ("account", "account", _COLORS["account"]),
    ):
        group = [a for a in accounts if bucket(a) == key]
        if not group:
            continue
        fig.add_trace(
            go.Scatter(
                x=[pos[a][0] for a in group],
                y=[pos[a][1] for a in group],
                mode="markers+text",
                marker={
                    "size": 26 if key == "focal" else 18,
                    "color": color,
                    "line": {"width": 1.6, "color": _COLORS["font"]},
                },
                text=group,
                textposition="top center",
                textfont={"size": 10, "color": _COLORS["font"]},
                name=name, legendgroup=name,
                customdata=[[["node", a]] for a in group],
                hovertext=[f"account {a}" for a in group],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_COLORS["bg"],
        font={"color": _COLORS["font"], "size": 12},
        showlegend=True,
        legend={
            "orientation": "h", "yanchor": "bottom", "y": -0.16,
            "x": 0, "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={"visible": False}, yaxis={"visible": False},
        dragmode="pan",
        clickmode="event+select",
        hovermode="closest",
    )
    return fig


def selected_from_event(event: Any) -> dict[str, Any] | None:
    """Map a ``st.plotly_chart(on_select=...)`` result to a real record.

    Returns ``{"kind": "node", "account_id": ...}`` or
    ``{"kind": "edge", "src": ..., "dst": ...}``, or ``None`` when nothing is
    selected. Tolerant of the several shapes Streamlit/Plotly return.
    """
    if not event:
        return None
    selection = event.get("selection") if isinstance(event, dict) else None
    points = None
    if isinstance(selection, dict):
        points = selection.get("points")
    if points is None and isinstance(event, dict):
        points = event.get("points")
    if not points:
        return None
    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else None
    # A node selection may arrive as the flat ["node", id] or nested [[...]].
    if isinstance(custom, list) and len(custom) == 1 and isinstance(custom[0], list):
        custom = custom[0]
    if not isinstance(custom, (list, tuple)) or not custom:
        return None
    kind = custom[0]
    if kind == "node" and len(custom) >= 2:
        return {"kind": "node", "account_id": custom[1]}
    if kind == "edge" and len(custom) >= 3:
        return {"kind": "edge", "src": custom[1], "dst": custom[2]}
    return None


def node_details(account_id: str, *, edges: list[dict[str, Any]],
                 transactions: list[dict[str, Any]] | None = None,
                 cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Real relationships of one account, from the returned rows only."""
    outgoing = [e for e in edges if e.get("src") == account_id]
    incoming = [e for e in edges if e.get("dst") == account_id]
    related_txns = [
        t for t in (transactions or [])
        if t.get("src") == account_id or t.get("dst") == account_id
    ]
    related_cases = [
        c for c in (cases or [])
        if account_id in (c.get("entity_ids") or []) or c.get("account_id") == account_id
    ]
    return {
        "account_id": account_id,
        "out_degree": len(outgoing),
        "in_degree": len(incoming),
        "outgoing": sorted(outgoing, key=lambda e: -float(e.get("total_amount") or 0)),
        "incoming": sorted(incoming, key=lambda e: -float(e.get("total_amount") or 0)),
        "transactions": related_txns,
        "cases": related_cases,
    }


def edge_details(src: str, dst: str, *, edges: list[dict[str, Any]],
                 transactions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Real evidence behind one directed edge (timestamps and ids included)."""
    edge = next(
        (e for e in edges if e.get("src") == src and e.get("dst") == dst), None
    )
    txns = [
        t for t in (transactions or [])
        if t.get("src") == src and t.get("dst") == dst
    ]
    txn_ids: list[str] = []
    timestamps: list[str] = []
    for txn in txns:
        txn_id = txn.get("txn_id") or txn.get("transaction_id")
        if txn_id and str(txn_id) not in txn_ids:
            txn_ids.append(str(txn_id))
        stamp = txn.get("ts") or txn.get("timestamp")
        if stamp and str(stamp) not in timestamps:
            timestamps.append(str(stamp))
    timestamps.sort()
    return {
        "src": src, "dst": dst,
        "edge": edge,
        "transactions": txns,
        "txn_ids": txn_ids,
        "timestamps": timestamps,
        "first_ts": timestamps[0] if timestamps else None,
        "last_ts": timestamps[-1] if timestamps else None,
        "has_aggregate": edge is not None,
        "has_transactions": bool(txns),
    }