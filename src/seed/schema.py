"""Schema helpers for the HABEAS demo graph (trusted setup code).

This module can clear and rewrite the graph; only seed scripts, tests, and
explicit infrastructure commands may use it (PLAN.md §0/§3). All queries are
parameterized; labels/relationship types are code-owned constants validated
against the strict identifier pattern before interpolation.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator

from src.db import ensure_index, run_query, validate_identifier

BATCH_SIZE = 500

# PLAN.md §5: index the id of every entity label the investigation flow uses.
# Case/Evidence/Simulation/Action labels do not exist until Phase 5 (or are
# first written by it); creating their indexes in setup is harmless and keeps
# seed/test setup repeat-safe.
INDEX_SPECS: list[tuple[str, str]] = [
    ("Customer", "id"),
    ("Account", "id"),
    ("Transaction", "id"),
    ("Alert", "id"),
    ("Case", "id"),
    ("Evidence", "id"),
    ("Simulation", "id"),
    ("Action", "id"),
]


def ensure_indexes(graph) -> dict[str, str]:
    """Create all schema indexes; repeat-safe (db.ensure_index pre-checks)."""
    return {
        f"{label}.{prop}": ensure_index(graph, label, prop)
        for label, prop in INDEX_SPECS
    }


def clear_graph(graph) -> None:
    """Erase ALL data in the graph. Only call after the reset guard passed."""
    run_query(graph, "MATCH (n) DETACH DELETE n")


def node_count(graph) -> int:
    res = run_query(graph, "MATCH (n) RETURN count(n) AS c", read_only=True)
    return int(res.result_set[0][0])


def graph_counts(graph) -> dict[str, int]:
    """Per-label node counts and per-type relationship counts."""
    counts: dict[str, int] = {}
    nodes = run_query(
        graph,
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY label",
        read_only=True,
    )
    for label, c in nodes.result_set:
        counts[f"nodes:{label}"] = int(c)
    rels = run_query(
        graph,
        "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t",
        read_only=True,
    )
    for t, c in rels.result_set:
        counts[f"rels:{t}"] = int(c)
    return counts


def _chunk(rows: list[Any], size: int = BATCH_SIZE) -> Iterator[list[Any]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def insert_nodes(graph, label: str, rows: list[dict[str, Any]]) -> None:
    """Batch-create nodes whose full property map is each row (no extras)."""
    validate_identifier(label, "label")
    for chunk in _chunk(rows):
        run_query(
            graph,
            f"UNWIND $rows AS row CREATE (n:{label}) SET n = row",
            {"rows": chunk},
        )


def insert_relationships(
    graph,
    rel_type: str,
    rows: list[dict[str, Any]],
    *,
    src_label: str,
    dst_label: str,
) -> None:
    """Batch-create relationships. Each row: {"src", "dst", "props": {...}}."""
    validate_identifier(rel_type, "relationship type")
    validate_identifier(src_label, "label")
    validate_identifier(dst_label, "label")
    for chunk in _chunk(rows):
        run_query(
            graph,
            f"UNWIND $rows AS row "
            f"MATCH (a:{src_label} {{id: row.src}}), (b:{dst_label} {{id: row.dst}}) "
            f"CREATE (a)-[r:{rel_type}]->(b) SET r = row.props",
            {"rows": chunk},
        )


def insert_transactions(graph, rows: list[dict[str, Any]]) -> None:
    """Batch-create Transaction nodes plus SENT/RECEIVED_BY in one pass.

    Each row: {"txn": {node properties}, "src": account_id, "dst": account_id}.
    """
    for chunk in _chunk(rows):
        run_query(
            graph,
            "UNWIND $rows AS row "
            "CREATE (t:Transaction) SET t = row.txn "
            "WITH row, t "
            "MATCH (a:Account {id: row.src}), (b:Account {id: row.dst}) "
            "CREATE (a)-[:SENT]->(t)-[:RECEIVED_BY]->(b)",
            {"rows": chunk},
        )
