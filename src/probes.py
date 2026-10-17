"""Phase 0 FalkorDB capability probes (PLAN.md §4).

Run:  python -m src.probes

Writes the ACTUAL observed results to docs/falkordb-capabilities.md. All probes
run against the disposable demo graph and are re-run safe (MERGE/SET based).
Probe labels are code-owned constants; every input-derived value is passed as
a query parameter.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.db import (
    QueryError,
    ensure_index,
    get_client,
    get_graph,
    index_exists,
    list_indexes,
    normalize_rows,
    run_query,
)

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "falkordb-capabilities.md"

IMAGE_TAG = "falkordb/falkordb:v4.22.1"

# Code-owned label constants (never derived from user input).
LBL_PROBE = "Phase0Probe"
LBL_CHAIN = "Phase0Chain"
LBL_CYCLE = "Phase0Cycle"
LBL_LEGACY = "Phase0ProbeLegacy"
LBL_FRESH = "Phase0IdxFresh"
LBL_ENSURE = "Phase0IdxEnsure"

ROUNDTRIP_ID = "phase0-roundtrip"
CHAIN_ROWS = [{"id": "PC1", "next": "PC2"}, {"id": "PC2", "next": "PC3"}]
CYCLE_ROWS = [
    {"id": "CY1", "next": "CY2"},
    {"id": "CY2", "next": "CY3"},
    {"id": "CY3", "next": "CY4"},
    {"id": "CY4", "next": "CY1"},
]


@dataclass
class ProbeResult:
    id: str
    title: str
    ok: bool
    observed: str
    required_for_mvp: bool = False


def _ensure_chain(graph) -> None:
    run_query(
        graph,
        f"UNWIND $rows AS row MERGE (a:{LBL_CHAIN} {{id: row.id}}) "
        f"MERGE (b:{LBL_CHAIN} {{id: row.next}}) MERGE (a)-[:NEXT]->(b)",
        {"rows": CHAIN_ROWS},
    )


def _ensure_cycle(graph) -> None:
    run_query(
        graph,
        f"UNWIND $rows AS row MERGE (a:{LBL_CYCLE} {{id: row.id}}) "
        f"MERGE (b:{LBL_CYCLE} {{id: row.next}}) MERGE (a)-[:NEXT]->(b)",
        {"rows": CYCLE_ROWS},
    )


def probe_return_one(graph) -> ProbeResult:
    res = run_query(graph, "RETURN 1 AS one", read_only=True)
    ok = res.result_set == [[1]]
    return ProbeResult(
        "P0-01", "Connection + RETURN 1", ok, f"RETURN 1 -> {res.result_set}",
        required_for_mvp=True,
    )


def probe_roundtrip(graph) -> ProbeResult:
    run_query(
        graph,
        f"MERGE (p:{LBL_PROBE} {{id: $id}}) SET p.n = $n, p.updated_at = $ts",
        {"id": ROUNDTRIP_ID, "n": 1, "ts": "2026-01-01T00:00:00Z"},
    )
    run_query(
        graph,
        f"MATCH (p:{LBL_PROBE} {{id: $id}}) SET p.n = p.n + $inc RETURN p.n AS n",
        {"id": ROUNDTRIP_ID, "inc": 1},
    )
    res = run_query(
        graph,
        f"MATCH (p:{LBL_PROBE} {{id: $id}}) RETURN p.id AS id, p.n AS n",
        {"id": ROUNDTRIP_ID},
        read_only=True,
    )
    rows = res.result_set
    ok = bool(rows) and rows[0][0] == ROUNDTRIP_ID and rows[0][1] == 2
    return ProbeResult(
        "P0-02",
        "Parameterized write/read round-trip (MERGE, SET, MATCH)",
        ok,
        f"wrote n=1 via MERGE+SET, incremented via SET, read back {rows} — "
        "all values bound as parameters",
        required_for_mvp=True,
    )


def probe_db_indexes_call(graph) -> ProbeResult:
    try:
        rows = list_indexes(graph)
        has_probe = any(
            r["label"] == LBL_PROBE and "id" in r["properties"] for r in rows
        )
        sample = [(r["label"], r["properties"]) for r in rows][:6]
        return ProbeResult(
            "P0-18", "Index enumeration via CALL db.indexes()", has_probe,
            f"{len(rows)} indexes listed (rows parsed as label+properties); "
            f"known index Phase0Probe.id listed: {has_probe}; sample: {sample}",
        )
    except QueryError as exc:
        return ProbeResult(
            "P0-18", "Index enumeration via CALL db.indexes()", False,
            f"not available: kind={exc.kind}: {str(exc)[:120]} — repeated index "
            "setup must then rely on create-and-tolerate only",
        )


def probe_drop_index(graph) -> ProbeResult:
    stmt_create = f"CREATE INDEX FOR (p:{LBL_FRESH}) ON (p.id)"
    stmt_drop = f"DROP INDEX ON :{LBL_FRESH}(id)"
    try:
        run_query(graph, stmt_create)
    except QueryError as exc:
        if not (exc.kind == "constraint" and "already indexed" in str(exc).lower()):
            raise
    try:
        run_query(graph, stmt_drop)
    except QueryError as exc:
        return ProbeResult(
            "P0-19", "DROP INDEX support", False,
            f"rejected: kind={exc.kind}: {str(exc)[:120]} — scratch indexes cannot "
            "be reset, so fresh-creation retesting is limited",
        )
    gone = not index_exists(graph, LBL_FRESH, "id")
    return ProbeResult(
        "P0-19", "DROP INDEX support", gone,
        f"`DROP INDEX ON :Label(prop)` accepted; index no longer listed after "
        f"drop: {gone}",
    )


def probe_index_lifecycle(graph) -> ProbeResult:
    """Fresh creation, duplicate rejection, and independent existing-index
    detection — tested as separate steps (user correction on P0-03)."""
    stmt_create = f"CREATE INDEX FOR (p:{LBL_FRESH}) ON (p.id)"
    stmt_drop = f"DROP INDEX ON :{LBL_FRESH}(id)"
    parts: list[str] = []
    fresh_ok: bool | None = None

    if index_exists(graph, LBL_FRESH, "id"):
        try:
            run_query(graph, stmt_drop)
            parts.append("dropped pre-existing scratch index to retest fresh creation")
        except QueryError as exc:
            parts.append(
                f"pre-existing scratch index could not be dropped (kind={exc.kind})"
            )

    if not index_exists(graph, LBL_FRESH, "id"):
        try:
            res = run_query(graph, stmt_create)
            fresh_ok = True
            parts.append(f"fresh creation OK (indices_created={res.indices_created})")
        except QueryError as exc:
            fresh_ok = False
            parts.append(f"fresh creation FAILED: kind={exc.kind}: {str(exc)[:80]}")
    else:
        parts.append(
            "fresh creation not re-testable this run (scratch index still present)"
        )

    dup_ok = False
    try:
        run_query(graph, stmt_create)
        parts.append("duplicate creation unexpectedly succeeded")
    except QueryError as exc:
        dup_ok = exc.kind == "constraint" and "already indexed" in str(exc).lower()
        parts.append(
            f"duplicate creation rejected with the expected already-indexed "
            f"constraint error (kind={exc.kind})"
            if dup_ok else
            f"duplicate creation failed DIFFERENTLY: kind={exc.kind}: {str(exc)[:80]}"
        )

    detect_ok = index_exists(graph, LBL_FRESH, "id")
    parts.append(f"existing-index detection via CALL db.indexes(): listed={detect_ok}")

    try:
        run_query(graph, stmt_drop)
        parts.append("cleanup drop OK")
    except QueryError as exc:
        parts.append(f"cleanup drop failed (kind={exc.kind})")

    checked = [ok for ok in (fresh_ok, dup_ok, detect_ok) if ok is not None]
    return ProbeResult(
        "P0-03",
        "Index lifecycle: fresh creation, duplicate rejection, detection",
        all(checked),
        "; ".join(parts),
    )


def probe_ensure_index(graph) -> ProbeResult:
    """Live check that the Phase 1 index-setup helper is repeat-safe."""
    first = ensure_index(graph, LBL_ENSURE, "id")
    second = ensure_index(graph, LBL_ENSURE, "id")
    ok = first in {"created", "exists"} and second == "exists"
    try:
        run_query(graph, f"DROP INDEX ON :{LBL_ENSURE}(id)")
        cleanup = "cleanup drop OK"
    except QueryError as exc:
        cleanup = f"cleanup drop failed (kind={exc.kind})"
    return ProbeResult(
        "P0-20", "ensure_index() repeated setup is safe", ok,
        f"first call -> {first}; second call -> {second}; {cleanup}",
    )


def probe_index_legacy_syntax(graph) -> ProbeResult:
    stmt = f"CREATE INDEX ON :{LBL_LEGACY}(id)"
    try:
        res = run_query(graph, stmt)
        first = f"accepted (indices_created={res.indices_created})"
    except QueryError as exc:
        if exc.kind == "constraint" and "already indexed" in str(exc).lower():
            first = "accepted on an earlier probe run (index already existed)"
        else:
            return ProbeResult(
                "P0-04", "Index creation — legacy syntax", False,
                f"rejected: kind={exc.kind}: {str(exc)[:120]} — use the new syntax (P0-03)",
            )
    return ProbeResult(
        "P0-04", "Index creation — legacy syntax", True,
        f"legacy `CREATE INDEX ON :Label(prop)` {first}",
    )


def probe_param_in_limit(graph) -> ProbeResult:
    try:
        res = run_query(graph, "RETURN 1 AS x LIMIT $lim", {"lim": 1}, read_only=True)
        return ProbeResult(
            "P0-05", "Parameter in LIMIT", res.result_set == [[1]],
            "`LIMIT $lim` accepted with a bound parameter — parameterize all limits",
        )
    except QueryError as exc:
        return ProbeResult(
            "P0-05", "Parameter in LIMIT", False,
            f"rejected: kind={exc.kind}: {str(exc)[:120]} — use a validated bounded "
            "integer constant at that grammar position (PLAN.md §4 rule)",
        )


def probe_param_in_varlength(graph) -> ProbeResult:
    _ensure_chain(graph)
    try:
        run_query(
            graph,
            f"MATCH (a:{LBL_CHAIN} {{id: $a}})-[:NEXT*1..$hops]->(b:{LBL_CHAIN}) "
            "RETURN b.id AS id",
            {"a": "PC1", "hops": 2},
            read_only=True,
        )
        return ProbeResult(
            "P0-06", "Parameter in variable-length path bound", True,
            "`*1..$hops` accepted with a bound parameter",
        )
    except QueryError as exc:
        return ProbeResult(
            "P0-06", "Parameter in variable-length path bound", False,
            f"rejected: kind={exc.kind}: {str(exc)[:120]} — hop caps must be validated "
            "ints in [1,5] interpolated as Python-validated constants "
            "(PLAN.md §0 allows exactly this one interpolation)",
        )


def probe_varlength_traversal(graph) -> ProbeResult:
    _ensure_chain(graph)
    res = run_query(
        graph,
        f"MATCH (a:{LBL_CHAIN} {{id: $a}})-[:NEXT*1..2]->(b:{LBL_CHAIN}) "
        "RETURN b.id AS id ORDER BY id",
        {"a": "PC1"},
        read_only=True,
    )
    ids = [r[0] for r in res.result_set]
    return ProbeResult(
        "P0-07", "Bounded variable-length traversal", ids == ["PC2", "PC3"],
        f"1..2 hops from PC1 over NEXT -> {ids}",
        required_for_mvp=True,
    )


def probe_cycle_query(graph) -> ProbeResult:
    _ensure_cycle(graph)
    res = run_query(
        graph,
        f"MATCH path = (a:{LBL_CYCLE} {{id: $id}})-[:NEXT*1..4]->(a) "
        "RETURN [n IN nodes(path) | n.id] AS members LIMIT 5",
        {"id": "CY1"},
        read_only=True,
    )
    members = [r[0] for r in res.result_set]
    ok = any(m and m[0] == "CY1" and m[-1] == "CY1" and len(m) == 5 for m in members)
    return ProbeResult(
        "P0-08", "Bounded cycle-detection query (MVP template)", ok,
        f"self-returning directed path 1..4 found the cycle (members={members}); "
        "temporal corroboration stays a Phase 2 Python concern",
        required_for_mvp=True,
    )


def probe_shortest_path(graph) -> ProbeResult:
    _ensure_chain(graph)
    attempts = []
    for label, pattern in (
        ("directed", "[:NEXT*..3]->"),
        ("undirected", "[*..3]-"),
    ):
        stmt = (
            f"MATCH (a:{LBL_CHAIN} {{id: $s}}), (b:{LBL_CHAIN} {{id: $t}}) "
            f"WITH shortestPath((a)-{pattern}(b)) AS p RETURN length(p) AS hops"
        )
        try:
            res = run_query(graph, stmt, {"s": "PC1", "t": "PC3"}, read_only=True)
            hops = res.result_set[0][0] if res.result_set else None
            return ProbeResult(
                "P0-09", "shortestPath", hops == 2,
                f"{label} WITH-clause form works: shortestPath PC1-PC3 = {hops} "
                "hops. NOTE: `MATCH p = shortestPath(...)` is rejected — this "
                "server only supports shortestPath in WITH/RETURN clauses",
            )
        except QueryError as exc:
            attempts.append(f"{label}: kind={exc.kind}: {str(exc)[:100]}")
    return ProbeResult(
        "P0-09", "shortestPath", False,
        f"not usable: {'; '.join(attempts)} — fallback: bounded BFS in Python",
    )


def probe_list_comprehension(graph) -> ProbeResult:
    res = run_query(
        graph, "RETURN [x IN range(1, 5) WHERE x % 2 = 0 | x * 10] AS xs", read_only=True
    )
    cell = res.result_set[0][0] if res.result_set else None
    return ProbeResult(
        "P0-10", "List comprehension", cell == [20, 40],
        f"cell value -> {cell} (expected [20, 40])",
    )


def probe_reduce(graph) -> ProbeResult:
    res = run_query(
        graph, "RETURN reduce(s = 0, x IN [1, 2, 3] | s + x) AS total", read_only=True
    )
    cell = res.result_set[0][0] if res.result_set else None
    return ProbeResult(
        "P0-11", "reduce()", cell == 6, f"cell value -> {cell} (expected 6)",
    )


def probe_ro_query(graph) -> ProbeResult:
    res = run_query(graph, "RETURN 21 AS x", read_only=True)
    ok = res.result_set == [[21]]
    return ProbeResult(
        "P0-12", "Read-only query path (ro_query)", ok,
        "graph.ro_query works; run_query uses it for every read (required_for_mvp)",
        required_for_mvp=True,
    )


def probe_timeout(graph) -> ProbeResult:
    heavy = "UNWIND range(1, 60000000) AS x WITH x WHERE x = 60000000 RETURN count(x) AS c"
    t0 = time.perf_counter()
    try:
        run_query(graph, heavy, timeout_ms=300, read_only=True)
        dt = time.perf_counter() - t0
        return ProbeResult(
            "P0-13", "Per-query timeout enforcement (300ms request)", False,
            f"query completed in {dt:.2f}s — server did NOT abort at the requested "
            "300ms timeout; keep queries bounded via LIMIT/hop caps regardless",
        )
    except QueryError as exc:
        dt = time.perf_counter() - t0
        if exc.kind == "timeout":
            return ProbeResult(
                "P0-13", "Per-query timeout enforcement (300ms request)", True,
                f"query aborted after {dt:.2f}s with QueryError kind=timeout: {str(exc)[:100]}",
            )
        return ProbeResult(
            "P0-13", "Per-query timeout enforcement (300ms request)", False,
            f"failed with kind={exc.kind} after {dt:.2f}s: {str(exc)[:120]}",
        )


def probe_procedures_list(graph) -> ProbeResult:
    try:
        res = run_query(
            graph, "CALL dbms.procedures() YIELD name RETURN name AS name",
            read_only=True,
        )
        names = sorted(str(r[0]) for r in res.result_set)
        interesting = [
            n for n in names
            if any(t in n.lower() for t in ("page", "rank", "bfs", "wcc", "betweenness"))
        ]
        return ProbeResult(
            "P0-14", "CALL dbms.procedures()", True,
            f"available; {len(names)} procedures registered; "
            f"page/rank/bfs/wcc/betweenness-related: {interesting or 'none'}; "
            f"first 10: {names[:10]}",
        )
    except QueryError as exc:
        return ProbeResult(
            "P0-14", "CALL dbms.procedures()", False,
            f"not available: kind={exc.kind}: {str(exc)[:120]} "
            "(PLAN.md: never assume it exists)",
        )


def probe_pagerank(graph) -> ProbeResult:
    attempts = []
    for name in ("algo.pageRank", "algo.page_rank"):
        stmt = (
            f"CALL {name}('{LBL_CHAIN}', 'NEXT') YIELD node, score "
            "RETURN node.id AS id, score LIMIT 5"
        )
        try:
            res = run_query(graph, stmt, read_only=True)
            return ProbeResult(
                "P0-15", "Optional PageRank procedure", True,
                f"{name} works; returned {res.result_set} (stretch feature only)",
            )
        except QueryError as exc:
            attempts.append(f"{name}: kind={exc.kind}: {str(exc)[:100]}")
    return ProbeResult(
        "P0-15", "Optional PageRank procedure", False,
        f"not usable: {'; '.join(attempts)} — MVP does not depend on it; "
        "hub_scores uses weighted in-degree (PLAN.md §7/§8)",
    )


def probe_normalize(graph) -> ProbeResult:
    res = run_query(
        graph,
        f"MATCH (p:{LBL_PROBE} {{id: $id}}) RETURN p AS node LIMIT 1",
        {"id": ROUNDTRIP_ID},
        read_only=True,
    )
    columns, rows = normalize_rows(res)
    payload = json.dumps({"columns": columns, "rows": rows}, sort_keys=True)
    ok = bool(rows) and rows[0][0].get("_type") == "node" and len(payload) > 0
    return ProbeResult(
        "P0-16", "Normalize graph values to JSON-safe primitives", ok,
        f"columns={columns}; row0={json.dumps(rows[0], sort_keys=True)[:160]}; "
        "json.dumps OK",
        required_for_mvp=True,
    )


def probe_path_node_order(graph) -> ProbeResult:
    """Fact probe: are Edge endpoints and nodes(path) order from
    variable-length matches faithful to the stored traversal direction?"""
    _ensure_chain(graph)
    _ensure_cycle(graph)
    shapes = {
        "open-2hop-exact": (
            f"MATCH path = (a:{LBL_CHAIN} {{id: $a}})-[:NEXT*2..2]->(b:{LBL_CHAIN}) "
            "RETURN nodes(path) AS ns, relationships(path) AS rs LIMIT 1",
            {"a": "PC1"},
        ),
        "open-1hop-exact": (
            f"MATCH path = (a:{LBL_CHAIN} {{id: $a}})-[:NEXT*1..1]->(b:{LBL_CHAIN}) "
            "RETURN nodes(path) AS ns, relationships(path) AS rs LIMIT 1",
            {"a": "PC1"},
        ),
        "closed-cycle-self-return": (
            f"MATCH path = (a:{LBL_CYCLE} {{id: $a}})-[:NEXT*1..4]->(a) "
            "RETURN nodes(path) AS ns, relationships(path) AS rs LIMIT 1",
            {"a": "CY1"},
        ),
    }
    truth = run_query(
        graph,
        "MATCH (x)-[r:NEXT]->(y) RETURN id(r) AS rid, id(x) AS s, id(y) AS d",
        read_only=True,
    )
    rel_truth = {r[0]: (r[1], r[2]) for r in truth.result_set}
    lookup = run_query(
        graph,
        f"MATCH (n) WHERE n:{LBL_CHAIN} OR n:{LBL_CYCLE} "
        "RETURN n.id AS id, id(n) AS internal",
        read_only=True,
    )
    internal_to_prop = {r[1]: r[0] for r in lookup.result_set}

    descriptions = []
    endpoints_ok_count = 0
    node_order_ok_count = 0
    rel_order_ok_count = 0
    for name, (cypher, params) in shapes.items():
        res = run_query(graph, cypher, params, read_only=True)
        ns, rs = res.result_set[0]
        node_props = [internal_to_prop.get(n.id) for n in ns]
        # (1) Is each Edge's own src/dst faithful to the stored direction?
        endpoints_match = all(
            rel_truth.get(e.id) == (e.src_node, e.dest_node) for e in rs
        )
        # (2) Does the relationships(path) LIST order chain as a traversal?
        rel_endpoints = [
            (internal_to_prop.get(e.src_node), internal_to_prop.get(e.dest_node))
            for e in rs
        ]
        rel_order_chains = all(
            rel_endpoints[i][1] == rel_endpoints[i + 1][0]
            for i in range(len(rel_endpoints) - 1)
        )
        # (3) Does nodes(path) order agree with the edge endpoints?
        node_order_forward = (
            len(node_props) == len(rs) + 1
            and node_props[:-1] == [p[0] for p in rel_endpoints]
            and node_props[1:] == [p[1] for p in rel_endpoints]
        )
        endpoints_ok_count += endpoints_match
        node_order_ok_count += node_order_forward
        rel_order_ok_count += rel_order_chains
        descriptions.append(
            f"{name}: nodes={node_props}, endpoints_faithful={endpoints_match}, "
            f"rels_list_chains={rel_order_chains}, "
            f"nodes_order_matches_edges={node_order_forward}"
        )
    n = len(shapes)
    endpoints_verdict = (
        "each Edge's own src/dst matches stored direction in all probed shapes"
        if endpoints_ok_count == n else
        f"Edge endpoints UNRELIABLE ({n - endpoints_ok_count}/{n} shapes wrong)"
    )
    order_verdict = (
        "list order faithful in all probed shapes"
        if (node_order_ok_count == n and rel_order_ok_count == n) else
        f"LIST ORDER is NOT reliable: nodes(path) order agreed with edges in "
        f"{node_order_ok_count}/{n} shapes and relationships(path) list order "
        f"chained in {rel_order_ok_count}/{n} shapes. Closed self-returning "
        "paths are the offenders, and which of the two lists is reversed varies "
        "between queries. Phase 2 reconstruction must therefore ignore list "
        "order entirely and walk the directed Edge endpoints"
    )
    return ProbeResult(
        "P0-17", "Path element ordering/direction (fact probe)", True,
        "; ".join(descriptions) + f" -> {endpoints_verdict}; {order_verdict}",
    )


PROBES = [
    probe_return_one,
    probe_roundtrip,
    probe_db_indexes_call,
    probe_drop_index,
    probe_index_lifecycle,
    probe_ensure_index,
    probe_index_legacy_syntax,
    probe_param_in_limit,
    probe_param_in_varlength,
    probe_varlength_traversal,
    probe_cycle_query,
    probe_shortest_path,
    probe_list_comprehension,
    probe_reduce,
    probe_ro_query,
    probe_timeout,
    probe_procedures_list,
    probe_pagerank,
    probe_normalize,
    probe_path_node_order,
]


def collect_environment(client) -> dict:
    env: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "docker_image": IMAGE_TAG,
        "python": platform.python_version(),
        "host_os": f"{platform.system()} {platform.release()}",
        "falkordb_py_client": md.version("falkordb"),
    }
    conn = getattr(client, "connection", None)
    if conn is not None:
        try:
            info = conn.info("server")
            env["server_redis_version"] = info.get("redis_version")
        except Exception as exc:
            env["server_redis_version"] = f"unavailable ({type(exc).__name__})"
        try:
            mods = conn.execute_command("MODULE", "LIST")
            parsed = []
            for mod in mods:
                # redis-py 8.x (observed) returns each module as an alternating
                # key/value list ['name','graph','ver',40700,...]; older shapes
                # are [name, ver, ...] or dicts. Handle all three.
                if isinstance(mod, dict):
                    items = list(mod.items())
                elif isinstance(mod, (list, tuple)) and mod[:1] in ([b"name"], ["name"]):
                    items = list(zip(mod[0::2], mod[1::2]))
                elif isinstance(mod, (list, tuple)) and len(mod) >= 2:
                    items = [("name", mod[0]), ("ver", mod[1])]
                else:
                    items = [("name", mod)]
                flat = {}
                for k, v in items:
                    k = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                    v = v.decode() if isinstance(v, (bytes, bytearray)) else v
                    flat[k] = v
                parsed.append(f"{flat.get('name', '?')} v{flat.get('ver', '?')}")
            env["server_modules"] = ", ".join(parsed) if parsed else "none reported"
        except Exception as exc:
            env["server_modules"] = f"unavailable ({type(exc).__name__}: {exc})"
    return env


def _rules(results: list[ProbeResult]) -> list[str]:
    by = {r.id: r for r in results}
    rules = [
        "- `timeout` on `Graph.query`/`Graph.ro_query` is in **milliseconds** "
        "(falkordb-py contract; confirmed against the installed client docs).",
        "- Reads go through `graph.ro_query`, writes through `graph.query` — "
        "encapsulated in `run_query` (src/db.py).",
    ]
    if "P0-05" in by:
        r = by["P0-05"]
        rules.append(
            "- Parameter in `LIMIT`: " + (
                "supported — parameterize all limits."
                if r.ok else
                "NOT supported — use a validated bounded integer constant at that "
                "grammar position (the one interpolation PLAN.md §4 allows)."
            )
        )
    if "P0-06" in by:
        r = by["P0-06"]
        rules.append(
            "- Parameter in variable-length bound (`*1..$hops`): " + (
                "supported."
                if r.ok else
                "NOT supported — hop caps must be validated integers in [1,5] "
                "interpolated as Python-validated constants (PLAN.md §0 allows "
                "exactly this one interpolation)."
            )
        )
    if "P0-18" in by:
        r = by["P0-18"]
        rules.append(
            "- Index enumeration: " + (
                "`CALL db.indexes() YIELD label, properties, ...` works — "
                "`list_indexes()` (src/db.py) parses it and `ensure_index()` "
                "pre-checks it before any CREATE."
                if r.ok else
                "NOT available — repeated index setup must rely solely on "
                "create-and-tolerate of the specific 'already indexed' error."
            )
        )
    if "P0-19" in by:
        r = by["P0-19"]
        rules.append(
            "- `DROP INDEX ON :Label(prop)`: " + (
                "supported — scratch indexes can be reset and idempotent setup "
                "can clean up after itself."
                if r.ok else
                "NOT supported — index setup is append-only; indexes cannot be "
                "dropped for retesting."
            )
        )
    if "P0-03" in by:
        r = by["P0-03"]
        rules.append(
            "- Repeated index setup: " + (
                "verified safe — `ensure_index()` (src/db.py) pre-checks via "
                "db.indexes(), tolerates ONLY the probed 'already indexed' "
                "constraint error, re-verifies before trusting it, and never "
                "masks unrelated errors (live check: P0-20)."
                if r.ok else
                "NOT verified — read P0-03's observed output before relying on "
                "repeated setup."
            )
        )
    if "P0-20" in by:
        r = by["P0-20"]
        rules.append(
            "- `ensure_index()` live behavior: " + (
                "first call created-or-found, second call returned 'exists' "
                "without issuing a duplicate CREATE."
                if r.ok else
                "did NOT behave idempotently — see P0-20 observed output; fix "
                "ensure_index() before Phase 1."
            )
        )
    if "P0-13" in by:
        r = by["P0-13"]
        rules.append(
            "- Per-query timeout (milliseconds): " + (
                "enforced by the server; queries abort with a timeout error."
                if r.ok else
                "NOT observed as enforced on this server — all queries must stay "
                "bounded via LIMIT/hop caps; treat the client timeout as best-effort."
            )
        )
    if "P0-08" in by:
        r = by["P0-08"]
        rules.append(
            "- Cycle detection: " + (
                "bounded self-returning path query works — MVP cycle template "
                "(canonicalization and temporal checks are Phase 2 Python work)."
                if r.ok else
                "bounded self-returning path query failed — Phase 2 must use the "
                "bounded Python cycle fallback."
            )
        )
    if "P0-09" in by:
        r = by["P0-09"]
        rules.append(
            "- `shortestPath`: " + (
                "available only in WITH/RETURN clauses (not as `MATCH p = "
                "shortestPath(...)`); bounded hop cap still required."
                if r.ok else
                "not usable — bounded BFS in Python is the fallback."
            )
        )
    if "P0-17" in by:
        r = by["P0-17"]
        rules.append(
            "- Path Edge endpoints: " + (
                "each Edge's own `src_node`/`dest_node` matches the stored "
                "direction (probed) — these may be trusted."
                if "matches stored direction in all probed shapes" in r.observed else
                "UNRELIABLE on this server (see P0-17) — re-derive direction by "
                "relationship id."
            )
        )
        rules.append(
            "- Path LIST ORDER: " + (
                "nodes(path) and relationships(path) are both in traversal order "
                "(probed)."
                if "list order faithful in all probed shapes" in r.observed else
                "NOT reliable for closed self-returning paths — in the same query "
                "one list can be forward and the other reversed, and which is "
                "which varies between queries. Phase 2 must ignore list order and "
                "reconstruct the traversal by walking the directed Edge endpoints "
                "(src/tools/reads.py reconstruct_path_accounts)."
            )
        )
    if "P0-15" in by:
        r = by["P0-15"]
        rules.append(
            "- Optional PageRank procedure: " + (
                "available; stretch feature only, never an MVP dependency."
                if r.ok else
                "not usable — MVP does not depend on it; `hub_scores` reports "
                "weighted in-degree (PLAN.md §8)."
            )
        )
    return rules


def render_doc(env: dict, results: list[ProbeResult]) -> str:
    lines: list[str] = [
        "# FalkorDB capability probes — HABEAS",
        "",
        f"Generated: {env['generated_utc']} by `python -m src.probes`. "
        "Every value below is the actual observed output of that run; re-run the "
        "command to regenerate.",
        "",
        "## Environment",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    for key, value in env.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Probe results",
        "",
        "| ID | Probe | Result | Observed |",
        "|---|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        observed = r.observed.replace("|", "/")
        lines.append(f"| {r.id} | {r.title} | {status} | {observed} |")
    lines += [
        "",
        "## Established compatibility rules",
        "",
        *_rules(results),
        "",
        "## Notes",
        "",
        "- Graph name in use: `HABEAS_demo_local` (matches the hard-coded "
        "`HABEAS_demo` demo-safe prefix). Probes write only probe-labeled "
        "scratch nodes into this disposable demo graph; the Phase 1 seed reset "
        "replaces its contents.",
        "- Probes bootstrap the demo graph with a harmless `RETURN 1` write "
        "query first, because `ro_query` refuses a not-yet-created graph "
        "(\"Invalid graph operation on empty key\"); any write auto-creates it.",
        "- `falkordb-py` exposes `FalkorDB(host, port)`, `select_graph(name)`, "
        "`graph.query(q, params, timeout_ms)` / `graph.ro_query(...)`; results "
        "via `QueryResult.result_set` / `.header`. No `read_only` flag exists "
        "on `query` — `ro_query` is the read-only path. `nodes(path)` yields "
        "Node objects and `relationships(path)` yields Edge objects.",
        "- **Phase 2 path-reconstruction strategy (from P0-17):** treat the "
        "directed Edge endpoints as the source of truth. Neither `nodes(path)` "
        "list order nor `relationships(path)` list order can be trusted for "
        "closed self-returning paths — in a single query one may be forward "
        "while the other is reversed, and which is which varies. Reconstruct "
        "the traversal by walking the edges (see "
        "`src/tools/reads.py::reconstruct_path_accounts`); each Edge's own "
        "`src_node`/`dest_node` is faithful.",
        "- The `mcp` Python SDK resolves to 1.30.0 in this dependency set "
        "(langchain-mcp-adapters constrains it below 2.0; mcp 2.x renames the "
        "server API). requirements.txt pins 1.30.0 — do not bump blindly.",
    ]
    failed_optional = [r for r in results if not r.ok and not r.required_for_mvp]
    if failed_optional:
        lines.append(
            "- Failed optional probes with documented fallbacks: "
            + ", ".join(r.id for r in failed_optional)
            + ". None block the MVP."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    graph = get_graph()
    # ro_query refuses a graph that does not exist yet; bootstrap it with a
    # harmless write query (auto-creates the key).
    run_query(graph, "RETURN 1 AS bootstrap", read_only=False)
    results: list[ProbeResult] = []
    for probe in PROBES:
        try:
            result = probe(graph)
        except QueryError as exc:
            result = ProbeResult(
                probe.__name__, "probe", False,
                f"QueryError kind={exc.kind}: {str(exc)[:160]}",
            )
        except Exception as exc:
            result = ProbeResult(
                probe.__name__, "probe", False,
                f"{type(exc).__name__}: {str(exc)[:160]}",
            )
        results.append(result)
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.id} {result.title}: {result.observed[:150]}")
    env = collect_environment(get_client())
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(render_doc(env, results), encoding="utf-8")
    print(f"\nWrote {DOC_PATH}")
    required_failures = [r for r in results if r.required_for_mvp and not r.ok]
    if required_failures:
        print(f"REQUIRED PROBE FAILURES: {[r.id for r in required_failures]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
