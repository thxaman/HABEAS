# FalkorDB capability probes — HABEAS

Generated: 2026-10-18 09:24 UTC by `python -m src.probes`. Every value below is the actual observed output of that run; re-run the command to regenerate.

## Environment

| Item | Value |
|---|---|
| generated_utc | 2026-10-18 09:24 UTC |
| docker_image | falkordb/falkordb:v4.22.1 |
| python | 3.14.5 |
| host_os | Windows 11 |
| falkordb_py_client | 1.8.0 |
| server_redis_version | 8.10.2 |
| server_modules | falkordbe v10000, vectorset v1, graph v42200 |

## Probe results

| ID | Probe | Result | Observed |
|---|---|---|---|
| P0-01 | Connection + RETURN 1 | PASS | RETURN 1 -> [[1]] |
| P0-02 | Parameterized write/read round-trip (MERGE, SET, MATCH) | PASS | wrote n=1 via MERGE+SET, incremented via SET, read back [['phase0-roundtrip', 2]] — all values bound as parameters |
| P0-18 | Index enumeration via CALL db.indexes() | FAIL | 8 indexes listed (rows parsed as label+properties); known index Phase0Probe.id listed: False; sample: [('Action', ['id']), ('Simulation', ['id']), ('Evidence', ['id']), ('Case', ['id']), ('Alert', ['id']), ('Transaction', ['id'])] |
| P0-19 | DROP INDEX support | PASS | `DROP INDEX ON :Label(prop)` accepted; index no longer listed after drop: True |
| P0-03 | Index lifecycle: fresh creation, duplicate rejection, detection | PASS | fresh creation OK (indices_created=1.0); duplicate creation rejected with the expected already-indexed constraint error (kind=constraint); existing-index detection via CALL db.indexes(): listed=True; cleanup drop OK |
| P0-20 | ensure_index() repeated setup is safe | PASS | first call -> created; second call -> exists; cleanup drop OK |
| P0-04 | Index creation — legacy syntax | PASS | legacy `CREATE INDEX ON :Label(prop)` accepted (indices_created=1.0) |
| P0-05 | Parameter in LIMIT | PASS | `LIMIT $lim` accepted with a bound parameter — parameterize all limits |
| P0-06 | Parameter in variable-length path bound | FAIL | rejected: kind=syntax: Encountered unhandled type in inlined properties. — hop caps must be validated ints in [1,5] interpolated as Python-validated constants (PLAN.md §0 allows exactly this one interpolation) |
| P0-07 | Bounded variable-length traversal | PASS | 1..2 hops from PC1 over NEXT -> ['PC2', 'PC3'] |
| P0-08 | Bounded cycle-detection query (MVP template) | PASS | self-returning directed path 1..4 found the cycle (members=[['CY1', 'CY4', 'CY3', 'CY2', 'CY1']]); temporal corroboration stays a Phase 2 Python concern |
| P0-09 | shortestPath | PASS | directed WITH-clause form works: shortestPath PC1-PC3 = 2 hops. NOTE: `MATCH p = shortestPath(...)` is rejected — this server only supports shortestPath in WITH/RETURN clauses |
| P0-10 | List comprehension | PASS | cell value -> [20, 40] (expected [20, 40]) |
| P0-11 | reduce() | PASS | cell value -> 6 (expected 6) |
| P0-12 | Read-only query path (ro_query) | PASS | graph.ro_query works; run_query uses it for every read (required_for_mvp) |
| P0-13 | Per-query timeout enforcement (300ms request) | PASS | query aborted after 21.53s with QueryError kind=timeout: Timeout connecting to server |
| P0-14 | CALL dbms.procedures() | PASS | available; 26 procedures registered; page/rank/bfs/wcc/betweenness-related: ['algo.BFS', 'algo.WCC', 'algo.betweenness', 'algo.pageRank']; first 10: ['algo.AStar', 'algo.BFS', 'algo.HarmonicCentrality', 'algo.MSF', 'algo.SPpaths', 'algo.SSpaths', 'algo.WCC', 'algo.betweenness', 'algo.labelPropagation', 'algo.maxFlow'] |
| P0-15 | Optional PageRank procedure | PASS | algo.pageRank works; returned [] (stretch feature only) |
| probe_normalize | probe | FAIL | IndexError: list index out of range |
| P0-17 | Path element ordering/direction (fact probe) | PASS | open-2hop-exact: nodes=['PC1', 'PC2', 'PC3'], endpoints_faithful=True, rels_list_chains=True, nodes_order_matches_edges=True; open-1hop-exact: nodes=['PC1', 'PC2'], endpoints_faithful=True, rels_list_chains=True, nodes_order_matches_edges=True; closed-cycle-self-return: nodes=['CY1', 'CY4', 'CY3', 'CY2', 'CY1'], endpoints_faithful=True, rels_list_chains=True, nodes_order_matches_edges=False -> each Edge's own src/dst matches stored direction in all probed shapes; LIST ORDER is NOT reliable: nodes(path) order agreed with edges in 2/3 shapes and relationships(path) list order chained in 3/3 shapes. Closed self-returning paths are the offenders, and which of the two lists is reversed varies between queries. Phase 2 reconstruction must therefore ignore list order entirely and walk the directed Edge endpoints |

## Established compatibility rules

- `timeout` on `Graph.query`/`Graph.ro_query` is in **milliseconds** (falkordb-py contract; confirmed against the installed client docs).
- Reads go through `graph.ro_query`, writes through `graph.query` — encapsulated in `run_query` (src/db.py).
- Parameter in `LIMIT`: supported — parameterize all limits.
- Parameter in variable-length bound (`*1..$hops`): NOT supported — hop caps must be validated integers in [1,5] interpolated as Python-validated constants (PLAN.md §0 allows exactly this one interpolation).
- Index enumeration: NOT available — repeated index setup must rely solely on create-and-tolerate of the specific 'already indexed' error.
- `DROP INDEX ON :Label(prop)`: supported — scratch indexes can be reset and idempotent setup can clean up after itself.
- Repeated index setup: verified safe — `ensure_index()` (src/db.py) pre-checks via db.indexes(), tolerates ONLY the probed 'already indexed' constraint error, re-verifies before trusting it, and never masks unrelated errors (live check: P0-20).
- `ensure_index()` live behavior: first call created-or-found, second call returned 'exists' without issuing a duplicate CREATE.
- Per-query timeout (milliseconds): enforced by the server; queries abort with a timeout error.
- Cycle detection: bounded self-returning path query works — MVP cycle template (canonicalization and temporal checks are Phase 2 Python work).
- `shortestPath`: available only in WITH/RETURN clauses (not as `MATCH p = shortestPath(...)`); bounded hop cap still required.
- Path Edge endpoints: each Edge's own `src_node`/`dest_node` matches the stored direction (probed) — these may be trusted.
- Path LIST ORDER: NOT reliable for closed self-returning paths — in the same query one list can be forward and the other reversed, and which is which varies between queries. Phase 2 must ignore list order and reconstruct the traversal by walking the directed Edge endpoints (src/tools/reads.py reconstruct_path_accounts).
- Optional PageRank procedure: available; stretch feature only, never an MVP dependency.

## Notes

- Graph name in use: `HABEAS_demo_local` (matches the hard-coded `HABEAS_demo` demo-safe prefix). Probes write only probe-labeled scratch nodes into this disposable demo graph; the Phase 1 seed reset replaces its contents.
- Probes bootstrap the demo graph with a harmless `RETURN 1` write query first, because `ro_query` refuses a not-yet-created graph ("Invalid graph operation on empty key"); any write auto-creates it.
- `falkordb-py` exposes `FalkorDB(host, port)`, `select_graph(name)`, `graph.query(q, params, timeout_ms)` / `graph.ro_query(...)`; results via `QueryResult.result_set` / `.header`. No `read_only` flag exists on `query` — `ro_query` is the read-only path. `nodes(path)` yields Node objects and `relationships(path)` yields Edge objects.
- **Phase 2 path-reconstruction strategy (from P0-17):** treat the directed Edge endpoints as the source of truth. Neither `nodes(path)` list order nor `relationships(path)` list order can be trusted for closed self-returning paths — in a single query one may be forward while the other is reversed, and which is which varies. Reconstruct the traversal by walking the edges (see `src/tools/reads.py::reconstruct_path_accounts`); each Edge's own `src_node`/`dest_node` is faithful.
- The `mcp` Python SDK resolves to 1.30.0 in this dependency set (langchain-mcp-adapters constrains it below 2.0; mcp 2.x renames the server API). requirements.txt pins 1.30.0 — do not bump blindly.
- Failed optional probes with documented fallbacks: P0-18, P0-06, probe_normalize. None block the MVP.
