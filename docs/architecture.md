# HABEAS architecture

Companion to `README.md` §2–§4. Everything here reflects the implemented code;
capability claims are backed by `docs/falkordb-capabilities.md` (probes) and
`eval/results.json` (measured results).

## Component map

| Component | Module | Responsibility | Trust level |
|---|---|---|---|
| Config | `src/config.py` | env-backed immutable settings; demo-safety predicates | trusted config |
| DB layer | `src/db.py` | `run_query`/`ro_query`, timeouts (ms), structured errors, normalization, safe index helpers | trusted |
| Seed | `src/seed/*` | deterministic synthetic graph + validated manifest + reset guard | trusted setup |
| Query catalog | `src/tools/query_catalog.py` | approved parameterized templates (read + narrow write), validators | trusted |
| Read tools | `src/tools/reads.py`, `src/tools/algorithms.py` | the 9 read tools + bounded algorithms | agent-reachable (read-only) |
| Action tools | `src/tools/actions.py` | case/evidence/flags/simulation/freeze/verify via approved write templates | agent-reachable (sandbox only) |
| MCP server | `src/tools/server.py` | exposes the tools over stdio; read vs action separation | boundary |
| MCP client | `src/agent/mcp_client.py` | long-lived stdio session, envelope-preserving calls | boundary |
| Agent | `src/agent/{loop,state,analysis,llm,prompts,policy,trace}.py` | bounded LangGraph investigation; deterministic interpretation + policy | proposes only |
| Report | `src/report/{builder,verifier}.py` | Markdown report from the persisted case; independent re-verification | verification |
| Evaluation | `eval/*` | harness, metrics, memory comparison, ablation (reads manifest) | eval only |
| Demo | `src/demo/*` | three-scenario CLI demo + browser hint | presentation |

## Data model (write-created memory)

```text
(Alert {id,reason})─[:TRIGGERED_BY]→(Transaction)         (already seeded)
        └──────────[:CONCERNS]──────→(Account)

Investigating an alert produces:
    (Case {id=CASE-<alert>, verdict, severity, summary, status})
        ├─[:ABOUT]────────→(Account|Customer)

        ├─[:HAS_EVIDENCE]─→(Evidence {id=<case>:<n>, claim, stance,
        │                             query_id, params_json, rows_json,
        │                             result_hash, truncated,
        │                             sources_json, interpretation, …})
        │                       └─[:SUPPORTS]→(Account|Customer)

        ├─[:HAS_SIMULATION]→(Simulation {graph_fingerprint, metrics_json,
        │                                window_days, hops, simulation_complete})

        ├─[:HAS_ACTION]────→(Action {id=ACT-<case>-<acct>, status="applied",
        │                            reason, requested_by, simulation_id,
        │                            policy_json, before_state_json,
        │                            after_state_json})

        └─[:REUSES]────────→(older:Case)        # lead link only; no inheritance
```

A freeze sets `Account.status='frozen'` and marks incident `TRANSFERRED_TO`
edges `active=false`. **No node or relationship is ever deleted.** The
`Action.policy_json` preserves the exact eligibility decision (checks,
blocking reasons, provenance families, qualifying evidence ids) at the moment
of the action.

## Investigation state machine

```text
TRIAGE → FORM_HYPOTHESES → INVESTIGATE → UPDATE_EVIDENCE → CHALLENGE → DECIDE → REPORT
             ▲                                      │
             └────────── (evidence insufficient) ───┘
```

Bounded by a hard tool budget; each node returns an explicit state update
(LangGraph does not persist in-place mutation). Hypothesis discipline: a
provisional `suspicious_activity` is never treated as confirmed fraud; a failed
or truncated call is recorded as *unresolved*, never as evidence of absence.

## Why the boundaries matter

- The agent can only reach the graph through MCP tool schemas; it cannot read
  the manifest, write arbitrary Cypher, or delete data.
- Interpretations, signal families and verdicts are computed in Python, never
  authored by the model.
- Writes are confined to `src/tools/actions.py` behind the sandbox opt-in and
  approved templates; the only mutation is the simulated freeze.
- The verifier re-queries the graph with the same approved templates and reports
  mismatches and unverifiable records rather than asserting success.