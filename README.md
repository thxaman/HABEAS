# HABEAS

**Graph-native financial-crime investigation on FalkorDB.**  
Track: **Agents That Act on Connected Data**


HABEAS investigates synthetic financial-transfer alerts by traversing connected accounts and transactions, gathering evidence through MCP tools, testing competing hypotheses, applying a deterministic intervention policy, simulating a freeze before acting, and independently verifying the resulting graph state.
try: (https://habeas.streamlit.app/)

> **Synthetic data only.** HABEAS does not connect to real accounts, payment systems, or external financial services. Any freeze is a simulated mutation inside a disposable demo graph. Findings are risk indicators for human review, not legal conclusions.

## Demo video

[Watch the HABEAS demo](https://github.com/user-attachments/assets/b12e6278-e677-4516-8ff6-01dbf8e1b9d8)

[Watch the HABEAS working demo](https://github.com/user-attachments/assets/e6084bd8-02c8-4184-907d-ff3336ab2dd1)



## Why a graph?

Suspicious activity may only become visible across multiple connected transfers: a directed cycle, fan-in/fan-out concentration, rapid pass-through movement, or customers sharing synthetic devices and identifiers. FalkorDB lets HABEAS query those relationships directly instead of treating every transaction as an isolated row.

## Architecture

```text
Streamlit UI / CLI
        |
        v
LangGraph investigation agent
(TRIAGE -> HYPOTHESES -> INVESTIGATE -> CHALLENGE -> DECIDE -> REPORT)
        | tool calls over MCP / JSON-RPC stdio
        v
HABEAS domain-specific MCP server
(approved, typed, bounded tools; no arbitrary LLM Cypher)
        |
        v
FalkorDB
(accounts, transactions, derived transfer graph, cases, evidence, actions)

Deterministic Policy v2 -> permits or declines sandbox action
Independent verifier   -> replays approved queries and checks recorded claims
```

The LLM proposes investigation tool choices; deterministic Python interprets evidence and enforces action eligibility. Agent graph access goes through the domain-specific MCP tools. The report verifier independently re-checks evidence without asking an LLM to validate its own claims.

## Graph model

The event ledger is the source of truth for transaction events:

```text
(Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account)
```

HABEAS also builds a **derived transfer-topology projection** for graph traversal:

```text
(Account)-[:TRANSFERRED_TO {
  count, total_amount, last_ts, active
}]->(Account)
```

This projection summarizes transaction events; it is not another set of transactions. Other nodes include `Customer`, `Device`, `Identifier`, `Alert`, `Case`, `Evidence`, `Simulation`, and `Action`, connected by ownership, identity, alert, case, evidence, simulation and audit relationships.

A simulated freeze sets the target account status to `frozen` and deactivates its incident derived routing edges. Historical transactions and evidence are retained. Prior cases are investigation leads, not inherited verdicts.

Full schema: [`docs/graph-model.md`](docs/graph-model.md).

## Cypher, graph traversal and algorithms

All agent query execution is selected from the approved query catalog in [`src/tools/query_catalog.py`](src/tools/query_catalog.py) by **query ID**. The agent cannot submit free-form Cypher. Values such as account and alert IDs are parameterized; traversals and result counts are bounded. FalkorDB's tested variable-length traversal-bound limitation is handled by validating the hop count as an integer before placing it into the query syntax.

Representative query IDs:

| Query ID | What it checks |
|---|---|
| `alert_context_v1` | Resolves an alert to its trigger transaction and concerned account |
| `trace_paths_out_v1` / `trace_paths_in_v1` | Bounded outgoing or incoming multi-hop paths |
| `detect_closed_paths_v1` | Bounded search for directed cycles |
| `fan_in_senders_v1` / `fan_out_receivers_v1` | Sender/receiver concentration |
| `hub_weighted_in_degree_v1` | Incoming volume over active transfer edges |
| `shared_devices_v1` / `shared_identifiers_v1` | Shared synthetic identity infrastructure |
| `txn_events_for_legs_v1` / `txn_max_ts_v1` | Timestamped events and observation-window anchor |
| `active_path_exists_v1` | Tests active reachability after a simulated action |
| `prior_cases_v1` | Retrieves earlier cases as leads to re-check |

The bounded traversal query family follows this Cypher pattern (illustrative shape; the application executes the catalog's validated template):

```cypher
MATCH p = (a:Account {id: $account_id})-[:TRANSFERRED_TO*1..4]->(b:Account)
RETURN p
LIMIT $limit
```

The upper hop bound shown is a validated integer literal because the tested FalkorDB dialect does not support the parameterized variable-length bound form used in the initial probe. Other values remain parameterized. Exact executable templates and validation rules live in the query catalog; the verifier dispatches by trusted query ID, not by executing arbitrary Cypher saved in a report or supplied by a model.

Implemented graph analysis includes:

- **Structural vs. temporal cycles:** a closed path is not called chronologically corroborated circulation unless timestamped transaction events chain around the ring for at least two full laps.
- **Bounded multi-hop tracing:** paths show graph connectivity; event-level timestamps are needed to make chronological claims.
- **Fan-in/fan-out:** counts distinct senders and receivers within a declared time window, with explicit handling for zero denominators.
- **Weighted in-degree:** incoming transfer volume over active edges; this is **not PageRank**.
- **Signal provenance:** signal families are compared by underlying transaction IDs so overlapping observations are not presented as independent corroboration.

See [`docs/falkordb-capabilities.md`](docs/falkordb-capabilities.md) for tested database capabilities and compatibility workarounds.

## MCP tools and investigation logic

Read tools include `alert_context`, `entity_profile`, `trace_funds`, `detect_cycle`, `fan_in_out`, `shared_identifiers`, `hub_scores`, `prior_cases`, and `active_path_exists`. Narrow action tools support case creation, evidence citation, entity flagging, simulation, guarded `freeze_account`, and `verify_freeze`.

Tool results use structured evidence envelopes that retain approved query IDs, parameters, row hashes, row counts, truncation indicators, derived facts and deterministic interpretations. Truncated results are marked as incomplete instead of being treated as exhaustive.

The bounded LangGraph workflow maintains competing hypotheses, chooses tools through MCP, records a tool trace, gathers evidence, decides whether evidence is sufficient, and produces a report. The LLM is not the source of factual findings, and its confidence is never sufficient to authorize an action.

## Policy v2: simulate before acting

Before a sandbox freeze, HABEAS computes a bounded simulation including recent direct outgoing transaction volume, downstream reachability, and routing edges that would be disabled. Historical transaction volume is **not** money that will be frozen or stranded.

Policy v2 requires, among other safeguards:

- a review-supported verdict and sufficient evidence;
- at least two distinct-provenance signal families that implicate the proposed target;
- independent transaction observations rather than multiple labels for the same transfers;
- no disqualifying material truncation;
- a completed, fresh simulation for the same case and account;
- explicit demo-mode and disposable-graph safety conditions.

If a gate fails, the action is declined and the blocking reasons are recorded. The sandbox action is followed by an independent read-back of account and routing state. Historical actions and their audit records are not silently rewritten when policy logic changes.

## Independent evidence verification

The report verifier is deterministic Python, not an LLM. For each recorded evidence source it resolves the approved query ID, validates parameters, replays the permitted read where possible, and compares canonicalized result hashes. It distinguishes outcomes such as:

- `reproduced_current_state`
- `verified_against_recorded_evidence`
- `changed_after_action`
- `mismatch`
- `unverifiable`

A claim that changed because an authorized sandbox action modified current topology is reported differently from a mismatch. Mutable metadata and evidence without a replayable source are not falsely presented as independently reproduced.

## Quick start

Requirements: Docker and Python 3.11+.

```bash
git clone https://github.com/thxaman/HABEAS.git
cd HABEAS
docker compose up -d
python -m venv .venv
```

Activate the virtual environment, then install dependencies and seed the disposable graph:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux / Git Bash (use this instead of the Windows command)
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
python -m src.seed.generate --seed 42 --reset-demo-graph
```

The reset command is explicitly opt-in and restricted to a demo-safe graph name. Configure a provider key in `.env` only if using real-provider mode; never commit `.env` or credentials. FakeLLM mode runs without an API key.

## FalkorDB Cloud / Streamlit Community Cloud

HABEAS also runs against a managed FalkorDB Cloud database. Set
`FALKORDB_USERNAME`, `FALKORDB_PASSWORD` (kept as a `SecretStr`, never logged
or rendered) and `FALKORDB_TLS=true`, plus `FALKORDB_HOST`/`FALKORDB_PORT`;
leaving them empty keeps the local Docker default. On Streamlit Community
Cloud the same values go into the app's Secrets, which are promoted into the
process environment and inherited by the MCP server subprocess.

The cloud database is seeded exactly once with the same guarded seeder
(`GRAPH_NAME=HABEAS_demo_showcase_s42 python -m src.seed.generate --seed 42
--reset-demo-graph` with the cloud variables set) — the app never reseeds on
startup, on a rerun, or on a page load. Full steps, including the secrets
format and verification, are in [`docs/falkordb-cloud.md`](docs/falkordb-cloud.md).

## Run the application and checks

```bash
# Analyst UI
streamlit run app.py

# Three-scenario deterministic CLI demo (FakeLLM)
python -m src.demo.run_demo

# Tests, FalkorDB capability probes and dependency consistency
python -m pytest -q
python -m src.probes
python -m pip check

# Build or verify an existing report
python -m src.report build --case CASE-ALR-0021 --reports-dir reports/demo
python -m src.report verify --case CASE-ALR-0021

# Reproducible evaluation
python -m eval.run --alerts all --seed 42 --mode fake
```

The UI loads alerts from the graph through MCP, displays the real neighbourhood and investigation evidence, streams workflow events, supports account/edge inspection, and exposes decision, simulation, action and claim-verification details. The reset remains a separately confirmed operation; selecting or investigating an alert does not automatically reseed the graph.

## Evaluation snapshot

Measured on the synthetic seed-42 benchmark using FakeLLM over the real MCP stdio transport:

| Metric | Result |
|---|---:|
| Alerts evaluated | 21 |
| Strict recall | 4/10 (40.0%) |
| False-positive rate on benign alerts | 0/11 (0.0%) |
| Strict planted pattern-family coverage | 2/4 (50.0%) |
| Investigation completion | 21/21 (100%) |
| Evidence claims passing verifier | 129/158 (81.6%) |
| Verifier mismatches | 0 |
| Unverifiable claims | 29 |

These are results on a small synthetic benchmark, not real-world detection performance. Strict recall and pattern-family coverage are limited; the abstention rate is high, and 29 verifier claims were unverifiable rather than proven. The measured case-memory comparison showed no tool-call savings, so HABEAS does not claim that memory improves tool efficiency.

Full details and methodology: [`eval/results.md`](eval/results.md) and [`eval/results.json`](eval/results.json).

## Project documentation

- [Architecture](docs/architecture.md)
- [Graph model](docs/graph-model.md)
- [FalkorDB capabilities and probes](docs/falkordb-capabilities.md)
- [FalkorDB Cloud deployment](docs/falkordb-cloud.md)
- [Evaluation results](eval/results.md)
- [Demo script](docs/demo-script.md)

## AI-assisted development

This project was built with an AI coding assistant under human direction. Test results, capability probes, and evaluation artifacts are recorded in the repository. The evaluation ground-truth manifest is isolated from the live investigation agent.

**Selected track:** Agents That Act on Connected Data.
