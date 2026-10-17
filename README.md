# HABEAS

**Graph-native fraud investigation on FalkorDB.**
Selected track: **Agents That Act on Connected Data**.

HABEAS is a synthetic-data investigation agent that follows a financial-crime
alert through an evidence-driven investigation over a graph, records the case,
simulates an intervention before acting, and applies a **simulated** account
freeze only when a deterministic policy says the evidence supports it — then
independently verifies the resulting graph state.

> **Synthetic data only.** HABEAS never connects to real accounts, payment
> systems, or external financial services. The only "action" is a simulated
> mutation inside an explicitly configured disposable demo graph. Findings are
> **risk indicators for human review, not legal conclusions**, and the
> repository makes no claim of guilt.

---

## 1. Problem, user, and why a graph

**Problem.** Fraud alerts rarely make sense one transaction at a time. The
signal is in the *shape* of the money movement: funds circulating around a
ring, many senders concentrating into one cash-out hub, a pass-through account
that forwards deposits within minutes, or customers who share a device or an
IP identifier. These are relationship patterns, and answering them one hop at
a time loses the story.

**Intended user.** A financial-crime analyst / investigator triaging an alert
queue. The analyst needs a defensible, auditable record of *what was checked*,
*what the evidence showed*, and *what action (if any) is justified* — not a
black-box score.

**Why a graph.** Rings, fan-in/fan-out, shared infrastructure and multi-hop
paths are first-class traversals in a graph. FalkorDB lets us run **bounded**
variable-length path queries and keep the historical event ledger alongside a
derived transfer-topology projection, with case memory persisted as graph
entities (a repeated alert can be linked to prior cases as *leads*).

---

## 2. Architecture

```text
                           ┌───────────────────────────────────────────┐
                           │             Analyst / CLI / UI             │
                           └───────────────────────┬───────────────────┘
                                                   │
                     python -m src.demo.run_demo    │   python -m src.report build|verify
                                                   ▼
        ┌──────────────────────────── Investigation agent (LangGraph) ───────────────────────────┐
        │  TRIAGE → FORM_HYPOTHESES → INVESTIGATE → UPDATE_EVIDENCE → CHALLENGE → DECIDE → REPORT │
        │  • bounded state machine with a hard tool budget                                       │
        │  • LLM proposes tool choices; DETERMINISTIC Python computes every interpretation,       │
        │    signal family and verdict (LLM confidence is never treated as calibrated)            │
        └───────────────┬──────────────────────────────────────────────────────────┬────────────┘
                        │ tool calls (JSON-RPC over stdio)                          │ evidence + envelopes
                        ▼                                                           ▼
        ┌──────────────────────────── MCP server (src/tools/server.py) ────────────────────────────┐
        │  read tools  : alert_context, entity_profile, trace_funds, detect_cycle, fan_in_out,       │
        │                shared_identifiers, hub_scores, prior_cases, active_path_exists             │
        │  action tools: create_case, cite_evidence, flag_entity, close_case, simulate_freeze,       │
        │                freeze_account, verify_freeze   (opt-in; sandbox graph only)                │
        │  • approved, parameterized query templates only — never free-form LLM Cypher               │
        │  • canonical evidence envelope: query ids, params, row hashes, truncation flags           │
        └───────────────┬──────────────────────────────────────────────────────────┬────────────────┘
                        │ Cypher (ro_query for reads)                              │ narrow write templates
                        ▼                                                           ▼
        ┌──────────────────────────────────────── FalkorDB ─────────────────────────────────────────┐
        │  event ledger: (Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account)                  │
        │  derived projection: (Account)-[:TRANSFERRED_TO {count,total_amount,last_ts,active}]->(…)  │
        │  memory: Case / Evidence / Simulation / Action  (+ Alert, Customer, Device, Identifier)     │
        └──────────────────────────────────────────────────────────────────────────────────────────┘
                        ▲                                                           ▲
                        │ read-only replay of approved templates                     │ independent re-query
        ┌───────────────┴───────────────────────┐               ┌──────────────────┴───────────────────┐
        │  Deterministic policy (Policy v2)     │               │  Independent verifier                 │
        │  decides freeze eligibility from      │               │  (src/report/verifier.py):            │
        │  persisted evidence — never from an   │               │  re-checks every material claim        │
        │  LLM's confidence                     │               │  against the graph; no LLM             │
        └───────────────────────────────────────┘               └───────────────────────────────────────┘
```

The separation is deliberate: the **agent proposes**, deterministic Python
**validates**, a narrow sandbox **action** executes, and an independent
**verifier** re-checks the recorded claims. Graph access from the agent only
ever happens through the MCP boundary.

---

## 3. Graph schema and key relationships

Node labels: `Customer`, `Account`, `Transaction`, `Device`, `Identifier`,
`Alert`, and the write-created `Case`, `Evidence`, `Simulation`, `Action`.

Core relationships:

- `(Customer)-[:OWNS]->(Account)`
- `(Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account)` — the event
  ledger (source of truth for money movement)
- `(Customer)-[:USES_DEVICE]->(Device)`, `(Customer)-[:HAS_IDENTIFIER]->(Identifier)`
- `(Alert)-[:TRIGGERED_BY]->(Transaction)`, `(Alert)-[:CONCERNS]->(Account)`
- `(Account)-[:TRANSFERRED_TO {count,total_amount,last_ts,active}]->(Account)`
  — a **derived** aggregate of the ledger (never an extra set of transactions)
- `(Case)-[:ABOUT]->(Account|Customer)`, `(Case)-[:HAS_EVIDENCE]->(Evidence)`,
  `(Evidence)-[:SUPPORTS]->(…)`, `(Case)-[:HAS_SIMULATION]->(Simulation)`,
  `(Case)-[:HAS_ACTION]->(Action)`, `(Case)-[:REUSES]->(older:Case)`

A simulated freeze sets `Account.status = 'frozen'` and marks the account's
incident `TRANSFERRED_TO` edges `active = false`; **historical transactions and
evidence are never deleted**. `REUSES` links prior cases as *leads* only —
verdicts are never inherited.

Full property lists: [`docs/graph-model.md`](docs/graph-model.md).

---

## 4. Key Cypher templates and algorithms

All queries come from an approved parameterized catalog
(`src/tools/query_catalog.py`) selected by **query id** — never free-form Cypher
from the model. Traversal bounds are validated integers (1–5), every query is
bounded with explicit row limits, and reads go through `ro_query`.

Representative read templates:

| Query id | Purpose |
|---|---|
| `alert_context_v1` | resolve an Alert id to its trigger transaction and concerned account |
| `trace_paths_out_v1` / `trace_paths_in_v1` | bounded `[:TRANSFERRED_TO*1..<<hops>>]` path traversal |
| `detect_closed_paths_v1` | bounded closed-path search for rings |
| `fan_in_senders_v1` / `fan_out_receivers_v1` | concentration of senders/receivers |
| `hub_weighted_in_degree_v1` | weighted in-degree over **active** incoming edges |
| `shared_devices_v1` / `shared_identifiers_v1` | shared-infrastructure association |
| `entity_profile_*_v1`, `active_path_exists_v1`, `prior_cases_v1` | profile, active-route check, case memory |
| `txn_events_for_legs_v1`, `txn_max_ts_v1` | timestamped evidence for corroboration and window anchoring |

Algorithms (plain Python in `src/tools/algorithms.py` and
`src/agent/analysis.py`, independent of the LLM):

- **Structural vs temporally-corroborated cycles** — a closed path is reported
  structurally; it is only called *temporally corroborated* when the ts-ordered
  events chain around the ring for ≥2 laps. Path direction is reconstructed
  from relationship endpoints, never from `nodes(path)` order.
- **Hub score** — honest label: weighted in-degree over the active projection,
  **not** PageRank.
- **Signal families** — `circulation`, `fan_in_out`, `rapid_forwarding` (plus
  association evidence from shared identifiers), used by the policy to require
  *independent* corroboration.
- **Policy v2** (`src/agent/policy.py`) — requires a supported verdict, ≥2
  signal families, ≥2 **distinct-provenance** families that implicate the
  target, no material truncation, a fresh complete simulation for the same
  case+account, and the sandbox opt-in. An LLM confidence value never
  authorizes an action.

---

## 5. Installation and clean-clone setup

Requires **Docker** and **Python 3.11+** (developed on 3.14).

```bash
git clone <your-fork-url> HABEAS && cd HABEAS
docker compose up -d                       # starts pinned FalkorDB v4.22.1 on :6379
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                       # then edit .env (see §6)
python -m src.seed.generate --seed 42 --reset-demo-graph
python -m pytest -q                        # full test suite
python -m src.demo.run_demo                # three-scenario CLI demo (FakeLLM)
```

The demo reseeds its **own** disposable graph (`HABEAS_demo_showcase_s42`); it
never resets `HABEAS_demo_local`, the evaluation graphs, or an arbitrary graph.

---

## 6. Environment variables and key handling

Copy `.env.example` to `.env`. `.env` is gitignored and must never be committed.

| Variable | Meaning |
|---|---|
| `FALKORDB_HOST` / `FALKORDB_PORT` | FalkorDB endpoint (default `localhost:6379`) |
| `GRAPH_NAME` | target graph; must start with the demo-safe prefix `HABEAS_demo` for resets/actions |
| `HABEAS_DEMO_MODE` | `true` enables simulated actions on a demo-safe graph |
| `HABEAS_ENABLE_ACTIONS` | separate opt-in that registers the write-capable action tools |
| `FALKORDB_QUERY_TIMEOUT_MS` | per-query timeout in **milliseconds** |
| `LLM_MODE` | `fake` (deterministic, no key) or `real` |
| `LLM_PROVIDER` | `openai` or `groq` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI credentials (real mode) |
| `GROQ_API_KEY` / `GROQ_MODEL` | Groq credentials (real mode); Groq's OpenAI-compatible endpoint is used |
| `LLM_MODEL` | generic model fallback when the provider-specific var is empty |
| `TOOL_BUDGET` | hard per-investigation tool-call budget |

**Safe key handling.** Secrets live in `SecretStr`, are never logged, and never
appear in trace or error output. A missing key produces a clear configuration
error naming the *variable* (never the value). `OPENAI_API_KEY` is never
repurposed for Groq. `LLM_MODE=fake` needs no credentials at all, so the whole
test suite and the deterministic demo run offline.

---

## 7. Exact commands

```bash
# Database
docker compose up -d
docker compose down

# Seed the deterministic synthetic demo graph (seed 42)
python -m src.seed.generate --seed 42 --reset-demo-graph

# Tests / probes / dependency check
python -m pytest -q
python -m src.probes          # FalkorDB capability probes → docs/falkordb-capabilities.md
python -m pip check

# Demo: three investigations over the real MCP stdio server
python -m src.demo.run_demo                      # deterministic FakeLLM (recordable backup)
python -m src.demo.run_demo --scenarios A,C      # a subset
python -m src.demo.run_demo --mode real          # bounded live provider demo (needs creds)
python -m src.demo.run_demo --mode auto          # real when configured, else fake
python -m src.demo.run_demo --json-out logs/demo/last_run.json

# Reports: build and independently verify
python -m src.report build  --case CASE-ALR-0021 --reports-dir reports/demo
python -m src.report verify --case CASE-ALR-0021

# Evaluation harness (Phase 7): full deterministic run
python -m eval.run --alerts all --seed 42 --mode fake

# Analyst UI (Phase 8.5)
streamlit run app.py        # then open http://localhost:8501
```

The demo prints, per scenario: the MCP tools selected, concise evidence
summaries, provenance overlap/distinctness, the policy decision and its
blocking reasons, the actual action result, the counterfactual simulation, the
post-action verification, and the independent report verification status.

### Analyst UI — investigation workspace

```bash
streamlit run app.py        # then open http://localhost:8501
```

A three-panel financial-intelligence workstation over the existing backend —
not a chatbot and not a row of pre-scripted demo buttons.

1. **Alert inbox (left)** — loaded *from the graph* over MCP
   (`alert_inbox_v1`), searchable by id/reason/account, showing each alert's
   reason, trigger transaction, concerned account, and — where a prior run
   persisted one — the case verdict and action status. The three established
   scenarios are marked ★ **recommended examples**, but they are ordinary rows
   from that query; no alert id branches into a manufactured result.
2. **Evidence graph (centre, the primary visual)** — a real interactive
   directed graph built from the bounded `alert_neighborhood` query
   (accounts + aggregate `TRANSFERRED_TO` edges with active flags + bounded
   transaction events) merged with whatever the investigation recorded. Click
   a **node** to inspect its real in/out relationships and transactions; click
   an **edge** to inspect the aggregate transfer row and its underlying
   transactions. Encoding never relies on colour alone: solid cyan = active
   routing, dashed amber = deactivated routing, solid teal = corroborated
   circulation, dotted slate = structural connectivity only. A legend and a
   **Reset view** control are provided.
3. **Activity timeline (right)** — the agent's real workflow as it happens:
   stage transitions, the tool it selected and the question it addresses, a
   summary of what each tool returned, hypothesis re-evaluation, provenance
   and target-relevance checks, the action gate, simulation, action outcome
   and verification. Events come from the backend's `ProgressEmitter`; a
   broken consumer can never break a run. An investigation runs in a
   background thread, so the timeline updates live rather than appearing only
   at the end.

Below that: **Decision** (canonical verdict, Policy v2 checks, blocking
reasons, distinct-provenance families, target relevance, policy version),
**Audit** (counterfactual simulation with its honest metric semantics —
historical volume is explicitly *not* labelled as frozen funds — the recorded
action id and post-action checks, the unchanged account state for a decline),
and **claim-level verification** (reproduced / changed-after-action / mismatch
/ unverifiable) with a viewable, downloadable report.

Operator safety invariants (all regression-tested in `tests/test_ui.py` and
`tests/test_ui_workspace.py`):

- the target graph is fixed to the disposable demo graph — the UI exposes **no
  graph-name field and no Cypher input** (its only text box is the inbox
  search);
- the demo graph is reseeded **only** by the explicit, confirmed control —
  never on a rerun, page load, alert selection, or investigation;
- switching alerts clears the previous investigation's graph overlay,
  timeline, decision and report, and reloads the graph for the new alert;
- all investigation, inbox, neighbourhood and alert-resolution access goes
  through **MCP**; the UI never reimplements the agent, Policy v2, the
  simulation, the report builder or the verifier;
- API keys are never collected or displayed; FakeLLM always works without
  credentials, and a missing real-provider key surfaces as a clear
  configuration error.

---

## 8. Phase 7 evaluation — measured results

Full artifacts: [`eval/results.md`](eval/results.md) and `eval/results.json`.
Command: `python -m eval.run --alerts all --seed 42 --mode fake --alt-seed 43
--real-smoke`. Dataset: the seeded manifest — **21 alerts (10 suspicious, 11
benign across 4 planted pattern families)**. Mode: FakeLLM over the real MCP
stdio server, Policy v2, and the independent verifier.

Outcome mapping: `supported_for_review` = strict positive;
`suspicious_activity` = a separate *unconfirmed* tier (counted as neither
positive nor benign); `likely_benign` = benign;
`insufficient_evidence`/`incomplete_investigation` = abstention; harness or
provider failures are a visible `failed` tier that **stays in every
denominator**.

| Metric | Result (seed 42) |
|---|---|
| Alert recall — strict (`suspicion_supported_for_review`) | **4/10 (40.0%)** |
| Alert recall — lenient (+ `suspicious_activity`) | **4/10 (40.0%)** |
| False-positive rate on benign (strict) | **0/11 (0.0%)** |
| Inconclusive / abstention rate (overall) | **15/21 (71.4%)** — suspicious 6/10, benign 9/11 |
| Investigation completion rate | **21/21 (100%)**; tool-budget exhaustion 0/21 |
| Failure rate (harness/provider) | **0/21 (0.0%)** |
| Pattern-family coverage (strict) | **2/4 (50%)** — `cycle` 3/3, `cycle_rapid` 1/1; `fan_in_out` 0/3, `rapid_shared` 0/3 |
| Pattern detection (expected signal families observed) | **9/10 (90%)** |
| Action eligibility (Policy v2 allowed, of attempted) | **1/21 (4.8%)** |
| Sandbox actions applied / declined | **1 applied**, **20 declined** (95.2%) |
| Independent verifier — claims passing | **129/158 (81.6%)**; 0 mismatches, 29 unverifiable, 21/21 cases checked |
| Tool calls per investigation | mean **7.5**, median **7**, p90 **8** (n=21) |

Reproducibility: a second sweep on a rebuilt graph produced an **identical**
semantic digest. Alternative seed 43: recall 4/10, FPR 0/11, coverage 2/4
(separate denominators, never combined). Bounded real-LLM smoke: completed
(provider `groq`, model `openai/gpt-oss-120b`).

**Multi-hop ablation (controlled).** For the circulation alerts, a one-hop
profile (`entity_profile`) and a ≤3-account traversal (`detect_cycle max_len=3`)
**cannot express the closed ring** (0/4 each), while the bounded 4-account
traversal **temporally corroborates it on 4/4** — the concrete contribution of
bounded multi-hop graph traversal over a one-hop baseline, on the same MCP
boundary and the same 90-day window.

### Case memory: no measured tool-call saving

A controlled paired comparison (memory enabled vs `prior_cases` withheld, on
identically seeded graphs with an identical warm-up) is in `eval/results.json`
→ `memory_comparison`. Across **5 pairs**, memory-on made **42** calls and
memory-off **37** — a net **+5 calls** (the `prior_cases` lookups) that avoided
**0** recorded steps. **HABEAS therefore does not claim that graph memory
reduces tool calls.** Prior cases function as *leads to re-check*, and in this
deterministic configuration they cost calls rather than save them.

---

## 9. Known limitations

- **Strict recall is 40%.** Two of four planted families (`fan_in_out`,
  `rapid_shared`) are detected as signals but do not reach a
  review-supported verdict in the deterministic baseline; they are reported as
  abstentions, not as benign.
- **Inconclusive rate is high (71.4%).** Most alerts end in
  `insufficient_evidence` — an explicit "not enough evidence", which is the
  honest outcome here rather than a false accusation.
- **Pattern coverage is 2/4 families** under the strict (all-alerts-in-family)
  definition.
- **Verifier: 81.6% of claims pass.** The remaining 29 records are
  **unverifiable**, not mismatched (0 mismatches): they lack a preserved
  snapshot or a replayable approved query, and the verifier reports them
  explicitly instead of hiding them.
- **Case memory saves no tool calls** in the measured comparison (see §8).
- Metrics are measured on a **synthetic** benchmark and do not generalize.
  Findings are risk indicators for review, never legal conclusions.

---

## 10. Demo video and submission assets


https://github.com/user-attachments/assets/b12e6278-e677-4516-8ff6-01dbf8e1b9d8




---

## 11. AI coding-assistant disclosure

This project was built with an AI coding assistant (an agentic LLM coding
tool) under human direction, phase by phase, against the contract in
`PLAN.md`. Every phase gate was executed and its **actual** output recorded;
capability claims are backed by the saved probe results and evaluation
artifacts. No test output, detection metric, performance number, or
verification success in this repository is fabricated. The assistant did not
have access to the evaluation ground truth while investigating (enforced and
regression-tested).

Selected track: **Agents That Act on Connected Data**.
