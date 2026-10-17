# HABEAS graph model

Synthetic, pseudonymous data only (PLAN.md §5). The live demo graph is
`HABEAS_demo_local` (matches the hard-coded `HABEAS_demo` demo-safe prefix).

## Node types

| Label | Properties |
|---|---|
| `Customer` | `id`, `display_name` (Faker-generated fictional name), `created_at` |
| `Account` | `id`, `status` (`active`; `frozen` only after a Phase 5 simulated freeze), `balance`, `currency` (all `USD`), `created_at` |
| `Transaction` | `id`, `amount` (2-decimal float), `currency`, `ts` (`YYYY-MM-DDTHH:MM:SSZ`), `channel`, `status` |
| `Device` | `id`, `first_seen`, `last_seen` |
| `Identifier` | `id`, `kind` (`email` / `phone` / `ip`), `value_hash` — the raw value is never stored; hashes are SHA-256 prefixes of clearly synthetic values (`*.invalid` e-mails, `555-01xx` fictional numbers, RFC 5737 `198.51.100.0/24` IPs) |
| `Alert` | `id`, `transaction_id`, `reason`, `created_at`, `triage_status` (`open` at seed time) |

Phase 5 adds four write-created labels (still synthetic-only):

| Label | Properties |
|---|---|
| `Case` | `id` (`CASE-<alert_id>`), `alert_id`, `title`, `status`, `verdict` (canonical), `severity`, `summary`, `created_at`, `updated_at` |
| `Evidence` | `id` (`<case_id>:<local_id>`, `-v2…` on re-citation), `claim`, `stance` (`supporting`/`contradicting`/`unresolved`), `query_id` (approved catalog only), `source_tool`, `params_json`, `rows_json` (the preserved normalized snapshot, or `[]` for hash-only citations), `derived_json`, `result_hash` (sha256 over all normalized rows, canonicalized order-independently as of Phase 6), `row_count`, `truncated`, `interpretation`, `evidence_kind` (`immutable_history`/`current_state`/`captured_snapshot`), `case_id`, `captured_at`; Phase 6 audit fields: `columns_json` (column names of the preserved snapshot), `interpolation_json` (validated `{"<<token>>": int}` bounds used to build the query), `template_sha256` (server-computed fingerprint of the approved catalog template at capture time); **Phase 6.1 claim-level fields**: `sources_json` (the complete per-invocation source manifest: an ordered array of `{query_id, params, interpolation, result_hash, row_count, truncated, template_sha256, role}` entries — `role` is server-computed: `window_anchor` for `txn_max_ts_v1`, else `claim_source`), `source_count` (number of recorded sources), `required_source_queries_json` (claim-material source query ids, i.e. every source except the window anchor). The top-level `query_id`/`params_json`/`result_hash`/… mirror `sources[0]` for compatibility; a record without `sources_json` is a pre-6.1 legacy citation |
| `Simulation` | `id`, `case_id`, `account_id`, `graph_fingerprint` (account state + bounded active topology), `metrics_json`, `window_days`, `hops`, `simulation_complete`, `status` (`complete`/`incomplete`), `created_at` |
| `Action` | `id` (`ACT-<case_id>-<account_id>`), `case_id`, `account_id`, `action` (`freeze_account`), `status` (`applied`), `reason`, `requested_by`, `simulation_id`, `evidence_ids_json`, `policy_json`, `before_state_json`, `after_state_json`, `created_at` |

`flagged_for_review`, `flag_reason`, `flag_risk`, `flag_case_id`, `flagged_at`
are set on `Account`/`Customer` nodes by `flag_entity` (review requested, not
guilt established). Frozen accounts carry `frozen_at`, `frozen_reason`,
`frozen_case_id`.

## Relationships

- `(Customer)-[:OWNS]->(Account)`
- `(Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account)` — the event
  ledger; the only source of truth for money movement.
- `(Customer)-[:USES_DEVICE]->(Device)`
- `(Customer)-[:HAS_IDENTIFIER]->(Identifier)`
- `(Alert)-[:TRIGGERED_BY]->(Transaction)`
- `(Alert)-[:CONCERNS]->(Account)` — added in Phase 1: PLAN §6 requires each
  alert to link to its triggering transaction *and* relevant account; recorded
  in PLAN §5 per the no-silent-changes rule.

Phase 5 relationships (all write-created by `src/tools/actions.py`; the only
sanctioned writer):

- `(Case)-[:ABOUT]->(Account|Customer)`
- `(Case)-[:HAS_EVIDENCE]->(Evidence)` and `(Evidence)-[:SUPPORTS]->(Account|Customer)`
- `(Case)-[:HAS_SIMULATION]->(Simulation)`
- `(Case)-[:HAS_ACTION]->(Action)`
- `(Case)-[:REUSES]->(older:Case)` — a **lead** link added when an earlier
  case contributed to the investigation. Verdicts are never inherited along
  this edge; the new case re-derives its own from fresh evidence.

### Derived transfer-topology projection

`(Account)-[:TRANSFERRED_TO {count, total_amount, last_ts, active}]->(Account)`

A **derived summary** of the event ledger, computed once at seed time — never
an additional set of transactions. Invariant (tested in
`tests/test_seed.py::test_projection_matches_event_ledger`): for every edge,
`count`, `total_amount`, and `last_ts` equal the aggregate of the underlying
`SENT → RECEIVED_BY` events. Metrics must never add raw transaction amounts
and aggregate edge amounts in the same figure (PLAN §5/§8).

When a simulated freeze is applied (Phase 5), the account's `status` becomes
`frozen` and its incident `TRANSFERRED_TO` edges are marked `active=false`;
historical events and evidence are never deleted. Active-routing queries
filter to `active = true` and non-frozen accounts.

## Indexes

Created via the Phase 0 `ensure_index()` helper (pre-checks
`CALL db.indexes()`, repeat-safe): `Customer.id`, `Account.id`,
`Transaction.id`, `Alert.id`, `Case.id`, `Evidence.id`, `Simulation.id`,
`Action.id`.

## Seeded population (seed 42, actual counts)

356 customers, 412 accounts, 1,603 transactions over the fixed synthetic
window 2026-01-01 → 2026-03-31 (90 days), 501 devices, 661 identifiers,
21 alerts, 1,540 `TRANSFERRED_TO` aggregate edges.

## Planted patterns (detectable by deterministic tests)

1. **`cycle`** — 4 dedicated accounts circulating funds in 3 rounds, legs
   strictly ordered inside each round (temporally corroborated: the
   ts-ordered events chain around the ring). Detected structurally via
   bounded `TRANSFERRED_TO` traversal (live Cypher, P0-08 template) and by
   the bounded Python cycle finder with rotation canonicalization.
2. **`fan_in_out`** — 8 low-activity source accounts feed 2 intermediaries;
   each intermediary forwards ~85% of intake to a single cash-out hub.
   Verified live: hub in-edges are exactly the two intermediaries at the 0.85
   forwarded share; weighted-in-degree rank of the hub is 9.
3. **`rapid_shared`** — 5 in/out chains (feeder → receiver → forwarder →
   exit) with each hop 10–90 minutes after the previous; the two receiver
   customers share one device **and** one IP identifier.
4. **`cycle_rapid`** (append-only addition, corrective Phase 5 pass; alert
   `ALR-0021`) — a slow ring (internal legs 150–240 minutes apart, far outside
   the 90-minute rapid window) plus exactly ONE rapid pass-through on the
   alerted ring member (feeder → member → exit, 25–45 minutes, ≥0.9 forwarded
   share). The circulation observations (ring legs) and the rapid observations
   (the two pass-through transactions) share **zero** transaction ids, and the
   trigger transaction implicates the alerted member — the pattern the
   corrected provenance-aware policy can legitimately authorize against. The
   pattern is appended last so every earlier id, amount, timestamp and
   manifest entry is unchanged.

Benign noise includes deliberate traps: an account receiving 8+ salary-like
credits (naive fan-in fires, but there is no forwarding concentration and no
rapid movement), three high-volume legitimate accounts, and an innocently
shared device. The ground-truth manifest lists them as
`tempting_false_positives`.

## Ground-truth isolation

The manifest (`src/seed/ground_truth.json`, Pydantic-validated, gitignored,
regenerated by the seeder) records per-alert family/label/members/signals.
It is **never** written onto graph nodes and never read by the agent, its
prompts, or MCP tools — only tests and the evaluation harness read it.
`tests/test_seed.py` enforces: no ground-truth-like node properties in the
graph, no `ground_truth` references in `src/tools|agent|report|demo`, and the
gitignore entry.

## Reset safety

`python -m src.seed.generate --seed 42 --reset-demo-graph` is the only
destructive path. The guard (tested) refuses: any graph name without the
`HABEAS_demo` prefix (even with the flag), and any non-empty graph without
the explicit flag. `DETACH DELETE` exists nowhere in agent-facing code.
