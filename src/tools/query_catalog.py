"""Approved query catalog and tool-input validators (PLAN.md §4/§7).

- Every template is a constant string; tool code selects by ``query_id``.
- All values are bound parameters. The ONLY interpolation is ``<<hops>>`` /
  ``<<max_len>>`` tokens replaced with a Python-validated integer in the
  documented range (FalkorDB rejects parameters in variable-length bounds —
  probe P0-06); ``LIMIT`` accepts parameters (probe P0-05).
- Path predicates use ``relationships(p)``/``nodes(p)`` — iterating a bound
  relationship variable directly fails on this server (probed).
- Path direction is reconstructed from Edge endpoints, never from
  ``nodes(path)`` order (P0-17: unreliable for closed cycles).
"""
from __future__ import annotations

import re
from typing import Any

from src.tools.envelopes import ROW_LIMIT_DEFAULT, ROW_LIMIT_MAX, ToolValidationError

_HOP_MIN, _HOP_MAX = 1, 5
_CYCLE_LEN_MIN, _CYCLE_LEN_MAX = 2, 5
_WINDOW_MIN, _WINDOW_MAX = 1, 365
_AMOUNT_MAX = 1_000_000_000.0
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# ``<<hops>>`` is replaced (post-validation) with an integer literal.
_QUERY_TEMPLATES: dict[str, str] = {
    "txn_max_ts_v1": "MATCH (t:Transaction) RETURN max(t.ts) AS max_ts",
    # Phase 4 prerequisite: `entity_profile` only resolves Account/Customer
    # (verified: Alert ids return found=false), so the agent needs one narrow
    # read to turn an alert id into its involved entities and trigger txn.
    "alert_context_v1": (
        "MATCH (al:Alert {id: $alert_id})-[:TRIGGERED_BY]->(t:Transaction) "
        "OPTIONAL MATCH (al)-[:CONCERNS]->(conc:Account) "
        "OPTIONAL MATCH (src:Account)-[:SENT]->(t) "
        "OPTIONAL MATCH (t)-[:RECEIVED_BY]->(dst:Account) "
        "RETURN al.id AS alert_id, al.reason AS reason, al.created_at AS created_at, "
        "al.triage_status AS triage_status, t.id AS txn_id, t.ts AS ts, "
        "t.amount AS amount, conc.id AS concerned_account_id, "
        "src.id AS sender_account_id, dst.id AS receiver_account_id LIMIT 1"
    ),
    "entity_profile_lookup_v1": (
        "MATCH (n) WHERE (n:Account OR n:Customer) AND n.id = $entity_id "
        "RETURN labels(n)[0] AS kind, properties(n) AS props LIMIT 1"
    ),
    "entity_profile_flows_v1": (
        "MATCH (a:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(b:Account) "
        "WHERE (a.id = $entity_id OR b.id = $entity_id) AND t.ts >= $cutoff "
        "RETURN a.id AS src, b.id AS dst, count(t) AS txn_count, "
        "sum(t.amount) AS total_amount ORDER BY src, dst LIMIT $limit"
    ),
    "entity_profile_owner_v1": (
        "MATCH (c:Customer)-[:OWNS]->(a:Account {id: $entity_id}) "
        "RETURN c.id AS customer_id, c.display_name AS display_name "
        "ORDER BY customer_id LIMIT $limit"
    ),
    "entity_profile_owned_accounts_v1": (
        "MATCH (c:Customer {id: $entity_id})-[:OWNS]->(a:Account) "
        "RETURN a.id AS account_id, a.status AS status "
        "ORDER BY account_id LIMIT $limit"
    ),
    # Phase 5: entity existence check for flag_entity. Unlike account_state_v1
    # this resolves Account *or* Customer ids, because a case may flag either.
    "entity_exists_v1": (
        "MATCH (n) WHERE (n:Account OR n:Customer) AND n.id = $entity_id "
        "RETURN n.id AS entity_id, labels(n)[0] AS kind LIMIT 1"
    ),
    "trace_paths_out_v1": (
        "MATCH p=(a:Account {id: $account_id})-[:TRANSFERRED_TO*1..<<hops>>]->(b) "
        "WHERE all(rel IN relationships(p) WHERE rel.active = true "
        "AND rel.total_amount >= $min_amount AND rel.last_ts >= $cutoff) "
        "AND all(n IN nodes(p) WHERE n.status <> 'frozen') "
        "RETURN relationships(p) AS rels, nodes(p) AS nodes, length(p) AS hops "
        "ORDER BY hops, b.id LIMIT $limit"
    ),
    "trace_paths_in_v1": (
        "MATCH p=(a:Account {id: $account_id})<-[:TRANSFERRED_TO*1..<<hops>>]-(b) "
        "WHERE all(rel IN relationships(p) WHERE rel.active = true "
        "AND rel.total_amount >= $min_amount AND rel.last_ts >= $cutoff) "
        "AND all(n IN nodes(p) WHERE n.status <> 'frozen') "
        "RETURN relationships(p) AS rels, nodes(p) AS nodes, length(p) AS hops "
        "ORDER BY hops, b.id LIMIT $limit"
    ),
    "txn_events_for_legs_v1": (
        "UNWIND $legs AS leg "
        "MATCH (a:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(b:Account) "
        "WHERE a.id = leg.src AND b.id = leg.dst AND t.ts >= $cutoff "
        "RETURN a.id AS src, b.id AS dst, t.id AS txn_id, t.ts AS ts, "
        "t.amount AS amount ORDER BY ts, txn_id LIMIT $limit"
    ),
    "detect_closed_paths_v1": (
        "MATCH p=(a:Account {id: $account_id})-[:TRANSFERRED_TO*1..<<max_len>>]->(a) "
        "WHERE all(rel IN relationships(p) WHERE rel.active = true) "
        "AND all(n IN nodes(p) WHERE n.status <> 'frozen') "
        "RETURN relationships(p) AS rels, nodes(p) AS nodes, length(p) AS hops "
        "ORDER BY hops LIMIT $limit"
    ),
    "shared_devices_v1": (
        "MATCH (c1:Customer {id: $customer_a_id})-[:USES_DEVICE]->(d:Device)"
        "<-[:USES_DEVICE]-(c2:Customer {id: $customer_b_id}) "
        "RETURN d.id AS device_id, d.first_seen AS first_seen, "
        "d.last_seen AS last_seen ORDER BY device_id LIMIT $limit"
    ),
    "shared_identifiers_v1": (
        "MATCH (c1:Customer {id: $customer_a_id})-[:HAS_IDENTIFIER]->(i:Identifier)"
        "<-[:HAS_IDENTIFIER]-(c2:Customer {id: $customer_b_id}) "
        "RETURN i.id AS identifier_id, i.kind AS kind "
        "ORDER BY identifier_id LIMIT $limit"
    ),
    "fan_in_senders_v1": (
        "MATCH (a:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(b:Account {id: $account_id}) "
        "WHERE t.ts >= $cutoff "
        "RETURN a.id AS sender, count(t) AS txn_count, sum(t.amount) AS total_amount "
        "ORDER BY total_amount DESC, sender LIMIT $limit"
    ),
    "fan_out_receivers_v1": (
        "MATCH (a:Account {id: $account_id})-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(b:Account) "
        "WHERE t.ts >= $cutoff "
        "RETURN b.id AS receiver, count(t) AS txn_count, sum(t.amount) AS total_amount "
        "ORDER BY total_amount DESC, receiver LIMIT $limit"
    ),
    "hub_weighted_in_degree_v1": (
        "MATCH (a:Account)-[r:TRANSFERRED_TO]->(b:Account) "
        "WHERE r.active = true AND r.last_ts >= $cutoff "
        "RETURN b.id AS account_id, sum(r.total_amount) AS weighted_in_amount, "
        "count(DISTINCT a) AS distinct_senders, sum(r.count) AS in_event_count "
        "ORDER BY weighted_in_amount DESC, account_id LIMIT $limit"
    ),
    "prior_cases_v1": (
        "MATCH (c:Case)-[:ABOUT]->(e) WHERE e.id = $entity_id "
        "RETURN c.id AS case_id, c.status AS status, c.verdict AS verdict, "
        "c.severity AS severity, c.updated_at AS updated_at "
        "ORDER BY updated_at DESC, case_id LIMIT $limit"
    ),
    # ---- Phase 8.5 UI: data-driven alert inbox + neighbourhood subgraph ----
    # The alert inbox is a bounded, read-only listing of every seeded alert
    # with its concerned account, trigger transaction, and — where an earlier
    # investigation already ran — the case verdict and latest action status.
    # The UI never hard-codes case ids; it renders rows this query returns.
    "alert_inbox_v1": (
        "MATCH (al:Alert) "
        "OPTIONAL MATCH (al)-[:CONCERNS]->(acc:Account) "
        "OPTIONAL MATCH (al)-[:TRIGGERED_BY]->(t:Transaction) "
        "OPTIONAL MATCH (c:Case) WHERE c.alert_id = al.id "
        "OPTIONAL MATCH (c)-[:HAS_ACTION]->(act:Action) "
        "RETURN al.id AS alert_id, al.reason AS reason, "
        "al.created_at AS created_at, al.triage_status AS triage_status, "
        "acc.id AS concerned_account_id, t.id AS txn_id, t.ts AS ts, "
        "t.amount AS amount, c.id AS case_id, c.verdict AS case_verdict, "
        "c.status AS case_status, act.id AS action_id, "
        "act.status AS action_status "
        "ORDER BY al.id LIMIT $limit"
    ),
    # Bounded neighbourhood of the alerted account: the reachable accounts in
    # each direction (``<<hops>>`` is a validated integer, never a parameter —
    # FalkorDB rejects parameters in variable-length path bounds) plus every
    # ACTIVE-or-inactive TRANSFERRED_TO edge incident to that set, so the UI
    # can distinguish active from deactivated routing.
    "neighborhood_out_accounts_v1": (
        "MATCH (a:Account {id: $account_id})"
        "-[:TRANSFERRED_TO*1..<<hops>>]->(b:Account) "
        "RETURN DISTINCT b.id AS account_id LIMIT $limit"
    ),
    "neighborhood_in_accounts_v1": (
        "MATCH (a:Account {id: $account_id})"
        "<-[:TRANSFERRED_TO*1..<<hops>>]-(b:Account) "
        "RETURN DISTINCT b.id AS account_id LIMIT $limit"
    ),
    "neighborhood_edges_v1": (
        "MATCH (s:Account)-[r:TRANSFERRED_TO]->(e:Account) "
        "WHERE s.id IN $account_ids OR e.id IN $account_ids "
        "RETURN s.id AS src, e.id AS dst, r.count AS count, "
        "r.total_amount AS total_amount, r.last_ts AS last_ts, "
        "r.active AS active "
        "ORDER BY total_amount DESC, src, dst LIMIT $limit"
    ),
    "active_path_min_hops_v1": (
        "MATCH p=(a:Account {id: $source_account_id})"
        "-[:TRANSFERRED_TO*1..<<max_hops>>]->(b:Account {id: $target_account_id}) "
        "WHERE all(rel IN relationships(p) WHERE rel.active = true) "
        "AND all(n IN nodes(p) WHERE n.status <> 'frozen') "
        "RETURN length(p) AS hops ORDER BY hops LIMIT 1"
    ),
    # ---- Phase 5: case memory, simulation and the sandbox action ----------
    # Every template below is a narrow, purpose-built, fully parameterized
    # operation. There is deliberately NO generic write / arbitrary-Cypher tool.
    "case_lookup_v1": (
        "MATCH (c:Case {id: $case_id}) RETURN properties(c) AS props LIMIT 1"
    ),
    "case_upsert_v1": (
        "MERGE (c:Case {id: $case_id}) "
        "SET c.alert_id = $alert_id, c.title = $title, c.status = $status, "
        "c.verdict = $verdict, c.severity = $severity, c.summary = $summary, "
        "c.created_at = $created_at, c.updated_at = $updated_at "
        "RETURN c.id AS case_id, c.created_at AS created_at, c.updated_at AS updated_at"
    ),
    "case_about_v1": (
        "MATCH (c:Case {id: $case_id}) "
        "MATCH (n) WHERE (n:Account OR n:Customer) AND n.id = $entity_id "
        "MERGE (c)-[:ABOUT]->(n) RETURN n.id AS entity_id"
    ),
    "case_about_entities_v1": (
        "MATCH (c:Case {id: $case_id})-[:ABOUT]->(n) "
        "RETURN n.id AS entity_id ORDER BY entity_id LIMIT $limit"
    ),
    "case_close_v1": (
        "MATCH (c:Case {id: $case_id}) "
        "SET c.status = $status, c.verdict = $verdict, c.summary = $summary, "
        "c.updated_at = $updated_at "
        "RETURN c.id AS case_id, c.status AS status, c.verdict AS verdict"
    ),
    "evidence_upsert_v1": (
        "MERGE (e:Evidence {id: $evidence_id}) "
        "SET e.claim = $claim, e.stance = $stance, e.query_id = $query_id, "
        "e.params_json = $params_json, e.rows_json = $rows_json, "
        "e.derived_json = $derived_json, "
        "e.result_hash = $result_hash, e.row_count = $row_count, "
        "e.truncated = $truncated, e.interpretation = $interpretation, "
        "e.evidence_kind = $evidence_kind, e.source_tool = $source_tool, "
        "e.captured_at = $captured_at, e.case_id = $case_id, "
        # Phase 6 audit fidelity: preserved columns, the recorded interpolation
        # values (so the verifier can rebuild the exact bounded query), and the
        # catalog template fingerprint captured at citation time.
        "e.columns_json = $columns_json, "
        "e.interpolation_json = $interpolation_json, "
        "e.template_sha256 = $template_sha256, "
        # Phase 6.1 claim-level coverage: the complete per-sub-query source
        # manifest, its declared size, and the claim-material query ids.
        "e.sources_json = $sources_json, "
        "e.source_count = $source_count, "
        "e.required_source_queries_json = $required_source_queries_json "
        "RETURN e.id AS evidence_id"
    ),
    "evidence_supports_v1": (
        "MATCH (e:Evidence {id: $evidence_id}) "
        "MATCH (n) WHERE (n:Account OR n:Customer) AND n.id = $entity_id "
        "MERGE (e)-[:SUPPORTS]->(n) RETURN e.id AS evidence_id"
    ),
    "evidence_lookup_v1": (
        "MATCH (e:Evidence {id: $evidence_id}) "
        "RETURN e.result_hash AS result_hash, e.source_count AS source_count, "
        "e.sources_json AS sources_json LIMIT 1"
    ),
    "case_link_evidence_v1": (
        "MATCH (c:Case {id: $case_id}) MATCH (e:Evidence {id: $evidence_id}) "
        "MERGE (c)-[:HAS_EVIDENCE]->(e) RETURN e.id AS evidence_id"
    ),
    "case_reuses_v1": (
        "MATCH (c:Case {id: $case_id}) MATCH (older:Case {id: $older_case_id}) "
        "WHERE older.id <> $case_id "
        "MERGE (c)-[:REUSES]->(older) RETURN older.id AS older_case_id"
    ),
    "entity_flag_v1": (
        "MATCH (n) WHERE (n:Account OR n:Customer) AND n.id = $entity_id "
        "SET n.flagged_for_review = true, n.flag_reason = $reason, "
        "n.flag_risk = $risk, n.flag_case_id = $case_id, n.flagged_at = $now "
        "RETURN n.id AS entity_id, n.flagged_for_review AS flagged"
    ),
    "simulation_lookup_v1": (
        "MATCH (s:Simulation {id: $simulation_id}) "
        "RETURN properties(s) AS props LIMIT 1"
    ),
    "simulation_for_case_v1": (
        "MATCH (c:Case {id: $case_id})-[:HAS_SIMULATION]->(s:Simulation) "
        "WHERE s.account_id = $account_id "
        "RETURN properties(s) AS props ORDER BY s.created_at DESC LIMIT 1"
    ),
    "simulation_upsert_v1": (
        "MERGE (s:Simulation {id: $simulation_id}) "
        "SET s.case_id = $case_id, s.account_id = $account_id, "
        "s.graph_fingerprint = $graph_fingerprint, s.metrics_json = $metrics_json, "
        "s.window_days = $window_days, s.hops = $hops, "
        "s.simulation_complete = $simulation_complete, s.status = $status, "
        "s.created_at = $created_at "
        "RETURN s.id AS simulation_id"
    ),
    "simulation_link_v1": (
        "MATCH (c:Case {id: $case_id}) MATCH (s:Simulation {id: $simulation_id}) "
        "MERGE (c)-[:HAS_SIMULATION]->(s) RETURN s.id AS simulation_id"
    ),
    "account_state_v1": (
        "MATCH (a:Account {id: $account_id}) "
        "RETURN a.id AS account_id, a.status AS status, "
        "a.balance AS balance, a.currency AS currency LIMIT 1"
    ),
    "account_incident_active_edges_v1": (
        "MATCH (x:Account)-[r:TRANSFERRED_TO]->(y:Account) "
        "WHERE (x.id = $account_id OR y.id = $account_id) AND r.active = true "
        "RETURN x.id AS src, y.id AS dst, r.count AS count, "
        "r.total_amount AS total_amount, r.last_ts AS last_ts "
        "ORDER BY src, dst LIMIT $limit"
    ),
    "active_topology_v1": (
        "MATCH (x:Account)-[r:TRANSFERRED_TO]->(y:Account) "
        "WHERE r.active = true "
        "RETURN x.id AS src, y.id AS dst, r.count AS count, "
        "r.total_amount AS total_amount, r.last_ts AS last_ts "
        "ORDER BY src, dst LIMIT $limit"
    ),
    "direct_outgoing_in_window_v1": (
        "MATCH (a:Account {id: $account_id})-[:SENT]->(t:Transaction)"
        "-[:RECEIVED_BY]->(:Account) WHERE t.ts >= $cutoff "
        "RETURN count(t) AS txn_count, sum(t.amount) AS total_amount, "
        "max(t.ts) AS last_ts LIMIT 1"
    ),
    "freeze_account_v1": (
        "MATCH (a:Account {id: $account_id}) "
        "WHERE a.status <> 'frozen' "
        "SET a.status = 'frozen', a.frozen_at = $now, "
        "a.frozen_reason = $reason, a.frozen_case_id = $case_id "
        "RETURN a.id AS account_id, a.status AS status, a.frozen_case_id AS case_id"
    ),
    "deactivate_incident_edges_v1": (
        "MATCH (x:Account)-[r:TRANSFERRED_TO]->(y:Account) "
        "WHERE (x.id = $account_id OR y.id = $account_id) AND r.active = true "
        "SET r.active = false, r.deactivated_at = $now, r.deactivated_by = $case_id "
        "RETURN count(r) AS deactivated"
    ),
    "action_upsert_v1": (
        "MERGE (a:Action {id: $action_id}) "
        "SET a.case_id = $case_id, a.account_id = $account_id, "
        "a.action = $action, a.status = $status, a.reason = $reason, "
        "a.requested_by = $requested_by, a.simulation_id = $simulation_id, "
        "a.evidence_ids_json = $evidence_ids_json, a.policy_json = $policy_json, "
        "a.before_state_json = $before_state_json, "
        "a.after_state_json = $after_state_json, "
        "a.verification_json = $verification_json, a.created_at = $created_at "
        "RETURN a.id AS action_id, a.status AS status"
    ),
    "action_for_case_v1": (
        "MATCH (c:Case {id: $case_id})-[:HAS_ACTION]->(a:Action) "
        "WHERE a.account_id = $account_id AND a.action = $action "
        "RETURN properties(a) AS props ORDER BY a.created_at DESC LIMIT 1"
    ),
    "action_link_v1": (
        "MATCH (c:Case {id: $case_id}) MATCH (a:Action {id: $action_id}) "
        "MERGE (c)-[:HAS_ACTION]->(a) RETURN a.id AS action_id"
    ),
    "case_actions_v1": (
        "MATCH (c:Case {id: $case_id})-[:HAS_ACTION]->(a:Action) "
        "RETURN a.id AS action_id, a.account_id AS account_id, "
        "a.action AS action, a.status AS status, a.created_at AS created_at "
        "ORDER BY created_at LIMIT $limit"
    ),
    "case_evidence_v1": (
        "MATCH (c:Case {id: $case_id})-[:HAS_EVIDENCE]->(e:Evidence) "
        "RETURN e.id AS evidence_id, e.claim AS claim, e.stance AS stance, "
        "e.query_id AS query_id, e.source_tool AS source_tool, "
        "e.params_json AS params_json, e.derived_json AS derived_json, "
        "e.result_hash AS result_hash, e.row_count AS row_count, "
        "e.truncated AS truncated, e.interpretation AS interpretation, "
        "e.evidence_kind AS evidence_kind, e.captured_at AS captured_at, "
        "e.columns_json AS columns_json, "
        "e.interpolation_json AS interpolation_json, "
        "e.template_sha256 AS template_sha256, "
        "e.case_id AS case_id, "
        "e.sources_json AS sources_json, "
        "e.source_count AS source_count, "
        "e.required_source_queries_json AS required_source_queries_json, "
        "e.rows_json AS rows_json "
        "ORDER BY e.captured_at LIMIT $limit"
    ),
    # Phase 6: report/verifier reads. Both are narrow, bounded, read-only.
    "simulation_for_case_any_v1": (
        "MATCH (c:Case {id: $case_id})-[:HAS_SIMULATION]->(s:Simulation) "
        "RETURN properties(s) AS props ORDER BY s.created_at DESC LIMIT 1"
    ),
    "case_reuses_list_v1": (
        "MATCH (c:Case {id: $case_id})-[:REUSES]->(older:Case) "
        "RETURN older.id AS older_case_id, older.verdict AS verdict, "
        "older.status AS status, older.updated_at AS updated_at "
        "ORDER BY older.id LIMIT $limit"
    ),
}

# Tokens that may ONLY be replaced with Python-validated integers.
_INTERPOLATION_TOKENS = ("<<hops>>", "<<max_len>>", "<<max_hops>>")

# Phase 5 templates that mutate the graph. They may be executed ONLY by
# src/tools/actions.py, only against the configured sandbox graph, and only
# after the deterministic eligibility policy permits the action.
WRITE_QUERY_IDS = frozenset({
    "case_upsert_v1",
    "case_about_v1",
    "case_close_v1",
    "case_reuses_v1",
    "evidence_upsert_v1",
    "evidence_supports_v1",
    "case_link_evidence_v1",
    "entity_flag_v1",
    "simulation_upsert_v1",
    "simulation_link_v1",
    "freeze_account_v1",
    "deactivate_incident_edges_v1",
    "action_upsert_v1",
    "action_link_v1",
})


def is_write_query(query_id: str) -> bool:
    return query_id in WRITE_QUERY_IDS


def get_template(query_id: str) -> str:
    try:
        return _QUERY_TEMPLATES[query_id]
    except KeyError as exc:
        raise KeyError(f"unknown query_id {query_id!r}; not in approved catalog") from exc


def bind_hop_bound(query_id: str, token: str, value: int) -> str:
    """Return the template with ``token`` replaced by a validated integer."""
    if token not in _INTERPOLATION_TOKENS:
        raise ToolValidationError(f"illegal interpolation token {token!r}")
    if token not in get_template(query_id):
        raise ToolValidationError(f"template {query_id} has no {token} token")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(f"{token} bound must be an int, got {type(value).__name__}")
    return get_template(query_id).replace(token, str(value))


# ---- input validators ------------------------------------------------------


def validate_entity_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ToolValidationError(
            f"{field} must match {_ID_RE.pattern} (1-64 chars, alphanumeric-ish); "
            f"got {value!r}"
        )
    return value


def validate_hop_count(value: Any, field: str = "hops") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(f"{field} must be an integer, got {type(value).__name__}")
    if not (_HOP_MIN <= value <= _HOP_MAX):
        raise ToolValidationError(f"{field} must be in [{_HOP_MIN}, {_HOP_MAX}], got {value}")
    return value


def validate_cycle_len(value: Any, field: str = "max_len") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(f"{field} must be an integer, got {type(value).__name__}")
    if not (_CYCLE_LEN_MIN <= value <= _CYCLE_LEN_MAX):
        raise ToolValidationError(
            f"{field} must be in [{_CYCLE_LEN_MIN}, {_CYCLE_LEN_MAX}], got {value}"
        )
    return value


def validate_limit(value: Any = ROW_LIMIT_DEFAULT) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(f"limit must be an integer, got {type(value).__name__}")
    if not (1 <= value <= ROW_LIMIT_MAX):
        raise ToolValidationError(f"limit must be in [1, {ROW_LIMIT_MAX}], got {value}")
    return value


def validate_window_days(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(
            f"window_days must be an integer, got {type(value).__name__}"
        )
    if not (_WINDOW_MIN <= value <= _WINDOW_MAX):
        raise ToolValidationError(
            f"window_days must be in [{_WINDOW_MIN}, {_WINDOW_MAX}], got {value}"
        )
    return value


def validate_amount(value: Any, field: str = "min_amount") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolValidationError(f"{field} must be numeric, got {type(value).__name__}")
    value = float(value)
    if not (0.0 <= value <= _AMOUNT_MAX):
        raise ToolValidationError(f"{field} must be in [0, {_AMOUNT_MAX}], got {value}")
    return value


def validate_min_senders(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(
            f"min_senders must be an integer, got {type(value).__name__}"
        )
    if not (1 <= value <= 100):
        raise ToolValidationError(f"min_senders must be in [1, 100], got {value}")
    return value


def validate_direction(value: Any) -> str:
    if value not in ("in", "out"):
        raise ToolValidationError(f"direction must be 'in' or 'out', got {value!r}")
    return value


_SEVERITIES = ("low", "medium", "high", "critical")


def validate_severity(value: Any, field: str = "severity") -> str:
    if value not in _SEVERITIES:
        raise ToolValidationError(
            f"{field} must be one of {list(_SEVERITIES)}, got {value!r}"
        )
    return value


# ---- Phase 6: parameter contract, template fingerprints, interpolation -----
#
# The verifier must never execute Cypher reconstructed from stored evidence
# text. Instead it resolves the query id against THIS catalog, checks the
# stored parameters against the argument contract of the approved template,
# rebuilds interpolated bounds from recorded integer values only, and re-runs
# the approved template through the read-only database abstraction.

_PARAM_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")
_INTERP_RE = re.compile(r"<<[a-z_]+>>")


def _validate_cutoff(value: Any, field: str = "cutoff") -> str:
    if value == "":
        return ""  # no transactions at all: the window matches nothing
    if not isinstance(value, str):
        raise ToolValidationError(f"{field} must be an ISO-8601 UTC string")
    try:
        from datetime import datetime

        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ToolValidationError(
            f"{field} must be 'YYYY-MM-DDTHH:MM:SSZ', got {value!r}"
        ) from exc
    return value


def _validate_text(value: Any, field: str = "text") -> str:
    if not isinstance(value, str) or not value:
        raise ToolValidationError(f"{field} must be a non-empty string")
    if len(value) > 100_000:
        raise ToolValidationError(f"{field} exceeds the 100000-char cap")
    return value


def _validate_bool(value: Any, field: str = "flag") -> bool:
    if not isinstance(value, bool):
        raise ToolValidationError(f"{field} must be a boolean, got {type(value).__name__}")
    return value


def _validate_nonneg_int(value: Any, field: str = "count") -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ToolValidationError(f"{field} must be a non-negative integer")
    return value


def _validate_account_ids(value: Any, field: str = "account_ids") -> list[str]:
    """A bounded list of validated Account ids (Phase 8.5 neighbourhood query)."""
    if not isinstance(value, list) or not value:
        raise ToolValidationError(f"{field} must be a non-empty list of account ids")
    if len(value) > 500:
        raise ToolValidationError(f"{field} exceeds the 500-id cap")
    seen: list[str] = []
    for item in value:
        account_id = validate_entity_id(item, field)
        if account_id not in seen:
            seen.append(account_id)
    return seen


def _validate_legs(value: Any, field: str = "legs") -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ToolValidationError(f"{field} must be a non-empty list of {{src, dst}}")
    if len(value) > 100:
        raise ToolValidationError(f"{field} exceeds the 100-leg cap")
    out: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"src", "dst"}:
            raise ToolValidationError(f"{field} entries must be exactly {{src, dst}}")
        out.append({
            "src": validate_entity_id(item["src"], f"{field}.src"),
            "dst": validate_entity_id(item["dst"], f"{field}.dst"),
        })
    return out


# One validator per documented parameter name. A catalog template that uses a
# parameter missing from this table fails loudly in validate_cited_params
# rather than silently skipping a check (a completeness test pins this).
_PARAM_VALIDATORS: dict[str, Any] = {
    # entity references
    "alert_id": lambda v: validate_entity_id(v, "alert_id"),
    "entity_id": lambda v: validate_entity_id(v, "entity_id"),
    "account_id": lambda v: validate_entity_id(v, "account_id"),
    "customer_a_id": lambda v: validate_entity_id(v, "customer_a_id"),
    "customer_b_id": lambda v: validate_entity_id(v, "customer_b_id"),
    "source_account_id": lambda v: validate_entity_id(v, "source_account_id"),
    "target_account_id": lambda v: validate_entity_id(v, "target_account_id"),
    "case_id": lambda v: validate_entity_id(v, "case_id"),
    "older_case_id": lambda v: validate_entity_id(v, "older_case_id"),
    "evidence_id": lambda v: validate_entity_id(v, "evidence_id"),
    "simulation_id": lambda v: validate_entity_id(v, "simulation_id"),
    "action_id": lambda v: validate_entity_id(v, "action_id"),
    # bounded query arguments
    "limit": validate_limit,
    "min_amount": validate_amount,
    "direction": validate_direction,
    "cutoff": _validate_cutoff,
    "legs": _validate_legs,
    "account_ids": _validate_account_ids,
    "window_days": validate_window_days,
    "hops": validate_hop_count,
    # write-path scalars (never replayable; still structurally checked)
    "action": lambda v: _validate_text(v, "action"),
    "status": lambda v: _validate_text(v, "status"),
    "verdict": lambda v: _validate_text(v, "verdict"),
    "severity": validate_severity,
    "title": lambda v: _validate_text(v, "title"),
    "summary": lambda v: _validate_text(v, "summary"),
    "reason": lambda v: _validate_text(v, "reason"),
    "requested_by": lambda v: _validate_text(v, "requested_by"),
    "risk": lambda v: _validate_text(v, "risk"),
    "now": lambda v: _validate_text(v, "now"),
    "created_at": lambda v: _validate_text(v, "created_at"),
    "updated_at": lambda v: _validate_text(v, "updated_at"),
    "captured_at": lambda v: _validate_text(v, "captured_at"),
    "claim": lambda v: _validate_text(v, "claim"),
    "stance": lambda v: _validate_text(v, "stance"),
    "query_id": lambda v: _validate_text(v, "query_id"),
    "source_tool": lambda v: _validate_text(v, "source_tool"),
    "result_hash": lambda v: _validate_text(v, "result_hash"),
    "interpretation": lambda v: _validate_text(v, "interpretation"),
    "evidence_kind": lambda v: _validate_text(v, "evidence_kind"),
    "graph_fingerprint": lambda v: _validate_text(v, "graph_fingerprint"),
    "params_json": lambda v: _validate_text(v, "params_json"),
    "rows_json": lambda v: _validate_text(v, "rows_json"),
    "derived_json": lambda v: _validate_text(v, "derived_json"),
    "metrics_json": lambda v: _validate_text(v, "metrics_json"),
    "columns_json": lambda v: _validate_text(v, "columns_json"),
    "interpolation_json": lambda v: _validate_text(v, "interpolation_json"),
    "template_sha256": lambda v: _validate_text(v, "template_sha256"),
    "sources_json": lambda v: _validate_text(v, "sources_json"),
    "required_source_queries_json": lambda v: _validate_text(
        v, "required_source_queries_json"
    ),
    "evidence_ids_json": lambda v: _validate_text(v, "evidence_ids_json"),
    "policy_json": lambda v: _validate_text(v, "policy_json"),
    "before_state_json": lambda v: _validate_text(v, "before_state_json"),
    "after_state_json": lambda v: _validate_text(v, "after_state_json"),
    "verification_json": lambda v: _validate_text(v, "verification_json"),
    "row_count": _validate_nonneg_int,
    "source_count": _validate_nonneg_int,
    "truncated": _validate_bool,
    "simulation_complete": _validate_bool,
}

_INTERPOLATION_VALIDATORS: dict[str, Any] = {
    "<<hops>>": validate_hop_count,
    "<<max_len>>": validate_cycle_len,
    "<<max_hops>>": validate_hop_count,
}


def template_tokens(query_id: str) -> tuple[str, ...]:
    """The ``<<token>>`` interpolation sites in an approved template."""
    return tuple(sorted(set(_INTERP_RE.findall(get_template(query_id)))))


def param_names(query_id: str) -> tuple[str, ...]:
    """The ``$name`` bind-parameter sites in an approved template."""
    return tuple(sorted(set(_PARAM_RE.findall(get_template(query_id)))))


def template_sha256(query_id: str) -> str:
    """Fingerprint of the approved template text (tokens unbound)."""
    import hashlib

    digest = hashlib.sha256(get_template(query_id).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def validate_interpolation_value(query_id: str, token: str, value: Any) -> int:
    """Validate one recorded interpolation value against its token contract."""
    if token not in _INTERPOLATION_TOKENS:
        raise ToolValidationError(f"illegal interpolation token {token!r}")
    if token not in _INTERPOLATION_VALIDATORS:
        raise ToolValidationError(f"no validator for interpolation token {token!r}")
    return _INTERPOLATION_VALIDATORS[token](value)


def validate_interpolation(query_id: str, values: Any) -> dict[str, int]:
    """Validate the interpolation map recorded alongside a citation."""
    if values in (None, {}):
        return {}
    if not isinstance(values, dict):
        raise ToolValidationError("interpolation must be a token->integer object")
    tokens = set(template_tokens(query_id))
    out: dict[str, int] = {}
    for token, value in values.items():
        if token not in _INTERPOLATION_TOKENS or token not in tokens:
            raise ToolValidationError(
                f"template {query_id} has no interpolation token {token!r}"
            )
        out[token] = validate_interpolation_value(query_id, token, value)
    return out


def validate_cited_params(query_id: str, params: Any) -> dict[str, Any]:
    """Validate stored parameters against the approved template's contract.

    Every ``$name`` the template uses must be present and pass its documented
    validator; parameters the template does not use are rejected. Unknown
    templates are rejected outright. This is the verifier's argument contract.
    """
    if query_id not in _QUERY_TEMPLATES:
        raise ToolValidationError(f"unapproved query id {query_id!r}")
    if not isinstance(params, dict):
        raise ToolValidationError("params must be a JSON object")
    expected = set(param_names(query_id))
    unknown = set(params) - expected
    if unknown:
        raise ToolValidationError(
            f"params contain keys not used by {query_id}: {sorted(unknown)}"
        )
    missing = expected - set(params)
    if missing:
        raise ToolValidationError(
            f"params are missing required keys for {query_id}: {sorted(missing)}"
        )
    out: dict[str, Any] = {}
    for name in sorted(expected):
        validator = _PARAM_VALIDATORS.get(name)
        if validator is None:
            raise ToolValidationError(
                f"no stored validator for parameter ${name} of {query_id}; "
                "the catalog and the verifier contract are out of sync"
            )
        out[name] = validator(params[name])
    return out
