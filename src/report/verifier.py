"""Phase 6 independent evidence verifier (PLAN.md §11) — deterministic, no LLM.

The verifier never trusts report prose and never executes Cypher that came
from graph properties, evidence text or model output. It resolves every cited
query id against the approved catalog, validates stored parameters against
the template's argument contract, recomputes the canonical full-row hash of
any preserved snapshot, and — where possible — rebuilds the exact bounded
query from recorded integer interpolation values and replays it read-only
against the current graph.

Phase 6.1 — claim-level scope. A single tool invocation can execute several
approved queries (a window anchor plus the claim-material queries; e.g.
``fan_in_out`` = ``fan_in_senders_v1`` + ``fan_out_receivers_v1``). The
verifier therefore verifies EVERY recorded source of an evidence record
individually — catalog membership, parameter contract, template fingerprint,
preserved-snapshot integrity and bounded replay — and derives the record's
status from the per-source outcomes by the documented precedence
``mismatch > changed_after_action > unverifiable > reproduced_current_state >
verified_against_recorded_evidence``. A record is only
``reproduced_current_state`` when EVERY required source replay-matches; a
missing, malformed or internally inconsistent source manifest fails closed.
For fan-in/fan-out records the recorded derived aggregates are additionally
recomputed from the replayed source rows (derived-result consistency), and a
full capture without a verifiable derived result is never labeled fully
reproduced. Pre-6.1 legacy records whose only recorded query is the observation
window anchor are never labeled reproduced either: the claim-bearing sources of
the invocation were never recorded, so a replay of the anchor alone cannot
carry the claim.

Verification status vocabulary (see docs/phase-6-reports.md for the mapping to
the earlier per-record concepts):

- ``verified_against_recorded_evidence``: the preserved, normalized snapshot
  and its integrity checks pass; no live replay was needed or possible.
  Hash-only citations with no replayable query are **unverifiable**, not
  verified: there is no snapshot to compare against.
- ``reproduced_current_state``: an independent read-only replay of the
  approved query, with the ORIGINAL recorded parameters and window,
  reproduces the recorded full-row hash against the current graph.
- ``changed_after_action``: the replay differs, and the difference is
  consistent with a recorded, authorized action on this case whose account
  is the subject of the recorded query.
- ``mismatch``: malformed or tampered records, unknown query ids, drifted
  templates, invalid parameters, incomplete source manifests, or replay
  differences that no recorded action explains. Never silently ignored.
- ``unverifiable``: missing evidence, unsupported replay (write-path or
  interpolation bounds not recorded), truncation without a preserved
  snapshot, mutable case/action metadata that legitimately changed since
  capture, or a failed replay.

Row ordering: ``rows_hash`` (src.tools.envelopes) canonicalizes row order, so
replayed results are compared regardless of server-side ordering. The
fan-in/fan-out derived-aggregate recomputation mirrors
``src.tools.reads.fan_in_out`` exactly and only runs when both its sources
replay-matched untruncated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, Field

from src.db import QueryError
from src.tools import query_catalog as qc
from src.tools.actions import verify_freeze
from src.tools.envelopes import ToolValidationError, rows_hash, run_evidence

# ---- status vocabulary ------------------------------------------------------

STATUS_VERIFIED_RECORDED = "verified_against_recorded_evidence"
STATUS_REPRODUCED = "reproduced_current_state"
STATUS_CHANGED = "changed_after_action"
STATUS_MISMATCH = "mismatch"
STATUS_UNVERIFIABLE = "unverifiable"

ALL_STATUSES = (
    STATUS_VERIFIED_RECORDED,
    STATUS_REPRODUCED,
    STATUS_CHANGED,
    STATUS_MISMATCH,
    STATUS_UNVERIFIABLE,
)

# ---- replay classification --------------------------------------------------
#
# Every READ query id must appear in exactly one bucket (a test pins this):
# - immutable: transaction-history reads that a graph-state action must not
#   change; a replay difference is always a genuine discrepancy.
# - action_sensitive: topology/status reads that an authorized freeze is
#   EXPECTED to change for the frozen account.
# - mutable_metadata: case/action/simulation metadata that legitimately
#   changes as cases progress; differences are flagged, not attributed.
_IMMUTABLE_QUERIES = frozenset({
    "alert_context_v1", "txn_max_ts_v1",
    "entity_profile_flows_v1", "entity_profile_owner_v1",
    "entity_exists_v1", "txn_events_for_legs_v1",
    "shared_devices_v1", "shared_identifiers_v1",
    "fan_in_senders_v1", "fan_out_receivers_v1",
})
_ACTION_SENSITIVE_QUERIES = frozenset({
    "detect_closed_paths_v1", "trace_paths_out_v1", "trace_paths_in_v1",
    "active_path_min_hops_v1", "active_topology_v1",
    "account_state_v1", "account_incident_active_edges_v1",
    "direct_outgoing_in_window_v1", "hub_weighted_in_degree_v1",
    # Both read mutable account properties/status: an authorized freeze on the
    # queried account legitimately changes what they return, so they are
    # status reads (attributed to the recorded action when the frozen account
    # is the subject of the parameters), never "immutable history".
    "entity_profile_lookup_v1", "entity_profile_owned_accounts_v1",
    # Phase 8.5 UI neighbourhood reads: active flags and reachability change
    # when an authorized freeze deactivates routing, so a post-action
    # difference is attributed to the recorded action.
    "neighborhood_out_accounts_v1", "neighborhood_in_accounts_v1",
    "neighborhood_edges_v1",
})
_MUTABLE_METADATA_QUERIES = frozenset({
    "case_lookup_v1", "case_about_entities_v1", "case_evidence_v1",
    "case_actions_v1", "case_reuses_list_v1", "prior_cases_v1",
    "simulation_for_case_v1", "simulation_for_case_any_v1",
    "simulation_lookup_v1", "action_for_case_v1", "evidence_lookup_v1",
    # Phase 8.5 UI alert inbox: reads Case/Action metadata, which legitimately
    # evolves as investigations run — a snapshot, not a replayable fact.
    "alert_inbox_v1",
})

_HASH_RE_LEN = 64

_STANCES = ("supporting", "contradicting", "unresolved")
_EVIDENCE_KINDS = ("immutable_history", "current_state", "captured_snapshot")


# ---- models -----------------------------------------------------------------


class CheckOutcome(BaseModel):
    name: str
    ok: bool
    detail: str


class SourceVerification(BaseModel):
    """Per-sub-query outcome inside one compound evidence record."""

    query_id: str
    role: str = "claim_source"
    required: bool = True
    status: str = STATUS_UNVERIFIABLE
    reason: str = ""
    replay: dict[str, Any] | None = None


class ClaimVerification(BaseModel):
    """Outcome for one persisted evidence record's claim.

    ``sources`` carries the per-sub-query outcomes; ``status``/``reason`` are
    the aggregate by the documented precedence. ``compound`` is True when a
    persisted source manifest was present. ``replay`` mirrors the primary
    (first) source's replay for convenience and backward compatibility.
    """

    evidence_id: str
    claim: str
    query_id: str | None = None
    source_tool: str | None = None
    status: str = STATUS_UNVERIFIABLE
    reason: str = ""
    checks: list[CheckOutcome] = Field(default_factory=list)
    replay: dict[str, Any] | None = None
    sources: list[SourceVerification] = Field(default_factory=list)
    compound: bool = False
    source_count: int = 0
    required_source_queries: list[str] = Field(default_factory=list)
    checked_source_queries: list[str] = Field(default_factory=list)
    unavailable_source_queries: list[str] = Field(default_factory=list)
    mismatched_source_queries: list[str] = Field(default_factory=list)


class ReferenceVerification(BaseModel):
    """Outcome for one material report claim mapped to evidence ids."""

    claim: str
    source: str
    evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence_ids: list[str] = Field(default_factory=list)
    per_evidence: list[dict[str, Any]] = Field(default_factory=list)
    checked_source_queries: list[str] = Field(default_factory=list)
    unavailable_source_queries: list[str] = Field(default_factory=list)
    mismatched_source_queries: list[str] = Field(default_factory=list)
    status: str = STATUS_UNVERIFIABLE
    reason: str = ""


class CaseVerification(BaseModel):
    case_id: str
    exists: bool
    claims: list[ClaimVerification] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    references: list[ReferenceVerification] = Field(default_factory=list)
    reference_summary: dict[str, int] = Field(default_factory=dict)
    current_state: list[dict[str, Any]] = Field(default_factory=list)
    post_action: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@dataclass
class CaseRecords:
    """Everything the persisted case record exposes, via approved reads only."""

    case: dict[str, Any] | None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    action_rows: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    simulation: dict[str, Any] | None = None
    reuses: list[dict[str, Any]] = field(default_factory=list)
    about_entities: list[str] = field(default_factory=list)
    query_ids_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---- reads ------------------------------------------------------------------


def _read(graph, query_id: str, params: Mapping[str, Any] | None = None, *, row_cap=None):
    return run_evidence(
        graph, query_id, qc.get_template(query_id), dict(params or {}), row_cap=row_cap
    )


def _rows_to_dicts(ev) -> list[dict[str, Any]]:
    return [dict(zip(ev.columns, row)) for row in ev.rows]


def load_case_records(graph, case_id: str) -> CaseRecords:
    """Load the persisted case, evidence, actions, simulation and REUSES links."""
    used: list[str] = []

    case_ev = _read(graph, "case_lookup_v1", {"case_id": case_id}, row_cap=1)
    used.append("case_lookup_v1")
    case = case_ev.rows[0][0] if case_ev.rows else None
    if case is None:
        return CaseRecords(
            case=None, query_ids_used=used,
            notes=[f"case {case_id} does not exist in the graph"],
        )

    evidence_ev = _read(
        graph, "case_evidence_v1", {"case_id": case_id, "limit": 500}, row_cap=500
    )
    used.append("case_evidence_v1")
    evidence = _rows_to_dicts(evidence_ev)

    action_rows_ev = _read(
        graph, "case_actions_v1", {"case_id": case_id, "limit": 100}, row_cap=100
    )
    used.append("case_actions_v1")
    action_rows = _rows_to_dicts(action_rows_ev)
    actions: list[dict[str, Any]] = []
    for row in action_rows:
        full = _read(
            graph, "action_for_case_v1",
            {"case_id": case_id, "account_id": row["account_id"],
             "action": row["action"]},
            row_cap=1,
        )
        used.append("action_for_case_v1")
        if full.rows:
            actions.append(full.rows[0][0])

    sim_ev = _read(graph, "simulation_for_case_any_v1", {"case_id": case_id}, row_cap=1)
    used.append("simulation_for_case_any_v1")
    simulation = sim_ev.rows[0][0] if sim_ev.rows else None

    reuses_ev = _read(
        graph, "case_reuses_list_v1", {"case_id": case_id, "limit": 100}, row_cap=100
    )
    used.append("case_reuses_list_v1")
    reuses = _rows_to_dicts(reuses_ev)

    about_ev = _read(
        graph, "case_about_entities_v1", {"case_id": case_id, "limit": 100}, row_cap=100
    )
    used.append("case_about_entities_v1")
    about = [row[0] for row in about_ev.rows]

    return CaseRecords(
        case=case, evidence=evidence, action_rows=action_rows, actions=actions,
        simulation=simulation, reuses=reuses, about_entities=about,
        query_ids_used=used,
    )


# ---- per-record verification ------------------------------------------------


def _get(item: dict[str, Any], key: str, default=None):
    return item.get(key, default)


def _parse_json_field(record: dict[str, Any], key: str) -> tuple[Any, str | None]:
    raw = record.get(key)
    if raw in (None, ""):
        return ({} if key.endswith("interpolation_json") else [] if key in (
            "rows_json", "columns_json") else {}), None
    try:
        return json.loads(raw), None
    except (TypeError, ValueError):
        return None, f"{key} is not valid JSON"


def _explained_by_action(
    query_id: str, params: Mapping[str, Any], actions: Iterable[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the applied action that explains a replay difference, if any.

    Conservative rule: the action must be an applied freeze on THIS case, the
    query must be a topology/status read, and the frozen account must be the
    subject of the recorded query parameters. Anything else is a mismatch for
    review, not attributed to the action.
    """
    if query_id not in _ACTION_SENSITIVE_QUERIES:
        return None
    subjects = {
        params.get("account_id"), params.get("source_account_id"),
        params.get("target_account_id"), params.get("entity_id"),
    }
    # Phase 8.5 neighbourhood queries take a list of account ids.
    listed = params.get("account_ids")
    if isinstance(listed, (list, tuple, set)):
        subjects.update(listed)
    for action in actions:
        if action.get("action") != "freeze_account":
            continue
        if action.get("status") != "applied":
            continue
        if action.get("account_id") in subjects:
            return action
    return None


# Tool-semantic source requirements (Phase 6.1). A compound manifest that
# self-describes as a FULL tool-invocation capture — it records the
# observation-window anchor (``txn_max_ts_v1``), which every windowed tool runs
# first — must contain the query ids the tool ALWAYS executes; a manifest that
# lost one of them is refused as incomplete even if its declared source_count
# was adjusted to match. A manifest WITHOUT the anchor is a deliberate
# sub-query citation (a single claim-material query recorded on its own), not a
# full capture, so the minima does not apply to it. Queries a tool runs
# conditionally (e.g. txn_events_for_legs_v1 only when a path or cycle exists)
# are cross-checked against the capture-time required_source_queries
# declaration instead, which the verifier never rewrites.
_WINDOW_ANCHOR_QUERY = "txn_max_ts_v1"

_TOOL_REQUIRED_SOURCES: dict[str, dict[str, tuple[str, ...]]] = {
    "alert_context": {"all_of": ("alert_context_v1",)},
    "entity_profile": {"all_of": ("entity_profile_lookup_v1",)},
    "detect_cycle": {"all_of": ("detect_closed_paths_v1",)},
    "fan_in_out": {"all_of": ("fan_in_senders_v1", "fan_out_receivers_v1")},
    "trace_funds": {"one_of": ("trace_paths_out_v1", "trace_paths_in_v1")},
    "shared_identifiers": {"all_of": ("shared_devices_v1", "shared_identifiers_v1")},
    "hub_scores": {"all_of": ("hub_weighted_in_degree_v1",)},
    "prior_cases": {"all_of": ("prior_cases_v1",)},
}

_SOURCE_ROLES = ("window_anchor", "claim_source")

# Source-level statuses that mean "independently checked" in the report view.
_CHECKED_SOURCE_STATUSES = (STATUS_REPRODUCED, STATUS_CHANGED, STATUS_VERIFIED_RECORDED)


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 7 + _HASH_RE_LEN
    )


def _distinct(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _normalize_manifest_entry(
    raw: Any, ordinal: int, problems: list[str]
) -> dict[str, Any]:
    """Structurally validate one stored source-manifest entry.

    Missing or malformed required fields are collected as problems: a
    compound record with an incomplete manifest is never partially trusted.
    """
    where = f"source[{ordinal}]"
    if not isinstance(raw, dict):
        problems.append(f"{where} is not a JSON object")
        return {}
    entry: dict[str, Any] = {
        "query_id": "", "role": "claim_source", "params": {},
        "interpolation": {}, "result_hash": "", "row_count": 0,
        "truncated": False, "template_sha256": None, "rows": [], "columns": [],
    }
    query_id = raw.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        problems.append(f"{where} has no query_id")
    else:
        entry["query_id"] = query_id
    role = raw.get("role")
    if role not in _SOURCE_ROLES:
        problems.append(f"{where} role {role!r} is not window_anchor/claim_source")
    else:
        entry["role"] = role
    result_hash = raw.get("result_hash")
    if not _is_hash(result_hash):
        problems.append(f"{where} result_hash is not 'sha256:' plus 64 hex chars")
    else:
        entry["result_hash"] = result_hash
    row_count = raw.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        problems.append(f"{where} row_count is not a non-negative integer")
    else:
        entry["row_count"] = row_count
    if not isinstance(raw.get("truncated"), bool):
        problems.append(f"{where} truncated is not a boolean")
    else:
        entry["truncated"] = raw["truncated"]
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        problems.append(f"{where} params is not a JSON object")
    else:
        entry["params"] = params
    interpolation = raw.get("interpolation") or {}
    if not isinstance(interpolation, dict):
        problems.append(f"{where} interpolation is not a JSON object")
    else:
        entry["interpolation"] = interpolation
    template_sha256 = raw.get("template_sha256")
    if template_sha256 is not None and not isinstance(template_sha256, str):
        problems.append(f"{where} template_sha256 is not a string")
    else:
        entry["template_sha256"] = template_sha256
    rows = raw.get("rows") or []
    columns = raw.get("columns") or []
    if not isinstance(rows, list) or not isinstance(columns, list):
        problems.append(f"{where} rows/columns must be lists")
    else:
        entry["rows"] = rows
        entry["columns"] = columns
        if rows and not columns:
            problems.append(
                f"{where} supplies rows without columns; the snapshot cannot "
                "be verified"
            )
        elif rows and all(isinstance(c, str) for c in columns) and _is_hash(
            entry["result_hash"]
        ):
            try:
                agreed = rows_hash(list(columns), rows) == entry["result_hash"]
            except Exception:
                agreed = False
            if not agreed:
                problems.append(
                    f"{where} preserved rows do not match its declared result hash"
                )
    return entry


def _extract_sources(record: dict[str, Any]) -> dict[str, Any]:
    """Resolve the recorded source manifest (compound) or legacy single query.

    Returns ``sources`` (normalized entries: query_id, role, params,
    interpolation, result_hash, row_count, truncated, template_sha256,
    rows/columns for the preserved snapshot, and ``required``), ``problems``
    (fail-closed integrity findings), ``compound``, and
    ``required_source_queries``. The primary source's preserved snapshot is
    the top-level rows/columns; secondary snapshots travel inside their
    manifest entries.
    """
    problems: list[str] = []
    raw_sources = record.get("sources_json")
    source_count = record.get("source_count")
    raw_required = record.get("required_source_queries_json")
    compound = raw_sources not in (None, "") or source_count is not None

    query_id = record.get("query_id")
    result_hash = str(record.get("result_hash") or "")
    row_count = record.get("row_count")
    truncated = bool(record.get("truncated"))

    top_params, params_err = _parse_json_field(record, "params_json")
    if params_err:
        problems.append(f"malformed parameters: {params_err}")
    top_interpolation, interp_err = _parse_json_field(record, "interpolation_json")
    if interp_err:
        problems.append(f"malformed interpolation bounds: {interp_err}")
    top_rows, rows_err = _parse_json_field(record, "rows_json")
    if rows_err:
        problems.append(f"malformed snapshot: {rows_err}")
    top_columns, columns_err = _parse_json_field(record, "columns_json")
    if columns_err:
        problems.append(f"malformed snapshot: {columns_err}")

    if not compound:
        return {
            "sources": [{
                "query_id": query_id if isinstance(query_id, str) else "",
                "role": "claim_source",
                "params": top_params if isinstance(top_params, dict) else {},
                "interpolation": top_interpolation if isinstance(top_interpolation, dict) else {},
                "result_hash": result_hash,
                "row_count": row_count,
                "truncated": truncated,
                "template_sha256": record.get("template_sha256"),
                "rows": top_rows if isinstance(top_rows, list) else [],
                "columns": top_columns if isinstance(top_columns, list) else [],
                "required": True,
            }],
            "problems": problems,
            "compound": False,
            "required_source_queries": [],
        }

    parsed: Any = None
    if raw_sources in (None, ""):
        problems.append(
            "source manifest is missing (sources_json absent or empty) while "
            "the record declares a compound citation"
        )
    else:
        try:
            parsed = json.loads(raw_sources)
        except (TypeError, ValueError):
            problems.append("sources_json is not valid JSON")
    entries: list[dict[str, Any]] = []
    if parsed is not None:
        if not isinstance(parsed, list) or not parsed:
            problems.append("source manifest is not a non-empty list")
        else:
            for ordinal, raw_entry in enumerate(parsed):
                entries.append(
                    _normalize_manifest_entry(raw_entry, ordinal, problems)
                )
    if isinstance(source_count, int) and not isinstance(source_count, bool):
        if source_count != len(entries):
            problems.append(
                f"source_count declares {source_count} source(s) but the "
                f"manifest stores {len(entries)}; the source manifest is "
                "incomplete"
            )
    else:
        problems.append("source_count is not a non-negative integer")

    required: list[str] = []
    if raw_required not in (None, ""):
        try:
            parsed_required = json.loads(raw_required)
        except (TypeError, ValueError):
            problems.append("required_source_queries_json is not valid JSON")
            parsed_required = None
        if parsed_required is not None:
            if not isinstance(parsed_required, list) or not all(
                isinstance(q, str) for q in parsed_required
            ):
                problems.append(
                    "required_source_queries_json is not a list of query ids"
                )
            else:
                required = _distinct(parsed_required)

    # The top-level citation fields must mirror sources[0].
    if entries:
        primary = entries[0]
        if primary.get("query_id") != query_id:
            problems.append(
                "top-level query_id does not match sources[0] "
                f"({primary.get('query_id')!r} != {query_id!r})"
            )
        if primary.get("result_hash") != result_hash:
            # The declared hash disagrees between the top-level field and
            # sources[0]. When preserved rows exist they arbitrate which
            # declaration the snapshot supports; fail closed either way.
            if isinstance(top_rows, list) and top_rows and isinstance(
                top_columns, list
            ) and top_columns:
                recomputed = rows_hash(list(top_columns), top_rows)
                if recomputed != result_hash:
                    problems.append(
                        "preserved snapshot rows do not match the declared "
                        "result hash"
                    )
                elif recomputed != primary.get("result_hash"):
                    problems.append(
                        "sources[0] result_hash does not match the preserved "
                        "snapshot rows"
                    )
                else:
                    problems.append(
                        "top-level result_hash does not match sources[0]; the "
                        "record was edited inconsistently"
                    )
            else:
                problems.append(
                    "top-level result_hash does not match sources[0]; the "
                    "record was edited inconsistently"
                )
        if primary.get("row_count") != row_count:
            problems.append("top-level row_count does not match sources[0]")
        if bool(primary.get("truncated")) != truncated:
            problems.append("top-level truncated does not match sources[0]")
        # Params and interpolation are NOT compared here: both the top-level
        # declaration and the manifest entry are validated against the
        # template contract during per-source verification, and only then are
        # they compared for consistency, so a contract violation is reported
        # with its specific reason instead of a generic manifest error.
        primary["declared_params"] = (
            top_params if isinstance(top_params, dict) else {}
        )
        primary["declared_interpolation"] = (
            top_interpolation if isinstance(top_interpolation, dict) else {}
        )
        if isinstance(top_rows, list) and top_rows:
            # The top-level rows are the record's declared preserved snapshot
            # (this is the copy the classic tamper tests modify); they always
            # take precedence for the primary source. The entry's own rows —
            # when a caller supplied them — were already validated against the
            # entry's declared hash above.
            primary["rows"] = top_rows
            primary["columns"] = (
                top_columns if isinstance(top_columns, list) else []
            )
        primary["declared_template_sha256"] = record.get("template_sha256")

    # Capture-time required sources: each must be present and approved, and
    # every claim-material source must be declared.
    manifest_qids = {entry.get("query_id") for entry in entries}
    required_set = set(required)
    for required_qid in required:
        if required_qid not in qc._QUERY_TEMPLATES:
            problems.append(
                "required_source_queries contains unapproved query id "
                f"{required_qid!r}"
            )
        if required_qid not in manifest_qids:
            problems.append(
                f"required source-query record for {required_qid!r} is missing "
                "from the source manifest"
            )
    if raw_required not in (None, ""):
        for entry in entries:
            if entry.get("role") == "claim_source" and \
                    entry.get("query_id") not in required_set:
                problems.append(
                    f"source {entry.get('query_id')!r} is claim-material but is "
                    "not declared in required_source_queries"
                )

    # Tool-semantic minima: a manifest that self-describes as a full
    # tool-invocation capture (it records the observation-window anchor) must
    # contain a query the tool always runs even if the count and the required
    # list were edited consistently. A manifest without the anchor is a
    # deliberate sub-query citation and the minima does not apply.
    source_tool = str(record.get("source_tool") or "")
    requirement = _TOOL_REQUIRED_SOURCES.get(source_tool)
    full_capture = any(
        entry.get("query_id") == _WINDOW_ANCHOR_QUERY for entry in entries
    )
    if requirement and full_capture:
        present = set(manifest_qids)
        for required_qid in requirement.get("all_of", ()):
            if required_qid not in present:
                problems.append(
                    f"required source-query record for {required_qid!r} is "
                    f"missing from the {source_tool} manifest"
                )
        one_of = requirement.get("one_of", ())
        if one_of and not (set(one_of) & present):
            problems.append(
                "not one of the required source queries for "
                f"{source_tool} ({', '.join(one_of)}) is present in its "
                "manifest"
            )

    for entry in entries:
        entry["required"] = entry.get("query_id") in required_set
    return {
        "sources": entries,
        "problems": problems,
        "compound": True,
        "required_source_queries": required,
    }


def _verify_source(
    graph,
    source: dict[str, Any],
    *,
    actions: Iterable[dict[str, Any]],
    checks: list[CheckOutcome],
    name_prefix: str,
) -> tuple[str, str, dict[str, Any] | None, Any]:
    """Verify ONE recorded source query end to end.

    Mirrors the single-record verifier steps: catalog membership, template
    fingerprint, parameter contract, interpolation bounds, preserved-snapshot
    integrity, bounded read-only replay, and outcome resolution. Returns
    ``(status, reason, replay_info, replayed_evidence)``.
    """
    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(CheckOutcome(name=f"{name_prefix}{name}", ok=ok, detail=detail))

    query_id = str(source.get("query_id") or "")
    result_hash = str(source.get("result_hash") or "")
    truncated = bool(source.get("truncated"))
    params = source.get("params") if isinstance(source.get("params"), dict) else {}
    interpolation = (
        source.get("interpolation")
        if isinstance(source.get("interpolation"), dict) else {}
    )
    rows = list(source.get("rows") or [])
    columns = list(source.get("columns") or [])
    row_count = source.get("row_count")

    # catalog resolution
    if query_id not in qc._QUERY_TEMPLATES:
        check("query_in_catalog", False,
              f"{query_id!r} is unknown; no execution attempted")
        return STATUS_MISMATCH, f"unknown query id {query_id!r}; rejected", None, None
    check("query_in_catalog", True, f"{query_id!r} is an approved catalog query")

    # template drift (both the top-level citation fingerprint and the
    # manifest entry's fingerprint must match the catalog when present)
    stored_fps = [
        fp for fp in (
            source.get("template_sha256"),
            source.get("declared_template_sha256"),
        ) if fp
    ]
    if stored_fps:
        current_fp = qc.template_sha256(query_id)
        drifted = next((fp for fp in stored_fps if fp != current_fp), None)
        if drifted is not None:
            check(
                "template_fingerprint", False,
                f"stored template fingerprint {drifted} differs from the "
                f"current catalog template {current_fp}; the template "
                "definition changed after capture",
            )
            return (STATUS_MISMATCH,
                    f"template definition for {query_id} changed after capture",
                    None, None)
        check("template_fingerprint", True,
              "template matches the catalog fingerprint")
    else:
        check("template_fingerprint", True,
              "no template fingerprint recorded (legacy citation); drift "
              "cannot be checked")

    # parameter contract: BOTH the record's top-level declaration and the
    # manifest entry (when present) must satisfy the template contract.
    declared_params = source.get("declared_params")
    declaration_checks: list[tuple[str, dict[str, Any]]] = [("recorded", params)]
    if isinstance(declared_params, dict):
        declaration_checks.append(("top-level", declared_params))
    validated_params: dict[str, Any] = {}
    for label, candidate in declaration_checks:
        try:
            validated = qc.validate_cited_params(query_id, candidate)
        except ToolValidationError as exc:
            check("params_contract", False, f"{label} parameters: {exc}")
            return (STATUS_MISMATCH,
                    f"stored parameters violate the {query_id} contract: {exc}",
                    None, None)
        if label == "recorded":
            validated_params = validated
    if (
        isinstance(declared_params, dict)
        and json.dumps(declared_params, sort_keys=True, default=str)
        != json.dumps(params, sort_keys=True, default=str)
    ):
        check("params_contract", False,
              "top-level params_json does not match sources[0]")
        return (STATUS_MISMATCH,
                "top-level params_json does not match sources[0]; the record "
                "was edited inconsistently",
                None, None)
    check("params_contract", True,
          "stored parameters satisfy the approved template contract")

    # interpolation bounds: same both-copies validation.
    declared_interpolation = source.get("declared_interpolation")
    interpolation_checks: list[tuple[str, dict[str, Any]]] = [
        ("recorded", interpolation)
    ]
    if isinstance(declared_interpolation, dict):
        interpolation_checks.append(("top-level", declared_interpolation))
    validated_interpolation: dict[str, int] = {}
    for label, candidate in interpolation_checks:
        try:
            validated = qc.validate_interpolation(query_id, candidate)
        except ToolValidationError as exc:
            check("interpolation_bounds", False, f"{label} bounds: {exc}")
            return STATUS_MISMATCH, f"invalid interpolation bounds: {exc}", None, None
        if label == "recorded":
            validated_interpolation = validated
    if (
        isinstance(declared_interpolation, dict)
        and json.dumps(declared_interpolation, sort_keys=True, default=str)
        != json.dumps(interpolation, sort_keys=True, default=str)
    ):
        check("interpolation_bounds", False,
              "top-level interpolation_json does not match sources[0]")
        return (STATUS_MISMATCH,
                "top-level interpolation_json does not match sources[0]; the "
                "record was edited inconsistently",
                None, None)
    check("interpolation_bounds", True,
          f"recorded bounds: {validated_interpolation or 'none'}")

    # preserved snapshot integrity
    snapshot_ok = False
    if rows:
        if not columns:
            check("snapshot_integrity", False,
                  "non-empty rows without columns cannot be verified")
            return STATUS_MISMATCH, "snapshot rows have no column names", None, None
        recomputed = rows_hash(list(columns), rows)
        if recomputed != result_hash:
            check("snapshot_integrity", False,
                  f"stored rows hash to {recomputed} but the record declares "
                  f"{result_hash}")
            return (STATUS_MISMATCH,
                    "preserved snapshot rows do not match the declared result "
                    "hash", None, None)
        if row_count != len(rows):
            check("snapshot_integrity", False,
                  f"row_count {row_count} disagrees with {len(rows)} stored rows")
            return STATUS_MISMATCH, "row_count disagrees with stored rows", None, None
        snapshot_ok = True
        check("snapshot_integrity", True,
              f"recomputed full-row hash over {len(rows)} row(s) matches")
    else:
        check("snapshot_integrity", True,
              "hash-only citation: rows were not preserved, so the snapshot "
              "itself cannot be re-hashed")

    # live replay
    replay_possible = True
    replay_reason = ""
    if query_id in qc.WRITE_QUERY_IDS:
        replay_possible, replay_reason = False, (
            "write-path record: replay would execute mutations and is never run"
        )
    else:
        missing_tokens = [
            t for t in qc.template_tokens(query_id)
            if t not in validated_interpolation
        ]
        if missing_tokens:
            replay_possible, replay_reason = False, (
                f"replay unavailable: interpolation bounds not recorded for "
                f"{', '.join(missing_tokens)}"
            )

    replayed = None
    replay_info: dict[str, Any] | None = None
    if replay_possible:
        try:
            cypher = qc.get_template(query_id)
            for token in qc.template_tokens(query_id):
                cypher = cypher.replace(
                    token,
                    str(qc.validate_interpolation_value(
                        query_id, token, validated_interpolation[token])),
                )
            replayed = run_evidence(
                graph, query_id, cypher, validated_params,
                interpolated=[
                    t.strip("<>") for t in qc.template_tokens(query_id)
                ],
                bound={k: v for k, v in validated_interpolation.items()},
                row_cap=(validated_params.get("limit")
                         if isinstance(validated_params.get("limit"), int)
                         else None),
            )
            replay_info = {
                "query_id": query_id,
                "params": validated_params,
                "interpolation": dict(validated_interpolation),
                "recorded_hash": result_hash,
                "replayed_hash": replayed.rows_hash,
                "replayed_rows": replayed.row_count,
                "matches": replayed.rows_hash == result_hash,
            }
            check(
                "live_replay", True,
                f"replayed {query_id} with the original recorded parameters "
                f"and window over all {replayed.row_count} row(s)",
            )
        except (QueryError, ToolValidationError) as exc:
            replayed = None
            replay_possible = False
            replay_reason = f"replay failed ({type(exc).__name__}): {exc}"
            check("live_replay", False, replay_reason)
        except Exception as exc:  # defensive: never let a verifier crash a report
            replayed = None
            replay_possible = False
            replay_reason = f"replay raised {type(exc).__name__}: {exc}"
            check("live_replay", False, replay_reason)
    else:
        check("live_replay", True, replay_reason or "replay not applicable")

    # outcome resolution
    if replayed is None:
        if snapshot_ok:
            return (STATUS_VERIFIED_RECORDED,
                    "preserved snapshot integrity checks pass; "
                    + (replay_reason or "no live replay was applicable"),
                    replay_info, None)
        return (STATUS_UNVERIFIABLE,
                (replay_reason or "no live replay was applicable")
                + "; no preserved snapshot rows to verify against",
                replay_info, None)

    if replayed.rows_hash == result_hash:
        if truncated:
            return (STATUS_VERIFIED_RECORDED,
                    "live replay matches the preserved result, but the "
                    "original observation was truncated, so the claim cannot "
                    "be qualified as fully reproduced",
                    replay_info, replayed)
        return (STATUS_REPRODUCED,
                "an independent read-only replay with the original parameters "
                "reproduced the recorded full-row hash against the current "
                "graph",
                replay_info, replayed)

    explaining = _explained_by_action(query_id, validated_params, actions)
    if explaining is not None:
        return (STATUS_CHANGED,
                "the replay differs from the recorded result and the "
                "difference is consistent with the recorded, authorized "
                f"action {explaining.get('id')} "
                f"({explaining.get('action')} on {explaining.get('account_id')})",
                replay_info, replayed)
    if query_id in _MUTABLE_METADATA_QUERIES:
        # Case/action/simulation metadata legitimately evolves (new linked
        # cases, later actions). The recorded row set is a snapshot of what
        # existed at capture time; exact replay is not expected, and this is
        # reported precisely instead of being called a mismatch.
        return (STATUS_UNVERIFIABLE,
                "case/action metadata changed since capture (for example a new "
                "linked case or later action); the recorded result is a "
                "snapshot of the metadata at capture time and exact "
                "re-derivation is not possible for mutable metadata",
                replay_info, replayed)
    return (STATUS_MISMATCH,
            "the replay differs from the recorded result and no recorded "
            "action explains the difference",
            replay_info, replayed)


def _fan_in_out_derived_consistency(
    results: list[dict[str, Any]], derived: Any,
) -> str | None:
    """Recompute fan_in_out's row-derivable aggregates from replayed rows.

    Mirrors ``src.tools.reads.fan_in_out`` exactly (including rounding) and
    only runs when BOTH the sender and receiver sources replay-matched
    untruncated. Only the fields the agent-level derived schema retains
    (``src.agent.analysis.derive_fan_in_out``) are compared: a missing key or
    a disagreeing value is a derived-result inconsistency; fields that depend
    on tool arguments rather than the rows (min_senders threshold, window) are
    out of scope. Returns a disagreement summary, or None when not applicable.
    """
    if not isinstance(derived, dict) or "distinct_senders" not in derived:
        return None

    def replayed_rows(query_id: str) -> list[Any] | None:
        for item in results:
            if (
                item["query_id"] == query_id
                and item["status"] == STATUS_REPRODUCED
                and not item["truncated"]
                and item.get("replayed") is not None
            ):
                return item["replayed"].rows
        return None

    sender_rows = replayed_rows("fan_in_senders_v1")
    receiver_rows = replayed_rows("fan_out_receivers_v1")
    if sender_rows is None or receiver_rows is None:
        return None
    in_amount = round(sum(r[2] for r in sender_rows), 2)
    out_amount = round(sum(r[2] for r in receiver_rows), 2)
    top_receiver = receiver_rows[0][0] if receiver_rows else None
    top_share = (
        round(receiver_rows[0][2] / out_amount, 4)
        if top_receiver and out_amount > 0 else None
    )
    computed = {
        "distinct_senders": len(sender_rows),
        "distinct_receivers": len(receiver_rows),
        "in_amount": in_amount,
        "out_amount": out_amount,
        "out_in_ratio": round(out_amount / in_amount, 4) if in_amount > 0 else None,
        "top_receiver": top_receiver,
        "top_receiver_share": top_share,
    }
    disagreements = [
        f"{key}: recorded {derived.get(key)!r} vs replayed rows {value!r}"
        for key, value in computed.items()
        if key not in derived or derived.get(key) != value
    ]
    if disagreements:
        return (
            "the recorded derived result disagrees with the replayed source "
            "rows: " + "; ".join(disagreements)
        )
    return None


def verify_evidence_record(
    graph,
    case_id: str,
    record: dict[str, Any],
    *,
    actions: Iterable[dict[str, Any]] = (),
) -> ClaimVerification:
    """Validate one persisted evidence record against the catalog and graph."""
    checks: list[CheckOutcome] = []
    replay_info: dict[str, Any] | None = None

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append(CheckOutcome(name=name, ok=ok, detail=detail))
        return ok

    def finish(
        status: str,
        reason: str,
        *,
        sources: Iterable[dict[str, Any]] = (),
        compound: bool = False,
        source_count: int = 0,
        required: Iterable[str] = (),
    ) -> ClaimVerification:
        source_rows = list(sources)
        return ClaimVerification(
            evidence_id=str(record.get("evidence_id") or ""),
            claim=str(record.get("claim") or ""),
            query_id=record.get("query_id"),
            source_tool=record.get("source_tool"),
            status=status, reason=reason, checks=checks, replay=replay_info,
            sources=[
                SourceVerification(
                    query_id=str(item.get("query_id") or ""),
                    role=str(item.get("role") or "claim_source"),
                    required=bool(item.get("required", True)),
                    status=str(item.get("status") or STATUS_UNVERIFIABLE),
                    reason=str(item.get("reason") or ""),
                    replay=item.get("replay"),
                )
                for item in source_rows
            ],
            compound=compound,
            source_count=source_count,
            required_source_queries=list(required),
            checked_source_queries=_distinct(
                str(item.get("query_id") or "")
                for item in source_rows
                if item.get("status") in _CHECKED_SOURCE_STATUSES
            ),
            unavailable_source_queries=_distinct(
                str(item.get("query_id") or "")
                for item in source_rows
                if item.get("status") == STATUS_UNVERIFIABLE
            ),
            mismatched_source_queries=_distinct(
                str(item.get("query_id") or "")
                for item in source_rows
                if item.get("status") == STATUS_MISMATCH
            ),
        )

    evidence_id = str(record.get("evidence_id") or "")
    claim = str(record.get("claim") or "")
    query_id = record.get("query_id")
    stance = record.get("stance")
    evidence_kind = record.get("evidence_kind")
    result_hash = str(record.get("result_hash") or "")

    # 1. record integrity -----------------------------------------------------
    problems: list[str] = []
    if not evidence_id:
        problems.append("missing evidence_id")
    if not claim:
        problems.append("missing claim text")
    if not isinstance(query_id, str) or not query_id:
        problems.append("missing query_id")
    if stance not in _STANCES:
        problems.append(f"stance {stance!r} is not supporting/contradicting/unresolved")
    if evidence_kind not in _EVIDENCE_KINDS:
        problems.append(f"evidence_kind {evidence_kind!r} is not a known kind")
    if not (result_hash.startswith("sha256:") and len(result_hash) == 7 + _HASH_RE_LEN):
        problems.append("result_hash is not 'sha256:' plus 64 hex chars")
    row_count = record.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        problems.append("row_count is not a non-negative integer")
    if not isinstance(record.get("truncated"), bool):
        problems.append("truncated is not a boolean")
    associated_case = record.get("case_id")
    if associated_case and associated_case != case_id:
        problems.append(
            f"record is linked to case {associated_case!r}, not {case_id!r}"
        )
    check("record_integrity", not problems, "; ".join(problems) or "record fields are well formed")
    if problems:
        return finish(STATUS_MISMATCH, "malformed evidence record: " + "; ".join(problems))

    # 2. source manifest (Phase 6.1) ----------------------------------------------
    extracted = _extract_sources(record)
    manifest_sources = extracted["sources"]
    manifest_problems: list[str] = list(extracted["problems"])
    compound = bool(extracted["compound"])
    required_queries: list[str] = list(extracted["required_source_queries"])
    # Pre-6.1 legacy records of a multi-query tool whose only recorded query is
    # the observation-window anchor cannot carry the tool's claim: the
    # claim-bearing sources were never recorded, so replaying the anchor must
    # never brand such a record fully reproduced.
    legacy_anchor_capture = (
        not compound
        and str(record.get("source_tool") or "") in _TOOL_REQUIRED_SOURCES
        and str(record.get("query_id") or "") == _WINDOW_ANCHOR_QUERY
    )
    if manifest_problems:
        check("source_manifest", False, "; ".join(manifest_problems))
        return finish(
            STATUS_MISMATCH,
            "source manifest integrity failed: " + "; ".join(manifest_problems),
            sources=[], compound=compound, source_count=len(manifest_sources),
            required=required_queries,
        )
    check(
        "source_manifest", True,
        f"compound source manifest validated: {len(manifest_sources)} source "
        "query(ies); every source is verified individually"
        if compound else
        "legacy citation: only the cited query is recorded; sub-query coverage "
        "was not captured",
    )

    # 3. per-source verification (Phase 6.1) ---------------------------------------
    composite = compound and len(manifest_sources) > 1
    source_results: list[dict[str, Any]] = []
    for ordinal, source in enumerate(manifest_sources):
        prefix = f"{source.get('query_id')}: " if composite else ""
        status, reason, source_replay, replayed = _verify_source(
            graph, source, actions=actions, checks=checks, name_prefix=prefix,
        )
        source_results.append({
            "query_id": str(source.get("query_id") or ""),
            "role": str(source.get("role") or "claim_source"),
            "required": bool(source.get("required", True)),
            "status": status,
            "reason": reason,
            "replay": source_replay,
            "replayed": replayed,
            "truncated": bool(source.get("truncated")),
        })
        if ordinal == 0:
            replay_info = source_replay

    # 4. derived-result consistency (fan_in_out) -----------------------------------
    # A full fan_in_out capture (its manifest carries the window anchor) must
    # have its recorded derived aggregates verified whenever both source
    # queries replay-match: the claim is never labeled fully reproduced from
    # the source rows alone.
    derived_parsed, _derived_err = _parse_json_field(record, "derived_json")
    full_capture = compound and any(
        str(source.get("query_id") or "") == _WINDOW_ANCHOR_QUERY
        for source in manifest_sources
    )
    if record.get("source_tool") == "fan_in_out" and full_capture:
        both_reproduced = (
            any(r["query_id"] == "fan_in_senders_v1"
                and r["status"] == STATUS_REPRODUCED for r in source_results)
            and any(r["query_id"] == "fan_out_receivers_v1"
                    and r["status"] == STATUS_REPRODUCED for r in source_results)
        )
        if not (
            isinstance(derived_parsed, dict)
            and "distinct_senders" in derived_parsed
        ):
            if both_reproduced:
                message = (
                    "the recorded derived result is missing or malformed, so "
                    "the fan-in/fan-out aggregates cannot be verified; the "
                    "claim is not labeled fully reproduced without them"
                )
                check("derived_consistency", False, message)
                return finish(
                    STATUS_MISMATCH,
                    "derived-result verification failed: " + message,
                    sources=[], compound=compound,
                    source_count=len(manifest_sources),
                    required=required_queries,
                )
        else:
            disagreement = _fan_in_out_derived_consistency(
                source_results, derived_parsed
            )
            if disagreement:
                check("derived_consistency", False, disagreement)
                return finish(
                    STATUS_MISMATCH,
                    "derived-result verification failed: " + disagreement,
                    sources=[], compound=compound,
                    source_count=len(manifest_sources),
                    required=required_queries,
                )
            if both_reproduced:
                check(
                    "derived_consistency", True,
                    "fan-in/fan-out aggregates recomputed from the replayed "
                    "source rows match the recorded derived result",
                )

    # 5. aggregate outcome by the documented precedence -----------------------------
    statuses = [r["status"] for r in source_results]
    aggregate = next(
        (s for s in _STATUS_PRECEDENCE if s in statuses), STATUS_UNVERIFIABLE
    )
    if len(source_results) == 1:
        reason = source_results[0]["reason"]
    else:
        summary = ", ".join(
            f"{r['query_id']}={r['status']}" for r in source_results
        )
        if aggregate == STATUS_REPRODUCED:
            reason = (
                f"all {len(source_results)} recorded source queries were "
                "independently replayed with the original parameters and "
                f"reproduced their recorded full-row hashes ({summary})"
            )
        elif aggregate == STATUS_VERIFIED_RECORDED:
            reason = (
                f"all {len(source_results)} recorded source queries passed "
                f"their integrity checks without a full live replay ({summary})"
            )
        else:
            worst = next(
                r for r in source_results if r["status"] == aggregate
            )
            reason = f"{worst['reason']} | source outcomes: {summary}"

    if legacy_anchor_capture and aggregate in (STATUS_REPRODUCED, STATUS_CHANGED):
        check(
            "source_coverage", False,
            "legacy citation records only the observation-window anchor "
            "(txn_max_ts_v1); the tool's claim-bearing source queries were "
            "never recorded, so a full verification is impossible",
        )
        aggregate = STATUS_UNVERIFIABLE
        reason = (
            "the recorded citation captures only the observation-window anchor "
            "query (pre-6.1 legacy record); the claim-bearing source queries of "
            "the tool invocation were never recorded, so the claim cannot be "
            "verified from this record"
        )

    return finish(
        aggregate, reason,
        sources=source_results,
        compound=compound,
        source_count=len(manifest_sources),
        required=required_queries,
    )


# ---- current-state checks ----------------------------------------------------


def _current_state_checks(graph, case_id: str, records: CaseRecords) -> list[dict[str, Any]]:
    """Independently re-read the current graph state for the case's entities."""
    out: list[dict[str, Any]] = []
    for entity in records.about_entities:
        exists_ev = _read(graph, "entity_exists_v1", {"entity_id": entity}, row_cap=1)
        if not exists_ev.rows:
            out.append({"entity_id": entity, "kind": None, "status": None,
                        "check": "entity_exists", "ok": False})
            continue
        kind = exists_ev.rows[0][1]
        entry: dict[str, Any] = {"entity_id": entity, "kind": kind,
                                 "check": "entity_exists", "ok": True}
        if kind == "Account":
            st = _read(graph, "account_state_v1", {"account_id": entity}, row_cap=1)
            entry["status"] = st.rows[0][1] if st.rows else None
            entry["check"] = "account_state"
        out.append(entry)
    return out


def _post_action_checks(graph, case_id: str, records: CaseRecords) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for action in records.actions:
        if action.get("action") != "freeze_account" or action.get("status") != "applied":
            continue
        envelope = verify_freeze(graph, case_id, str(action.get("account_id")))
        result = envelope.result if envelope.ok else {}
        out.append({
            "action_id": action.get("id"),
            "account_id": action.get("account_id"),
            "verified": bool(result.get("verified")),
            "account_status": result.get("account_status"),
            "active_incident_edges": result.get("active_incident_edges"),
            "active_edges_total": result.get("active_edges_total"),
            "simulation_expected_active_edges": result.get("simulation_expected_active_edges"),
            "checks": result.get("checks") or {},
        })
    return out


# ---- claim references --------------------------------------------------------


_STATUS_PRECEDENCE = (
    STATUS_MISMATCH, STATUS_CHANGED, STATUS_UNVERIFIABLE,
    STATUS_REPRODUCED, STATUS_VERIFIED_RECORDED,
)

_STATUS_REASONS = {
    STATUS_MISMATCH: "at least one referenced evidence record is a mismatch",
    STATUS_CHANGED: "the strongest referenced evidence changed after a recorded action",
    STATUS_UNVERIFIABLE: "no referenced evidence record could be independently checked",
    STATUS_REPRODUCED: "a referenced evidence record was independently reproduced",
    STATUS_VERIFIED_RECORDED: "referenced evidence snapshots pass their integrity checks",
}


def verify_references(
    references: Iterable[Mapping[str, Any]],
    claims_by_id: Mapping[str, ClaimVerification],
) -> list[ReferenceVerification]:
    """Map material report claims (evidence-id references) to outcomes.

    A claim points at one or more persisted evidence ids. Missing ids count as
    unverifiable; the claim's outcome is the strongest (by precedence) outcome
    across its references, so an unverifiable or mismatched backing record can
    never be hidden behind a good one. Phase 6.1: the referenced records'
    per-source outcomes are folded in, so the report can state which source
    queries were checked and which remain unavailable per claim.
    """
    out: list[ReferenceVerification] = []
    for ref in references:
        claim = str(ref.get("claim") or "")
        source = str(ref.get("source") or "report")
        ids = [str(i) for i in (ref.get("evidence_ids") or [])]
        per: list[dict[str, Any]] = []
        missing: list[str] = []
        statuses: list[str] = []
        query_status: dict[str, str] = {}
        for evidence_id in ids:
            claim_verification = claims_by_id.get(evidence_id)
            if claim_verification is None:
                missing.append(evidence_id)
                per.append({"evidence_id": evidence_id,
                            "status": STATUS_UNVERIFIABLE, "queries": []})
                statuses.append(STATUS_UNVERIFIABLE)
                continue
            per.append({
                "evidence_id": evidence_id,
                "status": claim_verification.status,
                "queries": [
                    {"query_id": s.query_id, "status": s.status,
                     "required": s.required, "role": s.role}
                    for s in claim_verification.sources
                ],
            })
            statuses.append(claim_verification.status)
            for s in claim_verification.sources:
                current = query_status.get(s.query_id)
                if current is None or (
                    _STATUS_PRECEDENCE.index(s.status)
                    < _STATUS_PRECEDENCE.index(current)
                ):
                    query_status[s.query_id] = s.status
        if not ids:
            status = STATUS_UNVERIFIABLE
            reason = "the claim references no evidence at all"
        else:
            status = next((s for s in _STATUS_PRECEDENCE if s in statuses),
                          STATUS_UNVERIFIABLE)
            reason = _STATUS_REASONS[status]
            if missing:
                reason += f"; references missing evidence ids {sorted(missing)}"
        checked = sorted(
            qid for qid, st in query_status.items()
            if st in _CHECKED_SOURCE_STATUSES
        )
        unavailable = sorted(
            qid for qid, st in query_status.items() if st == STATUS_UNVERIFIABLE
        )
        mismatched = sorted(
            qid for qid, st in query_status.items() if st == STATUS_MISMATCH
        )
        out.append(ReferenceVerification(
            claim=claim, source=source, evidence_ids=ids,
            missing_evidence_ids=sorted(missing), per_evidence=per,
            checked_source_queries=checked,
            unavailable_source_queries=unavailable,
            mismatched_source_queries=mismatched,
            status=status, reason=reason,
        ))
    return out


# ---- case-level verification --------------------------------------------------


def verify_case(
    graph,
    case_id: str,
    *,
    claim_refs: Iterable[Mapping[str, Any]] | None = None,
    records: CaseRecords | None = None,
) -> CaseVerification:
    """Independently verify every persisted evidence record of one case.

    ``claim_refs`` optionally maps material report claims to evidence ids so a
    report-level claim view can be computed from the SAME per-record outcomes.
    ``records`` may supply a pre-loaded :class:`CaseRecords` so callers that
    already read the case (the report builder) do not read it twice.
    """
    records = records or load_case_records(graph, case_id)
    if records.case is None:
        return CaseVerification(
            case_id=case_id, exists=False,
            notes=list(records.notes),
        )

    claims: list[ClaimVerification] = []
    for record in records.evidence:
        claims.append(verify_evidence_record(
            graph, case_id, record, actions=records.actions,
        ))
    summary = {status: 0 for status in ALL_STATUSES}
    for claim in claims:
        summary[claim.status] = summary.get(claim.status, 0) + 1

    references = verify_references(claim_refs or [], {c.evidence_id: c for c in claims})
    reference_summary = {status: 0 for status in ALL_STATUSES}
    for ref in references:
        reference_summary[ref.status] = reference_summary.get(ref.status, 0) + 1

    notes: list[str] = []
    unverifiable = [c.evidence_id for c in claims if c.status == STATUS_UNVERIFIABLE]
    if unverifiable:
        notes.append(
            "Records without a preserved snapshot and without a replayable "
            f"query remain unverifiable: {sorted(unverifiable)}"
        )
    mismatches = [c.evidence_id for c in claims if c.status == STATUS_MISMATCH]
    if mismatches:
        notes.append(f"Mismatches require review: {sorted(mismatches)}")
    write_path = [c.evidence_id for c in claims
                  if c.query_id in qc.WRITE_QUERY_IDS]
    if write_path:
        notes.append(
            f"Write-path records are never replayed: {sorted(write_path)}"
        )
    compound = [c for c in claims if c.compound]
    if compound:
        notes.append(
            f"{len(compound)} record(s) carry a compound source manifest; "
            "every recorded source query was verified individually"
        )
    unavailable_sources = sorted({
        s.query_id for c in claims for s in c.sources
        if s.status == STATUS_UNVERIFIABLE
    })
    if unavailable_sources:
        notes.append(
            "Source queries that could not be independently checked: "
            f"{unavailable_sources}"
        )
    legacy = [c.evidence_id for c in claims if not c.compound]
    if legacy:
        notes.append(
            "Legacy citations without a source manifest (sub-query coverage "
            f"was not recorded): {sorted(legacy)}"
        )

    return CaseVerification(
        case_id=case_id, exists=True, claims=claims, summary=summary,
        references=references, reference_summary=reference_summary,
        current_state=_current_state_checks(graph, case_id, records),
        post_action=_post_action_checks(graph, case_id, records),
        notes=notes,
    )
