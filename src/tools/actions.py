"""Phase 5 action tools: persistent case memory, counterfactual simulation,
and the guarded sandbox freeze (PLAN.md §5/§7/§8/§9).

Boundaries enforced here:

- This is the ONLY module permitted to execute write templates
  (``query_catalog.WRITE_QUERY_IDS``). There is no generic write or
  arbitrary-Cypher tool anywhere in the MCP surface.
- Every mutation requires an explicitly configured sandbox
  (``Settings.is_sandbox``: HABEAS_DEMO_MODE plus a demo-safe graph name) and
  ``HABEAS_ENABLE_ACTIONS``. The graph name always comes from configuration,
  never from tool arguments.
- ``freeze_account`` re-runs the deterministic eligibility policy itself; a
  caller-supplied flag or an LLM's judgement cannot authorize a mutation.
- If any precondition fails, nothing is written.
- Historical evidence is preserved: the pre-action simulation and the cited
  evidence are never overwritten by the mutated graph's state.

Metric semantics (PLAN.md §8) are kept strictly separate:
``downstream_accounts_before/after`` are unique account sets reached by a
bounded Python BFS over the active projection; ``direct_outgoing_volume`` is
the sum of *unique* transaction events in the window. Paths are never summed,
so overlapping routes cannot double-count. No metric is called "funds frozen"
or "cash stranded": the graph stores no accounting model to justify that.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from src.agent import policy as action_policy
from src.config import Settings, get_settings
from src.tools import query_catalog as qc
from src.tools.envelopes import (
    QueryEvidence,
    ToolEnvelope,
    ToolValidationError,
    error_envelope,
    make_envelope,
    rows_hash as compute_rows_hash,
    run_evidence,
)

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# A stored result hash is always "sha256:" + 64 hex chars (see envelopes).
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

DEFAULT_TOPOLOGY_LIMIT = 5000

STANCE_SUPPORTING = "supporting"
STANCE_CONTRADICTING = "contradicting"
STANCE_UNRESOLVED = "unresolved"

EVIDENCE_KIND_IMMUTABLE = "immutable_history"
EVIDENCE_KIND_CURRENT = "current_state"
EVIDENCE_KIND_SNAPSHOT = "captured_snapshot"

ACTION_FREEZE = "freeze_account"

# Phase 6.1 source-manifest roles: the observation-window anchor positions the
# window ("as of" / cutoff) but cannot carry a claim on its own; claim-material
# queries carry the cited observation. Roles and fingerprints are derived
# server-side, never accepted from the caller.
ANCHOR_QUERIES = frozenset({"txn_max_ts_v1"})
SOURCE_ROLE_WINDOW_ANCHOR = "window_anchor"
SOURCE_ROLE_CLAIM = "claim_source"

# Serializes the read-check-mutate section of freeze_account within this
# process (the demo deployment runs exactly one MCP server process; tests
# exercise concurrent attempts with threads). Cross-process duplicates are
# still blocked by the guarded database write itself: freeze_account_v1 only
# updates an account whose status is not already 'frozen', and the action
# record is keyed by (case, account), so a second mutation is impossible even
# without the lock.
_FREEZE_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FMT)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _safe_json(value: Any, default: Any) -> Any:
    """Parse a stored JSON text field, falling back to ``default``."""
    if not value:
        return default
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _canonical_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _write(graph, query_id: str, params: dict[str, Any], *, row_cap: int | None = None):
    """Run one approved WRITE template. Refuses anything not in the catalog."""
    if not qc.is_write_query(query_id):
        raise ToolValidationError(
            f"{query_id!r} is not an approved write template; refusing to execute"
        )
    return run_evidence(
        graph, query_id, qc.get_template(query_id), params,
        row_cap=row_cap, read_only=False,
    )


def _read(graph, query_id: str, params: dict[str, Any], *, row_cap: int | None = None):
    return run_evidence(graph, query_id, qc.get_template(query_id), params, row_cap=row_cap)


def _require_sandbox(settings: Settings) -> None:
    if not settings.habeas_enable_actions:
        raise ToolValidationError(
            "action tools are disabled; set HABEAS_ENABLE_ACTIONS=true to opt in"
        )
    if not settings.is_sandbox:
        raise ToolValidationError(
            "refusing to mutate: this is not the disposable sandbox graph "
            "(requires HABEAS_DEMO_MODE=true and a demo-safe graph name)"
        )


def _window(graph, window_days: int) -> tuple[dict[str, Any], QueryEvidence]:
    ev = _read(graph, "txn_max_ts_v1", {})
    max_ts = ev.rows[0][0] if ev.rows else None
    if max_ts:
        cutoff = (
            datetime.strptime(max_ts, _TS_FMT) - timedelta(days=window_days)
        ).strftime(_TS_FMT)
    else:
        cutoff = ""
    return {"as_of": max_ts, "window_days": window_days, "cutoff": cutoff}, ev


def reachable_accounts(
    edges: Iterable[tuple[str, str]], start: str, max_hops: int
) -> set[str]:
    """Unique accounts reachable from ``start`` within ``max_hops`` (excl. start)."""
    adjacency: dict[str, list[str]] = {}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
    seen: set[str] = set()
    frontier = [start]
    for _ in range(max_hops):
        nxt: list[str] = []
        for node in frontier:
            for neighbour in adjacency.get(node, []):
                if neighbour not in seen and neighbour != start:
                    seen.add(neighbour)
                    nxt.append(neighbour)
        if not nxt:
            break
        frontier = nxt
    return seen


# ---- case memory -----------------------------------------------------------


def case_id_for_alert(alert_id: str) -> str:
    """Deterministic case id: one case per alert, so repeats cannot duplicate."""
    return f"CASE-{alert_id}"


def reconstruct_case_state(graph, case_id: str):
    """Rebuild a CaseState from the PERSISTED case and its cited evidence.

    This is what lets the MCP freeze tool re-derive eligibility without any
    model input: the evidence was captured by real tool calls at investigation
    time, and the policy is then applied to that persisted record — not to a
    fresh claim by the caller.
    """
    state, _ = _reconstruct_case_state_full(graph, case_id)
    return state


def _reconstruct_case_state_full(
    graph, case_id: str,
) -> tuple[Any, list[QueryEvidence]]:
    """Rebuild the CaseState and return the approved-query evidence it read."""
    from src.agent import analysis
    from src.agent.state import CaseState, EvidenceRecord

    case_ev = _read(graph, "case_lookup_v1", {"case_id": case_id}, row_cap=1)
    if not case_ev.rows:
        raise ToolValidationError(f"case {case_id} does not exist")
    case = case_ev.rows[0][0] or {}

    ev_ev = _read(graph, "case_evidence_v1", {"case_id": case_id, "limit": 500},
                  row_cap=500)
    records: list[EvidenceRecord] = []
    for row in ev_ev.rows:
        cols = ev_ev.columns
        item = dict(zip(cols, row))
        # Phase 6.1: reconstruct the COMPLETE source manifest when it was
        # persisted, so the policy and report layers see every contributing
        # sub-query (truncation flags included). Legacy records (no manifest)
        # fall back to the single top-level citation as before.
        raw_sources = _safe_json(item.get("sources_json"), None)
        sources = (
            [s for s in raw_sources
             if isinstance(s, dict) and s.get("query_id")]
            if isinstance(raw_sources, list) else []
        )
        if sources:
            query_ids = [str(s["query_id"]) for s in sources]
            params = [_safe_dict(s.get("params")) for s in sources]
            rows_hash = [str(s.get("result_hash") or "") for s in sources]
            row_count = [_safe_int(s.get("row_count")) for s in sources]
            truncated = [bool(s.get("truncated")) for s in sources]
            bounds = [_safe_dict(s.get("interpolation")) for s in sources]
        else:
            query_ids = [item.get("query_id")] if item.get("query_id") else []
            params = [_safe_dict(_safe_json(item.get("params_json"), {}))]
            rows_hash = [item.get("result_hash") or ""]
            row_count = [_safe_int(item.get("row_count"))]
            truncated = [bool(item.get("truncated"))]
            bounds = [_safe_dict(_safe_json(item.get("interpolation_json"), {}))]
        records.append(EvidenceRecord(
            evidence_id=item["evidence_id"],
            tool=item.get("source_tool") or "",
            stage="persisted",
            ok=True,
            interpretation=item.get("interpretation") or "",
            query_ids=query_ids,
            params=params,
            rows_hash=rows_hash,
            row_count=row_count,
            truncated=truncated,
            bounds=bounds,
            derived=_safe_dict(_safe_json(item.get("derived_json"), {})),
        ))

    state = CaseState(
        alert_id=case.get("alert_id") or "",
        case_id=case_id,
        provisional_verdict=case.get("verdict"),
        evidence=records,
    )
    entities = _read(graph, "case_about_entities_v1",
                     {"case_id": case_id, "limit": 100}, row_cap=100)
    state.primary_entity_ids = [row[0] for row in entities.rows]
    state.hypotheses = analysis.recompute_hypotheses(state)
    return state, [case_ev, ev_ev, entities]


def create_case(
    graph,
    alert_id: str,
    *,
    title: str | None = None,
    severity: str = "medium",
    verdict: str | None = None,
    summary: str = "",
    entity_ids: Iterable[str] = (),
    reuses_case_ids: Iterable[str] = (),
    settings: Settings | None = None,
) -> ToolEnvelope:
    """Create the case for an alert, or return the existing one (idempotent)."""
    t0 = time.perf_counter()
    settings = settings or get_settings()
    try:
        _require_sandbox(settings)
        alert_id = qc.validate_entity_id(alert_id, "alert_id")
        severity = qc.validate_severity(severity)
        canonical = action_policy.canonical_verdict(verdict)
        case_id = case_id_for_alert(alert_id)

        existing = _read(graph, "case_lookup_v1", {"case_id": case_id}, row_cap=1)
        evidence: list[QueryEvidence] = [existing]
        created_at = _now()
        created = True
        if existing.rows:
            props = existing.rows[0][0] or {}
            created = False
            created_at = props.get("created_at") or created_at

        # Validate EVERY input before the first write, so an invalid reference
        # can never leave a half-written case behind.
        entity_list: list[str] = []
        for entity_id in dict.fromkeys(entity_ids):
            entity_list.append(qc.validate_entity_id(entity_id, "entity_id"))

        # REUSES links: pointers to earlier cases whose verdicts are LEADS to
        # re-check (PLAN.md §7), never inherited truth. Referencing a missing
        # case is refused rather than silently skipped.
        reuse_targets: list[str] = []
        for older_case_id in dict.fromkeys(reuses_case_ids):
            older_case_id = qc.validate_entity_id(older_case_id, "older_case_id")
            if older_case_id == case_id:
                raise ToolValidationError(
                    f"case {case_id} cannot reuse itself; reuses_case_ids must "
                    "reference earlier cases"
                )
            older = _read(graph, "case_lookup_v1", {"case_id": older_case_id}, row_cap=1)
            evidence.append(older)
            if not older.rows:
                raise ToolValidationError(
                    f"prior case {older_case_id} does not exist; refusing to "
                    "record a REUSES link to a missing case"
                )
            reuse_targets.append(older_case_id)

        now = _now()
        title = title or f"Investigation of {alert_id}"
        upsert = _write(
            graph, "case_upsert_v1",
            {
                "case_id": case_id, "alert_id": alert_id, "title": title,
                "status": "open", "verdict": canonical, "severity": severity,
                "summary": summary, "created_at": created_at, "updated_at": now,
            },
        )
        evidence.append(upsert)

        for entity_id in entity_list:
            evidence.append(_write(graph, "case_about_v1",
                                   {"case_id": case_id, "entity_id": entity_id}))

        for older_case_id in reuse_targets:
            evidence.append(_write(
                graph, "case_reuses_v1",
                {"case_id": case_id, "older_case_id": older_case_id},
            ))

        result = {
            "case_id": case_id, "alert_id": alert_id, "created": created,
            "status": "open", "verdict": canonical, "severity": severity,
            "title": title, "created_at": created_at, "updated_at": now,
            "entity_ids": entity_list, "reused_case_ids": reuse_targets,
        }
        interpretation = (
            f"Case {case_id} for alert {alert_id} was "
            + ("created" if created else "reused (already existed; timestamps preserved)")
            + f" with canonical verdict {canonical!r} and severity {severity}, linked "
            f"to {len(entity_list)} entity/entities. Case ids derive from the alert "
            "id, so repeated investigations cannot create duplicate cases."
        )
        if reuse_targets:
            interpretation += (
                f" Earlier case(s) linked via REUSES: {reuse_targets}; their "
                "verdicts are leads to re-check against current evidence, not "
                "settled truth."
            )
        return make_envelope("create_case", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("create_case", exc, t0)


def _source_role(query_id: str) -> str:
    return (
        SOURCE_ROLE_WINDOW_ANCHOR if query_id in ANCHOR_QUERIES
        else SOURCE_ROLE_CLAIM
    )


def _normalize_source_entry(
    entry: Any, claim: str, ordinal: int
) -> dict[str, Any]:
    """Validate ONE per-sub-query source record of a compound citation.

    Mirrors the citation integrity gate per source: approved query id,
    well-formed hash (reconciled with any supplied rows/columns snapshot),
    non-negative row count, boolean truncation, query-specific interpolation
    bounds, and the server-computed catalog fingerprint + role.
    """
    where = f"citation for {claim!r} source[{ordinal}]"
    if not isinstance(entry, dict):
        raise ToolValidationError(f"{where} must be an object")
    query_id = entry.get("query_id")
    if query_id not in qc._QUERY_TEMPLATES:
        raise ToolValidationError(
            f"{where} references unapproved query id {query_id!r}; only "
            "approved catalog queries may be cited"
        )
    params = entry.get("params") or {}
    if not isinstance(params, dict):
        raise ToolValidationError(f"{where} params must be a JSON object")
    interpolation = qc.validate_interpolation(
        query_id, entry.get("interpolation") or {}
    )
    rows = list(entry.get("rows") or [])
    columns = list(entry.get("columns") or [])
    if columns and not all(isinstance(c, str) for c in columns):
        raise ToolValidationError(f"{where} has non-string column names")
    if rows and not columns:
        raise ToolValidationError(
            f"{where} supplies non-empty rows without columns; the snapshot "
            "hash cannot be verified, so it is refused"
        )
    supplied_hash = entry.get("result_hash") or None
    if supplied_hash is not None and not (
        isinstance(supplied_hash, str) and _HASH_RE.match(supplied_hash)
    ):
        raise ToolValidationError(
            f"{where} has a malformed result_hash {supplied_hash!r}; expected "
            "'sha256:' plus 64 hex chars"
        )
    if rows:
        expected_hash = compute_rows_hash(columns, rows)
        if supplied_hash is None:
            supplied_hash = expected_hash
        elif supplied_hash != expected_hash:
            raise ToolValidationError(
                f"{where} declares a result_hash that does not match its "
                "normalized rows; refusing to persist inconsistent evidence"
            )
    result_hash = supplied_hash or compute_rows_hash(columns, rows)
    row_count = entry.get("row_count")
    if row_count is None:
        row_count = len(rows)
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise ToolValidationError(f"{where} row_count must be a non-negative integer")
    return {
        "query_id": query_id,
        "role": _source_role(query_id),
        "params": params,
        "interpolation": interpolation,
        "result_hash": result_hash,
        "row_count": row_count,
        "truncated": bool(entry.get("truncated")),
        "template_sha256": qc.template_sha256(query_id),
        "columns": columns,
        "rows": rows,
    }


def _build_sources_manifest(
    citation: dict[str, Any],
    claim: str,
    *,
    query_id: str,
    result_hash: str,
    rows: list[Any],
    columns: list[Any],
    interpolation: dict[str, int],
    row_count: int,
    truncated: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build + validate the complete source manifest for one citation.

    When the caller supplies ``sources`` (the workflow does: every executed
    approved sub-query), each entry is validated individually and the
    top-level citation fields must mirror sources[0]. Without ``sources`` the
    citation is recorded as a single-source manifest so no new record is ever
    persisted without a manifest. ``required_source_queries`` defaults to the
    claim-material query ids (everything except window anchors).
    """
    raw = citation.get("sources")
    if raw is None:
        raw = [{
            "query_id": query_id,
            "params": citation.get("params") or {},
            "interpolation": interpolation,
            "result_hash": result_hash,
            "row_count": row_count,
            "truncated": truncated,
            "rows": rows,
            "columns": columns,
        }]
    if not isinstance(raw, list) or not raw:
        raise ToolValidationError(
            f"citation for {claim!r} has a malformed sources manifest; expected "
            "a non-empty list of per-sub-query source records"
        )
    manifest = [
        _normalize_source_entry(entry, claim, ordinal)
        for ordinal, entry in enumerate(raw)
    ]
    primary = manifest[0]
    if primary["query_id"] != query_id or primary["result_hash"] != result_hash:
        raise ToolValidationError(
            f"citation for {claim!r}: the top-level citation fields must mirror "
            "sources[0]; refusing to persist an inconsistent source manifest"
        )
    required = citation.get("required_source_queries")
    if required is None:
        required = [
            s["query_id"] for s in manifest if s["role"] == SOURCE_ROLE_CLAIM
        ]
    if not isinstance(required, list) or not all(
        isinstance(q, str) for q in required
    ):
        raise ToolValidationError(
            f"citation for {claim!r} required_source_queries must be a list of "
            "query ids"
        )
    required = list(dict.fromkeys(required))
    manifest_qids = {s["query_id"] for s in manifest}
    for required_qid in required:
        if required_qid not in qc._QUERY_TEMPLATES:
            raise ToolValidationError(
                f"citation for {claim!r} declares unapproved required source "
                f"query id {required_qid!r}"
            )
        if required_qid not in manifest_qids:
            raise ToolValidationError(
                f"citation for {claim!r} declares required source query "
                f"{required_qid!r} that is missing from its source manifest"
            )
    return manifest, required


def _manifest_signature(
    result_hash: str, sources: Any
) -> list[list[Any]]:
    """Comparable (query_id, result_hash) signature of a stored manifest."""
    signature: list[list[Any]] = []
    if isinstance(sources, list):
        for entry in sources:
            if isinstance(entry, dict) and (entry.get("query_id") or
                                            entry.get("result_hash")):
                signature.append([entry.get("query_id"), entry.get("result_hash")])
    return signature or [[None, result_hash]]


def _parsed_sources(sources_json: Any) -> Any:
    if not sources_json:
        return None
    try:
        parsed = json.loads(sources_json)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def cite_evidence(
    graph,
    case_id: str,
    citations: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> ToolEnvelope:
    """Persist references to evidence already returned during the investigation.

    Each citation must name a query id from the approved catalog and carry the
    normalized rows/hash captured at investigation time. Rows are stored as
    JSON text and are never executed as Cypher.
    """
    t0 = time.perf_counter()
    settings = settings or get_settings()
    try:
        _require_sandbox(settings)
        case_id = qc.validate_entity_id(case_id, "case_id")
        if not citations:
            raise ToolValidationError("at least one citation is required")

        evidence: list[QueryEvidence] = []
        stored: list[dict[str, Any]] = []
        for citation in citations:
            claim = str(citation.get("claim") or "").strip()
            if not claim:
                raise ToolValidationError("every citation needs a non-empty claim")
            stance = citation.get("stance", STANCE_SUPPORTING)
            if stance not in {STANCE_SUPPORTING, STANCE_CONTRADICTING, STANCE_UNRESOLVED}:
                raise ToolValidationError(
                    "stance must be supporting/contradicting/unresolved, got "
                    f"{stance!r}"
                )
            query_id = citation.get("query_id")
            if query_id not in qc._QUERY_TEMPLATES:
                raise ToolValidationError(
                    f"citation for {claim!r} references unapproved query id "
                    f"{query_id!r}; only approved catalog queries may be cited"
                )
            local_id = qc.validate_entity_id(
                str(citation.get("evidence_id") or ""), "evidence_id"
            )
            evidence_id = f"{case_id}:{local_id}"
            rows = citation.get("rows") or []
            columns = citation.get("columns") or []
            # Integrity gate (PLAN.md §7/§11): a supplied hash must be a
            # well-formed sha256; when normalized rows AND their columns are
            # supplied, the hash must equal the canonical envelope hash over
            # them. Rows that disagree with the declared hash are never
            # persisted, and non-empty rows without columns are refused so a
            # snapshot can never be stored in an unverifiable shape.
            supplied_hash = citation.get("result_hash") or None
            if supplied_hash is not None and not (
                isinstance(supplied_hash, str) and _HASH_RE.match(supplied_hash)
            ):
                raise ToolValidationError(
                    f"citation for {claim!r} has a malformed result_hash "
                    f"{supplied_hash!r}; expected 'sha256:' plus 64 hex chars"
                )
            if rows:
                if not columns:
                    raise ToolValidationError(
                        f"citation for {claim!r} supplies non-empty rows without "
                        "columns; the snapshot hash cannot be verified, so it is "
                        "refused"
                    )
                expected_hash = compute_rows_hash(list(columns), rows)
                if supplied_hash is None:
                    supplied_hash = expected_hash
                elif supplied_hash != expected_hash:
                    raise ToolValidationError(
                        f"citation for {claim!r} declares a result_hash that does "
                        "not match its normalized rows; refusing to persist "
                        "inconsistent evidence"
                    )
            result_hash = supplied_hash or compute_rows_hash(list(columns), rows)

            # Phase 6 audit fidelity: record the validated interpolation bounds
            # and (server-side, never caller-supplied) the catalog template
            # fingerprint, so the verifier can rebuild the exact bounded query
            # and detect template drift.
            if columns and not all(isinstance(c, str) for c in columns):
                raise ToolValidationError(
                    f"citation for {claim!r} has non-string column names"
                )
            interpolation = qc.validate_interpolation(
                query_id, citation.get("interpolation") or {}
            )
            primary_row_count = int(citation.get("row_count") or len(rows))

            # Phase 6.1 claim-level coverage: persist the COMPLETE source
            # manifest of the tool invocation — one record per executed
            # approved sub-query — not just the first one. The top-level
            # citation fields keep mirroring sources[0]; every source is
            # validated individually, fingerprinted server-side, and
            # classified as window anchor vs claim-material.
            manifest, required_queries = _build_sources_manifest(
                citation, claim,
                query_id=query_id, result_hash=result_hash,
                rows=rows, columns=columns, interpolation=interpolation,
                row_count=primary_row_count,
                truncated=bool(citation.get("truncated")),
            )
            sources_json = _json(manifest)
            manifest_signature = _manifest_signature(result_hash, manifest)

            # Never overwrite an earlier observation (PLAN.md §7): if this
            # evidence id already holds a DIFFERENT result or source manifest,
            # the new observation is stored under a versioned id so the
            # original record stays auditable. An identical manifest is an
            # idempotent re-citation.
            versioned = False
            existing = _read(graph, "evidence_lookup_v1",
                             {"evidence_id": evidence_id}, row_cap=1)
            evidence.append(existing)

            def _stored_signature(probe) -> Any:
                if not probe.rows:
                    return None
                row = probe.rows[0]
                probe_sources = _parsed_sources(
                    row[2] if len(row) > 2 else None
                )
                return _manifest_signature(row[0], probe_sources)

            if existing.rows and (
                existing.rows[0][0] != result_hash
                or _stored_signature(existing) != manifest_signature
            ):
                version = 2
                while True:
                    if version > 50:
                        raise ToolValidationError(
                            f"too many stored versions for {evidence_id}"
                        )
                    candidate = f"{evidence_id}-v{version}"
                    probe = _read(graph, "evidence_lookup_v1",
                                  {"evidence_id": candidate}, row_cap=1)
                    if not probe.rows or _stored_signature(probe) == manifest_signature:
                        evidence.append(probe)
                        evidence_id = candidate
                        versioned = True
                        break
                    version += 1

            evidence_kind = citation.get("evidence_kind") or EVIDENCE_KIND_IMMUTABLE
            if evidence_kind not in {
                EVIDENCE_KIND_IMMUTABLE, EVIDENCE_KIND_CURRENT, EVIDENCE_KIND_SNAPSHOT
            }:
                raise ToolValidationError(
                    f"evidence_kind must be immutable_history/current_state/"
                    f"captured_snapshot, got {evidence_kind!r}"
                )

            evidence.append(_write(
                graph, "evidence_upsert_v1",
                {
                    "evidence_id": evidence_id, "claim": claim, "stance": stance,
                    "query_id": query_id,
                    "params_json": _json(citation.get("params") or {}),
                    "rows_json": _json(rows),
                    "derived_json": _json(citation.get("derived") or {}),
                    "result_hash": result_hash,
                    "row_count": primary_row_count,
                    "truncated": bool(citation.get("truncated")),
                    "interpretation": str(citation.get("interpretation") or ""),
                    "evidence_kind": evidence_kind,
                    "source_tool": str(citation.get("source_tool") or ""),
                    "captured_at": citation.get("captured_at") or _now(),
                    "case_id": case_id,
                    "columns_json": _json([str(c) for c in columns]),
                    "interpolation_json": _json(interpolation),
                    "template_sha256": qc.template_sha256(query_id),
                    "sources_json": sources_json,
                    "source_count": len(manifest),
                    "required_source_queries_json": _json(required_queries),
                },
            ))
            evidence.append(_write(graph, "case_link_evidence_v1",
                                   {"case_id": case_id, "evidence_id": evidence_id}))
            entity_id = citation.get("entity_id")
            if entity_id:
                entity_id = qc.validate_entity_id(entity_id, "entity_id")
                evidence.append(_write(
                    graph, "evidence_supports_v1",
                    {"evidence_id": evidence_id, "entity_id": entity_id},
                ))
            stored.append({
                "evidence_id": evidence_id, "claim": claim, "stance": stance,
                "query_id": query_id, "result_hash": result_hash,
                "truncated": bool(citation.get("truncated")),
                "evidence_kind": evidence_kind, "versioned": versioned,
                "interpolation": interpolation,
                "source_count": len(manifest),
                "required_source_queries": required_queries,
                "source_query_ids": [s["query_id"] for s in manifest],
            })

        counts = {
            stance: sum(1 for r in stored if r["stance"] == stance)
            for stance in (STANCE_SUPPORTING, STANCE_CONTRADICTING, STANCE_UNRESOLVED)
        }
        result = {"case_id": case_id, "cited_count": len(stored),
                  "stance_counts": counts, "citations": stored}
        interpretation = (
            f"Persisted {len(stored)} evidence citation(s) on case {case_id} "
            f"({counts[STANCE_SUPPORTING]} supporting, "
            f"{counts[STANCE_CONTRADICTING]} contradicting, "
            f"{counts[STANCE_UNRESOLVED]} unresolved), each bound to an approved "
            "query id with its parameters, rows, hash, truncation flag and capture "
            "time. Every citation persists the complete per-sub-query source "
            "manifest (query ids, parameters, hashes, truncation and the "
            "claim-material subset), so each contributing approved query can be "
            "verified individually. Supporting, contradicting and unresolved "
            "evidence remain distinguishable."
        )
        return make_envelope("cite_evidence", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("cite_evidence", exc, t0)


def flag_entity(
    graph,
    case_id: str,
    entity_id: str,
    reason: str,
    *,
    risk: str = "medium",
    settings: Settings | None = None,
) -> ToolEnvelope:
    """Mark an entity for review — review requested, not guilt established."""
    t0 = time.perf_counter()
    settings = settings or get_settings()
    try:
        _require_sandbox(settings)
        case_id = qc.validate_entity_id(case_id, "case_id")
        entity_id = qc.validate_entity_id(entity_id, "entity_id")
        risk = qc.validate_severity(risk, "risk")
        if not str(reason).strip():
            raise ToolValidationError("a non-empty reason is required")

        case = _read(graph, "case_lookup_v1", {"case_id": case_id}, row_cap=1)
        evidence = [case]
        if not case.rows:
            raise ToolValidationError(f"case {case_id} does not exist")

        # An entity may be either an Account or a Customer (a case can flag the
        # controlling customer, not just one account).
        entity = _read(graph, "entity_exists_v1", {"entity_id": entity_id}, row_cap=1)
        evidence.append(entity)
        if not entity.rows:
            raise ToolValidationError(f"entity {entity_id} does not exist")
        kind = entity.rows[0][1] if entity.rows else None

        evidence.append(_write(
            graph, "entity_flag_v1",
            {"entity_id": entity_id, "reason": reason, "risk": risk,
             "case_id": case_id, "now": _now()},
        ))
        result = {"case_id": case_id, "entity_id": entity_id, "kind": kind,
                  "reason": reason, "risk": risk, "flagged_for_review": True}
        interpretation = (
            f"Entity {entity_id} (a synthetic {kind or 'entity'}) was flagged for "
            f"review on case {case_id} (reason: {reason}; assessed risk: {risk}). "
            "Flagging requests human review; it does not establish wrongdoing and "
            "does not modify routing."
        )
        return make_envelope("flag_entity", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("flag_entity", exc, t0)


def close_case(
    graph,
    case_id: str,
    *,
    verdict: str,
    summary: str,
    status: str = "closed",
    settings: Settings | None = None,
) -> ToolEnvelope:
    """Store the permitted final disposition for a case.

    A case may be closed as ``suspicion_supported_for_review`` only when the
    evidence PERSISTED on the case still satisfies the deterministic escalation
    conditions (independent signal families, enough successful records, no
    failed queries, no material truncation, benign explanation ruled out). The
    verdict is re-derived from the stored evidence here; a caller cannot assert
    it into existence.
    """
    t0 = time.perf_counter()
    settings = settings or get_settings()
    try:
        _require_sandbox(settings)
        case_id = qc.validate_entity_id(case_id, "case_id")
        if status not in {"closed", "open"}:
            raise ToolValidationError(f"status must be 'closed' or 'open', got {status!r}")
        canonical = action_policy.canonical_verdict(verdict)
        if canonical not in set(action_policy.PHASE4_VERDICT_MIGRATION.values()) | {
            action_policy.VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
            action_policy.VERDICT_INCOMPLETE_INVESTIGATION,
        }:
            raise ToolValidationError(f"unsupported verdict {verdict!r}")

        state, reads = _reconstruct_case_state_full(graph, case_id)
        evidence = list(reads)
        if canonical == action_policy.VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW:
            escalated = action_policy.escalation_verdict(
                state, action_policy.VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION
            )
            if escalated != action_policy.VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW:
                raise ToolValidationError(
                    "cannot close case "
                    f"{case_id} as {canonical!r}: the evidence persisted on the "
                    "case does not satisfy the escalation conditions (at least "
                    f"{action_policy.MIN_EVIDENCE_RECORDS} successful evidence "
                    f"records, at least {action_policy.MIN_SIGNAL_FAMILIES} "
                    "independent signal families, no failed queries, no material "
                    "truncation, and the benign explanation ruled out)"
                )

        wrote = _write(
            graph, "case_close_v1",
            {"case_id": case_id, "status": status, "verdict": canonical,
             "summary": summary, "updated_at": _now()},
        )
        evidence.append(wrote)
        if not wrote.rows:
            raise ToolValidationError(f"case {case_id} does not exist")
        result = {"case_id": case_id, "status": status, "verdict": canonical,
                  "summary": summary}
        interpretation = (
            f"Case {case_id} was set to status {status!r} with canonical verdict "
            f"{canonical!r}. The disposition is a risk-review outcome, not a legal "
            "determination."
        )
        if canonical == action_policy.VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION:
            interpretation += (
                " This verdict remains provisional (suspicious_activity); it is "
                "not a confirmed finding."
            )
        return make_envelope("close_case", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("close_case", exc, t0)


# ---- simulation ------------------------------------------------------------


def _fingerprint(account_props: dict[str, Any], topology_rows: list[list[Any]]) -> str:
    """Hash the candidate account state plus the BOUNDED ACTIVE TOPOLOGY snapshot.

    The fingerprint is the simulation's evidence snapshot: if any active routing
    edge (or its aggregate properties) anywhere in the bounded topology
    snapshot changes after the simulation ran, the simulation is stale and a
    fresh one is required (PLAN.md §8). The bounded topology limit is part of
    the simulation parameters, and an incomplete (truncated) simulation can
    never authorize an action.
    """
    return _canonical_hash({
        "account_id": account_props.get("account_id"),
        "status": account_props.get("status"),
        "active_topology": [
            {"src": row[0], "dst": row[1], "count": row[2],
             "total_amount": row[3], "last_ts": row[4]}
            for row in topology_rows
        ],
    })


def simulate_freeze(
    graph,
    case_id: str,
    account_id: str,
    *,
    window_days: int = 30,
    hops: int = 3,
    topology_limit: int = DEFAULT_TOPOLOGY_LIMIT,
    settings: Settings | None = None,
    persist: bool = True,
) -> ToolEnvelope:
    """Read-only counterfactual: what would a freeze of ``account_id`` affect?

    Distinct metrics, never conflated:
    - ``downstream_accounts_before/after``: unique reachable account sets;
    - ``newly_unreachable_accounts``: before minus after on the same bounded model;
    - ``direct_outgoing_volume``: sum of UNIQUE transaction events in the window;
    - ``active_edges_disabled``: incident active routing edges a freeze disables.

    It does NOT report funds "frozen" or "stranded": the graph holds no balance
    or accounting model that would support that claim.
    """
    t0 = time.perf_counter()
    settings = settings or get_settings()
    try:
        _require_sandbox(settings)
        case_id = qc.validate_entity_id(case_id, "case_id")
        account_id = qc.validate_entity_id(account_id, "account_id")
        window_days = qc.validate_window_days(window_days)
        hops = qc.validate_hop_count(hops)

        case = _read(graph, "case_lookup_v1", {"case_id": case_id}, row_cap=1)
        evidence: list[QueryEvidence] = [case]
        if not case.rows:
            raise ToolValidationError(f"case {case_id} does not exist")

        state = _read(graph, "account_state_v1", {"account_id": account_id}, row_cap=1)
        evidence.append(state)
        if not state.rows:
            raise ToolValidationError(f"account {account_id} does not exist")
        account_props = {
            "account_id": state.rows[0][0], "status": state.rows[0][1],
            "balance": state.rows[0][2], "currency": state.rows[0][3],
        }

        window, window_ev = _window(graph, window_days)
        evidence.append(window_ev)
        direct = _read(
            graph, "direct_outgoing_in_window_v1",
            {"account_id": account_id, "cutoff": window["cutoff"]},
        )
        evidence.append(direct)
        direct_count = int(direct.rows[0][0]) if direct.rows else 0
        direct_volume = float(direct.rows[0][1] or 0.0) if direct.rows else 0.0

        incident = _read(
            graph, "account_incident_active_edges_v1",
            {"account_id": account_id, "limit": topology_limit}, row_cap=topology_limit,
        )
        evidence.append(incident)
        incident_edges = incident.rows
        disabled = [(e[0], e[1]) for e in incident_edges]
        disabled_set = set(disabled)

        topology = _read(
            graph, "active_topology_v1", {"limit": topology_limit}, row_cap=topology_limit,
        )
        evidence.append(topology)
        edges = [(r[0], r[1]) for r in topology.rows]
        remaining = [e for e in edges if e not in disabled_set]

        before = reachable_accounts(edges, account_id, hops)
        after = reachable_accounts(remaining, account_id, hops)
        newly_unreachable = sorted(before - after)

        truncated = bool(topology.truncated or incident.truncated or direct.truncated)
        # The fingerprint covers the account state plus the whole bounded active
        # topology snapshot (edge ids AND aggregate properties), so any material
        # routing change invalidates the simulation (PLAN.md §8).
        fingerprint = _fingerprint(account_props, topology.rows)
        metrics = {
            "account_id": account_id,
            "window": window,
            "hops": hops,
            "direct_outgoing_volume": round(direct_volume, 2),
            "direct_outgoing_transaction_count": direct_count,
            "downstream_accounts_before": len(before),
            "downstream_accounts_after": len(after),
            "newly_unreachable_accounts": newly_unreachable,
            "newly_unreachable_count": len(newly_unreachable),
            "active_edges_disabled": len(disabled),
            "disabled_edges": [{"src": s, "dst": d} for s, d in disabled],
            "active_edges_total": len(edges),
            "account_status": account_props.get("status"),
            "simulation_complete": not truncated,
            "limits": {"topology_limit": topology_limit, "hops": hops,
                       "window_days": window_days},
            "excluded": [
                "accounts unreachable within the hop cap",
                "inactive routing edges",
                "frozen accounts",
                "funds actually unavailable (no accounting model exists)",
            ],
            "truncated_queries": [
                ev.query_id for ev in (topology, incident, direct) if ev.truncated
            ],
            "metric_semantics": {
                "direct_outgoing_volume":
                    "sum of unique transaction events in the window — historical "
                    "volume exposed to disruption, NOT money that would be frozen",
                "downstream_accounts_before/after":
                    "unique accounts reachable under the bounded active routing model",
                "active_edges_disabled":
                    "incident active TRANSFERRED_TO edges a freeze would deactivate",
            },
        }

        simulation_id = None
        if persist:
            simulation_id = f"SIM-{case_id}-{account_id}-{hops}-{window_days}"
            evidence.append(_write(
                graph, "simulation_upsert_v1",
                {
                    "simulation_id": simulation_id, "case_id": case_id,
                    "account_id": account_id, "graph_fingerprint": fingerprint,
                    "metrics_json": _json(metrics), "window_days": window_days,
                    "hops": hops, "simulation_complete": metrics["simulation_complete"],
                    "status": "complete" if metrics["simulation_complete"] else "incomplete",
                    "created_at": _now(),
                },
            ))
            evidence.append(_write(
                graph, "simulation_link_v1",
                {"case_id": case_id, "simulation_id": simulation_id},
            ))

        result = {
            "case_id": case_id, "simulation_id": simulation_id,
            "graph_fingerprint": fingerprint, "metrics": metrics,
        }
        interpretation = (
            f"Simulating a freeze of {account_id} over {hops} hops and a "
            f"{window_days}-day window: {metrics['active_edges_disabled']} active "
            f"routing edge(s) would be disabled, "
            f"{metrics['downstream_accounts_before']} downstream account(s) are "
            f"currently reachable and {metrics['newly_unreachable_count']} would "
            f"become unreachable. Historical outgoing volume of "
            f"{metrics['direct_outgoing_volume']:.2f} across {direct_count} unique "
            "transaction event(s) is exposed to disruption — that is transaction "
            "volume, not money that would be frozen or stranded, which this graph "
            "cannot compute."
            + ("" if metrics["simulation_complete"]
               else " WARNING: the snapshot was truncated, so this simulation is "
                    "incomplete and cannot justify an action.")
        )
        return make_envelope("simulate_freeze", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("simulate_freeze", exc, t0)


# ---- the guarded freeze ----------------------------------------------------


def _load_simulation(graph, case_id: str, account_id: str):
    ev = _read(graph, "simulation_for_case_v1",
               {"case_id": case_id, "account_id": account_id}, row_cap=1)
    return (ev.rows[0][0] if ev.rows else None), ev


# Evidence ids are stored on case evidence as f"{case_id}:{local_id}"; a
# permissive but bounded shape check keeps control characters, whitespace and
# unbounded blobs out of persisted references.
_EVIDENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


def _case_scoped_evidence_ids(case_id: str, evidence_ids: Iterable[str]) -> list[str]:
    """Normalize investigation-side evidence ids to their case-scoped stored form."""
    prefix = f"{case_id}:"
    out: list[str] = []
    for raw in evidence_ids:
        value = str(raw)
        scoped = value if value.startswith(prefix) else f"{prefix}{value}"
        if scoped not in out:
            out.append(scoped)
    return out


def _supplied_evidence_problems(
    case_id: str, supplied: Iterable[str], allowed: set[str],
) -> list[str]:
    """Reject caller-supplied evidence references that cannot back this action.

    A supplied id must be well-formed, linked to this case, and part of the
    policy's qualifying evidence for this target — never fabricated or reused
    from an unrelated account or case.
    """
    prefix = f"{case_id}:"
    problems: list[str] = []
    for evidence_id in supplied:
        if not _EVIDENCE_ID_RE.match(evidence_id):
            problems.append(f"evidence id {evidence_id!r} is malformed")
        elif not evidence_id.startswith(prefix):
            problems.append(
                f"evidence id {evidence_id!r} is not linked to case {case_id}"
            )
        elif evidence_id not in allowed:
            problems.append(
                f"evidence id {evidence_id!r} does not back the qualifying "
                "evidence for this target"
            )
    return problems


def freeze_account(
    graph,
    case_id: str,
    account_id: str,
    reason: str,
    *,
    eligibility_inputs: dict[str, Any],
    requested_by: str = "policy",
    evidence_ids: Iterable[str] = (),
    settings: Settings | None = None,
) -> ToolEnvelope:
    """Freeze an account in the sandbox — only if every precondition holds.

    Public entry point: serializes the read-check-mutate section in-process via
    ``_FREEZE_LOCK`` and delegates to ``_freeze_account_unlocked``. Duplicate
    and concurrent attempts therefore resolve to ``already_applied`` or
    ``in_progress`` instead of racing; the database-level guard in
    ``freeze_account_v1`` is the second line of defence for multi-process use.
    """
    with _FREEZE_LOCK:
        return _freeze_account_unlocked(
            graph, case_id, account_id, reason,
            eligibility_inputs=eligibility_inputs,
            requested_by=requested_by, evidence_ids=evidence_ids,
            settings=settings,
        )


def _freeze_account_unlocked(
    graph,
    case_id: str,
    account_id: str,
    reason: str,
    *,
    eligibility_inputs: dict[str, Any],
    requested_by: str = "policy",
    evidence_ids: Iterable[str] = (),
    settings: Settings | None = None,
) -> ToolEnvelope:
    """Freeze an account in the sandbox — only if every precondition holds.

    ``eligibility_inputs`` carries the investigation-side facts the policy needs
    (the CaseState plus the canonical verdict). This function re-runs the policy
    itself; it never trusts a caller-supplied "allowed" flag, and the LLM has no
    way to bypass it. On any failure nothing is written.
    """
    t0 = time.perf_counter()
    settings = settings or get_settings()
    evidence: list[QueryEvidence] = []
    try:
        _require_sandbox(settings)
        case_id = qc.validate_entity_id(case_id, "case_id")
        account_id = qc.validate_entity_id(account_id, "account_id")
        if not str(reason).strip():
            raise ToolValidationError("a non-empty reason is required")

        case_state = eligibility_inputs.get("case_state")
        if case_state is None:
            raise ToolValidationError(
                "eligibility_inputs.case_state is required: eligibility is "
                "re-derived from the investigation evidence"
            )

        case = _read(graph, "case_lookup_v1", {"case_id": case_id}, row_cap=1)
        evidence.append(case)
        case_exists = bool(case.rows)
        case_props = (case.rows[0][0] or {}) if case.rows else {}

        # The verdict used for eligibility is the one RECORDED on the case (the
        # investigation's canonical outcome). A caller-supplied value is only a
        # fallback when no case has been persisted yet, so the in-process and
        # MCP paths decide identically.
        verdict = action_policy.canonical_verdict(
            case_props.get("verdict")
            or eligibility_inputs.get("verdict")
            or case_state.provisional_verdict
        )

        state = _read(graph, "account_state_v1", {"account_id": account_id}, row_cap=1)
        evidence.append(state)
        account_exists = bool(state.rows)
        account_status = state.rows[0][1] if state.rows else None

        simulation, sim_ev = _load_simulation(graph, case_id, account_id)
        evidence.append(sim_ev)

        prior_ev = _read(
            graph, "action_for_case_v1",
            {"case_id": case_id, "account_id": account_id, "action": ACTION_FREEZE},
            row_cap=1,
        )
        evidence.append(prior_ev)
        prior = prior_ev.rows[0][0] if prior_ev.rows else None
        if prior and prior.get("status") == "applied":
            result = {
                "case_id": case_id, "account_id": account_id,
                "action_status": "already_applied", "action_id": prior.get("id"),
                "frozen": True, "simulation_id": prior.get("simulation_id"),
                "before_state": json.loads(prior.get("before_state_json") or "{}"),
                "policy": json.loads(prior.get("policy_json") or "{}"),
                "mutation_performed": False,
            }
            interpretation = (
                f"Account {account_id} was already frozen by an earlier action "
                f"({prior.get('id')}) on case {case_id}; the freeze is idempotent, so "
                "no second mutation was performed."
            )
            return make_envelope("freeze_account", evidence, result, interpretation, t0)

        incident = _read(
            graph, "account_incident_active_edges_v1",
            {"account_id": account_id, "limit": DEFAULT_TOPOLOGY_LIMIT},
            row_cap=DEFAULT_TOPOLOGY_LIMIT,
        )
        evidence.append(incident)
        # The staleness snapshot is the WHOLE bounded active topology, not just
        # the candidate's incident edges: downstream reachability metrics depend
        # on it, so any material routing change requires re-simulation.
        topology = _read(
            graph, "active_topology_v1", {"limit": DEFAULT_TOPOLOGY_LIMIT},
            row_cap=DEFAULT_TOPOLOGY_LIMIT,
        )
        evidence.append(topology)
        current_fingerprint = (
            _fingerprint({"account_id": account_id, "status": account_status},
                         topology.rows)
            if account_exists else None
        )
        simulation_stale = bool(
            simulation and current_fingerprint
            and simulation.get("graph_fingerprint") != current_fingerprint
        )
        sim_metrics = (
            json.loads(simulation.get("metrics_json") or "{}") if simulation else None
        )

        decision = action_policy.evaluate_freeze_eligibility(
            case_state,
            verdict=verdict,
            target_account_id=account_id,
            simulation=sim_metrics,
            simulation_stale=simulation_stale,
            account_exists=account_exists,
            account_already_frozen=account_status == "frozen",
            sandbox_enabled=settings.habeas_enable_actions,
            graph_is_demo=settings.is_demo_graph,
            actions_enabled=settings.habeas_enable_actions,
        )
        policy_record = {
            "allowed": decision.allowed,
            "verdict": decision.verdict,
            "blocking_reasons": decision.blocking_reasons,
            "caveats": decision.caveats,
            "checks": decision.checks,
            "signal_families": decision.signal_families,
            "target_relevant_families": decision.target_relevant_families,
            "distinct_provenance_families": decision.distinct_provenance_families,
            "qualifying_evidence_ids": decision.qualifying_evidence_ids,
            "provenance": decision.provenance,
            "policy_version": action_policy.POLICY_VERSION,
        }

        # Caller-supplied evidence references are validated against the
        # policy's qualifying set for THIS case and target before anything is
        # read for mutation: a fabricated, unrelated or malformed reference
        # fails the call without touching the graph.
        supplied_evidence_ids = [
            str(item).strip() for item in evidence_ids if str(item).strip()
        ]
        allowed_evidence_ids = set(_case_scoped_evidence_ids(
            case_id, decision.qualifying_evidence_ids,
        ))
        if supplied_evidence_ids:
            problems = _supplied_evidence_problems(
                case_id, supplied_evidence_ids, allowed_evidence_ids,
            )
            if problems:
                raise ToolValidationError(
                    "caller-supplied evidence_ids cannot back this action: "
                    + "; ".join(problems)
                )

        if not (case_exists and decision.allowed):
            reasons = list(decision.blocking_reasons)
            if not case_exists:
                reasons.insert(0, f"case {case_id} does not exist")
            result = {
                "case_id": case_id, "account_id": account_id,
                "action_status": "declined", "frozen": False,
                "policy": policy_record, "mutation_performed": False,
            }
            interpretation = (
                f"Freeze DECLINED for {account_id} on case {case_id}; no graph "
                "mutation was performed. " + "; ".join(reasons)
            )
            return make_envelope("freeze_account", evidence, result, interpretation, t0)

        # Resolve the exact evidence references this action will persist, and
        # verify they exist among the case's PERSISTED, cited evidence — not
        # merely in the in-memory decision. Missing or fabricated references
        # abort before any mutation. The stored list is sorted: the
        # reconstructed evidence order depends on graph storage order, so only
        # a sorted set is a stable, auditable record.
        action_evidence_ids = sorted(dict.fromkeys(
            supplied_evidence_ids or _case_scoped_evidence_ids(
                case_id, decision.qualifying_evidence_ids,
            )
        ))
        if not action_evidence_ids:
            raise ToolValidationError(
                "no qualifying evidence ids could be resolved for this action"
            )
        stored_ev = _read(
            graph, "case_evidence_v1", {"case_id": case_id, "limit": 500},
            row_cap=500,
        )
        evidence.append(stored_ev)
        id_col = stored_ev.columns.index("evidence_id")
        stored_ids = {row[id_col] for row in stored_ev.rows}
        missing_refs = [eid for eid in action_evidence_ids if eid not in stored_ids]
        if missing_refs:
            raise ToolValidationError(
                "the action's qualifying evidence references are not present in "
                f"the case's persisted evidence: {missing_refs}"
            )

        before_state = {
            "status": account_status,
            "active_incident_edges": len(incident.rows),
            "active_edges_total": len(topology.rows),
        }
        now = _now()
        # The only account mutation. The WHERE guard makes it atomic at the
        # database level: if any concurrent attempt already froze the account
        # (or a cross-process caller raced this one), this matches zero rows
        # and nothing is written below.
        freeze_ev = _write(
            graph, "freeze_account_v1",
            {"account_id": account_id, "now": now, "reason": reason, "case_id": case_id},
            row_cap=1,
        )
        evidence.append(freeze_ev)
        if not freeze_ev.rows:
            state2 = _read(graph, "account_state_v1", {"account_id": account_id},
                           row_cap=1)
            evidence.append(state2)
            prior2_ev = _read(
                graph, "action_for_case_v1",
                {"case_id": case_id, "account_id": account_id, "action": ACTION_FREEZE},
                row_cap=1,
            )
            evidence.append(prior2_ev)
            prior2 = prior2_ev.rows[0][0] if prior2_ev.rows else None
            status2 = state2.rows[0][1] if state2.rows else None
            if prior2 and prior2.get("status") == "applied":
                result = {
                    "case_id": case_id, "account_id": account_id,
                    "action_status": "already_applied",
                    "action_id": prior2.get("id"),
                    "frozen": status2 == "frozen",
                    "simulation_id": prior2.get("simulation_id"),
                    "policy": policy_record, "mutation_performed": False,
                }
                interpretation = (
                    f"Account {account_id} was frozen by a concurrent/earlier "
                    f"action ({prior2.get('id')}) on case {case_id}; the guarded "
                    "database write matched no rows, so this attempt performed no "
                    "second mutation."
                )
            elif status2 == "frozen":
                result = {
                    "case_id": case_id, "account_id": account_id,
                    "action_status": "in_progress", "frozen": True,
                    "policy": policy_record, "mutation_performed": False,
                }
                interpretation = (
                    f"A concurrent freeze of {account_id} won the guarded "
                    "database write; this attempt performed no mutation. The "
                    "winning action's audit record is written by its own call."
                )
            else:
                result = {
                    "case_id": case_id, "account_id": account_id,
                    "action_status": "declined", "frozen": False,
                    "policy": policy_record, "mutation_performed": False,
                }
                interpretation = (
                    f"Freeze DECLINED for {account_id} on case {case_id}: the "
                    "guarded database write matched no rows and no concurrent "
                    "freeze is visible; nothing was mutated."
                )
            return make_envelope("freeze_account", evidence, result, interpretation, t0)

        deactivated = _write(
            graph, "deactivate_incident_edges_v1",
            {"account_id": account_id, "now": now, "case_id": case_id},
        )
        evidence.append(deactivated)
        deactivated_count = int(deactivated.rows[0][0] or 0) if deactivated.rows else 0

        after_state = {"status": "frozen", "deactivated_edges": deactivated_count,
                       "frozen_at": now}
        action_id = f"ACT-{case_id}-{account_id}"
        action_payload = {
            "action_id": action_id, "case_id": case_id, "account_id": account_id,
            "action": ACTION_FREEZE, "status": "applied", "reason": reason,
            "requested_by": requested_by,
            "simulation_id": simulation.get("id"),
            "evidence_ids_json": _json(action_evidence_ids),
            "policy_json": _json(policy_record),
            "before_state_json": _json(before_state),
            "after_state_json": _json(after_state),
            "created_at": now,
        }
        # First write: the record exists as applied (the immediate readback
        # below verifies the record itself, so it must exist first).
        evidence.append(_write(
            graph, "action_upsert_v1",
            {**action_payload,
             "verification_json": _json({"status": "readback_pending"})},
        ))
        evidence.append(_write(graph, "action_link_v1",
                               {"case_id": case_id, "action_id": action_id}))

        # Immediate post-action readback: the same deterministic checks the
        # Phase 6 verifier runs, appended to the action record so the audit
        # trail shows what was true at action time. Later verification never
        # rewrites this snapshot.
        verification = verify_freeze(graph, case_id, account_id, settings=settings)
        verification_result = dict(verification.result or {})
        evidence.append(_write(
            graph, "action_upsert_v1",
            {**action_payload, "verification_json": _json(verification_result)},
        ))

        result = {
            "case_id": case_id, "account_id": account_id, "action_id": action_id,
            "action_status": "applied", "frozen": True,
            "simulation_id": simulation.get("id"),
            "deactivated_edges": deactivated_count,
            "evidence_ids": action_evidence_ids,
            "post_action_verification": verification_result,
            "before_state": before_state, "after_state": after_state,
            "policy": policy_record, "mutation_performed": True,
        }
        interpretation = (
            f"SIMULATED FREEZE APPLIED in the sandbox graph only: {account_id} is now "
            f"frozen (case {case_id}, action {action_id}) and {deactivated_count} "
            f"incident active routing edge(s) were deactivated. Justified by "
            f"simulation {simulation.get('id')} and "
            f"{len(decision.qualifying_evidence_ids)} qualifying evidence record(s) "
            f"across {len(decision.distinct_provenance_families)} distinct-provenance "
            f"signal family/families; the record persists {len(action_evidence_ids)} "
            f"exact evidence id(s) under policy version "
            f"{action_policy.POLICY_VERSION}. Historical transactions and transfer "
            "summaries were preserved; no data was deleted."
        )
        return make_envelope("freeze_account", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("freeze_account", exc, t0)


# ---- post-action verification ----------------------------------------------


def verify_freeze(
    graph,
    case_id: str,
    account_id: str,
    *,
    settings: Settings | None = None,
) -> ToolEnvelope:
    """Re-query the graph after the action and report what actually changed.

    Kept separate from the pre-action simulation and evidence: nothing here
    rewrites what was observed before the freeze.
    """
    t0 = time.perf_counter()
    settings = settings or get_settings()
    try:
        case_id = qc.validate_entity_id(case_id, "case_id")
        account_id = qc.validate_entity_id(account_id, "account_id")

        state = _read(graph, "account_state_v1", {"account_id": account_id}, row_cap=1)
        incident = _read(
            graph, "account_incident_active_edges_v1",
            {"account_id": account_id, "limit": DEFAULT_TOPOLOGY_LIMIT},
            row_cap=DEFAULT_TOPOLOGY_LIMIT,
        )
        topology = _read(
            graph, "active_topology_v1", {"limit": DEFAULT_TOPOLOGY_LIMIT},
            row_cap=DEFAULT_TOPOLOGY_LIMIT,
        )
        actions = _read(graph, "case_actions_v1",
                        {"case_id": case_id, "limit": 50}, row_cap=50)
        case = _read(graph, "case_lookup_v1", {"case_id": case_id}, row_cap=1)
        simulation, sim_ev = _load_simulation(graph, case_id, account_id)
        evidence = [state, incident, topology, actions, case, sim_ev]

        status = state.rows[0][1] if state.rows else None
        action_rows = [
            {"action_id": r[0], "account_id": r[1], "action": r[2],
             "status": r[3], "created_at": r[4]}
            for r in actions.rows
        ]
        matching = [
            a for a in action_rows
            if a["account_id"] == account_id and a["action"] == ACTION_FREEZE
        ]

        # The pre-action simulation is the baseline every post-action count is
        # compared against: the freeze promised to deactivate N incident active
        # edges, so the graph-wide active edge count must now be exactly
        # (baseline total - N). A mismatch means the routing changed beyond
        # what the action recorded.
        sim_metrics = (
            json.loads(simulation.get("metrics_json") or "{}") if simulation else None
        )
        expected_active_edges = None
        if sim_metrics and sim_metrics.get("simulation_complete"):
            expected_active_edges = int(sim_metrics.get("active_edges_total") or 0) - int(
                sim_metrics.get("active_edges_disabled") or 0
            )
        routing_matches = (
            expected_active_edges is not None
            and len(topology.rows) == expected_active_edges
        )

        checks = {
            "account_is_frozen": status == "frozen",
            "no_active_incident_edges": len(incident.rows) == 0,
            "action_record_present": bool(matching),
            "case_exists": bool(case.rows),
            "routing_edge_count_matches_simulation": routing_matches,
        }
        result = {
            "case_id": case_id, "account_id": account_id,
            "account_status": status,
            "active_incident_edges": len(incident.rows),
            "active_edges_total": len(topology.rows),
            "simulation_id": simulation.get("id") if simulation else None,
            "simulation_expected_active_edges": expected_active_edges,
            "actions": action_rows, "checks": checks,
            "verified": all(checks.values()),
        }
        interpretation = (
            f"Post-action graph state for {account_id}: status is {status!r}, "
            f"{len(incident.rows)} active routing edge(s) remain incident to it, and "
            f"{len(topology.rows)} active edge(s) remain graph-wide"
            + (
                f" (the recorded simulation expected {expected_active_edges})"
                if expected_active_edges is not None
                else " (no complete recorded simulation to compare against)"
            )
            + ". An action record "
            f"for this case and account "
            f"{'exists' if matching else 'is MISSING'}. "
            + ("All post-action checks passed." if result["verified"]
               else "One or more post-action checks FAILED: "
                    + ", ".join(k for k, v in checks.items() if not v))
        )
        return make_envelope("verify_freeze", evidence, result, interpretation, t0)
    except Exception as exc:
        return error_envelope("verify_freeze", exc, t0)