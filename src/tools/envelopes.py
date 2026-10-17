"""Canonical JSON-safe tool result envelopes (PLAN.md §7).

Every Phase 2 tool returns a :class:`ToolEnvelope`:

- ``query_evidence``: one record per executed approved query — query id,
  template text, parameters, columns, normalized rows, a deterministic hash
  over ALL rows, truncation flag, and duration;
- ``result``: JSON-safe derived payload;
- ``interpretation``: a deterministic string built in Python from result
  fields — never LLM-authored, never a legal conclusion;
- errors are structured (``error.kind``/``message``) and secret-free.

Rows are normalized through ``src.db.normalize_rows``/``normalize_value`` so
Node/Edge/Path cells become plain dictionaries before hashing/serialization.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

from pydantic import BaseModel, Field

from src.db import QueryError, normalize_rows, run_query

ROW_LIMIT_DEFAULT = 100
ROW_LIMIT_MAX = 500


class ToolValidationError(ValueError):
    """A tool argument failed validation (never retried, never surfaced raw)."""


class ToolErrorBody(BaseModel):
    kind: str
    message: str


class QueryEvidence(BaseModel):
    query_id: str
    cypher: str
    params: dict[str, Any]
    interpolated: list[str] = Field(default_factory=list)
    # Phase 6: the integer values substituted for ``<<token>>`` bounds, keyed by
    # the token text. Without these the exact bounded query cannot be rebuilt
    # for verification, so they are recorded at execution time.
    bound: dict[str, int] = Field(default_factory=dict)
    columns: list[str]
    rows: list[Any]
    rows_hash: str
    truncated: bool
    row_count: int
    duration_ms: float


class ToolEnvelope(BaseModel):
    ok: bool
    tool: str
    query_evidence: list[QueryEvidence] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    interpretation: str
    duration_ms: float = 0.0
    error: ToolErrorBody | None = None


def rows_hash(columns: list[str], rows: list[Any]) -> str:
    """Stable sha256 over ALL normalized rows (PLAN.md §7).

    Rows must already be JSON-safe; dumps with sorted keys and fixed
    separators so the hash depends only on content. Row ORDER is canonicalized
    (sorted by each row's canonical JSON) because several approved queries do
    not guarantee a total order among ties; without this, an identical replay
    could hash differently purely because the server returned rows in another
    order (Phase 6 verifier requirement).
    """
    canonical_rows = sorted(
        rows,
        key=lambda row: json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    )
    payload = json.dumps(
        {"columns": columns, "rows": canonical_rows},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_evidence(
    graph,
    query_id: str,
    cypher: str,
    params: Mapping[str, Any],
    *,
    interpolated: list[str] | None = None,
    bound: Mapping[str, int] | None = None,
    row_cap: int | None = None,
    timeout_ms: int | None = None,
    read_only: bool = True,
) -> QueryEvidence:
    """Execute one approved template and capture its evidence.

    Reads (the default) use the read-only routing. ``read_only=False`` is used
    ONLY by the Phase 5 action layer, which executes purpose-built write
    templates from the approved catalog — never model-authored Cypher.
    ``bound`` records the validated integer values used for ``<<token>>``
    interpolation so Phase 6 can rebuild the exact bounded query.
    """
    params = dict(params)
    t0 = time.perf_counter()
    result = run_query(graph, cypher, params, timeout_ms=timeout_ms, read_only=read_only)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    columns, rows = normalize_rows(result)
    # Belt and braces: guarantee JSON safety even if normalization misses a type.
    safe_rows = json.loads(json.dumps(rows, default=str))
    return QueryEvidence(
        query_id=query_id,
        cypher=cypher,
        params=params,
        interpolated=list(interpolated or []),
        bound={str(k): int(v) for k, v in dict(bound or {}).items()},
        columns=columns,
        rows=safe_rows,
        rows_hash=rows_hash(columns, safe_rows),
        truncated=bool(row_cap is not None and len(safe_rows) >= row_cap),
        row_count=len(safe_rows),
        duration_ms=round(duration_ms, 3),
    )


def make_envelope(
    tool: str,
    evidence: list[QueryEvidence],
    result: dict[str, Any],
    interpretation: str,
    t0: float,
) -> ToolEnvelope:
    assert interpretation and interpretation.strip(), "interpretation must be non-empty"
    return ToolEnvelope(
        ok=True,
        tool=tool,
        query_evidence=evidence,
        result=result,
        interpretation=interpretation,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
    )


def error_envelope(tool: str, exc: Exception, t0: float) -> ToolEnvelope:
    """Structured, secret-free failure envelope for any tool error."""
    if isinstance(exc, ToolValidationError):
        kind, message = "validation", str(exc)
    elif isinstance(exc, QueryError):
        kind = f"database:{exc.kind}"
        message = f"query failed ({exc.kind}); see evidence log for the template"
    else:
        kind, message = "internal", f"{type(exc).__name__}: {exc}"
    return ToolEnvelope(
        ok=False,
        tool=tool,
        result={},
        interpretation=f"The {tool} tool could not produce a result: {message}",
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        error=ToolErrorBody(kind=kind, message=message[:500]),
    )
