"""FalkorDB connection layer and the shared run_query helper.

This module is trusted infrastructure code (PLAN.md §0/§3): seed scripts, test
fixtures, and the domain tool layer use it. The investigation agent itself
never imports this module — agent graph access goes through the MCP tools
(Phase 2+).

Query rules enforced here (PLAN.md §0/§4):
- values are passed as query parameters; callers never interpolate input;
- timeouts are MILLISECONDS (falkordb-py contract, verified by Phase 0 probes);
- read-only queries are retried at most once on a classified transient
  connection failure; writes are never retried automatically;
- failures raise QueryError with a structured, secret-free classification.

Note: `ro_query` refuses a graph that does not exist yet ("Invalid graph
operation on empty key"); any write query auto-creates the graph. Trusted
setup code bootstraps with a harmless write before read-only use.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any, Mapping

from falkordb import FalkorDB, Graph

from src.config import Settings, get_settings

_TRANSIENT_KINDS = frozenset({"connection"})

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, what: str) -> str:
    """Guard for the only allowed interpolation: code-owned Cypher identifiers."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Cypher {what} must match [A-Za-z_][A-Za-z0-9_]*, got {name!r}")
    return name


class QueryError(Exception):
    """Structured query failure.

    kind is one of: connection | timeout | syntax | constraint | unknown.
    retryable means the failure classifies as a transient connection issue
    (whether a retry actually happens depends on read/write mode).
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        retryable: bool,
        cypher: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.cypher = cypher
        self.params = dict(params) if params is not None else None


def classify_failure(
    exc: Exception,
    *,
    cypher: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> QueryError:
    """Map a driver exception to a structured QueryError (never leaks secrets)."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if (
        "connectionerror" in name
        or "connection refused" in text
        or "connection reset" in text
        or "broken pipe" in text
    ):
        kind, retryable = "connection", True
    elif "timeout" in name or "timed out" in text:
        kind, retryable = "timeout", False
    elif "already exists" in text or "already indexed" in text or "constraint" in text:
        kind, retryable = "constraint", False
    elif (
        "responseerror" in name
        or "syntax" in text
        or "invalid" in text
        or "undefined" in text
        or "unknown" in text
    ):
        kind, retryable = "syntax", False
    else:
        kind, retryable = "unknown", False
    return QueryError(
        redact_secrets(str(exc)), kind=kind, retryable=retryable,
        cypher=cypher, params=params,
    )


def redact_secrets(text: str, settings: Settings | None = None) -> str:
    """Scrub credential material out of free text (messages, logs, errors)."""
    s = settings or get_settings()
    out = text
    password = s.falkordb_password.get_secret_value()
    if password:
        out = out.replace(password, "***")
    return out


def _scrub_exception(exc: Exception, settings: Settings | None = None) -> Exception:
    """Redact credentials from an exception's own message, in place."""
    scrubbed = tuple(
        redact_secrets(arg, settings) if isinstance(arg, str) else arg
        for arg in exc.args
    )
    if scrubbed != exc.args:
        exc.args = scrubbed
    return exc


@lru_cache(maxsize=4)
def _client_cached(
    host: str,
    port: int,
    username: str,
    password: str,
    tls: bool,
) -> FalkorDB:
    """One client per DISTINCT connection configuration (all settings keyed).

    The password participates in the cache key so a settings change can never
    silently reuse a client authenticated against different credentials; the
    value lives only in this process's memory, never in a log.
    """
    return FalkorDB(
        host=host,
        port=port,
        username=username or None,
        password=password or None,
        ssl=tls,
    )


def clear_client_cache() -> None:
    """Drop cached clients (tests and config changes call this)."""
    _client_cached.cache_clear()


def get_client(settings: Settings | None = None) -> FalkorDB:
    s = settings or get_settings()
    try:
        return _client_cached(
            s.falkordb_host,
            s.falkordb_port,
            s.falkordb_username,
            s.falkordb_password.get_secret_value(),
            s.falkordb_tls,
        )
    except Exception as exc:
        # connection failures must name the cause, never the credentials
        raise _scrub_exception(exc, s) from exc


def get_graph(settings: Settings | None = None, name: str | None = None) -> Graph:
    s = settings or get_settings()
    return get_client(s).select_graph(name or s.graph_name)


def run_query(
    graph: Graph,
    cypher: str,
    params: Mapping[str, Any] | None = None,
    *,
    timeout_ms: int | None = None,
    read_only: bool = False,
    retry_delay_s: float = 0.25,
):
    """Run one parameterized query and return the driver QueryResult.

    timeout_ms is in milliseconds; None uses the configured default.
    Read-only queries are retried at most once on a classified transient
    connection failure. Writes are never retried automatically.
    Uses ro_query for reads so the server can route them as read-only.
    """
    if timeout_ms is None:
        timeout_ms = get_settings().falkordb_query_timeout_ms
    method = graph.ro_query if read_only else graph.query
    attempts = 2 if read_only else 1
    for attempt in range(1, attempts + 1):
        try:
            return method(cypher, params=params, timeout=timeout_ms)
        except Exception as exc:  # classified below; never swallowed silently
            qe = classify_failure(exc, cypher=cypher, params=params)
            if read_only and qe.kind in _TRANSIENT_KINDS and attempt < attempts:
                if retry_delay_s > 0:
                    time.sleep(retry_delay_s)
                continue
            raise qe from exc
    raise AssertionError("unreachable: retry loop must return or raise")


def normalize_value(value: Any) -> Any:
    """Convert FalkorDB values into JSON-safe primitives (PLAN.md §4).

    Node/Edge/Path objects are duck-typed so this keeps working across
    driver versions; anything unrecognized degrades to its string form.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if hasattr(value, "labels") and hasattr(value, "properties"):  # Node-like
        return {
            "_type": "node",
            "id": value.id,
            "labels": sorted(value.labels or []),
            "properties": normalize_value(dict(value.properties or {})),
        }
    if hasattr(value, "relation") and hasattr(value, "properties"):  # Edge-like
        return {
            "_type": "edge",
            "id": value.id,
            "relation": value.relation,
            "src": value.src_node,
            "dst": value.dest_node,
            "properties": normalize_value(dict(value.properties or {})),
        }
    if hasattr(value, "nodes") and hasattr(value, "edges"):  # Path-like
        nodes_attr, edges_attr = value.nodes, value.edges
        nodes = nodes_attr() if callable(nodes_attr) else nodes_attr
        edges = edges_attr() if callable(edges_attr) else edges_attr
        return {
            "_type": "path",
            "nodes": [normalize_value(n) for n in nodes],
            "edges": [normalize_value(e) for e in edges],
        }
    return str(value)


def normalize_rows(result) -> tuple[list[str], list[list[Any]]]:
    """Return (column names, JSON-safe rows) from a driver QueryResult."""
    columns = [
        col[1] if isinstance(col, (list, tuple)) and len(col) > 1 else str(col)
        for col in result.header
    ]
    rows = [[normalize_value(cell) for cell in row] for row in result.result_set]
    return columns, rows


def list_indexes(graph) -> list[dict]:
    """Enumerate indexes via CALL db.indexes() (probed: P0-18).

    Returns [{"label": str, "properties": [str, ...], "types": [...]}, ...].
    """
    res = run_query(
        graph,
        "CALL db.indexes() YIELD label, properties, types, language, stopwords, info",
        read_only=True,
    )
    names = [
        col[1] if isinstance(col, (list, tuple)) and len(col) > 1 else str(col)
        for col in res.header
    ]
    rows = []
    for raw in res.result_set:
        row = dict(zip(names, raw))
        rows.append({
            "label": row.get("label"),
            "properties": list(row.get("properties") or []),
            "types": list(row.get("types") or []),
        })
    return rows


def index_exists(graph, label: str, prop: str) -> bool:
    """True if an index on (label, prop) is listed by CALL db.indexes()."""
    return any(
        row["label"] == label and prop in row["properties"]
        for row in list_indexes(graph)
    )


def ensure_index(graph, label: str, prop: str, *, timeout_ms: int | None = None) -> str:
    """Idempotently ensure an index on (label, prop). Returns 'created' or 'exists'.

    Safe to call repeatedly (Phase 1 index setup):
    - pre-checks via CALL db.indexes() and skips CREATE when already present;
    - if CREATE still hits the specific 'already indexed' constraint error
      (exact probed message, P0-03 — e.g. a concurrent creator), re-verifies
      via db.indexes() and returns 'exists';
    - if the error cannot be confirmed as an existing index, or any other
      failure occurs, the original error propagates unmasked.

    label/prop must be code-owned schema constants matching
    [A-Za-z_][A-Za-z0-9_]* (they are interpolated into the template; no
    input-derived values are ever passed here).
    """
    validate_identifier(label, "label")
    validate_identifier(prop, "property")
    if index_exists(graph, label, prop):
        return "exists"
    try:
        run_query(graph, f"CREATE INDEX FOR (n:{label}) ON (n.{prop})", timeout_ms=timeout_ms)
        return "created"
    except QueryError as exc:
        if exc.kind == "constraint" and "already indexed" in str(exc).lower():
            if index_exists(graph, label, prop):
                return "exists"
        raise
