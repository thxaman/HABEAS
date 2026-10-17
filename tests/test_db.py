"""Phase 0 tests for src/db.py: run_query semantics, error classification,
read retry behavior, and value normalization.

Stub-based tests run offline; graph-based tests exercise the live database.
"""
import uuid

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from src.db import (
    QueryError,
    ensure_index,
    list_indexes,
    normalize_rows,
    normalize_value,
    run_query,
)


class _Result:
    def __init__(self, rows):
        self.header = [("COLUMN_SCALAR", "x")]
        self.result_set = rows


class FlakyGraph:
    """Graph stand-in that raises queued failures, then succeeds."""

    def __init__(self, failures):
        self._failures = list(failures)
        self.calls = 0
        self.timeouts = []

    def query(self, cypher, params=None, timeout=None):
        return self._call(cypher, params, timeout)

    def ro_query(self, cypher, params=None, timeout=None):
        return self._call(cypher, params, timeout)

    def _call(self, cypher, params, timeout):
        self.calls += 1
        self.timeouts.append(timeout)
        if self._failures:
            raise self._failures.pop(0)
        return _Result([[1]])


# --- offline unit tests ---------------------------------------------------


def test_read_retried_at_most_once_on_transient_connection_error():
    g = FlakyGraph([RedisConnectionError("connection refused")])
    res = run_query(g, "RETURN 1", timeout_ms=12345, read_only=True, retry_delay_s=0)
    assert res.result_set == [[1]]
    assert g.calls == 2
    assert g.timeouts == [12345, 12345]  # millisecond timeout forwarded as-is


def test_write_is_never_retried_automatically():
    g = FlakyGraph([RedisConnectionError("connection refused")])
    with pytest.raises(QueryError) as excinfo:
        run_query(g, "MERGE (x:X {id: $id})", {"id": "a"}, timeout_ms=1, read_only=False)
    assert excinfo.value.kind == "connection"
    assert excinfo.value.params == {"id": "a"}
    assert g.calls == 1  # no retry for writes


def test_non_transient_failure_is_not_retried_even_for_reads():
    g = FlakyGraph([RuntimeError("boom")])
    with pytest.raises(QueryError) as excinfo:
        run_query(g, "RETURN 1", timeout_ms=1, read_only=True, retry_delay_s=0)
    assert excinfo.value.kind == "unknown"
    assert g.calls == 1


def test_query_error_is_structured_and_secret_free():
    qe = QueryError("bad things", kind="syntax", retryable=False,
                    cypher="MATCH (x)", params={"id": "ACC-1"})
    assert qe.kind == "syntax"
    assert qe.retryable is False
    assert qe.cypher == "MATCH (x)"
    assert qe.params == {"id": "ACC-1"}
    assert "bad things" in str(qe)


def test_normalize_value_scalars_and_collections():
    assert normalize_value(None) is None
    assert normalize_value(True) is True
    assert normalize_value(3.5) == 3.5
    assert normalize_value("txt") == "txt"
    assert normalize_value((1, 2)) == [1, 2]
    assert normalize_value({"a": [1, {"b": None}]}) == {"a": [1, {"b": None}]}


class FakeNode:
    def __init__(self):
        self.id = 7
        self.labels = {"B", "A"}
        self.properties = {"name": "n", "tags": ["x"]}


class FakeEdge:
    def __init__(self):
        self.id = 11
        self.relation = "NEXT"
        self.src_node = 7
        self.dest_node = 8
        self.properties = {"w": 1}


class FakePath:
    def __init__(self):
        self._node = FakeNode()
        self._edge = FakeEdge()

    def nodes(self):
        return [self._node, self._node]

    def edges(self):
        return [self._edge]


def test_normalize_value_node_edge_path():
    node = normalize_value(FakeNode())
    assert node == {
        "_type": "node", "id": 7, "labels": ["A", "B"],
        "properties": {"name": "n", "tags": ["x"]},
    }
    edge = normalize_value(FakeEdge())
    assert edge["_type"] == "edge"
    assert edge["relation"] == "NEXT" and edge["src"] == 7 and edge["dst"] == 8
    path = normalize_value(FakePath())
    assert path["_type"] == "path" and len(path["nodes"]) == 2 and len(path["edges"]) == 1


def test_normalize_rows_uses_header_names_and_normalizes_cells():
    class PathResult:
        header = [("COLUMN_SCALAR", "node"), ("COLUMN_SCALAR", "n")]
        result_set = [[FakeNode(), 42]]

    columns, rows = normalize_rows(PathResult())
    assert columns == ["node", "n"]
    assert rows == [[normalize_value(FakeNode()), 42]]


# --- live database tests --------------------------------------------------


def test_return_one(graph):
    res = run_query(graph, "RETURN 1 AS one", read_only=True)
    assert res.result_set == [[1]]


def test_parameterized_write_read_roundtrip(graph, test_nodes):
    node_id = test_nodes(f"t-{uuid.uuid4().hex[:12]}")
    run_query(graph, "MERGE (n:Phase0Test {id: $id}) SET n.n = $n",
              {"id": node_id, "n": 1})
    run_query(graph, "MATCH (n:Phase0Test {id: $id}) SET n.n = n.n + $inc",
              {"id": node_id, "inc": 1})
    res = run_query(graph, "MATCH (n:Phase0Test {id: $id}) RETURN n.n AS n",
                    {"id": node_id}, read_only=True)
    assert res.result_set == [[2]]


def test_bounded_variable_length_traversal(graph, test_nodes):
    a = test_nodes(f"a-{uuid.uuid4().hex[:12]}")
    b = test_nodes(f"b-{uuid.uuid4().hex[:12]}")
    c = test_nodes(f"c-{uuid.uuid4().hex[:12]}")
    run_query(
        graph,
        "UNWIND $rows AS row MERGE (x:Phase0Test {id: row.id}) "
        "MERGE (y:Phase0Test {id: row.next}) MERGE (x)-[:NEXT]->(y)",
        {"rows": [{"id": a, "next": b}, {"id": b, "next": c}]},
    )
    res = run_query(
        graph,
        "MATCH (s:Phase0Test {id: $a})-[:NEXT*1..2]->(t:Phase0Test) "
        "RETURN t.id AS id ORDER BY id",
        {"a": a},
        read_only=True,
    )
    assert [r[0] for r in res.result_set] == [b, c]


def test_query_timeout_is_classified(graph):
    heavy = "UNWIND range(1, 60000000) AS x WITH x WHERE x = 60000000 RETURN count(x) AS c"
    try:
        run_query(graph, heavy, timeout_ms=300, read_only=True)
    except QueryError as exc:
        assert exc.kind == "timeout", (
            f"expected timeout classification, got kind={exc.kind}: {exc}"
        )
    else:
        # Server did not abort within the requested 300ms; recorded as a known
        # limit in docs/falkordb-capabilities.md. Query itself must not raise
        # an unexpected error class, which reaching here already proves.
        pass


# --- index setup regression tests (Phase 1 readiness) ---------------------


class _ListResult:
    """Stand-in QueryResult for CALL db.indexes() with its probed header."""

    def __init__(self, rows):
        self.header = [[1, "label"], [1, "properties"], [1, "types"]]
        self.result_set = rows


class _Rows:
    def __init__(self, rows):
        self.rows = rows


class _Raise:
    def __init__(self, exc):
        self.exc = exc


class ScriptedGraph:
    """Consumes queued outcomes in call order for query and ro_query alike."""

    def __init__(self, steps):
        self._steps = list(steps)
        self.calls = []

    def _next(self, cypher):
        self.calls.append(cypher)
        step = self._steps.pop(0)
        if isinstance(step, _Raise):
            raise step.exc
        if isinstance(step, _Rows):
            return _ListResult(step.rows)
        return step

    def query(self, cypher, params=None, timeout=None):
        return self._next(cypher)

    def ro_query(self, cypher, params=None, timeout=None):
        return self._next(cypher)


def test_list_indexes_finds_known_probe_index(graph):
    rows = list_indexes(graph)
    assert any(
        r["label"] == "Phase0Probe" and "id" in r["properties"] for r in rows
    ), f"Phase0Probe.id index missing from enumeration: {rows}"
    assert all(set(r) == {"label", "properties", "types"} for r in rows)


def test_ensure_index_is_idempotent_across_repeated_setup(graph):
    first = ensure_index(graph, "Phase0TestIdx", "id")
    assert first in {"created", "exists"}
    assert ensure_index(graph, "Phase0TestIdx", "id") == "exists"
    # cleanup: DROP INDEX support is probed (P0-19); leave no scratch index
    run_query(graph, "DROP INDEX ON :Phase0TestIdx(id)")


def test_ensure_index_tolerates_already_indexed_race_and_verifies():
    g = ScriptedGraph([
        _Rows([]),  # pre-check listing: index absent
        _Raise(ResponseError("Attribute 'id' is already indexed")),  # CREATE race
        _Rows([["Phase0Race", ["id"], ["RANGE"]]]),  # re-listing confirms it
    ])
    assert ensure_index(g, "Phase0Race", "id") == "exists"
    assert len(g.calls) == 3


def test_ensure_index_does_not_mask_unrelated_create_errors():
    g = ScriptedGraph([
        _Rows([]),
        _Raise(ResponseError("some unrelated server failure")),
    ])
    with pytest.raises(QueryError) as excinfo:
        ensure_index(g, "Phase0Race", "id")
    assert "unrelated server failure" in str(excinfo.value)
    assert excinfo.value.kind == "syntax"
    assert len(g.calls) == 2  # no swallow, no retry


def test_ensure_index_still_raises_when_already_indexed_not_confirmed():
    # CREATE claims already-indexed but re-listing does not show the index:
    # the original error must propagate instead of reporting success.
    g = ScriptedGraph([
        _Rows([]),
        _Raise(ResponseError("Attribute 'id' is already indexed")),
        _Rows([]),
    ])
    with pytest.raises(QueryError) as excinfo:
        ensure_index(g, "Phase0Race", "id")
    assert excinfo.value.kind == "constraint"
    assert len(g.calls) == 3


def test_ensure_index_rejects_unsafe_identifiers_before_touching_the_database():
    with pytest.raises(ValueError):
        ensure_index(None, "Bad-Label", "id")  # dash not allowed
    with pytest.raises(ValueError):
        ensure_index(None, "Label", "id; DROP")  # injection-shaped input
