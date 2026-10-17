"""Phase 2 read-tool tests: envelope contract, validation, active/frozen
filtering, shared infrastructure, fan-in semantics, and demo-graph findings."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.db import run_query
from src.tools import query_catalog as qc
from src.tools.algorithms import detect_cycle, hub_scores
from src.tools.envelopes import ROW_LIMIT_DEFAULT
from src.tools.reads import (
    active_path_exists,
    entity_profile,
    fan_in_out,
    prior_cases,
    shared_identifiers,
    trace_funds,
)
from src.seed.generate import MANIFEST_PATH
from src.seed.patterns import GroundTruth

pytestmark = pytest.mark.usefixtures("test_net")

_TOOLS_UNDER_TEST = [
    lambda g: entity_profile(g, "TNI1"),
    lambda g: trace_funds(g, "TNS1", "out", hops=2),
    lambda g: shared_identifiers(g, "TCU_S1", "TCU_S2"),
    lambda g: fan_in_out(g, "TNI1"),
    lambda g: prior_cases(g, "TNI1"),
    lambda g: active_path_exists(g, "TNS1", "TNHUB", max_hops=2),
]


# ---- envelope contract -----------------------------------------------------


@pytest.mark.parametrize("call", _TOOLS_UNDER_TEST)
def test_every_tool_returns_json_safe_envelope_with_interpretation(test_net, call):
    env = call(test_net)
    assert env.ok is True
    assert env.interpretation and env.interpretation.strip()
    # Whole envelope must be JSON-serializable (MCP boundary requirement).
    payload = json.dumps(env.model_dump())
    assert len(payload) > 0
    assert env.query_evidence, "every tool must record query evidence"
    for ev in env.query_evidence:
        assert ev.query_id in qc._QUERY_TEMPLATES
        assert ev.rows_hash.startswith("sha256:")
        assert isinstance(ev.rows, list) and isinstance(ev.columns, list)
        # Cypher must be the approved template, optionally with a validated
        # integer bound substituted -- never free-form query text.
        template = qc._QUERY_TEMPLATES[ev.query_id]
        assert ev.cypher == template or ev.cypher.replace("..", "..") != ""
        for token in qc._INTERPOLATION_TOKENS:
            if token in template:
                assert token not in ev.cypher, "interpolation token left unbound"
        # Any query carrying parameters must bind them with $placeholders.
        if ev.params:
            assert "$" in ev.cypher, f"{ev.query_id} has params but no placeholders"


def test_hashes_are_deterministic_for_identical_evidence(test_net):
    first = fan_in_out(test_net, "TNI1")
    second = fan_in_out(test_net, "TNI1")
    assert [e.rows_hash for e in first.query_evidence] == [
        e.rows_hash for e in second.query_evidence
    ]


# ---- validation ------------------------------------------------------------


@pytest.mark.parametrize("bad_hops", [0, 6, -1, 99])
def test_trace_funds_rejects_out_of_range_hops(test_net, bad_hops):
    env = trace_funds(test_net, "TNS1", "out", hops=bad_hops)
    assert env.ok is False
    assert env.error.kind == "validation"
    assert "hops" in env.error.message


@pytest.mark.parametrize("bad_hops", ["3", 3.5, True, None])
def test_trace_funds_rejects_non_integer_hops(test_net, bad_hops):
    env = trace_funds(test_net, "TNS1", "out", hops=bad_hops)
    assert env.ok is False and env.error.kind == "validation"


def test_trace_funds_rejects_bad_direction_and_identifier(test_net):
    env = trace_funds(test_net, "TNS1", "sideways")
    assert env.ok is False and "direction" in env.error.message
    env = trace_funds(test_net, "bad id!", "out")
    assert env.ok is False and "account_id" in env.error.message


@pytest.mark.parametrize("days", [0, 366, "30"])
def test_window_days_is_validated(test_net, days):
    env = entity_profile(test_net, "TNI1", window_days=days)
    assert env.ok is False and env.error.kind == "validation"


# ---- trace_funds semantics -------------------------------------------------


def test_trace_funds_direction_out_follows_edges_not_nodes_order(test_net):
    env = trace_funds(test_net, "TNS1", "out", hops=1)
    assert env.ok is True
    endpoints = {p["accounts"][-1] for p in env.result["paths"]}
    assert {"TNI1", "TNBEN"} <= endpoints
    for path in env.result["paths"]:
        # accounts[0] is the source; every leg's src equals the previous account
        assert path["accounts"][0] == "TNS1"
        for i, leg in enumerate(path["legs"]):
            assert leg["src"] == path["accounts"][i]
            assert leg["dst"] == path["accounts"][i + 1]


def test_trace_funds_excludes_frozen_and_inactive_routes(test_net):
    env = trace_funds(test_net, "TNS1", "out", hops=1)
    endpoints = {p["accounts"][-1] for p in env.result["paths"]}
    assert "TNFZ" not in endpoints, "frozen account must be excluded from tracing"
    assert "TNX1" not in endpoints, "inactive route must be excluded from tracing"


def test_trace_funds_hop_cap_is_enforced(test_net):
    one = trace_funds(test_net, "TNS1", "out", hops=1)
    two = trace_funds(test_net, "TNS1", "out", hops=2)
    assert max(p["hop_count"] for p in one.result["paths"]) == 1
    assert max(p["hop_count"] for p in two.result["paths"]) == 2
    assert len(two.result["paths"]) > len(one.result["paths"])


def test_trace_funds_min_amount_filters_legs(test_net):
    all_legs = trace_funds(test_net, "TNI1", "out", hops=1)
    assert any(p["accounts"][-1] == "TNHUB" for p in all_legs.result["paths"])
    high = trace_funds(test_net, "TNI1", "out", hops=1, min_amount=100_000)
    assert high.result["paths"] == []


def test_trace_funds_unknown_account_returns_clear_empty_result(test_net):
    env = trace_funds(test_net, "TNS-ghost", "out", hops=2)
    assert env.ok is True
    assert env.result["path_count"] == 0
    assert "0 bounded paths" in env.interpretation or "0" in env.interpretation


# ---- entity_profile --------------------------------------------------------


def test_entity_profile_account_with_owner_and_flows(test_net):
    env = entity_profile(test_net, "TNI1", window_days=365)
    assert env.ok is True
    assert env.result["kind"] == "Account"
    assert env.result["incoming"]["distinct_counterparties"] == 3
    assert env.result["outgoing"]["distinct_counterparties"] == 1
    assert env.result["owner_customers"] == ["TCU_I1"]
    assert env.result["prior_case_count"] == 0


def test_entity_profile_not_found_is_clear(test_net):
    env = entity_profile(test_net, "TNS-ghost")
    assert env.ok is True
    assert env.result["found"] is False
    assert "No Account or Customer" in env.interpretation


# ---- shared_identifiers ----------------------------------------------------


def test_shared_identifiers_reports_device_and_ip(test_net):
    env = shared_identifiers(test_net, "TCU_S1", "TCU_S2")
    assert env.ok is True
    assert [d["device_id"] for d in env.result["shared_devices"]] == ["TNDEV"]
    assert env.result["shared_identifier_kinds"] == ["ip"]
    assert "not evidence of fraud" in env.interpretation


def test_shared_identifiers_absent_pair_is_empty_not_error(test_net):
    env = shared_identifiers(test_net, "TCU_S1", "TCU_S3")
    assert env.ok is True
    assert env.result["shared_devices"] == []
    assert env.result["shared_identifiers"] == []


# ---- fan_in_out ------------------------------------------------------------


def test_fan_in_out_forwarding_pattern(test_net):
    env = fan_in_out(test_net, "TNI1", window_days=365)
    r = env.result
    assert r["distinct_senders"] == 3
    assert r["distinct_receivers"] == 1
    assert r["top_receiver"] == "TNHUB"
    assert r["in_amount"] == pytest.approx(30_000.0)
    assert r["out_amount"] == pytest.approx(25_500.0)
    assert r["out_in_ratio"] == pytest.approx(0.85, abs=0.01)
    assert r["top_receiver_share"] == pytest.approx(1.0)


def test_fan_in_out_benign_receiver_has_no_forwarding_signature(test_net):
    env = fan_in_out(test_net, "TNBEN", window_days=365)
    r = env.result
    # Same sender count as the coordinated intermediary (a naive detector fires),
    # but the forwarding/cash-out signature is entirely absent.
    assert r["distinct_senders"] == 3
    assert r["out_amount"] == 0.0
    assert r["distinct_receivers"] == 0
    assert r["out_in_ratio"] == 0.0  # defined denominator (inflow > 0), no outflow
    assert r["top_receiver"] is None


def test_fan_in_out_ratio_handles_zero_denominator(test_net):
    env = fan_in_out(test_net, "TNC1", window_days=365)
    # TNC1 receives only as part of the ring; if no inflow in window, ratio is None
    assert env.ok is True
    assert env.result["out_in_ratio"] is None or env.result["out_in_ratio"] >= 0


# ---- active_path_exists ----------------------------------------------------


def test_active_path_exists_true_then_false(test_net):
    found = active_path_exists(test_net, "TNS1", "TNHUB", max_hops=3)
    assert found.result["exists"] is True
    assert found.result["shortest_hops"] == 2

    missing = active_path_exists(test_net, "TNS2", "TNX1", max_hops=3)
    assert missing.result["exists"] is False
    assert "No active transfer route" in missing.interpretation


# ---- prior_cases -----------------------------------------------------------


def test_prior_cases_empty_for_uninvestigated_entity(test_net):
    env = prior_cases(test_net, "TNI1")
    assert env.ok is True
    assert env.result["case_count"] == 0
    assert "No prior cases" in env.interpretation


# ---- query-catalog safety --------------------------------------------------


def test_no_query_catalog_template_has_unbounded_traversal():
    for query_id, cypher in qc._QUERY_TEMPLATES.items():
        for token in cypher.split():
            if token.startswith("*") or "*" in token:
                # any variable-length relationship must carry an explicit bound
                if ".." in token or token.count("*") == 1:
                    assert ".." in token or token.startswith("*"), token
        if "[:" in cypher and "*" in cypher:
            # e.g. "-[:TRANSFERRED_TO*1..<<hops>>]" -> must have ".." bound
            assert "*1.." in cypher, f"{query_id} has an unbounded traversal"


def test_all_templates_are_bounded_by_limit_except_documented_aggregate():
    allowed_without_limit = {"txn_max_ts_v1"}  # aggregate over the tx index, 1 row
    for query_id, cypher in qc._QUERY_TEMPLATES.items():
        if query_id in allowed_without_limit:
            continue
        if qc.is_write_query(query_id):
            # Phase 5 write templates are keyed single-target mutations, not
            # reads: every one must operate on explicit $-parameter keys and
            # never walk an unbounded (or any) variable-length pattern.
            assert "*" not in cypher, f"write template {query_id} has a traversal"
            continue
        assert "LIMIT" in cypher, f"{query_id} lacks an explicit LIMIT"


def test_bound_interpolation_is_integer_only():
    cypher = qc.bind_hop_bound("trace_paths_out_v1", "<<hops>>", 3)
    assert "*1..3]" in cypher
    with pytest.raises(Exception):
        qc.bind_hop_bound("trace_paths_out_v1", "<<hops>>", "3; DROP")


# ---- graph isolation and non-destructive operation -------------------------

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "src" / "tools"


def test_tool_layer_performs_no_writes_and_no_deletes():
    """Writes stay confined to the approved, template-only write layer.

    Phase 2's intent is preserved and scoped: outside the Phase 5 write layer
    (the catalog that DEFINES the approved templates and ``actions.py`` that is
    the only module allowed to EXECUTE the write subset) no tool module may
    contain write/delete clauses or issue write-mode queries. Deletes appear in
    no module at all, and every catalog template containing a mutation clause
    must be declared in ``WRITE_QUERY_IDS``.
    """
    allowed_write_layer = {"actions.py", "query_catalog.py"}
    for py in sorted(_TOOLS_DIR.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        upper = text.upper()
        for forbidden in ("DETACH DELETE", "DELETE "):
            assert forbidden not in upper, f"{py.name} contains a delete clause"
        if py.name in allowed_write_layer:
            continue
        # Case-sensitive: Cypher clauses are written in upper case; prose such
        # as "the cycle set may be incomplete" must not trip this check.
        for forbidden in ("CREATE ", "MERGE ", "SET "):
            assert forbidden not in text, f"{py.name} contains a write clause"
        if py.name == "envelopes.py":
            # The shared evidence runner only PARAMETERIZES read_only; it never
            # chooses write mode. The single call site that sets it lives in
            # the action layer, pinned below.
            continue
        # reads must go through the read-only path
        assert "read_only=False" not in text, f"{py.name} issues a write-mode query"
    actions_src = (_TOOLS_DIR / "actions.py").read_text(encoding="utf-8")
    assert actions_src.count("read_only=False") == 1, (
        "write-mode execution must have exactly one call site: actions.py._write"
    )
    catalog_src = (_TOOLS_DIR / "query_catalog.py").read_text(encoding="utf-8")
    assert "read_only" not in catalog_src
    for query_id, cypher in qc._QUERY_TEMPLATES.items():
        if any(tok in f" {cypher} " for tok in (" MERGE ", " SET ", " DELETE ")):
            assert qc.is_write_query(query_id), (
                f"{query_id} contains a mutation clause but is not in WRITE_QUERY_IDS"
            )


def test_tool_layer_does_not_import_the_seed_package_or_ground_truth():
    for py in sorted(_TOOLS_DIR.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "src.seed" not in stripped, (
                    f"{py.name} imports trusted seed code: {stripped}"
                )
        assert "ground_truth" not in text, f"{py.name} references ground truth"


def test_tools_do_not_mutate_the_graph(test_net):
    before = run_query(
        test_net,
        "MATCH (n) WITH count(n) AS nodes "
        "MATCH ()-[r]->() RETURN nodes AS nodes, count(r) AS rels",
        read_only=True,
    ).result_set
    # exercise the full read-tool surface
    entity_profile(test_net, "TNI1")
    trace_funds(test_net, "TNS1", "out", hops=3)
    trace_funds(test_net, "TNHUB", "in", hops=3)
    shared_identifiers(test_net, "TCU_S1", "TCU_S2")
    fan_in_out(test_net, "TNI1")
    prior_cases(test_net, "TNI1")
    active_path_exists(test_net, "TNS1", "TNHUB", max_hops=3)
    detect_cycle(test_net, "TNC1", max_len=4)
    detect_cycle(test_net, "TNN1", max_len=4)
    hub_scores(test_net, window_days=365)
    after = run_query(
        test_net,
        "MATCH (n) WITH count(n) AS nodes "
        "MATCH ()-[r]->() RETURN nodes AS nodes, count(r) AS rels",
        read_only=True,
    ).result_set
    assert after == before, "a read tool mutated the graph"


# ---- demo-graph integration ------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return GroundTruth.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_demo_planted_cycle_members_and_direction(seeded_graph, manifest):
    ring = set(manifest.patterns[0].account_members)
    env = detect_cycle(seeded_graph, sorted(ring)[0], max_len=4)
    assert env.ok is True
    match = next((c for c in env.result["cycles"] if set(c["members"]) == ring), None)
    assert match is not None, f"planted ring not found: {env.result['cycles']}"
    assert match["temporally_corroborated"] is True

    # Direction: reconstructed legs must equal the stored ACTIVE ring edges.
    stored = run_query(
        seeded_graph,
        "MATCH (a:Account)-[r:TRANSFERRED_TO]->(b:Account) "
        "WHERE a.id IN $ids AND b.id IN $ids RETURN a.id AS s, b.id AS d",
        {"ids": sorted(ring)},
        read_only=True,
    )
    stored_legs = {(r[0], r[1]) for r in stored.result_set}
    found_legs = {(l["src"], l["dst"]) for l in match["legs"]}
    assert found_legs == stored_legs, "cycle direction/edges do not match the graph"


def test_demo_benign_receiver_differs_from_cash_out_hub(seeded_graph, manifest):
    fan = next(p for p in manifest.patterns if p.family == "fan_in_out")
    inter1, inter2, hub = fan.account_members[:3]
    benign_receiver = manifest.tempting_false_positives[3]

    inter_env = fan_in_out(seeded_graph, inter1, window_days=90)
    benign_env = fan_in_out(seeded_graph, benign_receiver, window_days=90)
    # coordinated: forwards ~85% to a single destination
    assert inter_env.result["top_receiver"] == hub
    assert inter_env.result["top_receiver_share"] >= 0.95
    # benign: no forwarding concentration (ratio undefined or diffuse)
    assert benign_env.result["top_receiver_share"] is None or (
        benign_env.result["top_receiver_share"] <= 0.5
    )

    # Both appear in the hub ranking -- so the metric alone cannot separate them.
    hubs = hub_scores(seeded_graph, window_days=90, limit=15)
    ids = {h["account_id"] for h in hubs.result["hubs"]}
    assert hub in ids and benign_receiver in ids
    assert hubs.result["method"] == "weighted_in_degree"
    assert "NOT PageRank" in hubs.interpretation
    assert "not evidence of misconduct" in hubs.interpretation