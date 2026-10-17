"""Phase 1 tests: deterministic seed model, manifest validation, planted
pattern detection (model-level and live), projection invariants, and the
reset guard (PLAN.md §6/§14 Phase 1 gate)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.db import run_query
from src.seed import patterns as P
from src.seed.generate import (
    MANIFEST_PATH,
    SeedRefused,
    build_model,
    main,
    plan_reset,
    seed,
)
from src.seed.patterns import GroundTruth

DEMO_GRAPH = "HABEAS_demo_local"
EVENT_FETCH_LIMIT = 5000  # fixture graph holds ~1.6k events; assert no truncation


# ---- pure model tests (no database) ---------------------------------------


@pytest.fixture(scope="module")
def model():
    return build_model(seed=42, graph_name=DEMO_GRAPH)


def test_same_seed_builds_identical_model():
    assert build_model(seed=42, graph_name=DEMO_GRAPH) == build_model(
        seed=42, graph_name=DEMO_GRAPH
    )


def test_different_seed_changes_the_data():
    m1 = build_model(seed=42, graph_name=DEMO_GRAPH)
    m2 = build_model(seed=43, graph_name=DEMO_GRAPH)
    assert m1.txn_rows != m2.txn_rows
    assert m1.manifest.alerts[0].alert_id == m2.manifest.alerts[0].alert_id  # ids are counters
    first_txn_42 = m1.txn_rows[0]["txn"]
    first_txn_43 = m2.txn_rows[0]["txn"]
    assert (first_txn_42["amount"], first_txn_42["ts"]) != (
        first_txn_43["amount"],
        first_txn_43["ts"],
    )


def test_no_id_collisions(model):
    seen: set[str] = set()
    id_lists = [
        [c["id"] for c in model.customers],
        [a["id"] for a in model.accounts],
        [r["txn"]["id"] for r in model.txn_rows],
        [d["id"] for d in model.devices],
        [i["id"] for i in model.identifiers],
        [a["id"] for a in model.alerts],
    ]
    for id_list in id_lists:
        assert len(id_list) == len(set(id_list)), "collision within one id type"
        assert not (set(id_list) & seen), "collision across id types"
        seen |= set(id_list)


def test_all_transfers_positive_and_valid(model):
    account_ids = {a["id"] for a in model.accounts}
    window = ("2026-01-01T00:00:00Z", "2026-03-31T23:59:59Z")
    for row in model.txn_rows:
        txn = row["txn"]
        assert txn["amount"] > 0
        assert window[0] <= txn["ts"] <= window[1]
        assert row["src"] in account_ids and row["dst"] in account_ids
        assert row["src"] != row["dst"]


def test_population_within_plan_ranges(model):
    counts = model.manifest.counts
    assert 300 <= counts["nodes:Customer"] <= 400
    assert 350 <= counts["nodes:Account"] <= 500
    assert 1500 <= counts["nodes:Transaction"] <= 2500
    assert counts["nodes:Alert"] == 21


def test_manifest_validates(model):
    manifest = GroundTruth.model_validate(model.manifest.model_dump())
    assert manifest.seed == 42
    planted = [a for a in manifest.alerts if a.label == "suspicious"]
    benign = [a for a in manifest.alerts if a.label == "benign"]
    assert len(planted) + len(benign) == 21
    assert {a.family for a in planted} == {
        "cycle", "cycle_rapid", "fan_in_out", "rapid_shared"}
    assert all(a.family == "benign" for a in benign)


def test_reset_guard_all_combos():
    assert plan_reset(DEMO_GRAPH, False, 0) == "seed"
    assert plan_reset(DEMO_GRAPH, True, 0) == "seed"  # nothing to reset
    assert plan_reset(DEMO_GRAPH, True, 1234) == "reset_and_seed"
    with pytest.raises(SeedRefused):
        plan_reset(DEMO_GRAPH, False, 1234)  # non-empty, no explicit flag
    with pytest.raises(SeedRefused):
        plan_reset("customer_prod", False, 0)  # never touch non-demo names
    with pytest.raises(SeedRefused):
        plan_reset("customer_prod", True, 0)  # even with the flag


def test_planted_patterns_detectable_in_model(model):
    events = model.events
    adjacency: dict[str, list[str]] = {}
    for src, dst, _, _ in events:
        adjacency.setdefault(src, []).append(dst)

    cycles = P.find_cycles(adjacency, max_len=4)
    cycle_members = set(model.manifest.patterns[0].account_members)
    assert any(set(c) == cycle_members for c in cycles), "planted cycle not found"
    cycle_events = [
        e for e in events
        if e[0] in cycle_members and e[1] in cycle_members and e[0] != e[1]
    ]
    # planted cycle has exactly its ring edges (plus nothing else between members)
    assert all({e[0], e[1]} <= cycle_members for e in cycle_events)
    assert P.cycle_temporally_consistent(cycle_events), (
        "planted cycle should be temporally corroborated"
    )

    fan = next(p for p in model.manifest.patterns if p.family == "fan_in_out")
    inter1, inter2, hub = fan.account_members[:3]
    for inter in (inter1, inter2):
        prof = P.fan_in_out(events, inter)
        assert prof["distinct_senders"] >= 5
        assert 0.8 <= prof["out_in_ratio"] <= 0.9  # ~85% forwarded
        assert prof["top_receiver"] == hub
        assert prof["top_receiver_share"] >= 0.95  # single cash-out destination

    rapid = next(p for p in model.manifest.patterns if p.family == "rapid_shared")
    r1, r2 = rapid.account_members[:2]
    for acc in (r1, r2):
        matches = P.rapid_in_out(events, acc, max_gap_minutes=90)
        assert len(matches) >= 4
        assert all(m["gap_minutes"] <= 90 for m in matches)  # design: 10–90 min


def test_cycle_rapid_pattern_has_distinct_provenance(model):
    """The corrected-policy qualifying pattern (append-only addition).

    A slow ring (internal gaps far above the rapid threshold) plus exactly ONE
    rapid pass-through on the alerted member, whose two transactions are
    disjoint from the ring's transactions — so the circulation and rapid
    signal families have genuinely distinct provenance.
    """
    events = model.events
    pat = next(p for p in model.manifest.patterns if p.family == "cycle_rapid")
    ring = pat.ring_members
    assert len(ring) == 4
    member_set = set(ring)
    feeder, exit_acc = [a for a in pat.account_members if a not in member_set]

    ring_events = [e for e in events if e[0] in member_set and e[1] in member_set]
    assert len(ring_events) == 12  # 3 rounds x 4 legs
    assert P.cycle_temporally_consistent(ring_events)
    ring_tx_ts = {e[2] for e in ring_events}

    alerted = pat.account_members[0]
    assert alerted in member_set
    matches = P.rapid_in_out(events, alerted)
    assert len(matches) == 1, "the ring alone must never produce rapid matches"
    match = matches[0]
    assert match["gap_minutes"] <= 90
    assert match["forward_ratio"] >= 0.8
    rapid_ts = {match["in_txn_ts"], match["out_ts"]}
    assert not (rapid_ts & ring_tx_ts), "rapid evidence overlaps the ring"
    for member in ring[1:]:
        assert P.rapid_in_out(events, member) == []

    entries = {(r["src"], r["dst"], r["txn"]["ts"]): r["txn"]["id"]
               for r in model.txn_rows}
    assert (feeder, alerted, match["in_txn_ts"]) in entries
    assert (alerted, exit_acc, match["out_ts"]) in entries

    truth = next(a for a in model.manifest.alerts if a.family == "cycle_rapid")
    assert truth.account_id == alerted
    assert truth.trigger_transaction_id == entries[
        (feeder, alerted, match["in_txn_ts"])]
    assert pat.customer_members


def test_tempting_false_positives_need_multiple_signals(model):
    events = model.events
    benign_receiver = model.manifest.tempting_false_positives[3]
    prof = P.fan_in_out(events, benign_receiver)
    # Naive fan-in detector WOULD fire on this benign account...
    assert prof["distinct_senders"] >= 8
    # ...but the distinguishing signals are absent: no forwarding concentration
    # and no rapid movement.
    assert prof["top_receiver_share"] is None or prof["top_receiver_share"] <= 0.5
    assert P.rapid_in_out(events, benign_receiver) == []
    # Random noise can still place it in an accidental structural cycle; such
    # cycles must NOT be temporally consistent — that is the distinguishing
    # signal vs the planted ring (which is temporally corroborated).
    adjacency: dict[str, list[str]] = {}
    for src, dst, _, _ in events:
        adjacency.setdefault(src, []).append(dst)
    member_set = set(model.manifest.patterns[0].account_members)
    for cycle in P.find_cycles(adjacency, max_len=4):
        if benign_receiver in cycle:
            legs = [e for e in events if e[0] in cycle and e[1] in cycle]
            assert not P.cycle_temporally_consistent(legs), (
                f"accidental cycle through the benign trap is temporally "
                f"consistent: {cycle}"
            )
        assert not (set(cycle) <= member_set) or set(cycle) == member_set


def test_agent_layers_never_reference_ground_truth():
    """The live agent must not reach the manifest (PLAN.md §5/§10); eval may."""
    for pkg in ("src/tools", "src/agent", "src/report", "src/demo"):
        pkg_dir = Path(pkg)
        if not pkg_dir.exists():
            continue
        for py in pkg_dir.rglob("*.py"):
            assert "ground_truth" not in py.read_text(encoding="utf-8"), (
                f"{py} references the ground-truth manifest"
            )


# ---- live graph tests ------------------------------------------------------


@pytest.fixture(scope="module")
def manifest(seeded_graph):
    return GroundTruth.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))


def _fetch_events(graph) -> list[P.Event]:
    res = run_query(
        graph,
        "MATCH (a:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(b:Account) "
        "RETURN a.id AS src, b.id AS dst, t.ts AS ts, t.amount AS amount "
        f"ORDER BY t.id LIMIT {EVENT_FETCH_LIMIT}",
        read_only=True,
    )
    rows = [(r[0], r[1], r[2], float(r[3])) for r in res.result_set]
    assert len(rows) < EVENT_FETCH_LIMIT, "event fetch truncated; raise the limit"
    return rows


def _fetch_projection_edges(graph) -> list[dict]:
    res = run_query(
        graph,
        "MATCH (a:Account)-[e:TRANSFERRED_TO]->(b:Account) "
        "RETURN a.id AS src, b.id AS dst, e.count AS count, "
        "e.total_amount AS total, e.last_ts AS last_ts "
        "ORDER BY a.id, b.id LIMIT 5000",
        read_only=True,
    )
    rows = [
        {"src": r[0], "dst": r[1], "count": int(r[2]), "total_amount": float(r[3]),
         "last_ts": r[4]}
        for r in res.result_set
    ]
    assert len(rows) < 5000
    return rows


def test_seeded_graph_counts_match_manifest(seeded_graph, manifest):
    from src.seed.schema import graph_counts

    assert graph_counts(seeded_graph) == manifest.counts
    # Manifest file itself validates (Pydantic) and is deterministic content.
    assert GroundTruth.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_same_seed_reseeds_identically(seeded_graph, settings):
    from src.seed.schema import graph_counts

    counts_before = graph_counts(seeded_graph)
    manifest_before = MANIFEST_PATH.read_bytes()
    seed(settings, seed=42, reset=True, graph=seeded_graph)
    assert graph_counts(seeded_graph) == counts_before
    assert MANIFEST_PATH.read_bytes() == manifest_before


def test_projection_matches_event_ledger(seeded_graph, manifest):
    events = _fetch_events(seeded_graph)
    ledger: dict[tuple[str, str], dict] = {}
    for src, dst, ts, amount in events:
        agg = ledger.setdefault((src, dst), {"count": 0, "total": 0.0, "last_ts": ts})
        agg["count"] += 1
        agg["total"] += amount
        agg["last_ts"] = max(agg["last_ts"], ts)

    edges = _fetch_projection_edges(seeded_graph)
    assert len(edges) == manifest.counts["rels:TRANSFERRED_TO"]
    assert {(e["src"], e["dst"]) for e in edges} == set(ledger)
    for edge in edges:
        agg = ledger[(edge["src"], edge["dst"])]
        assert edge["count"] == agg["count"]
        assert edge["total_amount"] == pytest.approx(agg["total"], abs=0.005)
        assert edge["last_ts"] == agg["last_ts"]
        assert edge["last_ts"] == max(
            ts for s, d, ts, _ in events if s == edge["src"] and d == edge["dst"]
        )


def test_planted_cycle_detectable_via_live_cypher(seeded_graph, manifest):
    cycle_members = set(manifest.patterns[0].account_members)
    probe = sorted(cycle_members)[0]
    # P0-08 template (bounded self-returning directed path). Known quirk: for
    # closed paths nodes(path) order is reversed, so compare as sets minus the
    # repeated closing element (see docs/falkordb-capabilities.md P0-17).
    res = run_query(
        seeded_graph,
        "MATCH path = (a:Account {id: $id})-[:TRANSFERRED_TO*1..4]->(a) "
        "RETURN [n IN nodes(path) | n.id] AS members LIMIT 5",
        {"id": probe},
        read_only=True,
    )
    found = [set(row[0][:-1]) for row in res.result_set if len(row[0]) == 5]
    assert any(members == cycle_members for members in found), (
        f"planted cycle not found via live Cypher; got {found}"
    )


def test_planted_hub_pattern_via_live_graph(seeded_graph, manifest):
    events = _fetch_events(seeded_graph)
    edges = _fetch_projection_edges(seeded_graph)
    fan = next(p for p in manifest.patterns if p.family == "fan_in_out")
    inter1, inter2, hub = fan.account_members[:3]

    # structural: the cash-out hub receives from exactly the two intermediaries
    in_edges = [e for e in edges if e["dst"] == hub]
    assert {e["src"] for e in in_edges} == {inter1, inter2}
    for edge in in_edges:
        intake = sum(a for s, d, _, a in events if d == edge["src"])
        assert edge["total_amount"] == pytest.approx(intake * 0.85, rel=0.01)

    # hub metric (weighted in-degree, NOT PageRank): hub and the benign
    # fan-in trap both rank high — the hub must need a second signal.
    hubs = P.weighted_in_degree(edges)
    top10 = {h["account_id"] for h in hubs[:10]}
    assert hub in top10
    assert manifest.tempting_false_positives[3] in top10


def test_planted_rapid_and_shared_infrastructure_live(seeded_graph, manifest):
    events = _fetch_events(seeded_graph)
    rapid = next(p for p in manifest.patterns if p.family == "rapid_shared")
    r1, r2 = rapid.account_members[:2]
    cust_a, cust_b = rapid.customer_members
    for acc in (r1, r2):
        matches = P.rapid_in_out(events, acc, max_gap_minutes=90)
        assert len(matches) >= 4

    # planted shared infrastructure: both rapid customers share device AND ip
    res = run_query(
        seeded_graph,
        "MATCH (c1:Customer {id: $a})-[:USES_DEVICE]->(d:Device)"
        "<-[:USES_DEVICE]-(c2:Customer {id: $b}) RETURN d.id AS device LIMIT 5",
        {"a": cust_a, "b": cust_b},
        read_only=True,
    )
    assert [r[0] for r in res.result_set], "planted shared device missing"
    res = run_query(
        seeded_graph,
        "MATCH (c1:Customer {id: $a})-[:HAS_IDENTIFIER]->(i:Identifier)"
        "<-[:HAS_IDENTIFIER]-(c2:Customer {id: $b}) "
        "RETURN i.id AS ident, i.kind AS kind LIMIT 5",
        {"a": cust_a, "b": cust_b},
        read_only=True,
    )
    kinds = {r[1] for r in res.result_set}
    assert "ip" in kinds, "planted shared identifier missing"

    # tempting trap: the innocent shared device exists but its customers
    # show no rapid movement and are not planted members.
    device_id = manifest.tempting_false_positives[4]
    res = run_query(
        seeded_graph,
        "MATCH (c:Customer)-[:USES_DEVICE]->(d:Device {id: $d}) "
        "RETURN collect(c.id) AS customers",
        {"d": device_id},
        read_only=True,
    )
    sharers = set(res.result_set[0][0])
    assert len(sharers) >= 3  # owner + 2 innocent extra sharers
    planted_members = {
        m for p in manifest.patterns for m in p.account_members
    }
    assert not (sharers & planted_members)


def test_reset_without_flag_is_refused(seeded_graph, settings):
    from src.seed.schema import graph_counts

    counts_before = graph_counts(seeded_graph)
    assert main(["--seed", "42"]) == 2  # refused: graph is non-empty
    assert graph_counts(seeded_graph) == counts_before  # nothing was touched


def test_no_ground_truth_markers_on_graph_nodes(seeded_graph):
    res = run_query(
        seeded_graph,
        "MATCH (n) UNWIND keys(n) AS k RETURN DISTINCT k",
        read_only=True,
    )
    keys = {r[0] for r in res.result_set}
    forbidden = {
        "is_fraud", "pattern_type", "ground_truth_members", "is_planted",
        "family", "signals", "label",
    }
    assert not (keys & forbidden), f"ground-truth-like properties on nodes: {keys & forbidden}"


def test_alerts_are_linked_to_transaction_and_account(seeded_graph, manifest):
    res = run_query(
        seeded_graph,
        "MATCH (al:Alert)-[:TRIGGERED_BY]->(t:Transaction), (al)-[:CONCERNS]->(a:Account) "
        "RETURN al.id AS id, t.id AS txn, a.id AS acc ORDER BY id LIMIT 100",
        read_only=True,
    )
    rows = {r[0]: (r[1], r[2]) for r in res.result_set}
    assert len(rows) == manifest.counts["nodes:Alert"] == 21
    for truth in manifest.alerts:
        assert rows[truth.alert_id] == (
            truth.trigger_transaction_id, truth.account_id
        )


def test_manifest_file_is_not_committed():
    # .gitignore must cover the manifest (it is generated, never committed).
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "src/seed/ground_truth.json" in gitignore
