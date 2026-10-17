"""Phase 2 algorithm tests: canonicalization, temporal corroboration, cycle
detection behavior on controlled and demo data, and hub-score labeling."""
from __future__ import annotations

import pytest

from src.seed.generate import MANIFEST_PATH
from src.seed.patterns import GroundTruth
from src.tools.algorithms import (
    canonical_cycle,
    cycle_temporally_consistent,
    detect_cycle,
    hub_scores,
)

# ---- pure helpers (no database) --------------------------------------------


def test_canonical_cycle_rotates_to_smallest_and_dedupes_rotations():
    assert canonical_cycle(["D", "A", "C", "B"]) == ("A", "C", "B", "D")
    assert canonical_cycle(["A", "C", "B", "D"]) == ("A", "C", "B", "D")
    # a path-shape input that closes on itself drops the repeat
    assert canonical_cycle(["B", "D", "A", "C", "B"]) == ("A", "C", "B", "D")


def test_cycle_temporally_consistent_true_for_repeated_rounds():
    # two full laps around the ring, correctly ordered -> corroborated
    events = [
        ("A", "B", "2026-02-01T00:00:00Z", 10.0),
        ("B", "C", "2026-02-01T01:00:00Z", 9.0),
        ("C", "A", "2026-02-01T02:00:00Z", 8.0),
        ("A", "B", "2026-02-01T03:00:00Z", 10.0),
        ("B", "C", "2026-02-01T04:00:00Z", 9.0),
        ("C", "A", "2026-02-01T05:00:00Z", 8.0),
    ]
    assert cycle_temporally_consistent(events, ["A", "B", "C"]) is True


def test_cycle_temporally_consistent_false_for_single_lap():
    # one clean lap is weak evidence and must NOT be certified
    events = [
        ("A", "B", "2026-02-01T00:00:00Z", 10.0),
        ("B", "C", "2026-02-01T01:00:00Z", 9.0),
        ("C", "A", "2026-02-01T02:00:00Z", 8.0),
    ]
    assert cycle_temporally_consistent(events, ["A", "B", "C"]) is False


def test_cycle_temporally_consistent_false_when_order_breaks():
    # two laps, but the second is out of ring order
    events = [
        ("A", "B", "2026-02-01T00:00:00Z", 10.0),
        ("B", "C", "2026-02-01T01:00:00Z", 9.0),
        ("C", "A", "2026-02-01T02:00:00Z", 8.0),
        ("A", "B", "2026-02-01T03:00:00Z", 10.0),
        ("C", "A", "2026-02-01T04:00:00Z", 8.0),
        ("B", "C", "2026-02-01T05:00:00Z", 9.0),
    ]
    assert cycle_temporally_consistent(events, ["A", "B", "C"]) is False


def test_cycle_temporally_consistent_false_with_duplicate_leg():
    # an extra leg means the ring traffic is not a clean chronological round
    events = [
        ("A", "B", "2026-02-01T00:00:00Z", 10.0),
        ("A", "B", "2026-02-01T01:00:00Z", 10.0),
        ("B", "C", "2026-02-01T02:00:00Z", 9.0),
        ("C", "A", "2026-02-01T03:00:00Z", 8.0),
    ]
    assert cycle_temporally_consistent(events, ["A", "B", "C"]) is False


def test_cycle_temporally_consistent_false_for_off_ring_event():
    events = [
        ("A", "B", "2026-02-01T00:00:00Z", 10.0),
        ("B", "C", "2026-02-01T01:00:00Z", 9.0),
        ("C", "A", "2026-02-01T02:00:00Z", 8.0),
        ("C", "Z", "2026-02-01T03:00:00Z", 7.0),  # leaves the ring
    ]
    assert cycle_temporally_consistent(events, ["A", "B", "C"]) is False


def test_cycle_temporally_consistent_requires_multiple_events():
    assert cycle_temporally_consistent([]) is False
    assert cycle_temporally_consistent(
        [("A", "B", "2026-02-01T00:00:00Z", 1.0)], ["A", "B"]
    ) is False


# ---- detect_cycle on the controlled network --------------------------------

pytestmark_net = pytest.mark.usefixtures("test_net")


def test_detect_cycle_finds_corroborated_ring_with_canonical_members(test_net):
    env = detect_cycle(test_net, "TNC1", max_len=4)
    assert env.ok is True
    rings = [c for c in env.result["cycles"] if set(c["members"]) == {"TNC1", "TNC2", "TNC3", "TNC4"}]
    assert rings, f"planted ring not found: {env.result['cycles']}"
    ring = rings[0]
    assert ring["temporally_corroborated"] is True
    assert ring["structurally_closed"] is True
    # canonical membership: smallest id leads, no closing repeat
    assert ring["members"][0] == min(ring["members"])
    assert len(ring["members"]) == len(set(ring["members"]))


def test_detect_cycle_does_not_overclaim_structural_only_ring(test_net):
    env = detect_cycle(test_net, "TNN1", max_len=4)
    assert env.ok is True
    noise = [c for c in env.result["cycles"] if set(c["members"]) == {"TNN1", "TNN2", "TNN3"}]
    assert noise, f"structural ring not found: {env.result['cycles']}"
    assert noise[0]["temporally_corroborated"] is False
    assert env.result["temporally_corroborated_count"] == 0
    # the interpretation must state that a structural cycle is not proof of flow
    assert "structural cycle" in env.interpretation.lower()
    assert "not evidence" in env.interpretation.lower()


def test_detect_cycle_reports_no_cycle_for_acyclic_account(test_net):
    env = detect_cycle(test_net, "TNBEN", max_len=4)
    assert env.ok is True
    assert env.result["cycles"] == []
    assert env.result["account_exists"] is True


def test_detect_cycle_max_len_is_bounded(test_net):
    for bad in (1, 6, 0, "4"):
        env = detect_cycle(test_net, "TNC1", max_len=bad)
        assert env.ok is False and env.error.kind == "validation"


def test_detect_cycle_respects_tighter_max_len(test_net):
    # The 4-ring is invisible at max_len=3 but the 3-ring is visible at both.
    three = detect_cycle(test_net, "TNN1", max_len=3)
    assert any(len(c["members"]) == 3 for c in three.result["cycles"])
    four_at_three = detect_cycle(test_net, "TNC1", max_len=3)
    assert all(len(c["members"]) <= 3 for c in four_at_three.result["cycles"])


# ---- hub_scores ------------------------------------------------------------


def test_hub_scores_labels_method_and_disclaims_proof(test_net):
    env = hub_scores(test_net, window_days=365, limit=10)
    assert env.ok is True
    assert env.result["method"] == "weighted_in_degree"
    assert "weighted in-degree" in env.result["description"]
    assert "NOT PageRank" in env.interpretation
    ids = {h["account_id"] for h in env.result["hubs"]}
    assert "TNI1" in ids and "TNBEN" in ids
    # highest weighted in-degree first (TNI1 receives 30k from 3 senders)
    assert env.result["hubs"][0]["account_id"] == "TNI1"
    assert env.result["hubs"][0]["distinct_senders"] == 3


def test_hub_scores_limit_is_bounded(test_net):
    for bad in (0, 51, "10"):
        env = hub_scores(test_net, limit=bad)
        assert env.ok is False and env.error.kind == "validation"


# ---- demo-graph integration ------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return GroundTruth.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_demo_planted_cycle_is_temporally_corroborated(seeded_graph, manifest):
    ring = manifest.patterns[0].account_members
    env = detect_cycle(seeded_graph, sorted(ring)[0], max_len=4)
    assert env.ok is True
    match = next(c for c in env.result["cycles"] if set(c["members"]) == set(ring))
    assert match["temporally_corroborated"] is True
    assert env.result["temporally_corroborated_count"] >= 1


def test_demo_accidental_cycles_are_not_all_certified(seeded_graph, manifest):
    """Accidental structural cycles must not all be treated as corroborated.

    Real noise produces many structural rings. The tool must certify only
    rings whose timestamps genuinely advance around them, and only the
    planted ring may be claimed as the intended pattern.
    """
    planted = set()
    for p in manifest.patterns:
        if p.family == "cycle":
            planted.add(tuple(sorted(p.account_members)))
        elif p.family == "cycle_rapid":
            # the ring is explicit; account_members also includes the
            # feeder/exit accounts that only participate in the pass-through
            planted.add(tuple(sorted(p.ring_members)))

    total_cycles = 0
    uncorroborated = 0
    for truth in manifest.alerts:
        env = detect_cycle(seeded_graph, truth.account_id, max_len=4)
        assert env.ok is True
        total_cycles += env.result["cycle_count"]
        uncorroborated += (
            env.result["cycle_count"] - env.result["temporally_corroborated_count"]
        )
        for cycle in env.result["cycles"]:
            if cycle["temporally_corroborated"]:
                assert tuple(sorted(cycle["members"])) in planted, (
                    f"a noise cycle was wrongly corroborated: {cycle['members']}"
                )
    assert total_cycles > 0, "expected some structural cycles in the demo graph"
    assert uncorroborated > 0, (
        "every structural cycle was reported as corroborated — the temporal "
        "check is not discriminating"
    )


def test_demo_hub_metric_does_not_prove_fraud(seeded_graph, manifest):
    env = hub_scores(seeded_graph, window_days=90, limit=20)
    assert env.ok is True
    assert env.result["method"] == "weighted_in_degree"
    assert "prioritization metric" in env.interpretation
    assert "not evidence of misconduct" in env.interpretation
    # highest-ranked account is a real account id from the graph, not a label
    assert env.result["hubs"][0]["account_id"].startswith("ACC-")