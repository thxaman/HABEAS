"""Phase 6 signal-provenance tests (PLAN.md §11), updated for policy v2.

Different signal NAMES are not independent evidence. These tests pin the
classifier that traces signal families back to the underlying transaction
ids, so a report can never describe correlated families as independent
corroboration. The classifier is pure; since the corrective Phase 5 pass it
also feeds the action-eligibility policy (v2): overlapping or
target-irrelevant families cannot authorize an action. The
provenance-aware eligibility rules themselves are covered by
``tests/test_policy_provenance.py``.
"""
from __future__ import annotations

from src.agent import analysis
from src.agent.state import CaseState, EvidenceRecord

ACC = "ACC-TEST"


def _trace_record(evidence_id: str, legs: list[dict]) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        tool="trace_funds",
        stage="TEST",
        ok=True,
        derived={"account_id": ACC, "direction": "out", "leg_events": legs},
    )


def _cycle_record(evidence_id: str, txns: list[str]) -> EvidenceRecord:
    events = [
        {"src": "A", "dst": "B", "txn_id": txn, "ts": f"2026-01-01T0{i}:00:00Z",
         "amount": 100.0}
        for i, txn in enumerate(txns)
    ]
    return EvidenceRecord(
        evidence_id=evidence_id,
        tool="detect_cycle",
        stage="TEST",
        ok=True,
        derived={
            "corroborated_count": 1,
            "cycles": [{
                "members": ["A", "B"],
                "temporally_corroborated": True,
                "event_count": len(events),
                "legs": [{"src": "A", "dst": "B"}],
                "events": events,
            }],
        },
    )


def _state(*records: EvidenceRecord) -> CaseState:
    return CaseState(
        alert_id="ALRT-TEST",
        primary_entity_ids=[ACC],
        evidence=list(records),
    )


def _leg(src: str, dst: str, txn: str, minutes: int) -> dict:
    return {
        "src": src, "dst": dst, "txn_id": txn,
        "ts": f"2026-01-01T{minutes // 60:02d}:{minutes % 60:02d}:00Z",
        "amount": 100.0,
    }


def test_shared_observations_when_rapid_legs_are_a_subset_of_the_cycle():
    """A rapid match on ring legs shares the cycle's own transactions."""
    # In at txn IN, out at txn OUT: a rapid match whose observations are the
    # cycle's transactions.
    trace = _trace_record("EV-1", [
        _leg("Z", ACC, "IN", 10),
        _leg(ACC, "Y", "OUT", 20),
    ])
    cycle = _cycle_record("EV-2", ["IN", "OUT", "R3", "R4"])
    state = _state(trace, cycle)

    provenance = analysis.signal_provenance(state)
    assert sorted(provenance["families_detected"]) == ["circulation", "rapid_forwarding"]
    pair = provenance["pairs"][0]
    assert pair["classification"] == "shared_observations"
    assert set(pair["shared_txn_ids"]) == {"IN", "OUT"}
    # circulation observed more than the rapid match: unique ids are reported
    assert set(pair["unique_txn_ids"]["circulation"]) == {"R3", "R4"}
    assert pair["unique_txn_ids"]["rapid_forwarding"] == []
    assert provenance["overlapping_pairs"] == [pair["pair"]]


def test_partially_shared_observations():
    """Some shared transactions, some unique to each family."""
    trace = _trace_record("EV-1", [
        _leg("Z", ACC, "SHARED", 10),
        _leg(ACC, "Y", "RAPID-ONLY", 20),
    ])
    cycle = _cycle_record("EV-2", ["SHARED", "CYCLE-ONLY"])
    pair = analysis.signal_provenance(_state(trace, cycle))["pairs"][0]
    assert pair["classification"] == "partially_shared"
    assert pair["shared_txn_ids"] == ["SHARED"]
    assert pair["unique_txn_ids"]["circulation"] == ["CYCLE-ONLY"]
    assert pair["unique_txn_ids"]["rapid_forwarding"] == ["RAPID-ONLY"]


def test_distinct_observations_when_no_transaction_is_shared():
    trace = _trace_record("EV-1", [
        _leg("Z", ACC, "R1", 10),
        _leg(ACC, "Y", "R2", 20),
    ])
    cycle = _cycle_record("EV-2", ["C1", "C2", "C3", "C4"])
    pair = analysis.signal_provenance(_state(trace, cycle))["pairs"][0]
    assert pair["classification"] == "distinct_observations"
    assert pair["shared_count"] == 0
    assert len(pair["unique_txn_ids"]["circulation"]) == 4
    assert len(pair["unique_txn_ids"]["rapid_forwarding"]) == 2


def test_aggregate_only_family_is_provenance_unavailable():
    """fan_in_out returns rollups, not transaction ids: independence unknown."""
    fan = EvidenceRecord(
        evidence_id="EV-1", tool="fan_in_out", stage="TEST", ok=True,
        derived={
            "distinct_senders": 3, "top_receiver_share": 1.0,
            "out_in_ratio": 0.9, "out_amount": 900.0,
            "account_id": ACC, "direction": "in",
        },
    )
    cycle = _cycle_record("EV-2", ["C1", "C2", "C3", "C4"])
    provenance = analysis.signal_provenance(_state(fan, cycle))
    assert provenance["families_detected"] == ["circulation", "fan_in_out"]
    pair = provenance["pairs"][0]
    assert pair["classification"] == "provenance_unavailable"
    assert provenance["unavailable_pairs"] == [pair["pair"]]
    assert provenance["families"]["fan_in_out"]["available"] is False
    assert provenance["families"]["fan_in_out"]["txn_ids"] is None
    assert "cannot be established" in provenance["families"]["fan_in_out"]["note"]


def test_corroborated_cycle_without_retained_events_is_unavailable_not_shared():
    """Legacy records without retained events must not claim shared provenance."""
    cycle = EvidenceRecord(
        evidence_id="EV-1", tool="detect_cycle", stage="TEST", ok=True,
        derived={"cycles": [{
            "members": ["A", "B"], "temporally_corroborated": True,
            "event_count": 4,  # events were not retained (legacy citation)
        }]},
    )
    trace = _trace_record("EV-2", [_leg("Z", ACC, "IN", 10), _leg(ACC, "Y", "OUT", 20)])
    # circulation is available only if events with txn ids were retained
    circulation = analysis.circulation_observations(_state(cycle, trace))
    assert circulation["available"] is False or circulation["txn_ids"] == []
    availability = analysis.classify_observation_overlap(
        circulation, analysis.rapid_observations(_state(cycle, trace))
    )
    assert availability == "provenance_unavailable"


def test_rapid_observations_trace_to_individual_transactions():
    trace = _trace_record("EV-1", [
        _leg("Z", ACC, "IN", 10),
        _leg(ACC, "Y", "OUT", 20),
        _leg(ACC, "Y2", "OUT2", 30),
    ])
    rapid = analysis.rapid_observations(_state(trace))
    assert rapid["available"] is True
    # First outbound that satisfies the ratio: IN -> OUT.
    assert rapid["by_account"] == {ACC: ["IN", "OUT"]}
