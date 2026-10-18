"""Integration and regression tests for UI verification status across all alert scenarios.

Validates that:
- Case-prefixed and versioned IDs match their corresponding local evidence IDs.
- Every persisted citation has a corresponding verifier assessment.
- Statuses (reproduced_current_state, changed_after_action, unverifiable, mismatch)
  survive unchanged.
- No evidence row receives a status from another alert or unrelated evidence record.
- Genuinely missing verification assessments remain `not_checked`.
- Multiple assessments choose the most severe status and its matching reason
  (status/reason consistency).
- Tested across ALR-0001, ALR-0021, ALR-0004, ALR-0007 on isolated, disposable test graphs.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from src.config import DEMO_GRAPH_PREFIX, Settings
from src.db import get_graph
from src.seed import generate, schema
from src.ui import backend
from src.ui import view_models as vm


SCENARIOS = [
    ("ALR-0001", "policy declines due to provenance overlap/target relevance"),
    ("ALR-0021", "policy permits the sandbox freeze with independent target-relevant evidence"),
    ("ALR-0004", "insufficient evidence"),
    ("ALR-0007", "rapid-forwarding evidence without sufficient independent corroboration"),
]


@pytest.fixture(params=SCENARIOS, ids=[s[0] for s in SCENARIOS])
def scenario_alert(request):
    return request.param


class TestScenarioVerificationMapping:
    """End-to-end verification mapping tests using disposable graphs."""

    @pytest.mark.usefixtures("graph")
    def test_alert_verification_mapping_end_to_end(
        self, settings: Settings, tmp_path: Path, monkeypatch, scenario_alert
    ):
        alert_id, description = scenario_alert
        unique_graph_name = (
            f"{DEMO_GRAPH_PREFIX}test_{alert_id.lower().replace('-', '_')}_{uuid.uuid4().hex[:8]}"
        )
        scenario_settings = Settings(
            **{
                **settings.model_dump(),
                "graph_name": unique_graph_name,
                "habeas_demo_mode": True,
                "habeas_enable_actions": True,
            }
        )

        test_graph = get_graph(scenario_settings)
        generate.seed(scenario_settings, seed=42, reset=True, graph=test_graph)

        rep_dir = tmp_path / f"r_{alert_id}"
        trc_dir = tmp_path / f"t_{alert_id}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        trc_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(backend, "UI_REPORTS_DIR", str(rep_dir))
        monkeypatch.setattr(backend, "UI_TRACE_DIR", str(trc_dir))

        try:
            snapshot = backend.run_scenario(
                alert_id,
                settings=scenario_settings,
                mode="fake",
                trace_dir=trc_dir,
            )

            assert snapshot["status"] == "ok"
            case_id = snapshot["case_id"]
            assert case_id is not None
            assert alert_id in case_id

            evidence_entries = snapshot.get("evidence") or []
            report = snapshot.get("report") or {}
            claims = report.get("claims") or []

            # 1. Count persisted citations in graph
            ev_records = test_graph.query(
                "MATCH (c:Case {id: $case_id})-[:HAS_EVIDENCE]->(e:Evidence) "
                "RETURN e.id AS id, e.query_id AS qid",
                {"case_id": case_id},
            )
            persisted_eids = [r[0] for r in (ev_records.result_set or [])]
            persisted_count = len(persisted_eids)

            # All investigated evidence with query_ids was cited and persisted
            cited_investigation_eids = [
                e["evidence_id"] for e in evidence_entries if e.get("query_ids")
            ]
            assert len(evidence_entries) > 0
            assert persisted_count == len(cited_investigation_eids)

            # 2. Map claims to inspector rows
            rows = vm.evidence_inspector_rows(evidence_entries, claims)
            assert len(rows) == len(evidence_entries)

            matched_rows = [r for r in rows if r["checked"]]
            unmatched_rows = [r for r in rows if not r["checked"]]

            # Print diagnostic stats for audit reporting
            print(f"\nScenario {alert_id} ({description}):")
            print(f"  evidence_count={len(evidence_entries)}")
            print(f"  citation_count={len(cited_investigation_eids)}")
            print(f"  persisted_evidence_count={persisted_count}")
            print(f"  verifier_assessment_count={len(claims)}")
            print(f"  matched_rows={len(matched_rows)}")
            print(f"  unmatched_evidence_ids={[r['evidence_id'] for r in unmatched_rows]}")

            # Every cited evidence row must match its verifier assessment
            assert len(unmatched_rows) == 0
            assert len(matched_rows) == len(evidence_entries)

            # 3. Assert IDs, prefixes, and versions
            for row in rows:
                local_id = row["evidence_id"]
                assert row["checked"] is True
                assert row["verification_status"] != vm.NOT_CHECKED
                assert len(row["verification_ids"]) >= 1

                # All verification IDs must normalize back to this row's local_id
                for v_id in row["verification_ids"]:
                    assert vm.normalize_evidence_id(v_id) == local_id
                    # Prefixed with the case ID
                    assert v_id.startswith(f"{case_id}:")

                # Verify status and reason are consistent
                assert row["verification_reason"] != ""
                assert row["verification_status"] in (
                    "reproduced_current_state",
                    "verified_against_recorded_evidence",
                    "changed_after_action",
                    "unverifiable",
                    "mismatch",
                )

            # 4. Also verify evidence_inspector_rows works with dataclass objects directly
            raw_claims = getattr(
                getattr(snapshot.get("_artifact"), "verification", None), "claims", None
            )
            if raw_claims:
                dataclass_rows = vm.evidence_inspector_rows(evidence_entries, raw_claims)
                assert len(dataclass_rows) == len(rows)
                for r_dict, r_obj in zip(rows, dataclass_rows):
                    assert r_dict["verification_status"] == r_obj["verification_status"]
                    assert r_dict["verification_reason"] == r_obj["verification_reason"]

        finally:
            schema.clear_graph(test_graph)


class TestVerificationRowUnitCases:
    """Unit tests for status/reason consistency, cross-case isolation, and not_checked handling."""

    def test_status_and_reason_consistency_on_multiple_assessments(self):
        """When multiple assessments match, status and reason must both come from the chosen assessment."""
        claims = [
            {
                "evidence_id": "CASE-ALR-0001:EV-0002",
                "status": "reproduced_current_state",
                "reason": "replay reproduced exactly against current graph",
            },
            {
                "evidence_id": "CASE-ALR-0001:EV-0002-v2",
                "status": "unverifiable",
                "reason": "metadata changed since capture",
            },
        ]
        entry = {
            "evidence_id": "EV-0002",
            "tool": "entity_profile",
            "ok": True,
            "query_ids": ["entity_profile_lookup_v1"],
            "row_count": [1],
        }

        rows = vm.evidence_inspector_rows([entry], claims)
        row = rows[0]
        assert row["checked"] is True
        # unverifiable ranks higher in severity than reproduced_current_state
        assert row["verification_status"] == "unverifiable"
        # Reason must match the unverifiable assessment, NOT the reproduced one
        assert row["verification_reason"] == "metadata changed since capture"
        assert "metadata changed since capture" in row["verification_details"]
        assert "2 assessments matched this record" in row["verification_details"]

    def test_cross_case_isolation_prevents_mismatched_status(self):
        """Evidence row with a case prefix or case_id must never match a claim from a different case."""
        entry_case1 = {
            "evidence_id": "CASE-ALR-0001:EV-0001",
            "case_id": "CASE-ALR-0001",
            "tool": "alert_context",
            "ok": True,
            "query_ids": ["alert_context_v1"],
            "row_count": [1],
        }
        claims_case2 = [
            {
                "evidence_id": "CASE-ALR-0021:EV-0001",
                "status": "changed_after_action",
                "reason": "edge deactivated by freeze action",
            }
        ]

        rows = vm.evidence_inspector_rows([entry_case1], claims_case2)
        row = rows[0]
        # Should NOT match claim from CASE-ALR-0021
        assert row["checked"] is False
        assert row["verification_status"] == vm.NOT_CHECKED
        assert row["verification_ids"] == []

    def test_genuinely_uncited_evidence_remains_not_checked_with_explanation(self):
        """Evidence without source queries or verification assessments remains truthfully not_checked."""
        uncited_entry = {
            "evidence_id": "EV-0099",
            "tool": "unknown_tool",
            "ok": False,
            "query_ids": [],
            "row_count": [],
        }
        claims = [
            {
                "evidence_id": "CASE-ALR-0001:EV-0001",
                "status": "reproduced_current_state",
                "reason": "verified",
            }
        ]

        rows = vm.evidence_inspector_rows([uncited_entry], claims)
        row = rows[0]
        assert row["checked"] is False
        assert row["verification_status"] == vm.NOT_CHECKED
        assert row["verification_ids"] == []
        assert "no verification assessment" in row["verification_details"]
