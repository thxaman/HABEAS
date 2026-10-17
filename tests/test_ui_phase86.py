"""Phase 8.6 tests: evidence export, ID matching, live views, graph hover.

These pin the fixes this phase is judged on:

- verification claims are matched to evidence by *normalized* id, so a report
  claim stored as ``CASE-ALR-0001:EV-0001`` (or a re-citation ``-v2``) still
  resolves to the record ``EV-0001`` — while the original ids are preserved in
  the output. A record with no assessment stays ``not_checked`` and is never
  silently reported as verified.
- the explicit CSV download has an exact header, one consistent list
  serialization, and turns ``unverifiable`` / ``mismatch`` into nothing
  friendlier.
- each grouped edge segment carries its OWN hover text and customdata, so
  clicking either of two edges in one trace opens the matching inspector.
- live (partial) views are built strictly from emitted events: nothing is
  predicted, and a stage that has not been reached stays empty.

DB-backed tests skip cleanly when FalkorDB is unreachable.
"""
from __future__ import annotations

import csv
import io
import threading
from pathlib import Path

import pytest

from src.ui import backend
from src.ui import graph_view as gv
from src.ui import view_models as vm

REPO = Path(__file__).resolve().parents[1]
APP_SRC = (REPO / "app.py").read_text(encoding="utf-8")


# --------------------------------------------------------------- helpers


def _entry(eid: str = "EV-0001", **kw) -> dict:
    entry = {
        "evidence_id": eid, "tool": "detect_cycle", "ok": True,
        "truncated": [False], "stance": "supporting",
        "query_ids": ["detect_closed_paths_v1"], "row_count": [3],
        "derived": {}, "interpretation": "",
    }
    entry.update(kw)
    return entry


def _claim(eid: str, status: str, reason: str = "assessment recorded",
           *, checked: list[str] | None = None,
           mismatched: list[str] | None = None,
           local: str | None = None) -> dict:
    claim = {
        "evidence_id": eid, "status": status, "reason": reason,
        "checked_source_queries": list(checked or []),
        "mismatched_source_queries": list(mismatched or []),
    }
    if local is not None:
        claim["local_evidence_id"] = local
    return claim


def _event(seq: int, type_: str, **kw) -> dict:
    event = {
        "seq": seq, "at": 0.0, "type": type_, "phase": None, "stage": None,
        "tool": None, "title": "t", "detail": "", "ok": None, "meta": {},
    }
    event.update(kw)
    return event


# ====================== 1. evidence id matching for the UI ==================


class TestEvidenceIdMatching:
    def test_normalize_strips_case_prefix_and_recitation_suffix(self):
        assert vm.normalize_evidence_id("EV-0001") == "EV-0001"
        assert vm.normalize_evidence_id("CASE-ALR-0001:EV-0001") == "EV-0001"
        assert vm.normalize_evidence_id("CASE-ALR-0001:EV-0002-v3") == "EV-0002"
        assert vm.normalize_evidence_id(None) == ""
        assert vm.normalize_evidence_id("  EV-0001  ") == "EV-0001"

    def test_prefixed_claim_matches_the_unprefixed_evidence_record(self):
        rows = vm.evidence_inspector_rows(
            [_entry("EV-0001")],
            [_claim("CASE-ALR-0001:EV-0001", "reproduced_current_state")],
        )
        assert rows[0]["checked"] is True
        assert rows[0]["verification_status"] == "reproduced_current_state"
        assert rows[0]["verification_count"] == 1

    def test_local_evidence_id_is_used_when_present(self):
        rows = vm.evidence_inspector_rows(
            [_entry("EV-0001")],
            [_claim("CASE-ALR-0001:EV-0001-v2", "unverifiable", "no snapshot",
                    local="EV-0001")],
        )
        assert rows[0]["verification_status"] == "unverifiable"

    def test_record_without_a_claim_is_never_marked_checked(self):
        rows = vm.evidence_inspector_rows(
            [_entry("EV-0002")],
            [_claim("CASE-ALR-0001:EV-0001", "reproduced_current_state")],
        )
        row = rows[0]
        assert row["checked"] is False
        assert row["verification_status"] == vm.NOT_CHECKED
        assert row["verification_ids"] == []
        assert "no verification assessment" in row["verification_details"]

    def test_duplicate_assessments_collapse_to_the_most_severe(self):
        claims = [
            _claim(f"CASE-ALR-0001:EV-0002{suffix}", status)
            for suffix, status in (
                ("", "reproduced_current_state"),
                ("-v2", "verified_against_recorded_evidence"),
                ("-v3", "unverifiable"),
                ("-v4", "reproduced_current_state"),
                ("-v5", "unverifiable"),
            )
        ]
        rows = vm.evidence_inspector_rows([_entry("EV-0002")], claims)
        row = rows[0]
        assert row["checked"] is True
        assert row["verification_status"] == "unverifiable"  # worst wins
        assert row["verification_count"] == 5
        # every ORIGINAL id is preserved, not just the one that won
        assert len(row["verification_ids"]) == 5
        assert "CASE-ALR-0001:EV-0002-v5" in row["verification_ids"]
        assert "5 assessments matched" in row["verification_details"]

    def test_every_outcome_stays_distinguishable(self):
        statuses = ("reproduced_current_state",
                    "verified_against_recorded_evidence",
                    "changed_after_action", "unverifiable", "mismatch")
        rows = vm.evidence_inspector_rows(
            [_entry(f"EV-000{i}") for i in range(1, len(statuses) + 1)],
            [_claim(f"CASE-ALR-0001:EV-000{i}", s)
             for i, s in enumerate(statuses, start=1)],
        )
        seen = [r["verification_status"] for r in rows]
        assert seen == list(statuses)
        assert len(set(seen)) == len(statuses)  # none collapsed together
        assert all(r["checked"] is True for r in rows)


# ============================== 2. evidence CSV ==============================

EXPECTED_COLUMNS = (
    "evidence", "tool", "stance", "queries", "query_ids", "rows", "truncated",
    "checked", "verification_status", "verification_details",
    "checked_source_queries", "mismatched_source_queries",
)


class TestEvidenceCsv:
    def test_header_matches_the_documented_columns_exactly(self):
        assert vm.EVIDENCE_CSV_COLUMNS == EXPECTED_COLUMNS
        text = vm.evidence_csv([])
        assert text.splitlines()[0] == ",".join(EXPECTED_COLUMNS)

    def test_checked_reports_attempt_and_never_converts_a_bad_outcome(self):
        rows = vm.evidence_inspector_rows(
            [_entry("EV-0001"), _entry("EV-0002"), _entry("EV-0003")],
            [_claim("CASE-ALR-0001:EV-0001", "unverifiable", "graph moved"),
             _claim("CASE-ALR-0001:EV-0002", "mismatch", "hash differs",
                    mismatched=["detect_closed_paths_v1"])],
        )
        parsed = list(csv.DictReader(io.StringIO(vm.evidence_csv(rows))))
        assert len(parsed) == 3
        assert (parsed[0]["checked"], parsed[0]["verification_status"]) == (
            "yes", "unverifiable")
        assert (parsed[1]["checked"], parsed[1]["verification_status"]) == (
            "yes", "mismatch")
        assert parsed[1]["mismatched_source_queries"] == "detect_closed_paths_v1"
        # the record with no assessment is unchecked, and says so
        assert (parsed[2]["checked"], parsed[2]["verification_status"]) == (
            "no", "not_checked")

    def test_list_fields_share_one_serialization(self):
        entry = _entry("EV-0004", query_ids=["a_v1", "b_v1"], row_count=[2, 3])
        claim = _claim("CASE-ALR-0001:EV-0004", "reproduced_current_state",
                       checked=["a_v1", "b_v1"])
        parsed = list(csv.DictReader(io.StringIO(
            vm.evidence_csv(vm.evidence_inspector_rows([entry], [claim])))))
        assert parsed[0]["query_ids"] == "a_v1; b_v1"
        assert parsed[0]["checked_source_queries"] == "a_v1; b_v1"
        assert parsed[0]["queries"] == "2"
        assert parsed[0]["rows"] == "5"

    def test_scalar_fields_match_the_visible_table(self):
        entry = _entry("EV-0005", stance="contradicting", truncated=[True])
        row = vm.evidence_inspector_rows([entry])[0]
        parsed = list(csv.DictReader(io.StringIO(vm.evidence_csv([row]))))[0]
        assert parsed["evidence"] == row["evidence_id"]
        assert parsed["tool"] == row["tool"]
        assert parsed["stance"] == row["stance"]
        assert parsed["truncated"] == ("yes" if row["truncated_any"] else "no")
        assert parsed["truncated"] == "yes"
        assert parsed["checked"] == ("yes" if row["checked"] else "no")

    @pytest.mark.usefixtures("graph")
    def test_real_run_csv_carries_successful_and_unsuccessful_outcomes(
        self, settings, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(backend, "UI_REPORTS_DIR", str(tmp_path / "r"))
        monkeypatch.setattr(backend, "UI_TRACE_DIR", str(tmp_path / "t"))
        settings = backend.ui_settings(settings)
        snap = backend.run_scenario("ALR-0021", settings=settings, mode="fake",
                                    trace_dir=tmp_path / "t")
        rows = vm.evidence_inspector_rows(snap["evidence"],
                                          snap["report"]["claims"])
        parsed = list(csv.DictReader(io.StringIO(vm.evidence_csv(rows))))
        assert len(parsed) == len(snap["evidence"])
        assert all({"evidence", "checked", "verification_status"} <= set(p)
                   for p in parsed)
        for row in parsed:
            if row["checked"] == "yes":
                assert row["verification_status"] != vm.NOT_CHECKED
            else:
                assert row["verification_status"] == vm.NOT_CHECKED
        # a real, independently verified run yields at least one success
        assert any(r["verification_status"] in (
            "reproduced_current_state", "verified_against_recorded_evidence")
            for r in parsed)
        assert "mismatch" not in {r["verification_status"] for r in parsed}


# ======================= 3. graph hover and edge selection ==================


class TestGraphEdgeSegments:
    def _two_edge_figure(self):
        edges = [
            {"src": "ACC-A", "dst": "ACC-B", "count": 3,
             "total_amount": 100.0, "last_ts": "2026-01-02T00:00:00Z",
             "active": True},
            {"src": "ACC-B", "dst": "ACC-C", "count": 0,
             "total_amount": None, "active": True},
        ]
        fig = gv.build_figure(
            accounts=["ACC-A", "ACC-B", "ACC-C"], edges=edges,
            transactions=[{"src": "ACC-A", "dst": "ACC-B", "txn_id": "TXN-1"}],
        )
        trace = next(t for t in fig.data if t.mode and "lines" in t.mode
                     and t.name.startswith("active"))
        return trace

    def test_each_segment_carries_its_own_identity(self):
        trace = self._two_edge_figure()
        assert len(trace.customdata) == len(trace.x) == 6
        assert list(trace.customdata[0]) == ["edge", "ACC-A", "ACC-B"]
        assert list(trace.customdata[3]) == ["edge", "ACC-B", "ACC-C"]
        # the hover text follows the segment, not a copy of the first one
        assert trace.text[0].startswith("ACC-A → ACC-B")
        assert trace.text[3].startswith("ACC-B → ACC-C")
        assert trace.text[0] != trace.text[3]

    def test_clicking_either_edge_selects_the_matching_segment(self):
        trace = self._two_edge_figure()
        first = gv.selected_from_event({"selection": {"points": [
            {"customdata": list(trace.customdata[0])}]}})
        second = gv.selected_from_event({"selection": {"points": [
            {"customdata": list(trace.customdata[3])}]}})
        assert first == {"kind": "edge", "src": "ACC-A", "dst": "ACC-B"}
        assert second == {"kind": "edge", "src": "ACC-B", "dst": "ACC-C"}
        assert first != second

    def test_hover_quotes_only_real_returned_data(self):
        trace = self._two_edge_figure()
        # A->B has aggregate count/volume/timestamp and one returned event
        assert "3 transfer(s) aggregated" in trace.text[0]
        assert "100.00 total volume" in trace.text[0]
        assert "last transfer 2026-01-02T00:00:00Z" in trace.text[0]
        assert "1 transaction event(s) returned by the query" in trace.text[0]
        # B->C returned no transaction events and has no volume: nothing invented
        assert "transaction event" not in trace.text[3]
        assert "total volume" not in trace.text[3]
        assert "→" in trace.text[3]  # direction is always shown

    def test_structural_connectivity_is_never_labelled_circulation(self):
        fig = gv.build_figure(
            accounts=["ACC-A", "ACC-B"],
            edges=[{"src": "ACC-A", "dst": "ACC-B", "count": 1}],
            cycle_edges=[{"src": "ACC-A", "dst": "ACC-B", "corroborated": False}],
        )
        structural = next(t for t in fig.data if t.name == "structural cycle only")
        assert "not circulation" in structural.text[0]

    def test_live_overlay_marks_reportable_nodes_and_legs(self):
        fig = gv.build_figure(
            accounts=["ACC-A", "ACC-B", "ACC-C"],
            edges=[{"src": "ACC-A", "dst": "ACC-B", "active": True}],
            live_nodes=["ACC-C"],
            live_edges=[["ACC-B", "ACC-C"]],
        )
        names = {t.name for t in fig.data}
        assert "implicated during this run" in names


# ============================== 4. live views ===============================


class TestLiveViews:
    def test_empty_event_stream_produces_an_empty_view(self):
        view = vm.live_view_from_events([])
        assert view["evidence"] == [] and view["nodes"] == [] and view["edges"] == []
        assert view["tool_calls"] == 0
        assert view["policy"] is None and view["action"] is None
        assert view["simulation"] is None and view["verification"] is None
        assert view["event_count"] == 0

    def test_only_reached_stages_appear(self):
        events = [_event(1, "tool", tool="entity_profile", phase="started")]
        view = vm.live_view_from_events(events)
        assert view["tool_calls"] == 0  # only completed tool calls count
        assert view["evidence"] == []
        assert view["policy"] is None and view["action"] is None
        assert view["hypotheses"] == []

    def test_a_full_event_stream_is_reflected_faithfully(self):
        events = [
            _event(1, "tool", tool="entity_profile", phase="started"),
            _event(2, "tool", tool="entity_profile", phase="completed", ok=True),
            _event(3, "evidence", ok=True, tool="detect_cycle", meta={
                "evidence_id": "EV-0001", "tool": "detect_cycle",
                "stance": "supporting", "query_ids": ["detect_closed_paths_v1"],
                "row_count": [4], "truncated": [False],
                "nodes": ["ACC-A", "ACC-B", "ACC-C"],
                "edges": [["ACC-A", "ACC-B"], ["ACC-B", "ACC-C"]]}),
            _event(4, "hypotheses", meta={
                "hypotheses": [{"name": "ring", "status": "supported"}],
                "signal_families": ["circulation"]}),
            _event(5, "policy", stage="POLICY", ok=True, meta={
                "policy_version": 2, "allowed": True,
                "blocking_reasons": [], "checks": {"two_families": True},
                "distinct_provenance_families": ["circulation"]}),
            _event(6, "action", stage="SIMULATE", ok=True, meta={
                "simulation_complete": True, "downstream_accounts_before": 5}),
            _event(7, "action", stage="ACT", ok=True, meta={
                "action_status": "applied", "frozen": True,
                "action_id": "ACT-1", "sandbox": True}),
            _event(8, "verification", stage="VERIFY", ok=True, meta={
                "target": "ACC-A", "checks": {"account_is_frozen": True}}),
            _event(9, "error", title="warn", detail="truncated window"),
        ]
        view = vm.live_view_from_events(events)
        assert view["tool_calls"] == 1
        assert [e["evidence_id"] for e in view["evidence"]] == ["EV-0001"]
        assert view["evidence_count"] == 1
        assert view["nodes"] == ["ACC-A", "ACC-B", "ACC-C"]
        assert view["edges"] == [["ACC-A", "ACC-B"], ["ACC-B", "ACC-C"]]
        assert view["signal_families"] == ["circulation"]
        assert [h["name"] for h in view["hypotheses"]] == ["ring"]
        assert view["policy"]["allowed"] is True
        assert view["simulation"]["downstream_accounts_before"] == 5
        assert view["action"]["action_status"] == "applied"
        assert view["verification"]["checks"] == {"account_is_frozen": True}
        assert view["errors"] == ["truncated window"]

    def test_duplicate_evidence_events_do_not_duplicate_rows(self):
        meta = {"evidence_id": "EV-0001", "tool": "detect_cycle",
                "stance": "supporting", "query_ids": ["q_v1"],
                "row_count": [1], "truncated": [False],
                "nodes": ["ACC-A"], "edges": []}
        view = vm.live_view_from_events([
            _event(1, "evidence", meta=dict(meta)),
            _event(2, "evidence", meta=dict(meta)),
        ])
        assert [e["evidence_id"] for e in view["evidence"]] == ["EV-0001"]

    def test_partial_snapshot_reports_running_then_final(self, settings):
        settings = backend.ui_settings(settings)
        gate = threading.Event()

        def runner(alert_id, *, settings, mode, on_event, trace_dir=None):
            on_event(_event(1, "tool", tool="entity_profile", phase="started"))
            gate.wait(10)
            on_event(_event(2, "tool", tool="entity_profile",
                            phase="completed", ok=True))
            return {"alert_id": alert_id, "status": "ok", "completed": True}

        job = backend.start_investigation("ALR-9001", settings=settings,
                                          mode="fake", runner=runner)
        running = job.partial_snapshot()
        assert running["partial"] is True and running["status"] == "running"
        assert running["evidence"] == []
        assert "final" not in running  # nothing final while still running

        gate.set()
        job.wait(timeout=15)
        view = job.partial_snapshot()
        assert job.status() == "completed"
        assert view["partial"] is False and view["status"] == "completed"
        assert view["final"]["completed"] is True
        assert view["tool_calls"] == 1

    def test_partial_snapshot_status_covers_incomplete_and_failed(self, settings):
        settings = backend.ui_settings(settings)

        def incomplete(alert_id, **_kw):
            return {"alert_id": alert_id, "status": "ok", "completed": False}

        job = backend.start_investigation("ALR-9002", settings=settings,
                                          mode="fake", runner=incomplete)
        job.wait(timeout=15)
        assert job.status() == "incomplete"
        assert job.partial_snapshot()["status"] == "incomplete"

        def broken(*_a, **_k):
            raise RuntimeError("MCP died")

        job2 = backend.start_investigation("ALR-9003", settings=settings,
                                           mode="fake", runner=broken)
        job2.wait(timeout=15)
        assert job2.status() == "failed"
        snap = job2.partial_snapshot()
        assert snap["status"] == "failed" and "MCP died" in snap["error"]


class TestDuplicateJobGuard:
    def test_same_alert_while_running_returns_the_existing_job(self, settings):
        settings = backend.ui_settings(settings)
        gate = threading.Event()

        def runner(alert_id, **_kw):
            gate.wait(10)
            return {"alert_id": alert_id, "status": "ok", "completed": True}

        job = backend.start_investigation("ALR-0001", settings=settings,
                                          mode="fake", runner=runner)
        again = backend.start_investigation("ALR-0001", settings=settings,
                                            mode="fake", runner=runner,
                                            existing=job)
        assert again is job
        other = backend.start_investigation("ALR-0002", settings=settings,
                                            mode="fake", runner=runner,
                                            existing=job)
        assert other is not job and other.alert_id == "ALR-0002"
        gate.set()
        job.wait(15)
        other.wait(15)


class TestSessionStateRobustness:
    @pytest.mark.usefixtures("graph")
    def test_a_non_job_value_in_session_state_never_crashes_the_app(self):
        """Regression: run-state helpers must ignore a value they did not write.

        The Phase 8.5 AppTest injects a string sentinel as the session job;
        the app must degrade to "no live job" instead of raising on it.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=180)
        at.run()
        assert not at.exception
        at.session_state["job"] = "sentinel-job"
        at.run()
        assert not at.exception
        assert len(at.radio) >= 1  # the script ran past the run-state plumbing


# ============================ 5. HTML escaping ==============================


class TestEscaping:
    def test_escape_html_neutralises_markup(self):
        assert vm.escape_html('<b>"x" & </b>') == (
            "&lt;b&gt;&quot;x&quot; &amp; &lt;/b&gt;")
        assert vm.escape_html(None) == ""
        assert vm.escape_html(3) == "3"

    def test_app_escapes_the_dynamic_text_it_injects_as_html(self):
        assert 'vm.escape_html(event.get("title"))' in APP_SRC
        assert 'vm.escape_html(event.get("detail")' in APP_SRC
        # the hypothesis name goes through chip(), which escapes its label
        assert 'chip(hypothesis.get("name")' in APP_SRC
        assert "vm.escape_html(INTRO)" in APP_SRC
        assert "vm.escape_html(label)" in APP_SRC


# ========================== 6. workflow progress ===========================


class TestWorkflowProgress:
    def test_steps_are_labelled_and_ordered(self):
        steps = vm.workflow_progress([])
        assert [s["label"] for s in steps] == [
            "Alert", "Network investigation", "Evidence and hypotheses",
            "Policy decision", "Impact simulation", "Action outcome",
            "Independent verification",
        ]
        assert [s["key"] for s in steps] == [k for k, _ in vm.WORKFLOW_STEPS]

    def test_progress_advances_only_on_real_events(self):
        assert vm.workflow_progress([])[0]["state"] == "current"
        assert all(s["state"] == "pending" for s in vm.workflow_progress([])[1:])

        selected = vm.workflow_progress([], alert_selected=True)
        assert selected[0]["state"] == "done"
        assert selected[1]["state"] == "current"

        # a run that has only produced evidence is at "policy", not further
        mid = vm.workflow_progress([_event(1, "evidence")])
        states = {s["key"]: s["state"] for s in mid}
        assert states["alert"] == "done" and states["investigation"] == "done"
        assert states["evidence"] == "done"
        assert states["policy"] == "current"
        assert states["simulation"] == "pending"
        assert states["verification"] == "pending"

    def test_a_finished_run_marks_every_step_done(self):
        events = [
            _event(1, "tool", phase="completed"), _event(2, "evidence"),
            _event(3, "hypotheses"), _event(4, "policy"),
            _event(5, "action", stage="SIMULATE"),
            _event(6, "action", stage="ACT"),
            _event(7, "verification"),
        ]
        assert all(s["state"] == "done" for s in vm.workflow_progress(events))


# ======================= 7. app wiring for this phase ======================


class TestAppWiring:
    def test_explicit_csv_download_exists(self):
        assert "Download evidence CSV" in APP_SRC
        assert "vm.evidence_csv(rows)" in APP_SRC
        assert "text/csv" in APP_SRC

    def test_live_views_and_overlay_are_wired(self):
        assert "partial_snapshot()" in APP_SRC
        assert "workflow_progress(" in APP_SRC
        assert 'live_nodes=view.get("nodes") if partial else None' in APP_SRC
        assert "evidence_for_entity" in APP_SRC and "evidence_for_edge" in APP_SRC

    def test_rerun_cannot_launch_a_duplicate_job(self):
        assert "existing=_job()" in APP_SRC
        # only a real InvestigationJob is treated as a live job, so a rerun can
        # neither reuse nor crash on any other value left in session state
        assert "isinstance(job, backend.InvestigationJob)" in APP_SRC

    def test_technical_details_are_progressively_disclosed(self):
        assert "Technical details" in APP_SRC
        assert "st.expander" in APP_SRC