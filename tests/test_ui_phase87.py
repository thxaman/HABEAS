"""Phase 8.7 tests: FalkorDB-first product polish.

These pin the phase's judged behaviour:

- the product identity names FalkorDB as the graph engine, states the
  synthetic/sandbox scope, and describes the *implemented* data paths — the
  investigation agent over MCP and the trusted report/verifier path that reads
  the graph directly;
- the header connection state is derived from a real approved ``alert_inbox``
  read with three distinguishable states (connected / unavailable /
  not checked), an age and a retry — never a constant "connected" chip;
- the guided demonstration uses the real scenario metadata and starts a
  scenario's real investigation only after the user presses its button; the
  result chips are copied from the recorded policy/action envelope;
- an evidence record can be highlighted in the graph strictly from the
  accounts/legs THAT record reported; an unmapped record is called out instead
  of being given an invented link; edge details carry timestamps and
  transaction ids when the queries returned them;
- offline evaluation metrics are shown only when the artifact is present AND
  identified by seed, mode and dataset, and never as the current run's result.

DB-backed tests skip cleanly when FalkorDB is unreachable.
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest

from src.ui import backend
from src.ui import graph_view as gv
from src.ui import view_models as vm

REPO = Path(__file__).resolve().parents[1]
APP_SRC = (REPO / "app.py").read_text(encoding="utf-8")


# --------------------------------------------------------------- helpers


def _fake_inbox() -> dict:
    alerts = []
    for index, scenario in enumerate(backend.list_scenarios()):
        alerts.append({
            "alert_id": scenario["alert_id"],
            "reason": scenario["title"],
            "concerned_account_id": f"ACC-{index:04d}",
            "triage_status": "open", "case_id": None, "case_verdict": None,
            "action_status": None, "trigger_transaction": None,
        })
    return {"ok": True, "alerts": alerts, "interpretation": None,
            "query_ids": ["alert_inbox_v1"], "error": None}


def _patched_app(monkeypatch, *, status_state: str = "connected"):
    """AppTest-safe backend patches: no MCP, deterministic connection state."""
    moment = time.time()
    status = {
        "state": status_state, "checked_at": moment, "tool": "alert_inbox",
        "query_ids": ["alert_inbox_v1"] if status_state == "connected" else [],
        "duration_ms": 4.2,
        "error": None if status_state == "connected" else "offline test",
    }
    monkeypatch.setattr(backend, "check_graph_status",
                        lambda settings=None: dict(status))
    monkeypatch.setattr(backend, "load_alert_inbox",
                        lambda settings=None: _fake_inbox())
    monkeypatch.setattr(backend, "load_neighborhood",
                        lambda *a, **k: {"ok": False, "error": "offline test"})
    return status


class _StubJob:
    """A recording stand-in for InvestigationJob (not the real class)."""

    def __init__(self, alert_id: str) -> None:
        self.alert_id = alert_id
        self.done = True
        self.error = None
        self.snapshot = {
            "alert_id": alert_id, "status": "ok", "completed": True,
            "errors": [], "evidence": [], "hypotheses": [], "tool_calls": 0,
            "tool_budget": 12, "signal_families": [], "primary_entities": [],
            "report": None, "action": None, "simulation": None,
            "post_action": None, "truncated_evidence": [],
        }

    @property
    def running(self) -> bool:
        return False


# ==================== 1. product identity and architecture ==================


class TestProductIdentity:
    def test_header_names_falkordb_and_the_synthetic_sandbox_scope(self):
        assert "built on FalkorDB" in APP_SRC
        assert "connected-data reasoning" in APP_SRC
        assert "synthetic data only" in APP_SRC
        assert "not an action against a real" in APP_SRC
        # the brand copy is rendered escaped like every other HTML injection
        assert "vm.escape_html(BRAND)" in APP_SRC
        assert "vm.escape_html(DATA_NOTE)" in APP_SRC

    def test_architecture_strip_matches_the_implemented_path(self):
        for stage in ("FalkorDB graph", "MCP tools",
                      "HABEAS investigation agent", "Policy v2 gate",
                      "Sandbox simulation / action",
                      "Independent evidence verifier"):
            assert stage in APP_SRC, stage
        # the direct (non-MCP) audit path is represented, not forced into the
        # linear agent flow
        assert "Trusted audit path" in APP_SRC
        assert "read FalkorDB directly" in APP_SRC or (
            "reads FalkorDB directly" in APP_SRC)
        assert "not over MCP" in APP_SRC or "not part of the" in APP_SRC
        assert "no partnership or endorsement is implied" in APP_SRC

    def test_architecture_claims_name_real_catalog_queries(self):
        from src.tools import query_catalog as qc

        for query_id in ("detect_closed_paths_v1", "trace_paths_out_v1",
                         "shared_identifiers_v1", "fan_in_senders_v1",
                         "fan_out_receivers_v1"):
            assert query_id in APP_SRC, query_id
            assert qc.get_template(query_id), query_id  # really in the catalog


# ======================== 2. connection status ==============================


class TestConnectionStatus:
    def test_three_states_have_distinct_labels_and_tones(self):
        assert vm.connection_chip({"state": "connected"}) == (
            "FalkorDB connected", "ok")
        assert vm.connection_chip({"state": "unavailable"}) == (
            "FalkorDB unavailable", "error")
        assert vm.connection_chip({"state": "not_checked"}) == (
            "FalkorDB not checked", "muted")
        assert vm.connection_chip(None) == ("FalkorDB not checked", "muted")
        assert vm.connection_chip({}) == ("FalkorDB not checked", "muted")

    def test_age_is_explicit_and_computed_from_the_check_time(self):
        base = 1_000_000.0
        assert vm.connection_age({"checked_at": base}, now=base) == "0s ago"
        assert vm.connection_age({"checked_at": base}, now=base + 5) == "5s ago"
        assert vm.connection_age({"checked_at": base}, now=base + 65) == "1m 5s ago"
        assert vm.connection_age({"checked_at": base}, now=base + 120) == "2m ago"
        assert vm.connection_age({"checked_at": base}, now=base + 3700) == "1h 1m ago"
        assert vm.connection_age({"checked_at": base}, now=base + 90000) == "1d 1h ago"
        assert vm.connection_age(None) is None
        assert vm.connection_age({"checked_at": "not-a-time"}) is None

    def test_stale_after_five_minutes_but_never_hidden(self):
        base = 1_000_000.0
        assert vm.connection_is_stale({"checked_at": base}, now=base + 100) is False
        assert vm.connection_is_stale({"checked_at": base}, now=base + 301) is True
        assert vm.connection_is_stale({"checked_at": None}) is False

    def test_check_uses_one_bounded_approved_read(self, monkeypatch):
        seen: dict = {}

        class _Result:
            ok = True
            error = None
            interpretation = "bounded inbox read"
            envelope = {
                "query_evidence": [{"query_id": "alert_inbox_v1"}],
                "result": {"alerts": []},
            }

        def fake(settings, tool, arguments):
            seen["tool"] = tool
            seen["arguments"] = dict(arguments)
            return _Result()

        monkeypatch.setattr(backend, "_mcp_call", fake)
        status = backend.check_graph_status()
        assert seen == {"tool": "alert_inbox", "arguments": {"limit": 1}}
        assert status["state"] == "connected"
        assert status["tool"] == "alert_inbox"
        assert status["query_ids"] == ["alert_inbox_v1"]
        assert status["error"] is None and status["checked_at"]

    def test_check_reports_failure_without_raising(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("no db")

        monkeypatch.setattr(backend, "_mcp_call", boom)
        status = backend.check_graph_status()
        assert status["state"] == "unavailable"
        assert "no db" in status["error"]
        assert status["checked_at"]

        class _BadResult:
            ok = False
            error = "graph down"
            envelope: dict = {"query_evidence": []}

        monkeypatch.setattr(backend, "_mcp_call",
                            lambda *a, **k: _BadResult())
        status = backend.check_graph_status()
        assert status["state"] == "unavailable"
        assert "graph down" in status["error"]

    def test_check_accepts_no_graph_name_or_cypher(self):
        params = inspect.signature(backend.check_graph_status).parameters
        assert "graph_name" not in params and "cypher" not in params

    def test_app_renders_the_recorded_state_and_offers_a_retry(self):
        assert "check_graph_status" in APP_SRC
        assert "Re-check FalkorDB connection" in APP_SRC
        assert "st.session_state.graph_status = None" in APP_SRC
        assert "connection_age" in APP_SRC and "connection_is_stale" in APP_SRC
        # regression: the old constant chip is gone
        assert 'chip("FalkorDB", "ok")' not in APP_SRC


# ========================= 3. guided demonstration ==========================


class TestGuidedDemo:
    def test_scenarios_come_from_the_real_demo_metadata(self):
        scenarios = backend.list_scenarios()
        assert [s["key"] for s in scenarios] == ["A", "B", "C"]
        assert all(set(s) == {"key", "alert_id", "title", "stake"}
                   for s in scenarios)
        assert "backend.list_scenarios()" in APP_SRC

    def test_result_summary_is_copied_from_the_recorded_envelope(self):
        declined = {
            "alert_id": "SOME-ALERT", "completed": True,
            "canonical_verdict": "suspicion_supported_for_review",
            "action": {
                "action_status": "declined", "allowed": False,
                "blocking_reasons": ["the cited families share one story"],
                "distinct_provenance_families": ["circulation"],
            },
        }
        summary = vm.guided_result_summary(declined, status="completed")
        assert summary["action_status"] == "declined"
        assert summary["allowed"] is False
        assert summary["blocking_reasons"] == [
            "the cited families share one story"]
        assert summary["verdict"] == "suspicion_supported_for_review"

        applied = {
            "alert_id": "OTHER", "completed": True,
            "action": {
                "action_status": "applied", "allowed": True,
                "distinct_provenance_families": ["circulation",
                                                 "rapid_forwarding"],
            },
        }
        summary = vm.guided_result_summary(applied, status="completed")
        assert summary["allowed"] is True
        assert summary["qualifying_families"] == ["circulation",
                                                  "rapid_forwarding"]

        live = {"alert_id": "LIVE", "policy": {
            "allowed": False, "blocking_reasons": ["not enough"],
            "distinct_provenance_families": ["fan_in_out"]}}
        summary = vm.guided_result_summary(live, status="running")
        assert summary["allowed"] is False
        assert summary["blocking_reasons"] == ["not enough"]

        empty = vm.guided_result_summary(None, status="incomplete")
        assert empty["allowed"] is None and empty["blocking_reasons"] == []

    def test_app_wiring_never_auto_runs_and_never_reseeds(self):
        assert "pending_run" in APP_SRC
        assert "guided-run-" in APP_SRC
        assert 'disabled=status == "running"' in APP_SRC
        assert "auto-run or reseeds" in APP_SRC or (
            "auto-runs or reseeds" in APP_SRC)
        assert "backend.list_scenarios()" in APP_SRC

    def test_plain_load_never_starts_a_run_and_click_starts_the_alert(
        self, monkeypatch
    ):
        from streamlit.testing.v1 import AppTest

        _patched_app(monkeypatch, status_state="connected")
        calls: list[str] = []

        def fake_start(alert_id, **kwargs):
            calls.append(alert_id)
            assert kwargs.get("mode") == "fake"
            return _StubJob(alert_id)

        monkeypatch.setattr(backend, "start_investigation", fake_start)
        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=60)
        at.run()
        assert not at.exception
        assert calls == []  # a page load never starts an investigation

        joined = " ".join(str(m.value) for m in at.markdown)
        assert "FalkorDB connected" in joined
        titles = [s["title"] for s in backend.list_scenarios()]
        for title in titles:
            assert title in joined, title  # real scenario metadata shown
        assert all(f"guided-run-{key}" in {b.key for b in at.button}
                   for key in ("A", "B", "C"))

        scenario_b = backend.list_scenarios()[1]
        button = next(b for b in at.button if b.key == "guided-run-B")
        button.click().run()
        assert not at.exception
        assert calls == [scenario_b["alert_id"]]
        assert at.session_state["selected_alert"] == scenario_b["alert_id"]
        assert at.session_state["job"] is not None
        assert at.session_state["pending_run"] is None

    def test_failed_status_renders_without_claiming_a_result(self, monkeypatch):
        from streamlit.testing.v1 import AppTest

        _patched_app(monkeypatch, status_state="unavailable")
        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=60)
        at.run()
        assert not at.exception
        joined = " ".join(str(m.value) for m in at.markdown)
        assert "FalkorDB unavailable" in joined
        assert "FalkorDB connected" not in joined


# ==================== 4. evidence <-> graph interaction =====================


class TestEvidenceGraphMapping:
    def test_maps_cycles_leg_events_derived_account_and_entities(self):
        entry = {
            "evidence_id": "EV-1",
            "entity_ids": ["ACC-X"],
            "derived": {
                "account_id": "ACC-Y",
                "cycles": [{
                    "members": ["ACC-A"],
                    "legs": [{"src": "ACC-A", "dst": "ACC-B"},
                             {"src": "ACC-A", "dst": "ACC-B"}],
                    "events": [{"src": "ACC-B", "dst": "ACC-C",
                                "txn_id": "T-1", "ts": "2026-01-01"}],
                }],
                "leg_events": [{"src": "ACC-C", "dst": "ACC-A"}],
            },
        }
        mapping = vm.evidence_graph_elements(entry)
        assert mapping["mapped"] is True
        assert mapping["nodes"] == ["ACC-A", "ACC-B", "ACC-C", "ACC-Y", "ACC-X"]
        assert mapping["edges"] == [["ACC-A", "ACC-B"], ["ACC-B", "ACC-C"],
                                    ["ACC-C", "ACC-A"]]

    def test_live_records_keep_their_own_nodes_and_edges(self):
        events = [{
            "seq": 1, "at": 0.0, "type": "evidence", "phase": "completed",
            "stage": None, "tool": "detect_cycle", "title": "t", "detail": "",
            "ok": True,
            "meta": {
                "evidence_id": "EV-1", "tool": "detect_cycle",
                "stance": "supporting", "query_ids": ["detect_closed_paths_v1"],
                "row_count": [1], "truncated": [False],
                "nodes": ["ACC-A", "ACC-B"], "edges": [["ACC-A", "ACC-B"]],
            },
        }]
        view = vm.live_view_from_events(events)
        entry = view["evidence"][0]
        assert entry["nodes"] == ["ACC-A", "ACC-B"]
        assert entry["edges"] == [["ACC-A", "ACC-B"]]
        mapping = vm.evidence_graph_elements(entry)
        assert mapping["edges"] == [["ACC-A", "ACC-B"]]

    def test_unmapped_record_is_reported_not_invented(self):
        mapping = vm.evidence_graph_elements(
            {"evidence_id": "EV-9", "tool": "policy", "derived": {},
             "entity_ids": []})
        assert mapping == {"nodes": [], "edges": [], "mapped": False}
        assert mapping["mapped"] is False
        assert "does not map" in APP_SRC
        assert "nothing is highlighted rather" in APP_SRC

    def test_focus_overlay_maps_clicks_back_to_the_real_edge(self):
        fig = gv.build_figure(
            accounts=["ACC-A", "ACC-B"],
            edges=[{"src": "ACC-A", "dst": "ACC-B", "count": 1}],
            focus_edges=[["ACC-A", "ACC-B"]],
            focus_nodes=["ACC-B"],
        )
        focus = [t for t in fig.data if t.name == "highlighted evidence"]
        assert focus, "the focus overlay trace must exist"
        line = next(t for t in focus if t.mode and "lines" in t.mode)
        assert len([x for x in line.x if x is not None]) // 2 == 1
        identity = list(line.customdata[0])
        assert identity == ["edge", "ACC-A", "ACC-B"]
        selected = gv.selected_from_event(
            {"selection": {"points": [{"customdata": identity}]}})
        assert selected == {"kind": "edge", "src": "ACC-A", "dst": "ACC-B"}

    def test_focus_never_adds_elements_outside_the_drawn_graph(self):
        fig = gv.build_figure(
            accounts=["ACC-A"], edges=[],
            focus_nodes=["ACC-UNKNOWN"], focus_edges=[["ACC-A", "ACC-MISSING"]],
        )
        assert not any(t.name == "highlighted evidence" for t in fig.data)
        base = gv.build_figure(accounts=["ACC-A"], edges=[])
        assert not any(t.name == "highlighted evidence" for t in base.data)

    def test_edge_details_expose_timestamps_and_transaction_ids(self):
        details = gv.edge_details(
            "ACC-A", "ACC-B",
            edges=[{"src": "ACC-A", "dst": "ACC-B", "count": 2}],
            transactions=[
                {"src": "ACC-A", "dst": "ACC-B", "txn_id": "T-2",
                 "ts": "2026-02-02T00:00:00Z"},
                {"src": "ACC-A", "dst": "ACC-B", "txn_id": "T-1",
                 "ts": "2026-01-01T00:00:00Z"},
                {"src": "ACC-A", "dst": "ACC-B", "txn_id": "T-1"},
            ],
        )
        assert details["txn_ids"] == ["T-2", "T-1"]
        assert details["timestamps"] == ["2026-01-01T00:00:00Z",
                                         "2026-02-02T00:00:00Z"]
        assert details["first_ts"] == "2026-01-01T00:00:00Z"
        assert details["last_ts"] == "2026-02-02T00:00:00Z"

        blank = gv.edge_details(
            "ACC-A", "ACC-Z",
            edges=[{"src": "ACC-A", "dst": "ACC-Z", "count": 1}],
            transactions=[{"src": "ACC-A", "dst": "ACC-X"}],
        )
        assert blank["txn_ids"] == [] and blank["timestamps"] == []
        assert blank["first_ts"] is None

    def test_app_wires_the_selector_edge_ids_and_mapping_captions(self):
        assert "Highlight one evidence record in the graph" in APP_SRC
        assert "evidence_graph_elements" in APP_SRC
        assert "focus_nodes=focus_nodes or None" in APP_SRC
        assert "first_ts" in APP_SRC and "txn_ids" in APP_SRC
        assert "graph mapping: none" in APP_SRC


# ====================== 5. evaluation artifact view =========================


class TestEvaluationArtifact:
    def test_missing_artifact_is_explicit_and_shows_nothing(self, tmp_path):
        summary = backend.load_eval_summary(tmp_path / "missing.json")
        assert summary["ok"] is False and summary["identified"] is False
        assert "no evaluation artifact" in summary["reason"]
        assert vm.eval_metric_rows(summary) == []

    def test_unidentified_artifact_withholds_its_metrics(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps({
            "run": {"mode": "fake"},  # no seed, no dataset graph name
            "metrics": {"counts": {"evaluated": 3},
                        "recall": {"strict": {"display": "1/1 (100%)"}}},
        }), encoding="utf-8")
        summary = backend.load_eval_summary(path)
        assert summary["ok"] is True and summary["identified"] is False
        assert "seed" in summary["reason"] and "withheld" in summary["reason"]
        assert vm.eval_metric_rows(summary) == []

    def test_malformed_artifact_is_a_state_not_a_crash(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text("not json", encoding="utf-8")
        summary = backend.load_eval_summary(path)
        assert summary["ok"] is False
        assert "could not be read" in summary["reason"]

    def test_committed_artifact_is_identified_and_aggregate_only(self):
        summary = backend.load_eval_summary()
        if not summary.get("ok"):
            pytest.skip("no committed evaluation artifact in this tree")
        if not summary.get("identified"):
            pytest.skip("committed artifact is not identified")
        assert isinstance(summary["seed"], int)
        assert summary["mode"]
        assert str(summary["dataset"]["graph_name"]).startswith(
            "HABEAS_demo_eval")
        blob = json.dumps(summary)
        assert "ALR-" not in blob and "per_alert" not in blob
        assert "alerts" not in summary  # never the per-alert rows
        rows = vm.eval_metric_rows(summary)
        labels = {row["metric"] for row in rows}
        assert "Recall — strict" in labels
        assert "Verifier claims checked" in labels
        assert "False-positive rate — strict" in labels

    def test_app_gates_display_on_presence_and_identification(self):
        assert "load_eval_summary" in APP_SRC
        assert "identified" in APP_SRC
        assert "NOT this investigation's results" in APP_SRC
        assert "Reload artifact" in APP_SRC


# ======================= 6. wiring / finishing pass =========================


class TestAppWiring87:
    def test_third_tab_hosts_the_technical_view(self):
        assert ('st.tabs(["Graph & evidence", "Decision & audit", '
                '"Technical & evaluation"])') in APP_SRC

    def test_truncation_is_flagged_in_the_decision_view(self):
        assert "were truncated by the query" in APP_SRC

    def test_existing_export_and_verification_semantics_are_retained(self):
        assert "Download evidence CSV" in APP_SRC
        assert "vm.evidence_csv(rows)" in APP_SRC
        assert "verification" in APP_SRC
        assert "not checked" in APP_SRC
