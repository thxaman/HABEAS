"""Phase 8.5 workspace tests: data-driven inbox, real graph, live events.

These pin the contracts the redesign is judged on:

- the alert inbox is loaded from the graph over the MCP read tool, never a
  hard-coded list, and no alert id branches into a manufactured outcome;
- graph nodes/edges come from the neighbourhood query rows and nothing is
  invented; node/edge selection resolves to the same source records;
- progress events correspond to the investigation's actual tool calls and
  stage transitions (and a broken consumer can never break the run);
- switching alerts clears the previous graph/timeline/decision/report, and a
  rerun never restarts an investigation or reseeds the graph;
- Policy v2 and the graph-safety invariants are untouched by the UI.

DB-backed tests skip cleanly when FalkorDB is unreachable.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.agent.events import ProgressEmitter
from src.tools import query_catalog as qc
from src.ui import backend
from src.ui import graph_view as gv
from src.ui import view_models as vm

REPO = Path(__file__).resolve().parents[1]
APP_SRC = (REPO / "app.py").read_text(encoding="utf-8")
UI_MODULES = [
    REPO / "app.py",
    REPO / "src" / "ui" / "backend.py",
    REPO / "src" / "ui" / "view_models.py",
    REPO / "src" / "ui" / "graph_view.py",
]


@pytest.fixture()
def ui_dirs(tmp_path, monkeypatch):
    reports = tmp_path / "reports-ui"
    traces = tmp_path / "traces-ui"
    monkeypatch.setattr(backend, "UI_REPORTS_DIR", str(reports))
    monkeypatch.setattr(backend, "UI_TRACE_DIR", str(traces))
    return reports, traces


# ============================ 1. data-driven inbox ==========================


class TestAlertInbox:
    def test_new_read_tools_are_cataloged_read_only_and_bounded(self):
        for query_id in ("alert_inbox_v1", "neighborhood_out_accounts_v1",
                         "neighborhood_in_accounts_v1", "neighborhood_edges_v1"):
            assert query_id in qc._QUERY_TEMPLATES
            assert qc.is_write_query(query_id) is False
            assert "LIMIT" in qc.get_template(query_id)

    @pytest.mark.usefixtures("graph")
    def test_inbox_is_loaded_from_the_graph_through_mcp(self, settings):
        settings = backend.ui_settings(settings)
        inbox = backend.load_alert_inbox(settings)
        assert inbox["ok"] is True, inbox.get("error")
        assert inbox["query_ids"] == ["alert_inbox_v1"]  # came over MCP
        alerts = inbox["alerts"]
        assert len(alerts) > 5
        row = alerts[0]
        for field in ("alert_id", "reason", "concerned_account_id",
                      "trigger_transaction", "case_verdict", "action_status"):
            assert field in row
        assert all(str(a["alert_id"]).startswith("ALR-") for a in alerts)

    @pytest.mark.usefixtures("graph")
    def test_inbox_reports_existing_investigation_state(self, settings):
        settings = backend.ui_settings(settings)
        inbox = backend.load_alert_inbox(settings)
        investigated = [a for a in inbox["alerts"] if a.get("case_id")]
        assert investigated, "the demo graph should already hold cases"
        assert all(a["case_verdict"] for a in investigated)

    def test_inbox_failure_is_reported_not_hidden(self, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("no db")

        monkeypatch.setattr(backend, "_mcp_call", boom)
        inbox = backend.load_alert_inbox(backend.ui_settings())
        assert inbox["ok"] is False and "no db" in inbox["error"]

    def test_app_hardcodes_no_alert_outcomes(self):
        """Only the three *recommended markers* may appear; no branching on id."""
        id_lines = [ln for ln in APP_SRC.splitlines() if "ALR-000" in ln]
        assert len(id_lines) == 1, id_lines
        assert "RECOMMENDED_ALERTS" in id_lines[0]
        for forbidden in ('verdict = "suspicion', 'action_status = "applied"',
                          'if row["alert_id"] ==', "MOCK", "FIXTURE_RESULT"):
            assert forbidden not in APP_SRC, forbidden

    def test_app_renders_inbox_rows_from_state_not_literals(self):
        assert "_load_inbox" in APP_SRC and "load_alert_inbox" in APP_SRC
        assert "RECOMMENDED_ALERTS" in APP_SRC


# ============================== 2. real graph ==============================


def _fake_neighborhood() -> dict:
    return {
        "ok": True,
        "focal_account_id": "ACC-A",
        "accounts": ["ACC-A", "ACC-B", "ACC-C"],
        "edges": [
            {"src": "ACC-A", "dst": "ACC-B", "count": 3,
             "total_amount": 100.0, "last_ts": "2026-01-02T00:00:00Z", "active": True},
            {"src": "ACC-B", "dst": "ACC-C", "count": 2,
             "total_amount": 50.0, "last_ts": "2026-01-03T00:00:00Z", "active": True},
            {"src": "ACC-C", "dst": "ACC-A", "count": 1,
             "total_amount": 25.0, "last_ts": "2026-01-04T00:00:00Z", "active": False},
        ],
        "transactions": [
            {"src": "ACC-A", "dst": "ACC-B", "txn_id": "TXN-1",
             "ts": "2026-01-02T00:00:00Z", "amount": 100.0},
        ],
        "truncated": False,
        "interpretation": "Bounded 2-hop neighbourhood of ACC-A",
    }


class TestGraphData:
    def test_figure_contains_exactly_the_query_nodes_and_edges(self):
        nb = _fake_neighborhood()
        fig = gv.build_figure(accounts=nb["accounts"], edges=nb["edges"],
                              focal_account_id="ACC-A")
        assert fig is not None
        node_points = []
        edge_segments = 0
        for trace in fig.data:
            if trace.mode and "markers" in trace.mode:
                node_points.extend(list(trace.text))
            elif trace.mode and "lines" in trace.mode:
                edge_segments += len([x for x in trace.x if x is not None]) // 2
        assert sorted(node_points) == sorted(nb["accounts"])  # nothing invented
        assert edge_segments == len(nb["edges"])

    def test_empty_graph_renders_nothing_rather_than_filling_the_gap(self):
        assert gv.build_figure(accounts=[], edges=[]) is None

    def test_active_and_inactive_edges_are_visually_distinct(self):
        fig = gv.build_figure(accounts=["ACC-A", "ACC-C"],
                              edges=[{"src": "ACC-C", "dst": "ACC-A", "count": 1,
                                      "total_amount": 1.0, "active": False}])
        dashes = {t.line.dash for t in fig.data if t.mode and "lines" in t.mode}
        assert "dash" in dashes, "deactivated routing must be dashed, not just recoloured"

    def test_structural_cycle_is_distinguished_from_circulation(self):
        fig = gv.build_figure(
            accounts=["ACC-A", "ACC-B"],
            edges=[{"src": "ACC-A", "dst": "ACC-B", "count": 1}],
            cycle_edges=[{"src": "ACC-A", "dst": "ACC-B", "corroborated": False}],
        )
        names = {t.name for t in fig.data}
        assert "structural cycle only" in names or "corroborated circulation" in names
        dashes = {t.line.dash for t in fig.data if t.mode and "lines" in t.mode}
        assert "dot" in dashes or "dash" in dashes

    @pytest.mark.usefixtures("graph")
    def test_neighborhood_comes_from_the_read_only_tool(self, settings):
        settings = backend.ui_settings(settings)
        nb = backend.load_neighborhood("ACC-0387", settings=settings)
        assert nb["ok"] is True, nb.get("error")
        assert nb["focal_account_id"] == "ACC-0387"
        assert "ACC-0387" in nb["accounts"]
        assert nb["edges"], "the demo ring must have real edges"
        assert all({"src", "dst", "active"} <= set(e) for e in nb["edges"])
        assert nb["transactions"], "underlying transactions should be returned"

    def test_merge_keeps_both_sources_and_invents_nothing(self):
        nb = _fake_neighborhood()
        ev = vm.graph_from_evidence(
            [{"evidence_id": "EV-1", "tool": "detect_cycle", "ok": True,
              "truncated": [], "stance": "supporting",
              "query_ids": ["detect_closed_paths_v1"], "row_count": [1],
              "derived": {"cycles": [{"temporally_corroborated": True,
                                      "legs": [{"src": "ACC-A", "dst": "ACC-D"}],
                                      "events": []}]}}],
            ["ACC-A"],
        )
        merged = vm.merge_graphs(nb, ev, focal="ACC-A")
        assert set(merged["accounts"]) == {"ACC-A", "ACC-B", "ACC-C", "ACC-D"}
        assert merged["focal"] == "ACC-A"
        assert any(c["corroborated"] for c in merged["cycle_edges"])


class TestSelection:
    def test_node_and_edge_selection_map_to_real_records(self):
        node = gv.selected_from_event(
            {"selection": {"points": [{"customdata": ["node", "ACC-B"]}]}})
        assert node == {"kind": "node", "account_id": "ACC-B"}
        edge = gv.selected_from_event(
            {"selection": {"points": [{"customdata": ["edge", "ACC-A", "ACC-B"]}]}})
        assert edge == {"kind": "edge", "src": "ACC-A", "dst": "ACC-B"}
        assert gv.selected_from_event({"selection": {"points": []}}) is None
        assert gv.selected_from_event(None) is None

    def test_node_details_match_the_source_edges(self):
        nb = _fake_neighborhood()
        details = gv.node_details("ACC-B", edges=nb["edges"],
                                 transactions=nb["transactions"])
        assert details["out_degree"] == 1 and details["in_degree"] == 1
        assert details["outgoing"][0]["dst"] == "ACC-C"
        assert details["incoming"][0]["src"] == "ACC-A"

    def test_edge_details_expose_the_aggregate_and_transactions(self):
        nb = _fake_neighborhood()
        details = gv.edge_details("ACC-A", "ACC-B", edges=nb["edges"],
                                  transactions=nb["transactions"])
        assert details["edge"]["count"] == 3
        assert details["has_aggregate"] is True
        assert details["transactions"][0]["txn_id"] == "TXN-1"

    def test_edge_without_transactions_is_reported_honestly(self):
        nb = _fake_neighborhood()
        details = gv.edge_details("ACC-B", "ACC-C", edges=nb["edges"],
                                  transactions=nb["transactions"])
        assert details["has_aggregate"] is True
        assert details["has_transactions"] is False


# ============================ 3. progress events ============================


class TestProgressEvents:
    def test_emitter_orders_events_and_survives_a_broken_consumer(self):
        def broken(_event):
            raise RuntimeError("consumer exploded")

        emitter = ProgressEmitter(broken)  # must not raise
        first = emitter.emit(type="stage", title="a")
        second = emitter.emit(type="tool", title="b", tool="entity_profile")
        assert (first["seq"], second["seq"]) == (1, 2)
        assert [e["title"] for e in emitter.events] == ["a", "b"]
        with pytest.raises(ValueError):
            emitter.emit(type="not-a-type", title="x")

    @pytest.mark.usefixtures("graph")
    def test_events_track_the_real_tool_calls_and_stages(self, settings, ui_dirs):
        settings = backend.ui_settings(settings)
        events: list[dict] = []
        snapshot = backend.run_scenario(
            "ALR-0001", settings=settings, mode="fake",
            on_event=events.append, trace_dir=ui_dirs[1],
        )
        assert snapshot["status"] == "ok"
        assert events, "a real run must emit progress events"

        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        assert all(e["type"] in ("stage", "tool", "evidence", "hypotheses",
                                 "decision", "policy", "action",
                                 "verification", "error") for e in events)

        action_tools = {"create_case", "cite_evidence", "simulate_freeze",
                        "freeze_account", "verify_freeze"}
        called = [e for e in events if e["type"] == "tool"]
        for tool in snapshot["tool_sequence"]:
            if tool in action_tools:
                continue  # action tools are reported as 'action' events
            assert any(e.get("tool") == tool for e in called), tool

        stages = [e.get("stage") for e in events if e["type"] == "action"]
        assert stages == ["RECORD_CASE", "CITE_EVIDENCE", "SIMULATE", "ACT"]
        # the policy gate and the post-action verification are their own event
        # types, so the UI can show the decision and the independent check as
        # distinct stages rather than folded into "action".
        assert any(e["type"] == "policy" for e in events)
        assert any(e["type"] == "verification" for e in events)
        assert any(e["type"] == "evidence" for e in events)
        # every tool call is reported as a started/completed pair
        pairs = [(e.get("tool"), e.get("phase")) for e in called]
        started = {t for t, p in pairs if p == "started"}
        completed = {t for t, p in pairs if p == "completed"}
        assert started == completed, (started ^ completed)
        assert any(e["type"] == "decision" for e in events)
        assert any(e["type"] == "hypotheses" for e in events)

    @pytest.mark.usefixtures("graph")
    def test_events_expose_no_secret_ground_truth_or_path(self, settings, ui_dirs):
        settings = backend.ui_settings(settings)
        events: list[dict] = []
        backend.run_scenario("ALR-0001", settings=settings, mode="fake",
                             on_event=events.append, trace_dir=ui_dirs[1])
        blob = json.dumps(events, default=str)
        for forbidden in ("gt_", "ground_truth", "sk-", "gsk-", "API_KEY"):
            assert forbidden not in blob, forbidden

    def test_job_runs_in_a_thread_and_exposes_live_state(self, settings):
        settings = backend.ui_settings(settings)
        captured = {}

        def fake_runner(alert_id, *, settings, mode, on_event, trace_dir=None):
            captured["alert_id"] = alert_id
            on_event({"seq": 1, "type": "stage", "title": "started",
                      "detail": "", "ok": None, "meta": {}})
            on_event({"seq": 2, "type": "tool", "title": "done",
                      "detail": "", "ok": True, "meta": {}})
            return {"alert_id": alert_id, "status": "ok", "completed": True}

        job = backend.start_investigation(
            "ALR-9999", settings=settings, mode="fake", runner=fake_runner
        )
        job.wait(timeout=15)
        assert job.done is True and job.error is None
        assert captured["alert_id"] == "ALR-9999"
        assert [e["title"] for e in job.events()] == ["started", "done"]
        assert job.snapshot["status"] == "ok"

    def test_job_reports_a_runner_exception_without_crashing(self, settings):
        def exploding_runner(alert_id, **_kw):
            raise RuntimeError("backend exploded")

        job = backend.start_investigation(
            "ALR-0001", settings=backend.ui_settings(settings), mode="fake",
            runner=exploding_runner,
        )
        job.wait(timeout=15)
        assert job.done is True
        assert "backend exploded" in (job.error or "")
        assert job.snapshot is None


# ======================= 4. state scoping / no side effects =================


class TestStateScoping:
    def test_stale_run_is_never_shown_for_another_alert(self):
        assert vm.is_stale({"alert_id": "ALR-0001"}, "ALR-0004") is True
        assert vm.is_stale({"alert_id": "ALR-0001"}, "ALR-0001") is False
        assert vm.is_stale(None, "ALR-0001") is False

    @pytest.mark.usefixtures("graph")
    def test_switching_alerts_clears_graph_timeline_decision_and_report(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=180)
        at.run()
        assert not at.exception
        at.session_state["run"] = {"alert_id": "ALR-0001", "status": "ok"}
        at.session_state["job"] = "sentinel-job"
        at.session_state["neighborhood"] = {"ok": True, "accounts": ["ACC-0387"]}
        at.run()
        assert at.session_state["run"] is not None  # same alert: kept

        alert_radio = at.radio[0]
        alert_radio.set_value(alert_radio.options[3]).run()
        # the previous investigation artifacts are cleared ...
        assert at.session_state["run"] is None
        assert at.session_state["job"] is None
        # ... and the graph is reloaded for the NEWLY selected alert
        nb = at.session_state["neighborhood"]
        assert nb is not None and nb["focal_account_id"] != "ACC-0387"

    @pytest.mark.usefixtures("graph")
    def test_plain_reruns_never_start_a_job_or_reseed(self, monkeypatch):
        from streamlit.testing.v1 import AppTest

        def forbidden(*_a, **_k):
            raise AssertionError("the demo graph was reseeded automatically")

        monkeypatch.setattr("src.demo.run_demo.reseed_demo_graph", forbidden)
        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=180)
        at.run()
        at.run()
        at.run()
        assert not at.exception
        assert at.session_state["job"] is None
        assert at.session_state["run"] is None

    def test_reset_is_only_reachable_through_the_confirmed_control(self):
        assert APP_SRC.count("reset_demo_graph") == 1
        assert "I confirm: rebuild the demo graph" in APP_SRC
        assert "disabled=not confirm" in APP_SRC
        backend_src = (REPO / "src" / "ui" / "backend.py").read_text(encoding="utf-8")
        assert backend_src.count("reseed_demo_graph(") == 1


# ===================== 5. states: errors / outcomes / verification ==========


class TestStateRendering:
    def test_run_status_distinguishes_every_documented_state(self):
        assert vm.run_status_chip({"status": "configuration_error"})[1] == "error"
        assert vm.run_status_chip({"status": "failed"}) == ("Investigation failed",
                                                            "error")
        assert vm.run_status_chip({})[0] == "No investigation run yet"
        assert vm.run_status_chip({"status": "ok", "completed": False})[1] == "watch"
        assert vm.run_status_chip({"status": "ok", "completed": True})[1] == "positive"

    def test_action_and_verification_states_are_explicit(self):
        assert vm.action_chip("applied")[1] == "positive"
        assert vm.action_chip("declined")[1] == "abstain"
        assert vm.verification_chip("mismatch")[1] == "error"
        assert vm.verification_chip("unverifiable")[1] == "abstain"
        counts = vm.verification_counts({"reproduced_current_state": 2,
                                        "changed_after_action": 1,
                                        "mismatch": 0, "unverifiable": 3})
        assert counts == {"reproduced": 2, "changed_after_action": 1,
                          "mismatch": 0, "unverifiable": 3}

    def test_summarize_run_survives_a_failed_snapshot(self):
        summary = vm.summarize_run({"status": "failed", "errors": ["boom"]})
        assert summary["status"] == "failed"
        assert summary["tool_calls"] == 0 and summary["evidence_count"] == 0
        assert summary["errors"] == ["boom"]

    @pytest.mark.usefixtures("graph")
    def test_failed_snapshot_renders_without_crashing(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(REPO / "app.py"), default_timeout=180)
        at.run()
        at.session_state["run"] = {
            "alert_id": "ALR-0001", "status": "failed", "error": "mcp down",
            "errors": ["mcp down"], "completed": False, "tool_calls": 0,
            "tool_budget": 12, "evidence": [], "hypotheses": [], "action": None,
            "report": None, "signal_families": [], "primary_entities": [],
            "mode_label": "FakeLLM", "graph_name": "x",
        }
        at.run()
        assert not at.exception

    @pytest.mark.usefixtures("graph")
    def test_real_run_populates_decision_action_and_verification(self, settings,
                                                                 ui_dirs):
        settings = backend.ui_settings(settings)
        snapshot = backend.run_scenario("ALR-0021", settings=settings,
                                        mode="fake", trace_dir=ui_dirs[1])
        action = snapshot["action"]
        assert action["policy_version"] is not None  # Policy v2 decided it
        assert action["checks"]
        assert snapshot["simulation"] is not None
        assert snapshot["post_action"] is not None
        report = snapshot["report"]
        assert report["summary"] and report.get("claims") is not None
        assert Path(report["written_to"]).exists()
        assert Path(report["written_to"]).parent == ui_dirs[0]

    @pytest.mark.usefixtures("graph")
    def test_declined_action_shows_no_mutation_and_the_real_account_state(
        self, settings, ui_dirs
    ):
        settings = backend.ui_settings(settings)
        snapshot = backend.run_scenario("ALR-0001", settings=settings,
                                        mode="fake", trace_dir=ui_dirs[1])
        assert snapshot["action"]["action_status"] == "declined"
        assert snapshot["action"]["mutation_performed"] is False
        assert snapshot["post_action"]["account_status"] == "active"
        assert snapshot["action"]["blocking_reasons"]


# ==================== 6. safety invariants (policy / graph) =================


class TestSafetyInvariants:
    def test_ui_backend_takes_no_graph_name_or_cypher(self):
        for func in (backend.run_scenario, backend.load_alert_inbox,
                     backend.load_neighborhood, backend.reset_demo_graph,
                     backend.start_investigation):
            params = inspect.signature(func).parameters
            assert "graph_name" not in params, func.__name__
            assert "cypher" not in params, func.__name__
        assert backend.ui_settings().graph_name.startswith("HABEAS_demo_showcase")

    def test_agent_graph_access_goes_through_mcp_only(self):
        backend_src = (REPO / "src" / "ui" / "backend.py").read_text(encoding="utf-8")
        assert "call_tool_sync" in backend_src  # MCP transport
        for source in (backend_src, APP_SRC):
            assert "run_query(" not in source
            assert "run_evidence(" not in source
            assert "src.tools.reads" not in source
            assert "src.tools.query_catalog" not in source
        # one trusted graph handle, passed to the Phase 6 report builder
        assert backend_src.count("get_graph(") == 1

    def test_ui_never_recomputes_policy_or_action_eligibility(self):
        for path in UI_MODULES:
            source = path.read_text(encoding="utf-8")
            assert "evaluate_freeze_eligibility" not in source, path.name
            assert "canonical_verdict(" not in source, path.name
            assert "escalation_verdict(" not in source, path.name
        backend_src = (REPO / "src" / "ui" / "backend.py").read_text(encoding="utf-8")
        assert "policy" in backend_src

    def test_no_text_area_chat_or_free_form_cypher_input(self):
        for forbidden in ("st.text_area", "st.chat_input", "st.chat_message"):
            assert forbidden not in APP_SRC, forbidden
        # exactly one text input: the inbox search (never a query/target)
        assert APP_SRC.count("st.text_input") == 1
        assert "Search alerts" in APP_SRC

    def test_no_ground_truth_secret_or_local_path_in_ui_sources(self):
        for path in UI_MODULES:
            source = path.read_text(encoding="utf-8")
            for forbidden in ("ground_truth", "gt_label", "API_KEY", "sk-",
                              "gsk-"):
                assert forbidden not in source, f"{path.name}: {forbidden}"


# ============================ 7. report integrity ===========================


class TestReports:
    def test_reports_use_the_existing_builder_and_verifier(self):
        backend_src = (REPO / "src" / "ui" / "backend.py").read_text(encoding="utf-8")
        assert "build_report" in backend_src
        assert "write_report" in backend_src
        assert "verification" in backend_src

    def test_report_never_overwrites_the_historical_root(self):
        from src.ui.backend import UI_REPORTS_DIR

        parts = Path(UI_REPORTS_DIR).parts
        assert parts[0] == "reports" and len(parts) > 1

    def test_build_report_cli_still_refuses_silent_overwrite(self):
        source = (REPO / "src" / "report" / "__main__.py").read_text(encoding="utf-8")
        assert "refusing to overwrite" in source
        assert "--force" in source