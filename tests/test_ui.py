"""Phase 8.5 UI tests.

The UI is a presentation layer, so these tests pin the contracts that make it
trustworthy:

- the three curated scenarios are presented from REAL backend snapshots
  (FakeLLM over the real MCP stdio server) — A and C declined, B applied —
  and never from hard-coded outcomes;
- failed / incomplete / unconfigured runs are explicit states, never a green
  status left over from an earlier success;
- switching alerts discards the stored run (no stale results);
- the demo graph is never reseeded by rendering, reruns, or investigations —
  only by the explicit confirmed control;
- the UI exposes no graph-name or Cypher input and never computes policy
  decisions itself (Policy v2 stays authoritative in src/agent/policy.py);
- reports go to the UI's own subdirectory (no silent overwrite of historical
  artifacts);
- no ground-truth or credential material can appear in a snapshot.

DB-backed tests skip cleanly when FalkorDB is unreachable.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.ui import backend, view_models as vm

REPO = Path(__file__).resolve().parents[1]
APP_SRC = (REPO / "app.py").read_text(encoding="utf-8")


# ----------------------------------------------------------------- scenarios


@pytest.fixture()
def ui_dirs(tmp_path, monkeypatch):
    """Keep the UI's report/trace output inside the test's tmp directory."""
    reports = tmp_path / "reports-ui"
    traces = tmp_path / "traces-ui"
    monkeypatch.setattr(backend, "UI_REPORTS_DIR", str(reports))
    monkeypatch.setattr(backend, "UI_TRACE_DIR", str(traces))
    return reports, traces


def _all_scenario_snapshots(settings, ui_dirs):
    reports, _traces = ui_dirs
    return {
        s["alert_id"]: backend.run_scenario(
            s["alert_id"], settings=settings, mode="fake"
        )
        for s in backend.list_scenarios()
    }, reports


@pytest.mark.usefixtures("graph")
def test_all_three_scenarios_present_real_outcomes(settings, ui_dirs):
    """The UI's three scenarios, driven through the real MCP backend."""
    settings = backend.ui_settings(settings)
    # Explicit guarded rebuild (the same control the UI exposes) so the
    # assertions reflect a fresh run and never leftover freeze state from an
    # earlier demo or test session.
    backend.reset_demo_graph(settings)
    snapshots, reports = _all_scenario_snapshots(settings, ui_dirs)

    a = snapshots["ALR-0001"]
    assert a["status"] == "ok" and a["completed"] is True
    assert a["action"]["action_status"] == "declined"
    assert a["action"]["allowed"] is False
    assert a["action"]["blocking_reasons"], "a declined action must say why"
    assert a["action"]["distinct_provenance_families"] == ["circulation"]
    assert a["post_action"]["account_status"] == "active"  # nothing changed

    b = snapshots["ALR-0021"]
    assert b["status"] == "ok" and b["completed"] is True
    assert b["action"]["action_status"] == "applied"
    assert b["action"]["frozen"] is True
    assert b["action"]["action_id"] and b["action"]["action_id"].startswith("ACT-")
    assert sorted(b["action"]["distinct_provenance_families"]) == [
        "circulation", "rapid_forwarding",
    ]
    assert b["post_action"]["checks"]["account_is_frozen"] is True
    assert b["post_action"]["verified"] is True

    c = snapshots["ALR-0004"]
    assert c["action"]["action_status"] == "declined"
    assert c["canonical_verdict"] == "insufficient_evidence"
    assert len(c["action"]["distinct_provenance_families"]) == 1

    # every snapshot carries real multi-query evidence + a verified report
    for alert_id, snap in snapshots.items():
        assert snap["evidence"], alert_id
        assert all(e["query_ids"] for e in snap["evidence"])
        assert snap["report"], alert_id
        assert "error" not in snap["report"]
        assert snap["report"]["summary"].get("mismatch", 0) == 0
        written = Path(snap["report"]["written_to"])
        assert written.exists()
        assert written.parent == reports  # the UI's own directory, never reports/


@pytest.mark.usefixtures("graph")
def test_scenario_snapshots_never_carry_ground_truth_or_secrets(settings, ui_dirs):
    settings = backend.ui_settings(settings)
    snapshots, _ = _all_scenario_snapshots(settings, ui_dirs)
    for snap in snapshots.values():
        blob = json.dumps(snap, default=str)
        assert "gt_" not in blob
        assert "ground_truth" not in blob
        assert "manifest_path" not in blob
        assert "GROQ_API_KEY" not in blob
        for forbidden in ("sk-", "gsk-"):
            assert forbidden not in blob


# ------------------------------------------------------------- failure states


def test_failed_investigation_is_an_explicit_state(settings, ui_dirs, monkeypatch):
    settings = backend.ui_settings(settings)

    def boom(*_a, **_k):
        raise RuntimeError("MCP server died")

    monkeypatch.setattr(backend, "run_case_workflow", boom)
    snap = backend.run_scenario("ALR-0001", settings=settings, mode="fake")
    assert snap["status"] == "failed"
    assert "MCP server died" in snap["error"]
    assert snap["completed"] is False
    assert snap["tool_calls"] == 0 and snap["evidence"] == []
    assert vm.run_status_chip(snap) == ("Investigation failed", "error")
    assert snap["termination_reason"] == "unrecoverable_error"


def test_unconfigured_provider_is_a_clear_error_without_secrets(settings):
    from src.config import Settings

    bare = Settings(
        **{**settings.model_dump(), "llm_provider": "groq",
           "groq_api_key": "", "groq_model": ""}
    )
    snap = backend.run_scenario("ALR-0001", settings=bare, mode="real")
    assert snap["status"] == "configuration_error"
    assert "GROQ_API_KEY" in snap["error"]
    # the error names the missing variable, never a credential value
    assert "gsk" not in snap["error"].lower()
    assert vm.run_status_chip(snap) == ("Provider not configured", "error")


def test_incomplete_investigation_is_never_green():
    snap = {
        "status": "ok", "completed": False,
        "termination_reason": "tool_budget_exhausted",
        "canonical_verdict": None, "action": None, "mode_label": "FakeLLM",
    }
    assert vm.run_status_chip(snap) == ("Investigation incomplete", "watch")


# ------------------------------------------------------------ view models


def test_status_chips_cover_the_documented_vocabulary():
    assert vm.verdict_chip("suspicion_supported_for_review") == (
        "Supported for review", "positive")
    assert vm.verdict_chip("insufficient_evidence") == (
        "Insufficient evidence", "abstain")
    assert vm.verdict_chip(None)[1] == "abstain"
    assert vm.action_chip("declined") == ("Declined by policy", "abstain")
    assert vm.action_chip("applied") == ("Freeze applied (sandbox)", "positive")
    assert vm.hypothesis_chip("contradicted")[1] == "error"
    assert vm.hypothesis_chip("open") == ("Open / untested", "muted")
    assert vm.verification_chip("mismatch") == ("Mismatch", "error")
    for chip_fn in (vm.verdict_chip, vm.action_chip, vm.hypothesis_chip,
                    vm.stance_chip, vm.verification_chip):
        for status in ("supported", "contradicted", "open", "insufficient",
                       "unverifiable", "mismatch", "applied", "declined", None):
            assert chip_fn(status)[1] in vm.TONES


def test_verification_counts_merge_the_two_pass_statuses():
    counts = vm.verification_counts({
        "reproduced_current_state": 3,
        "verified_against_recorded_evidence": 2,
        "changed_after_action": 4,
        "mismatch": 0,
        "unverifiable": 1,
    })
    assert counts == {"reproduced": 5, "changed_after_action": 4,
                      "mismatch": 0, "unverifiable": 1}
    assert vm.verification_counts(None) == {
        "reproduced": 0, "changed_after_action": 0, "mismatch": 0,
        "unverifiable": 0,
    }


def _cycle_entry(corroborated: bool, truncated: bool = False) -> dict:
    return {
        "evidence_id": "EV-0002", "tool": "detect_cycle", "ok": True,
        "truncated": [truncated, truncated], "stance": "supporting",
        "query_ids": ["detect_closed_paths_v1", "txn_events_for_legs_v1"],
        "row_count": [1, 4], "derived": {
            "cycles": [{
                "members": ["ACC-0387", "ACC-0388", "ACC-0389", "ACC-0390"],
                "temporally_corroborated": corroborated,
                "legs": [
                    {"src": "ACC-0387", "dst": "ACC-0388"},
                    {"src": "ACC-0388", "dst": "ACC-0389"},
                ],
                "events": [
                    {"src": "ACC-0387", "dst": "ACC-0388",
                     "txn_id": "TXN-1", "ts": "2026-01-05T10:00:00Z"},
                ],
            }],
        },
        "interpretation": "cycle detected",
    }


def _trace_entry(truncated: bool = False) -> dict:
    return {
        "evidence_id": "EV-0004", "tool": "trace_funds", "ok": True,
        "truncated": [truncated], "stance": "unresolved",
        "query_ids": ["trace_paths_out_v1", "txn_events_for_legs_v1"],
        "row_count": [2, 2], "derived": {
            "leg_events": [
                {"src": "ACC-0387", "dst": "ACC-0390",
                 "txn_id": "TXN-9", "ts": "2026-01-06T09:00:00Z"},
            ],
        },
        "interpretation": "outbound trace",
    }


def test_graph_is_built_only_from_recorded_evidence():
    graph = vm.graph_from_evidence(
        [_cycle_entry(corroborated=True), _trace_entry()],
        ["ACC-0387"],
    )
    ids = {n["id"] for n in graph["nodes"]}
    assert ids == {"ACC-0387", "ACC-0388", "ACC-0389", "ACC-0390"}
    assert graph["nodes"][0]["kind"] == "primary"  # ACC-0387 first, highlighted
    edges = {(e["src"], e["dst"]): e for e in graph["edges"]}
    assert ("ACC-0387", "ACC-0388") in edges
    assert edges[("ACC-0387", "ACC-0388")]["corroborated"] is True
    assert edges[("ACC-0387", "ACC-0388")]["txn_ids"] == ["TXN-1"]
    # the traced leg appears dotted with its own txn id
    assert edges[("ACC-0387", "ACC-0390")]["kind"] == "traced"
    assert graph["truncated"] is False and graph["has_content"] is True


def test_structural_cycles_stay_visually_distinct_from_circulation():
    graph = vm.graph_from_evidence([_cycle_entry(corroborated=False)])
    edge = graph["edges"][0]
    assert edge["corroborated"] is False
    assert "structural connectivity only" in " ".join(graph["notes"])
    dot = vm.to_dot(graph)
    assert "structure only" in dot and "style=dashed" in dot


def test_truncated_evidence_marks_the_graph_incomplete():
    graph = vm.graph_from_evidence([_cycle_entry(True, truncated=True)])
    assert graph["truncated"] is True
    assert any("truncated" in n for n in graph["notes"])


def test_empty_evidence_renders_no_invented_graph():
    graph = vm.graph_from_evidence([], ["ACC-0387"])
    assert graph["has_content"] is False
    assert graph["nodes"] == [] and graph["edges"] == []


def test_dot_rendering_is_bounded_and_brands_corroboration():
    graph = vm.graph_from_evidence(
        [_cycle_entry(True), _trace_entry()], ["ACC-0387"]
    )
    dot = vm.to_dot(graph)
    assert dot.startswith("digraph HABEAS {")
    assert '"ACC-0387"' in dot and "circulated" in dot
    assert "2026-01-06T09:00:00Z" in dot and "TXN-9" in dot


def test_evidence_inspector_keeps_every_query_of_a_multi_query_record():
    rows = vm.evidence_inspector_rows(
        [_cycle_entry(True)],
        [{"evidence_id": "EV-0002", "status": "reproduced_current_state",
          "reason": "replayed", "checked_source_queries": [
              "detect_closed_paths_v1", "txn_events_for_legs_v1"],
          "mismatched_source_queries": []}],
    )
    row = rows[0]
    assert row["query_count"] == 2 and row["multi_query"] is True
    assert row["query_ids"] == [
        "detect_closed_paths_v1", "txn_events_for_legs_v1",
    ]
    assert row["total_rows"] == 5
    assert row["verification_status"] == "reproduced_current_state"


# ------------------------------------------------------------ stale results


def test_is_stale_flags_runs_from_another_alert():
    run = {"alert_id": "ALR-0001"}
    assert vm.is_stale(run, "ALR-0004") is True
    assert vm.is_stale(run, "ALR-0001") is False
    assert vm.is_stale(None, "ALR-0001") is False


def test_app_clears_the_stored_run_when_the_alert_changes():
    """Streamlit AppTest: switching the selector must not keep the old run."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    at.session_state["run"] = {"alert_id": "ALR-0001", "status": "ok"}
    at.session_state["selected_alert"] = "ALR-0001"
    at.run()
    assert at.session_state["run"] == {"alert_id": "ALR-0001", "status": "ok"}

    # selecting a different alert fires the on_change callback
    at.radio[0].set_value(at.radio[0].options[2]).run()
    assert at.session_state["run"] is None


# --------------------------------------------------------- safety invariants


def test_ui_backend_never_accepts_a_graph_name():
    for func in (backend.run_scenario, backend.reset_demo_graph,
                 backend.ui_settings):
        assert "graph_name" not in inspect.signature(func).parameters
    settings = backend.ui_settings()
    assert settings.graph_name.startswith("HABEAS_demo_showcase")
    assert settings.is_sandbox is True


def test_app_has_no_arbitrary_cypher_or_query_input():
    """Exactly one text input exists — the inbox search — and no field accepts
    a graph name, Cypher, or a target entity."""
    for forbidden in ("st.text_area", "st.chat_input", "st.chat_message"):
        assert forbidden not in APP_SRC, forbidden
    assert APP_SRC.count("st.text_input") == 1
    assert "Search alerts" in APP_SRC


def test_ui_never_recomputes_policy_decisions():
    """Policy v2 stays authoritative in src/agent/policy.py; the UI only reads
    the recorded decision from the action envelope."""
    for source in (APP_SRC,
                   (REPO / "src" / "ui" / "backend.py").read_text(encoding="utf-8"),
                   (REPO / "src" / "ui" / "view_models.py").read_text(encoding="utf-8")):
        assert "evaluate_freeze_eligibility" not in source
        assert "canonical_verdict(" not in source


def test_reset_is_explicit_only_never_automatic():
    # the UI references the reset exactly once — inside the confirmed control
    assert APP_SRC.count("reset_demo_graph") == 1
    assert "Rebuild demo graph" in APP_SRC
    assert "I confirm: rebuild the demo graph" in APP_SRC
    # and the investigation path never reseeds (count call sites, not the
    # docstring mention)
    backend_src = (REPO / "src" / "ui" / "backend.py").read_text(encoding="utf-8")
    assert backend_src.count("reseed_demo_graph(") == 1  # only inside reset_demo_graph


@pytest.mark.usefixtures("graph")
def test_investigation_and_rerun_never_reseed_the_graph(settings, ui_dirs, monkeypatch):
    """Behavioral guard: rendering + a real investigation never touch the seeder."""
    from streamlit.testing.v1 import AppTest

    def forbidden(*_a, **_k):
        raise AssertionError("the demo graph was reseeded automatically")

    monkeypatch.setattr("src.demo.run_demo.reseed_demo_graph", forbidden)
    monkeypatch.setattr(backend, "UI_REPORTS_DIR", str(ui_dirs[0]))
    monkeypatch.setattr(backend, "UI_TRACE_DIR", str(ui_dirs[1]))

    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    investigate = next(b for b in at.button if b.label == "Investigate alert")
    investigate.click().run()
    assert not at.exception
    assert at.session_state["run"] is not None
    assert at.session_state["run"]["alert_id"] == "ALR-0001"


# ------------------------------------------------------- reports and history


def test_ui_reports_dir_is_a_subdirectory_of_reports():
    from src.ui.backend import UI_REPORTS_DIR

    parts = Path(UI_REPORTS_DIR).parts
    assert parts[0] == "reports" and len(parts) > 1


@pytest.mark.usefixtures("graph")
def test_scenario_a_does_not_touch_historical_reports(settings, ui_dirs):
    settings = backend.ui_settings(settings)
    snap = backend.run_scenario("ALR-0001", settings=settings, mode="fake")
    written = Path(snap["report"]["written_to"])
    assert "ui" in written.parts or written.parent == Path(ui_dirs[0])
    # the Phase 6 historical artifacts in reports/ root are never the target
    historical = REPO / "reports" / "CASE-ALR-0001.md"
    if historical.exists():
        assert written.resolve() != historical.resolve()


# --------------------------------------------------------- scenario metadata


def test_scenario_list_describes_but_never_predicts_outcomes():
    for scenario in backend.list_scenarios():
        assert set(scenario) == {"key", "alert_id", "title", "stake"}
        text = (scenario["title"] + " " + scenario["stake"]).lower()
        assert "declined" not in text and "applied" not in text
        assert "frozen" not in text


def test_mode_label_never_contains_a_credential():
    label = backend.run_demo.mode_label("fake", object())
    assert "API_KEY" not in label and "key" not in label.lower()