"""Phase 8 demo tests: graph safety, no-artifact-overwrite, link honesty.

The demo is a thin presentation layer over the established Phase 2–7 code, so
these tests focus on the contracts the Phase 8 brief makes explicit:

- the demo may only reseed/mutate its own disposable graph;
- it never writes reports over the historical ``reports/CASE-*.md`` artifacts;
- the graph-browser helper never fabricates a deep link;
- scenario selection is explicit and validated.

DB-backed checks skip cleanly when FalkorDB is unreachable (matching the rest
of the suite); the full three-scenario run is exercised by the phase gate.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from src.demo import graph_link, run_demo
from src.demo.graph_link import build_graph_link, subgraph_cypher


# ---------------------------------------------------------------- graph guard


@pytest.mark.parametrize("bad", [
    "HABEAS_demo_local",          # the shared demo graph
    "HABEAS_demo_eval_main_s42",  # an evaluation graph
    "HABEAS_demo_test_net",       # the controlled test graph
    "HABEAS_demo",                # bare prefix
    "arbitrary_graph",            # arbitrary graph
    "",
])
def test_demo_refuses_non_showcase_graphs(bad):
    with pytest.raises(ValueError):
        run_demo.assert_demo_graph(bad)


def test_demo_accepts_its_own_disposable_namespace():
    run_demo.assert_demo_graph("HABEAS_demo_showcase_s42")
    assert run_demo.demo_graph_name(42) == "HABEAS_demo_showcase_s42"
    assert run_demo.demo_graph_name(7).startswith("HABEAS_demo_showcase")


def test_demo_namespace_is_inside_the_demo_safe_prefix():
    from src.config import DEMO_GRAPH_PREFIX

    assert run_demo.DEMO_GRAPH_NAMESPACE.startswith(DEMO_GRAPH_PREFIX)
    # and distinct from the shared demo graph and the evaluation namespace
    assert run_demo.DEMO_GRAPH_NAMESPACE != "HABEAS_demo_local"
    assert not run_demo.DEMO_GRAPH_NAMESPACE.startswith("HABEAS_demo_eval")


def test_demo_settings_enable_sandbox_actions_on_the_demo_graph(settings):
    demo = run_demo.demo_settings(
        settings, graph_name="HABEAS_demo_showcase_s42", mode="fake", tool_budget=9
    )
    assert demo.graph_name == "HABEAS_demo_showcase_s42"
    assert demo.habeas_demo_mode is True
    assert demo.habeas_enable_actions is True
    assert demo.is_sandbox is True
    assert demo.tool_budget == 9
    with pytest.raises(ValueError):
        run_demo.demo_settings(
            settings, graph_name="HABEAS_demo_local", mode="fake", tool_budget=None
        )


# ----------------------------------------------------------- no overwrite


def test_default_reports_dir_is_a_subdirectory_of_reports():
    default = Path(run_demo.DEFAULT_REPORTS_DIR)
    assert default.parts[0] == "reports"
    assert len(default.parts) > 1, "demo reports must not land in reports/ itself"


def test_main_refuses_to_write_reports_into_reports_root(capsys):
    """The guard fires before any graph is touched, so no DB is required."""
    code = run_demo.main(["--reports", "reports", "--no-reseed"])
    assert code == 2
    err = capsys.readouterr().err
    assert "refusing to write demo reports directly into reports/" in err


# ------------------------------------------------------------- link honesty


def test_graph_link_never_fabricates_a_deep_link():
    link = build_graph_link("HABEAS_demo_showcase_s42", "MATCH (n) RETURN n",
                            probe=False)
    assert link.deep_link is None
    assert link.graph_name == "HABEAS_demo_showcase_s42"
    assert link.browser_url.startswith("http")
    rendered = link.render()
    assert "not provided" in rendered
    assert "was not verified" in rendered
    # no query string / fragment that could look like a fabricated deep link
    assert "?" not in link.browser_url and "#" not in link.browser_url


def test_graph_link_reports_unreachable_browser_truthfully(monkeypatch):
    monkeypatch.setattr(graph_link, "probe_browser", lambda *a, **k: False)
    link = build_graph_link("HABEAS_demo_showcase_s42", "", probe=True)
    assert link.reachable is False
    assert "no FalkorDB Browser answered" in link.render()


def test_subgraph_cypher_is_bounded_and_lists_the_focal_accounts():
    cypher = subgraph_cypher(["ACC-0387", "ACC-0388"])
    assert "ACC-0387" in cypher and "ACC-0388" in cypher
    assert "LIMIT 200" in cypher
    assert "TRANSFERRED_TO" in cypher
    # empty input stays a valid, bounded query
    assert "LIMIT 200" in subgraph_cypher([])


# --------------------------------------------------------- scenario selection


def test_scenario_selection_is_explicit_and_validated():
    assert [s.key for s in run_demo._select_scenarios("all")] == ["A", "B", "C"]
    assert [s.key for s in run_demo._select_scenarios("")] == ["A", "B", "C"]
    assert [s.key for s in run_demo._select_scenarios("C,A")] == ["A", "C"]
    with pytest.raises(ValueError):
        run_demo._select_scenarios("Z")


def test_scenarios_target_the_documented_alerts():
    by_key = {s.key: s.alert_id for s in run_demo.SCENARIOS}
    assert by_key == {"A": "ALR-0001", "B": "ALR-0021", "C": "ALR-0004"}


def test_mode_resolution_auto_uses_real_only_when_configured():
    from src.config import Settings

    bare = Settings(llm_provider="groq", groq_api_key="", groq_model="m")
    assert run_demo.resolve_mode("auto", bare) == "fake"
    assert run_demo.resolve_mode("fake", bare) == "fake"
    assert run_demo.resolve_mode("real", bare) == "real"  # explicit request, not auto
    configured = Settings(llm_provider="groq", groq_api_key="k", groq_model="m")
    assert run_demo.resolve_mode("auto", configured) == "real"


# --------------------------------------- bounded real end-to-end (scenario A)


def test_scenario_a_runs_and_declines_on_a_disposable_graph(graph, settings, tmp_path):
    """Scenario A is non-destructive (it declines), so it is safe as a test.

    Proves the demo writes reports under the given directory and leaves the
    shared report root untouched.
    """
    graph_name = run_demo.demo_graph_name(42)
    demo = run_demo.demo_settings(
        settings, graph_name=graph_name, mode="fake", tool_budget=None
    )
    args = argparse.Namespace(
        scenarios="A", reports=str(tmp_path / "reports"),
        trace_dir=str(tmp_path / "traces"), no_reseed=False, seed=42,
        mode="fake", provider=None, tool_budget=None, graph=graph_name,
        json_out=None,
    )
    summaries = asyncio.run(run_demo._run_all(args, demo, run_demo.FakeLLM()))

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["alert_id"] == "ALR-0001"
    assert summary["action_status"] == "declined"
    assert summary["frozen"] is False
    assert summary["tool_calls"] >= 1  # real MCP calls happened
    # the report went to the requested directory, never reports/ root
    written = Path(summary["report_path"])
    assert written.exists()
    assert written.parent == (tmp_path / "reports")