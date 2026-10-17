"""Ground-truth isolation at the evaluation-to-agent boundary (Phase 7 §2).

The evaluation may read ``src/seed/ground_truth.json`` to score predictions;
the investigating agent must never receive it. These tests pin that contract:

- a missing, malformed, or tampered manifest is a clear evaluation error whose
  message never carries planted answers (check C);
- nothing manifest-derived except the alert id crosses the harness boundary
  into the workflow, and no ``src.agent``/``src.tools`` module imports the
  seed manifest or the evaluation package (check D);
- the MCP tool schemas exposed to the agent contain no planted ids or labels.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from eval import EvaluationConfigError
from eval.ground_truth import load_manifest
from eval.harness import AlertRunner, eval_graph_name, eval_settings
from src.agent.mcp_client import list_tools_async
from src.agent.state import CaseState
from src.seed.generate import build_model

REPO = Path(__file__).resolve().parents[1]

# Marker fragments that would leak planted answers if they ever crossed the
# boundary (ids from the seed-42 manifest; see load_manifest below).
_ID_FRAGMENT = re.compile(r"ACC-\d{4}|ALR-\d{4}|TXN-[A-Za-z0-9]+")


def _leak_check(message: str) -> None:
    match = _ID_FRAGMENT.search(message)
    assert match is None, f"error message leaks a planted id: {match!r}"


# ----------------------------- C. manifest failures --------------------------


def test_missing_manifest_is_a_clear_error_without_answers(tmp_path):
    with pytest.raises(EvaluationConfigError) as excinfo:
        load_manifest(tmp_path / "does_not_exist.json", expected_seed=42)
    message = str(excinfo.value)
    assert "not found" in message
    assert "src.seed.generate" in message or "regenerate" in message
    _leak_check(message)


def test_malformed_json_manifest_is_a_clear_error_without_answers(tmp_path):
    bad = tmp_path / "ground_truth.json"
    bad.write_text("{definitely not json", encoding="utf-8")
    with pytest.raises(EvaluationConfigError) as excinfo:
        load_manifest(bad, expected_seed=42)
    assert "not valid JSON" in str(excinfo.value)
    _leak_check(str(excinfo.value))


def test_schema_malformed_manifest_is_a_clear_error_without_answers(tmp_path):
    bad = tmp_path / "ground_truth.json"
    bad.write_text(json.dumps({"schema_version": 1, "seed": 42}),
                   encoding="utf-8")
    with pytest.raises(EvaluationConfigError) as excinfo:
        load_manifest(bad, expected_seed=42)
    message = str(excinfo.value)
    assert "malformed" in message
    # the message names the schema location, never the planted values
    _leak_check(message)


def test_wrong_seed_manifest_is_refused(tmp_path):
    manifest = build_model(seed=42, graph_name="HABEAS_demo_local").manifest
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(manifest.model_dump(mode="json")),
                    encoding="utf-8")
    with pytest.raises(EvaluationConfigError) as excinfo:
        load_manifest(path, expected_seed=43)
    message = str(excinfo.value)
    assert "seed 42" in message and "seed 43" in message


def test_tampered_manifest_is_refused_and_names_only_section_names(tmp_path):
    manifest = build_model(seed=42, graph_name="HABEAS_demo_local").manifest
    raw = manifest.model_dump(mode="json")
    tampered_account = raw["alerts"][0]["account_id"] + "9"
    raw["alerts"][0]["account_id"] = tampered_account
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(EvaluationConfigError) as excinfo:
        load_manifest(path, expected_seed=42)
    message = str(excinfo.value)
    assert "does not match the deterministic generator output" in message
    assert "alerts" in message
    # no answer values: not the tampered id, not any planted id
    assert tampered_account not in message
    _leak_check(message)


# --------------------- D. nothing crosses the boundary ----------------------


_IMPORT_LINE = re.compile(r"^\s*(?:from|import)\s+", re.MULTILINE)


def _module_import_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if _IMPORT_LINE.match(line)
    ]


@pytest.mark.parametrize("package", ["src/agent", "src/tools"])
def test_agent_and_tool_layers_never_import_the_seed_or_eval_layers(package):
    """Import-level isolation: agent/tools may not read the manifest or the
    evaluation package. (The trusted seed code itself is the only writer.)"""
    root = REPO / package
    modules = sorted(root.glob("*.py"))
    assert modules, f"no modules found under {package}"
    for module in modules:
        for line in _module_import_lines(module):
            assert "src.seed" not in line, f"{module.name}: {line}"
            assert re.match(r"(from|import)\s+eval", line) is None, (
                f"{module.name}: {line}")


def test_case_state_has_no_manifest_derived_fields():
    forbidden = {"gt_label", "gt_family", "gt_signals", "label", "family",
                 "truth", "expected_account_members", "pattern"}
    held = set(CaseState.model_fields)
    assert held.isdisjoint(forbidden), held & forbidden


def test_only_the_alert_id_crosses_into_the_workflow(settings, tmp_path):
    """Structural proof at the harness boundary: the workflow is invoked with
    exactly the alert id plus runtime configuration — never any manifest
    field, planted account, trigger transaction, or label."""
    manifest = load_manifest(expected_seed=42)
    truth = manifest.alerts[0]

    captured: dict = {}

    async def stub_workflow(alert_id, **kwargs):
        captured["alert_id"] = alert_id
        captured.update(kwargs)
        return None  # the runner records this as a failed run; we don't care

    graph_name = eval_graph_name(42, "tstiso")
    runner = AlertRunner(
        settings=eval_settings(settings, graph_name=graph_name, mode="fake"),
        graph_name=graph_name, seed=42, mode="fake", trace_dir=tmp_path,
        llm=None,  # the stub never uses it
        workflow=stub_workflow,
    )
    runner.run(truth)

    assert captured["alert_id"] == truth.alert_id
    assert set(captured) == {
        "alert_id", "settings", "llm", "trace_dir", "case_memory", "freeze",
    }
    # no planted identifier anywhere in what crossed the boundary
    blob = repr(captured)
    assert truth.account_id not in blob
    for member in truth.expected_account_members:
        assert member not in blob
    assert truth.trigger_transaction_id not in blob
    for signal in truth.signals:
        assert signal not in blob


def test_mcp_tool_schemas_carry_no_planted_ids_or_labels(settings):
    """The tool interface the agent (and any LLM) sees lists no planted
    account ids, alert ids, or ground-truth vocabulary."""
    manifest = load_manifest(expected_seed=42)
    planted = {
        account
        for pattern in manifest.patterns
        for account in pattern.account_members
    }
    for alert in manifest.alerts:
        planted.add(alert.account_id)
        planted.update(alert.expected_account_members)

    tools = asyncio.run(list_tools_async(
        settings, graph_name=settings.graph_name + "_test_net"))
    assert tools, "the MCP server exposed no tools"
    blob = json.dumps(tools)
    for planted_id in sorted(planted):
        assert planted_id not in blob, planted_id
    assert "ground_truth" not in blob
    assert "tempting_false_positives" not in blob
