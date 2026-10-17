"""Phase 7 evaluation-harness tests (PLAN.md §12).

Covers the required checks A, B, E, F, G, H, I, J plus an inventory guard for
K and L. Pure tests pin the deterministic metric semantics; DB-backed tests
run bounded but REAL investigations through the MCP stdio server on disposable
evaluation graphs (``HABEAS_demo_eval_*``), never the shared demo graph.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from eval import EvaluationConfigError, EvaluationError
from eval.ground_truth import family_denominators, load_manifest, manifest_summary
from eval.harness import (
    AlertRunner,
    _failed_record,
    assert_eval_graph,
    compare_runs,
    eval_graph_name,
    eval_settings,
    seed_eval_graph,
)
from eval.memory import aggregate_pairs, run_comparison, run_pair, select_repeat_pairs
from eval.metrics import (
    OUTCOME_ABSTENTION,
    OUTCOME_BENIGN,
    OUTCOME_FAILED,
    OUTCOME_SUPPORTED,
    OUTCOME_UNCONFIRMED,
    checklist_failures,
    compute_metrics,
    outcome_tier,
)
from eval.render import render_markdown
from eval.run import resolve_alert_truths
from src.agent.llm import FakeLLM
from src.agent.policy import (
    VERDICT_INSUFFICIENT_EVIDENCE,
    VERDICT_INCOMPLETE_INVESTIGATION,
    VERDICT_LIKELY_BENIGN,
    VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
    VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION,
)
from src.agent.state import (
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_EVIDENCE_SUFFICIENT,
    TERMINATION_UNRECOVERABLE_ERROR,
)
from src.config import Settings
from src.seed.patterns import (
    FAMILY_BENIGN,
    FAMILY_CYCLE,
    FAMILY_FAN_IN_OUT,
    AlertTruth,
    GroundTruth,
    PatternTruth,
)

REPO = Path(__file__).resolve().parents[1]

_CTX = {
    "seed": 42,
    "graph_name": "HABEAS_demo_eval_test_s42",
    "mode": "fake",
    "provider": None,
    "model": None,
    "case_memory": True,
}


def _truth(alert_id: str = "ALR-0001", label: str = "suspicious",
           family: str = FAMILY_CYCLE) -> AlertTruth:
    return AlertTruth(
        alert_id=alert_id, family=family, label=label,
        trigger_transaction_id="TXN-X", account_id="ACC-0387",
        expected_account_members=["ACC-0387", "ACC-0388"],
        signals=["circular_transfer"],
    )


def _record(alert_id: str, label: str, family: str = FAMILY_CYCLE, **overrides):
    """A minimal-but-complete per-alert record in the harness schema."""
    record = {
        "alert_id": alert_id,
        "gt_label": label,
        "gt_family": family,
        "gt_signals": ["circular_transfer"],
        "gt_account_id": "ACC-0387",
        "gt_expected_members": ["ACC-0387", "ACC-0388"],
        **_CTX,
        "elapsed_s": 1.0,
        "status": "ok",
        "error": None,
        "case_id": f"CASE-{alert_id}",
        "canonical_verdict": None,
        "investigation_verdict": None,
        "termination_reason": TERMINATION_EVIDENCE_SUFFICIENT,
        "completed": True,
        "tool_calls": 5,
        "tool_sequence": ["alert_context", "entity_profile"],
        "tool_steps": [
            {"tool": "alert_context", "arguments": {"alert_id": alert_id}},
        ],
        "signal_families": ["circulation"],
        "claims": {"supported": 1, "contradicted": 0, "unverified": 0},
        "hypotheses": [{"name": "cycle", "status": "supported"}],
        "limitations": [],
        "errors": [],
        "truncated_evidence": [],
        "prior_cases_used": False,
        "prior_case_leads": [],
        "reused_case_ids": [],
        "case_created": True,
        "action": {"attempted": False},
        "simulation": {"present": False},
        "post_action_verification": None,
        "verifier": {"status": "skipped"},
    }
    record.update(overrides)
    record["outcome_tier"] = outcome_tier(record)
    return record


# ============================== A. metric semantics ==========================


def test_outcome_tier_mapping_is_explicit():
    supported = _record("A1", "suspicious",
                        canonical_verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW)
    unconfirmed = _record("A2", "suspicious",
                          canonical_verdict=VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION)
    benign = _record("A3", "benign", canonical_verdict=VERDICT_LIKELY_BENIGN)
    abstain_1 = _record("A4", "suspicious",
                        canonical_verdict=VERDICT_INSUFFICIENT_EVIDENCE)
    abstain_2 = _record("A5", "suspicious",
                        canonical_verdict=VERDICT_INCOMPLETE_INVESTIGATION)
    no_verdict = _record("A6", "suspicious")
    failed_status = _record("A7", "suspicious", status="error", error="boom")
    failed_termination = _record(
        "A8", "suspicious", termination_reason=TERMINATION_UNRECOVERABLE_ERROR)

    assert outcome_tier(supported) == OUTCOME_SUPPORTED
    assert outcome_tier(unconfirmed) == OUTCOME_UNCONFIRMED
    assert outcome_tier(benign) == OUTCOME_BENIGN
    assert outcome_tier(abstain_1) == OUTCOME_ABSTENTION
    assert outcome_tier(abstain_2) == OUTCOME_ABSTENTION
    assert outcome_tier(no_verdict) == OUTCOME_ABSTENTION
    assert outcome_tier(failed_status) == OUTCOME_FAILED
    assert outcome_tier(failed_termination) == OUTCOME_FAILED


def test_recall_false_positives_and_confusion_matrix():
    records = [
        _record("A1", "suspicious",
                canonical_verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW),
        # suspicious_activity is deliberately NOT counted as strict recall
        _record("A2", "suspicious",
                canonical_verdict=VERDICT_SUSPICIOUS_REQUIRES_INVESTIGATION),
        # missed: an explicit abstention
        _record("A3", "suspicious",
                canonical_verdict=VERDICT_INSUFFICIENT_EVIDENCE),
        # false positive on a benign alert
        _record("B1", "benign",
                canonical_verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW),
        _record("B2", "benign", canonical_verdict=VERDICT_LIKELY_BENIGN),
    ]
    m = compute_metrics(records)

    assert m["counts"] == {
        "evaluated": 5, "suspicious": 3, "benign": 2, "families": 1,
    }
    assert (m["recall"]["strict"]["numerator"],
            m["recall"]["strict"]["denominator"]) == (1, 3)
    assert (m["recall"]["lenient"]["numerator"],
            m["recall"]["lenient"]["denominator"]) == (2, 3)
    assert (m["false_positive_rate"]["strict"]["numerator"],
            m["false_positive_rate"]["strict"]["denominator"]) == (1, 2)
    assert m["recall"]["strict"]["display"] == "1/3 (33.3%)"

    confusion = m["confusion"]
    assert confusion["columns"] == [
        OUTCOME_SUPPORTED, OUTCOME_UNCONFIRMED, OUTCOME_BENIGN,
        OUTCOME_ABSTENTION, OUTCOME_FAILED,
    ]
    suspicious_row = confusion["rows"]["suspicious"]
    assert (suspicious_row[OUTCOME_SUPPORTED],
            suspicious_row[OUTCOME_UNCONFIRMED],
            suspicious_row[OUTCOME_ABSTENTION],
            suspicious_row["total"]) == (1, 1, 1, 3)
    benign_row = confusion["rows"]["benign"]
    assert (benign_row[OUTCOME_SUPPORTED],
            benign_row[OUTCOME_BENIGN], benign_row["total"]) == (1, 1, 2)

    # the false positive and the miss are listed, never folded into "correct"
    kinds = {item["kind"] for item in checklist_failures(records)}
    assert "false_positive_on_benign" in kinds
    assert "missed_suspicious_alert" in kinds


def test_budget_exhaustion_is_visible_not_completed():
    records = [_record("A1", "suspicious",
                       termination_reason=TERMINATION_BUDGET_EXHAUSTED,
                       completed=False)]
    m = compute_metrics(records)
    assert m["completion"]["tool_budget_exhausted"]["numerator"] == 1
    assert m["completion"]["completed"]["numerator"] == 0


# ==================== B. pattern denominators from the manifest ==============


def test_denominators_are_derived_from_the_validated_manifest():
    manifest = load_manifest(expected_seed=42)
    den = family_denominators(manifest)
    suspicious = [a for a in manifest.alerts if a.label == "suspicious"]

    assert den["alert_total"] == len(manifest.alerts)
    assert den["suspicious_total"] == len(suspicious)
    assert den["benign_total"] == len(manifest.alerts) - len(suspicious)
    assert set(den["pattern_families"]) == {p.family for p in manifest.patterns}
    for family, ids in den["suspicious_by_family"].items():
        assert ids == sorted(
            a.alert_id for a in suspicious if a.family == family
        )
    # and the stored summary reproduces the same denominators for results.json
    summary = manifest_summary(manifest)
    assert summary["suspicious_by_family"] == den["suspicious_by_family"]
    assert summary["pattern_account_members"] == den["pattern_account_members"]


def test_metric_family_denominator_follows_the_records():
    records = [
        _record("A1", "suspicious", family=FAMILY_CYCLE,
                canonical_verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW),
        _record("A2", "suspicious", family=FAMILY_FAN_IN_OUT,
                canonical_verdict=VERDICT_INSUFFICIENT_EVIDENCE),
    ]
    m = compute_metrics(records)
    assert m["counts"]["families"] == 2
    assert set(m["pattern_coverage"]["by_family"]) == {
        FAMILY_CYCLE, FAMILY_FAN_IN_OUT,
    }
    # per-family detection rates use the per-family record counts (1 each)
    assert m["pattern_coverage"]["by_family"][FAMILY_CYCLE][
        "detection_rate"]["numerator"] == 1
    assert m["pattern_coverage"]["by_family"][FAMILY_FAN_IN_OUT][
        "detection_rate"]["numerator"] == 0


def _gt(alerts, patterns) -> GroundTruth:
    return GroundTruth(
        schema_version=1, seed=7, graph_name="HABEAS_demo_eval_x_s7",
        counts={"alerts": len(alerts)}, alerts=alerts, patterns=patterns,
        tempting_false_positives=[],
    )


def test_inconsistent_manifests_are_rejected_not_scored():
    suspicious = _truth("ALR-9001", "suspicious", FAMILY_CYCLE)
    benign = _truth("ALR-9002", "benign", FAMILY_BENIGN)
    # a suspicious family with no planted pattern
    with pytest.raises(EvaluationConfigError) as excinfo:
        family_denominators(_gt([suspicious, benign], patterns=[]))
    assert "no planted pattern" in str(excinfo.value)
    # a planted family with no suspicious alerts (the cycle pattern covers the
    # suspicious alert, so only the uncovered fan_in_out family is the problem)
    with pytest.raises(EvaluationConfigError) as excinfo:
        family_denominators(_gt(
            [suspicious, benign],
            patterns=[
                PatternTruth(family=FAMILY_CYCLE, account_members=["ACC-0387"]),
                PatternTruth(family=FAMILY_FAN_IN_OUT, account_members=["ACC-B"]),
            ],
        ))
    assert "no suspicious alerts" in str(excinfo.value)
    # the benign label can never be a planted pattern family
    with pytest.raises(EvaluationConfigError) as excinfo:
        family_denominators(_gt(
            [suspicious, benign],
            patterns=[PatternTruth(family=FAMILY_BENIGN, account_members=[])],
        ))
    assert "benign" in str(excinfo.value)


# ================================ E. reproducibility =========================


def test_semantic_comparison_ignores_timing_and_finds_field_changes():
    base = _record("A1", "suspicious",
                   canonical_verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW)
    same_but_slower = dict(base, elapsed_s=999.0)
    verdict_changed = dict(base, canonical_verdict=VERDICT_LIKELY_BENIGN)

    assert compare_runs([base], [same_but_slower])["identical"] is True
    cmp = compare_runs([base], [verdict_changed])
    assert cmp["identical"] is False
    assert cmp["mismatched_alerts"] == [
        {"alert_id": "A1", "fields": ["canonical_verdict"]}
    ]
    assert compare_runs([base], [base])["digest_first"] == \
        compare_runs([base], [base])["digest_second"]


def test_repeat_seed_fakellm_sweep_is_reproducible(settings, tmp_path):
    """Two bounded sweeps on a REBUILT graph produce identical semantics."""
    manifest = load_manifest(expected_seed=42)
    truths = resolve_alert_truths("2", manifest)
    graph_name = eval_graph_name(42, "tstrepro")
    run_settings = eval_settings(settings, graph_name=graph_name, mode="fake")

    digests = []
    for _ in range(2):
        seed_eval_graph(
            run_settings, seed=42, graph_name=graph_name,
            manifest_path=tmp_path / "manifest_repro.json",
        )
        runner = AlertRunner(
            settings=run_settings, graph_name=graph_name, seed=42,
            mode="fake", trace_dir=tmp_path, llm=FakeLLM(),
        )
        records = [runner.run(truth) for truth in truths]
        digests.append(compare_runs(records, records)["digest_first"])
    assert digests[0] == digests[1]


# ======================== F/G. paired memory comparison ======================


class _FakeRunner:
    """Stands in for AlertRunner so run_pair formulas are testable offline."""

    instances: list["_FakeRunner"] = []
    records_by_alert: dict[str, dict] = {}

    def __init__(self, *, settings, graph_name, seed, mode, llm, trace_dir,
                 case_memory=True, verify=True, **_ignored):
        self.settings = settings
        self.graph_name = graph_name
        self.case_memory = case_memory
        _FakeRunner.instances.append(self)

    def run(self, truth):
        record = _FakeRunner.records_by_alert[truth.alert_id]
        if record.get("_repeat"):
            return record["_on"] if self.case_memory else record["_off"]
        return record


@pytest.fixture()
def fake_memory_env(monkeypatch):
    _FakeRunner.instances = []
    _FakeRunner.records_by_alert = {}
    monkeypatch.setattr("eval.memory.AlertRunner", _FakeRunner)
    monkeypatch.setattr("eval.memory.seed_eval_graph", lambda *_a, **_kw: None)
    return _FakeRunner


def _pair_truths(warmup_id: str, repeat_id: str):
    return {
        "warmup": _truth(warmup_id, "suspicious", FAMILY_CYCLE),
        "repeat": _truth(repeat_id, "suspicious", FAMILY_CYCLE),
        "selection_rule": "shared_family_members",
    }


def _memory_records(*, warmup_steps, on_steps, off_steps, leads, reuses,
                    verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
                    verifier_summary=None):
    def _repeat(steps):
        return _record(
            "ALR-R", "suspicious", tool_calls=len(steps),
            tool_sequence=[s["tool"] for s in steps], tool_steps=steps,
            prior_case_leads=list(leads), reused_case_ids=list(reuses),
            prior_cases_used=any(s["tool"] == "prior_cases" for s in steps),
            canonical_verdict=verdict,
            verifier=verifier_summary or {"status": "skipped"},
        )

    return {
        "ALR-W": _record("ALR-W", "suspicious", tool_calls=len(warmup_steps),
                         tool_sequence=[s["tool"] for s in warmup_steps],
                         tool_steps=warmup_steps),
        "ALR-R": {
            "_repeat": True,
            "_on": _repeat(on_steps),
            "_off": _repeat(off_steps),
        },
    }


def test_memory_formulas_count_recorded_steps_only(fake_memory_env):
    fake_memory_env.records_by_alert = _memory_records(
        warmup_steps=[{"tool": "alert_context", "arguments": {}}],
        # memory-on repeat: prior_cases inserted; every off step still re-derived
        on_steps=[
            {"tool": "alert_context", "arguments": {}},
            {"tool": "prior_cases", "arguments": {"entity_id": "ACC-0387"}},
            {"tool": "entity_profile", "arguments": {}},
            {"tool": "trace_funds", "arguments": {}},
        ],
        off_steps=[
            {"tool": "alert_context", "arguments": {}},
            {"tool": "entity_profile", "arguments": {}},
            {"tool": "trace_funds", "arguments": {}},
        ],
        leads=["CASE-ALR-W"], reuses=["CASE-ALR-W"],
        verifier_summary={"status": "ran", "summary": {"mismatch": 0}},
    )
    pair = run_pair(_pair_truths("ALR-W", "ALR-R"), index=9,
                    base_settings=Settings(), seed=42,
                    trace_root=Path("unused"), verify=True)
    measures = pair["repeat_measures"]

    assert pair["warmup_equivalent"] is True
    assert (measures["calls_on"], measures["calls_off"]) == (4, 3)
    assert measures["net_call_delta"] == 1  # the extra prior_cases call
    assert measures["prior_lookup_calls_on"] == 1
    assert measures["calls_avoided"] == 0  # nothing was skipped
    assert measures["reduction_pct"] == round((3 - 4) / 3, 4)  # negative: cost
    assert pair["conditions"]["on"]["case_memory"] is True
    assert pair["conditions"]["off"]["case_memory"] is False
    assert pair["tool_budget"] == 12  # from the Settings default
    # reused leads were validated: nothing skipped, verifier replay clean
    assert measures["reused_findings_validated"] is True

    aggregate = aggregate_pairs([pair])
    assert aggregate["pairs"] == 1
    assert aggregate["net_call_delta_total"] == 1
    assert aggregate["avoided_calls_total"] == 0
    # honesty: a cost is reported as a cost, never spun as a saving
    assert aggregate["conclusion"] == (
        "memory introduced 1 additional prior-case lookup(s) and avoided none "
        "of the recorded steps across 1 pair(s); no tool-call saving was "
        "measured in this configuration"
    )


def test_missing_prior_cases_produce_no_fabricated_benefit(fake_memory_env):
    fake_memory_env.records_by_alert = _memory_records(
        warmup_steps=[{"tool": "alert_context", "arguments": {}}],
        on_steps=[{"tool": "alert_context", "arguments": {}},
                  {"tool": "entity_profile", "arguments": {}}],
        off_steps=[{"tool": "alert_context", "arguments": {}},
                   {"tool": "entity_profile", "arguments": {}}],
        leads=[], reuses=[],  # no prior case existed
    )
    pair = run_pair(_pair_truths("ALR-W", "ALR-R"), index=3,
                    base_settings=Settings(), seed=42,
                    trace_root=Path("unused"), verify=True)
    measures = pair["repeat_measures"]
    assert measures["calls_avoided"] == 0
    assert measures["prior_lookup_calls_on"] == 0
    assert measures["reused_findings_validated"] is False
    assert aggregate_pairs([pair])["validated_pairs"] == 0


def test_memory_comparison_is_paired_on_real_graphs(settings, tmp_path):
    """DB-backed bounded comparison: identical warm-up, only the tool differs."""
    manifest = load_manifest(expected_seed=42)
    base = settings.model_copy(update={"tool_budget": 10})
    block = run_comparison(
        manifest=manifest, base_settings=base, seed=42,
        trace_root=tmp_path, max_pairs=1,
    )
    assert block["aggregate"]["pairs"] == 1
    pair = block["pairs"][0]
    assert pair["tool_budget"] == 10
    assert pair["warmup_equivalent"] is True, (
        "warm-up investigations must be identical before the only difference "
        "is introduced"
    )
    on = pair["conditions"]["on"]
    off = pair["conditions"]["off"]
    assert on["case_memory"] is True and off["case_memory"] is False
    assert on["graph_name"] != off["graph_name"]
    # memory-off really does not see the prior_cases tool
    assert "prior_cases" not in off["repeat"]["tool_sequence"]
    measures = pair["repeat_measures"]
    assert measures["net_call_delta"] == (
        measures["calls_on"] - measures["calls_off"])
    # verdicts are re-derived in both conditions — memory is a lead, not truth
    assert measures["verdicts"]["on_off_agree"] is True
    if measures["calls_avoided"] == 0:
        conclusion = block["aggregate"]["conclusion"]
        assert "no tool-call saving" in conclusion or (
            "additional prior-case lookup" in conclusion)


# ============================ H. failures are reported =======================


def test_failed_runs_stay_in_denominators_with_reasons():
    truth = _truth("ALR-ERR", "suspicious")
    failed = _failed_record(truth, _CTX, "RuntimeError: harness boom")
    ok = _record("ALR-0001", "suspicious",
                 canonical_verdict=VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW)

    assert outcome_tier(failed) == OUTCOME_FAILED
    assert failed["errors"] == ["RuntimeError: harness boom"]
    assert failed["tool_calls"] == 0

    m = compute_metrics([ok, failed])
    assert m["failure_rate"]["numerator"] == 1
    assert m["failure_rate"]["denominator"] == 2
    assert m["completion"]["early_termination"]["numerator"] == 1
    assert m["counts"]["evaluated"] == 2  # never dropped from the denominator

    items = checklist_failures([ok, failed])
    problem = next(i for i in items if i["kind"] == "run_problem"
                   and i["alert_id"] == "ALR-ERR")
    assert problem["harness_error"] == "RuntimeError: harness boom"

    provider_down = _record(
        "ALR-PROV", "suspicious",
        termination_reason=TERMINATION_UNRECOVERABLE_ERROR,
        errors=["LLM provider failure: provider call failed (TimeoutError)"],
    )
    assert outcome_tier(provider_down) == OUTCOME_FAILED
    items = checklist_failures([provider_down])
    problem = next(i for i in items if i["kind"] == "run_problem")
    assert any("provider failure" in r for r in problem["errors"])


def test_truncated_evidence_is_reported_not_hidden():
    truncated = _record("ALR-T", "suspicious",
                        truncated_evidence=["EV-0007"],
                        limitations=["one or more tool results were truncated"])
    items = checklist_failures([truncated])
    notice = next(i for i in items if i["kind"] == "truncation_notice")
    assert notice["truncated_evidence"] == ["EV-0007"]


# ============================ I. evaluation graph isolation ==================


@pytest.mark.parametrize("bad", [
    "HABEAS_demo_local",      # the shared demo graph
    "HABEAS_demo",            # bare demo prefix
    "HABEAS_demo_test_net",   # the controlled test graph
    "my_graph",               # arbitrary graph
    "",
])
def test_assert_eval_graph_refuses_everything_but_eval_graphs(bad):
    with pytest.raises(EvaluationError):
        assert_eval_graph(bad)


def test_assert_eval_graph_accepts_eval_graphs():
    assert_eval_graph("HABEAS_demo_eval_main_s42")


def test_eval_graph_names_are_validated():
    assert eval_graph_name(42, "main") == "HABEAS_demo_eval_main_s42"
    for bad_tag in ("Main", "with-dash", "with space", "with.dot", ""):
        with pytest.raises(EvaluationError):
            eval_graph_name(42, bad_tag)


def test_eval_settings_refuse_non_evaluation_graphs(settings):
    with pytest.raises(EvaluationError):
        eval_settings(settings, graph_name="HABEAS_demo_local")
    with pytest.raises(EvaluationError):
        eval_settings(settings, graph_name="HABEAS_demo_test_net")


def test_alert_runner_refuses_non_evaluation_graphs(settings):
    with pytest.raises(EvaluationError):
        AlertRunner(settings=settings, graph_name="HABEAS_demo_local",
                    seed=42, mode="fake", trace_dir=Path("unused"),
                    llm=FakeLLM())


def test_seed_eval_graph_requires_matching_settings(settings):
    local = settings.model_copy(update={"graph_name": "HABEAS_demo_local"})
    with pytest.raises(EvaluationError):
        seed_eval_graph(local, seed=42,
                        graph_name="HABEAS_demo_eval_main_s42")


# ============================ J. scorecard integrity =========================


def test_scorecard_is_recomputable_from_results_json():
    path = REPO / "eval" / "results.json"
    if not path.exists():
        pytest.skip("eval/results.json has not been generated yet")
    results = json.loads(path.read_text(encoding="utf-8"))
    # every displayed number is recomputed from the per-alert records
    assert compute_metrics(results["alerts"]) == results["metrics"]

    markdown = render_markdown(results)
    metrics = results["metrics"]
    strict = metrics["recall"]["strict"]
    assert f"{strict['numerator']}/{strict['denominator']}" in markdown
    if strict["rate"] is not None:
        assert f"{100.0 * strict['rate']:.1f}%" in markdown
    counts = metrics["counts"]
    assert f"**{counts['evaluated']}** alert(s)" in markdown
    # the pattern diagnostics blocks exist for every planted family
    for family in results["run"]["manifest"]["suspicious_by_family"]:
        assert family in markdown


# ===================== K/L. prior-phase regressions remain ===================


def test_policy_v2_and_report_regressions_remain_in_the_suite():
    """The Phase 7 spec's K and L checks: the named regression scenarios from
    Policy v2, reports and claim verification must still exist and run as part
    of the suite (their pass/fail is asserted by pytest itself)."""
    policy_tests = importlib.import_module("test_policy_provenance")
    for name in (
        "test_a_overlapping_families_count_once",
        "test_b_identical_observations_are_deduplicated_not_double_counted",
        "test_c_partial_overlap_is_handled_by_the_documented_rule",
        "test_d_distinct_target_relevant_families_can_qualify",
        "test_e_sibling_only_signal_cannot_authorize_the_target",
        "test_h_policy_is_independent_of_llm_confidence_and_ground_truth",
        "test_k_historical_alr0001_record_and_reports_stay_intact",
    ):
        assert callable(getattr(policy_tests, name, None)), name

    report_tests = importlib.import_module("test_report")
    for name in (
        "test_d_tampered_result_hash_is_detected",
        "test_j_truncated_evidence_is_qualified_never_fully_reproduced",
        "test_m_report_and_verifier_never_mutate_the_graph",
    ):
        assert callable(getattr(report_tests, name, None)), name

    claim_tests = importlib.import_module("test_claim_verification")
    for name in (
        "test_f_valid_but_irrelevant_queries_cannot_establish_the_claim",
        "test_h_absence_is_never_reported_as_a_contradiction",
    ):
        assert callable(getattr(claim_tests, name, None)), name


# ====================== pair selection is data-driven ========================


def test_repeat_pairs_are_selected_from_the_manifest_without_hardcoding():
    manifest = load_manifest(expected_seed=42)
    pairs = select_repeat_pairs(manifest)
    assert pairs, "the manifest must yield at least one repeat pair"
    seen: set[str] = set()
    for pair in pairs:
        warmup, repeat = pair["warmup"], pair["repeat"]
        assert warmup.alert_id < repeat.alert_id
        for alert in (warmup, repeat):
            assert alert.alert_id not in seen  # each alert is consumed once
            seen.add(alert.alert_id)
        if pair["selection_rule"] == "same_alerted_account":
            assert warmup.account_id == repeat.account_id
        else:
            assert warmup.family == repeat.family
            assert set(warmup.expected_account_members) & set(
                repeat.expected_account_members)
    # bounded selection honours max_pairs
    assert len(select_repeat_pairs(manifest, max_pairs=2)) == 2


def test_dataset_counts_come_from_the_manifest_not_constants():
    manifest = load_manifest(expected_seed=42)
    truths = resolve_alert_truths("all", manifest)
    assert len(truths) == len(manifest.alerts)
    first_two = resolve_alert_truths("2", manifest)
    assert len(first_two) == 2
    explicit = resolve_alert_truths(
        f"{manifest.alerts[0].alert_id},{manifest.alerts[1].alert_id}", manifest)
    assert [t.alert_id for t in explicit] == [
        manifest.alerts[0].alert_id, manifest.alerts[1].alert_id]
    with pytest.raises(EvaluationConfigError):
        resolve_alert_truths("ALR-DOES-NOT-EXIST", manifest)
