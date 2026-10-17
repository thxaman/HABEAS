"""Controlled memory-on vs memory-off paired comparison (PLAN.md §10).

The comparison is *paired*: for every repeat-investigation scenario the same
disposable graph is seeded from the same seed for both conditions, the same
warm-up investigation runs first on each (creating identical prior-case
history — the warm-up semantic records are compared to prove equivalence), and
only then does the repeat investigation differ in exactly one way:

- memory ON  — the agent's tool interface includes ``prior_cases``;
- memory OFF — ``prior_cases`` is withheld from the interface
  (``run_investigation(..., case_memory=False)``); every other tool behaves
  identically, and the profile's pre-existing ``prior_case_count`` statistic
  is unchanged (see the note below).

Measures (all reported with explicit denominators/formulas):

- tool calls of the repeat investigation under each condition;
- calls attributable to the prior-case lookup (``prior_cases`` occurrences in
  the memory-on repeat);
- calls avoided: steps recorded in the memory-off repeat whose (tool,
  arguments) pair does not appear in the memory-on repeat;
- net delta and reduction percentage over the paired denominators;
- wall-clock per condition (descriptive only — small sample, includes MCP
  server startup; never presented as a benchmark);
- whether reused leads were validated: REUSES links recorded, no verification
  step skipped, per-case verifier replay clean, verdict re-derived locally.

No alert id or account id is special-cased anywhere; pairs are selected from
the validated manifest (same alerted account first, then same planted family
with overlapping expected members — e.g. the ring).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from src.agent.llm import FakeLLM
from src.config import Settings
from src.seed.patterns import FAMILY_BENIGN, AlertTruth, GroundTruth

from eval import EvaluationError
from eval.harness import (
    AlertRunner,
    assert_eval_graph,
    eval_graph_name,
    eval_settings,
    seed_eval_graph,
    semantic_view,
)


def select_repeat_pairs(
    manifest: GroundTruth, *, max_pairs: int | None = None
) -> list[dict[str, Any]]:
    """Select repeat-investigation candidate pairs from the manifest alone.

    Preference order for the partner of each earlier alert: (1) a later alert
    about the *same alerted account*; (2) a later alert in the same planted
    family whose expected account members overlap (the ring scenario). Both
    alerts are consumed once. Deterministic; no hard-coded ids.
    """
    alerts = sorted(manifest.alerts, key=lambda a: a.alert_id)
    consumed: set[str] = set()
    pairs: list[dict[str, Any]] = []

    def later(a: AlertTruth):
        return [b for b in alerts if b.alert_id > a.alert_id and b.alert_id not in consumed]

    for alert in alerts:
        if alert.alert_id in consumed:
            continue
        partner: AlertTruth | None = None
        rule = ""
        for candidate in later(alert):
            if candidate.account_id == alert.account_id:
                partner, rule = candidate, "same_alerted_account"
                break
        if partner is None and alert.family != FAMILY_BENIGN:
            for candidate in later(alert):
                if candidate.family == alert.family and (
                    set(candidate.expected_account_members)
                    & set(alert.expected_account_members)
                ):
                    partner, rule = candidate, "shared_family_members"
                    break
        if partner is not None:
            pairs.append(
                {"warmup": alert, "repeat": partner, "selection_rule": rule}
            )
            consumed.add(alert.alert_id)
            consumed.add(partner.alert_id)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def _step_keys(record: dict[str, Any]) -> Counter:
    keys: Counter = Counter()
    for step in record.get("tool_steps") or []:
        arguments = json.dumps(step.get("arguments") or {}, sort_keys=True, default=str)
        keys[(step.get("tool"), arguments)] += 1
    return keys


def _failed_condition(alert_id: str, error: str) -> dict[str, Any]:
    return {
        "alert_id": alert_id,
        "status": "error",
        "error": error,
        "tool_calls": 0,
        "tool_steps": [],
        "tool_sequence": [],
        "prior_case_leads": [],
        "reused_case_ids": [],
        "canonical_verdict": None,
        "verifier": {"status": "skipped"},
        "elapsed_s": 0.0,
    }


def run_pair(
    pair: dict[str, Any],
    *,
    index: int,
    base_settings: Settings,
    seed: int,
    trace_root: Path,
    verify: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one pair under both conditions on two freshly seeded graphs."""
    conditions: dict[str, Any] = {}
    for condition, memory_flag in (("on", True), ("off", False)):
        tag = f"mem{index}_{condition}"
        graph_name = eval_graph_name(seed, tag)
        settings = eval_settings(
            base_settings,
            graph_name=graph_name,
            mode="fake",
            tool_budget=base_settings.tool_budget,
        )
        if progress:
            progress(f"memory pair {index} [{condition}]: (re)seeding {graph_name}")
        seed_eval_graph(
            settings,
            seed=seed,
            graph_name=graph_name,
            manifest_path=Path(trace_root) / f"manifest_{graph_name}.json",
        )
        # The warm-up is identical for both conditions: memory is irrelevant
        # on a graph with no prior cases, which is exactly why it can serve as
        # the equivalence anchor.
        warm_runner = AlertRunner(
            settings=settings, graph_name=graph_name, seed=seed, mode="fake",
            llm=FakeLLM(), trace_dir=trace_root, case_memory=True, verify=verify,
        )
        if progress:
            progress(f"memory pair {index} [{condition}]: warm-up {pair['warmup'].alert_id}")
        warmup = warm_runner.run(pair["warmup"])
        repeat_runner = AlertRunner(
            settings=settings, graph_name=graph_name, seed=seed, mode="fake",
            llm=FakeLLM(), trace_dir=trace_root, case_memory=memory_flag,
            verify=verify,
        )
        if progress:
            progress(f"memory pair {index} [{condition}]: repeat {pair['repeat'].alert_id}")
        repeat = repeat_runner.run(pair["repeat"])
        conditions[condition] = {
            "graph_name": graph_name,
            "case_memory": memory_flag,
            "warmup": warmup,
            "repeat": repeat,
        }

    on = conditions["on"]["repeat"]
    off = conditions["off"]["repeat"]
    warm_on = conditions["on"]["warmup"]
    warm_off = conditions["off"]["warmup"]

    warmup_equivalent = semantic_view(warm_on) == semantic_view(warm_off)
    calls_on = int(on.get("tool_calls") or 0)
    calls_off = int(off.get("tool_calls") or 0)
    on_steps = _step_keys(on)
    off_steps = _step_keys(off)
    avoided = off_steps - on_steps
    avoided_list = sorted(
        (
            {"tool": tool, "arguments": json.loads(arguments), "count": count}
            for (tool, arguments), count in avoided.items()
        ),
        key=lambda item: (item["tool"], json.dumps(item["arguments"], sort_keys=True)),
    )
    calls_avoided = sum(item["count"] for item in avoided_list)
    prior_lookups = sum(1 for step in on.get("tool_steps") or []
                        if step.get("tool") == "prior_cases")

    verifier = on.get("verifier") or {}
    verifier_summary = verifier.get("summary") or {}
    leads = list(on.get("prior_case_leads") or [])
    reuses = list(on.get("reused_case_ids") or [])
    validated = bool(
        leads
        and reuses
        and calls_avoided == 0
        and verifier.get("status") == "ran"
        and verifier_summary.get("mismatch", 0) == 0
        and on.get("canonical_verdict")
    )

    return {
        "pair_id": f"mem{index}",
        "family": pair["warmup"].family,
        "selection_rule": pair["selection_rule"],
        "warmup_alert": pair["warmup"].alert_id,
        "repeat_alert": pair["repeat"].alert_id,
        "tool_budget": int(base_settings.tool_budget),
        "warmup_equivalent": warmup_equivalent,
        "conditions": {
            condition: {
                "graph_name": data["graph_name"],
                "case_memory": data["case_memory"],
                "warmup": {
                    "status": data["warmup"].get("status"),
                    "canonical_verdict": data["warmup"].get("canonical_verdict"),
                    "tool_calls": data["warmup"].get("tool_calls"),
                    "elapsed_s": data["warmup"].get("elapsed_s"),
                },
                "repeat": {
                    "alert_id": data["repeat"].get("alert_id"),
                    "status": data["repeat"].get("status"),
                    "canonical_verdict": data["repeat"].get("canonical_verdict"),
                    "termination_reason": data["repeat"].get("termination_reason"),
                    "tool_calls": data["repeat"].get("tool_calls"),
                    "tool_sequence": data["repeat"].get("tool_sequence"),
                    "prior_case_leads": data["repeat"].get("prior_case_leads"),
                    "reused_case_ids": data["repeat"].get("reused_case_ids"),
                    "verifier": data["repeat"].get("verifier"),
                    "elapsed_s": data["repeat"].get("elapsed_s"),
                    "error": data["repeat"].get("error"),
                },
            }
            for condition, data in conditions.items()
        },
        "repeat_measures": {
            "calls_on": calls_on,
            "calls_off": calls_off,
            "net_call_delta": calls_on - calls_off,
            "prior_lookup_calls_on": prior_lookups,
            "calls_avoided": calls_avoided,
            "avoided_steps": avoided_list,
            "reduction_pct": (
                round((calls_off - calls_on) / calls_off, 4) if calls_off else None
            ),
            "wall_clock_on_s": on.get("elapsed_s"),
            "wall_clock_off_s": off.get("elapsed_s"),
            "verdicts": {
                "warmup": conditions["on"]["warmup"].get("canonical_verdict"),
                "on": on.get("canonical_verdict"),
                "off": off.get("canonical_verdict"),
                "on_off_agree": on.get("canonical_verdict") == off.get("canonical_verdict"),
            },
            "reused_findings_validated": validated,
            "validation_evidence": {
                "prior_case_leads": leads,
                "reuses_links": reuses,
                "verification_steps_skipped": calls_avoided,
                "repeat_verifier_status": verifier_summary,
                "repeat_verdict": on.get("canonical_verdict"),
            },
        },
    }


def aggregate_pairs(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired measures with explicit formulas and denominators."""
    calls_on = sum(p["repeat_measures"]["calls_on"] for p in pair_results)
    calls_off = sum(p["repeat_measures"]["calls_off"] for p in pair_results)
    prior_lookups = sum(p["repeat_measures"]["prior_lookup_calls_on"] for p in pair_results)
    avoided = sum(p["repeat_measures"]["calls_avoided"] for p in pair_results)
    wall_on = round(sum((p["repeat_measures"]["wall_clock_on_s"] or 0) for p in pair_results), 3)
    wall_off = round(sum((p["repeat_measures"]["wall_clock_off_s"] or 0) for p in pair_results), 3)
    if avoided > 0:
        conclusion = (
            f"memory avoided {avoided} repeat-investigation call(s) across "
            f"{len(pair_results)} pair(s) — measured, not assumed"
        )
    elif calls_on > calls_off:
        conclusion = (
            f"memory introduced {calls_on - calls_off} additional prior-case "
            f"lookup(s) and avoided none of the recorded steps across "
            f"{len(pair_results)} pair(s); no tool-call saving was measured in "
            "this configuration"
        )
    elif calls_on == calls_off:
        conclusion = (
            f"no measured tool-call difference between memory on and off across "
            f"{len(pair_results)} pair(s)"
        )
    else:
        conclusion = (
            f"memory reduced repeat-investigation calls by {calls_off - calls_on} "
            f"across {len(pair_results)} pair(s); note that reductions without a "
            "recorded avoided step would need manual review"
        )
    return {
        "pairs": len(pair_results),
        "calls_on_total": calls_on,
        "calls_off_total": calls_off,
        "net_call_delta_total": calls_on - calls_off,
        "prior_lookup_calls_total": prior_lookups,
        "avoided_calls_total": avoided,
        "reduction_pct": round((calls_off - calls_on) / calls_off, 4) if calls_off else None,
        "wall_clock_on_total_s": wall_on,
        "wall_clock_off_total_s": wall_off,
        "wall_clock_note": (
            "descriptive only: small sample, includes MCP server startup and "
            "tool execution; not a performance benchmark"
        ),
        "warmups_equivalent": all(p["warmup_equivalent"] for p in pair_results),
        "validated_pairs": sum(
            1 for p in pair_results if p["repeat_measures"]["reused_findings_validated"]
        ),
        "conclusion": conclusion,
    }


def run_comparison(
    *,
    manifest: GroundTruth,
    base_settings: Settings,
    seed: int,
    trace_root: Path,
    max_pairs: int | None = None,
    verify: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the full paired comparison and return the reportable block."""
    pairs = select_repeat_pairs(manifest, max_pairs=max_pairs)
    if not pairs:
        raise EvaluationError(
            "no repeat-investigation pairs could be derived from the manifest; "
            "the paired memory comparison cannot run"
        )
    pair_results = [
        run_pair(
            pair, index=index, base_settings=base_settings, seed=seed,
            trace_root=trace_root, verify=verify, progress=progress,
        )
        for index, pair in enumerate(pairs, start=1)
    ]
    aggregate = aggregate_pairs(pair_results)
    return {
        "method": (
            "paired comparison on disposable graphs rebuilt from the same seed per "
            "condition; identical warm-up investigation; the only difference is "
            "whether the prior_cases tool is part of the agent's interface. The "
            "entity_profile tool's pre-existing prior_case_count statistic is "
            "unchanged in both conditions (documented limitation)."
        ),
        "formulas": {
            "net_call_delta": "calls(memory on) - calls(memory off), summed over pairs",
            "prior_lookup_calls": "occurrences of prior_cases in the memory-on repeat tool sequence",
            "calls_avoided": (
                "multiset difference of (tool, arguments) steps: steps in the "
                "memory-off repeat absent from the memory-on repeat"
            ),
            "reduction_pct": "(calls_off_total - calls_on_total) / calls_off_total",
        },
        "pairs": pair_results,
        "aggregate": aggregate,
    }
