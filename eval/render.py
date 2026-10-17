"""Markdown scorecard rendering — strictly derived from results.json.

``render_markdown(results)`` is pure: every number shown comes from the
structured results, so the rendered scorecard and the stored metrics can
never drift (regression-tested by recomputing metrics from the same records
and comparing).
"""
from __future__ import annotations

from typing import Any

from eval.metrics import OUTCOME_TIERS


def _fmt_rate(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "—"
    rate = entry.get("rate")
    pct = f"{100.0 * rate:.1f}%" if rate is not None else "n/a"
    return f"{entry.get('numerator', '?')}/{entry.get('denominator', '?')} ({pct})"


def _metric_row(label: str, entry: dict[str, Any] | None, note: str = "") -> str:
    if not entry:
        return f"| {label} | — | — | {note or '—'} |"
    rate = entry.get("rate")
    pct = f"{100.0 * rate:.1f}%" if rate is not None else "n/a"
    sample = entry.get("sample_size", entry.get("denominator", "?"))
    trailing = note if note else f"n={sample}"
    return (
        f"| {label} | {entry.get('numerator', '?')}/{entry.get('denominator', '?')} "
        f"| {pct} | {trailing} |"
    )


def _stats_row(label: str, stats: dict[str, Any]) -> str:
    if not stats or not stats.get("n"):
        return f"| {label} | 0 | — | — | — | — |"
    return (
        f"| {label} | {stats['n']} | {stats['mean']} | {stats['median']} | "
        f"{stats['p90']} | {stats['min']}–{stats['max']} |"
    )


def render_markdown(results: dict[str, Any]) -> str:
    run = results.get("run", {})
    metrics = results.get("metrics", {})
    manifest = run.get("manifest", {})
    lines: list[str] = []

    lines.append("# HABEAS evaluation — Phase 7")
    lines.append("")
    lines.append(f"Command: `{run.get('command', 'n/a')}`")
    lines.append("")
    lines.append(
        f"- Generated: {results.get('generated_at', 'n/a')}\n"
        f"- Seed: {run.get('seed')} · mode: {run.get('mode')} · "
        f"provider: {run.get('provider') or '—'} · model: {run.get('model') or '—'}\n"
        f"- Graph: `{run.get('graph_name')}` (disposable evaluation graph)\n"
        f"- Tool budget: {run.get('tool_budget')} · trace dir: `{run.get('trace_dir')}`\n"
        f"- Manifest: `{manifest.get('path')}` (seed {manifest.get('seed')}, "
        f"schema v{manifest.get('schema_version')})"
    )
    lines.append("")
    counts = metrics.get("counts", {})
    lines.append(
        f"Evaluated **{counts.get('evaluated')}** alert(s): "
        f"{counts.get('suspicious')} suspicious / {counts.get('benign')} benign · "
        f"planted pattern families: {counts.get('families')}."
    )
    if run.get("bounded_subset"):
        lines.append("")
        lines.append(
            "> **BOUNDED SUBSET RUN — debugging aid only.** Denominators cover "
            "only the evaluated alerts; this is not a complete evaluation."
        )
    lines.append("")

    lines.append("## Outcome mapping")
    lines.append("")
    lines.append(
        "| tier | meaning | counts as |\n"
        "| --- | --- | --- |\n"
        "| `supported_for_review` | canonical verdict `suspicion_supported_for_review` | positive risk finding (strict) |\n"
        "| `suspicious_unconfirmed` | canonical verdict `suspicious_activity` | signal present, escalation not met — neither positive nor benign |\n"
        "| `benign` | canonical verdict `likely_benign` | benign outcome |\n"
        "| `abstention` | `insufficient_evidence` / `incomplete_investigation` | inconclusive, visible in the matrix |\n"
        "| `failed` | harness error or unrecoverable investigation failure | stays in every denominator, reported |"
    )
    lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| metric | value | rate | sample |")
    lines.append("| --- | --- | --- | --- |")
    recall = metrics.get("recall", {})
    fpr = metrics.get("false_positive_rate", {})
    incon = metrics.get("inconclusive_rate", {})
    completion = metrics.get("completion", {})
    coverage = metrics.get("pattern_coverage", {})
    detection = metrics.get("pattern_detection", {})
    action = metrics.get("action", {})
    verifier = metrics.get("verifier", {})
    lines.append(_metric_row("Alert-level recall (strict)", recall.get("strict")))
    lines.append(_metric_row("Alert-level recall (lenient, incl. unconfirmed)", recall.get("lenient")))
    lines.append(_metric_row("False-positive rate on benign (strict)", fpr.get("strict")))
    lines.append(_metric_row("False-positive rate on benign (lenient)", fpr.get("lenient")))
    lines.append(_metric_row("Inconclusive rate (all alerts)", incon.get("overall")))
    lines.append(_metric_row("Inconclusive rate (suspicious)", incon.get("suspicious")))
    lines.append(_metric_row("Inconclusive rate (benign)", incon.get("benign")))
    lines.append(_metric_row("Failure rate", metrics.get("failure_rate")))
    lines.append(_metric_row("Investigation completion rate", completion.get("completed")))
    lines.append(_metric_row("Tool-budget exhaustion rate", completion.get("tool_budget_exhausted")))
    lines.append(_metric_row("Early termination rate", completion.get("early_termination")))
    lines.append(_metric_row("Pattern-family coverage (strict)", coverage.get("strict")))
    lines.append(_metric_row("Pattern-family coverage (>=1 detection)", coverage.get("lenient")))
    lines.append(_metric_row("Planted signal families observed", detection.get("expected_families_observed")))
    lines.append(_metric_row("Freeze-eligible under Policy v2 (of attempts)", action.get("eligible_for_freeze")))
    lines.append(_metric_row("Freeze applied (of attempts)", action.get("applied")))
    lines.append(_metric_row("Verifier pass (of checked claims)", verifier.get("claims_checked")))
    lines.append("")

    lines.append("## Confusion matrix (ground truth × outcome tier)")
    lines.append("")
    columns = metrics.get("confusion", {}).get("columns", list(OUTCOME_TIERS))
    lines.append("| ground truth | " + " | ".join(f"`{c}`" for c in columns) + " | total |")
    lines.append("| --- | " + " | ".join("---" for _ in columns) + " | --- |")
    for row_label in ("suspicious", "benign"):
        row = metrics.get("confusion", {}).get("rows", {}).get(row_label, {})
        cells = " | ".join(str(row.get(c, 0)) for c in columns)
        lines.append(f"| {row_label} | {cells} | {row.get('total', 0)} |")
    lines.append("")

    lines.append("## Per-alert results")
    lines.append("")
    lines.append("| alert | family | label | verdict | outcome | action | calls | elapsed_s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for record in results.get("alerts", []):
        action_status = (record.get("action") or {}).get("status")
        action_text = action_status if record.get("status") == "ok" else "—"
        if action_text == "applied":
            action_text = "applied ✅"
        lines.append(
            f"| {record.get('alert_id')} | {record.get('gt_family')} | "
            f"{record.get('gt_label')} | {record.get('canonical_verdict') or '—'} | "
            f"{record.get('outcome_tier')} | {action_text or '—'} | "
            f"{record.get('tool_calls')} | {record.get('elapsed_s')} |"
        )
    lines.append("")

    lines.append("## Per-pattern diagnostics")
    lines.append("")
    for family, info in (results.get("patterns", {}).get("by_family") or {}).items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append(
            f"- suspicious alerts: {info.get('suspicious_total')} · detected: "
            f"{', '.join(info.get('detected') or []) or '—'} · missed: "
            f"{', '.join(info.get('missed') or []) or '—'}"
        )
        lines.append(f"- detection rate: {_fmt_rate(info.get('detection_rate'))}")
        tools = info.get("evidence_tools_used") or {}
        if tools:
            top = ", ".join(f"{name}×{count}" for name, count in list(tools.items())[:10])
            lines.append(f"- evidence tools used: {top}")
        lines.append(
            f"- target-relevant evidence: {_fmt_rate(info.get('target_relevant_evidence'))} · "
            f"distinct-provenance evidence: {_fmt_rate(info.get('distinct_provenance_evidence'))} · "
            f"freeze-eligible: {_fmt_rate(info.get('freeze_eligible'))}"
        )
        reasons = info.get("failure_reasons_by_miss") or {}
        unresolved = info.get("unresolved_hypotheses_by_miss") or {}
        for alert_id, why in reasons.items():
            lines.append(f"- miss {alert_id}: {'; '.join(why)}")
        for alert_id, names in unresolved.items():
            lines.append(f"  - unresolved hypotheses {alert_id}: {', '.join(names) or '—'}")
        shared = info.get("benign_false_positives_sharing_pattern_accounts") or []
        if shared:
            lines.append(f"- benign alerts flagged that share pattern accounts: {', '.join(shared)}")
        lines.append("")
    benign_fps = (results.get("patterns", {}) or {}).get("benign_false_positives") or []
    lines.append(f"Benign alerts flagged (all): {', '.join(benign_fps) or '—'}")
    not_evaluated = (results.get("patterns", {}) or {}).get("families_not_evaluated") or []
    if not_evaluated:
        lines.append(
            f"Planted families not evaluated in this run: {', '.join(not_evaluated)}"
        )
    lines.append("")

    lines.append("## Action and intervention outcomes")
    lines.append("")
    lines.append("| outcome | value | sample |")
    lines.append("| --- | --- | --- |")
    for key, label in (
        ("attempted", "freeze attempts"),
        ("eligible_for_freeze", "eligible under Policy v2"),
        ("applied", "applied (sandbox mutation)"),
        ("declined", "declined with reasons"),
        ("already_applied", "already applied"),
        ("in_progress", "in progress"),
        ("failed", "failed attempts"),
    ):
        entry = action.get(key) or {}
        lines.append(f"| {label} | {_fmt_rate(entry)} | n={entry.get('sample_size', entry.get('denominator', '?'))} |")
    lines.append("")
    blocking = action.get("blocking_check_counts") or {}
    if blocking:
        lines.append("Policy checks that blocked attempts:")
        lines.append("")
        for check, count in blocking.items():
            lines.append(f"- `{check}`: {count}")
        lines.append("")

    lines.append("## Verifier integrity")
    lines.append("")
    lines.append(
        f"- cases verified: {verifier.get('cases_verified')} · verifier errors: "
        f"{verifier.get('cases_with_error')}"
    )
    lines.append(f"- claims checked: {_fmt_rate(verifier.get('claims_checked'))}")
    for status, count in (verifier.get("status_counts") or {}).items():
        lines.append(f"- `{status}`: {count}")
    lines.append("")

    lines.append("## Distributions")
    lines.append("")
    tool_calls = metrics.get("tool_calls", {})
    lines.append("| group | n | mean | median | p90 | range |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append(_stats_row("tool calls / alert (all)", tool_calls.get("all", {})))
    lines.append(_stats_row("tool calls / alert (suspicious)", tool_calls.get("suspicious", {})))
    lines.append(_stats_row("tool calls / alert (benign)", tool_calls.get("benign", {})))
    timing = results.get("timing", {})
    lines.append(_stats_row("wall-clock s / alert (all)", timing.get("all", {})))
    lines.append("")
    lines.append(
        f"Total tool calls: {tool_calls.get('total')}. Wall-clock includes MCP server "
        "startup and tool execution; FakeLLM adds no network latency."
    )
    lines.append("")

    ablation = results.get("one_hop_baseline")
    lines.append("## One-hop baseline vs bounded traversal (ablation)")
    lines.append("")
    if not ablation:
        lines.append("Not run for this evaluation (no circulation-family alert evaluated).")
    else:
        lines.append(f"Definition: {ablation.get('definition')}")
        lines.append("")
        lines.append(
            "| alert | target | 1-hop counterparties (in/out) | ≤3-account cycles "
            "(corroborated) | 4-account cycles (corroborated) | contribution |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in ablation.get("alerts", []):
            if item.get("error"):
                lines.append(
                    f"| {item.get('alert_id')} | {item.get('target')} | — | — | — | "
                    f"error: {item['error']} |"
                )
                continue
            one = item.get("one_hop", {})
            narrow = item.get("narrow_traversal", {})
            bounded = item.get("bounded_traversal", {})
            lines.append(
                f"| {item.get('alert_id')} | {item.get('target')} | "
                f"{one.get('counterparties_in')}/{one.get('counterparties_out')} | "
                f"{narrow.get('cycle_count')} ({narrow.get('corroborated_count')}) | "
                f"{bounded.get('cycle_count')} ({bounded.get('corroborated_count')}) | "
                f"{item.get('contribution')} |"
            )
        aggregate = ablation.get("aggregate", {})
        lines.append("")
        lines.append(
            f"**Aggregate:** alerts checked={aggregate.get('alerts_checked')} · "
            f"one-hop closures visible={aggregate.get('one_hop_closed_loop_visible')} · "
            f"narrow corroborated={aggregate.get('narrow_traversal_corroborated')} · "
            f"bounded corroborated={aggregate.get('bounded_traversal_corroborated')}"
        )
        lines.append("")
        lines.append(f"Conclusion: {aggregate.get('conclusion')}")
    lines.append("")

    memory = results.get("memory_comparison")
    lines.append("## Case-memory comparison (paired)")
    lines.append("")
    if not memory:
        lines.append("Not run for this evaluation (bounded subset or skipped).")
    else:
        lines.append(f"Method: {memory.get('method')}")
        lines.append("")
        formulas = memory.get("formulas") or {}
        for name, formula in formulas.items():
            lines.append(f"- `{name}`: {formula}")
        lines.append("")
        lines.append(
            "| pair | warm-up → repeat | rule | calls on | calls off | net Δ | "
            "prior lookups | avoided | validated | verdicts agree |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for pair in memory.get("pairs", []):
            measures = pair.get("repeat_measures", {})
            lines.append(
                f"| {pair.get('pair_id')} | {pair.get('warmup_alert')} → "
                f"{pair.get('repeat_alert')} | {pair.get('selection_rule')} | "
                f"{measures.get('calls_on')} | {measures.get('calls_off')} | "
                f"{measures.get('net_call_delta')} | "
                f"{measures.get('prior_lookup_calls_on')} | "
                f"{measures.get('calls_avoided')} | "
                f"{'yes' if measures.get('reused_findings_validated') else 'no'} | "
                f"{'yes' if (measures.get('verdicts') or {}).get('on_off_agree') else 'no'} |"
            )
        aggregate = memory.get("aggregate", {})
        lines.append("")
        lines.append(
            f"**Aggregate:** pairs={aggregate.get('pairs')} · calls on={aggregate.get('calls_on_total')} · "
            f"calls off={aggregate.get('calls_off_total')} · net Δ={aggregate.get('net_call_delta_total')} · "
            f"prior-case lookups introduced={aggregate.get('prior_lookup_calls_total')} · "
            f"steps avoided={aggregate.get('avoided_calls_total')} · "
            f"reduction={aggregate.get('reduction_pct')} · "
            f"warm-ups equivalent: {'yes' if aggregate.get('warmups_equivalent') else 'NO'} · "
            f"reused findings validated: {aggregate.get('validated_pairs')}/{aggregate.get('pairs')}"
        )
        lines.append("")
        lines.append(f"Conclusion: {aggregate.get('conclusion')}")
        lines.append("")
        lines.append(f"Wall-clock note: {aggregate.get('wall_clock_note')} "
                     f"(on total {aggregate.get('wall_clock_on_total_s')}s, off total "
                     f"{aggregate.get('wall_clock_off_total_s')}s)")
    lines.append("")

    reproducibility = results.get("reproducibility")
    lines.append("## Reproducibility check")
    lines.append("")
    if not reproducibility:
        lines.append("Not run for this evaluation (bounded subset or skipped).")
    else:
        lines.append(
            f"Same seed + FakeLLM configuration, graph rebuilt: "
            f"**{'identical' if reproducibility.get('identical') else 'MISMATCH'}** "
            f"({reproducibility.get('digest_first')} vs {reproducibility.get('digest_second')})."
        )
        for mismatch in reproducibility.get("mismatched_alerts") or []:
            lines.append(f"- mismatch {mismatch.get('alert_id')}: {mismatch.get('fields')}")
    lines.append("")

    alt = results.get("alt_seed")
    lines.append("## Alternative-seed validation")
    lines.append("")
    if not alt:
        lines.append("Not run for this evaluation.")
    else:
        alt_metrics = alt.get("metrics", {})
        alt_counts = alt_metrics.get("counts", {})
        lines.append(
            f"Seed {alt.get('seed')} (separate denominators, never combined): "
            f"{alt_counts.get('evaluated')} alert(s), "
            f"{alt_counts.get('suspicious')} suspicious / {alt_counts.get('benign')} benign."
        )
        lines.append("")
        lines.append("| metric | value | rate | sample |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(_metric_row("recall (strict)", alt_metrics.get("recall", {}).get("strict")))
        lines.append(_metric_row("false-positive rate (strict)", alt_metrics.get("false_positive_rate", {}).get("strict")))
        lines.append(_metric_row("pattern coverage (strict)", alt_metrics.get("pattern_coverage", {}).get("strict")))
        lines.append(_metric_row("completion rate", alt_metrics.get("completion", {}).get("completed")))
        lines.append("")

    real = results.get("real_llm")
    lines.append("## Real-LLM evaluation")
    lines.append("")
    if not real:
        lines.append("Not requested.")
    else:
        lines.append(
            f"- status: **{real.get('status')}**\n"
            f"- provider: {real.get('provider') or '—'} · model: {real.get('model') or '—'}"
        )
        if real.get("error"):
            lines.append(f"- error: {real['error']}")
        real_metrics = real.get("metrics") or {}
        if real_metrics:
            real_counts = real_metrics.get("counts", {})
            lines.append(
                f"- alerts evaluated: {real_counts.get('evaluated')} "
                f"({real_counts.get('suspicious')} suspicious / {real_counts.get('benign')} benign)"
            )
            lines.append(
                f"- recall (strict): {_fmt_rate(real_metrics.get('recall', {}).get('strict'))} · "
                f"FP rate (strict): {_fmt_rate(real_metrics.get('false_positive_rate', {}).get('strict'))} · "
                f"completion: {_fmt_rate(real_metrics.get('completion', {}).get('completed'))}"
            )
        for record in real.get("alerts") or []:
            lines.append(
                f"- {record.get('alert_id')}: status={record.get('status')}, "
                f"verdict={record.get('canonical_verdict')}, "
                f"calls={record.get('tool_calls')}, elapsed={record.get('elapsed_s')}s, "
                f"errors={record.get('errors') or record.get('error') or '—'}"
            )
    lines.append("")

    lines.append("## Failures, misses, and limitations")
    lines.append("")
    failures = results.get("failures") or []
    if not failures:
        lines.append("No missed suspicious alerts, false positives, or run problems recorded.")
    else:
        for item in failures:
            kind = item.get("kind")
            if kind == "missed_suspicious_alert":
                lines.append(
                    f"- MISSED {item.get('alert_id')} ({item.get('gt_family')}): "
                    f"{item.get('outcome_tier')} / {item.get('canonical_verdict')} — "
                    f"{'; '.join(item.get('reasons') or []) or 'no reason recorded'}"
                )
            elif kind == "false_positive_on_benign":
                lines.append(
                    f"- FALSE POSITIVE {item.get('alert_id')}: {item.get('outcome_tier')} / "
                    f"{item.get('canonical_verdict')}"
                )
            elif kind == "truncation_notice":
                lines.append(
                    f"- truncated evidence on {item.get('alert_id')}: "
                    f"{', '.join(item.get('truncated_evidence') or [])} (bounded row caps; "
                    "recorded, not hidden)"
                )
            else:
                lines.append(
                    f"- run problem {item.get('alert_id')}: "
                    f"status={item.get('status')}, termination={item.get('termination_reason')}, "
                    f"errors={item.get('errors') or item.get('harness_error') or '—'}"
                )
    lines.append("")
    lines.append("Limitations:")
    lines.append("")
    for limitation in results.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append(
        "Isolation: the agent receives alert ids and graph evidence only; the "
        "ground-truth manifest is read exclusively by this evaluation layer, and "
        "every evaluation graph is disposable (`HABEAS_demo_eval_*`). Actions run "
        "only behind the deterministic Policy v2 checks in the sandbox."
    )
    lines.append("")
    return "\n".join(lines)
