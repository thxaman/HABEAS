"""Phase 7 evaluation CLI (PLAN.md §12).

Complete deterministic evaluation:

    python -m eval.run --alerts all --seed 42 --mode fake

Extended gate run (reproducibility is automatic for full runs; these add the
alternative-seed validation and the bounded real-LLM smoke):

    python -m eval.run --alerts all --seed 42 --mode fake --alt-seed 43 --real-smoke

Bounded real-LLM evaluation (explicit, billed, never required by tests):

    python -m eval.run --mode real --provider groq --alerts 2 --seed 42 \\
        --out eval/results.real.json

Debugging subsets: ``--alerts 5`` (first five by id) or
``--alerts ALR-0001,ALR-0004``. Subset runs are labelled as bounded and skip
the memory comparison / reproducibility sweeps.

Artifacts: ``eval/results.json`` + ``eval/results.md`` (fake mode) or the
``--out`` path. The Markdown scorecard is rendered from the same structured
results, and metrics are recomputable from the JSON alone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.llm import FakeLLM, LLMConfigurationError, make_llm
from src.config import Settings, load_settings
from src.seed.patterns import AlertTruth, GroundTruth

from eval import EvaluationConfigError, EvaluationError
from eval.ground_truth import (
    _portable_path,
    alert_index,
    load_manifest,
    manifest_summary,
)
from eval.harness import (
    AlertRunner,
    compare_runs,
    eval_graph_name,
    eval_settings,
    run_one_hop_ablation,
    run_sweep,
    seed_eval_graph,
)
from eval.memory import run_comparison
from eval.metrics import checklist_failures, compute_metrics, describe, pattern_diagnostics
from eval.render import render_markdown

MAX_ALERTS_PER_RUN = 500

_STATIC_LIMITATIONS = [
    "FakeLLM is a deterministic regression baseline: it exercises the real MCP "
    "stdio server, Policy v2, the verifier and the production tool-budget "
    "enforcement, but it does not demonstrate live model tool selection. That "
    "is covered separately by the bounded real-LLM mode/smoke run.",
    "Wall-clock timings include MCP server startup and are descriptive, not "
    "benchmarks.",
    "The memory comparison is small by construction (pairs derived from the "
    "synthetic manifest); its wall-clock deltas are not statistically "
    "meaningful, and only tool-call deltas with recorded step evidence are "
    "treated as measured savings.",
    "Alert-level metrics are measured on the synthetic benchmark and do not "
    "generalize to production data.",
    "The agent never receives ground truth; conversely the evaluation can only "
    "observe what the agent recorded as evidence. Interpretations are "
    "deterministic Python, never LLM-authored.",
    "Findings are risk indicators for review, not legal conclusions.",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _progress_printer(prefix: str):
    def _progress(index: int, total: int, record: dict[str, Any]) -> None:
        action = (record.get("action") or {}).get("status") or "n/a"
        print(
            f"[{prefix} {index}/{total}] {record.get('alert_id')}: "
            f"status={record.get('status')} verdict={record.get('canonical_verdict')} "
            f"tier={record.get('outcome_tier')} calls={record.get('tool_calls')} "
            f"action={action} elapsed={record.get('elapsed_s')}s",
            file=sys.stderr,
            flush=True,
        )

    return _progress


def _timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_alert = {
        r["alert_id"]: round(float(r.get("elapsed_s") or 0.0), 3) for r in records
    }
    values = list(per_alert.values())
    suspicious = [
        float(r.get("elapsed_s") or 0.0) for r in records if r.get("gt_label") == "suspicious"
    ]
    benign = [
        float(r.get("elapsed_s") or 0.0) for r in records if r.get("gt_label") == "benign"
    ]
    return {
        "per_alert": per_alert,
        "all": describe(values),
        "suspicious": describe(suspicious),
        "benign": describe(benign),
        "total_s": round(sum(values), 3),
        "note": (
            "wall-clock includes MCP stdio server startup and tool execution; "
            "FakeLLM adds no network latency"
        ),
    }


def resolve_alert_truths(spec: str, manifest: GroundTruth) -> list[AlertTruth]:
    """``all`` | first N | comma-separated alert ids (validated against manifest)."""
    index = alert_index(manifest)
    ordered = sorted(manifest.alerts, key=lambda a: a.alert_id)
    if spec is None or spec.strip().lower() == "all":
        return ordered
    text = spec.strip()
    if text.isdigit():
        count = int(text)
        if count <= 0:
            raise EvaluationConfigError("--alerts count must be positive")
        return ordered[:count]
    requested: list[str] = []
    for raw in text.split(","):
        alert_id = raw.strip()
        if not alert_id:
            continue
        if alert_id not in index:
            raise EvaluationConfigError(
                f"unknown alert id {alert_id!r}; available ids: "
                f"{', '.join(a.alert_id for a in ordered)}"
            )
        if alert_id not in requested:
            requested.append(alert_id)
    if not requested:
        raise EvaluationConfigError("no alert ids were provided")
    return [index[alert_id] for alert_id in requested]


def _command_line(args: argparse.Namespace) -> str:
    parts = ["python -m eval.run", f"--alerts {args.alerts}", f"--seed {args.seed}",
             f"--mode {args.mode}"]
    if args.provider:
        parts.append(f"--provider {args.provider}")
    if args.tool_budget:
        parts.append(f"--tool-budget {args.tool_budget}")
    if args.alt_seed is not None:
        parts.append(f"--alt-seed {args.alt_seed}")
    if args.real_smoke:
        parts.append("--real-smoke")
        if args.real_smoke_alerts != "2":
            parts.append(f"--real-smoke-alerts {args.real_smoke_alerts}")
    if args.skip_memory:
        parts.append("--skip-memory")
    if args.skip_reproducibility:
        parts.append("--skip-reproducibility")
    if args.out:
        parts.append(f"--out {args.out}")
    if args.manifest:
        parts.append(f"--manifest {args.manifest}")
    return " ".join(parts)


def _real_llm_block_status(records: list[dict[str, Any]]) -> str:
    if any(r.get("status") != "ok" for r in records):
        return "completed_with_failures"
    if any(r.get("errors") for r in records):
        return "completed_with_failures"
    return "completed"


def _run_real_sweep(
    *,
    truths: list[AlertTruth],
    base_settings: Settings,
    seed: int,
    provider: str | None,
    tool_budget: int | None,
    trace_dir: Path,
    tag: str,
) -> dict[str, Any]:
    """Bounded real-LLM sweep; configuration problems are reported, not raised."""
    try:
        llm = make_llm(base_settings, mode="real", provider=provider)
    except LLMConfigurationError as exc:
        message = str(exc)
        status = (
            "not_run_missing_credentials"
            if "API_KEY" in message
            else "not_run_configuration_error"
        )
        return {"status": status, "provider": provider or base_settings.llm_provider,
                "model": None, "error": message, "alerts": [], "metrics": None}
    graph_name = eval_graph_name(seed, tag)
    settings = eval_settings(
        base_settings, graph_name=graph_name, mode="real", provider=provider,
        tool_budget=tool_budget,
    )
    seed_eval_graph(
        settings, seed=seed, graph_name=graph_name,
        manifest_path=Path(trace_dir) / f"manifest_{graph_name}.json",
    )
    runner = AlertRunner(
        settings=settings, graph_name=graph_name, seed=seed, mode="real",
        llm=llm, trace_dir=trace_dir, provider=llm.provider, model=llm.model_name,
    )
    records = run_sweep(truths, runner=runner, progress=_progress_printer(f"real:{tag}"))
    return {
        "status": _real_llm_block_status(records),
        "provider": llm.provider,
        "model": llm.model_name,
        "alerts": records,
        "metrics": compute_metrics(records),
        "note": (
            "bounded opt-in real-LLM evaluation; provider failures and malformed "
            "tool choices are recorded per alert and stay in the denominators"
        ),
    }


def _run_alt_seed(
    *,
    alt_seed: int,
    base_settings: Settings,
    tool_budget: int | None,
    trace_dir: Path,
) -> dict[str, Any]:
    graph_name = eval_graph_name(alt_seed, f"alt{alt_seed}")
    manifest_file = Path(trace_dir) / f"alt_seed_{alt_seed}_ground_truth.json"
    settings = eval_settings(
        base_settings, graph_name=graph_name, mode="fake", tool_budget=tool_budget
    )
    seed_eval_graph(
        settings, seed=alt_seed, graph_name=graph_name, manifest_path=manifest_file
    )
    manifest = load_manifest(manifest_file, expected_seed=alt_seed)
    truths = resolve_alert_truths("all", manifest)
    runner = AlertRunner(
        settings=settings, graph_name=graph_name, seed=alt_seed, mode="fake",
        llm=FakeLLM(), trace_dir=trace_dir,
    )
    records = run_sweep(truths, runner=runner, progress=_progress_printer(f"alt{alt_seed}"))
    return {
        "seed": alt_seed,
        "manifest": manifest_summary(manifest, manifest_file),
        "alerts": records,
        "metrics": compute_metrics(records),
        "note": (
            "separate validation run on an alternative seed with its own "
            "denominators; never combined with the primary seed's metrics"
        ),
    }


def run_deterministic_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    base = load_settings()
    manifest = load_manifest(args.manifest, expected_seed=args.seed)
    summary = manifest_summary(manifest, args.manifest)
    truths = resolve_alert_truths(args.alerts, manifest)
    full = len(truths) == len(manifest.alerts)

    trace_dir = Path(args.trace_dir) if args.trace_dir else Path("logs") / f"eval_fake_s{args.seed}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    graph_name = eval_graph_name(args.seed, "main")
    settings = eval_settings(
        base, graph_name=graph_name, mode="fake", tool_budget=args.tool_budget
    )
    seed_eval_graph(
        settings, seed=args.seed, graph_name=graph_name,
        manifest_path=trace_dir / f"manifest_{graph_name}.json",
    )
    # PLAN.md §12: only the slice of the fixture that can form rings is used,
    # measured on the pristine seeded graph before any sandbox mutation.
    ablation = run_one_hop_ablation(
        truths=truths, settings=settings, graph_name=graph_name
    )
    runner = AlertRunner(
        settings=settings, graph_name=graph_name, seed=args.seed, mode="fake",
        llm=FakeLLM(), trace_dir=trace_dir,
    )
    started = time.perf_counter()
    records = run_sweep(truths, runner=runner, progress=_progress_printer("fake"))
    elapsed = time.perf_counter() - started

    metrics = compute_metrics(records)
    patterns = pattern_diagnostics(records, summary)
    failures = checklist_failures(records)

    memory = None
    if full and not args.skip_memory:
        print("running paired memory comparison...", file=sys.stderr, flush=True)
        memory = run_comparison(
            manifest=manifest, base_settings=base, seed=args.seed,
            trace_root=trace_dir, progress=lambda msg: print(msg, file=sys.stderr, flush=True),
        )

    reproducibility = None
    if full and not args.skip_reproducibility:
        print("running reproducibility sweep (graph rebuilt)...", file=sys.stderr, flush=True)
        seed_eval_graph(
            settings, seed=args.seed, graph_name=graph_name,
            manifest_path=trace_dir / f"manifest_{graph_name}.json",
        )
        runner2 = AlertRunner(
            settings=settings, graph_name=graph_name, seed=args.seed, mode="fake",
            llm=FakeLLM(), trace_dir=trace_dir,
        )
        records2 = run_sweep(
            truths, runner=runner2, progress=_progress_printer("repro")
        )
        reproducibility = compare_runs(records, records2)

    alt = None
    if args.alt_seed is not None:
        print(f"running alternative-seed validation (seed {args.alt_seed})...",
              file=sys.stderr, flush=True)
        alt = _run_alt_seed(
            alt_seed=args.alt_seed, base_settings=base,
            tool_budget=args.tool_budget, trace_dir=trace_dir,
        )

    real = None
    if args.real_smoke:
        print("running bounded real-LLM smoke evaluation...", file=sys.stderr, flush=True)
        smoke_truths = resolve_alert_truths(args.real_smoke_alerts, manifest)
        real = _run_real_sweep(
            truths=smoke_truths, base_settings=base, seed=args.seed,
            provider=args.provider, tool_budget=args.real_smoke_budget,
            trace_dir=trace_dir, tag="realsmoke",
        )

    limitations = list(_STATIC_LIMITATIONS)
    if not full:
        limitations.insert(
            0,
            f"BOUNDED SUBSET: {len(truths)} of {len(manifest.alerts)} alerts were "
            "evaluated; metrics and denominators cover only the evaluated alerts.",
        )
    if memory is not None:
        aggregate = memory.get("aggregate", {})
        if aggregate.get("avoided_calls_total", 0) == 0:
            limitations.insert(
                0,
                "No tool-call saving from case memory was measured in this "
                "configuration (see the memory comparison block for the measured "
                "delta); nothing is assumed.",
            )

    results = {
        "schema_version": 1,
        "generated_at": _stamp(),
        "run": {
            "command": _command_line(args),
            "seed": args.seed,
            "mode": "fake",
            "provider": None,
            "model": None,
            "graph_name": graph_name,
            "tool_budget": settings.tool_budget,
            "alerts_requested": args.alerts,
            "bounded_subset": not full,
            "manifest": summary,
            "trace_dir": _portable_path(trace_dir),
            "elapsed_s": round(elapsed, 3),
        },
        "alerts": records,
        "metrics": metrics,
        "patterns": patterns,
        "one_hop_baseline": ablation,
        "timing": _timing(records),
        "failures": failures,
        "memory_comparison": memory,
        "reproducibility": reproducibility,
        "alt_seed": alt,
        "real_llm": real,
        "limitations": limitations,
    }
    return results


def run_real_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    base = load_settings()
    if args.alerts.strip().lower() == "all":
        raise EvaluationConfigError(
            "real-LLM evaluation is bounded by design: pass --alerts N or a "
            "comma-separated alert-id list"
        )
    manifest = load_manifest(args.manifest, expected_seed=args.seed)
    summary = manifest_summary(manifest, args.manifest)
    truths = resolve_alert_truths(args.alerts, manifest)
    full = len(truths) == len(manifest.alerts)

    trace_dir = Path(args.trace_dir) if args.trace_dir else Path("logs") / f"eval_real_s{args.seed}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    graph_name = eval_graph_name(args.seed, "real")
    try:
        llm = make_llm(base, mode="real", provider=args.provider)
    except LLMConfigurationError as exc:
        raise EvaluationConfigError(str(exc)) from exc
    settings = eval_settings(
        base, graph_name=graph_name, mode="real", provider=args.provider,
        tool_budget=args.tool_budget,
    )
    seed_eval_graph(
        settings, seed=args.seed, graph_name=graph_name,
        manifest_path=trace_dir / f"manifest_{graph_name}.json",
    )
    ablation = run_one_hop_ablation(
        truths=truths, settings=settings, graph_name=graph_name
    )
    runner = AlertRunner(
        settings=settings, graph_name=graph_name, seed=args.seed, mode="real",
        llm=llm, trace_dir=trace_dir, provider=llm.provider, model=llm.model_name,
    )
    started = time.perf_counter()
    records = run_sweep(truths, runner=runner, progress=_progress_printer("real"))
    elapsed = time.perf_counter() - started

    metrics = compute_metrics(records)
    return {
        "schema_version": 1,
        "generated_at": _stamp(),
        "run": {
            "command": _command_line(args),
            "seed": args.seed,
            "mode": "real",
            "provider": llm.provider,
            "model": llm.model_name,
            "graph_name": graph_name,
            "tool_budget": settings.tool_budget,
            "alerts_requested": args.alerts,
            "bounded_subset": not full,
            "manifest": summary,
            "trace_dir": _portable_path(trace_dir),
            "elapsed_s": round(elapsed, 3),
        },
        "alerts": records,
        "metrics": metrics,
        "patterns": pattern_diagnostics(records, summary),
        "one_hop_baseline": ablation,
        "timing": _timing(records),
        "failures": checklist_failures(records),
        "memory_comparison": None,
        "reproducibility": None,
        "alt_seed": None,
        "real_llm": {
            "status": _real_llm_block_status(records),
            "provider": llm.provider,
            "model": llm.model_name,
            "note": "this evaluation ran with --mode real; per-alert results above",
        },
        "limitations": _STATIC_LIMITATIONS
        + [
            "Real-LLM runs are non-deterministic; failures and provider errors "
            "are recorded per alert and remain in the denominators.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run",
        description="HABEAS Phase 7 evaluation harness (synthetic data only).",
    )
    parser.add_argument("--alerts", default="all",
                        help="'all', first N (e.g. 5), or comma-separated alert ids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["fake", "real"], default="fake")
    parser.add_argument("--provider", choices=["openai", "groq"], default=None,
                        help="real-LLM provider override (LLM_PROVIDER otherwise)")
    parser.add_argument("--tool-budget", type=int, default=None,
                        help="override the production tool budget for this run")
    parser.add_argument("--alt-seed", type=int, default=None,
                        help="alternative-seed validation run (e.g. 43)")
    parser.add_argument("--real-smoke", action="store_true",
                        help="also run the bounded real-LLM smoke evaluation")
    parser.add_argument("--real-smoke-alerts", default="2",
                        help="bounded alert subset for the smoke run (default: first 2)")
    parser.add_argument("--real-smoke-budget", type=int, default=6,
                        help="tool budget for the bounded real-LLM smoke run")
    parser.add_argument("--skip-memory", action="store_true",
                        help="skip the paired memory comparison (full runs only)")
    parser.add_argument("--skip-reproducibility", action="store_true",
                        help="skip the rebuild-and-rerun reproducibility sweep")
    parser.add_argument("--out", default=None,
                        help="results JSON path (default eval/results.json; "
                             "eval/results.real.json in real mode)")
    parser.add_argument("--manifest", default=None, help="manifest path override")
    parser.add_argument("--trace-dir", default=None, help="trace/artifact directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    # The rendered scorecard contains typographic glyphs; make the console
    # output encoding-safe on Windows without crashing the run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    if args.mode == "real" and not args.out:
        args.out = "eval/results.real.json"
    out_path = Path(args.out) if args.out else Path("eval/results.json")
    md_path = out_path.with_suffix(".md")

    try:
        if args.mode == "real":
            results = run_real_evaluation(args)
        else:
            results = run_deterministic_evaluation(args)
    except (EvaluationError, EvaluationConfigError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(results)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nwrote {out_path} and {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
