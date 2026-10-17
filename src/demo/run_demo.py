"""HABEAS CLI demo (Phase 8, PLAN.md §13): three scripted investigations.

    python -m src.demo.run_demo                 # deterministic FakeLLM (backup demo)
    python -m src.demo.run_demo --mode real     # bounded live provider demo
    python -m src.demo.run_demo --mode auto     # real when credentials exist, else fake

Scenarios (all over the REAL MCP stdio server):

A. ALR-0001  overlapping circulation + rapid-forwarding observations -> the
             provenance-aware policy DECLINES the freeze.
B. ALR-0021  distinct observations with target-specific qualifying evidence ->
             counterfactual simulation, an APPLIED sandbox freeze, and a
             successful post-action verification.
C. ALR-0004  fan-in/fan-out evidence only -> one signal family cannot
             authorize a freeze, so the policy DECLINES.

Honesty rules baked in:

- every number printed comes from the actual tool result / persisted case
  record — nothing is prewritten or replayed;
- the mode that produced the result is labelled; FakeLLM output is never
  presented as proof of live model tool selection;
- the demo NEVER claims HABEAS detects every pattern or that memory saves
  calls (Phase 7 measured no tool-call saving);
- the freeze is a simulated mutation in a disposable demo graph only.

Safety:

- the demo graph must start with ``HABEAS_demo_showcase`` (a disposable,
  demo-safe namespace distinct from ``HABEAS_demo_local`` and the evaluation
  graphs); any other name is refused before it is touched;
- reports are written under ``reports/demo/`` by default, so the historical
  ``reports/CASE-*.md`` artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.llm import FakeLLM, LLMConfigurationError, make_llm
from src.agent.workflow import run_case_workflow
from src.config import DEMO_GRAPH_PREFIX, Settings, load_settings
from src.db import get_graph, run_query
from src.demo.graph_link import build_graph_link, subgraph_cypher
from src.report.builder import build_report, write_report
from src.seed import generate, schema

# Disposable demo namespace (distinct from the shared demo graph and the
# evaluation graphs; still inside the demo-safe prefix the reset guard checks).
DEMO_GRAPH_NAMESPACE = "HABEAS_demo_showcase"
DEFAULT_SEED = 42
DEFAULT_REPORTS_DIR = "reports/demo"
DEFAULT_TRACE_DIR = "logs/demo"

_RULE = "=" * 78


@dataclass(frozen=True)
class Scenario:
    key: str
    alert_id: str
    title: str
    stake: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "A", "ALR-0001", "Overlapping circulation and rapid-forwarding observations",
        "Both observations may come from the SAME underlying transactions, so the "
        "provenance-aware policy must decline a freeze that one shared story "
        "cannot justify.",
    ),
    Scenario(
        "B", "ALR-0021", "Distinct observations with target-specific evidence",
        "Two independent, target-relevant signal families plus a complete "
        "simulation should clear the deterministic policy and apply a sandbox "
        "freeze that post-action verification confirms.",
    ),
    Scenario(
        "C", "ALR-0004", "Fan-in / fan-out only (a single signal family)",
        "A concentrated fan-in pattern alone is a risk indicator, not sufficient "
        "evidence: one signal family cannot authorize a freeze.",
    ),
)


# --------------------------------------------------------------- graph guards


def assert_demo_graph(name: str) -> None:
    """Refuse anything that is not a disposable showcase demo graph."""
    if not name.startswith(DEMO_GRAPH_NAMESPACE) or not name.startswith(
        DEMO_GRAPH_PREFIX
    ):
        raise ValueError(
            f"refusing to use graph {name!r}: the demo may only reseed/mutate its "
            f"own disposable graphs named {DEMO_GRAPH_NAMESPACE!r}+ "
            f"(never the shared demo graph, evaluation graphs, or arbitrary graphs)"
        )


def demo_graph_name(seed: int) -> str:
    name = f"{DEMO_GRAPH_NAMESPACE}_s{seed}"
    assert_demo_graph(name)
    return name


def demo_settings(
    base: Settings, *, graph_name: str, mode: str, tool_budget: int | None
) -> Settings:
    assert_demo_graph(graph_name)
    updates: dict[str, Any] = {
        "graph_name": graph_name,
        "habeas_demo_mode": True,
        "habeas_enable_actions": True,
        "llm_mode": "real" if mode == "real" else "fake",
    }
    if tool_budget is not None:
        updates["tool_budget"] = tool_budget
    return Settings(**{**base.model_dump(), **updates})


def reseed_demo_graph(settings: Settings, *, seed: int, trace_dir: Path) -> None:
    """Deterministically rebuild the disposable demo graph (guarded)."""
    assert_demo_graph(settings.graph_name)
    graph = get_graph(settings)
    run_query(graph, "RETURN 1 AS bootstrap", read_only=False)  # create key if new
    action = generate.plan_reset(settings.graph_name, True, schema.node_count(graph))
    generate.seed(
        settings,
        seed=seed,
        reset=(action == "reset_and_seed"),
        graph=graph,
        # never clobber the canonical manifest used by the evaluation
        manifest_path=trace_dir / f"manifest_{settings.graph_name}.json",
    )


# ------------------------------------------------------------ mode resolution


def _credentials_configured(settings: Settings) -> bool:
    if settings.llm_provider == "groq":
        return bool(settings.groq_api_key.get_secret_value()) and bool(settings.groq_model)
    return bool(settings.openai_api_key.get_secret_value()) and bool(settings.openai_model)


def resolve_mode(requested: str, settings: Settings) -> str:
    """``auto`` -> real when the resolved provider is fully configured, else fake."""
    if requested in {"fake", "real"}:
        return requested
    return "real" if _credentials_configured(settings) else "fake"


def mode_label(mode: str, llm) -> str:
    if mode == "real":
        return f"real provider={llm.provider} model={llm.model_name}"
    return "FakeLLM (deterministic; NOT evidence of live model tool selection)"


# ------------------------------------------------------------------- printing


def _short(text: str, limit: int = 150) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_scenario_header(scenario: Scenario, *, index: int, total: int) -> None:
    print(_RULE)
    print(f"SCENARIO {scenario.key} ({index}/{total}) — alert {scenario.alert_id}")
    print(scenario.title)
    print(f"stake: {scenario.stake}")
    print(_RULE)


def print_investigation(result) -> None:
    state = result.state
    print(
        f"\n[investigation] case={result.case_id} "
        f"canonical_verdict={result.canonical_verdict} "
        f"termination={state.termination_reason}"
    )
    print(
        f"[investigation] tool calls used: {state.tool_calls_used} / budget "
        f"{state.tool_budget}"
    )
    print("[investigation] MCP tools selected, in order:")
    for i, tool in enumerate(result.tool_sequence, start=1):
        print(f"    {i:2d}. {tool}")

    print("[investigation] evidence records (deterministic interpretations):")
    for rec in state.evidence:
        flag = " [truncated]" if any(rec.truncated) else ""
        qids = ", ".join(rec.query_ids[:3]) or "—"
        status = "ok" if rec.ok else f"FAILED: {rec.error}"
        print(f"    {rec.evidence_id} {rec.tool} ({status}){flag}")
        print(f"        queries: {qids}")
        if rec.interpretation:
            print(f"        reading: {_short(rec.interpretation)}")


def print_simulation(envelopes: dict[str, Any]) -> None:
    env = envelopes.get("simulate_freeze") or {}
    metrics = (env.get("result") or {}).get("metrics") or {}
    if not metrics:
        print("\n[simulate_freeze] not available for this scenario")
        return
    print("\n[simulate_freeze] counterfactual (read-only; no mutation):")
    print(f"    simulation_complete: {metrics.get('simulation_complete')}")
    print(f"    direct_outgoing_volume (window): {metrics.get('direct_outgoing_volume')}")
    print(f"    downstream accounts before: {metrics.get('downstream_accounts_before')}")
    print(f"    newly unreachable if frozen: {metrics.get('newly_unreachable_count')}")
    print(f"    active edges that would be disabled: {metrics.get('active_edges_disabled')}")
    semantics = metrics.get("metric_semantics")
    if isinstance(semantics, dict):
        note = semantics.get("direct_outgoing_volume") or semantics.get("note")
        if note:
            print(f"    metric meaning: {_short(str(note), 150)}")


def print_action(envelopes: dict[str, Any]) -> None:
    env = envelopes.get("freeze_account") or {}
    payload = env.get("result") or {}
    if not payload:
        print("\n[freeze_account] not attempted")
        return
    policy = payload.get("policy") or {}
    print(
        "\n[freeze_account] deterministic policy "
        f"(v{policy.get('policy_version', '?')}):"
    )
    print(f"    allowed: {policy.get('allowed')}  verdict: {policy.get('verdict')}")
    print("    signal families: " + (", ".join(policy.get("signal_families") or []) or "none"))
    print(
        "    target-relevant families: "
        + (", ".join(policy.get("target_relevant_families") or []) or "none")
    )
    print(
        "    distinct-provenance families: "
        + (", ".join(policy.get("distinct_provenance_families") or []) or "none")
    )
    print(
        "    qualifying evidence ids: "
        + (", ".join(policy.get("qualifying_evidence_ids") or []) or "none")
    )
    checks = policy.get("checks") or {}
    failing = sorted(name for name, ok in checks.items() if not ok)
    if failing:
        print(f"    failing checks: {', '.join(failing)}")
    for reason in policy.get("blocking_reasons") or []:
        print(f"    blocking reason: {_short(reason, 170)}")
    print(
        f"    RESULT: action_status={payload.get('action_status')} "
        f"frozen={payload.get('frozen')} action_id={payload.get('action_id')} "
        f"mutation_performed={payload.get('mutation_performed')}"
    )


def print_post_action(envelopes: dict[str, Any], *, frozen: bool) -> None:
    env = envelopes.get("verify_freeze") or {}
    payload = env.get("result") or {}
    if not payload:
        return
    checks = payload.get("checks") or {}
    if not frozen:
        # Nothing was applied, so this is a confirmation that the graph is
        # unchanged — not a failed action. Present it that way.
        print("\n[verify_freeze] no action was applied; confirming the graph is unchanged:")
        print(f"    account_status: {payload.get('account_status')} (unchanged)")
        print(
            f"    action record present: {bool(checks.get('action_record_present'))} "
            "(expected False when declined)"
        )
        print("    no mutation was performed — the freeze was correctly withheld")
        return
    print("\n[verify_freeze] post-action graph state (re-queried):")
    print(f"    account_status: {payload.get('account_status')}")
    print(f"    verified: {payload.get('verified')}")
    for name, ok in sorted(checks.items()):
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    print(
        "    note: historical transfer evidence is retained; only routing "
        "edges were deactivated"
    )


def print_report(artifact, *, reports_dir: Path) -> None:
    verification = artifact.verification
    print("\n[report] case report written")
    print(f"    path: {reports_dir / (artifact.case_id + '.md')}")
    print(f"    report_version: {artifact.report_version}")
    print(f"    independent verifier — claim status counts: {verification.summary}")
    if verification.reference_summary:
        print(f"    material-claim reference checks: {verification.reference_summary}")
    print(
        f"    live current-state checks: {len(verification.current_state)} · "
        f"post-action checks: {len(verification.post_action)}"
    )
    for note in verification.notes[:3]:
        print(f"    note: {_short(note, 160)}")


def print_graph_link(*, graph_name: str, account_ids: list[str]) -> None:
    link = build_graph_link(graph_name, subgraph_cypher(account_ids))
    print("\n[graph] browser hint (visualization only; not used by the agent)")
    print(link.render())


# ------------------------------------------------------------------- runner


def _action_summary(envelopes: dict[str, Any]) -> dict[str, Any]:
    payload = (envelopes.get("freeze_account") or {}).get("result") or {}
    policy = payload.get("policy") or {}
    return {
        "action_status": payload.get("action_status"),
        "frozen": bool(payload.get("frozen")),
        "allowed": policy.get("allowed"),
        "distinct_provenance_families": list(policy.get("distinct_provenance_families") or []),
        "target_relevant_families": list(policy.get("target_relevant_families") or []),
        "blocking_reasons": list(policy.get("blocking_reasons") or []),
    }


async def _run_scenario(
    scenario: Scenario,
    *,
    index: int,
    total: int,
    settings: Settings,
    llm,
    trace_dir: Path,
    reports_dir: Path,
    graph,
) -> dict[str, Any]:
    print_scenario_header(scenario, index=index, total=total)
    result = await run_case_workflow(
        scenario.alert_id,
        settings=settings,
        graph_name=settings.graph_name,
        llm=llm,
        trace_dir=trace_dir,
        case_memory=True,
        freeze=True,
    )

    print_investigation(result)
    print_simulation(result.envelopes)
    print_action(result.envelopes)
    print_post_action(result.envelopes, frozen=result.frozen)

    artifact = build_report(
        graph, result.case_id, state=result.state, envelopes=result.envelopes,
        settings=settings,
    )
    write_report(artifact, reports_dir)
    print_report(artifact, reports_dir=reports_dir)

    print_graph_link(
        graph_name=settings.graph_name,
        account_ids=list(result.state.primary_entity_ids),
    )

    return {
        "key": scenario.key,
        "alert_id": scenario.alert_id,
        "case_id": result.case_id,
        "canonical_verdict": result.canonical_verdict,
        "tool_calls": result.state.tool_calls_used,
        "tool_sequence": list(result.tool_sequence),
        "report_path": str(reports_dir / (result.case_id + ".md")),
        "verifier_summary": artifact.verification.summary,
        "verifier_reference_summary": artifact.verification.reference_summary,
        **_action_summary(result.envelopes),
    }


async def _run_all(args: argparse.Namespace, settings: Settings, llm) -> list[dict[str, Any]]:
    trace_dir = Path(args.trace_dir) if args.trace_dir else Path(DEFAULT_TRACE_DIR)
    reports_dir = Path(args.reports)
    if reports_dir.resolve() == Path("reports").resolve():
        raise ValueError(
            "refusing to write demo reports directly into reports/ (that would "
            "overwrite historical artifacts); use a subdirectory such as reports/demo"
        )
    trace_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_reseed:
        print(
            f"\n[setup] reseeding disposable demo graph {settings.graph_name} "
            f"(seed {args.seed})"
        )
        reseed_demo_graph(settings, seed=args.seed, trace_dir=trace_dir)
        print("[setup] graph rebuilt deterministically; a prior demo freeze cannot leak in")

    graph = get_graph(settings)
    selected = _select_scenarios(args.scenarios)
    summaries: list[dict[str, Any]] = []
    for index, scenario in enumerate(selected, start=1):
        summaries.append(
            await _run_scenario(
                scenario, index=index, total=len(selected),
                settings=settings, llm=llm,
                trace_dir=trace_dir, reports_dir=reports_dir, graph=graph,
            )
        )
    return summaries


def _select_scenarios(spec: str) -> list[Scenario]:
    if not spec or spec.strip().lower() == "all":
        return list(SCENARIOS)
    wanted = {s.strip().upper() for s in spec.split(",") if s.strip()}
    selected = [s for s in SCENARIOS if s.key in wanted]
    missing = wanted - {s.key for s in selected}
    if missing:
        raise ValueError(
            f"unknown scenario key(s) {sorted(missing)}; available: "
            f"{[s.key for s in SCENARIOS]}"
        )
    return selected


def print_summary(summaries: list[dict[str, Any]], *, mode_line: str) -> None:
    print("\n" + _RULE)
    print("DEMO SUMMARY")
    print(_RULE)
    print(f"mode: {mode_line}")
    for s in summaries:
        print(
            f"  {s['key']}. {s['alert_id']}  verdict={s['canonical_verdict']}  "
            f"action={s['action_status'] or 'n/a'}  frozen={s['frozen']}  "
            f"calls={s['tool_calls']}"
        )
        if s["blocking_reasons"]:
            print(f"      declined because: {_short(s['blocking_reasons'][0], 150)}")
        print(f"      report: {s['report_path']}")
    declined = [s for s in summaries if s["action_status"] == "declined"]
    applied = [s for s in summaries if s["action_status"] == "applied"]
    print(
        f"\n  {len(applied)} sandbox freeze(s) applied · "
        f"{len(declined)} correctly declined by the deterministic policy"
    )
    print(
        "  Honest scope: risk indicators for review, not legal findings. Phase 7 "
        "measured no tool-call saving from case memory."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.demo.run_demo",
        description="HABEAS Phase 8 CLI demo (synthetic data only).",
    )
    parser.add_argument("--mode", choices=["fake", "real", "auto"], default="fake",
                        help="fake (deterministic default), real, or auto")
    parser.add_argument("--provider", choices=["openai", "groq"], default=None,
                        help="real-LLM provider override")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--graph", default=None,
                        help=f"demo graph (default {DEMO_GRAPH_NAMESPACE}_s<seed>)")
    parser.add_argument("--scenarios", default="all",
                        help="all, or a comma-separated subset such as A,C")
    parser.add_argument("--reports", default=DEFAULT_REPORTS_DIR,
                        help="report output directory (a subdirectory of reports/)")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--tool-budget", type=int, default=None)
    parser.add_argument("--no-reseed", action="store_true",
                        help="keep the existing demo graph (no reseed)")
    parser.add_argument("--json-out", default=None,
                        help="also write a JSON summary for this run")
    return parser


def _make_llm(settings: Settings, *, mode: str, provider: str | None):
    if mode == "fake":
        return FakeLLM()
    return make_llm(settings, mode="real", provider=provider)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    base = load_settings()
    mode = resolve_mode(args.mode, base)
    graph_name = args.graph or demo_graph_name(args.seed)

    print(_RULE)
    print("HABEAS — graph-native fraud investigation demo")
    print(_RULE)
    print("SYNTHETIC DATA ONLY. No real accounts or payment systems are touched.")
    print("Findings are risk indicators for human review, not legal conclusions.")
    print(f"graph: {graph_name} (disposable demo graph)")

    try:
        assert_demo_graph(graph_name)
    except ValueError as exc:
        print(f"\nrefused: {exc}", file=sys.stderr)
        return 2

    settings = demo_settings(
        base, graph_name=graph_name, mode=mode, tool_budget=args.tool_budget
    )

    try:
        llm = _make_llm(settings, mode=mode, provider=args.provider)
    except LLMConfigurationError as exc:
        print(f"\nconfiguration error: {exc}", file=sys.stderr)
        print(
            "run with the default --mode fake for the deterministic offline demo",
            file=sys.stderr,
        )
        return 3

    print(f"mode: {mode_label(mode, llm)}")

    try:
        # one event loop for the whole run keeps the provider client on one loop
        summaries = asyncio.run(_run_all(args, settings, llm))
    except ValueError as exc:
        print(f"\nrefused: {exc}", file=sys.stderr)
        return 2

    print_summary(summaries, mode_line=mode_label(mode, llm))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())