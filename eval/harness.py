"""Evaluation harness: disposable graphs, per-alert records, reproducibility.

Safety model (Phase 7 spec + PLAN.md §6):

- every evaluation graph is named ``HABEAS_demo_eval_*``, satisfying both the
  existing demo-safe prefix (``DEMO_GRAPH_PREFIX``) and this module's stricter
  evaluation-only prefix; ``assert_eval_graph`` refuses anything else, so the
  shared demo graph and arbitrary graphs are never seeded, reset, or mutated;
- resets go through the trusted seed code's ``plan_reset`` guard;
- the evaluation graph is rebuilt (reset + deterministic seed) at the start of
  every independent run, so sandbox mutations from a previous run cannot leak
  into the next one.

Per-alert records are JSON-safe dicts; they carry everything the metrics and
the scorecard need (including the ground-truth label, which stays inside the
evaluation output and is never forwarded to the agent — only the alert id is).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from src.agent.llm import FakeLLM, make_llm
from src.agent.loop import SessionToolClient
from src.agent.mcp_client import open_session, server_parameters
from src.agent.workflow import run_case_workflow
from src.agent.state import (
    TERMINATION_EVIDENCE_SUFFICIENT,
    TERMINATION_NO_DISCRIMINATING_QUERY,
)
from src.config import DEMO_GRAPH_PREFIX, Settings
from src.db import get_graph, run_query
from src.seed import generate, schema
from src.seed.patterns import AlertTruth

from eval import EvaluationError
from eval.metrics import SIGNAL_TO_FAMILY, outcome_tier

EVAL_GRAPH_PREFIX = "HABEAS_demo_eval"
_TAG_RE = re.compile(r"^[a-z0-9_]+$")

_VERIFIER_PASS_STATUSES = (
    "reproduced_current_state",
    "verified_against_recorded_evidence",
    "changed_after_action",
)


# --------------------------------------------------------------- graph names


def eval_graph_name(seed: int, tag: str) -> str:
    """Build a validated disposable-evaluation graph name."""
    if not _TAG_RE.match(tag):
        raise EvaluationError(
            f"evaluation graph tag {tag!r} must match {_TAG_RE.pattern}"
        )
    name = f"{EVAL_GRAPH_PREFIX}_{tag}_s{seed}"
    assert_eval_graph(name)
    return name


def assert_eval_graph(name: str) -> None:
    """Refuse any graph that is not an evaluation-only disposable graph."""
    if not (name.startswith(EVAL_GRAPH_PREFIX) and name.startswith(DEMO_GRAPH_PREFIX)):
        raise EvaluationError(
            f"refusing to touch graph {name!r}: the evaluation may only use its "
            f"own disposable graphs named {EVAL_GRAPH_PREFIX!r}+"
        )


def eval_settings(
    base: Settings,
    *,
    graph_name: str,
    mode: str | None = None,
    provider: str | None = None,
    tool_budget: int | None = None,
) -> Settings:
    """Settings for one evaluation run: eval graph, sandbox actions on."""
    assert_eval_graph(graph_name)
    if mode is not None and mode not in {"fake", "real"}:
        raise EvaluationError(f"unknown evaluation mode {mode!r}")
    if provider is not None and provider not in {"openai", "groq"}:
        raise EvaluationError(f"unknown LLM provider {provider!r}")
    updates: dict[str, Any] = {
        "graph_name": graph_name,
        "habeas_demo_mode": True,
        "habeas_enable_actions": True,
    }
    if mode is not None:
        updates["llm_mode"] = mode
    if provider is not None:
        updates["llm_provider"] = provider
    if tool_budget is not None:
        updates["tool_budget"] = tool_budget
    return Settings(**{**base.model_dump(), **updates})


def seed_eval_graph(
    settings: Settings,
    *,
    seed: int,
    graph_name: str,
    manifest_path: Path | None = None,
) -> generate.SeedRunResult:
    """Deterministically (re)build one disposable evaluation graph.

    The manifest is written to ``manifest_path`` (never to the canonical
    ``src/seed/ground_truth.json`` unless the caller explicitly asks for it),
    so alternative-seed runs cannot clobber the canonical seed-42 manifest.
    """
    assert_eval_graph(graph_name)
    if settings.graph_name != graph_name:
        raise EvaluationError(
            "seed_eval_graph requires settings whose graph_name matches the "
            "evaluation graph"
        )
    graph = get_graph(settings)
    run_query(graph, "RETURN 1 AS bootstrap", read_only=False)  # create key if missing
    action = generate.plan_reset(graph_name, True, schema.node_count(graph))
    if manifest_path is not None:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    return generate.seed(
        settings,
        seed=seed,
        reset=(action == "reset_and_seed"),
        graph=graph,
        manifest_path=manifest_path,
    )


def build_llm(settings: Settings, *, mode: str, provider: str | None = None):
    """Fake (deterministic) or the configured real provider."""
    if mode == "fake":
        return FakeLLM()
    return make_llm(settings, mode="real", provider=provider)


# ------------------------------------------------------------ per-alert runs


def _failed_record(truth: AlertTruth, runner_ctx: dict[str, Any], error: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "alert_id": truth.alert_id,
        "gt_label": truth.label,
        "gt_family": truth.family,
        "gt_signals": list(truth.signals),
        "gt_account_id": truth.account_id,
        "gt_expected_members": list(truth.expected_account_members),
        **runner_ctx,
        "status": "error",
        "error": error,
        "case_id": None,
        "canonical_verdict": None,
        "investigation_verdict": None,
        "termination_reason": None,
        "completed": False,
        "tool_calls": 0,
        "tool_sequence": [],
        "tool_steps": [],
        "signal_families": [],
        "claims": {"supported": 0, "contradicted": 0, "unverified": 0},
        "hypotheses": [],
        "limitations": [],
        "errors": [error],
        "truncated_evidence": [],
        "prior_cases_used": False,
        "prior_case_leads": [],
        "reused_case_ids": [],
        "case_created": None,
        "action": {"attempted": False},
        "simulation": {"present": False},
        "post_action_verification": None,
        "verifier": {"status": "skipped"},
        "outcome_tier": outcome_tier({"status": "error"}),
    }
    return record


class AlertRunner:
    """Runs one alert end-to-end and captures its evaluation record.

    ``workflow`` is injectable so isolation tests can prove that the only
    thing crossing the evaluation-to-agent boundary is the alert id plus
    runtime configuration.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        graph_name: str,
        seed: int,
        mode: str,
        llm,
        trace_dir: Path,
        provider: str | None = None,
        model: str | None = None,
        case_memory: bool = True,
        freeze: bool = True,
        verify: bool = True,
        workflow: Callable[..., Any] | None = None,
    ) -> None:
        assert_eval_graph(graph_name)
        self.settings = settings
        self.graph_name = graph_name
        self.seed = seed
        self.mode = mode
        self.provider = provider
        self.model = model
        self.llm = llm
        self.trace_dir = Path(trace_dir)
        self.case_memory = case_memory
        self.freeze = freeze
        self.verify = verify
        self.workflow = workflow or run_case_workflow

    # -- verifier ---------------------------------------------------------

    def _verify_case(self, case_id: str) -> dict[str, Any]:
        if not self.verify or not case_id:
            return {"status": "skipped"}
        try:
            from src.report.verifier import verify_case

            graph = get_graph(self.settings)
            verification = verify_case(graph, case_id)
            summary = {str(k): int(v) for k, v in (verification.summary or {}).items()}
            checked = sum(summary.values())
            passed = sum(summary.get(status, 0) for status in _VERIFIER_PASS_STATUSES)
            return {
                "status": "ran",
                "case_exists": bool(verification.exists),
                "claims_checked": checked,
                "claims_passed": passed,
                "summary": summary,
            }
        except Exception as exc:  # verifier problems are reported, never hidden
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    # -- one alert ---------------------------------------------------------

    def run(self, truth: AlertTruth) -> dict[str, Any]:
        context = {
            "seed": self.seed,
            "graph_name": self.graph_name,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "case_memory": self.case_memory,
        }
        started = time.perf_counter()
        result = None
        error: str | None = None
        try:
            result = asyncio.run(
                self.workflow(
                    truth.alert_id,  # the ONLY manifest-derived value that crosses
                    settings=self.settings,
                    llm=self.llm,
                    trace_dir=self.trace_dir,
                    case_memory=self.case_memory,
                    freeze=self.freeze,
                )
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        if result is None:
            record = _failed_record(truth, context, error or "workflow returned nothing")
            record["elapsed_s"] = round(elapsed, 3)
            return record

        state = result.state
        envelopes = result.envelopes
        freeze_env = envelopes.get("freeze_account") or {}
        freeze_payload = freeze_env.get("result") or {}
        policy = freeze_payload.get("policy") or {}
        checks = policy.get("checks") or {}
        failing_checks = sorted(name for name, ok in checks.items() if not ok)
        sim_env = envelopes.get("simulate_freeze") or {}
        sim_result = sim_env.get("result") or {}
        sim_metrics = sim_result.get("metrics") or {}
        verify_env = envelopes.get("verify_freeze") or {}
        verify_result = verify_env.get("result") or {}
        post_checks = verify_result.get("checks") or {}
        create_env = envelopes.get("create_case") or {}
        create_result = create_env.get("result") or {}

        claims: dict[str, int] = {"supported": 0, "contradicted": 0, "unverified": 0}
        for claim in state.claims:
            claims[claim.status] = claims.get(claim.status, 0) + 1

        termination = state.termination_reason
        record = {
            "alert_id": truth.alert_id,
            "gt_label": truth.label,
            "gt_family": truth.family,
            "gt_signals": list(truth.signals),
            "gt_account_id": truth.account_id,
            "gt_expected_members": list(truth.expected_account_members),
            **context,
            "elapsed_s": round(elapsed, 3),
            "status": "ok",
            "error": None,
            "case_id": result.case_id,
            "canonical_verdict": result.canonical_verdict,
            "investigation_verdict": result.investigation.verdict,
            "termination_reason": termination,
            "completed": termination in (
                TERMINATION_EVIDENCE_SUFFICIENT,
                TERMINATION_NO_DISCRIMINATING_QUERY,
            ),
            "tool_calls": int(state.tool_calls_used),
            "tool_sequence": list(result.tool_sequence),
            "tool_steps": [
                {"tool": step.tool, "arguments": step.arguments}
                for step in state.investigation_steps
                if step.tool
            ],
            "signal_families": list(state.signal_families),
            "claims": claims,
            "hypotheses": [
                {"name": h.name, "status": h.status} for h in state.hypotheses
            ],
            "limitations": list(state.limitations),
            "errors": list(state.errors),
            "truncated_evidence": [
                r.evidence_id for r in state.evidence if any(r.truncated)
            ],
            "prior_cases_used": "prior_cases" in (result.tool_sequence or []),
            "prior_case_leads": list(state.prior_case_leads),
            "reused_case_ids": list(create_result.get("reused_case_ids") or []),
            "case_created": create_result.get("created"),
            "action": {
                "attempted": bool(freeze_payload.get("action_status")),
                "status": freeze_payload.get("action_status"),
                "action_id": freeze_payload.get("action_id"),
                "frozen": bool(freeze_payload.get("frozen")),
                "eligible": (
                    bool(policy.get("allowed")) if policy else None
                ),
                "blocking_reasons": list(policy.get("blocking_reasons") or []),
                "failing_checks": failing_checks,
                "target_relevant_families": list(
                    policy.get("target_relevant_families") or []
                ),
                "distinct_provenance_families": list(
                    policy.get("distinct_provenance_families") or []
                ),
                "qualifying_evidence_ids": list(
                    policy.get("qualifying_evidence_ids") or []
                ),
            },
            "simulation": {
                "present": bool(sim_result),
                "complete": sim_metrics.get("simulation_complete"),
                "active_edges_disabled": sim_metrics.get("active_edges_disabled"),
                "newly_unreachable": sim_metrics.get("newly_unreachable_count"),
                "downstream_accounts_before": sim_metrics.get(
                    "downstream_accounts_before"
                ),
                "direct_outgoing_volume": sim_metrics.get("direct_outgoing_volume"),
            },
            "post_action_verification": (
                {
                    "verified": bool(verify_result.get("verified")),
                    "account_status": verify_result.get("account_status"),
                    "failing_checks": sorted(
                        name for name, ok in post_checks.items() if not ok
                    ),
                }
                if verify_result
                else None
            ),
            "verifier": self._verify_case(result.case_id),
        }
        record["outcome_tier"] = outcome_tier(record)
        return record


# ------------------------------------------- one-hop baseline ablation (§12)


async def _ablation_calls(
    settings: Settings, graph_name: str, target: str, *, window_days: int
):
    """First-hop profile vs narrow vs bounded cycle traversal, same MCP boundary.

    Every call is an approved read template invoked the exact way the
    investigation agent invokes it (same server, same schemas, same graph).
    """
    params = server_parameters(settings, graph_name=graph_name)
    async with open_session(params=params) as session:
        client = SessionToolClient(session)

        async def call(tool: str, arguments: dict[str, Any]):
            return await client.call_tool(tool, arguments)

        one_hop = await call(
            "entity_profile", {"entity_id": target, "window_days": window_days}
        )
        narrow = await call(
            "detect_cycle",
            {"account_id": target, "max_len": 3, "window_days": window_days},
        )
        bounded = await call(
            "detect_cycle",
            {"account_id": target, "max_len": 4, "window_days": window_days},
        )
    return one_hop, narrow, bounded


def _cycle_view(call, requested_max_len: int) -> dict[str, Any]:
    envelope = call.envelope or {}
    result = envelope.get("result") or {}
    members: set[str] = set()
    for cycle in result.get("cycles") or []:
        members.update(cycle.get("members") or [])
    return {
        "tool": "detect_cycle",
        "max_len": requested_max_len,
        "ok": bool(call.ok),
        "cycle_count": int(result.get("cycle_count") or 0),
        "corroborated_count": int(result.get("temporally_corroborated_count") or 0),
        "members": sorted(members),
        "truncated": bool(result.get("truncated")),
        "error": call.error,
    }


def _one_hop_view(call) -> dict[str, Any]:
    envelope = call.envelope or {}
    result = envelope.get("result") or {}
    incoming = result.get("incoming") or {}
    outgoing = result.get("outgoing") or {}
    return {
        "tool": "entity_profile",
        "ok": bool(call.ok),
        "found": bool(result.get("found")),
        "counterparties_in": int(incoming.get("distinct_counterparties") or 0),
        "counterparties_out": int(outgoing.get("distinct_counterparties") or 0),
        # By definition of the baseline: a one-hop profile lists direct
        # counterparties and cannot express that a path returns to its origin
        # (a 4-account loop needs 4 hops). The measured counterpart is the
        # narrow (<=3 accounts) traversal below.
        "closed_loop_visible": False,
        "error": call.error,
    }


def run_one_hop_ablation(
    *,
    truths: list[AlertTruth],
    settings: Settings,
    graph_name: str,
    window_days: int = 90,
) -> dict[str, Any] | None:
    """PLAN.md §12: show what bounded multi-hop traversal contributes.

    Runs only for alerts whose planted signals map to the *circulation*
    family (rings); other families are not ring patterns and are skipped.
    Returns ``None`` when no such alert was evaluated.
    """
    circulation = [
        truth
        for truth in truths
        if any(
            SIGNAL_TO_FAMILY.get(signal) == "circulation"
            for signal in truth.signals
        )
    ]
    if not circulation:
        return None

    per_alert: list[dict[str, Any]] = []
    for truth in circulation:
        try:
            one_hop, narrow, bounded = asyncio.run(
                _ablation_calls(
                    settings, graph_name, truth.account_id,
                    window_days=window_days,
                )
            )
            one = _one_hop_view(one_hop)
            narrow_view = _cycle_view(narrow, 3)
            bounded_view = _cycle_view(bounded, 4)
            if (
                bounded_view["corroborated_count"] > 0
                and narrow_view["corroborated_count"] == 0
            ):
                contribution = (
                    "ring invisible to the one-hop profile and to the "
                    "<=3-account traversal; found and temporally corroborated "
                    "by the bounded 4-account traversal"
                )
            elif bounded_view["cycle_count"] > 0:
                contribution = (
                    "bounded traversal found cycle evidence; visibility at "
                    "narrow depth differs — measured values recorded"
                )
            else:
                contribution = (
                    "no ring found by the bounded traversal on this alert — "
                    "recorded as measured"
                )
            per_alert.append(
                {
                    "alert_id": truth.alert_id,
                    "gt_family": truth.family,
                    "target": truth.account_id,
                    "one_hop": one,
                    "narrow_traversal": narrow_view,
                    "bounded_traversal": bounded_view,
                    "contribution": contribution,
                }
            )
        except Exception as exc:  # never hide an ablation failure
            per_alert.append(
                {
                    "alert_id": truth.alert_id,
                    "gt_family": truth.family,
                    "target": truth.account_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    checked = [item for item in per_alert if "error" not in item]
    aggregate = {
        "alerts_checked": len(per_alert),
        "errors": len(per_alert) - len(checked),
        "one_hop_closed_loop_visible": sum(
            1 for item in checked if item["one_hop"]["closed_loop_visible"]
        ),
        "narrow_traversal_corroborated": sum(
            1 for item in checked if item["narrow_traversal"]["corroborated_count"] > 0
        ),
        "bounded_traversal_corroborated": sum(
            1 for item in checked if item["bounded_traversal"]["corroborated_count"] > 0
        ),
    }
    if checked:
        aggregate["conclusion"] = (
            f"bounded multi-hop traversal temporally corroborated a planted ring "
            f"on {aggregate['bounded_traversal_corroborated']}/"
            f"{aggregate['alerts_checked']} circulation alert(s); the one-hop "
            f"profile alone can express closure on "
            f"{aggregate['one_hop_closed_loop_visible']}/{aggregate['alerts_checked']}, "
            f"and the narrow (<=3 accounts) search likewise "
            f"{aggregate['narrow_traversal_corroborated']}/{aggregate['alerts_checked']}."
        )
    else:
        aggregate["conclusion"] = "ablation could not run for any alert (see errors)"

    return {
        "definition": (
            "One-hop baseline vs bounded multi-hop traversal over the same MCP "
            "boundary and the same 90-day window. The one-hop profile "
            "(entity_profile) lists direct counterparties only; the narrow "
            "traversal (detect_cycle max_len=3) is limited to <=3-account "
            "cycles; the bounded traversal (detect_cycle max_len=4) is inside "
            "the validated 2..5 cycle-length range."
        ),
        "window_days": window_days,
        "alerts": per_alert,
        "aggregate": aggregate,
        "note": (
            "all calls are approved read templates; nothing here feeds the "
            "agent or changes any verdict, and counts are recorded as measured."
        ),
    }


def run_sweep(
    truths: list[AlertTruth],
    *,
    runner: AlertRunner,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run one alert after another (deterministic order) and collect records."""
    records: list[dict[str, Any]] = []
    for index, truth in enumerate(truths, start=1):
        record = runner.run(truth)
        records.append(record)
        if progress is not None:
            progress(index, len(truths), record)
    return records


# ------------------------------------------------------- reproducibility


_SEMANTIC_ACTION_FIELDS = (
    "attempted",
    "status",
    "action_id",
    "frozen",
    "eligible",
    "failing_checks",
    "blocking_reasons",
    "target_relevant_families",
    "distinct_provenance_families",
)


def semantic_view(record: dict[str, Any]) -> dict[str, Any]:
    """The deterministic projection of a record used for run comparisons.

    Wall-clock fields (``elapsed_s``) and anything computed from them are
    excluded; everything else — including tool steps with their arguments —
    must be byte-identical between two runs of the same seed and FakeLLM
    configuration on a freshly rebuilt graph.
    """
    action = record.get("action") or {}
    simulation = record.get("simulation") or {}
    verifier = record.get("verifier") or {}
    return {
        "alert_id": record.get("alert_id"),
        "status": record.get("status"),
        "error": record.get("error"),
        "case_id": record.get("case_id"),
        "case_created": record.get("case_created"),
        "canonical_verdict": record.get("canonical_verdict"),
        "investigation_verdict": record.get("investigation_verdict"),
        "termination_reason": record.get("termination_reason"),
        "completed": record.get("completed"),
        "tool_sequence": record.get("tool_sequence"),
        "tool_steps": record.get("tool_steps"),
        "signal_families": record.get("signal_families"),
        "claims": record.get("claims"),
        "hypotheses": record.get("hypotheses"),
        "errors": record.get("errors"),
        "truncated_evidence": record.get("truncated_evidence"),
        "prior_cases_used": record.get("prior_cases_used"),
        "prior_case_leads": record.get("prior_case_leads"),
        "reused_case_ids": record.get("reused_case_ids"),
        "action": {key: action.get(key) for key in _SEMANTIC_ACTION_FIELDS},
        "simulation": {
            key: simulation.get(key)
            for key in ("present", "complete", "active_edges_disabled", "newly_unreachable")
        },
        "verifier": (
            {"status": verifier.get("status"), "summary": verifier.get("summary")}
            if verifier
            else {}
        ),
    }


def semantic_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [semantic_view(r) for r in records], sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_runs(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> dict[str, Any]:
    """Field-level comparison of two sweeps (same seed, rebuilt graph)."""
    first_by_id = {r["alert_id"]: semantic_view(r) for r in first}
    second_by_id = {r["alert_id"]: semantic_view(r) for r in second}
    mismatched: list[dict[str, Any]] = []
    for alert_id in sorted(set(first_by_id) | set(second_by_id)):
        left = first_by_id.get(alert_id)
        right = second_by_id.get(alert_id)
        if left != right:
            differing = (
                sorted(
                    key
                    for key in set(left or {}) | set(right or {})
                    if (left or {}).get(key) != (right or {}).get(key)
                )
                if left is not None and right is not None
                else ["<missing record>"]
            )
            mismatched.append({"alert_id": alert_id, "fields": differing})
    return {
        "identical": not mismatched,
        "digest_first": semantic_digest(first),
        "digest_second": semantic_digest(second),
        "mismatched_alerts": mismatched,
    }
