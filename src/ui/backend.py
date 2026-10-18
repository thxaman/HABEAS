"""UI backend integration (Phase 8.5) — thin wrappers over existing interfaces.

Everything here reuses the established entry points and result schemas:

- investigation + actions: ``src.agent.workflow.run_case_workflow`` (one MCP
  stdio session; the agent still reaches the graph only through MCP);
- eligibility/simulation/verification: the Phase 5 action envelopes the
  workflow already records;
- report + independent verification: ``src.report.builder/verifier``;
- demo-data reset: the guarded ``src.demo.run_demo.reseed_demo_graph`` (the
  same disposable ``HABEAS_demo_showcase*`` namespace and reset guards).

Safety rules that the UI depends on:

- no function here accepts a graph name from the caller: the target graph is
  always the demo's own disposable showcase graph, so the UI cannot point the
  agent at an arbitrary graph;
- no Cypher is accepted, composed, or executed here;
- the reseed is explicit and never called automatically on import or run;
- nothing here reads the ground-truth manifest, and no credential ever enters
  a result snapshot.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from src.agent.llm import FakeLLM, LLMConfigurationError, make_llm
from src.agent.state import (
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_EVIDENCE_SUFFICIENT,
    TERMINATION_NO_DISCRIMINATING_QUERY,
)
from src.agent.mcp_client import call_tool_sync
from src.agent.workflow import evidence_stance, run_case_workflow
from src.config import Settings, load_settings
from src.db import get_graph
from src.demo import run_demo
from src.ui import view_models as _vm
from src.report.builder import build_report, write_report

# The UI's own working output, mirroring the CLI demo's reports/demo:
# a subdirectory of reports/, never the historical reports/ root.
UI_REPORTS_DIR = "reports/ui"
UI_TRACE_DIR = "logs/ui"

_COMPLETED = (TERMINATION_EVIDENCE_SUFFICIENT, TERMINATION_NO_DISCRIMINATING_QUERY)


# ------------------------------------------------------------------ settings

#: Streamlit Community Cloud exposes configured secrets via ``st.secrets``.
#: These keys, when present there and not already in the environment, are
#: promoted into ``os.environ`` so the FalkorDB Cloud connection settings
#: resolve exactly once and are inherited by the MCP server subprocess.
_STREAMLIT_SECRET_KEYS = (
    "FALKORDB_HOST",
    "FALKORDB_PORT",
    "FALKORDB_USERNAME",
    "FALKORDB_PASSWORD",
    "FALKORDB_TLS",
    "FALKORDB_QUERY_TIMEOUT_MS",
    "GRAPH_NAME",
)


def load_streamlit_secrets_into_env() -> list[str]:
    """Promote FALKORDB_*/GRAPH_NAME secrets into the environment (idempotent).

    A no-op outside Streamlit (no secrets file, no runtime): existing
    environment variables always win. Values are moved into the process
    environment only — they are never logged or echoed.
    """
    promoted: list[str] = []
    try:  # Streamlit is only available in the app process
        import streamlit as st

        secrets = st.secrets
        for key in _STREAMLIT_SECRET_KEYS:
            try:
                value = secrets[key]
            except Exception:
                continue  # key not configured in this deployment
            if key not in os.environ and str(value).strip() != "":
                os.environ[key] = str(value)
                promoted.append(key)
    except Exception:
        return promoted
    return promoted


def ui_settings(
    base: Settings | None = None,
    *,
    mode: str = "fake",
    tool_budget: int | None = None,
) -> Settings:
    """Settings for the UI's disposable demo graph (never caller-named)."""
    load_streamlit_secrets_into_env()
    base = base or load_settings()
    return run_demo.demo_settings(
        base,
        graph_name=run_demo.demo_graph_name(run_demo.DEFAULT_SEED),
        mode=mode,
        tool_budget=tool_budget,
    )


def credentials_configured(settings: Settings) -> bool:
    """Whether the resolved real provider is fully configured (no key values)."""
    return run_demo._credentials_configured(settings)


def resolve_mode(requested: str, settings: Settings) -> str:
    return run_demo.resolve_mode(requested, settings)


def make_ui_llm(settings: Settings, mode: str):
    if mode == "fake":
        return FakeLLM()
    return make_llm(settings, mode="real")


# ----------------------------------------------------------------- scenarios


def list_scenarios() -> list[dict[str, str]]:
    """The curated demo scenarios (names/stakes only — never outcomes)."""
    return [
        {"key": s.key, "alert_id": s.alert_id, "title": s.title, "stake": s.stake}
        for s in run_demo.SCENARIOS
    ]


def scenario_for(alert_id: str) -> dict[str, str] | None:
    return next((s for s in list_scenarios() if s["alert_id"] == alert_id), None)


# -------------------------------------------------------------- demo data


def reset_demo_graph(
    settings: Settings,
    *,
    seed: int = run_demo.DEFAULT_SEED,
    trace_dir: str | Path | None = None,
) -> dict[str, Any]:
    """EXPLICIT, guarded rebuild of the disposable demo graph (never automatic)."""
    trace = Path(trace_dir or UI_TRACE_DIR)
    trace.mkdir(parents=True, exist_ok=True)
    run_demo.reseed_demo_graph(settings, seed=seed, trace_dir=trace)
    return {"graph_name": settings.graph_name, "seed": seed, "reset": True}


# ------------------------------------------------------------- empty snapshot


def _empty_snapshot(
    alert_id: str,
    *,
    status: str,
    settings: Settings,
    mode_label: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "alert_id": alert_id,
        "graph_name": settings.graph_name,
        "mode_label": mode_label,
        "status": status,            # "ok" | "failed" | "configuration_error"
        "error": error,
        "errors": [error] if error else [],
        "elapsed_s": None,
        "case_id": None,
        "canonical_verdict": None,
        "investigation_verdict": None,
        "termination_reason": None,
        "completed": False,
        "tool_calls": 0,
        "tool_budget": settings.tool_budget,
        "tool_sequence": [],
        "signal_families": [],
        "primary_entities": [],
        "hypotheses": [],
        "claims": [],
        "evidence": [],
        "limitations": [],
        "truncated_evidence": [],
        "policy": None,
        "action": None,
        "simulation": None,
        "post_action": None,
        "report": None,
    }


def _evidence_entry(record) -> dict[str, Any]:
    return {
        "evidence_id": record.evidence_id,
        "tool": record.tool,
        "ok": record.ok,
        "error": record.error,
        "stance": evidence_stance(record),
        # Phase 6.1 multi-query records: keep EVERY query id/param/hash/count,
        # never only the first one.
        "query_ids": list(record.query_ids),
        "params": list(record.params),
        "rows_hash": list(record.rows_hash),
        "row_count": list(record.row_count),
        "truncated": list(record.truncated),
        "entity_ids": list(record.entity_ids),
        "derived": dict(record.derived or {}),
        "interpretation": record.interpretation,
    }


def _action_view(envelopes: dict[str, Any]) -> dict[str, Any] | None:
    payload = (envelopes.get("freeze_account") or {}).get("result") or {}
    if not payload:
        return None
    policy = payload.get("policy") or {}
    return {
        "action_status": payload.get("action_status"),
        "frozen": bool(payload.get("frozen")),
        "action_id": payload.get("action_id"),
        "mutation_performed": bool(payload.get("mutation_performed")),
        "allowed": policy.get("allowed"),
        "policy_version": policy.get("policy_version"),
        "verdict": policy.get("verdict"),
        "checks": dict(policy.get("checks") or {}),
        "blocking_reasons": list(policy.get("blocking_reasons") or []),
        "caveats": list(policy.get("caveats") or []),
        "signal_families": list(policy.get("signal_families") or []),
        "target_relevant_families": list(policy.get("target_relevant_families") or []),
        "distinct_provenance_families": list(
            policy.get("distinct_provenance_families") or []
        ),
        "qualifying_evidence_ids": list(policy.get("qualifying_evidence_ids") or []),
    }


def _simulation_view(envelopes: dict[str, Any]) -> dict[str, Any] | None:
    result = (envelopes.get("simulate_freeze") or {}).get("result") or {}
    if not result:
        return None
    metrics = result.get("metrics") or {}
    return {
        "simulation_id": result.get("simulation_id"),
        "simulation_complete": metrics.get("simulation_complete"),
        "direct_outgoing_volume": metrics.get("direct_outgoing_volume"),
        "downstream_accounts_before": metrics.get("downstream_accounts_before"),
        "downstream_accounts_after": metrics.get("downstream_accounts_after"),
        "newly_unreachable_count": metrics.get("newly_unreachable_count"),
        "active_edges_disabled": metrics.get("active_edges_disabled"),
        "active_edges_total": metrics.get("active_edges_total"),
        "metric_semantics": dict(metrics.get("metric_semantics") or {}),
    }


def _post_action_view(envelopes: dict[str, Any]) -> dict[str, Any] | None:
    result = (envelopes.get("verify_freeze") or {}).get("result") or {}
    if not result:
        return None
    return {
        "account_id": result.get("account_id"),
        "account_status": result.get("account_status"),
        "verified": result.get("verified"),
        "checks": dict(result.get("checks") or {}),
    }


def _report_view(artifact) -> dict[str, Any]:
    verification = getattr(artifact, "verification", None)
    claims = [
        {
            # original ids are preserved verbatim (report/audit fidelity);
            # local_evidence_id is the normalized UI matching key only.
            "evidence_id": c.evidence_id,
            "local_evidence_id": _vm.normalize_evidence_id(c.evidence_id),
            "status": c.status,
            "reason": c.reason,
            "source_count": c.source_count,
            "compound": c.compound,
            "checked_source_queries": list(c.checked_source_queries),
            "unavailable_source_queries": list(c.unavailable_source_queries),
            "mismatched_source_queries": list(c.mismatched_source_queries),
            "sources": [
                {"query_id": s.query_id, "status": s.status, "reason": s.reason}
                for s in c.sources
            ],
        }
        for c in (verification.claims if verification else [])
    ]
    return {
        "case_id": artifact.case_id,
        "report_version": artifact.report_version,
        "markdown": artifact.markdown,
        "summary": dict(verification.summary or {}) if verification else {},
        "reference_summary": dict(verification.reference_summary or {}) if verification else {},
        "claims": claims,
        "notes": list(verification.notes or []) if verification else [],
        "written_to": None,  # filled by the caller after write_report
    }


# ----------------------------------------------------------------- main entry


def run_scenario(
    alert_id: str,
    *,
    settings: Settings | None = None,
    mode: str = "fake",
    trace_dir: str | Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one scenario end-to-end and return a plain, UI-safe snapshot.

    Never raises: configuration problems and failures come back as a snapshot
    with ``status`` = ``configuration_error`` / ``failed`` and a message that
    names the missing variable (never its value).
    """
    settings = settings or ui_settings(mode=mode)
    trace = Path(trace_dir or UI_TRACE_DIR)
    trace.mkdir(parents=True, exist_ok=True)

    try:
        llm = make_ui_llm(settings, mode)
    except LLMConfigurationError as exc:
        return _empty_snapshot(
            alert_id, status="configuration_error", settings=settings,
            mode_label=f"real provider ({settings.llm_provider}) — not configured",
            error=str(exc),
        )

    started = time.perf_counter()
    try:
        result = asyncio.run(
            run_case_workflow(
                alert_id,
                settings=settings,
                graph_name=settings.graph_name,
                llm=llm,
                trace_dir=trace,
                case_memory=True,
                freeze=True,
                on_event=on_event,
            )
        )
    except Exception as exc:  # MCP/LLM/graph failures are states, not crashes
        snapshot = _empty_snapshot(
            alert_id, status="failed", settings=settings,
            mode_label=run_demo.mode_label(mode, llm),
            error=f"{type(exc).__name__}: {exc}",
        )
        snapshot["elapsed_s"] = round(time.perf_counter() - started, 3)
        snapshot["termination_reason"] = "unrecoverable_error"
        return snapshot

    state = result.state
    snapshot = _empty_snapshot(
        alert_id, status="ok", settings=settings,
        mode_label=run_demo.mode_label(mode, llm),
    )
    snapshot.update(
        {
            "elapsed_s": round(time.perf_counter() - started, 3),
            "case_id": result.case_id,
            "canonical_verdict": result.canonical_verdict,
            "investigation_verdict": result.investigation.verdict,
            "termination_reason": state.termination_reason,
            "completed": state.termination_reason in _COMPLETED,
            "tool_calls": int(state.tool_calls_used),
            "tool_budget": int(state.tool_budget),
            "tool_sequence": list(result.tool_sequence),
            "signal_families": list(state.signal_families),
            "primary_entities": list(state.primary_entity_ids),
            "hypotheses": [
                {
                    "name": h.name,
                    "status": h.status,
                    "supporting_evidence": list(h.supporting_evidence),
                    "contradicting_evidence": list(h.contradicting_evidence),
                    "uncertainties": list(h.uncertainties),
                    "rationale": h.rationale,
                }
                for h in state.hypotheses
            ],
            "claims": [
                {"claim": c.claim, "status": c.status,
                 "evidence_ids": list(c.evidence_ids)}
                for c in state.claims
            ],
            "evidence": [_evidence_entry(r) for r in state.evidence],
            "limitations": list(state.limitations),
            "truncated_evidence": [
                r.evidence_id for r in state.evidence if any(r.truncated)
            ],
            "errors": [e for e in state.errors if e],
            "policy": _action_view(result.envelopes),
            "action": _action_view(result.envelopes),
            "simulation": _simulation_view(result.envelopes),
            "post_action": _post_action_view(result.envelopes),
        }
    )
    # A budget-exhausted run completed nothing; keep the honest state visible.
    if state.termination_reason == TERMINATION_BUDGET_EXHAUSTED:
        snapshot["completed"] = False

    # Report + independent verification reuse the Phase 6 stack unchanged.
    try:
        graph = get_graph(settings)
        artifact = build_report(
            graph, result.case_id, state=state,
            envelopes=result.envelopes, settings=settings,
        )
        snapshot["report"] = _report_view(artifact)
        snapshot["report"]["written_to"] = str(write_report(artifact, UI_REPORTS_DIR))
    except Exception as exc:  # a report failure must not hide the investigation
        snapshot["report"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "summary": {}, "reference_summary": {}, "claims": [],
            "notes": [], "markdown": None,
        }
    return snapshot

# ------------------------------------------------------- MCP-backed inbox


def _mcp_call(settings: Settings, tool: str, arguments: dict[str, Any]):
    """Call one read tool over the real MCP stdio transport."""
    return call_tool_sync(tool, arguments, settings=settings,
                          graph_name=settings.graph_name)


def load_alert_inbox(settings: Settings | None = None) -> dict[str, Any]:
    """Load the alert inbox from the graph through the MCP read tool.

    Never raises: an MCP/DB failure comes back as ``{"ok": False, "error": ...}``
    so the UI can show the real problem instead of an empty table.
    """
    settings = settings or ui_settings()
    try:
        result = _mcp_call(settings, "alert_inbox", {"limit": 200})
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "alerts": []}
    if not result.ok:
        return {"ok": False, "error": result.error or "alert_inbox failed",
                "alerts": []}
    payload = result.envelope.get("result") or {}
    return {
        "ok": True,
        "alerts": list(payload.get("alerts") or []),
        "interpretation": result.interpretation,
        "query_ids": [e.get("query_id") for e in
                      (result.envelope.get("query_evidence") or [])],
        "error": None,
    }


def load_neighborhood(
    account_id: str,
    *,
    settings: Settings | None = None,
    hops: int = 2,
    window_days: int = 365,
) -> dict[str, Any]:
    """Load the bounded neighbourhood subgraph for one account over MCP."""
    settings = settings or ui_settings()
    try:
        result = _mcp_call(settings, "alert_neighborhood", {
            "account_id": account_id, "hops": hops, "limit": 200,
        })
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not result.ok:
        return {"ok": False, "error": result.error or "alert_neighborhood failed"}
    payload = result.envelope.get("result") or {}
    return {
        "ok": True,
        "focal_account_id": payload.get("focal_account_id"),
        "accounts": list(payload.get("accounts") or []),
        "edges": list(payload.get("edges") or []),
        "transactions": list(payload.get("transactions") or []),
        "truncated": bool(payload.get("truncated")),
        "window": payload.get("window") or {},
        "interpretation": result.interpretation,
        "error": None,
    }


def resolve_alert_context(alert_id: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """Resolve an alert through the MCP read tool (never a hard-coded target)."""
    settings = settings or ui_settings()
    try:
        result = _mcp_call(settings, "alert_context", {"alert_id": alert_id})
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not result.ok:
        return {"ok": False, "error": result.error or "alert_context failed"}
    payload = result.envelope.get("result") or {}
    return {"ok": True, "context": payload, "interpretation": result.interpretation,
            "error": None}


def check_graph_status(settings: Settings | None = None) -> dict[str, Any]:
    """Check the graph connection with one approved, bounded read.

    Reuses the existing ``alert_inbox`` MCP read tool with ``limit=1`` — no
    new endpoint, no free-form query. Returns a record the UI renders as
    connected / unavailable, with the time of the check and the real query id
    so the status cannot be a constant label.
    """
    settings = settings or ui_settings()
    started = time.time()
    common = {
        "tool": "alert_inbox",
        "checked_at": time.time(),
        "query_ids": [],
    }
    try:
        result = _mcp_call(settings, "alert_inbox", {"limit": 1})
    except Exception as exc:
        return dict(common, state="unavailable",
                    duration_ms=round((time.time() - started) * 1000, 1),
                    error=f"{type(exc).__name__}: {exc}")
    if not result.ok:
        return dict(common, state="unavailable",
                    duration_ms=round((time.time() - started) * 1000, 1),
                    error=result.error or "alert_inbox failed")
    query_ids = [
        entry.get("query_id")
        for entry in (result.envelope.get("query_evidence") or [])
        if entry.get("query_id")
    ]
    return dict(common, state="connected", query_ids=query_ids,
                duration_ms=round((time.time() - started) * 1000, 1),
                error=None)


# ---------------------------------------------------------------- evaluation


EVAL_RESULTS_PATH = "eval/results.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _metric_value(block: Any) -> Any:
    """The display value of one metric block, or the raw rate when absent."""
    if isinstance(block, dict):
        return block.get("display", block.get("rate"))
    return block


def load_eval_summary(path: str | Path | None = None) -> dict[str, Any]:
    """Aggregate view of the offline evaluation artifact (no per-alert rows).

    Metrics are returned only when the artifact itself identifies the run by
    seed, mode and dataset; otherwise the caller gets an explicit reason and
    no numbers, so stale or unidentified metrics are never shown as current.
    The loader deliberately never reads the seeded manifest and never returns
    per-alert outcome rows — only aggregate counts already summarized in the
    artifact.
    """
    candidate = Path(path) if path is not None else _REPO_ROOT / EVAL_RESULTS_PATH
    shown_path = EVAL_RESULTS_PATH if path is None else str(path)
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "ok": False, "identified": False, "path": shown_path,
            "reason": (f"no evaluation artifact found at {shown_path}; "
                       "offline metrics are not shown"),
        }
    except Exception as exc:  # a malformed artifact is a state, not a crash
        return {
            "ok": False, "identified": False, "path": shown_path,
            "reason": (f"the evaluation artifact could not be read "
                       f"({type(exc).__name__}); offline metrics are not shown"),
        }
    if not isinstance(data, dict):
        return {
            "ok": False, "identified": False, "path": shown_path,
            "reason": ("the evaluation artifact has an unexpected shape; "
                       "offline metrics are not shown"),
        }

    run = data.get("run") or {}
    metrics = data.get("metrics") or {}
    seed = run.get("seed")
    mode = run.get("mode")
    graph_name = run.get("graph_name")
    identified = isinstance(seed, int) and bool(mode) and bool(graph_name)
    tool_calls = (metrics.get("tool_calls") or {}).get("all") or {}
    verifier = metrics.get("verifier") or {}
    summary: dict[str, Any] = {
        "ok": True,
        "identified": identified,
        "path": shown_path,
        "schema_version": data.get("schema_version"),
        "generated_at": data.get("generated_at"),
        "seed": seed,
        "mode": mode,
        "provider": run.get("provider"),
        "model": run.get("model"),
        "dataset": {
            "graph_name": graph_name,
            "manifest_seed": (run.get("manifest") or {}).get("seed"),
            "alerts_requested": run.get("alerts_requested"),
            "bounded_subset": bool(run.get("bounded_subset")),
        },
        "counts": dict(metrics.get("counts") or {}),
        "recall": {
            "strict": _metric_value((metrics.get("recall") or {}).get("strict")),
            "lenient": _metric_value((metrics.get("recall") or {}).get("lenient")),
        },
        "false_positive_rate": {
            "strict": _metric_value(
                (metrics.get("false_positive_rate") or {}).get("strict")),
        },
        "pattern_coverage": {
            "strict": _metric_value(
                (metrics.get("pattern_coverage") or {}).get("strict")),
        },
        "verifier": {
            "cases_verified": verifier.get("cases_verified"),
            "cases_with_error": verifier.get("cases_with_error"),
            "claims_checked": _metric_value(verifier.get("claims_checked")),
            "status_counts": dict(verifier.get("status_counts") or {}),
        },
        "tool_calls": {
            "mean": tool_calls.get("mean"),
            "median": tool_calls.get("median"),
        },
    }
    if not identified:
        summary["reason"] = (
            "the evaluation artifact is present but not identified by seed, "
            "mode and dataset; its metrics are withheld"
        )
    return summary


# --------------------------------------------------------- live investigation


class InvestigationJob:
    """Runs one investigation in a background thread while emitting events.

    The UI polls ``events`` and ``done`` so the activity timeline updates from
    *real* backend events as they happen. A Streamlit rerun never restarts the
    job and never reseeds anything — the job object simply lives in session
    state until it finishes.
    """

    def __init__(self, alert_id: str, *, settings: Settings, mode: str,
                 runner: Callable[..., dict[str, Any]] | None = None,
                 trace_dir: str | Path | None = None) -> None:
        self.alert_id = alert_id
        self.settings = settings
        self.mode = mode
        self._runner = runner or run_scenario
        self._trace_dir = trace_dir
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self.snapshot: dict[str, Any] | None = None
        self.error: str | None = None
        self.done = False
        self.started_at = time.time()

    # -- event sink -------------------------------------------------------
    def _on_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "InvestigationJob":
        def _work() -> None:
            try:
                self.snapshot = self._runner(
                    self.alert_id, settings=self.settings, mode=self.mode,
                    on_event=self._on_event, trace_dir=self._trace_dir,
                )
            except Exception as exc:  # surface, never crash the UI thread
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.done = True

        self._thread = threading.Thread(
            target=_work, name=f"habeas-investigation-{self.alert_id}", daemon=True
        )
        self._thread.start()
        return self

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return not self.done

    @property
    def elapsed_s(self) -> float:
        return round(time.time() - self.started_at, 3)

    # -- progressive view -------------------------------------------------
    def status(self) -> str:
        """One of running / completed / incomplete / failed."""
        if not self.done:
            return "running"
        if self.error:
            return "failed"
        snapshot = self.snapshot or {}
        if snapshot.get("status") in ("configuration_error", "failed"):
            return "failed"
        return "completed" if snapshot.get("completed") else "incomplete"

    def partial_snapshot(self) -> dict[str, Any]:
        """A live view built only from the events that have actually arrived.

        While the run is in progress this exposes the evidence records added so
        far, the latest hypothesis assessment, the accounts/legs each tool
        really returned, and the policy/simulation/action/verification results
        *when the workflow has reached them* — nothing is predicted or
        placeholder. When the real snapshot exists it is attached as ``final``.
        """
        live = _vm.live_view_from_events(self.events())
        live["alert_id"] = self.alert_id
        live["graph_name"] = self.settings.graph_name
        live["status"] = self.status()
        live["elapsed_s"] = self.elapsed_s
        live["error"] = self.error
        live["partial"] = self.running
        if self.snapshot is not None:
            live["final"] = self.snapshot
        return live


def start_investigation(
    alert_id: str,
    *,
    settings: Settings | None = None,
    mode: str = "fake",
    runner: Callable[..., dict[str, Any]] | None = None,
    trace_dir: str | Path | None = None,
    existing: "InvestigationJob | None" = None,
) -> InvestigationJob:
    """Start one investigation thread.

    ``existing`` lets a caller pass the session's current job: if that job is
    for the SAME alert and is still running it is returned untouched, so a
    Streamlit rerun can never spawn a duplicate investigation.
    """
    if existing is not None and existing.alert_id == alert_id and existing.running:
        return existing
    settings = settings or ui_settings(mode="real" if mode == "real" else "fake")
    return InvestigationJob(
        alert_id, settings=settings, mode=mode, runner=runner, trace_dir=trace_dir
    ).start()
