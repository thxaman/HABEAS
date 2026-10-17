"""Bounded LangGraph investigation workflow (PLAN.md §9).

Stages: TRIAGE -> FORM_HYPOTHESES -> INVESTIGATE -> UPDATE_EVIDENCE -> CHALLENGE
-> (INVESTIGATE | DECIDE) -> REPORT.

Guarantees enforced in code, not by prompt text:

- one MCP tool call per pass, counted whether it succeeds or fails;
- ``tool_budget`` is a hard stop, plus ``max_steps`` as a second loop bound;
- every call appends a JSONL trace record (see ``src/agent/trace.py``);
- a tool failure becomes a failed evidence record — never a claim, and never
  treated as absence of risk;
- an unrecoverable transport or LLM-provider error terminates the loop while
  preserving the evidence already collected;
- the LLM never decides what evidence means; ``analysis.py`` does.

The whole investigation runs inside ONE MCP session (opened by
``run_investigation``), so the server is started once per investigation.

Implementation note: LangGraph applies only the dict a node RETURNS; mutating
the incoming state in place is discarded. Every node therefore works on a
deep copy and returns its full update.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Protocol

from langgraph.graph import END, START, StateGraph

from src.agent import analysis
from src.agent.events import ProgressEmitter
from src.agent.llm import Decision, FakeLLM, InvestigationLLM, LLMProviderError
from src.agent.mcp_client import (
    ToolCallResult,
    open_session,
    result_from_raw,
    server_parameters,
)
from src.agent.state import (
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_EVIDENCE_SUFFICIENT,
    TERMINATION_NO_DISCRIMINATING_QUERY,
    TERMINATION_UNRECOVERABLE_ERROR,
    CaseState,
    EvidenceRecord,
    InvestigationResult,
    InvestigationStep,
)
from src.agent.trace import InvestigationTrace, TraceError
from src.config import Settings, get_settings

logger = logging.getLogger("habeas.loop")

STAGE_TRIAGE = "TRIAGE"
STAGE_FORM_HYPOTHESES = "FORM_HYPOTHESES"
STAGE_INVESTIGATE = "INVESTIGATE"
STAGE_UPDATE_EVIDENCE = "UPDATE_EVIDENCE"
STAGE_CHALLENGE = "CHALLENGE"
STAGE_DECIDE = "DECIDE"
STAGE_REPORT = "REPORT"

TRIAGE_TOOL = "alert_context"


class ToolClient(Protocol):
    """Minimal interface the loop needs; satisfied by the MCP session client."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        ...

    async def list_tools(self) -> list[dict[str, Any]]:
        ...


class SessionToolClient:
    """Adapts a live MCP ``ClientSession`` to :class:`ToolClient`."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        raw = await self._session.call_tool(name, arguments)
        return result_from_raw(name, raw)

    async def list_tools(self) -> list[dict[str, Any]]:
        listed = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in listed.tools
        ]


class InvestigationRunner:
    """Holds the dependencies the graph nodes need (client, llm, trace).

    ``case_memory=False`` withholds the ``prior_cases`` tool from the agent's
    tool interface for that run (the controlled memory-disabled condition used
    by the Phase 7 paired comparison); every other tool is unchanged.
    """

    def __init__(
        self,
        *,
        client: ToolClient,
        llm: InvestigationLLM,
        trace: InvestigationTrace,
        max_steps: int | None = None,
        case_memory: bool = True,
        emitter: ProgressEmitter | None = None,
    ) -> None:
        self.client = client
        self.llm = llm
        self.trace = trace
        self.available_tools: list[dict[str, Any]] = []
        self._max_steps_override = max_steps
        self.case_memory = bool(case_memory)
        # Progress events describe the real workflow to a UI consumer; with no
        # emitter they are simply not produced.
        self.emitter = emitter or ProgressEmitter()

    # ---- helpers ---------------------------------------------------------

    def _record_trace(self, state: CaseState, **kwargs: Any) -> None:
        """Write a trace record; trace failures are recorded, never hidden."""
        try:
            self.trace.tool_call(state=state, **kwargs)
        except TraceError as exc:
            state.errors.append(f"trace write failed: {exc}")
            logger.error("trace write failed for case %s: %s", state.case_id, exc)

    def _record_stage(self, *, state: CaseState, stage: str, note: str = "") -> None:
        """Write a stage record; a trace failure is surfaced, never swallowed.

        A broken audit trail must be visible in the state and on stderr, but it
        must not abort an investigation that is otherwise producing evidence.
        """
        try:
            self.trace.stage(state=state, stage=stage, note=note)
        except TraceError as exc:
            state.errors.append(f"trace write failed: {exc}")
            logger.error("trace write failed for case %s: %s", state.case_id, exc)

    def _emit_evidence(self, record, *, stage: str, state: CaseState) -> None:
        """Emit an 'evidence' event built only from the record's real content.

        The meta carries the record's own id/query ids/row counts/truncation,
        its deterministic stance, and the accounts and directed legs the tool
        actually returned — so a UI can reveal evidence and highlight newly
        implicated graph elements while the run is still in progress.
        """
        nodes, edges = analysis.evidence_nodes_and_edges(record)
        self.emitter.emit(
            type="evidence", stage=stage, tool=record.tool, ok=record.ok,
            title=f"Evidence {record.evidence_id} added",
            detail=analysis.summarize_evidence(record) or record.interpretation,
            meta={
                "evidence_id": record.evidence_id,
                "tool": record.tool,
                "stance": analysis.evidence_stance(record),
                "query_ids": list(record.query_ids),
                "row_count": list(record.row_count),
                "truncated": list(record.truncated),
                "nodes": nodes,
                "edges": edges,
                "call": state.tool_calls_used,
                "budget": state.tool_budget,
            },
        )

    def _failed_evidence(
        self,
        state: CaseState,
        tool: str,
        stage: str,
        arguments: dict[str, Any],
        error: str,
    ) -> EvidenceRecord:
        record = EvidenceRecord(
            evidence_id=f"EV-{len(state.evidence) + 1:04d}",
            tool=tool,
            stage=stage,
            ok=False,
            error=error,
            params=[dict(arguments)],
            entity_ids=list(state.primary_entity_ids),
        )
        self.emitter.emit(
            type="error", stage=stage, tool=tool, ok=False,
            title=f"{tool} transport failure",
            detail=error,
            meta={"evidence_id": record.evidence_id},
        )
        return record

    def _to_evidence(
        self,
        state: CaseState,
        tool: str,
        stage: str,
        arguments: dict[str, Any],
        result: ToolCallResult,
    ) -> EvidenceRecord:
        envelope = result.envelope or {}
        evidence_id = f"EV-{len(state.evidence) + 1:04d}"
        queries = envelope.get("query_evidence") or []
        query_ids = [e.get("query_id", "") for e in queries]
        rows_hash = [e.get("rows_hash", "") for e in queries]
        row_count = [int(e.get("row_count") or 0) for e in queries]
        truncated = [bool(e.get("truncated")) for e in queries]
        params = [dict(e.get("params") or {}) for e in queries]
        bounds = [dict(e.get("bound") or {}) for e in queries]

        deriver = analysis.DERIVERS.get(tool)
        derived: dict[str, Any] = {}
        if result.ok and envelope and deriver is not None:
            target = arguments.get("account_id") or arguments.get("entity_id")
            derived = deriver(envelope, target)

        entity_ids = list(state.primary_entity_ids)
        for value in (arguments.get("account_id"), arguments.get("entity_id")):
            if value and value not in entity_ids:
                entity_ids.append(value)
        if isinstance(derived.get("entity_ids"), list):
            for value in derived["entity_ids"]:
                if value and value not in entity_ids:
                    entity_ids.append(value)

        error = result.error
        if not result.ok and envelope:
            error = (
                (envelope.get("error") or {}).get("message")
                or error
                or "tool returned ok=false"
            )

        return EvidenceRecord(
            evidence_id=evidence_id,
            tool=tool,
            stage=stage,
            ok=result.ok,
            interpretation=result.interpretation,
            query_ids=query_ids,
            params=params,
            rows_hash=rows_hash,
            row_count=row_count,
            truncated=truncated,
            bounds=bounds,
            entity_ids=entity_ids,
            derived=derived,
            error=error,
            duration_ms=float(envelope.get("duration_ms") or 0.0),
        )

    # ---- nodes (each returns its full update) ----------------------------

    async def triage(self, state: CaseState) -> dict[str, Any]:
        s = state.model_copy(deep=True)
        s.stage = STAGE_TRIAGE
        self._record_stage(state=s, stage=STAGE_TRIAGE)
        self.emitter.emit(
            type="stage", stage=STAGE_TRIAGE,
            title="Triage — resolving the alert",
            detail=f"alert {s.alert_id} to trigger transaction and implicated accounts",
        )
        if s.budget_exhausted():
            s.termination_reason = TERMINATION_BUDGET_EXHAUSTED
            self.emitter.emit(
                type="error", stage=STAGE_TRIAGE, ok=False,
                title="Tool budget exhausted before triage",
                detail=f"budget {s.tool_budget} already used",
            )
            return s.model_dump()

        arguments = {"alert_id": s.alert_id}
        self.emitter.emit(
            type="tool", phase="started", stage=STAGE_TRIAGE, tool=TRIAGE_TOOL,
            title=f"{TRIAGE_TOOL} — started",
            detail="resolving the alert to its trigger transaction and accounts",
            meta={"arguments": dict(arguments), "call": s.tool_calls_used + 1,
                  "budget": s.tool_budget},
        )
        result = await self.client.call_tool(TRIAGE_TOOL, arguments)
        s.tool_calls_used += 1
        record = self._to_evidence(s, TRIAGE_TOOL, STAGE_TRIAGE, arguments, result)
        s.evidence.append(record)
        self._emit_evidence(record, stage=STAGE_TRIAGE, state=s)
        s.investigation_steps.append(InvestigationStep(
            index=s.step_index, stage=STAGE_TRIAGE, tool=TRIAGE_TOOL,
            purpose="resolve the alert into its trigger transaction and entities",
            arguments=arguments, result_summary=analysis.summarize_evidence(record),
            hypothesis_update="initial targets identified from graph evidence",
            ok=record.ok, duration_ms=record.duration_ms,
        ))
        s.step_index += 1
        self.emitter.emit(
            type="tool", phase="completed", stage=STAGE_TRIAGE, tool=TRIAGE_TOOL,
            ok=record.ok,
            title=f"{TRIAGE_TOOL} → {'resolved' if record.ok else 'failed'}",
            detail=analysis.summarize_evidence(record),
            meta={
                "evidence_id": record.evidence_id,
                "query_ids": list(record.query_ids),
                "call": s.tool_calls_used,
                "budget": s.tool_budget,
            },
        )
        self._record_trace(
            s, stage=STAGE_TRIAGE, tool=TRIAGE_TOOL, arguments=arguments,
            ok=record.ok, query_ids=record.query_ids, rows_hash=record.rows_hash,
            row_count=record.row_count, truncated=record.truncated,
            duration_ms=record.duration_ms,
            transition="TRIAGE->FORM_HYPOTHESES" if record.ok else "TRIAGE->DECIDE",
            error=record.error,
        )
        if not record.ok:
            s.errors.append(f"alert_context failed: {record.error}")
            return s.model_dump()
        s.alert_context = record.derived
        s.primary_entity_ids = list(record.derived.get("entity_ids") or [])
        return s.model_dump()

    async def form_hypotheses(self, state: CaseState) -> dict[str, Any]:
        s = state.model_copy(deep=True)
        s.stage = STAGE_FORM_HYPOTHESES
        s.hypotheses = analysis.recompute_hypotheses(s)
        self._record_stage(state=s, stage=STAGE_FORM_HYPOTHESES,
                         note="competing explanations registered")
        self.emitter.emit(
            type="hypotheses", stage=STAGE_FORM_HYPOTHESES,
            title="Competing explanations registered",
            detail="; ".join(f"{h.name}={h.status}" for h in s.hypotheses),
            meta={"hypotheses": [{"name": h.name, "status": h.status}
                                 for h in s.hypotheses]},
        )
        return s.model_dump()

    async def investigate(self, state: CaseState) -> dict[str, Any]:
        s = state.model_copy(deep=True)
        s.stage = STAGE_INVESTIGATE
        s.pending_tool = None
        if s.budget_exhausted():  # hard bound #1: budget
            s.termination_reason = TERMINATION_BUDGET_EXHAUSTED
            self._record_stage(state=s, stage=STAGE_INVESTIGATE,
                             note="budget exhausted before selecting a tool")
            self.emitter.emit(
                type="error", stage=STAGE_INVESTIGATE, ok=False,
                title="Tool budget exhausted",
                detail=f"{s.tool_calls_used}/{s.tool_budget} calls used",
            )
            return s.model_dump()

        try:
            decision: Decision = await self.llm.decide(
                state=s, available_tools=self.available_tools, stage=STAGE_INVESTIGATE
            )
        except LLMProviderError as exc:
            # A provider/network/quota failure is unrecoverable for this run:
            # terminate with the evidence already collected instead of losing
            # the investigation (and letting the failure vanish from eval).
            s.errors.append(f"LLM provider failure: {exc}")
            s.termination_reason = TERMINATION_UNRECOVERABLE_ERROR
            self._record_stage(
                state=s, stage=STAGE_INVESTIGATE,
                note=f"LLM provider failure: {exc}",
            )
            self.emitter.emit(
                type="error", stage=STAGE_INVESTIGATE, ok=False,
                title="LLM provider failure",
                detail=str(exc),
            )
            return s.model_dump()
        if decision.kind == "finalize":
            s.termination_reason = TERMINATION_NO_DISCRIMINATING_QUERY
            self._record_stage(state=s, stage=STAGE_INVESTIGATE,
                             note=f"finalize: {decision.reason}")
            self.emitter.emit(
                type="decision", stage=STAGE_INVESTIGATE,
                title="No further discriminating query",
                detail=decision.reason or "the agent stopped collecting evidence",
            )
            return s.model_dump()

        tool = decision.tool
        if tool not in {t["name"] for t in self.available_tools}:
            s.errors.append(f"LLM requested unknown tool {tool!r}; ignored")
            s.termination_reason = TERMINATION_UNRECOVERABLE_ERROR
            self.emitter.emit(
                type="error", stage=STAGE_INVESTIGATE, ok=False,
                title="Model requested an unknown tool",
                detail=f"{tool!r} is not in the exposed MCP tool set; ignored",
            )
            return s.model_dump()

        s.pending_tool = {
            "tool": tool,
            "arguments": dict(decision.arguments or {}),
            "purpose": decision.purpose,
        }
        self.emitter.emit(
            type="tool", phase="started", stage=STAGE_INVESTIGATE, tool=tool,
            title=f"{tool} — started",
            detail=decision.purpose or "next discriminating question",
            meta={
                "arguments": dict(decision.arguments or {}),
                "question": decision.purpose or "",
                "call": s.tool_calls_used + 1,
                "budget": s.tool_budget,
            },
        )
        return s.model_dump()

    async def update_evidence(self, state: CaseState) -> dict[str, Any]:
        s = state.model_copy(deep=True)
        s.stage = STAGE_UPDATE_EVIDENCE
        pending = s.pending_tool
        if not pending:
            return s.model_dump()

        tool = pending["tool"]
        arguments = pending["arguments"]
        started = time.perf_counter()
        try:
            result = await self.client.call_tool(tool, arguments)
        except Exception as exc:  # transport-level failure
            elapsed = (time.perf_counter() - started) * 1000.0
            s.tool_calls_used += 1
            record = self._failed_evidence(
                s, tool, STAGE_UPDATE_EVIDENCE, arguments, f"{type(exc).__name__}: {exc}"
            )
            record.duration_ms = elapsed
            s.evidence.append(record)
            s.investigation_steps.append(InvestigationStep(
                index=s.step_index, stage=STAGE_UPDATE_EVIDENCE, tool=tool,
                purpose=pending.get("purpose", ""), arguments=arguments,
                result_summary=f"transport failure: {exc}", ok=False,
                duration_ms=elapsed,
            ))
            s.step_index += 1
            s.errors.append(f"{tool} transport failure: {exc}")
            s.termination_reason = TERMINATION_UNRECOVERABLE_ERROR
            s.pending_tool = None
            self._record_trace(
                s, stage=STAGE_UPDATE_EVIDENCE, tool=tool, arguments=arguments,
                ok=False, query_ids=[], rows_hash=[], row_count=[], truncated=[],
                duration_ms=elapsed, transition="UPDATE_EVIDENCE->DECIDE",
                error=str(exc)[:300],
            )
            return s.model_dump()

        s.tool_calls_used += 1
        record = self._to_evidence(s, tool, STAGE_UPDATE_EVIDENCE, arguments, result)
        s.evidence.append(record)
        self.emitter.emit(
            type="tool", phase="completed", stage=STAGE_UPDATE_EVIDENCE, tool=tool,
            ok=record.ok,
            title=f"{tool} → {'evidence captured' if record.ok else 'failed'}",
            detail=analysis.summarize_evidence(record),
            meta={
                "evidence_id": record.evidence_id,
                "query_ids": list(record.query_ids),
                "row_count": list(record.row_count),
                "truncated": list(record.truncated),
                "call": s.tool_calls_used,
                "budget": s.tool_budget,
            },
        )
        self._emit_evidence(record, stage=STAGE_UPDATE_EVIDENCE, state=s)
        if tool == "prior_cases" and record.ok:
            # Prior cases are leads to re-check; record their ids (deduplicated)
            # so the case layer can LINK them without inheriting their verdicts.
            for case in record.derived.get("cases") or []:
                lead_id = case.get("case_id")
                if lead_id and lead_id not in s.prior_case_leads:
                    s.prior_case_leads.append(lead_id)
        s.hypotheses = analysis.recompute_hypotheses(s)
        s.claims = analysis.recompute_claims(s)
        if not record.ok:
            s.errors.append(f"{tool} returned a failure envelope: {record.error}")

        s.investigation_steps.append(InvestigationStep(
            index=s.step_index, stage=STAGE_UPDATE_EVIDENCE, tool=tool,
            purpose=pending.get("purpose", ""), arguments=arguments,
            result_summary=analysis.summarize_evidence(record),
            hypothesis_update=f"{len(s.hypotheses)} hypotheses re-evaluated",
            ok=record.ok, duration_ms=record.duration_ms,
        ))
        s.step_index += 1
        s.pending_tool = None
        self._record_trace(
            s, stage=STAGE_UPDATE_EVIDENCE, tool=tool, arguments=arguments,
            ok=record.ok, query_ids=record.query_ids, rows_hash=record.rows_hash,
            row_count=record.row_count, truncated=record.truncated,
            duration_ms=record.duration_ms, transition="UPDATE_EVIDENCE->CHALLENGE",
            error=record.error,
        )
        return s.model_dump()

    async def challenge(self, state: CaseState) -> dict[str, Any]:
        """Name the strongest competing explanation and decide whether to continue."""
        s = state.model_copy(deep=True)
        s.stage = STAGE_CHALLENGE
        families = analysis.detect_signal_families(s)
        s.signal_families = sorted(families)
        benign = analysis.benign_discriminators(s)
        if len(s.signal_families) >= 2:
            note = (
                f"{len(s.signal_families)} families are supported; the strongest "
                "competing explanation remains a benign one, which the evidence must "
                "contradict before a verdict is issued"
            )
        elif benign:
            note = "benign discriminators observed: " + "; ".join(benign[:2])
        else:
            note = "competing explanations remain unresolved"
        self._record_stage(state=s, stage=STAGE_CHALLENGE, note=note)
        self.emitter.emit(
            type="hypotheses", stage=STAGE_CHALLENGE,
            title=f"Challenge — {len(s.signal_families)} signal family(ies)",
            detail=note,
            meta={
                "signal_families": list(s.signal_families),
                "benign_discriminators": list(benign[:3]),
                "hypotheses": [{"name": h.name, "status": h.status}
                               for h in s.hypotheses],
            },
        )

        max_steps = self._max_steps_override or (s.tool_budget + 6)
        if s.termination_reason is None:
            if s.tool_calls_used >= s.tool_budget:
                s.termination_reason = TERMINATION_BUDGET_EXHAUSTED
            elif s.step_index >= max_steps:
                s.termination_reason = TERMINATION_NO_DISCRIMINATING_QUERY
        return s.model_dump()

    async def decide(self, state: CaseState) -> dict[str, Any]:
        s = state.model_copy(deep=True)
        s.stage = STAGE_DECIDE
        s.hypotheses = analysis.recompute_hypotheses(s)
        s.claims = analysis.recompute_claims(s)
        verdict, rationale, families, limitations = analysis.compute_verdict(s)
        s.provisional_verdict = verdict
        s.rationale = rationale
        s.signal_families = families
        s.limitations = limitations
        if s.termination_reason is None:
            s.termination_reason = (
                TERMINATION_EVIDENCE_SUFFICIENT
                if len(families) >= 2
                else TERMINATION_NO_DISCRIMINATING_QUERY
            )
        self._record_stage(state=s, stage=STAGE_DECIDE,
                         note=f"verdict={verdict}; families={families}")
        self.emitter.emit(
            type="decision", stage=STAGE_DECIDE,
            title=f"Verdict — {verdict}",
            detail=rationale or "deterministic verdict computed from the evidence",
            meta={
                "verdict": verdict,
                "signal_families": list(families),
                "limitations": list(limitations or []),
                "termination_reason": s.termination_reason,
            },
        )
        return s.model_dump()

    async def report(self, state: CaseState) -> dict[str, Any]:
        s = state.model_copy(deep=True)
        s.stage = STAGE_REPORT
        self._record_stage(state=s, stage=STAGE_REPORT,
                         note=f"termination={s.termination_reason}")
        return s.model_dump()

    # ---- graph -----------------------------------------------------------

    def build(self):
        graph = StateGraph(CaseState)
        graph.add_node(STAGE_TRIAGE, self.triage)
        graph.add_node(STAGE_FORM_HYPOTHESES, self.form_hypotheses)
        graph.add_node(STAGE_INVESTIGATE, self.investigate)
        graph.add_node(STAGE_UPDATE_EVIDENCE, self.update_evidence)
        graph.add_node(STAGE_CHALLENGE, self.challenge)
        graph.add_node(STAGE_DECIDE, self.decide)
        graph.add_node(STAGE_REPORT, self.report)

        graph.add_edge(START, STAGE_TRIAGE)
        graph.add_conditional_edges(
            STAGE_TRIAGE,
            lambda s: STAGE_FORM_HYPOTHESES if s.primary_entity_ids else STAGE_DECIDE,
            {STAGE_FORM_HYPOTHESES: STAGE_FORM_HYPOTHESES, STAGE_DECIDE: STAGE_DECIDE},
        )
        graph.add_edge(STAGE_FORM_HYPOTHESES, STAGE_INVESTIGATE)
        graph.add_conditional_edges(
            STAGE_INVESTIGATE,
            lambda s: STAGE_UPDATE_EVIDENCE if s.pending_tool else STAGE_DECIDE,
            {STAGE_UPDATE_EVIDENCE: STAGE_UPDATE_EVIDENCE, STAGE_DECIDE: STAGE_DECIDE},
        )
        graph.add_edge(STAGE_UPDATE_EVIDENCE, STAGE_CHALLENGE)
        graph.add_conditional_edges(
            STAGE_CHALLENGE,
            lambda s: (
                STAGE_DECIDE
                if (s.termination_reason or s.budget_exhausted())
                else STAGE_INVESTIGATE
            ),
            {STAGE_INVESTIGATE: STAGE_INVESTIGATE, STAGE_DECIDE: STAGE_DECIDE},
        )
        graph.add_edge(STAGE_DECIDE, STAGE_REPORT)
        graph.add_edge(STAGE_REPORT, END)
        return graph.compile()


async def run_investigation(
    alert_id: str,
    *,
    settings: Settings | None = None,
    graph_name: str | None = None,
    llm: InvestigationLLM | None = None,
    client: ToolClient | None = None,
    trace_dir: Any = None,
    case_id: str | None = None,
    max_steps: int | None = None,
    case_memory: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    emitter: ProgressEmitter | None = None,
) -> InvestigationResult:
    """Run one bounded investigation over ONE long-lived MCP session.

    ``client`` may be injected for failure-path tests; the default opens a real
    stdio session to ``python -m src.tools.server``. ``case_memory=False``
    withholds the ``prior_cases`` tool from the agent (Phase 7 controlled
    memory-disabled condition); nothing else changes. Progress events
    describing the real workflow are delivered to ``on_event`` — or, when the
    caller already owns one (so investigation and action events share a single
    ordered stream), to the supplied ``emitter``.
    """
    settings = settings or get_settings()
    llm = llm or FakeLLM()
    case_id = case_id or f"CASE-{alert_id}"
    trace = InvestigationTrace(case_id, trace_dir)
    emitter = emitter or ProgressEmitter(on_event)

    initial = CaseState(
        alert_id=alert_id,
        case_id=case_id,
        tool_budget=settings.tool_budget,
        llm_mode=getattr(llm, "mode", "fake"),
        trace_path=str(trace.path),
    )

    async def _run(tool_client: ToolClient) -> CaseState:
        runner = InvestigationRunner(
            client=tool_client, llm=llm, trace=trace, max_steps=max_steps,
            case_memory=case_memory, emitter=emitter,
        )
        try:
            tools = await tool_client.list_tools()
            if not case_memory:
                tools = [t for t in tools if t.get("name") != "prior_cases"]
            runner.available_tools = tools
        except Exception as exc:
            initial.errors.append(f"could not list MCP tools: {exc}")
            initial.termination_reason = TERMINATION_UNRECOVERABLE_ERROR
            return initial
        app = runner.build()
        final = await app.ainvoke(initial)
        return CaseState.model_validate(final)

    if client is not None:
        final_state = await _run(client)
    else:
        params = server_parameters(settings, graph_name=graph_name)
        async with open_session(params=params) as session:
            final_state = await _run(SessionToolClient(session))

    verdict = final_state.provisional_verdict or "insufficient_evidence"
    return InvestigationResult(
        state=final_state,
        verdict=verdict,
        verdict_label=verdict.replace("_", " "),
        tool_sequence=final_state.tools_called(),
        termination_reason=(
            final_state.termination_reason or TERMINATION_NO_DISCRIMINATING_QUERY
        ),
        trace_path=final_state.trace_path,
    )