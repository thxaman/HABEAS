"""Phase 5 end-to-end workflow (PLAN.md §1/§9).

ALERT -> INVESTIGATE -> RECORD CASE -> CITE EVIDENCE -> SIMULATE INTERVENTION
-> CHECK ELIGIBILITY -> EXECUTE OR DECLINE -> VERIFY GRAPH STATE

Everything goes through ONE MCP session: the investigation uses the read tools
and the action tools are called as MCP tools too, so the demo path exercises
the same boundary the LLM would use.

The freeze is never automatic: ``freeze_account`` re-derives eligibility from
the evidence persisted on the case and declines with structured reasons when
anything is missing. Declining is a successful outcome, not an error.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.events import ProgressEmitter
from src.agent.loop import SessionToolClient, run_investigation
from src.agent.mcp_client import ToolCallResult, open_session, server_parameters
from src.agent.policy import (
    VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW,
    escalation_verdict,
)
from src.agent.state import CaseState, InvestigationResult
from src.config import Settings, get_settings
from src.tools.actions import case_id_for_alert

# Evidence stance is an analysis concern; it lives in analysis.py so the
# investigation loop can label records as they are appended. Re-exported here
# for backward compatibility (src/ui/backend.py and the report layer import
# it from this module).
from src.agent.analysis import evidence_stance  # noqa: E402,F401


def _at(sequence: list[Any] | None, index: int, default: Any) -> Any:
    if not sequence or index >= len(sequence) or index < 0:
        return default
    return sequence[index]


def build_citations(state: CaseState, entity_id: str | None) -> list[dict[str, Any]]:
    """Turn the investigation's evidence into semantics-preserving citations.

    Phase 6.1: every citation carries the COMPLETE source manifest of the tool
    invocation — one record per executed approved query, with its parameters,
    interpolation bounds, result hash, row count and truncation flag — so the
    report/verifier layers can check each contributing query individually.
    The top-level fields mirror ``sources[0]`` for backward compatibility, and
    ``required_source_queries`` lists the claim-material query ids (every
    source except the observation-window anchor, which positions the window
    but cannot carry a claim on its own).
    """
    from src.agent.analysis import rapid_forwarding_for_account

    aggregated_rapid = any(
        rapid_forwarding_for_account(state, account).get("rapid_forward_count", 0) > 0
        for account in (state.primary_entity_ids or [])
    )
    citations: list[dict[str, Any]] = []
    for record in state.evidence:
        stance = evidence_stance(record)
        if record.tool == "trace_funds":
            stance = "supporting" if aggregated_rapid else "contradicting"
        sources: list[dict[str, Any]] = []
        for index, query_id in enumerate(record.query_ids or []):
            if not query_id:
                continue
            sources.append({
                "query_id": query_id,
                "params": _at(record.params, index, {}),
                "interpolation": _at(record.bounds, index, {}),
                "result_hash": _at(record.rows_hash, index, ""),
                "row_count": _at(record.row_count, index, 0),
                "truncated": bool(_at(record.truncated, index, False)),
            })
        if not sources:
            continue
        primary = sources[0]
        claim = (record.interpretation or f"{record.tool} evidence captured")[:300]
        required = list(dict.fromkeys(
            s["query_id"] for s in sources
            if s["query_id"] != "txn_max_ts_v1"
        ))
        citation: dict[str, Any] = {
            "evidence_id": record.evidence_id,
            "claim": claim,
            "stance": stance,
            "query_id": primary["query_id"],
            "params": primary["params"],
            "interpolation": primary["interpolation"],
            "rows": [],
            "derived": record.derived,
            "result_hash": primary["result_hash"],
            "row_count": primary["row_count"],
            "truncated": primary["truncated"],
            "interpretation": record.interpretation,
            "evidence_kind": "captured_snapshot",
            "source_tool": record.tool,
            "sources": sources,
            "source_count": len(sources),
            "required_source_queries": required,
        }
        if primary["query_id"] and entity_id:
            citation["entity_id"] = entity_id
        citations.append(citation)
    return [c for c in citations if c["query_id"]]


@dataclass
class CaseWorkflowResult:
    alert_id: str
    case_id: str
    investigation: InvestigationResult
    canonical_verdict: str
    envelopes: dict[str, dict[str, Any]] = field(default_factory=dict)
    frozen: bool = False
    freeze_declined: bool = False
    tool_sequence: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.alert_id}: canonical verdict {self.canonical_verdict}; "
            f"case {self.case_id}; "
            + ("freeze APPLIED" if self.frozen else "no freeze applied")
        )

    @property
    def state(self) -> CaseState:
        """The investigation's CaseState (convenience accessor)."""
        return self.investigation.state


async def run_case_workflow(
    alert_id: str,
    *,
    settings: Settings | None = None,
    graph_name: str | None = None,
    llm=None,
    trace_dir: Any = None,
    window_days: int = 90,
    hops: int = 3,
    freeze: bool = True,
    case_memory: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> CaseWorkflowResult:
    """Run the full investigate -> case -> cite -> simulate -> freeze -> verify chain.

    ``on_event`` receives the investigation's ordered progress events plus the
    action/verification stages, so a UI can render the real workflow (Phase
    8.5). Nothing here carries a secret or model chain-of-thought.
    """
    settings = settings or get_settings()
    case_id = case_id_for_alert(alert_id)

    effective = settings
    if graph_name and graph_name != settings.graph_name:
        effective = Settings(**{**settings.model_dump(), "graph_name": graph_name})

    emitter = ProgressEmitter(on_event)
    params = server_parameters(effective, graph_name=graph_name)
    async with open_session(params=params) as session:
        client = SessionToolClient(session)
        investigation = await run_investigation(
            alert_id, settings=effective, graph_name=graph_name,
            llm=llm, client=client, trace_dir=trace_dir, case_id=case_id,
            case_memory=case_memory, emitter=emitter,
        )
        state = investigation.state
        canonical = escalation_verdict(state, state.provisional_verdict)

        async def call(tool: str, arguments: dict[str, Any]) -> ToolCallResult:
            return await client.call_tool(tool, arguments)

        envelopes: dict[str, dict[str, Any]] = {}
        target = state.primary_entity_ids[0] if state.primary_entity_ids else None

        created = await call("create_case", {
            "alert_id": alert_id,
            "severity": (
                "high" if canonical == VERDICT_SUSPICION_SUPPORTED_FOR_REVIEW else "medium"
            ),
            "verdict": canonical,
            "summary": state.rationale[:400],
            "entity_ids": state.primary_entity_ids,
            # LEADS only: link earlier cases surfaced by prior_cases, excluding
            # this investigation's own case id. Verdicts are re-derived, never
            # inherited from the linked cases.
            "reuses_case_ids": [
                lead for lead in state.prior_case_leads
                if lead and lead != case_id
            ],
        })
        envelopes["create_case"] = created.envelope or {}
        create_result = (created.envelope or {}).get("result") or {}
        emitter.emit(
            type="action", stage="RECORD_CASE",
            title=f"Case recorded — {case_id}",
            detail=(f"created={create_result.get('created')}; "
                    f"reused_leads={create_result.get('reused_case_ids') or []}"),
            ok=bool(created.ok),
            meta={"case_id": case_id, "verdict": canonical,
                  "created": create_result.get("created"),
                  "reused_case_ids": list(create_result.get("reused_case_ids") or [])},
        )

        citations = build_citations(state, target)
        if citations:
            cited = await call("cite_evidence",
                               {"case_id": case_id, "citations": citations})
            envelopes["cite_evidence"] = cited.envelope or {}
            emitter.emit(
                type="action", stage="CITE_EVIDENCE",
                title=f"Evidence cited — {len(citations)} citation(s)",
                detail="each citation links an evidence id to its approved source query ids",
                ok=bool(cited.ok),
                meta={"citation_count": len(citations),
                      "evidence_ids": [c["evidence_id"] for c in citations]},
            )

        result = CaseWorkflowResult(
            alert_id=alert_id, case_id=case_id, investigation=investigation,
            canonical_verdict=canonical, envelopes=envelopes,
            tool_sequence=list(investigation.tool_sequence),
        )

        if not freeze or not target:
            result.envelopes["freeze_account"] = {
                "ok": False,
                "interpretation": "freeze not attempted (disabled or no target entity)",
            }
            return result

        simulated = await call("simulate_freeze", {
            "case_id": case_id, "account_id": target,
            "window_days": window_days, "hops": hops,
        })
        envelopes["simulate_freeze"] = simulated.envelope or {}
        sim_metrics = ((simulated.envelope or {}).get("result") or {}).get("metrics") or {}
        emitter.emit(
            type="action", stage="SIMULATE",
            title="Counterfactual simulation (read-only, no mutation)",
            detail=(f"complete={sim_metrics.get('simulation_complete')}; "
                    f"downstream before={sim_metrics.get('downstream_accounts_before')}; "
                    f"edges that would be disabled={sim_metrics.get('active_edges_disabled')}"),
            ok=bool(simulated.ok),
            meta={"counterfactual": True, "target": target,
                  "simulation_complete": sim_metrics.get("simulation_complete"),
                  "direct_outgoing_volume": sim_metrics.get("direct_outgoing_volume"),
                  "downstream_accounts_before": sim_metrics.get("downstream_accounts_before"),
                  "newly_unreachable_count": sim_metrics.get("newly_unreachable_count"),
                  "active_edges_disabled": sim_metrics.get("active_edges_disabled")},
        )

        frozen = await call("freeze_account", {
            "case_id": case_id, "account_id": target,
            "reason": f"policy escalation: {canonical}",
            "requested_by": "phase5-workflow",
        })
        envelopes["freeze_account"] = frozen.envelope or {}
        payload = (frozen.envelope or {}).get("result") or {}
        result.frozen = bool(
            payload.get("frozen") and payload.get("action_status") == "applied"
        )
        result.freeze_declined = payload.get("action_status") == "declined"
        result.tool_sequence.append("freeze_account")
        policy = payload.get("policy") or {}
        checks = policy.get("checks") or {}
        failing = sorted(name for name, ok in checks.items() if not ok)
        # The deterministic gate's own outcome (real checks from the envelope).
        emitter.emit(
            type="policy", stage="POLICY",
            title=f"Policy v{policy.get('policy_version')} — "
                  f"{'permitted' if policy.get('allowed') else 'not permitted'}",
            detail=("; ".join(policy.get("blocking_reasons") or [])
                    or "every deterministic policy check passed"),
            ok=bool(policy.get("allowed")),
            meta={"policy_version": policy.get("policy_version"),
                  "allowed": policy.get("allowed"),
                  "checks": dict(checks),
                  "failing_checks": failing,
                  "blocking_reasons": list(policy.get("blocking_reasons") or []),
                  "caveats": list(policy.get("caveats") or []),
                  "signal_families": list(policy.get("signal_families") or []),
                  "target_relevant_families": list(
                      policy.get("target_relevant_families") or []),
                  "distinct_provenance_families": list(
                      policy.get("distinct_provenance_families") or []),
                  "qualifying_evidence_ids": list(
                      policy.get("qualifying_evidence_ids") or []),
                  "target": target},
        )
        # What actually happened to the graph (sandbox mutation or nothing).
        action_status = payload.get("action_status")
        emitter.emit(
            type="action", stage="ACT",
            title=f"Action outcome — {action_status}",
            detail=(
                f"sandbox freeze applied to {target}"
                if action_status == "applied"
                else "no mutation applied — the policy declined this freeze"
                if action_status == "declined"
                else f"action_status={action_status}; no new mutation performed"
            ),
            ok=bool(payload.get("allowed")),
            meta={"action_status": action_status,
                  "frozen": bool(payload.get("frozen")),
                  "mutation_performed": bool(payload.get("mutation_performed")),
                  "action_id": payload.get("action_id"),
                  "sandbox": True,
                  "target": target},
        )

        verified = await call("verify_freeze",
                              {"case_id": case_id, "account_id": target})
        envelopes["verify_freeze"] = verified.envelope or {}
        result.tool_sequence.append("verify_freeze")
        verify_result = (verified.envelope or {}).get("result") or {}
        emitter.emit(
            type="verification", stage="VERIFY",
            title="Independent verification completed",
            detail=(f"account={verify_result.get('account_status')}; "
                    f"verified={verify_result.get('verified')}"),
            ok=bool(verify_result.get("verified")),
            meta={"target": target,
                  "checks": dict(verify_result.get("checks") or {})},
        )
        return result


def run_case_workflow_sync(alert_id: str, **kwargs) -> CaseWorkflowResult:
    return asyncio.run(run_case_workflow(alert_id, **kwargs))