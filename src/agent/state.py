"""Phase 4 investigation state (PLAN.md §9).

`CaseState` is the single serializable artifact of one investigation: the
alert, its resolved entities, competing hypotheses with the evidence that
supports or contradicts each, every MCP tool call's normalized evidence, the
step trace, budget usage, and a qualified provisional verdict.

It deliberately holds NO secrets and NO ground-truth manifest data. Every
claim it records is backed by evidence ids produced by real MCP tool calls.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---- hypothesis names (stable code constants, not seeded ids) --------------

H_CIRCULATION = "chronologically_corroborated_circulation"
H_FAN_IN_OUT = "coordinated_fan_in_fan_out_cash_out"
H_RAPID_MOVEMENT = "rapid_movement_of_incoming_funds"
H_SHARED_IDENTITY = "shared_identifier_link_warranting_review"
H_BENIGN = "benign_high_volume_or_shared_infrastructure"
H_INSUFFICIENT = "insufficient_evidence_to_distinguish"

ALL_HYPOTHESES = (
    H_CIRCULATION,
    H_FAN_IN_OUT,
    H_RAPID_MOVEMENT,
    H_SHARED_IDENTITY,
    H_BENIGN,
    H_INSUFFICIENT,
)

# ---- independent signal families (PLAN.md §9 action-policy vocabulary) -----

FAMILY_CIRCULATION = "circulation"
FAMILY_FAN_IN_OUT = "fan_in_out"
FAMILY_RAPID_FORWARDING = "rapid_forwarding"

# ---- verdict labels (never a legal conclusion) -----------------------------

VERDICT_SUSPICIOUS = "suspicious_activity"
VERDICT_LIKELY_BENIGN = "likely_benign"
VERDICT_INSUFFICIENT = "insufficient_evidence"

Verdict = Literal["suspicious_activity", "likely_benign", "insufficient_evidence"]
HypothesisStatus = Literal[
    "open", "supported", "weakened", "contradicted", "insufficient"
]

TERMINATION_EVIDENCE_SUFFICIENT = "evidence_sufficient_for_verdict"
TERMINATION_NO_DISCRIMINATING_QUERY = "no_further_discriminating_query"
TERMINATION_BUDGET_EXHAUSTED = "tool_budget_exhausted"
TERMINATION_UNRECOVERABLE_ERROR = "unrecoverable_error"


class Hypothesis(BaseModel):
    """One competing explanation, with the evidence that bears on it."""

    name: str
    status: HypothesisStatus = "open"
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    rationale: str = ""


class EvidenceRecord(BaseModel):
    """Normalized result of exactly one MCP tool call."""

    evidence_id: str
    tool: str
    stage: str
    ok: bool
    interpretation: str = ""
    query_ids: list[str] = Field(default_factory=list)
    params: list[dict[str, Any]] = Field(default_factory=list)
    rows_hash: list[str] = Field(default_factory=list)
    row_count: list[int] = Field(default_factory=list)
    truncated: list[bool] = Field(default_factory=list)
    # Phase 6: interpolation bounds (token -> validated int) recorded per
    # sub-query so the verifier can rebuild the exact bounded query.
    bounds: list[dict[str, Any]] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    derived: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


class InvestigationStep(BaseModel):
    """One loop iteration: what was chosen, why, and what it changed."""

    index: int
    stage: str
    tool: str | None = None
    purpose: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    hypothesis_update: str = ""
    ok: bool = False
    duration_ms: float = 0.0


class ClaimStatus(BaseModel):
    """A factual claim and whether evidence supports, contradicts, or leaves it open."""

    claim: str
    status: Literal["supported", "contradicted", "unverified"]
    evidence_ids: list[str] = Field(default_factory=list)


class CaseState(BaseModel):
    """LangGraph state for one HABEAS investigation (fully serializable)."""

    # identity / target
    alert_id: str
    case_id: str = ""
    primary_entity_ids: list[str] = Field(default_factory=list)
    alert_context: dict[str, Any] = Field(default_factory=dict)
    # Case ids surfaced by prior_cases. They are LEADS to re-check, never
    # inherited truth, and are linked (not merged) when this case is recorded.
    prior_case_leads: list[str] = Field(default_factory=list)

    # reasoning
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    investigation_steps: list[InvestigationStep] = Field(default_factory=list)
    claims: list[ClaimStatus] = Field(default_factory=list)

    # control / budget
    tool_calls_used: int = 0
    tool_budget: int = 12
    step_index: int = 0
    stage: str = "START"
    termination_reason: str | None = None
    pending_tool: dict[str, Any] | None = None
    pending_result: dict[str, Any] | None = None

    # output
    provisional_verdict: str | None = None
    rationale: str = ""
    limitations: list[str] = Field(default_factory=list)
    signal_families: list[str] = Field(default_factory=list)
    llm_mode: str = "fake"
    errors: list[str] = Field(default_factory=list)
    trace_path: str | None = None

    # ---- helpers used by nodes and tests ---------------------------------

    def evidence_by_id(self, evidence_id: str) -> EvidenceRecord | None:
        return next((e for e in self.evidence if e.evidence_id == evidence_id), None)

    def hypothesis(self, name: str) -> Hypothesis | None:
        return next((h for h in self.hypotheses if h.name == name), None)

    def tools_called(self) -> list[str]:
        return [step.tool for step in self.investigation_steps if step.tool]

    def budget_exhausted(self) -> bool:
        return self.tool_calls_used >= self.tool_budget

    def derived(self, tool: str) -> list[dict[str, Any]]:
        """Deterministic derivations recorded for every successful call of ``tool``."""
        return [e.derived for e in self.evidence if e.tool == tool and e.ok]


class InvestigationResult(BaseModel):
    """Structured return value of one investigation run."""

    state: CaseState
    verdict: Verdict
    verdict_label: str
    tool_sequence: list[str]
    termination_reason: str
    trace_path: str | None = None