"""LLM integration for the investigation agent (PLAN.md §9).

Three implementations, one interface:

- ``FakeLLM`` — deterministic and offline. It does not replay a fixed script
  of seeded answers; it inspects the evidence the agent has *already* gathered
  and picks the next tool that would best discriminate between the competing
  hypotheses. That is what makes the recorded tool sequence meaningful while
  still being reproducible in tests.
- ``ScriptedLLM`` — replays an explicit list of decisions. Used by tests that
  need to force a specific path (budget exhaustion, provider failure, an
  intentionally premature conclusion).
- ``RealLLM`` — the configured provider: OpenAI, or Groq's OpenAI-compatible
  endpoint through the same langchain-openai client. It receives the tool
  schemas derived from the live MCP server, so it selects tools through the
  same interface the agent exposes, never by importing graph functions.

The LLM only ever chooses *which tool to call and with what arguments*. Every
interpretation, threshold, signal family and verdict is computed in Python
(``src/agent/analysis.py``). LLM self-reported confidence is never treated as
a calibrated probability and never authorizes a conclusion on its own.
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from src.agent.state import CaseState
from src.config import Settings, get_settings

DEFAULT_INVESTIGATION_WINDOW_DAYS = 90


class LLMConfigurationError(RuntimeError):
    """Raised when real-LLM mode is selected without usable credentials."""


class LLMProviderError(RuntimeError):
    """Raised when the provider fails (network, quota, malformed response)."""


class ToolDecision(BaseModel):
    """Ask the workflow to call one MCP tool."""

    kind: Literal["call_tool"] = "call_tool"
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = ""


class FinalDecision(BaseModel):
    """Stop investigating: either nothing further would discriminate, or done."""

    kind: Literal["finalize"] = "finalize"
    reason: str = ""


Decision = ToolDecision | FinalDecision
ToolSpec = dict[str, Any]


class InvestigationLLM(Protocol):
    """Interface the workflow depends on."""

    mode: str

    async def decide(
        self,
        *,
        state: CaseState,
        available_tools: list[ToolSpec],
        stage: str,
    ) -> Decision:
        ...


def _step_used(state: CaseState, tool: str, **argument_equals: Any) -> bool:
    """True if a step already called ``tool`` with these argument values."""
    for step in state.investigation_steps:
        if step.tool != tool:
            continue
        if all(step.arguments.get(k) == v for k, v in argument_equals.items()):
            return True
    return False


def _tool_used(state: CaseState, tool: str) -> bool:
    return any(step.tool == tool for step in state.investigation_steps)


class FakeLLM:
    """Deterministic, offline, evidence-driven tool selector."""

    mode = "fake"

    def __init__(self, window_days: int = DEFAULT_INVESTIGATION_WINDOW_DAYS) -> None:
        self.window_days = window_days

    async def decide(
        self,
        *,
        state: CaseState,
        available_tools: list[ToolSpec],
        stage: str,
    ) -> Decision:
        names = {t["name"] for t in available_tools}
        target = state.primary_entity_ids[0] if state.primary_entity_ids else None
        if target is None:
            return FinalDecision(reason="alert did not resolve to any entity to profile")
        if "entity_profile" not in names or "detect_cycle" not in names:
            return FinalDecision(reason="required MCP tools are unavailable")

        window = self.window_days

        # 1. Baseline: what is this account, and is it already under review?
        if not _tool_used(state, "entity_profile"):
            return ToolDecision(
                tool="entity_profile",
                arguments={"entity_id": target, "window_days": window},
                purpose="establish the target account's status, flows and prior cases",
            )

        # 1b. Case memory: if the profile reports an earlier case, pull it as a
        #     LEAD and re-check its claims against current evidence — prior
        #     verdicts are never inherited truth. Only the TARGET's profile can
        #     trigger this (matched on the recorded call parameters; the first
        #     query of a profile call is the window lookup, so all are scanned).
        if "prior_cases" in names and not _tool_used(state, "prior_cases"):
            for record in reversed(state.evidence):
                if record.tool != "entity_profile" or not record.ok:
                    continue
                if not any(
                    (params or {}).get("entity_id") == target
                    for params in record.params
                ):
                    continue
                if int(record.derived.get("prior_case_count") or 0) > 0:
                    return ToolDecision(
                        tool="prior_cases",
                        arguments={"entity_id": target},
                        purpose="pull prior cases as leads to re-check, not settled truth",
                    )
                break

        # 2. Structure: is it on a directed ring, and is that ring corroborated?
        if not _tool_used(state, "detect_cycle"):
            return ToolDecision(
                tool="detect_cycle",
                arguments={"account_id": target, "max_len": 4, "window_days": window},
                purpose="separate a structural ring from chronological circulation",
            )

        # 3. Concentration: is incoming volume forwarded onward to one place?
        #    This is the discriminator between a benign high-volume receiver
        #    and a coordinated cash-out intermediary.
        if not _tool_used(state, "fan_in_out"):
            return ToolDecision(
                tool="fan_in_out",
                arguments={"account_id": target, "window_days": window},
                purpose="test forwarding concentration against the benign alternative",
            )

        # 4. Routing and timing beyond the direct counterparties. Both directions are
        #    needed: an outbound trace alone never contains the inbound leg, and
        #    rapid in/out movement is only observable when both are combined.
        if not _step_used(state, "trace_funds", direction="out"):
            return ToolDecision(
                tool="trace_funds",
                arguments={
                    "account_id": target, "direction": "out", "hops": 2,
                    "window_days": window,
                },
                purpose="trace onward routing away from the target",
            )
        if not _step_used(state, "trace_funds", direction="in"):
            return ToolDecision(
                tool="trace_funds",
                arguments={
                    "account_id": target, "direction": "in", "hops": 2,
                    "window_days": window,
                },
                purpose="trace inbound routing so in/out timing can be assessed",
            )

        fan = next(
            (r.derived for r in state.evidence if r.tool == "fan_in_out" and r.ok),
            None,
        )
        from src.agent.analysis import fan_in_is_concentrated

        # 5/6. Only when forwarding is concentrated does identity infrastructure
        #      become the discriminating question.
        if fan and fan_in_is_concentrated(fan):
            receiver = fan.get("top_receiver")
            if receiver and not _step_used(state, "entity_profile", entity_id=receiver):
                return ToolDecision(
                    tool="entity_profile",
                    arguments={"entity_id": receiver, "window_days": window},
                    purpose="identify who controls the single cash-out destination",
                )
            if "shared_identifiers" in names and not _tool_used(state, "shared_identifiers"):
                owner_a = self._owner_of(state, target)
                owner_b = self._owner_of(state, receiver) if receiver else None
                if owner_a and owner_b and owner_a != owner_b:
                    return ToolDecision(
                        tool="shared_identifiers",
                        arguments={"customer_a_id": owner_a, "customer_b_id": owner_b},
                        purpose="test whether a shared identity link explains the flow",
                    )

        # 7. Context: a high weighted in-degree is a lead, not a finding — check
        #    how common it is before weighting it.
        if "hub_scores" in names and not _tool_used(state, "hub_scores"):
            return ToolDecision(
                tool="hub_scores",
                arguments={"window_days": window, "limit": 10},
                purpose="calibrate how exceptional this account's inbound volume is",
            )

        return FinalDecision(
            reason="every tool that could discriminate the competing hypotheses "
                   "has been consulted"
        )

    @staticmethod
    def _owner_of(state: CaseState, account_id: str | None) -> str | None:
        if not account_id:
            return None
        for record in state.evidence:
            if record.tool != "entity_profile" or not record.ok:
                continue
            if record.derived.get("found") and account_id in record.entity_ids:
                owners = record.derived.get("owner_customers") or []
                if owners:
                    return owners[0]
        return None


class ScriptedLLM:
    """Replays an explicit decision list (tests that need a forced path)."""

    mode = "fake"

    def __init__(self, decisions: list[Decision], *, fill: str = "finalize") -> None:
        self._decisions = list(decisions)
        self._fill = fill
        self.calls = 0

    async def decide(
        self,
        *,
        state: CaseState,
        available_tools: list[ToolSpec],
        stage: str,
    ) -> Decision:
        self.calls += 1
        if self._decisions:
            return self._decisions.pop(0)
        return FinalDecision(reason="scripted decisions exhausted")


class RealLLM:
    """Provider-backed tool selection over the MCP-derived tool interface.

    Providers (``LLM_PROVIDER``):
    - ``openai`` — the original integration (``OPENAI_API_KEY`` / ``OPENAI_MODEL``);
    - ``groq`` — Groq's OpenAI-compatible endpoint
      (``GROQ_API_KEY`` / ``GROQ_MODEL``, base URL
      ``https://api.groq.com/openai/v1``), through the same langchain-openai
      client. Credentials are always provider-specific: a Groq run never
      consumes or renames ``OPENAI_API_KEY``.
    """

    mode = "real"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ) -> None:
        self._settings = settings or get_settings()
        provider = self._settings.llm_provider
        if provider == "groq":
            if not self._settings.groq_api_key.get_secret_value():
                raise LLMConfigurationError(
                    "real LLM mode with LLM_PROVIDER=groq requires GROQ_API_KEY; "
                    "set it in the environment or .env, or run with LLM_MODE=fake "
                    "for the offline deterministic mode"
                )
            if not self._settings.groq_model:
                raise LLMConfigurationError(
                    "real LLM mode with LLM_PROVIDER=groq requires GROQ_MODEL "
                    "(or LLM_MODEL) to name a tool-calling model"
                )
        else:  # "openai" (Settings validates the Literal; no other value exists)
            if not self._settings.openai_api_key.get_secret_value():
                raise LLMConfigurationError(
                    "real LLM mode requires OPENAI_API_KEY; set it in the "
                    "environment or .env, or run with LLM_MODE=fake for the offline "
                    "deterministic mode"
                )
            if not self._settings.openai_model:
                raise LLMConfigurationError("real LLM mode requires OPENAI_MODEL to be set")
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._model = None
        # The event loop the cached client was built under (None = built
        # outside any running loop). Verified in Phase 7: langchain-openai
        # reuses a process-wide async client whose httpx pool is bound to the
        # loop of first use, so a second asyncio.run() fails every provider
        # call with "Event loop is closed" even when a fresh ChatOpenAI is
        # constructed. The fix is twofold and both parts are required: (1)
        # rebuild the client when the running loop changes, and (2) always
        # hand it an explicit httpx.AsyncClient created inside the current
        # loop. The loop object is held (never touched after close) because
        # id() can be recycled for a new loop.
        self._model_loop: asyncio.AbstractEventLoop | None = None
        self._http_client = None  # explicit per-loop transport (kept alive)

    @property
    def provider(self) -> str:
        return self._settings.llm_provider

    @property
    def model_name(self) -> str:
        """Configured model id (never a credential)."""
        return (
            self._settings.groq_model
            if self._settings.llm_provider == "groq"
            else self._settings.openai_model
        )

    def _chat(self):
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:  # not inside a running loop
            loop = None
        if self._model is not None and self._model_loop is not loop:
            self._model = None  # stale pool from another loop; rebuild below
        if self._model is None:
            from langchain_openai import ChatOpenAI  # imported lazily

            # An explicit httpx client created in THIS loop is what keeps the
            # transport from leaking across asyncio.run() boundaries.
            transport: dict[str, Any] = {}
            if loop is not None:
                import httpx

                self._http_client = httpx.AsyncClient(timeout=self._timeout_s)
                transport["http_async_client"] = self._http_client
            if self._settings.llm_provider == "groq":
                # Groq documents an OpenAI-compatible chat API; the static
                # OpenAI client works against it with an explicit base_url and
                # the Groq credential.
                self._model = ChatOpenAI(
                    model=self._settings.groq_model,
                    api_key=self._settings.groq_api_key,
                    base_url=self._settings.groq_base_url,
                    temperature=self._temperature,
                    timeout=self._timeout_s,
                    **transport,
                )
            else:
                self._model = ChatOpenAI(
                    model=self._settings.openai_model,
                    api_key=self._settings.openai_api_key,
                    temperature=self._temperature,
                    timeout=self._timeout_s,
                    **transport,
                )
            self._model_loop = loop
        return self._model

    async def decide(
        self,
        *,
        state: CaseState,
        available_tools: list[ToolSpec],
        stage: str,
    ) -> Decision:
        from src.agent.prompts import build_messages, to_openai_tools

        messages = build_messages(state=state, stage=stage)
        try:
            bound = self._chat().bind_tools(to_openai_tools(available_tools))
            response = await bound.ainvoke(messages)
        except LLMConfigurationError:
            raise
        except Exception as exc:  # provider/network/quota -> structured failure
            raise LLMProviderError(
                f"provider call failed ({type(exc).__name__}: {str(exc)[:200]})"
            ) from exc

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return FinalDecision(reason="model returned no tool call")

        call = tool_calls[0]
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        if not isinstance(name, str) or name not in {t["name"] for t in available_tools}:
            raise LLMProviderError(f"model requested unknown tool {name!r}")
        if not isinstance(args, dict):
            return FinalDecision(reason="model returned malformed tool arguments")
        return ToolDecision(tool=name, arguments=args, purpose="model-selected tool call")


def make_llm(
    settings: Settings | None = None,
    *,
    mode: str | None = None,
    provider: str | None = None,
) -> InvestigationLLM:
    """Build the configured LLM. Fake is the default for offline/test runs.

    ``provider`` (``openai``/``groq``) only applies to real mode and overrides
    ``LLM_PROVIDER`` for that instance without touching credentials.
    """
    settings = settings or get_settings()
    mode = mode or settings.llm_mode
    if mode == "real":
        if provider is not None:
            if provider not in {"openai", "groq"}:
                raise LLMConfigurationError(
                    f"unknown LLM provider {provider!r}; expected 'openai' or 'groq'"
                )
            settings = settings.model_copy(update={"llm_provider": provider})
        return RealLLM(settings)
    if mode == "fake":
        return FakeLLM()
    raise LLMConfigurationError(f"unknown LLM_MODE {mode!r}; expected 'fake' or 'real'")