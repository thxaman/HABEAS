"""Structured progress events for a running investigation (Phase 8.5/8.6).

The UI must show the agent's *actual* workflow, not an animation. Rather than
re-parsing the JSONL trace after the fact, the runner emits typed events as
each real step happens. A callback failure is logged and swallowed so it can
never abort an investigation.

Event shape (JSON-safe, ordered by ``seq``):

    {"seq": 4, "at": 1712345.678, "type": "tool", "phase": "completed",
     "stage": "UPDATE_EVIDENCE", "tool": "detect_cycle",
     "title": "...", "detail": "...", "ok": True, "meta": {...}}

``type`` is one of :data:`EVENT_TYPES`:

- ``stage``         — a workflow stage began (triage, report, …);
- ``tool``          — an MCP tool call, with ``phase`` ``started``/``completed``;
- ``evidence``      — an evidence record was appended (its id, queries, stance,
  truncated flags, and the nodes/edges the tool actually returned);
- ``hypotheses``    — competing explanations re-assessed from the evidence;
- ``policy``        — the deterministic Policy v2 gate completed (allowed,
  per-check outcomes, blocking reasons);
- ``action``        — a case/evidence/simulation write or the freeze outcome;
- ``verification``  — post-action graph verification completed;
- ``decision``      — the deterministic verdict;
- ``error``         — a failure, budget exhaustion or provider problem.

Events carry no secrets, no chain-of-thought and no ground truth — only the
auditable choice, its arguments, and a summary of what it returned. Nothing
here is a placeholder: every value comes from a real tool result or from
deterministic Python over the recorded evidence.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger("habeas.events")

EVENT_TYPES = (
    "stage",
    "tool",
    "evidence",
    "hypotheses",
    "policy",
    "action",
    "verification",
    "decision",
    "error",
)

# Only tool events carry a phase.
TOOL_PHASES = ("started", "completed")


class ProgressEmitter:
    """Collects/delivers ordered progress events. Safe to use with no callback."""

    def __init__(self, callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._callback = callback
        self._seq = 0
        self.events: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self._callback is not None

    def emit(
        self,
        *,
        type: str,
        title: str,
        detail: str = "",
        stage: str | None = None,
        tool: str | None = None,
        ok: bool | None = None,
        phase: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if type not in EVENT_TYPES:
            raise ValueError(f"unknown progress event type {type!r}")
        if phase is not None and phase not in TOOL_PHASES:
            raise ValueError(f"unknown tool phase {phase!r}")
        self._seq += 1
        event = {
            "seq": self._seq,
            "at": time.time(),
            "type": type,
            "phase": phase,
            "stage": stage,
            "tool": tool,
            "title": title,
            "detail": detail,
            "ok": ok,
            "meta": dict(meta or {}),
        }
        self.events.append(event)
        if self._callback is not None:
            try:
                self._callback(event)
            except Exception as exc:  # a broken consumer must not break the run
                logger.error("progress event callback failed: %s", exc)
        return event


__all__ = ["ProgressEmitter", "EVENT_TYPES", "TOOL_PHASES"]