"""JSONL investigation trace (PLAN.md §9).

One structured record per MCP tool call, plus stage transitions. Secrets are
never written: arguments are filtered through an allow-list of the sanitized
fields the tool schemas actually expose, and no environment values are logged.

Trace failures must be VISIBLE. If a record cannot be written the writer
raises, the loop records the failure in the investigation state, and the
problem is logged to stderr — an audit trail is never silently fabricated or
silently dropped.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.state import CaseState

logger = logging.getLogger("habeas.trace")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_DIR = REPO_ROOT / "logs"

# Only these argument keys are ever serialized (tool schemas use nothing else).
_SAFE_ARGUMENT_KEYS = frozenset({
    "alert_id", "entity_id", "account_id", "customer_a_id", "customer_b_id",
    "source_account_id", "target_account_id", "direction", "hops", "max_hops",
    "max_len", "min_amount", "min_senders", "window_days", "limit",
})


class TraceError(RuntimeError):
    """Raised when the audit trace could not be written."""


def sanitize_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only known, non-secret tool arguments."""
    safe: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if key not in _SAFE_ARGUMENT_KEYS:
            continue
        safe[key] = value
    return safe


class InvestigationTrace:
    """Append-only JSONL writer for one investigation."""

    def __init__(self, case_id: str, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_TRACE_DIR
        self.path = self.directory / f"investigation-{case_id}.jsonl"
        self._records = 0

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "event": event,
            **fields,
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except OSError as exc:
            logger.error("failed to write trace record to %s: %s", self.path, exc)
            raise TraceError(
                f"could not write investigation trace to {self.path}: {exc}"
            ) from exc
        self._records += 1
        return record

    def tool_call(
        self,
        *,
        state: CaseState,
        stage: str,
        tool: str,
        arguments: dict[str, Any],
        ok: bool,
        query_ids: list[str],
        rows_hash: list[str],
        row_count: list[int],
        truncated: list[bool],
        duration_ms: float,
        transition: str,
        error: str | None = None,
    ) -> None:
        self.write(
            "tool_call",
            case_id=state.case_id,
            alert_id=state.alert_id,
            stage=stage,
            tool=tool,
            arguments=sanitize_arguments(arguments),
            ok=ok,
            query_ids=query_ids,
            evidence_hashes=rows_hash,
            row_counts=row_count,
            truncated=truncated,
            duration_ms=round(duration_ms, 3),
            transition=transition,
            error=error,
            tool_calls_used=state.tool_calls_used,
            tool_budget=state.tool_budget,
        )

    def stage(self, *, state: CaseState, stage: str, note: str = "") -> None:
        self.write(
            "stage",
            case_id=state.case_id,
            alert_id=state.alert_id,
            stage=stage,
            note=note,
            tool_calls_used=state.tool_calls_used,
            tool_budget=state.tool_budget,
        )

    def read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]