"""HABEAS investigation MCP server (PLAN.md §3/§7).

Exposes ONLY the Phase 2 read tools over stdio. Deliberately absent:

- any arbitrary-Cypher tool,
- any graph write/mutation tool,
- any access to the ground-truth manifest.

Run:  python -m src.tools.server

Protocol rules honoured here:
- stdout carries ONLY MCP protocol traffic (never application prints);
- all diagnostics go to stderr via the logging module;
- every tool returns the full Phase 2 evidence envelope as a JSON object, so
  query ids, parameters, rows, hashes, truncation flags and deterministic
  interpretations all survive the MCP boundary.

Argument validation is single-sourced: each Phase 2 tool validates its inputs
with ``src.tools.query_catalog`` BEFORE touching the graph, and returns a
structured ``ok=false`` envelope with an empty ``query_evidence`` list on
rejection. Nothing is validated by the transport and then re-checked loosely.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.config import Settings, get_settings
from src.db import get_graph
from src.tools.algorithms import detect_cycle as _detect_cycle
from src.tools.algorithms import hub_scores as _hub_scores
from src.tools.reads import active_path_exists as _active_path_exists
from src.tools.reads import alert_context as _alert_context
from src.tools.reads import alert_inbox as _alert_inbox
from src.tools.reads import alert_neighborhood as _alert_neighborhood
from src.tools.reads import entity_profile as _entity_profile
from src.tools.reads import fan_in_out as _fan_in_out
from src.tools.reads import prior_cases as _prior_cases
from src.tools.reads import shared_identifiers as _shared_identifiers
from src.tools.reads import trace_funds as _trace_funds

# Diagnostics -> stderr only. stdout is reserved for the protocol.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s habeas-mcp %(message)s",
)
logger = logging.getLogger("habeas.mcp")

SERVER_NAME = "HABEAS-investigation-tools"

mcp = FastMCP(SERVER_NAME)


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def _graph():
    """Selected demo graph. Read tools only — never written by this server."""
    settings = _settings()
    graph = get_graph(settings)
    logger.info(
        "serving graph %r at %s:%s",
        settings.graph_name, settings.falkordb_host, settings.falkordb_port,
    )
    return graph


def _envelope(env) -> dict[str, Any]:
    """Serialize a Phase 2 ToolEnvelope without dropping any evidence."""
    return env.model_dump()


# ---- read tools (PLAN.md §7) ----------------------------------------------


@mcp.tool()
def alert_context(alert_id: str) -> dict:
    """Resolve an alert id into its trigger transaction and involved accounts.

    Added in Phase 4 so the investigation agent can derive its target from
    graph evidence: entity_profile resolves Account/Customer ids only.
    """
    return _envelope(_alert_context(_graph(), alert_id))


@mcp.tool()
def alert_inbox(limit: int = 200) -> dict:
    """List the graph's alerts with their real investigation/action state.

    Phase 8.5: the analyst UI's inbox. Read-only and bounded; each row carries
    the concerned account, trigger transaction, and (when a prior run already
    persisted one) the case verdict and latest action status. The UI renders
    exactly these rows — it never hard-codes a case list.
    """
    return _envelope(_alert_inbox(_graph(), limit=limit))


@mcp.tool()
def alert_neighborhood(account_id: str, hops: int = 2, limit: int = 200) -> dict:
    """Bounded neighbourhood subgraph of one account (for graph rendering).

    account_id is an Account id; hops must be 1..5. Returns the focal account,
    accounts reachable in either direction, and every incident TRANSFERRED_TO
    edge with its count/volume/last_ts and active flag. Structural
    connectivity only — never a chronological claim.
    """
    return _envelope(_alert_neighborhood(_graph(), account_id, hops=hops, limit=limit))


@mcp.tool()
def entity_profile(entity_id: str, window_days: int = 30, limit: int = 100) -> dict:
    """Profile an account or customer: bounded flows, status, prior-case count.

    entity_id must be a graph id such as "ACC-0100" or "CUS-0007".
    (Examples are arbitrary format illustrations, never accounts of interest.)
    """
    return _envelope(_entity_profile(_graph(), entity_id, window_days=window_days, limit=limit))


@mcp.tool()
def trace_funds(
    account_id: str,
    direction: str = "out",
    hops: int = 3,
    min_amount: float = 0,
    window_days: int = 30,
    limit: int = 100,
) -> dict:
    """Trace bounded active transfer topology from an account.

    direction is "in" or "out"; hops must be 1..5. Follows the derived
    TRANSFERRED_TO projection (structural connectivity, not chronological
    movement of one sum of money).
    """
    return _envelope(
        _trace_funds(
            _graph(), account_id, direction, hops=hops, min_amount=min_amount,
            window_days=window_days, limit=limit,
        )
    )


@mcp.tool()
def detect_cycle(
    account_id: str, max_len: int = 4, window_days: int = 90, limit: int = 100
) -> dict:
    """Detect bounded directed cycles through an account.

    max_len must be 2..5. Distinguishes structural cycles from temporally
    corroborated ones; a structural cycle alone is not evidence that funds
    circulated.
    """
    return _envelope(
        _detect_cycle(
            _graph(), account_id, max_len=max_len, window_days=window_days, limit=limit
        )
    )


@mcp.tool()
def shared_identifiers(
    customer_a_id: str, customer_b_id: str, limit: int = 100
) -> dict:
    """Return exact shared synthetic devices and identifiers for two customers.

    Shared infrastructure is an association signal only, not proof of fraud.
    """
    return _envelope(
        _shared_identifiers(
            _graph(), customer_a_id, customer_b_id, limit=limit
        )
    )


@mcp.tool()
def fan_in_out(
    account_id: str, window_days: int = 30, min_senders: int = 3, limit: int = 100
) -> dict:
    """Count distinct senders/receivers and amounts for one account's window."""
    return _envelope(
        _fan_in_out(
            _graph(), account_id, window_days=window_days,
            min_senders=min_senders, limit=limit,
        )
    )


@mcp.tool()
def hub_scores(window_days: int = 30, limit: int = 10) -> dict:
    """Rank accounts by weighted in-degree (NOT PageRank; a lead, not proof)."""
    return _envelope(_hub_scores(_graph(), window_days=window_days, limit=limit))


@mcp.tool()
def prior_cases(entity_id: str, limit: int = 100) -> dict:
    """Return bounded prior cases referencing an entity (leads to re-check)."""
    return _envelope(_prior_cases(_graph(), entity_id, limit=limit))


@mcp.tool()
def active_path_exists(
    source_account_id: str, target_account_id: str, max_hops: int = 5
) -> dict:
    """Check whether an active route remains between two accounts (<=5 hops)."""
    return _envelope(
        _active_path_exists(
            _graph(), source_account_id, target_account_id, max_hops=max_hops
        )
    )


# ---- write-capable action tools (Phase 5, OPT-IN) --------------------------
#
# Registered only when HABEAS_ENABLE_ACTIONS=true AND the configured graph is
# the disposable sandbox. Every one of them re-checks that configuration at
# call time. They are narrow, purpose-built operations: there is no generic
# write tool and no tool that accepts Cypher. The graph name always comes from
# configuration, never from a tool argument.

_ACTION_TOOLS_REGISTERED = False


def register_action_tools() -> bool:
    """Register the sandbox action tools if actions are enabled and sandboxed."""
    global _ACTION_TOOLS_REGISTERED
    if _ACTION_TOOLS_REGISTERED:
        return True
    settings = _settings()
    if not (settings.habeas_enable_actions and settings.is_sandbox):
        logger.info(
            "action tools NOT registered (actions_enabled=%s, sandbox=%s)",
            settings.habeas_enable_actions, settings.is_sandbox,
        )
        return False

    from src.tools.actions import (
        cite_evidence as _cite_evidence,
        close_case as _close_case,
        create_case as _create_case,
        flag_entity as _flag_entity,
        freeze_account as _freeze_account,
        reconstruct_case_state as _reconstruct_case_state,
        simulate_freeze as _simulate_freeze,
        verify_freeze as _verify_freeze,
    )

    @mcp.tool()
    def create_case(
        alert_id: str, severity: str = "medium",
        verdict: str = "insufficient_evidence", summary: str = "",
        entity_ids: list[str] | None = None,
        reuses_case_ids: list[str] | None = None,
    ) -> dict:
        """Create (or reuse) the persistent case for an alert. Sandbox only.

        ``reuses_case_ids`` links earlier cases as LEADS (REUSES edges); their
        verdicts are never inherited.
        """
        return _envelope(_create_case(
            _graph(), alert_id, severity=severity, verdict=verdict,
            summary=summary, entity_ids=entity_ids or [],
            reuses_case_ids=reuses_case_ids or [], settings=_settings(),
        ))

    @mcp.tool()
    def cite_evidence(case_id: str, citations: list[dict]) -> dict:
        """Persist references to evidence already returned during investigation."""
        return _envelope(_cite_evidence(
            _graph(), case_id, citations, settings=_settings()
        ))

    @mcp.tool()
    def flag_entity(
        case_id: str, entity_id: str, reason: str, risk: str = "medium"
    ) -> dict:
        """Flag an entity for human review (does not imply wrongdoing)."""
        return _envelope(_flag_entity(
            _graph(), case_id, entity_id, reason, risk=risk, settings=_settings()
        ))

    @mcp.tool()
    def close_case(
        case_id: str, verdict: str, summary: str, status: str = "closed"
    ) -> dict:
        """Record the final disposition of a case."""
        return _envelope(_close_case(
            _graph(), case_id, verdict=verdict, summary=summary,
            status=status, settings=_settings(),
        ))

    @mcp.tool()
    def simulate_freeze(
        case_id: str, account_id: str, window_days: int = 30, hops: int = 3
    ) -> dict:
        """Read-only counterfactual: what a freeze would affect. No mutation."""
        return _envelope(_simulate_freeze(
            _graph(), case_id, account_id, window_days=window_days, hops=hops,
            settings=_settings(),
        ))

    @mcp.tool()
    def freeze_account(
        case_id: str, account_id: str, reason: str,
        requested_by: str = "mcp-client", evidence_ids: list[str] | None = None,
    ) -> dict:
        """Guarded sandbox freeze.

        Eligibility is re-derived HERE from the evidence PERSISTED on the case,
        so a caller cannot authorize a freeze by asserting a verdict or by
        supplying an "allowed" flag. On any failed precondition nothing is
        written.
        """
        graph = _graph()
        case_state = _reconstruct_case_state(graph, case_id)
        return _envelope(_freeze_account(
            graph, case_id, account_id, reason,
            eligibility_inputs={"case_state": case_state},
            requested_by=requested_by, evidence_ids=evidence_ids or [],
            settings=_settings(),
        ))

    @mcp.tool()
    def verify_freeze(case_id: str, account_id: str) -> dict:
        """Re-query graph state after a freeze and report what changed."""
        return _envelope(_verify_freeze(
            _graph(), case_id, account_id, settings=_settings()
        ))

    _ACTION_TOOLS_REGISTERED = True
    logger.info("action tools registered on sandbox graph %r", settings.graph_name)
    return True


def main() -> int:
    """Run the server over stdio (stdout = protocol, stderr = diagnostics)."""
    settings = _settings()
    register_action_tools()
    logger.info(
        "starting %s over stdio (sandbox=%s, actions_enabled=%s)",
        SERVER_NAME, settings.is_sandbox, settings.habeas_enable_actions,
    )
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())