"""HABEAS domain tool layer (PLAN.md §3/§7).

Phase 2 scope: plain-Python read tools and graph algorithms that return the
canonical JSON-safe evidence envelope. No MCP transport (Phase 3), no LLM
(Phase 4). Every query is a constant template from ``query_catalog`` executed
read-only through ``src.db.run_query`` — this layer never writes and never
imports the seed package (ground-truth isolation).
"""
from src.tools.algorithms import detect_cycle, hub_scores
from src.tools.reads import (
    active_path_exists,
    alert_context,
    entity_profile,
    fan_in_out,
    prior_cases,
    shared_identifiers,
    trace_funds,
)

__all__ = [
    "alert_context",
    "entity_profile",
    "trace_funds",
    "detect_cycle",
    "shared_identifiers",
    "fan_in_out",
    "hub_scores",
    "prior_cases",
    "active_path_exists",
]
