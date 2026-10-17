"""HABEAS Phase 7 evaluation harness (PLAN.md §12).

Layout (the user-facing Phase 7 spec names these paths):

- ``eval/run.py``      — command-line evaluation runner
- ``eval/metrics.py``  — deterministic metric calculations and validation
- ``eval/harness.py``  — disposable evaluation graphs + per-alert records
- ``eval/memory.py``   — controlled memory-on/memory-off paired comparison
- ``eval/ground_truth.py`` — validated manifest loading (EVAL LAYER ONLY)
- ``eval/render.py``   — Markdown scorecard rendering
- ``eval/results.json`` / ``eval/results.md`` — artifacts of the last run

Ground-truth isolation contract:

- the manifest is read ONLY here and in the trusted seed code;
- the investigation agent receives an alert id and graph evidence, nothing
  else — the harness never forwards manifest fields across the boundary;
- ``src.agent`` and ``src.tools`` never import this package (regression
  tests enforce both directions).

Every evaluation graph is explicitly disposable and named with the
``HABEAS_demo_eval`` prefix, which satisfies the existing demo-safe reset
guards (``src.config.DEMO_GRAPH_PREFIX`` + ``src.seed.generate.plan_reset``)
and additionally restricts evaluation to its own graphs — the shared demo
graph and arbitrary graphs are never touched.
"""
from __future__ import annotations


class EvaluationError(RuntimeError):
    """Evaluation setup or harness failure (never an agent outcome)."""


class EvaluationConfigError(EvaluationError):
    """Manifest/configuration problem: missing, malformed, or inconsistent."""


__all__ = ["EvaluationError", "EvaluationConfigError"]
