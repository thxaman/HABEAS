"""HABEAS Phase 8 demo package (PLAN.md §13).

``python -m src.demo.run_demo`` runs three scripted scenarios over the real
MCP stdio server against a disposable, explicitly validated demo graph.

Naming note (recorded per the no-silent-changes rule): the Phase 8 brief names
``demo/run_demo.py``; the repository layout in PLAN.md §9 and the command in
``AGENTS.md`` both use ``src/demo/run_demo.py`` (``python -m src.demo.run_demo``).
This package follows the repository's documented layout so the documented
command works; the reconciliation is recorded in ``AGENTS.md`` and
``docs/phase-8-demo.md``.
"""
from __future__ import annotations

__all__: list[str] = []