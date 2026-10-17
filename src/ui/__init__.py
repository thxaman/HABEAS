"""HABEAS Phase 8.5 UI package.

Two narrowly scoped modules, neither of which imports Streamlit, so both are
unit-testable without a UI runtime:

- ``backend``     — thin wrappers around the existing application interfaces
  (``run_case_workflow``, the Phase 5 action envelopes, the report builder and
  verifier, the guarded demo reseed). No agent reasoning, evidence extraction,
  eligibility, simulation, or verification is reimplemented here.
- ``view_models`` — pure presentation formatting (status chips, graph
  construction from recorded evidence, DOT rendering, evidence-inspector
  rows). Every value they format comes from a real result object.

``app.py`` (repository root) is the only Streamlit-aware module.
"""