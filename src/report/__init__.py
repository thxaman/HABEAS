"""Phase 6: auditable reports and independent evidence verification (PLAN.md §11).

``builder`` renders a deterministic Markdown report for a persisted case;
``verifier`` checks the case's evidence records without an LLM and without
trusting any report prose. Both are trusted, read-only library code: every
graph read goes through approved query-catalog templates.
"""
from src.report.builder import REPORT_VERSION, ReportArtifact, build_report, write_report
from src.report.verifier import (
    ALL_STATUSES,
    STATUS_CHANGED,
    STATUS_MISMATCH,
    STATUS_REPRODUCED,
    STATUS_UNVERIFIABLE,
    STATUS_VERIFIED_RECORDED,
    CaseVerification,
    CaseRecords,
    load_case_records,
    verify_case,
)

__all__ = [
    "ALL_STATUSES",
    "STATUS_CHANGED",
    "STATUS_MISMATCH",
    "STATUS_REPRODUCED",
    "STATUS_UNVERIFIABLE",
    "STATUS_VERIFIED_RECORDED",
    "CaseRecords",
    "CaseVerification",
    "REPORT_VERSION",
    "ReportArtifact",
    "build_report",
    "load_case_records",
    "verify_case",
    "write_report",
]
