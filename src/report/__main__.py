"""Report CLI (Phase 6 report stack, Phase 8 submission surface).

    python -m src.report build  --case CASE-ALR-0021 [--reports-dir reports/demo]
    python -m src.report verify --case CASE-ALR-0021

``build`` writes ``<reports-dir>/<case_id>.md`` and refuses to overwrite an
existing file unless ``--force`` is given, so a historical report can never be
replaced by accident (Phase 8 §2/§6). ``verify`` is strictly read-only: it
re-runs the independent verifier against the persisted case record and prints
the per-claim status counts. Neither command mutates the graph, and neither
uses an LLM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_settings
from src.db import get_graph
from src.report.builder import build_report, write_report
from src.report.verifier import verify_case


def cmd_build(args: argparse.Namespace) -> int:
    settings = load_settings()
    graph = get_graph(settings)
    out_dir = Path(args.reports_dir)
    target = out_dir / f"{args.case}.md"
    if target.exists() and not args.force:
        print(
            f"refusing to overwrite existing report {target}; pass --force to "
            "replace it deliberately",
            file=sys.stderr,
        )
        return 2
    artifact = build_report(graph, args.case, settings=settings)
    path = write_report(artifact, out_dir)
    verification = artifact.verification
    print(f"wrote {path} (report_version {artifact.report_version})")
    print(f"verifier claim status counts: {verification.summary}")
    if verification.reference_summary:
        print(f"material-claim reference checks: {verification.reference_summary}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    settings = load_settings()
    graph = get_graph(settings)
    verification = verify_case(graph, args.case)
    if not verification.exists:
        print(
            f"case {args.case} does not exist in graph {settings.graph_name}",
            file=sys.stderr,
        )
        return 2
    print(f"case {args.case}: claim status counts: {verification.summary}")
    if verification.reference_summary:
        print(f"material-claim reference checks: {verification.reference_summary}")
    for note in verification.notes:
        print(f"note: {note}")
    mismatches = verification.summary.get("mismatch", 0)
    print(
        "result:"
        + (" MISMATCH detected (see report)" if mismatches else
           " no mismatches; unverifiable records are reported, not hidden")
    )
    return 1 if mismatches else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.report",
        description="Build or independently verify a HABEAS case report.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build and write a case report")
    build.add_argument("--case", required=True)
    build.add_argument("--reports-dir", default="reports")
    build.add_argument("--force", action="store_true",
                       help="overwrite an existing report file")
    build.set_defaults(func=cmd_build)

    verify = sub.add_parser("verify", help="re-run the independent verifier")
    verify.add_argument("--case", required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())