"""Ground-truth manifest loading and validation — EVALUATION LAYER ONLY.

The manifest (`src/seed/ground_truth.json`, written by the trusted seed code)
is the scoring source for the evaluation. It never crosses into the agent:
the harness passes exactly one thing to the investigation — an alert id.

Validation performed here (all failures are ``EvaluationConfigError`` with a
clear, actionable message):

1. the file exists and parses as JSON;
2. it conforms to the ``GroundTruth`` schema;
3. ``seed`` matches the evaluation request;
4. it byte-matches (per section) the *deterministic generator output* for that
   seed, so a hand-edited or stale manifest cannot silently change scoring;
5. alert ids are unique, every suspicious alert's family has a planted pattern,
   and every planted pattern family has at least one suspicious alert — the
   denominator source is therefore the validated manifest itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from eval import EvaluationConfigError
from src.seed.generate import MANIFEST_PATH, build_model
from src.seed.patterns import FAMILY_BENIGN, AlertTruth, GroundTruth

# Sections compared against a fresh deterministic build. ``graph_name`` is
# excluded on purpose: where the manifest was produced (demo graph vs a
# disposable evaluation graph) does not change any planted fact.
_CROSS_CHECKED_FIELDS = ("counts", "alerts", "patterns", "tempting_false_positives")


def load_manifest(
    path: str | Path | None = None, *, expected_seed: int | None = None
) -> GroundTruth:
    """Load and fully validate a ground-truth manifest."""
    target = Path(path) if path is not None else MANIFEST_PATH
    if not target.exists():
        raise EvaluationConfigError(
            f"ground-truth manifest not found at {target}; generate it with "
            "`python -m src.seed.generate --seed <seed> --reset-demo-graph`"
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationConfigError(
            f"ground-truth manifest at {target} is unreadable or not valid JSON "
            f"({type(exc).__name__})"
        ) from exc
    if not isinstance(raw, dict):
        raise EvaluationConfigError(
            f"ground-truth manifest at {target} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    try:
        manifest = GroundTruth.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
        raise EvaluationConfigError(
            f"ground-truth manifest at {target} is malformed: "
            f"{exc.error_count()} schema error(s); first at {location}: "
            f"{first.get('msg', 'unknown error')}"
        ) from exc
    if expected_seed is not None and manifest.seed != expected_seed:
        raise EvaluationConfigError(
            f"ground-truth manifest at {target} was generated for seed "
            f"{manifest.seed}, but the evaluation requested seed {expected_seed}; "
            "regenerate the manifest for the requested seed"
        )
    _cross_check_against_generator(manifest, target)
    return manifest


def _cross_check_against_generator(manifest: GroundTruth, path: Path) -> None:
    expected = build_model(seed=manifest.seed, graph_name=manifest.graph_name).manifest
    differing = [
        field
        for field in _CROSS_CHECKED_FIELDS
        if getattr(manifest, field) != getattr(expected, field)
    ]
    if differing:
        raise EvaluationConfigError(
            f"ground-truth manifest at {path} does not match the deterministic "
            f"generator output for seed {manifest.seed} (differing sections: "
            f"{', '.join(differing)}); regenerate it instead of editing it"
        )


def alert_index(manifest: GroundTruth) -> dict[str, AlertTruth]:
    """Alert id -> truth, with duplicate detection."""
    index: dict[str, AlertTruth] = {}
    for alert in manifest.alerts:
        if alert.alert_id in index:
            raise EvaluationConfigError(
                f"ground-truth manifest contains duplicate alert id "
                f"{alert.alert_id!r}; refusing to score against it"
            )
        index[alert.alert_id] = alert
    return index


def family_denominators(manifest: GroundTruth) -> dict[str, Any]:
    """The manifest-derived denominators every pattern metric must use.

    Suspicious alerts are grouped by family; the *pattern* denominator is the
    set of distinct planted pattern families. A suspicious alert whose family
    has no planted pattern (or a planted family with no suspicious alerts)
    makes the manifest internally inconsistent and is rejected here.
    """
    suspicious = [a for a in manifest.alerts if a.label == "suspicious"]
    benign = [a for a in manifest.alerts if a.label == "benign"]
    by_family: dict[str, list[str]] = {}
    for alert in suspicious:
        by_family.setdefault(alert.family, []).append(alert.alert_id)
    pattern_families = sorted({p.family for p in manifest.patterns})

    if FAMILY_BENIGN in pattern_families:
        raise EvaluationConfigError(
            "ground-truth manifest lists the benign label as a planted pattern "
            "family; patterns must describe suspicious families only"
        )
    unknown = sorted(set(by_family) - set(pattern_families))
    if unknown:
        raise EvaluationConfigError(
            "ground-truth manifest has suspicious alerts in families with no "
            f"planted pattern: {', '.join(unknown)}"
        )
    uncovered = sorted(set(pattern_families) - set(by_family))
    if uncovered:
        raise EvaluationConfigError(
            "ground-truth manifest has planted pattern families with no "
            f"suspicious alerts: {', '.join(uncovered)}"
        )

    return {
        "pattern_families": pattern_families,
        "suspicious_by_family": {fam: sorted(ids) for fam, ids in sorted(by_family.items())},
        "benign_alerts": sorted(a.alert_id for a in benign),
        "suspicious_total": len(suspicious),
        "benign_total": len(benign),
        "alert_total": len(manifest.alerts),
        # Account members per planted pattern: used to flag benign alerts that
        # share accounts with a planted pattern (possible confusions).
        "pattern_account_members": {
            p.family: sorted(p.account_members) for p in manifest.patterns
        },
    }


def _portable_path(path: str | Path) -> str:
    """A repo-relative path (never a local absolute one) for published output.

    Evaluation artifacts are committed, so they must not embed the developer's
    machine paths (Phase 8 §6). A path outside the repo falls back to its name.
    """
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return str(Path(path).resolve().relative_to(repo_root))
    except ValueError:
        return Path(path).name


def manifest_summary(manifest: GroundTruth, path: str | Path | None = None) -> dict[str, Any]:
    """JSON-safe manifest digest stored in results.json for recomputation."""
    target = Path(path) if path is not None else MANIFEST_PATH
    summary: dict[str, Any] = {
        "path": _portable_path(target),
        "schema_version": manifest.schema_version,
        "seed": manifest.seed,
        "graph_name": manifest.graph_name,
    }
    summary.update(family_denominators(manifest))
    return summary
