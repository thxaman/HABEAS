"""Submission hygiene (Phase 8 §6): the public repository carries no secrets,
no local machine paths, and no unsupported performance claims.

This test scans the *public* text files (source, docs, README, config
examples) — never ``.env``, generated logs, reports, caches, or this test file
itself, which legitimately names the patterns it looks for. It is a guard on
what a clean clone would publish, not a lint of local scratch files.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Directories that are generated, gitignored, or otherwise not published.
_SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", "logs", "reports",
    "New folder", ".venv", "venv", ".mypy_cache",
}
# Files that exist only locally.
_SKIP_FILES = {".env", "logs_eval_gate_stderr.txt", "logs_eval_gate_stdout.txt"}

_TEXT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".json"}


def _public_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _SKIP_FILES or path == SELF:
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(REPO).parts):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        # .env.example is published; any other dotfile is not scanned.
        if path.name.startswith(".") and path.name != ".env.example":
            continue
        files.append(path)
    return files


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_gitignore_protects_secrets_artifacts_and_ground_truth():
    rules = _read(REPO / ".gitignore")
    for required in (
        ".env",              # secrets
        "logs/",             # traces
        "reports/",          # generated reports
        "ground_truth.json", # evaluation manifest
    ):
        assert required in rules, f".gitignore must protect {required!r}"
    # the throwaway gate logs at the repo root are generated, never published
    assert "logs_*.txt" in rules


def test_env_example_ships_no_secret_values():
    example = _read(REPO / ".env.example")
    for line in example.splitlines():
        line = line.strip()
        if line.startswith(("OPENAI_API_KEY", "GROQ_API_KEY")):
            key, _, value = line.partition("=")
            assert value.strip() == "", f"{key} must be empty in .env.example"


# Secret-shaped tokens. The patterns require long random-looking runs, so
# documented placeholders like "sk-test-credential-not-real" are not flagged.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"gsk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

# Local machine paths that must never appear in published text.
_ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\+Users\\+"),   # C:\Users\...
    re.compile(r"[A-Za-z]:\\+HABEAS"),      # drive-letter project path
    re.compile(r"/Users/[A-Za-z]"),         # macOS home
    re.compile(r"/home/[a-z]"),             # Linux home
]


@pytest.mark.parametrize("pattern", _SECRET_PATTERNS + _ABSOLUTE_PATH_PATTERNS)
def test_public_files_contain_no_secrets_or_local_paths(pattern):
    offenders = []
    for path in _public_files():
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not offenders, f"{pattern.pattern} found in: {offenders}"


def test_readme_makes_the_honest_disclosures():
    readme = _read(REPO / "README.md")
    # required submission content
    for required in ("Agents That Act on Connected Data", "Synthetic data only",
                     "AI coding-assistant disclosure"):
        assert required in readme, f"README missing {required!r}"
    # honest metrics with denominators
    assert "4/10 (40.0%)" in readme          # strict recall
    assert "0/11 (0.0%)" in readme           # false-positive rate
    assert "129/158 (81.6%)" in readme       # verifier claims passing
    # the memory disclosure must be explicit
    assert "does not claim that graph memory" in readme
    assert "unverifiable" in readme
    # the multi-hop ablation must be stated
    assert "one-hop" in readme.lower()


def test_readme_contains_no_fabricated_overclaims():
    readme = _read(REPO / "README.md").lower()
    for forbidden in (
        "detects every fraud pattern",
        "detects all fraud",
        "100% recall",
        "all claims are verified",
        "saves 30%",
        "guaranteed detection",
    ):
        assert forbidden not in readme, f"README contains an overclaim: {forbidden!r}"


def test_docs_do_not_claim_a_video_was_recorded():
    """No artifact may claim the demo video is recorded unless it actually is."""
    script = _read(REPO / "docs" / "demo-script.md").lower()
    assert "no video has been recorded" in script
    readme = _read(REPO / "README.md").lower()
    assert "no demo video has been recorded" in readme