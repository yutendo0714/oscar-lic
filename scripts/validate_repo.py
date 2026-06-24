#!/usr/bin/env python3
"""Structural validation for the self-bootstrapping repository."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "external",
    "oscar_lic_package",
    ".venv",
    ".venv-mlic",
    "wandb",
}
IGNORED_PREFIXES = {
    ("data", "raw"),
    ("experiments", "checkpoints"),
    ("experiments", "logs"),
    ("experiments", "results"),
}
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.registry import (  # noqa: E402
    validate_baseline_registry,
    validate_data_registry,
    validate_experiment_matrix,
)

REQUIRED_FILES = [
    "README.md", "INDEX.md", "AGENTS.md", "PROMPT_TO_AGENT.md", "PROJECT_SPEC.md", "STATUS.md", "TASKS.md",
    "DECISIONS.md", "BLOCKERS.md", "NEGATIVE_RESULTS.md", "RISKS.md",
    "literature/paper_registry.csv", "literature/repositories.yaml",
    "literature/unverified_leads.md", "scripts/render_literature.py", "scripts/build_release.py",
    "baselines/registry.yaml", "baselines/checkpoint_manifest.yaml",
    "baselines/ocr_models.yaml", "data/registry.yaml",
    "docs/architecture.md", "docs/losses.md", "docs/utility_definition.md",
    "docs/BITSTREAM_SPEC.md", "docs/THREAT_MODEL.md",
    "evaluation/PROTOCOL.md", "evaluation/METRICS.md",
    "evaluation/OCR_NORMALIZATION.md", "evaluation/REPORTING_SCHEMA.json",
    "experiments/EXPERIMENT_MATRIX.csv", "experiments/hypotheses.yaml",
    "experiments/run_manifest.schema.json", "configs/model/oscar_mlicpp_v1.yaml",
    "scripts/verify_environment.py", "src/oscarlic/text_metrics.py",
    "tests/test_text_metrics.py",
]

REQUIRED_DIRS = [
    "configs", "data", "docs", "environment", "evaluation", "experiments",
    "literature", "baselines", "scripts", "src/oscarlic", "tests", "external",
]


def discover_structured_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.yaml", "*.yml", "*.json"):
        paths.extend(root.rglob(suffix))
    return sorted(p for p in paths if not should_ignore(root, p))


def should_ignore(root: Path, path: Path) -> bool:
    parts = path.relative_to(root).parts
    if IGNORED_PARTS & set(parts):
        return True
    return any(parts[: len(prefix)] == prefix for prefix in IGNORED_PREFIXES)


def parse_structured(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_DIRS:
        if not (ROOT / rel).is_dir():
            errors.append(f"missing directory: {rel}")
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required file: {rel}")

    for path in discover_structured_files(ROOT):
        try:
            parse_structured(path)
        except Exception as exc:
            errors.append(f"cannot parse {path.relative_to(ROOT)}: {exc}")

    # Reject hidden control characters in text/config/source files. They can corrupt
    # equations, commands, YAML or copy/paste without being visually obvious.
    binary_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".zip", ".pyc", ".pyo", ".so", ".jsonl"}
    max_text_scan_bytes = 5 * 1024 * 1024
    for path in ROOT.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() in binary_suffixes:
            continue
        if should_ignore(ROOT, path):
            continue
        if path.stat().st_size > max_text_scan_bytes:
            warnings.append(f"skipping large text-control scan: {path.relative_to(ROOT)}")
            continue
        raw = path.read_bytes()
        bad = [(i, b) for i, b in enumerate(raw) if b < 32 and b not in (9, 10, 13)]
        if bad:
            errors.append(f"control character in {path.relative_to(ROOT)} at byte {bad[0][0]} value={bad[0][1]}")

    errors.extend(f"data registry: {e}" for e in validate_data_registry(ROOT / "data/registry.yaml"))
    errors.extend(f"baseline registry: {e}" for e in validate_baseline_registry(ROOT / "baselines/registry.yaml"))
    errors.extend(f"experiment matrix: {e}" for e in validate_experiment_matrix(ROOT / "experiments/EXPERIMENT_MATRIX.csv"))

    # Paper registry IDs and mandatory source/status fields.
    paper_path = ROOT / "literature/paper_registry.csv"
    if paper_path.exists():
        with paper_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            seen: set[str] = set()
            for line, row in enumerate(reader, start=2):
                pid = row.get("id") or row.get("paper_id")
                if not pid:
                    errors.append(f"paper registry line {line}: missing id")
                elif pid in seen:
                    errors.append(f"paper registry line {line}: duplicate id {pid}")
                seen.add(pid or "")
                if not (row.get("url") or row.get("primary_url")):
                    warnings.append(f"paper registry {pid}: no URL")

    # All config paths referenced by the experiment matrix must exist unless blank.
    matrix = ROOT / "experiments/EXPERIMENT_MATRIX.csv"
    if matrix.exists():
        with matrix.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                config = row.get("config", "").strip()
                if config and not (ROOT / config).is_file():
                    errors.append(f"experiment {row.get('experiment_id')}: missing config {config}")

    # External unresolved resources are expected at bootstrap and are warnings.
    checkpoint_data = parse_structured(ROOT / "baselines/checkpoint_manifest.yaml")
    unresolved = [c.get("id") for c in checkpoint_data.get("checkpoints", []) if not c.get("sha256")]
    if unresolved:
        warnings.append(f"checkpoint hashes unresolved by design: {', '.join(map(str, unresolved))}")

    print(f"Repository: {ROOT}")
    print(f"Structured files parsed: {len(discover_structured_files(ROOT))}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"FAILED with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"OK: structural validation passed with {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
