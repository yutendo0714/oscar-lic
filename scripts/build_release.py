#!/usr/bin/env python3
"""Build a deterministic, integrity-manifested repository ZIP.

Transient caches and downloaded external assets are excluded. MANIFEST.sha256
intentionally excludes itself to avoid a self-referential digest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIXED_ZIP_TIME = (2026, 6, 24, 0, 0, 0)
EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
GENERATED = {"MANIFEST.sha256", "TREE.txt", "RELEASE_INFO.json"}


def is_included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if path.is_symlink():
        raise RuntimeError(f"symlinks are not permitted in release: {rel}")
    return path.is_file()


def files(include_manifest: bool = True) -> list[Path]:
    out = [p for p in ROOT.rglob("*") if is_included(p)]
    if not include_manifest:
        out = [p for p in out if p.name != "MANIFEST.sha256"]
    return sorted(out, key=lambda p: p.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paper_count() -> int:
    with (ROOT / "literature/paper_registry.csv").open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def experiment_count() -> int:
    with (ROOT / "experiments/EXPERIMENT_MATRIX.csv").open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def generate_metadata(test_count: int) -> None:
    # Remove old generated files before computing tree/counts.
    for name in GENERATED:
        (ROOT / name).unlink(missing_ok=True)

    tree_lines = [p.relative_to(ROOT).as_posix() for p in files()]
    (ROOT / "TREE.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")

    release = {
        "schema_version": 1,
        "name": "oscar_lic_research_repo",
        "version": "0.1.0-bootstrap",
        "snapshot_date": "2026-06-24",
        "purpose": "Self-bootstrapping LIC/OCR-aware compression research repository",
        "paper_registry_records": paper_count(),
        "experiment_matrix_records": experiment_count(),
        "unit_tests_last_passed": test_count,
        "toy_manifest_records": 5,
        "cuda_validated": False,
        "external_datasets_bundled": False,
        "external_checkpoints_bundled": False,
        "external_repositories_bundled": False,
        "manifest_excludes": ["MANIFEST.sha256"],
    }
    (ROOT / "RELEASE_INFO.json").write_text(
        json.dumps(release, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in files(include_manifest=False)]
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_zip(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    prefix = ROOT.name + "/"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files():
            rel = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(prefix + rel, date_time=FIXED_ZIP_TIME)
            mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def verify_zip(output: Path) -> None:
    with zipfile.ZipFile(output, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        names = set(archive.namelist())
        expected = {ROOT.name + "/" + p.relative_to(ROOT).as_posix() for p in files()}
        if names != expected:
            missing = expected - names
            extra = names - expected
            raise RuntimeError(f"ZIP member mismatch; missing={sorted(missing)}, extra={sorted(extra)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parent / f"{ROOT.name}.zip")
    parser.add_argument("--test-count", type=int, default=26)
    parser.add_argument("--clean-caches", action="store_true")
    args = parser.parse_args()

    if args.clean_caches:
        for path in ROOT.rglob("*"):
            if path.is_dir() and path.name in {".pytest_cache", "__pycache__"}:
                shutil.rmtree(path)

    generate_metadata(args.test_count)
    build_zip(args.output)
    verify_zip(args.output)
    archive_hash = sha256(args.output)
    hash_path = args.output.with_suffix(args.output.suffix + ".sha256")
    hash_path.write_text(f"{archive_hash}  {args.output.name}\n", encoding="utf-8")
    print(f"Built {args.output} ({args.output.stat().st_size} bytes)")
    print(f"SHA256 {archive_hash}")
    print(f"Files {len(files())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
