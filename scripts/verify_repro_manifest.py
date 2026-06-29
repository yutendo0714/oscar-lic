#!/usr/bin/env python3
"""Verify local files referenced by an OSCAR-LIC reproducibility manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_checks(node: Any, checks: dict[str, str]) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            checks[node["path"]] = node["sha256"]
        if isinstance(node.get("npz"), str) and isinstance(node.get("npz_sha256"), str):
            checks[node["npz"]] = node["npz_sha256"]
        if isinstance(node.get("npz"), str) and isinstance(node.get("meta_sha256"), str):
            checks[f"{node['npz']}.meta.json"] = node["meta_sha256"]
        for value in node.values():
            collect_checks(value, checks)
    elif isinstance(node, list):
        for value in node:
            collect_checks(value, checks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to a reproducibility manifest YAML file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root used to resolve relative paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, str] = {}
    collect_checks(data, checks)

    errors: list[str] = []
    for rel, expected in sorted(checks.items()):
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"missing: {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"sha256 mismatch: {rel}: expected {expected}, got {actual}")

    print(f"Manifest: {manifest_path.relative_to(root)}")
    print(f"Checked files: {len(checks)}")
    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("OK: all referenced hashes match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
