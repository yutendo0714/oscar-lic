#!/usr/bin/env python3
"""Clone explicitly selected external repositories at registered pins.

This script does not execute third-party code. It refuses moving branches unless
``--allow-unpinned`` is supplied, and writes a resolved lock after checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "literature/repositories.yaml"


def command(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed.returncode}")


def clone_url(url: str) -> str:
    if url.endswith(".git"):
        return url
    return url + ".git"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", nargs="+", required=True, help="repository IDs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unpinned", action="store_true")
    parser.add_argument("--destination", type=Path, default=ROOT / "external")
    args = parser.parse_args()

    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    repositories = {item["id"]: item for item in data["repositories"]}
    unknown = sorted(set(args.selected) - set(repositories))
    if unknown:
        raise SystemExit(f"unknown repository IDs: {', '.join(unknown)}")

    args.destination.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, dict] = {}
    for repo_id in args.selected:
        item = repositories[repo_id]
        pin = item.get("pinned_commit")
        if not pin and not args.allow_unpinned:
            raise SystemExit(f"{repo_id} is unpinned; resolve and record a commit or use --allow-unpinned for exploration only")
        target = args.destination / repo_id
        if target.exists() and any(target.iterdir()):
            print(f"SKIP {repo_id}: destination exists and is nonempty: {target}")
        else:
            command(["git", "clone", "--filter=blob:none", clone_url(item["url"]), str(target)], dry_run=args.dry_run)
        if pin:
            command(["git", "fetch", "origin", pin], cwd=target, dry_run=args.dry_run)
            command(["git", "checkout", "--detach", pin], cwd=target, dry_run=args.dry_run)
        elif not args.dry_run:
            command(["git", "checkout", item.get("default_branch", "main")], cwd=target)

        if args.dry_run:
            resolved_commit = pin or "UNRESOLVED_MOVING_BRANCH"
        else:
            completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=target, check=True, text=True, stdout=subprocess.PIPE)
            resolved_commit = completed.stdout.strip()
        resolved[repo_id] = {
            "url": item["url"],
            "registered_pin": pin,
            "resolved_commit": resolved_commit,
            "path": str(target.relative_to(ROOT)),
            "license_verified": item.get("license_verified", False),
        }

    if not args.dry_run:
        lock = ROOT / "baselines/locks/resolved_repositories.json"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
