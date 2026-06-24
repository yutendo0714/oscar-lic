#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oscarlic.manifests import git_commit, sha256_file, utc_now_iso  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a provenance manifest before a run")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--environment", type=Path, default=ROOT / "environment/resolved_environment.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = args.config if args.config.is_absolute() else ROOT / args.config
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")
    output = args.output or ROOT / "experiments/manifests" / f"{args.experiment_id}.json"
    manifest = {
        "schema_version": "1.0",
        "experiment_id": args.experiment_id,
        "hypothesis_id": args.hypothesis_id,
        "status": "planned",
        "command": args.command,
        "command_tokens": shlex.split(args.command),
        "config_path": str(config.relative_to(ROOT)),
        "config_sha256": sha256_file(config),
        "code_commit": git_commit(ROOT),
        "environment_sha256": sha256_file(args.environment) if args.environment.is_file() else None,
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest) if args.dataset_manifest and args.dataset_manifest.is_file() else None,
        "external_repository_pins": {},
        "seed": args.seed,
        "hardware": {},
        "started_at": utc_now_iso(),
        "finished_at": None,
        "metrics_path": None,
        "log_path": None,
        "checkpoint_path": None,
        "conclusion": None,
        "next_action": None,
        "validity": None,
    }
    lock = ROOT / "baselines/locks/resolved_repositories.json"
    if lock.is_file():
        entries = json.loads(lock.read_text(encoding="utf-8"))
        manifest["external_repository_pins"] = {k: v["resolved_commit"] for k, v in entries.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
