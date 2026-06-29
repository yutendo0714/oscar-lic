#!/usr/bin/env python3
"""Export explicit assignment rows for a simple exception-center policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from analyze_exception_center_selector_audit import build_group_rows, parse_seed_path, sha256_file


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_rows(args: argparse.Namespace) -> dict[str, Any]:
    rows = build_group_rows(args)
    selected = [
        row
        for row in rows
        if row["partition"] == args.partition and float(row["extra_relative_error"]) <= args.threshold
    ]
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        real_seed, split_seed, source_index, candidate_index = [int(value) for value in row["group_key"]]
        exported = {
            "source_index": source_index,
            "candidate_index": candidate_index,
            "assignment_code_index": int(row["extra_best_code"]),
            "code_index": int(row["extra_best_code"]),
            "nearest_code": int(row["nearest_code"]),
            "base_best_code": int(row["base_best_code"]),
            "extra_best_code": int(row["extra_best_code"]),
            "extra_relative_error": float(row["extra_relative_error"]),
            "extra_distance_margin": float(row["extra_distance_margin"]),
            "extra_center_utility": float(row["extra_center_utility"]),
            "base_center_utility": float(row["base_center_utility"]),
            "oracle_change_proxy": bool(row["oracle_change"]),
            "source": row.get("source"),
            "reference": row.get("reference"),
            "seed": real_seed,
            "assignment_table_seed": split_seed,
            "real_seed": real_seed,
            "selected_by_policy": True,
            "selection_policy": f"exception_rel_le_{args.threshold:.3f}",
        }
        by_seed.setdefault(real_seed, []).append(exported)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for seed in sorted(by_seed):
        seed_rows = sorted(by_seed[seed], key=lambda item: (int(item["source_index"]), int(item["candidate_index"])))
        path = args.output_dir / f"eval300_seed{seed}_val75_exception_rel{int(round(args.threshold * 1000)):04d}_k80_explicit_2026_06_26.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in seed_rows)
        path.write_text(text, encoding="utf-8")
        outputs.append(
            {
                "seed": seed,
                "path": str(path),
                "rows": len(seed_rows),
                "oracle_change_proxy_rows": sum(1 for row in seed_rows if row["oracle_change_proxy"]),
                "noheadroom_proxy_rows": sum(1 for row in seed_rows if not row["oracle_change_proxy"]),
                "sha256": sha256_text(text),
            }
        )
    return {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic_export",
        "code_commit": git_commit(),
        "threshold": args.threshold,
        "partition": args.partition,
        "outputs": outputs,
        "summary": {
            "rows": sum(item["rows"] for item in outputs),
            "oracle_change_proxy_rows": sum(item["oracle_change_proxy_rows"] for item in outputs),
            "noheadroom_proxy_rows": sum(item["noheadroom_proxy_rows"] for item in outputs),
        },
    }


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, result: dict[str, Any]) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "features", "path": str(args.features), "sha256": sha256_file(args.features)},
        {"name": "table", "path": str(args.table), "sha256": sha256_file(args.table)},
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
    ]
    for seed, codebook in args.codebook:
        files.append({"name": f"codebook_seed{seed}", "path": str(codebook), "sha256": sha256_file(codebook)})
    for output in result["outputs"]:
        files.append({"name": f"policy_seed{output['seed']}", "path": output["path"], "sha256": output["sha256"]})
    data = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "command": " ".join(["scripts/export_exception_center_policy.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--codebook", type=parse_seed_path, action="append", required=True)
    parser.add_argument("--base-size", type=int, default=64)
    parser.add_argument("--partition", default="val")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_exception_center_rel100_policy_export_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_rows(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(args.manifest, args, args.result, result)
    print(json.dumps({"result": str(args.result), "manifest": str(args.manifest), "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
