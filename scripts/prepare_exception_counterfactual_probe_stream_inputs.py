#!/usr/bin/env python3
"""Prepare per-seed split and selection tables for exception counterfactual probes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def remap_selection(row: dict[str, Any], local_index: int, *, exception: bool) -> dict[str, Any] | None:
    if exception and row["policy_case"] == "current_only":
        return None
    selection = dict(row["selection_row"] or {})
    selection["source_index"] = local_index
    selection["original_source_index"] = int(row["source_index"])
    selection["original_seed"] = int(row["seed"])
    selection["candidate_index"] = int(row["candidate_index"])
    selection["source_image"] = row["source_image"]
    selection["reference"] = row["reference"]
    selection["selected_by_policy"] = True
    selection["probe_index"] = int(row["probe_index"])
    selection["probe_tags"] = list(row["probe_tags"])
    selection["selection_policy"] = "probe_current" if not exception else "probe_exception_replace_or_drop"
    if exception:
        selection["assignment_code_index"] = int(row["extra_best_code"])
        selection["code_index"] = int(row["extra_best_code"])
        selection["nearest_code"] = int(row["nearest_code"])
        selection["extra_relative_error"] = float(row["extra_relative_error"])
    return selection


def split_row(row: dict[str, Any], local_index: int, method_id: str) -> dict[str, Any]:
    return {
        "image_path": row["source_image"],
        "text": row["reference"],
        "source": row["source"],
        "split": "train",
        "method_id": method_id,
        "probe_index": int(row["probe_index"]),
        "probe_source_index": int(row["source_index"]),
        "probe_candidate_index": int(row["candidate_index"]),
        "probe_seed": int(row["seed"]),
        "probe_tags": list(row["probe_tags"]),
        "local_index": local_index,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rows = [row for row in read_jsonl(args.probe_table) if bool(row["training_use"])]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    files = []
    seed_summaries = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed, seed_rows in sorted(grouped.items()):
        seed_rows = sorted(seed_rows, key=lambda item: int(item["probe_index"]))
        split_rows = []
        current_rows = []
        exception_rows = []
        for local_index, row in enumerate(seed_rows):
            split_rows.append(split_row(row, local_index, f"{args.method_id}_seed{seed}"))
            current = remap_selection(row, local_index, exception=False)
            if current is not None:
                current_rows.append(current)
            exception = remap_selection(row, local_index, exception=True)
            if exception is not None:
                exception_rows.append(exception)
        split_path = args.output_dir / f"text_crops_eval300_exception_probe_train_seed{seed}_v1.jsonl"
        current_path = args.output_dir / f"exception_probe_seed{seed}_current_selection.jsonl"
        exception_path = args.output_dir / f"exception_probe_seed{seed}_exception_selection.jsonl"
        write_jsonl(split_path, split_rows)
        write_jsonl(current_path, current_rows)
        write_jsonl(exception_path, exception_rows)
        files.extend(
            [
                {"name": f"seed{seed}_split", "path": str(split_path), "sha256": sha256_file(split_path)},
                {"name": f"seed{seed}_current_selection", "path": str(current_path), "sha256": sha256_file(current_path)},
                {"name": f"seed{seed}_exception_selection", "path": str(exception_path), "sha256": sha256_file(exception_path)},
            ]
        )
        seed_summaries[str(seed)] = {
            "split_rows": len(split_rows),
            "current_selection_rows": len(current_rows),
            "exception_selection_rows": len(exception_rows),
            "tags": dict(Counter(tag for row in seed_rows for tag in row["probe_tags"])),
            "sources": dict(Counter(str(row["source"]) for row in seed_rows)),
        }
    return {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-counterfactual-stream-inputs",
        "status": "completed_input_preparation",
        "code_commit": git_commit(),
        "probe_table": str(args.probe_table),
        "output_dir": str(args.output_dir),
        "seed_summaries": seed_summaries,
        "generated_files": files,
        "next_action": "Run current K64 nearest streams and K80 explicit exception/drop streams per seed, then OCR both recon sets.",
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception Counterfactual Probe Stream Inputs",
        "",
        "| seed | split rows | current selections | exception selections | tags |",
        "|---|---:|---:|---:|---|",
    ]
    for seed, data in result["seed_summaries"].items():
        lines.append(
            f"| {seed} | {data['split_rows']} | {data['current_selection_rows']} | "
            f"{data['exception_selection_rows']} | `{json.dumps(data['tags'], sort_keys=True)}` |"
        )
    lines.extend(["", "## Next Action", "", result["next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path, result: dict[str, Any]) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "probe_table", "path": str(args.probe_table), "sha256": sha256_file(args.probe_table)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
        *result["generated_files"],
    ]
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_input_preparation",
        "command": " ".join(["scripts/prepare_exception_counterfactual_probe_stream_inputs.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method-id", default="exception_counterfactual_probe")
    parser.add_argument("--experiment-id", default="eval300_exception_counterfactual_probe_stream_inputs_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.report, result)
    print(json.dumps({"result": str(args.result), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
