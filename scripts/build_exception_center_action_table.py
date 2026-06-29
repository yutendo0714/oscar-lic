#!/usr/bin/env python3
"""Build a current-preserving exception-center action-readiness table."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml

from analyze_exception_center_selector_audit import (
    build_group_rows,
    git_commit,
    parse_seed_path,
    read_jsonl,
    sha256_file,
)


SELECTION_RE = re.compile(r"seed(?P<seed>\d+)_(?P<partition>train225|val75)_")


def infer_selection_key(path: Path) -> tuple[int, str]:
    match = SELECTION_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer seed/partition from selection filename: {path}")
    partition = "train" if match.group("partition") == "train225" else "val"
    return int(match.group("seed")), partition


def load_current_selections(paths: list[Path]) -> dict[tuple[int, str], set[tuple[int, int]]]:
    selections: dict[tuple[int, str], set[tuple[int, int]]] = defaultdict(set)
    for path in paths:
        key = infer_selection_key(path)
        for row in read_jsonl(path):
            if "selected_by_policy" in row and not bool(row["selected_by_policy"]):
                continue
            selections[key].add((int(row["source_index"]), int(row["candidate_index"])))
    return selections


def action_space(current_selected: bool, exception_eligible: bool) -> list[str]:
    if current_selected:
        actions = ["keep_current", "drop_current"]
        if exception_eligible:
            actions.insert(1, "replace_with_exception")
        return actions
    actions = ["keep_nearest"]
    if exception_eligible:
        actions.append("add_exception")
    return actions


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_partition: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_case: dict[str, int] = defaultdict(int)
    for row in rows:
        part = row["partition"]
        case = row["policy_case"]
        by_case[case] += 1
        metrics = by_partition[part]
        metrics["groups"] += 1
        metrics["oracle_change_proxy"] += int(row["oracle_change_proxy"])
        metrics["current_selected"] += int(row["current_selected"])
        metrics["exception_eligible"] += int(row["exception_eligible"])
        metrics["current_and_exception"] += int(row["current_selected"] and row["exception_eligible"])
        metrics["current_only"] += int(row["current_selected"] and not row["exception_eligible"])
        metrics["exception_only"] += int((not row["current_selected"]) and row["exception_eligible"])
        metrics["neither"] += int((not row["current_selected"]) and not row["exception_eligible"])
        metrics["exception_proxy_true"] += int(row["exception_eligible"] and row["oracle_change_proxy"])
        metrics["exception_proxy_false"] += int(row["exception_eligible"] and not row["oracle_change_proxy"])
        metrics["current_drop_risk_proxy"] += int(row["current_selected"] and not row["exception_eligible"])
    return {
        "by_partition": {part: dict(values) for part, values in sorted(by_partition.items())},
        "policy_cases": dict(sorted(by_case.items())),
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    current = load_current_selections(args.current_selection)
    rows = []
    for row in build_group_rows(args):
        seed, _split_seed, source_index, candidate_index = [int(value) for value in row["group_key"]]
        partition = str(row["partition"])
        current_selected = (source_index, candidate_index) in current.get((seed, partition), set())
        exception_eligible = float(row["extra_relative_error"]) <= args.threshold
        if current_selected and exception_eligible:
            case = "current_and_exception"
        elif current_selected:
            case = "current_only"
        elif exception_eligible:
            case = "exception_only"
        else:
            case = "neither"
        rows.append(
            {
                "group_key": row["group_key"],
                "seed": seed,
                "partition": partition,
                "source_index": source_index,
                "candidate_index": candidate_index,
                "source": row.get("source"),
                "reference": row.get("reference"),
                "current_selected": current_selected,
                "exception_eligible": exception_eligible,
                "policy_case": case,
                "action_space": action_space(current_selected, exception_eligible),
                "oracle_change_proxy": bool(row["oracle_change"]),
                "nearest_code": int(row["nearest_code"]),
                "oracle_code": int(row["oracle_code"]),
                "base_best_code": int(row["base_best_code"]),
                "extra_best_code": int(row["extra_best_code"]),
                "extra_relative_error": float(row["extra_relative_error"]),
                "extra_distance_margin": float(row["extra_distance_margin"]),
                "extra_second_margin": float(row["extra_second_margin"]),
                "base_best_distance": float(row["base_best_distance"]),
                "extra_best_distance": float(row["extra_best_distance"]),
                "residual_norm": float(row["residual_norm"]),
                "base_center_utility": float(row["base_center_utility"]),
                "extra_center_utility": float(row["extra_center_utility"]),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception-Center Action Readiness Table",
        "",
        "This is a table-only precheck artifact for ADR-044. It does not claim OCR improvement.",
        "",
        "## Summary",
        "",
        "| partition | groups | current selected | exception eligible | both | current only | exception only | neither | exception proxy true | exception proxy false |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for part, values in result["summary"]["by_partition"].items():
        lines.append(
            f"| {part} | {values.get('groups', 0)} | {values.get('current_selected', 0)} | "
            f"{values.get('exception_eligible', 0)} | {values.get('current_and_exception', 0)} | "
            f"{values.get('current_only', 0)} | {values.get('exception_only', 0)} | "
            f"{values.get('neither', 0)} | {values.get('exception_proxy_true', 0)} | "
            f"{values.get('exception_proxy_false', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `current_only` rows are current selections that a sparse exception-only policy would drop.",
            "- `exception_only` rows are new exception additions with no current selected enhancement.",
            "- `current_and_exception` rows are the only direct replacement candidates.",
            "- `exception_proxy_false` is the no-headroom pressure seen by distance-only exception eligibility.",
            "",
            "Before actual streams, a verifier should reduce `exception_proxy_false`, preserve useful `current_only` rows, and justify every replacement or drop relative to current.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, table_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "features", "path": str(args.features), "sha256": sha256_file(args.features)},
        {"name": "assignment_table", "path": str(args.table), "sha256": sha256_file(args.table)},
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "action_table", "path": str(table_path), "sha256": sha256_file(table_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for seed, codebook in args.codebook:
        files.append({"name": f"codebook_seed{seed}", "path": str(codebook), "sha256": sha256_file(codebook)})
    for selection in args.current_selection:
        seed, partition = infer_selection_key(selection)
        files.append(
            {
                "name": f"current_selection_seed{seed}_{partition}",
                "path": str(selection),
                "sha256": sha256_file(selection),
            }
        )
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_table_precheck",
        "command": " ".join(["scripts/build_exception_center_action_table.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--codebook", type=parse_seed_path, action="append", required=True)
    parser.add_argument("--current-selection", type=Path, action="append", required=True)
    parser.add_argument("--base-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_exception_center_action_readiness_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_table, rows)
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-center-replace-drop-precheck",
        "status": "completed_table_precheck",
        "code_commit": git_commit(),
        "threshold": args.threshold,
        "rows": len(rows),
        "action_table": str(args.output_table),
        "summary": summarize(rows),
        "conclusion": (
            "This table exposes the ADR-044 action surface before any further OCR stream generation: "
            "current-only rows are drop risks, exception-only rows are add risks, and current-and-exception "
            "rows are direct replacement candidates."
        ),
        "next_action": (
            "Train or audit a current-preserving replace/drop verifier on this action table; do not export "
            "streams until table-level current-relative OCR and no-headroom checks pass."
        ),
    }
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.output_table, args.report)
    print(json.dumps({"result": str(args.result), "rows": len(rows), "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
