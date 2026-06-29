#!/usr/bin/env python3
"""Audit supervision balance for current-preserving exception action rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from analyze_exception_center_selector_audit import read_jsonl, sha256_file


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    partitions: dict[str, dict[str, Any]] = {}
    for part in sorted({str(row["partition"]) for row in rows}):
        part_rows = [row for row in rows if str(row["partition"]) == part]
        repl = [row for row in part_rows if row["policy_case"] == "current_and_exception"]
        current_only = [row for row in part_rows if row["policy_case"] == "current_only"]
        true_count = sum(1 for row in repl if bool(row["oracle_change_proxy"]))
        false_count = len(repl) - true_count
        by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in repl:
            source = str(row.get("source") or "unknown")
            by_source[source]["replacement_candidates"] += 1
            by_source[source]["proxy_true"] += int(bool(row["oracle_change_proxy"]))
            by_source[source]["proxy_false"] += int(not bool(row["oracle_change_proxy"]))
        partitions[part] = {
            "groups": len(part_rows),
            "replacement_candidates": len(repl),
            "replacement_proxy_true": true_count,
            "replacement_proxy_false": false_count,
            "replacement_proxy_positive_rate": true_count / len(repl) if repl else None,
            "current_only_drop_risk": len(current_only),
            "by_source": {source: dict(values) for source, values in sorted(by_source.items())},
        }
    train = partitions.get("train", {})
    val = partitions.get("val", {})
    train_false = int(train.get("replacement_proxy_false", 0) or 0)
    val_false = int(val.get("replacement_proxy_false", 0) or 0)
    return {
        "partitions": partitions,
        "gap": {
            "train_replacement_false": train_false,
            "val_replacement_false": val_false,
            "false_count_gap_val_minus_train": val_false - train_false,
            "train_has_insufficient_replacement_negatives": train_false < 5,
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception Action Supervision Gap",
        "",
        "This audit uses the N143 action-readiness table. It is a supervision diagnostic, not a policy result.",
        "",
        "## Replacement Balance",
        "",
        "| partition | groups | replacement candidates | proxy true | proxy false | positive rate | current-only drop risk |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for part, values in result["summary"]["partitions"].items():
        rate = values["replacement_proxy_positive_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.4f}"
        lines.append(
            f"| {part} | {values['groups']} | {values['replacement_candidates']} | "
            f"{values['replacement_proxy_true']} | {values['replacement_proxy_false']} | "
            f"{rate_text} | {values['current_only_drop_risk']} |"
        )
    lines.extend(["", "## Source Breakdown", ""])
    for part, values in result["summary"]["partitions"].items():
        lines.extend([f"### {part}", "", "| source | replacement candidates | proxy true | proxy false |", "|---|---:|---:|---:|"])
        for source, source_values in values["by_source"].items():
            lines.append(
                f"| {source} | {source_values.get('replacement_candidates', 0)} | "
                f"{source_values.get('proxy_true', 0)} | {source_values.get('proxy_false', 0)} |"
            )
        lines.append("")
    lines.extend(["## Conclusion", "", result["conclusion"], "", "## Next Action", "", result["next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    data = {
        "experiment_id": args.experiment_id,
        "status": "completed_supervision_audit",
        "command": " ".join(["scripts/analyze_exception_action_table_supervision_gap.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": [
            {"name": "action_table", "path": str(args.table), "sha256": sha256_file(args.table)},
            {"name": "script", "path": str(script), "sha256": sha256_file(script)},
            {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
            {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_exception_action_supervision_gap_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.table)
    summary = summarize_rows(rows)
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-center-replace-drop-precheck",
        "status": "completed_supervision_audit",
        "code_commit": git_commit(),
        "summary": summary,
        "conclusion": (
            "The rel<=1.0 replacement cohort is not a usable standalone training target: train has only "
            f"{summary['gap']['train_replacement_false']} proxy-false replacement candidate, while validation has "
            f"{summary['gap']['val_replacement_false']}. This explains why distance-only exception policies look "
            "high-precision on train and false-heavy on validation."
        ),
        "next_action": (
            "Collect or derive explicit current-relative replacement/drop labels, or add hard replacement-negative "
            "supervision, before training a replace/drop verifier."
        ),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.report)
    print(json.dumps({"result": str(args.result), "summary": summary["gap"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
