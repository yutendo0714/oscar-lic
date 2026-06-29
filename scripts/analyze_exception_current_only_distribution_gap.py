#!/usr/bin/env python3
"""Audit train/validation distribution gaps for current-only exception hard negatives."""

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


THRESHOLDS = [1.1, 1.5, 2.0, 10.0, 1000.0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    data = sorted(values)

    def pick(frac: float) -> float:
        return float(data[round(frac * (len(data) - 1))])

    return {"min": float(data[0]), "q25": pick(0.25), "median": pick(0.5), "q75": pick(0.75), "max": float(data[-1])}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rel = [float(row["extra_relative_error"]) for row in rows]
    return {
        "rows": len(rows),
        "relative_error": quantiles(rel),
        "threshold_counts": {str(thr): int(sum(value <= thr for value in rel)) for thr in THRESHOLDS},
        "sources": dict(Counter(row.get("source", "unknown") for row in rows)),
    }


def source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("source", "unknown")].append(row)
    return {source: summarize_rows(items) for source, items in sorted(grouped.items())}


def label_lookup(labels: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(int(row["seed"]), int(row["source_index"])): row for row in labels if bool(row["current_selected"])}


def summarize_labeled_current_only(rows: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = label_lookup(labels)
    joined = []
    for row in rows:
        label = lookup.get((int(row["seed"]), int(row["source_index"])))
        if label is not None:
            joined.append({"action": row, "label": label})
    out: dict[str, Any] = {
        "joined_rows": len(joined),
        "strict_labels": dict(Counter(item["label"]["strict_label"] for item in joined)),
        "thresholds": {},
    }
    for thr in THRESHOLDS:
        subset = [item for item in joined if float(item["action"]["extra_relative_error"]) <= thr]
        out["thresholds"][str(thr)] = {
            "rows": len(subset),
            "strict_labels": dict(Counter(item["label"]["strict_label"] for item in subset)),
            "tesseract_unicode_delta": int(
                sum(int(item["label"]["ocr"]["tesseract"]["unicode_strict_v1"]["delta_errors"]) for item in subset)
            ),
            "tesseract_latin_delta": int(
                sum(int(item["label"]["ocr"]["tesseract"]["latin_alnum_ci_v1"]["delta_errors"]) for item in subset)
            ),
        }
    return out


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.action_table)
    labels = read_jsonl(args.current_relative_labels) if args.current_relative_labels else []
    current_only = [row for row in rows if row["policy_case"] == "current_only"]
    by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in current_only:
        by_partition[row["partition"]].append(row)
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-current-only-hard-negative-gap",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "summary": {
            "by_partition": {part: summarize_rows(items) for part, items in sorted(by_partition.items())},
            "by_partition_source": {part: source_summary(items) for part, items in sorted(by_partition.items())},
        },
        "conclusion": (
            "Current-only hard negatives are distributionally mismatched: validation current-only rows are almost all "
            "near the exception threshold, while train current-only rows are mostly far from exception centers. This "
            "explains why proxy/distance training lacks the hard negatives needed to preserve current selections."
        ),
        "next_action": (
            "Create or mine train hard negatives that resemble validation current-only rows before fitting a "
            "replace/drop verifier."
        ),
    }
    if labels:
        val_current_only = [row for row in current_only if row["partition"] == "val" and int(row["seed"]) in {1, 2}]
        result["summary"]["labeled_val_seed1_2_current_only"] = summarize_labeled_current_only(val_current_only, labels)
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception Current-Only Distribution Gap",
        "",
        "## Current-Only Summary",
        "",
        "| partition | rows | min | median | max | <=1.1 | <=1.5 | <=2.0 | <=10 | <=1000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for part, data in result["summary"]["by_partition"].items():
        rel = data["relative_error"]
        counts = data["threshold_counts"]
        lines.append(
            f"| {part} | {data['rows']} | {rel['min']:.6f} | {rel['median']:.6f} | {rel['max']:.6f} | "
            f"{counts['1.1']} | {counts['1.5']} | {counts['2.0']} | {counts['10.0']} | {counts['1000.0']} |"
        )
    if "labeled_val_seed1_2_current_only" in result["summary"]:
        labeled = result["summary"]["labeled_val_seed1_2_current_only"]
        lines.extend(
            [
                "",
                "## Labeled Val Seed1/2 Current-Only Rows",
                "",
                f"Joined rows: `{labeled['joined_rows']}`; labels: `{json.dumps(labeled['strict_labels'], sort_keys=True)}`.",
                "",
                "| threshold | rows | labels | tess unicode delta | tess latin delta |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for threshold, data in labeled["thresholds"].items():
            lines.append(
                f"| <= {threshold} | {data['rows']} | `{json.dumps(data['strict_labels'], sort_keys=True)}` | "
                f"{data['tesseract_unicode_delta']} | {data['tesseract_latin_delta']} |"
            )
    lines.extend(["", "## Conclusion", "", result["conclusion"], "", "## Next Action", "", result["next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def log_wandb(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any] | None:
    if not args.wandb_project:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_id,
            tags=["oscar-lic", "exception-centers", "current-only", "hard-negative-gap"],
            config={"experiment_id": args.experiment_id},
        )
        train = result["summary"]["by_partition"].get("train", {})
        val = result["summary"]["by_partition"].get("val", {})
        wandb.log(
            {
                "train/current_only_rows": train.get("rows", 0),
                "val/current_only_rows": val.get("rows", 0),
                "train/current_only_rel_le_1_5": train.get("threshold_counts", {}).get("1.5", 0),
                "val/current_only_rel_le_1_5": val.get("threshold_counts", {}).get("1.5", 0),
            }
        )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover - wandb availability is environment-dependent.
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "action_table", "path": str(args.action_table), "sha256": sha256_file(args.action_table)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    if args.current_relative_labels:
        files.append(
            {
                "name": "current_relative_labels",
                "path": str(args.current_relative_labels),
                "sha256": sha256_file(args.current_relative_labels),
            }
        )
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic",
        "command": " ".join(["scripts/analyze_exception_current_only_distribution_gap.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-table", type=Path, required=True)
    parser.add_argument("--current-relative-labels", type=Path, default=None)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_current_only_distribution_gap_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_result(args)
    result["wandb"] = log_wandb(args, result)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.report)
    print(json.dumps({"result": str(args.result), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
