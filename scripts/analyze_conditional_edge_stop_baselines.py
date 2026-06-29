#!/usr/bin/env python3
"""Simple transition-aware stop/continue baselines for conditional edge tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Callable

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metrics(rows: list[dict], selected: list[bool], target: str, profile: str) -> dict:
    labels = [bool(row[target]) for row in rows]
    tp = sum(1 for y, s in zip(labels, selected, strict=True) if y and s)
    fp = sum(1 for y, s in zip(labels, selected, strict=True) if not y and s)
    tn = sum(1 for y, s in zip(labels, selected, strict=True) if not y and not s)
    fn = sum(1 for y, s in zip(labels, selected, strict=True) if y and not s)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    chosen = [row for row, s in zip(rows, selected, strict=True) if s]
    return {
        "rows": len(rows),
        "positives": int(sum(labels)),
        "selected": int(sum(selected)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tesseract_delta_sum": int(sum(row[f"tesseract_delta_{profile}"] for row in chosen)),
        "parseq_delta_sum": int(sum(row[f"parseq_delta_{profile}"] for row in chosen)),
        "mean_added_enhancement_bpp_selected": None
        if not chosen
        else float(np.mean([row["added_enhancement_bpp"] for row in chosen])),
    }


def fixed_policy(rows: list[dict], predicate: Callable[[dict], bool], target: str, profile: str) -> dict:
    return metrics(rows, [predicate(row) for row in rows], target, profile)


def transition_rates(rows: list[dict], target: str) -> dict[str, float]:
    counts: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        counts[row["transition"]].append(1 if row[target] else 0)
    return {key: float(np.mean(values)) for key, values in counts.items()}


def crossfit_transition_threshold(rows: list[dict], target: str, profile: str, folds: int, threshold: float) -> dict:
    selected = []
    fold_summaries = []
    for row in rows:
        val_fold = int(row["index"]) % folds
        train = [candidate for candidate in rows if int(candidate["index"]) % folds != val_fold]
        rates = transition_rates(train, target)
        select_transitions = {transition for transition, rate in rates.items() if rate >= threshold}
        selected.append(row["transition"] in select_transitions)
        fold_summaries.append({"fold": val_fold, "rates": rates, "selected_transitions": sorted(select_transitions)})
    out = metrics(rows, selected, target, profile)
    out["threshold"] = threshold
    # Collapse identical fold summaries.
    unique = {}
    for item in fold_summaries:
        key = item["fold"]
        unique[key] = item
    out["folds"] = [unique[key] for key in sorted(unique)]
    return out


def aggregate_metric(prefix: str, value: object, out: dict) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            aggregate_metric(f"{prefix}_{key}", child, out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-table", type=Path, required=True)
    parser.add_argument("--target", default="label_tesseract_improve_parseq_safe_unicode_strict_v1")
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="conditional_edge_stop_baselines")
    args = parser.parse_args()

    rows = read_jsonl(args.edge_table)
    if not rows:
        raise SystemExit("empty edge table")
    if args.target not in rows[0]:
        raise SystemExit(f"target not found: {args.target}")

    fixed = {
        "none": fixed_policy(rows, lambda row: False, args.target, args.profile),
        "all": fixed_policy(rows, lambda row: True, args.target, args.profile),
        "first_only": fixed_policy(rows, lambda row: row["transition"] == "0_to_1", args.target, args.profile),
        "first_or_third": fixed_policy(rows, lambda row: row["transition"] in {"0_to_1", "2_to_3"}, args.target, args.profile),
        "continue_only": fixed_policy(rows, lambda row: row["transition"] != "0_to_1", args.target, args.profile),
    }
    crossfit = {
        f"transition_rate_ge_{threshold:.2f}": crossfit_transition_threshold(
            rows, args.target, args.profile, args.folds, threshold
        )
        for threshold in (0.05, 0.10, 0.20, 0.50, 0.80, 0.95)
    }
    result = {
        "experiment_id": args.experiment_id,
        "edge_table": str(args.edge_table),
        "target": args.target,
        "profile": args.profile,
        "folds": args.folds,
        "transition_positive_rates": transition_rates(rows, args.target),
        "fixed_policies": fixed,
        "crossfit_policies": crossfit,
    }
    scalar_metrics = {}
    aggregate_metric("fixed_policies", fixed, scalar_metrics)
    aggregate_metric("crossfit_policies", crossfit, scalar_metrics)
    result["aggregate"] = {"scalar_metrics": {key: {"value": value} for key, value in scalar_metrics.items()}}

    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


def metric_cell(row: dict, key: str) -> str:
    value = row[key]
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(path: Path, result: dict) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        f"Target: `{result['target']}`",
        f"Transition positive rates: `{result['transition_positive_rates']}`",
        "",
        "## Fixed Policies",
        "",
        "| policy | selected | TP/FP/FN | precision | recall | F1 | Tess delta | PARSeq delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["fixed_policies"].items():
        lines.append(
            f"| {name} | {row['selected']} | {row['tp']}/{row['fp']}/{row['fn']} | "
            f"{metric_cell(row, 'precision')} | {metric_cell(row, 'recall')} | {metric_cell(row, 'f1')} | "
            f"{row['tesseract_delta_sum']} | {row['parseq_delta_sum']} |"
        )
    lines.extend(["", "## Cross-Fit Transition Thresholds", ""])
    lines.append("| policy | selected | TP/FP/FN | precision | recall | F1 | Tess delta | PARSeq delta |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for name, row in result["crossfit_policies"].items():
        lines.append(
            f"| {name} | {row['selected']} | {row['tp']}/{row['fp']}/{row['fn']} | "
            f"{metric_cell(row, 'precision')} | {metric_cell(row, 'recall')} | {metric_cell(row, 'f1')} | "
            f"{row['tesseract_delta_sum']} | {row['parseq_delta_sum']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a tiny diagnostic over OCR-derived labels. "
            "A strong first-only baseline means the next model needs transition/context awareness before capacity.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
