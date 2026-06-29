#!/usr/bin/env python3
"""Evaluate diagnostic OCR-difficulty gates on assignment score policies.

This is an upper-control analysis: distance features come from OCR/reference
tables and are not deployable unless an equivalent predictor is learned.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def variant_key(row: dict) -> tuple[int, int, int, int, int]:
    return (*group_key(row), int(row["code_index"]))


def policy_metrics(groups: dict[tuple[int, int, int, int], list[dict]], selected: list[dict]) -> dict:
    changed = 0
    for row in selected:
        nearest = next(candidate for candidate in groups[group_key(row)] if int(candidate["is_nearest"]))
        changed += int(int(row["code_index"]) != int(nearest["code_index"]))
    return {
        "groups": len(selected),
        "changed_groups": changed,
        "parseq_delta_sum": sum(int(row["parseq_delta_vs_nearest"]) for row in selected),
        "tesseract_delta_sum": sum(int(row["tesseract_delta_vs_nearest"]) for row in selected),
        "parseq_worsen_groups": sum(int(row["parseq_delta_vs_nearest"]) > 0 for row in selected),
        "tesseract_improve_groups": sum(int(row["tesseract_delta_vs_nearest"]) < 0 for row in selected),
        "tesseract_worsen_groups": sum(int(row["tesseract_delta_vs_nearest"]) > 0 for row in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, action="append", required=True)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--threshold", type=int, action="append")
    args = parser.parse_args()
    if args.threshold is None:
        args.threshold = [0, 1, 2, 3, 4, 5, 6, 8, 10]

    table_by_variant = {}
    for table_path in args.table:
        for row in read_jsonl(table_path):
            table_by_variant[variant_key(row)] = row

    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for score_path in args.score:
        for row in read_jsonl(score_path):
            if int(row.get("model_seed", -1)) != args.model_seed:
                continue
            if int(row.get("seed", -1)) != args.val_seed:
                continue
            table_row = table_by_variant.get(variant_key(row), {})
            merged = dict(row)
            for key in [
                "source",
                "reference",
                "tesseract_nearest_distance",
                "parseq_nearest_distance",
            ]:
                if key in table_row:
                    merged[key] = table_row[key]
            groups[group_key(merged)].append(merged)

    policies = []
    for threshold in args.threshold:
        selected = []
        for group in groups.values():
            nearest = next(row for row in group if int(row["is_nearest"]))
            best = max(group, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))
            distance = int(nearest.get("tesseract_nearest_distance", 0))
            selected.append(best if distance >= threshold else nearest)
        metrics = policy_metrics(groups, selected)
        policies.append({"tesseract_distance_threshold": threshold, **metrics})

    best = min(
        policies,
        key=lambda row: (
            row["parseq_worsen_groups"],
            row["tesseract_delta_sum"],
            row["tesseract_worsen_groups"],
            row["changed_groups"],
        ),
    )
    output = {
        "description": "Diagnostic assignment difficulty-gate sweep; uses OCR/reference table distances.",
        "tables": [str(path) for path in args.table],
        "scores": [str(path) for path in args.score],
        "config": {
            "val_seed": args.val_seed,
            "model_seed": args.model_seed,
            "thresholds": args.threshold,
            "difficulty_feature": "tesseract_nearest_distance",
            "deployable": False,
        },
        "policies": policies,
        "best_policy": best,
        "aggregate": {
            "best_policy": best,
            "by_threshold": {str(row["tesseract_distance_threshold"]): row for row in policies},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
