#!/usr/bin/env python3
"""Evaluate no-op gates based on assignment-score margins."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def load_groups(paths: list[Path], model_seed: int) -> dict[tuple[int, int, int, int], list[dict]]:
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            if int(row.get("model_seed", -1)) != model_seed:
                continue
            groups[group_key(row)].append(row)
    return dict(groups)


def policy_metrics(selected: list[dict]) -> dict:
    return {
        "groups": len(selected),
        "changed_groups": int(sum(not int(row["is_nearest"]) for row in selected)),
        "parseq_delta_sum": int(sum(int(row["parseq_delta_vs_nearest"]) for row in selected)),
        "tesseract_delta_sum": int(sum(int(row["tesseract_delta_vs_nearest"]) for row in selected)),
        "parseq_worsen_groups": int(sum(int(row["parseq_delta_vs_nearest"]) > 0 for row in selected)),
        "tesseract_improve_groups": int(sum(int(row["tesseract_delta_vs_nearest"]) < 0 for row in selected)),
        "tesseract_worsen_groups": int(sum(int(row["tesseract_delta_vs_nearest"]) > 0 for row in selected)),
        "exact_oracle_groups": int(sum(int(row.get("label_assignment_oracle_choice", 0)) for row in selected)),
    }


def best_rows(group: list[dict]) -> tuple[dict, dict, dict]:
    nearest = next(row for row in group if int(row["is_nearest"]))
    best = max(group, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))
    oracle = next(row for row in group if int(row.get("label_assignment_oracle_choice", 0)))
    return nearest, best, oracle


def score_stats(group: list[dict]) -> dict:
    nearest, best, _ = best_rows(group)
    non_nearest = [row for row in group if not int(row["is_nearest"])]
    best_non_nearest = max(non_nearest, key=lambda row: (float(row["score"]), -int(row["topk_rank"])), default=nearest)
    ordered = sorted(group, key=lambda row: float(row["score"]), reverse=True)
    second = ordered[1] if len(ordered) > 1 else nearest
    return {
        "nearest": nearest,
        "best": best,
        "oracle": next(row for row in group if int(row.get("label_assignment_oracle_choice", 0))),
        "best_non_nearest": best_non_nearest,
        "best_score": float(best["score"]),
        "nearest_score": float(nearest["score"]),
        "best_non_nearest_score": float(best_non_nearest["score"]),
        "top_margin": float(best["score"]) - float(second["score"]),
        "non_nearest_margin": float(best_non_nearest["score"]) - float(nearest["score"]),
    }


def select_by_margin(groups: dict[tuple[int, int, int, int], list[dict]], threshold: float, mode: str) -> list[dict]:
    selected = []
    for key in sorted(groups):
        stats = score_stats(groups[key])
        nearest = stats["nearest"]
        best = stats["best"]
        if int(best["is_nearest"]):
            selected.append(nearest)
            continue
        gate_value = stats[mode]
        selected.append(best if gate_value >= threshold else nearest)
    return selected


def threshold_grid(values: list[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values))
    if not unique:
        return [1.0]
    return [max(unique) + 1e-6] + unique + [min(unique) - 1e-6]


def tune_threshold(groups: dict[tuple[int, int, int, int], list[dict]], mode: str) -> float:
    values = [score_stats(group)[mode] for group in groups.values()]
    best_threshold = 1.0
    best_key = None
    for threshold in threshold_grid(values):
        metrics = policy_metrics(select_by_margin(groups, threshold, mode))
        key = (
            metrics["parseq_worsen_groups"],
            metrics["tesseract_delta_sum"],
            metrics["tesseract_worsen_groups"],
            metrics["changed_groups"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def evaluate(groups: dict[tuple[int, int, int, int], list[dict]], mode: str, thresholds: list[float]) -> dict:
    output = {
        "nearest": policy_metrics([best_rows(groups[key])[0] for key in sorted(groups)]),
        "assignment_always_best": policy_metrics([best_rows(groups[key])[1] for key in sorted(groups)]),
        "oracle": policy_metrics([best_rows(groups[key])[2] for key in sorted(groups)]),
    }
    for threshold in thresholds:
        output[f"{mode}_threshold_{str(threshold).replace('.', '_').replace('-', 'm')}"] = policy_metrics(
            select_by_margin(groups, threshold, mode)
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--mode", choices=["top_margin", "non_nearest_margin"], default="non_nearest_margin")
    parser.add_argument("--threshold", type=float, action="append")
    args = parser.parse_args()

    groups = load_groups(args.assignment_score, model_seed=args.model_seed)
    train_groups = {key: group for key, group in groups.items() if key[1] != args.val_seed}
    val_groups = {key: group for key, group in groups.items() if key[1] == args.val_seed}
    tuned_threshold = tune_threshold(train_groups, args.mode)
    thresholds = list(args.threshold or [])
    thresholds.append(tuned_threshold)
    thresholds = sorted(set(float(value) for value in thresholds))

    output = {
        "description": "Assignment no-op gate based on assignment score margins.",
        "assignment_scores": [str(path) for path in args.assignment_score],
        "config": {
            "val_seed": args.val_seed,
            "model_seed": args.model_seed,
            "mode": args.mode,
            "thresholds": thresholds,
            "train_tuned_threshold": tuned_threshold,
        },
        "train_metrics": evaluate(train_groups, args.mode, thresholds),
        "val_metrics": evaluate(val_groups, args.mode, thresholds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["val_metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
