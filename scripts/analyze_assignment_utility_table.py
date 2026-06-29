#!/usr/bin/env python3
"""Analyze simple policies over top-k code-assignment utility labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    hits = 0
    total = 0.0
    for rank, label in enumerate(sorted_labels, start=1):
        if label:
            hits += 1
            total += hits / rank
    return total / positives


def group_rows(rows: list[dict]) -> dict[tuple[int, int, int], list[dict]]:
    grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))].append(row)
    return dict(grouped)


def nearest(group: list[dict]) -> dict:
    return next(row for row in group if int(row["is_nearest"]))


def oracle(group: list[dict]) -> dict:
    return next(row for row in group if int(row["label_assignment_oracle_choice"]))


def choose_rank(group: list[dict], rank: int) -> dict:
    return next((row for row in group if int(row["topk_rank"]) == rank), nearest(group))


def choose_first_under_relative_error(group: list[dict], threshold: float) -> dict:
    candidates = [
        row
        for row in group
        if int(row["topk_rank"]) > 0 and float(row["assignment_relative_error"]) <= threshold
    ]
    if not candidates:
        return nearest(group)
    return min(candidates, key=lambda row: (int(row["topk_rank"]), float(row["assignment_relative_error"])))


def choose_best_parseq_confidence(group: list[dict]) -> dict:
    return max(group, key=lambda row: (float(row.get("parseq_confidence") or 0.0), -int(row["topk_rank"])))


def compact_metrics(selected: list[dict]) -> dict:
    parseq_delta = int(sum(int(row["parseq_delta_vs_nearest"]) for row in selected))
    tesseract_delta = int(sum(int(row["tesseract_delta_vs_nearest"]) for row in selected))
    changed = int(sum(1 for row in selected if not int(row["is_nearest"])))
    worsens_parseq = int(sum(1 for row in selected if int(row["parseq_delta_vs_nearest"]) > 0))
    improves_tesseract = int(sum(1 for row in selected if int(row["tesseract_delta_vs_nearest"]) < 0))
    worsens_tesseract = int(sum(1 for row in selected if int(row["tesseract_delta_vs_nearest"]) > 0))
    relative_errors = [float(row["assignment_relative_error"]) for row in selected if not int(row["is_nearest"])]
    return {
        "groups": len(selected),
        "changed_groups": changed,
        "parseq_delta_sum": parseq_delta,
        "tesseract_delta_sum": tesseract_delta,
        "multi_teacher_delta_sum": parseq_delta + tesseract_delta,
        "parseq_worsen_groups": worsens_parseq,
        "tesseract_improve_groups": improves_tesseract,
        "tesseract_worsen_groups": worsens_tesseract,
        "mean_changed_relative_error": float(np.mean(relative_errors)) if relative_errors else 1.0,
    }


def evaluate_policies(grouped: dict[tuple[int, int, int], list[dict]]) -> dict:
    groups = list(grouped.values())
    policies: dict[str, list[dict]] = {
        "nearest": [nearest(group) for group in groups],
        "parseq_safe_tesseract_oracle": [oracle(group) for group in groups],
        "parseq_confidence_max": [choose_best_parseq_confidence(group) for group in groups],
    }
    for rank in range(1, 8):
        policies[f"fixed_rank_{rank}"] = [choose_rank(group, rank) for group in groups]
    for threshold in [1.001, 1.01, 1.03, 1.05, 1.10, 1.15, 1.20]:
        policies[f"first_nonnearest_rel_le_{threshold:.3f}"] = [
            choose_first_under_relative_error(group, threshold) for group in groups
        ]
    return {name: compact_metrics(selected) for name, selected in policies.items()}


def feature_scores(rows: list[dict]) -> dict:
    labels = np.asarray([int(row["label_tesseract_parseq_safe_improves"]) for row in rows], dtype=np.int64)
    values = {
        "negative_topk_rank": -np.asarray([float(row["topk_rank"]) for row in rows], dtype=np.float64),
        "negative_relative_error": -np.asarray([float(row["assignment_relative_error"]) for row in rows], dtype=np.float64),
        "parseq_confidence": np.asarray([float(row.get("parseq_confidence") or 0.0) for row in rows], dtype=np.float64),
        "nonnearest": np.asarray([1.0 - float(row["is_nearest"]) for row in rows], dtype=np.float64),
    }
    output = {}
    for name, scores in values.items():
        output[name] = {
            "auc": roc_auc(scores, labels),
            "ap": average_precision(scores, labels),
        }
    nonnearest_mask = np.asarray([not int(row["is_nearest"]) for row in rows], dtype=bool)
    if int(nonnearest_mask.sum()) > 0:
        nn_labels = labels[nonnearest_mask]
        for name, scores in values.items():
            output[f"{name}_nonnearest_only"] = {
                "auc": roc_auc(scores[nonnearest_mask], nn_labels),
                "ap": average_precision(scores[nonnearest_mask], nn_labels),
            }
    return output


def per_seed_policy_metrics(grouped: dict[tuple[int, int, int], list[dict]]) -> dict:
    output = {}
    for seed in sorted({key[0] for key in grouped}):
        seed_grouped = {key: group for key, group in grouped.items() if key[0] == seed}
        output[str(seed)] = evaluate_policies(seed_grouped)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.table)
    grouped = group_rows(rows)
    summary = {
        "description": "Simple policy diagnostics for top-k code-assignment utility labels.",
        "table": str(args.table),
        "rows": len(rows),
        "groups": len(grouped),
        "policy_metrics": evaluate_policies(grouped),
        "per_seed_policy_metrics": per_seed_policy_metrics(grouped),
        "feature_separability_for_tesseract_parseq_safe_rows": feature_scores(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
