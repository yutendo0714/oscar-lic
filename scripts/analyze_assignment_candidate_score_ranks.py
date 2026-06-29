#!/usr/bin/env python3
"""Audit candidate score ranks for top-k assignment policies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", row.get("seed", 0)))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def quantiles(values: Iterable[float]) -> dict[str, float] | None:
    values = list(values)
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def analyze_model(rows: list[dict], eval_seed: int, model_seed: int, topk_values: list[int]) -> dict:
    eval_rows = [row for row in rows if int(row["seed"]) == eval_seed and int(row["model_seed"]) == model_seed]
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in eval_rows:
        groups[group_key(row)].append(row)

    oracle_rank_counts: Counter[int] = Counter()
    safe_rank_counts: Counter[int] = Counter()
    oracle_topk_hits = {f"top{k}": 0 for k in topk_values}
    safe_topk_hits = {f"top{k}": 0 for k in topk_values}
    oracle_scores = []
    safe_scores = []
    nearest_scores = []
    worsen_scores = []
    oracle_margins_vs_nearest = []
    safe_margins_vs_nearest = []
    harmful_top1_groups = 0
    improving_top1_groups = 0
    neutral_top1_groups = 0

    for group in groups.values():
        ordered = sorted(group, key=lambda row: (-float(row["score"]), int(row["topk_rank"])))
        nearest = next(row for row in group if int(row["is_nearest"]))
        nearest_score = float(nearest["score"])
        nearest_scores.append(nearest_score)
        top1 = ordered[0]
        if int(top1["tesseract_delta_vs_nearest"]) > 0:
            harmful_top1_groups += 1
        elif int(top1["tesseract_delta_vs_nearest"]) < 0:
            improving_top1_groups += 1
        else:
            neutral_top1_groups += 1

        for row in group:
            if int(row["tesseract_delta_vs_nearest"]) > 0:
                worsen_scores.append(float(row["score"]))

        oracle_rows = [row for row in group if int(row["label_assignment_oracle_choice"]) and not int(row["is_nearest"])]
        if oracle_rows:
            oracle_row = max(oracle_rows, key=lambda row: float(row["score"]))
            rank = ordered.index(oracle_row) + 1
            oracle_rank_counts[rank] += 1
            oracle_scores.append(float(oracle_row["score"]))
            oracle_margins_vs_nearest.append(float(oracle_row["score"]) - nearest_score)
            for topk in topk_values:
                if rank <= topk:
                    oracle_topk_hits[f"top{topk}"] += 1

        safe_rows = [row for row in group if int(row["label_tesseract_parseq_safe_improves"])]
        if safe_rows:
            safe_row = max(safe_rows, key=lambda row: float(row["score"]))
            rank = ordered.index(safe_row) + 1
            safe_rank_counts[rank] += 1
            safe_scores.append(float(safe_row["score"]))
            safe_margins_vs_nearest.append(float(safe_row["score"]) - nearest_score)
            for topk in topk_values:
                if rank <= topk:
                    safe_topk_hits[f"top{topk}"] += 1

    return {
        "model_seed": model_seed,
        "eval_seed": eval_seed,
        "groups": len(groups),
        "oracle_change_groups": int(sum(oracle_rank_counts.values())),
        "safe_improve_groups": int(sum(safe_rank_counts.values())),
        "oracle_rank_counts": {str(rank): int(count) for rank, count in sorted(oracle_rank_counts.items())},
        "safe_rank_counts": {str(rank): int(count) for rank, count in sorted(safe_rank_counts.items())},
        "oracle_topk_hits": oracle_topk_hits,
        "safe_topk_hits": safe_topk_hits,
        "top1_group_effect_counts": {
            "improving": improving_top1_groups,
            "neutral": neutral_top1_groups,
            "harmful": harmful_top1_groups,
        },
        "score_quantiles": {
            "oracle": quantiles(oracle_scores),
            "safe": quantiles(safe_scores),
            "nearest": quantiles(nearest_scores),
            "tesseract_worsen": quantiles(worsen_scores),
            "oracle_margin_vs_nearest": quantiles(oracle_margins_vs_nearest),
            "safe_margin_vs_nearest": quantiles(safe_margins_vs_nearest),
        },
    }


def aggregate(per_model: list[dict], topk_values: list[int]) -> dict:
    out = {
        "models": len(per_model),
        "groups_mean": float(np.mean([row["groups"] for row in per_model])) if per_model else 0.0,
    }
    for prefix in ["oracle", "safe"]:
        for topk in topk_values:
            key = f"{prefix}_top{topk}_hits"
            values = [row[f"{prefix}_topk_hits"][f"top{topk}"] for row in per_model]
            out[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
            out[f"{key}_min"] = int(min(values)) if values else 0
            out[f"{key}_max"] = int(max(values)) if values else 0
    for effect in ["improving", "neutral", "harmful"]:
        values = [row["top1_group_effect_counts"][effect] for row in per_model]
        out[f"top1_{effect}_groups_mean"] = float(np.mean(values)) if values else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-file", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument("--topk", type=int, action="append", default=[1, 2, 3, 4, 8])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.score_file) != len(args.label):
        raise SystemExit("--score-file and --label counts must match")

    outputs = []
    for path, label in zip(args.score_file, args.label):
        rows = read_jsonl(path)
        model_seeds = sorted({int(row["model_seed"]) for row in rows})
        per_model = [analyze_model(rows, args.eval_seed, model_seed, args.topk) for model_seed in model_seeds]
        outputs.append(
            {
                "label": label,
                "score_file": str(path),
                "eval_seed": args.eval_seed,
                "topk_values": args.topk,
                "per_model": per_model,
                "aggregate": aggregate(per_model, args.topk),
            }
        )
    result = {"description": "Candidate score-rank audit for top-k assignment policies.", "results": outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
