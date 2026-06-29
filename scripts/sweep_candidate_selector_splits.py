#!/usr/bin/env python3
"""Sweep group splits for one-candidate OSCAR-LIC selector diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from evaluate_candidate_selector import (
    baseline_score_feature,
    candidate_split_maps,
    choose_base,
    choose_oracle,
    choose_threshold,
    choose_top_images,
    evaluate_policy,
    read_analysis,
    read_jsonl,
    rows_by_image,
    train_linear_scores,
    tune_threshold,
)


def summarize_policy(policy: dict, evaluator_labels: list[str]) -> dict:
    result = {
        "selected_images": policy["selected_images"],
        "selected_fraction": policy["selected_fraction"],
        "avg_actual_total_bpp": policy["avg_actual_total_bpp"],
        "multi_teacher_delta_distance_sum": policy["multi_teacher_delta_distance_sum"],
    }
    for label in evaluator_labels:
        metrics = policy["evaluators"][label]
        result[f"{label}_cer_micro"] = metrics["cer_micro"]
        result[f"{label}_exact_rate"] = metrics["exact_rate"]
    return result


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    output = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--candidate-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", action="append", nargs=2, metavar=("LABEL", "ANALYSIS_JSON"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--image-budget", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--base-structural-overhead-bytes", type=int, default=88)
    args = parser.parse_args()

    if not args.evaluator:
        raise SystemExit("provide at least one --evaluator LABEL ANALYSIS_JSON")

    rows = read_jsonl(args.table)
    evaluator_labels = [label for label, _ in args.evaluator]
    analyses = {label: read_analysis(Path(path)) for label, path in args.evaluator}
    bpp_by_key = candidate_split_maps(args.candidate_split, args.base_structural_overhead_bytes)
    per_seed = []
    by_policy = defaultdict(list)

    for seed in args.seeds:
        scores, train_mask, val_mask, model_meta = train_linear_scores(
            rows,
            seed=seed,
            val_fraction=args.val_fraction,
            epochs=args.epochs,
            lr=args.lr,
        )
        baseline_feature, residual_scores = baseline_score_feature(rows)
        grouped_train = rows_by_image(rows, np.flatnonzero(train_mask))
        grouped_val = rows_by_image(rows, np.flatnonzero(val_mask))
        tuned = tune_threshold(grouped_train, scores)
        policy_specs = [
            ("base", choose_base(grouped_val)),
            ("oracle_single_candidate", choose_oracle(grouped_val)),
            ("linear_train_tuned", choose_threshold(grouped_val, scores, tuned["threshold"])),
        ]
        for budget in args.image_budget:
            budget = min(int(budget), len(grouped_val))
            policy_specs.append((f"linear_top{budget}_images", choose_top_images(grouped_val, scores, budget)))
            policy_specs.append(
                (f"residual_rms_top{budget}_images", choose_top_images(grouped_val, residual_scores, budget))
            )

        seed_rows = {}
        for name, selected in policy_specs:
            policy = evaluate_policy(name, selected, grouped_val, analyses, bpp_by_key, args.profile)
            compact = summarize_policy(policy, evaluator_labels)
            seed_rows[name] = compact
            by_policy[name].append(compact)
        per_seed.append(
            {
                "seed": seed,
                "model": {
                    key: model_meta[key]
                    for key in [
                        "train_rows",
                        "val_rows",
                        "train_positive_rate",
                        "val_positive_rate",
                        "linear_train_auc",
                        "linear_val_auc",
                        "linear_val_ap",
                    ]
                },
                "train_tuned_threshold": tuned,
                "residual_baseline_score_feature": baseline_feature,
                "policies": seed_rows,
            }
        )

    output = {
        "table": str(args.table),
        "candidate_split": str(args.candidate_split),
        "profile": args.profile,
        "seeds": args.seeds,
        "evaluators": evaluator_labels,
        "per_seed": per_seed,
        "aggregate_by_policy": {name: aggregate(rows) for name, rows in sorted(by_policy.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate_by_policy"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
