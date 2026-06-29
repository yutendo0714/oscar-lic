#!/usr/bin/env python3
"""Audit which assignment features separate oracle codes from scorer choices."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


CORE_FEATURES = {
    "topk_rank",
    "assignment_relative_error",
    "is_nearest",
    "code_equals_nearest",
    "reference_length",
    "parseq_nearest_distance",
    "tesseract_nearest_distance",
}
FEATURE_PREFIXES = ("codebook_", "img_", "latent_vec_")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def row_real_seed(row: dict) -> int:
    return int(row.get("real_seed", row.get("original_seed", row.get("seed", 0))))


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (row_real_seed(row), int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))


def code_key(row: dict) -> tuple[int, int, int, int, int]:
    return (*group_key(row), int(row["code_index"]))


def feature_family(name: str) -> str:
    if name.startswith("codebook_"):
        return "codebook"
    if name.startswith("img_"):
        return "image_diff"
    if name.startswith("latent_vec_"):
        return "latent_vec"
    return "core"


def numeric_features(rows: Iterable[dict]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in CORE_FEATURES or any(key.startswith(prefix) for prefix in FEATURE_PREFIXES):
                if is_number(value):
                    names.add(key)
    return sorted(names)


def feature_value(row: dict, name: str) -> float:
    value = float(row.get(name, 0.0))
    if name == "assignment_relative_error":
        return math.log1p(min(max(value, 0.0), 1.0e8))
    return value


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


def summarize_values(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "mean_abs": float(np.abs(arr).mean()),
        "pos_fraction": float((arr > 0).mean()),
        "neg_fraction": float((arr < 0).mean()),
        "zero_fraction": float((arr == 0).mean()),
    }


def policy_error_type(best_code: int, nearest_code: int, oracle_code: int) -> str:
    if best_code == oracle_code and oracle_code != nearest_code:
        return "exact_oracle"
    if best_code == oracle_code and oracle_code == nearest_code:
        return "correct_nearest"
    if best_code != nearest_code and oracle_code == nearest_code:
        return "false_change"
    if best_code == nearest_code and oracle_code != nearest_code:
        return "missed_oracle"
    return "wrong_change"


def summarize_feature_auc(rows: list[dict], features: list[str]) -> list[dict]:
    labels = np.asarray([int(row["label_assignment_oracle_choice"]) for row in rows], dtype=np.float64)
    records = []
    for name in features:
        values = np.asarray([feature_value(row, name) for row in rows], dtype=np.float64)
        if float(values.std()) < 1.0e-12:
            continue
        auc = roc_auc(values, labels)
        if auc is None:
            continue
        oriented_auc = max(float(auc), 1.0 - float(auc))
        records.append(
            {
                "feature": name,
                "family": feature_family(name),
                "auc": float(auc),
                "oriented_auc": oriented_auc,
                "direction": "high_is_oracle" if auc >= 0.5 else "low_is_oracle",
            }
        )
    return sorted(records, key=lambda row: row["oriented_auc"], reverse=True)


def load_scores(paths: list[Path], val_seed: int, model_seed: int) -> dict[tuple[int, int, int, int, int], float]:
    scores = {}
    for path in paths:
        for row in read_jsonl(path):
            if int(row.get("val_seed", val_seed)) != val_seed:
                continue
            if int(row.get("model_seed", model_seed)) != model_seed:
                continue
            if int(row.get("seed", val_seed)) != val_seed:
                continue
            scores[code_key(row)] = float(row["score"])
    return scores


def summarize_group_deltas(
    groups: dict[tuple[int, int, int, int], list[dict]],
    features: list[str],
    score_by_code: dict[tuple[int, int, int, int, int], float],
) -> tuple[dict, list[dict]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    group_records = []
    for key, rows in sorted(groups.items()):
        nearest = next(row for row in rows if int(row["is_nearest"]))
        oracle_matches = [row for row in rows if int(row["label_assignment_oracle_choice"])]
        oracle = oracle_matches[0] if oracle_matches else nearest
        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        scored = sorted(
            rows,
            key=lambda row: (
                score_by_code.get((*key, int(row["code_index"])), float("-inf")),
                -int(row["topk_rank"]),
            ),
            reverse=True,
        )
        best = scored[0] if score_by_code else nearest
        best_code = int(best["code_index"])
        error_type = policy_error_type(best_code, nearest_code, oracle_code) if score_by_code else "no_score"
        group_records.append(
            {
                "real_seed": key[0],
                "seed": key[1],
                "source_index": key[2],
                "candidate_index": key[3],
                "source": str(nearest.get("source", "")),
                "reference": str(nearest.get("reference", "")),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "best_score_code": best_code,
                "oracle_changed": int(oracle_code != nearest_code),
                "best_score_changed": int(best_code != nearest_code),
                "error_type": error_type,
                "oracle_tesseract_delta": int(oracle["tesseract_delta_vs_nearest"]),
                "best_score_tesseract_delta": int(best["tesseract_delta_vs_nearest"]),
                "oracle_parseq_delta": int(oracle["parseq_delta_vs_nearest"]),
                "best_score_parseq_delta": int(best["parseq_delta_vs_nearest"]),
            }
        )
        for name in features:
            oracle_delta = feature_value(oracle, name) - feature_value(nearest, name)
            best_delta = feature_value(best, name) - feature_value(nearest, name)
            if oracle_code != nearest_code:
                buckets["oracle_change_delta"][name].append(oracle_delta)
                buckets[f"oracle_change_{error_type}_oracle_delta"][name].append(oracle_delta)
                buckets[f"oracle_change_{error_type}_best_delta"][name].append(best_delta)
                buckets[f"oracle_change_{error_type}_oracle_minus_best_delta"][name].append(
                    oracle_delta - best_delta
                )
            if best_code != nearest_code:
                buckets[f"best_change_{error_type}_delta"][name].append(best_delta)
    summarized = {}
    for bucket_name, feature_values in buckets.items():
        records = []
        for feature, values in feature_values.items():
            stats = summarize_values(values)
            if stats["count"] == 0:
                continue
            records.append({"feature": feature, "family": feature_family(feature), **stats})
        summarized[bucket_name] = sorted(records, key=lambda row: row["mean_abs"], reverse=True)
    return summarized, group_records


def summarize_family_auc(feature_auc: list[dict]) -> dict:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in feature_auc:
        buckets[row["family"]].append(float(row["oriented_auc"]))
    return {
        family: {
            "features": len(values),
            "mean_oriented_auc": float(np.mean(values)),
            "max_oriented_auc": float(np.max(values)),
        }
        for family, values in sorted(buckets.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, action="append", required=True)
    parser.add_argument("--score", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    rows = []
    for path in args.table:
        rows.extend(row for row in read_jsonl(path) if int(row["seed"]) == args.val_seed)
    if not rows:
        raise SystemExit("no rows after val-seed filtering")

    features = numeric_features(rows)
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    score_by_code = load_scores(args.score, val_seed=args.val_seed, model_seed=args.model_seed) if args.score else {}

    feature_auc = summarize_feature_auc(rows, features)
    group_deltas, group_records = summarize_group_deltas(groups, features, score_by_code)
    error_counts = Counter(row["error_type"] for row in group_records)
    oracle_change = [row for row in group_records if row["oracle_changed"]]

    scalar_metrics = {
        "rows": len(rows),
        "groups": len(groups),
        "features": len(features),
        "oracle_change_groups": len(oracle_change),
        "best_score_tesseract_delta_sum": int(sum(row["best_score_tesseract_delta"] for row in group_records)),
        "oracle_tesseract_delta_sum": int(sum(row["oracle_tesseract_delta"] for row in group_records)),
        "best_score_parseq_delta_sum": int(sum(row["best_score_parseq_delta"] for row in group_records)),
        "oracle_parseq_delta_sum": int(sum(row["oracle_parseq_delta"] for row in group_records)),
    }
    for name, count in error_counts.items():
        scalar_metrics[f"error_count/{name}"] = int(count)

    output = {
        "description": "Feature audit for assignment oracle-choice rows, scorer choices and missed-oracle gaps.",
        "tables": [str(path) for path in args.table],
        "scores": [str(path) for path in args.score],
        "config": {"val_seed": args.val_seed, "model_seed": args.model_seed, "top_n": args.top_n},
        "rows": len(rows),
        "groups": len(groups),
        "features": features,
        "feature_family_summary": summarize_family_auc(feature_auc),
        "top_univariate_features": feature_auc[: args.top_n],
        "top_group_delta_features": {
            name: values[: args.top_n] for name, values in sorted(group_deltas.items())
        },
        "group_error_counts": dict(error_counts),
        "group_examples": {
            name: [
                row
                for row in sorted(
                    (record for record in group_records if record["error_type"] == name),
                    key=lambda record: (
                        record["oracle_tesseract_delta"] - record["best_score_tesseract_delta"],
                        record["source"],
                        record["reference"],
                    ),
                )[:20]
            ]
            for name in sorted(error_counts)
        },
        "aggregate": {
            "scalar_metrics": scalar_metrics,
            "feature_family_summary": summarize_family_auc(feature_auc),
            "group_error_counts": dict(error_counts),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
