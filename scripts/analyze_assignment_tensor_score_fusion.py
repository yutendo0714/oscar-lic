#!/usr/bin/env python3
"""Evaluate simple score fusion policies for tensor-backed assignment rerankers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def policy_error_type(selected_changed: bool, selected_is_oracle: bool, oracle_changed: bool) -> str:
    if selected_is_oracle and oracle_changed:
        return "exact_oracle"
    if selected_is_oracle and not oracle_changed:
        return "correct_nearest"
    if selected_changed and not oracle_changed:
        return "false_change"
    if (not selected_changed) and oracle_changed:
        return "missed_oracle"
    return "wrong_change"


def policy_metrics(data: dict[str, np.ndarray], group_indices: np.ndarray, selected_pos: np.ndarray) -> dict:
    labels = data["labels"][group_indices]
    nearest = data["is_nearest"][group_indices].astype(bool)
    tesseract = data["tesseract_delta"][group_indices]
    parseq = data["parseq_delta"][group_indices]
    oracle_pos = labels.argmax(axis=1)
    nearest_pos = nearest.argmax(axis=1)
    selected_t = tesseract[np.arange(len(group_indices)), selected_pos]
    selected_p = parseq[np.arange(len(group_indices)), selected_pos]
    changed = selected_pos != nearest_pos
    oracle_changed = oracle_pos != nearest_pos
    selected_is_oracle = selected_pos == oracle_pos
    errors = Counter(
        policy_error_type(bool(ch), bool(ok), bool(oc))
        for ch, ok, oc in zip(changed, selected_is_oracle, oracle_changed)
    )
    return {
        "groups": int(len(group_indices)),
        "changed_groups": int(changed.sum()),
        "parseq_delta_sum": int(selected_p.sum()),
        "tesseract_delta_sum": int(selected_t.sum()),
        "parseq_worsen_groups": int((selected_p > 0).sum()),
        "tesseract_improve_groups": int((selected_t < 0).sum()),
        "tesseract_worsen_groups": int((selected_t > 0).sum()),
        "oracle_change_groups": int(oracle_changed.sum()),
        "exact_oracle_groups": int(errors["exact_oracle"]),
        "missed_oracle_groups": int(errors["missed_oracle"]),
        "false_change_groups": int(errors["false_change"]),
        "wrong_change_groups": int(errors["wrong_change"]),
        "correct_nearest_groups": int(errors["correct_nearest"]),
    }


def select_argmax(values: np.ndarray, mask: np.ndarray, group_indices: np.ndarray) -> np.ndarray:
    selected = []
    for group_index in group_indices:
        row = np.asarray(values[group_index], dtype=np.float64).copy()
        row[~mask[group_index]] = -np.inf
        selected.append(int(np.argmax(row)))
    return np.asarray(selected, dtype=np.int64)


def load_reranker_scores(data: dict[str, np.ndarray], paths: list[Path]) -> dict[int, np.ndarray]:
    group_count, max_candidates = data["mask"].shape
    scores_by_seed = defaultdict(lambda: np.full((group_count, max_candidates), np.nan, dtype=np.float32))
    group_lookup = {tuple(int(v) for v in key): index for index, key in enumerate(data["group_keys"])}
    for path in paths:
        for row in read_jsonl(path):
            key = (
                int(row["real_seed"]),
                int(row["seed"]),
                int(row["source_index"]),
                int(row["candidate_index"]),
            )
            group_index = group_lookup.get(key)
            if group_index is None:
                continue
            row_position = int(row["row_position"])
            if row_position < 0 or row_position >= max_candidates:
                continue
            scores_by_seed[int(row["model_seed"])][group_index, row_position] = float(row["score"])
    return dict(scores_by_seed)


def zscore_valid(values: np.ndarray, mask: np.ndarray, group_indices: np.ndarray) -> np.ndarray:
    out = values.copy().astype(np.float64)
    valid = mask[group_indices] & np.isfinite(out[group_indices])
    selected = out[group_indices][valid]
    if selected.size == 0:
        return out
    mean = float(selected.mean())
    std = float(selected.std())
    if std < 1.0e-8:
        std = 1.0
    out[np.isfinite(out)] = (out[np.isfinite(out)] - mean) / std
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, action="append", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--zscore", action="store_true")
    args = parser.parse_args()

    data = {key: value for key, value in np.load(args.features).items()}
    mask = data["mask"].astype(bool)
    val_groups = np.flatnonzero(data["group_keys"][:, 1] == args.val_seed)
    nearest = data["is_nearest"][val_groups].argmax(axis=1)
    oracle = data["labels"][val_groups].argmax(axis=1)
    baseline = data["baseline_score"].astype(np.float64)
    scores_by_seed = load_reranker_scores(data, args.score)
    score_stack = np.stack([scores for _, scores in sorted(scores_by_seed.items())], axis=0)
    finite = np.isfinite(score_stack)
    summed = np.where(finite, score_stack, 0.0).sum(axis=0)
    counts = finite.sum(axis=0)
    model_mean = np.full_like(summed, np.nan, dtype=np.float64)
    np.divide(summed, counts, out=model_mean, where=counts > 0)
    model_values = zscore_valid(model_mean, mask, val_groups) if args.zscore else model_mean
    baseline_values = zscore_valid(baseline, mask, val_groups) if args.zscore else baseline

    policies = {
        "nearest": policy_metrics(data, val_groups, nearest),
        "oracle": policy_metrics(data, val_groups, oracle),
        "baseline_score": policy_metrics(data, val_groups, select_argmax(baseline, mask, val_groups)),
        "model_mean": policy_metrics(data, val_groups, select_argmax(model_mean, mask, val_groups)),
    }
    for seed, scores in sorted(scores_by_seed.items()):
        policies[f"model_seed_{seed}"] = policy_metrics(data, val_groups, select_argmax(scores, mask, val_groups))
    for alpha in sorted(set(args.alpha)):
        fused = alpha * model_values + (1.0 - alpha) * baseline_values
        policies[f"fusion_alpha_{alpha:.2f}"] = policy_metrics(data, val_groups, select_argmax(fused, mask, val_groups))

    output = {
        "description": "Score fusion policies for tensor-backed assignment rerankers.",
        "features": str(args.features),
        "scores": [str(path) for path in args.score],
        "config": {"val_seed": args.val_seed, "alpha": sorted(set(args.alpha)), "zscore": args.zscore},
        "policies": policies,
        "aggregate": {"policy_metrics": policies},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(policies, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
