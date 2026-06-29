#!/usr/bin/env python3
"""Sweep no-op margin gates for tensor-backed assignment reranker scores."""

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
            if 0 <= row_position < max_candidates:
                scores_by_seed[int(row["model_seed"])][group_index, row_position] = float(row["score"])
    return dict(scores_by_seed)


def mean_scores(scores_by_seed: dict[int, np.ndarray]) -> np.ndarray:
    stack = np.stack([scores for _, scores in sorted(scores_by_seed.items())], axis=0)
    finite = np.isfinite(stack)
    summed = np.where(finite, stack, 0.0).sum(axis=0)
    counts = finite.sum(axis=0)
    out = np.full_like(summed, np.nan, dtype=np.float64)
    np.divide(summed, counts, out=out, where=counts > 0)
    return out


def select_margin_gate(
    values: np.ndarray,
    mask: np.ndarray,
    nearest_mask: np.ndarray,
    group_indices: np.ndarray,
    threshold: float,
    parseq_delta: np.ndarray | None = None,
    parseq_safe: bool = False,
) -> np.ndarray:
    selected = []
    for group_index in group_indices:
        valid = mask[group_index] & np.isfinite(values[group_index])
        nearest_positions = np.flatnonzero(nearest_mask[group_index] & valid)
        if nearest_positions.size == 0:
            selected.append(int(np.flatnonzero(valid)[0]))
            continue
        nearest_pos = int(nearest_positions[0])
        candidate_valid = valid & (~nearest_mask[group_index])
        if parseq_safe and parseq_delta is not None:
            candidate_valid &= parseq_delta[group_index] <= 0
        if not candidate_valid.any():
            selected.append(nearest_pos)
            continue
        candidate_scores = values[group_index].copy()
        candidate_scores[~candidate_valid] = -np.inf
        best_pos = int(np.argmax(candidate_scores))
        margin = float(values[group_index, best_pos] - values[group_index, nearest_pos])
        selected.append(best_pos if margin >= threshold else nearest_pos)
    return np.asarray(selected, dtype=np.int64)


def best_policy(rows: list[dict], require_parseq_safe: bool) -> dict:
    candidates = rows
    if require_parseq_safe:
        safe = [row for row in rows if row["metrics"]["parseq_delta_sum"] <= 0]
        if safe:
            candidates = safe
    return min(
        candidates,
        key=lambda row: (
            row["metrics"]["tesseract_delta_sum"],
            row["metrics"]["parseq_delta_sum"],
            row["metrics"]["false_change_groups"],
            -row["metrics"]["exact_oracle_groups"],
        ),
    )


def sweep_thresholds(
    data: dict[str, np.ndarray],
    scores: np.ndarray,
    mask: np.ndarray,
    nearest_mask: np.ndarray,
    group_indices: np.ndarray,
    thresholds: np.ndarray,
    parseq_safe: bool,
) -> list[dict]:
    rows = []
    for threshold in thresholds:
        selected = select_margin_gate(
            scores,
            mask,
            nearest_mask,
            group_indices,
            float(threshold),
            parseq_delta=data.get("parseq_delta"),
            parseq_safe=parseq_safe,
        )
        rows.append({"threshold": float(threshold), "metrics": policy_metrics(data, group_indices, selected)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--threshold-min", type=float, default=-0.2)
    parser.add_argument("--threshold-max", type=float, default=0.4)
    parser.add_argument("--threshold-steps", type=int, default=121)
    parser.add_argument("--parseq-safe", action="store_true")
    args = parser.parse_args()

    data = {key: value for key, value in np.load(args.features).items()}
    mask = data["mask"].astype(bool)
    nearest_mask = data["is_nearest"].astype(bool)
    val_groups = np.flatnonzero(data["group_keys"][:, 1] == args.val_seed)
    train_groups = np.flatnonzero(data["group_keys"][:, 1] != args.val_seed)
    nearest = data["is_nearest"][val_groups].argmax(axis=1)
    oracle = data["labels"][val_groups].argmax(axis=1)
    baseline = data["baseline_score"].astype(np.float64)
    scores_by_seed = load_reranker_scores(data, args.score)
    model_scores = {"model_mean": mean_scores(scores_by_seed)}
    for seed, scores in sorted(scores_by_seed.items()):
        model_scores[f"model_seed_{seed}"] = scores
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)

    policies = {
        "nearest": policy_metrics(data, val_groups, nearest),
        "oracle": policy_metrics(data, val_groups, oracle),
        "baseline_score_argmax": policy_metrics(data, val_groups, select_argmax(baseline, mask, val_groups)),
    }
    sweeps = {}
    train_sweeps = {}
    best = {}
    train_tuned = {}
    for name, scores in model_scores.items():
        argmax_metrics = policy_metrics(data, val_groups, select_argmax(scores, mask, val_groups))
        policies[f"{name}_argmax"] = argmax_metrics
        rows = sweep_thresholds(data, scores, mask, nearest_mask, val_groups, thresholds, args.parseq_safe)
        train_rows = sweep_thresholds(data, scores, mask, nearest_mask, train_groups, thresholds, args.parseq_safe)
        sweeps[name] = rows
        train_sweeps[name] = train_rows
        best[name] = best_policy(rows, require_parseq_safe=True)
        best_train = best_policy(train_rows, require_parseq_safe=True)
        threshold = float(best_train["threshold"])
        selected = select_margin_gate(
            scores,
            mask,
            nearest_mask,
            val_groups,
            threshold,
            parseq_delta=data.get("parseq_delta"),
            parseq_safe=args.parseq_safe,
        )
        train_tuned[name] = {
            "selected_train": best_train,
            "val_at_train_threshold": {"threshold": threshold, "metrics": policy_metrics(data, val_groups, selected)},
        }

    output = {
        "description": "No-op margin gate sweep for tensor-backed assignment reranker scores.",
        "features": str(args.features),
        "scores": [str(path) for path in args.score],
        "config": {
            "val_seed": args.val_seed,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_steps": args.threshold_steps,
            "parseq_safe": args.parseq_safe,
        },
        "policies": policies,
        "best_parseq_nonworse": best,
        "train_tuned_parseq_nonworse": train_tuned,
        "sweeps": sweeps,
        "train_sweeps": train_sweeps,
        "aggregate": {
            "policy_metrics": policies,
            "best_parseq_nonworse": best,
            "train_tuned_parseq_nonworse": train_tuned,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"policies": policies, "best_parseq_nonworse": best}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
