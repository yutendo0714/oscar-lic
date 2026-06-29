#!/usr/bin/env python3
"""Train sklearn tree rankers for nearest-plus-top4 assignment diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if int(labels.sum()) == 0:
        return None
    return float(average_precision_score(labels, scores))


def summarize_last_dims(array: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    flat = array.reshape(*array.shape[:2], -1).astype(np.float32)
    rms = np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True))
    mean = np.mean(flat, axis=-1, keepdims=True)
    std = np.std(flat, axis=-1, keepdims=True)
    amin = np.min(flat, axis=-1, keepdims=True)
    amax = np.max(flat, axis=-1, keepdims=True)
    out = np.concatenate([mean, std, rms, amin, amax], axis=-1)
    names = [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_rms", f"{prefix}_min", f"{prefix}_max"]
    return out, names


def build_features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    parts = [data["core_features"].astype(np.float32)]
    names = [f"core_{idx}" for idx in range(data["core_features"].shape[-1])]
    for key in ["code_center", "nearest_center", "delta_center", "latent_yhat", "latent_target", "latent_residual"]:
        if key in data:
            summary, summary_names = summarize_last_dims(data[key], key)
            parts.append(summary)
            names.extend(summary_names)
    if "latent_window_yhat" in data:
        for key in ["latent_window_yhat", "latent_window_target", "latent_window_residual"]:
            summary, summary_names = summarize_last_dims(data[key], key)
            parts.append(summary)
            names.extend(summary_names)
    code = data.get("code_center")
    nearest = data.get("nearest_center")
    delta = data.get("delta_center")
    if code is not None and nearest is not None and delta is not None:
        code_f = code.reshape(*code.shape[:2], -1).astype(np.float32)
        nearest_f = nearest.reshape(*nearest.shape[:2], -1).astype(np.float32)
        delta_f = delta.reshape(*delta.shape[:2], -1).astype(np.float32)
        dot = np.sum(code_f * nearest_f, axis=-1, keepdims=True)
        code_norm = np.sqrt(np.sum(code_f * code_f, axis=-1, keepdims=True)).clip(1.0e-6)
        nearest_norm = np.sqrt(np.sum(nearest_f * nearest_f, axis=-1, keepdims=True)).clip(1.0e-6)
        delta_norm = np.sqrt(np.sum(delta_f * delta_f, axis=-1, keepdims=True))
        cosine = dot / (code_norm * nearest_norm)
        parts.extend([dot, cosine, delta_norm / nearest_norm])
        names.extend(["code_nearest_dot", "code_nearest_cosine", "delta_norm_over_nearest_norm"])
    return np.concatenate(parts, axis=-1).astype(np.float32), names


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


def select_argmax(scores: np.ndarray, mask: np.ndarray, group_indices: np.ndarray, parseq_delta: np.ndarray) -> np.ndarray:
    selected = []
    for group_index in group_indices:
        row = scores[group_index].copy()
        row[~mask[group_index]] = -np.inf
        row[parseq_delta[group_index] > 0] = -np.inf
        if not np.isfinite(row).any():
            row = scores[group_index].copy()
            row[~mask[group_index]] = -np.inf
        selected.append(int(np.argmax(row)))
    return np.asarray(selected, dtype=np.int64)


def select_threshold(
    scores: np.ndarray,
    mask: np.ndarray,
    nearest: np.ndarray,
    group_indices: np.ndarray,
    threshold: float,
    parseq_delta: np.ndarray,
) -> np.ndarray:
    selected = []
    for group_index in group_indices:
        nearest_pos = int(nearest[group_index].argmax())
        valid = mask[group_index] & (~nearest[group_index]) & (parseq_delta[group_index] <= 0)
        if not valid.any():
            selected.append(nearest_pos)
            continue
        row = scores[group_index].copy()
        row[~valid] = -np.inf
        best = int(np.argmax(row))
        selected.append(best if float(row[best]) >= threshold else nearest_pos)
    return np.asarray(selected, dtype=np.int64)


def select_top_budget(
    scores: np.ndarray,
    mask: np.ndarray,
    nearest: np.ndarray,
    group_indices: np.ndarray,
    budget: int,
    parseq_delta: np.ndarray,
) -> np.ndarray:
    best_by_group = []
    for group_index in group_indices:
        nearest_pos = int(nearest[group_index].argmax())
        valid = mask[group_index] & (~nearest[group_index]) & (parseq_delta[group_index] <= 0)
        if not valid.any():
            continue
        row = scores[group_index].copy()
        row[~valid] = -np.inf
        best_pos = int(np.argmax(row))
        best_by_group.append((float(row[best_pos]), int(group_index), best_pos))
    keep = {group_index: pos for _, group_index, pos in sorted(best_by_group, reverse=True)[:budget]}
    selected = []
    for group_index in group_indices:
        nearest_pos = int(nearest[group_index].argmax())
        selected.append(keep.get(int(group_index), nearest_pos))
    return np.asarray(selected, dtype=np.int64)


def threshold_rows(data: dict[str, np.ndarray], scores: np.ndarray, groups: np.ndarray) -> list[dict]:
    rows = []
    for threshold in np.linspace(0, 1, 101):
        selected = select_threshold(
            scores,
            data["mask"].astype(bool),
            data["is_nearest"].astype(bool),
            groups,
            float(threshold),
            data["parseq_delta"],
        )
        rows.append({"threshold": float(threshold), "metrics": policy_metrics(data, groups, selected)})
    return rows


def budget_rows(data: dict[str, np.ndarray], scores: np.ndarray, groups: np.ndarray, budgets: list[int]) -> list[dict]:
    rows = []
    for budget in budgets:
        selected = select_top_budget(
            scores,
            data["mask"].astype(bool),
            data["is_nearest"].astype(bool),
            groups,
            int(budget),
            data["parseq_delta"],
        )
        rows.append({"budget": int(budget), "metrics": policy_metrics(data, groups, selected)})
    return rows


def choose_train_threshold(rows: list[dict], false_change_max: int) -> dict:
    candidates = [
        row
        for row in rows
        if row["metrics"]["parseq_delta_sum"] <= 0
        and row["metrics"]["false_change_groups"] <= false_change_max
        and row["metrics"]["wrong_change_groups"] == 0
    ]
    if not candidates:
        candidates = [row for row in rows if row["metrics"]["parseq_delta_sum"] <= 0]
    if not candidates:
        candidates = rows
    return min(
        candidates,
        key=lambda row: (
            row["metrics"]["tesseract_delta_sum"],
            row["metrics"]["false_change_groups"],
            row["metrics"]["wrong_change_groups"],
            -row["metrics"]["exact_oracle_groups"],
        ),
    )


def choose_train_budget(rows: list[dict], false_change_max: int) -> dict:
    candidates = [
        row
        for row in rows
        if row["metrics"]["parseq_delta_sum"] <= 0
        and row["metrics"]["false_change_groups"] <= false_change_max
        and row["metrics"]["wrong_change_groups"] == 0
    ]
    if not candidates:
        candidates = [row for row in rows if row["metrics"]["parseq_delta_sum"] <= 0]
    if not candidates:
        candidates = rows
    return min(
        candidates,
        key=lambda row: (
            row["metrics"]["tesseract_delta_sum"],
            row["metrics"]["false_change_groups"],
            row["metrics"]["wrong_change_groups"],
            -row["metrics"]["exact_oracle_groups"],
        ),
    )


def make_model(kind: str, seed: int, args: argparse.Namespace):
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=args.class_weight,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=args.class_weight,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
    if kind == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            random_state=seed,
        )
    raise ValueError(kind)


def train_one(data: dict[str, np.ndarray], features: np.ndarray, args: argparse.Namespace, seed: int) -> dict:
    group_keys = data["group_keys"]
    val_key_value = args.val_seed if args.val_key_value is None else args.val_key_value
    train_groups = np.flatnonzero(group_keys[:, args.split_key_index] != val_key_value)
    val_groups = np.flatnonzero(group_keys[:, args.split_key_index] == val_key_value)
    mask = data["mask"].astype(bool)
    train_rows = mask[train_groups]
    val_rows = mask[val_groups]
    x_train = features[train_groups][train_rows]
    y_train = data["labels"][train_groups][train_rows].astype(np.int64)
    x_val = features[val_groups][val_rows]
    y_val = data["labels"][val_groups][val_rows].astype(np.int64)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    model = make_model(args.model, seed, args)
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        train_score_flat = model.predict_proba(x_train)[:, 1]
        val_score_flat = model.predict_proba(x_val)[:, 1]
    else:
        train_score_flat = model.decision_function(x_train)
        val_score_flat = model.decision_function(x_val)
    scores = np.full(mask.shape, np.nan, dtype=np.float64)
    scores[train_groups][train_rows] = train_score_flat
    scores[val_groups][val_rows] = val_score_flat
    # Advanced indexing above writes to a copy, so fill explicitly.
    for rows, flat_scores, groups in [(train_rows, train_score_flat, train_groups), (val_rows, val_score_flat, val_groups)]:
        offset = 0
        for group_index, row_mask in zip(groups, rows):
            count = int(row_mask.sum())
            scores[group_index, row_mask] = flat_scores[offset : offset + count]
            offset += count

    false_change_max = max(0, int(round(len(train_groups) * args.train_false_change_rate_max)))
    train_thresholds = threshold_rows(data, scores, train_groups)
    threshold = choose_train_threshold(train_thresholds, false_change_max)
    train_budgets = budget_rows(data, scores, train_groups, args.changed_budget)
    budget = choose_train_budget(train_budgets, false_change_max)
    selected_argmax = select_argmax(scores, mask, val_groups, data["parseq_delta"])
    selected_default = select_threshold(
        scores, mask, data["is_nearest"].astype(bool), val_groups, args.default_threshold, data["parseq_delta"]
    )
    selected_train_tuned = select_threshold(
        scores, mask, data["is_nearest"].astype(bool), val_groups, threshold["threshold"], data["parseq_delta"]
    )
    selected_budget_tuned = select_top_budget(
        scores,
        mask,
        data["is_nearest"].astype(bool),
        val_groups,
        int(budget["budget"]),
        data["parseq_delta"],
    )
    selected_nearest = data["is_nearest"][val_groups].argmax(axis=1)
    selected_oracle = data["labels"][val_groups].argmax(axis=1)
    return {
        "model_seed": seed,
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "split_key_index": int(args.split_key_index),
        "val_key_value": int(val_key_value),
        "train_auc": safe_auc(y_train, train_score_flat),
        "train_ap": safe_ap(y_train, train_score_flat),
        "val_auc": safe_auc(y_val, val_score_flat),
        "val_ap": safe_ap(y_val, val_score_flat),
        "selected_train_threshold": threshold,
        "selected_train_budget": budget,
        "policy_metrics": {
            "nearest": policy_metrics(data, val_groups, selected_nearest),
            "oracle": policy_metrics(data, val_groups, selected_oracle),
            "row_argmax": policy_metrics(data, val_groups, selected_argmax),
            f"threshold_{args.default_threshold:.3f}": policy_metrics(data, val_groups, selected_default),
            "train_tuned_threshold": policy_metrics(data, val_groups, selected_train_tuned),
            "train_tuned_budget": policy_metrics(data, val_groups, selected_budget_tuned),
        },
    }


def aggregate(results: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for metric in ["train_auc", "train_ap", "val_auc", "val_ap"]:
        values = [row[metric] for row in results if row[metric] is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            output[metric] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "count": int(arr.size)}
    policy_names = sorted({name for row in results for name in row["policy_metrics"]})
    output["policy_metrics"] = {}
    for policy in policy_names:
        output["policy_metrics"][policy] = {}
        metrics = sorted({name for row in results for name in row["policy_metrics"].get(policy, {})})
        for metric in metrics:
            values = [float(row["policy_metrics"][policy][metric]) for row in results if policy in row["policy_metrics"]]
            arr = np.asarray(values, dtype=np.float64)
            output["policy_metrics"][policy][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["extra_trees", "random_forest", "hist_gbdt"], default="extra_trees")
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--split-key-index", type=int, default=1)
    parser.add_argument("--val-key-value", type=int)
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--changed-budget", type=int, action="append")
    parser.add_argument("--train-false-change-rate-max", type=float, default=0.02)
    args = parser.parse_args()
    if args.model_seed is None:
        args.model_seed = [0]
    if args.changed_budget is None:
        args.changed_budget = [1, 2, 3, 5, 8, 10, 15, 20]
    with np.load(args.features) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    features, feature_names = build_features(data)
    results = [train_one(data, features, args, seed) for seed in args.model_seed]
    output = {
        "description": "Sklearn tree ranker for assignment diagnostics.",
        "features": str(args.features),
        "feature_names": feature_names,
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "per_run": results,
        "aggregate": aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
