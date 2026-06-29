#!/usr/bin/env python3
"""Audit top-8 assignment feature neighborhoods.

This diagnostic asks whether useful assignment choices are locally supported by
training examples in deployable-ish feature spaces. It uses measured OCR deltas
only as labels/evaluation, so it is not a deployable selector by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
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
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return None
    order = np.argsort(-scores)
    hits = 0
    total = 0.0
    for rank, label in enumerate(labels[order], start=1):
        if label:
            hits += 1
            total += hits / rank
    return float(total / pos)


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    seed = int(row["seed"])
    return (
        int(row.get("real_seed", row.get("original_seed", seed))),
        seed,
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def safe_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def numeric_keys(rows: list[dict[str, Any]], predicate) -> list[str]:
    keys: set[str] = set()
    for row in rows[: min(len(rows), 32)]:
        for key, value in row.items():
            if not predicate(key):
                continue
            if isinstance(value, (int, float)) or value is None:
                keys.add(key)
    return sorted(keys)


def stats_vec(vec: np.ndarray) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(arr.mean()),
        float(arr.std()),
        float(np.mean(np.abs(arr))),
        float(np.linalg.norm(arr) / np.sqrt(arr.size)),
        float(np.max(np.abs(arr))),
    ]


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def normalized_l2(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(av - bv) / np.sqrt(max(av.size, 1)))


def find_npz_slot(z: dict[str, np.ndarray], group_index: int, row: dict[str, Any]) -> int:
    code = int(row["code_index"])
    rank = int(row["topk_rank"])
    mask = np.asarray(z["mask"][group_index], dtype=bool)
    codes = np.asarray(z["code_indices"][group_index], dtype=np.int64)
    ranks = np.asarray(z["topk_ranks"][group_index], dtype=np.int64)
    matches = np.flatnonzero(mask & (codes == code) & (ranks == rank))
    if len(matches):
        return int(matches[0])
    matches = np.flatnonzero(mask & (codes == code))
    if len(matches):
        return int(matches[0])
    raise KeyError(f"no npz slot for group={group_key(row)} code={code} rank={rank}")


def build_latent_summary(rows: list[dict[str, Any]], artifact: Path) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.load(artifact)
    needed = [
        "code_center",
        "code_indices",
        "core_features",
        "delta_center",
        "group_keys",
        "is_nearest",
        "latent_residual",
        "latent_target",
        "latent_window_residual",
        "latent_window_target",
        "latent_window_yhat",
        "latent_yhat",
        "mask",
        "nearest_center",
        "nearest_indices",
        "topk_ranks",
    ]
    z = {key: np.asarray(raw[key]) for key in needed}
    group_to_index = {tuple(int(v) for v in group): idx for idx, group in enumerate(z["group_keys"])}
    features: list[list[float]] = []
    missing = 0
    for row in rows:
        key = group_key(row)
        group_index = group_to_index.get(key)
        if group_index is None:
            features.append([0.0] * 62)
            missing += 1
            continue
        slot = find_npz_slot(z, group_index, row)
        code = z["code_center"][group_index, slot].astype(np.float32)
        nearest = z["nearest_center"][group_index, slot].astype(np.float32)
        delta = z["delta_center"][group_index, slot].astype(np.float32)
        yhat = z["latent_yhat"][group_index, slot].astype(np.float32)
        target = z["latent_target"][group_index, slot].astype(np.float32)
        residual = z["latent_residual"][group_index, slot].astype(np.float32)
        wyhat = z["latent_window_yhat"][group_index, slot].astype(np.float32)
        wtarget = z["latent_window_target"][group_index, slot].astype(np.float32)
        wresidual = z["latent_window_residual"][group_index, slot].astype(np.float32)
        row_features: list[float] = []
        for vec in [code, nearest, delta, yhat, target, residual, target - yhat]:
            row_features.extend(stats_vec(vec))
        row_features.extend(
            [
                safe_cosine(delta, residual),
                safe_cosine(code, target),
                safe_cosine(nearest, yhat),
                safe_cosine(code - nearest, target - yhat),
                normalized_l2(delta, residual),
                normalized_l2(code, target),
                normalized_l2(nearest, yhat),
            ]
        )
        for tensor in [wyhat, wtarget, wresidual, wtarget - wyhat]:
            row_features.extend(stats_vec(tensor))
        features.append(row_features)
    out = np.asarray(features, dtype=np.float32)
    meta = {
        "path": str(artifact),
        "sha256": sha256_file(artifact),
        "rows": int(out.shape[0]),
        "feature_dim": int(out.shape[1]),
        "missing_rows": int(missing),
    }
    return out, meta


def table_features(rows: list[dict[str, Any]], keys: list[str]) -> np.ndarray:
    return np.asarray([[safe_float(row, key) for key in keys] for row in rows], dtype=np.float32)


def sanitize_and_standardize(x: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    out[~np.isfinite(out)] = np.nan
    train = out[train_indices]
    med = np.nanmedian(train, axis=0)
    med[~np.isfinite(med)] = 0.0
    inds = np.where(~np.isfinite(out))
    out[inds] = np.take(med, inds[1])
    mean = out[train_indices].mean(axis=0)
    std = out[train_indices].std(axis=0)
    std[std < 1.0e-6] = 1.0
    return ((out - mean) / std).astype(np.float32)


def knn_score(
    queries: np.ndarray,
    train: np.ndarray,
    train_labels: np.ndarray,
    *,
    k: int,
    train_group_ids: np.ndarray | None = None,
    query_group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(queries.shape[0], dtype=np.float32)
    mean_dists = np.zeros(queries.shape[0], dtype=np.float32)
    train_sq = np.sum(train * train, axis=1, keepdims=True).T
    for start in range(0, queries.shape[0], 512):
        end = min(start + 512, queries.shape[0])
        q = queries[start:end]
        dists = np.sum(q * q, axis=1, keepdims=True) + train_sq - 2.0 * (q @ train.T)
        dists = np.maximum(dists, 0.0)
        if train_group_ids is not None and query_group_ids is not None:
            dists[train_group_ids[None, :] == query_group_ids[start:end, None]] = np.inf
        kk = min(k, train.shape[0])
        order = np.argpartition(dists, kth=kk - 1, axis=1)[:, :kk]
        row = np.arange(end - start)[:, None]
        local_dists = dists[row, order]
        local_labels = train_labels[order]
        finite = np.isfinite(local_dists)
        denom = finite.sum(axis=1).clip(min=1)
        scores[start:end] = (local_labels * finite).sum(axis=1) / denom
        mean_dists[start:end] = np.where(finite.any(axis=1), np.nanmean(np.where(finite, local_dists, np.nan), axis=1), np.inf)
    return scores, mean_dists


def error_type(selected_changed: bool, selected_is_oracle: bool, oracle_changed: bool) -> str:
    if selected_is_oracle and oracle_changed:
        return "exact_oracle"
    if selected_is_oracle and not oracle_changed:
        return "correct_nearest"
    if selected_changed and not oracle_changed:
        return "false_change"
    if (not selected_changed) and oracle_changed:
        return "missed_oracle"
    return "wrong_change"


def policy_metrics(rows: list[dict[str, Any]], group_indices: list[list[int]], selected_rows: dict[int, int]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    parseq_sum = 0
    tess_sum = 0
    parseq_worsen = 0
    tess_improve = 0
    tess_worsen = 0
    changed = 0
    oracle_changed_total = 0
    for group_id, indices in enumerate(group_indices):
        selected = selected_rows[group_id]
        nearest = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
        oracle = next(idx for idx in indices if int(rows[idx]["label_assignment_oracle_choice"]))
        selected_changed = selected != nearest
        oracle_changed = oracle != nearest
        selected_is_oracle = selected == oracle
        pdelta = int(rows[selected]["parseq_delta_vs_nearest"])
        tdelta = int(rows[selected]["tesseract_delta_vs_nearest"])
        parseq_sum += pdelta
        tess_sum += tdelta
        parseq_worsen += int(pdelta > 0)
        tess_improve += int(tdelta < 0)
        tess_worsen += int(tdelta > 0)
        changed += int(selected_changed)
        oracle_changed_total += int(oracle_changed)
        counts[error_type(selected_changed, selected_is_oracle, oracle_changed)] += 1
    return {
        "groups": int(len(group_indices)),
        "changed_groups": int(changed),
        "parseq_delta_sum": int(parseq_sum),
        "tesseract_delta_sum": int(tess_sum),
        "parseq_worsen_groups": int(parseq_worsen),
        "tesseract_improve_groups": int(tess_improve),
        "tesseract_worsen_groups": int(tess_worsen),
        "oracle_change_groups": int(oracle_changed_total),
        "exact_oracle_groups": int(counts["exact_oracle"]),
        "missed_oracle_groups": int(counts["missed_oracle"]),
        "false_change_groups": int(counts["false_change"]),
        "wrong_change_groups": int(counts["wrong_change"]),
        "correct_nearest_groups": int(counts["correct_nearest"]),
    }


def select_argmax_scores(group_indices: list[list[int]], scores: np.ndarray) -> dict[int, int]:
    selected: dict[int, int] = {}
    for group_id, indices in enumerate(group_indices):
        best = max(indices, key=lambda idx: (float(scores[idx]), -int(idx)))
        selected[group_id] = int(best)
    return selected


def select_safe_threshold(
    rows: list[dict[str, Any]],
    group_indices: list[list[int]],
    scores: np.ndarray,
    threshold: float,
) -> dict[int, int]:
    selected: dict[int, int] = {}
    for group_id, indices in enumerate(group_indices):
        nearest = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
        candidates = [idx for idx in indices if not int(rows[idx]["is_nearest"])]
        if not candidates:
            selected[group_id] = nearest
            continue
        best = max(candidates, key=lambda idx: (float(scores[idx]), -int(idx)))
        selected[group_id] = int(best if float(scores[best]) >= threshold else nearest)
    return selected


def select_margin_threshold(
    rows: list[dict[str, Any]],
    group_indices: list[list[int]],
    scores: np.ndarray,
    threshold: float,
) -> dict[int, int]:
    selected: dict[int, int] = {}
    for group_id, indices in enumerate(group_indices):
        nearest = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
        best = max(indices, key=lambda idx: (float(scores[idx]), -int(idx)))
        margin = float(scores[best] - scores[nearest])
        selected[group_id] = int(best if best != nearest and margin >= threshold else nearest)
    return selected


def choose_threshold(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in candidates if row["metrics"]["parseq_delta_sum"] <= 0]
    strict = [row for row in valid if row["metrics"]["false_change_groups"] <= max(1, int(round(row["metrics"]["groups"] * 0.02)))]
    if strict:
        valid = strict
    strict = [row for row in valid if row["metrics"]["wrong_change_groups"] <= 0]
    if strict:
        valid = strict
    strict = [row for row in valid if row["metrics"]["tesseract_worsen_groups"] <= max(1, int(round(row["metrics"]["groups"] * 0.01)))]
    if strict:
        valid = strict
    if not valid:
        valid = candidates
    return min(
        valid,
        key=lambda row: (
            row["metrics"]["tesseract_delta_sum"],
            row["metrics"]["parseq_delta_sum"],
            row["metrics"]["false_change_groups"],
            row["metrics"]["wrong_change_groups"],
            -row["metrics"]["exact_oracle_groups"],
        ),
    )


def threshold_candidates(values: np.ndarray, mode: str) -> np.ndarray:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.asarray([0.0], dtype=np.float32)
    if mode == "margin":
        grid = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, 81)))
        return np.asarray(sorted(set([0.0, *[float(v) for v in grid]])), dtype=np.float32)
    grid = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, 81)))
    return np.asarray(sorted(set([0.0, 0.25, 0.5, 0.75, 1.0, *[float(v) for v in grid]])), dtype=np.float32)


def rank_support(rows: list[dict[str, Any]], group_indices: list[list[int]], scores: np.ndarray) -> dict[str, Any]:
    changed_groups = 0
    top_counts = {1: 0, 2: 0, 4: 0, 8: 0}
    ranks: list[int] = []
    false_pressure: list[float] = []
    change_pressure: list[float] = []
    for indices in group_indices:
        nearest = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
        oracle = next(idx for idx in indices if int(rows[idx]["label_assignment_oracle_choice"]))
        non_nearest = [idx for idx in indices if idx != nearest]
        max_non_nearest = max((float(scores[idx]) for idx in non_nearest), default=float("-inf"))
        if oracle == nearest:
            false_pressure.append(max_non_nearest)
            continue
        changed_groups += 1
        change_pressure.append(max_non_nearest)
        ordered = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)
        rank = ordered.index(oracle) + 1
        ranks.append(rank)
        for k in top_counts:
            top_counts[k] += int(rank <= k)
    return {
        "oracle_changed_groups": int(changed_groups),
        "oracle_score_top1_recall": float(top_counts[1] / changed_groups) if changed_groups else None,
        "oracle_score_top2_recall": float(top_counts[2] / changed_groups) if changed_groups else None,
        "oracle_score_top4_recall": float(top_counts[4] / changed_groups) if changed_groups else None,
        "oracle_score_top8_recall": float(top_counts[8] / changed_groups) if changed_groups else None,
        "oracle_rank_mean": float(np.mean(ranks)) if ranks else None,
        "oracle_rank_median": float(np.median(ranks)) if ranks else None,
        "no_headroom_max_nonnearest_score_mean": float(np.mean(false_pressure)) if false_pressure else None,
        "oracle_change_max_nonnearest_score_mean": float(np.mean(change_pressure)) if change_pressure else None,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"policy_metrics": {}, "row_metrics": {}, "support": {}}
    policies = sorted({policy for row in rows for policy in row["policies"]})
    for policy in policies:
        output["policy_metrics"][policy] = {}
        metric_names = sorted({key for row in rows for key in row["policies"][policy]})
        for metric in metric_names:
            values = [float(row["policies"][policy][metric]) for row in rows if row["policies"][policy].get(metric) is not None]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                output["policy_metrics"][policy][metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
    for metric in ["oracle_auc", "oracle_ap", "safe_auc", "safe_ap"]:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            output["row_metrics"][metric] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    support_keys = sorted({key for row in rows for key in row["support"] if row["support"][key] is not None})
    for key in support_keys:
        values = [float(row["support"][key]) for row in rows if row["support"].get(key) is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            output["support"][key] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    return output


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Eval300 Top8 Feature Neighbor Audit",
        "",
        "This is a diagnostic, not a deployable selector. Tesseract deltas are used only to audit neighborhood support and policy failure modes.",
        "",
        "## Key Aggregates",
        "",
    ]
    for family in result["families"]:
        for k, agg in family["aggregate_by_k"].items():
            safe = agg["policy_metrics"].get("safe_threshold_train_tuned", {})
            oracle = agg["policy_metrics"].get("oracle_margin_train_tuned", {})
            argmax = agg["policy_metrics"].get("oracle_argmax", {})
            support = agg.get("support", {})
            lines.append(
                f"- `{family['name']}` k={k}: "
                f"safe_tuned dT={safe.get('tesseract_delta_sum', {}).get('mean')}, "
                f"false={safe.get('false_change_groups', {}).get('mean')}, "
                f"exact={safe.get('exact_oracle_groups', {}).get('mean')}; "
                f"oracle_margin dT={oracle.get('tesseract_delta_sum', {}).get('mean')}, "
                f"argmax dT={argmax.get('tesseract_delta_sum', {}).get('mean')}, "
                f"oracle_top4_recall={support.get('oracle_score_top4_recall', {}).get('mean')}"
            )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(result.get("interpretation", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--latent-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--k", type=int, action="append", default=[5, 15, 31])
    parser.add_argument("--val-seed", type=int, action="append", default=[0, 1, 2])
    args = parser.parse_args()

    rows = read_jsonl(args.table)
    keys = [group_key(row) for row in rows]
    group_to_rows: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        group_to_rows[key].append(idx)
    group_items = sorted(group_to_rows.items())
    group_id_by_key = {key: idx for idx, (key, _) in enumerate(group_items)}
    row_group_ids = np.asarray([group_id_by_key[key] for key in keys], dtype=np.int64)

    code_keys = numeric_keys(rows, lambda key: key.startswith("codebook_") and "onehot" not in key)
    onehot_keys = numeric_keys(rows, lambda key: key.startswith("codebook_") and "onehot" in key)
    imgdiff_keys = numeric_keys(rows, lambda key: key.startswith("img_") and not key.startswith("img_teacher_parseq_"))
    parseq_teacher_keys = numeric_keys(rows, lambda key: key.startswith("img_teacher_parseq_"))
    rank_keys = ["topk_rank", "is_nearest", "assignment_relative_error", "reference_length"]
    latent_features, latent_meta = build_latent_summary(rows, args.latent_artifact)

    blocks = {
        "rank_relerr": table_features(rows, rank_keys),
        "codegeom_scalar": table_features(rows, code_keys),
        "codegeom_onehot": table_features(rows, code_keys + onehot_keys),
        "imgdiff_scalar": table_features(rows, imgdiff_keys),
        "latent_summary": latent_features,
        "code_img_latent": np.concatenate(
            [table_features(rows, code_keys), table_features(rows, imgdiff_keys), latent_features], axis=1
        ),
        "code_img_latent_parseqteacher": np.concatenate(
            [
                table_features(rows, code_keys),
                table_features(rows, imgdiff_keys),
                latent_features,
                table_features(rows, parseq_teacher_keys),
            ],
            axis=1,
        ),
    }

    oracle_labels = np.asarray([int(row["label_assignment_oracle_choice"]) for row in rows], dtype=np.float32)
    safe_labels = np.asarray([int(row["label_tesseract_parseq_safe_improves"]) for row in rows], dtype=np.float32)
    seed_by_row = np.asarray([int(row["seed"]) for row in rows], dtype=np.int64)

    result = {
        "description": "Top8 assignment feature-neighborhood audit. Diagnostic only; Tesseract labels are not deployable training signals.",
        "inputs": {
            "table": {"path": str(args.table), "sha256": sha256_file(args.table), "rows": len(rows)},
            "latent_artifact": latent_meta,
        },
        "feature_keys": {
            "rank_relerr": rank_keys,
            "codegeom_scalar_count": len(code_keys),
            "codegeom_onehot_count": len(onehot_keys),
            "imgdiff_scalar_count": len(imgdiff_keys),
            "parseq_teacher_count": len(parseq_teacher_keys),
        },
        "families": [],
        "interpretation": [],
    }

    for family_name, features in blocks.items():
        family_runs: list[dict[str, Any]] = []
        by_k: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for val_seed in args.val_seed:
            train_rows = np.flatnonzero(seed_by_row != val_seed)
            val_rows = np.flatnonzero(seed_by_row == val_seed)
            train_group_ids = row_group_ids[train_rows]
            val_group_ids = row_group_ids[val_rows]
            x = sanitize_and_standardize(features, train_rows)
            train_group_indices = [indices for key, indices in group_items if int(key[1]) != val_seed]
            val_group_indices = [indices for key, indices in group_items if int(key[1]) == val_seed]
            for k in args.k:
                train_oracle_scores, _ = knn_score(
                    x[train_rows],
                    x[train_rows],
                    oracle_labels[train_rows],
                    k=k,
                    train_group_ids=train_group_ids,
                    query_group_ids=train_group_ids,
                )
                val_oracle_scores, _ = knn_score(x[val_rows], x[train_rows], oracle_labels[train_rows], k=k)
                train_safe_scores, _ = knn_score(
                    x[train_rows],
                    x[train_rows],
                    safe_labels[train_rows],
                    k=k,
                    train_group_ids=train_group_ids,
                    query_group_ids=train_group_ids,
                )
                val_safe_scores, _ = knn_score(x[val_rows], x[train_rows], safe_labels[train_rows], k=k)

                oracle_scores = np.zeros(len(rows), dtype=np.float32)
                oracle_scores[train_rows] = train_oracle_scores
                oracle_scores[val_rows] = val_oracle_scores
                safe_scores = np.zeros(len(rows), dtype=np.float32)
                safe_scores[train_rows] = train_safe_scores
                safe_scores[val_rows] = val_safe_scores

                train_argmax = select_argmax_scores(train_group_indices, oracle_scores)
                val_argmax = select_argmax_scores(val_group_indices, oracle_scores)

                margins = []
                safe_values = []
                for group_id, indices in enumerate(train_group_indices):
                    nearest = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
                    best = max(indices, key=lambda idx: float(oracle_scores[idx]))
                    margins.append(float(oracle_scores[best] - oracle_scores[nearest]))
                    non_nearest = [idx for idx in indices if idx != nearest]
                    safe_values.append(max((float(safe_scores[idx]) for idx in non_nearest), default=0.0))
                margin_rows = []
                for threshold in threshold_candidates(np.asarray(margins, dtype=np.float32), mode="margin"):
                    selected = select_margin_threshold(rows, train_group_indices, oracle_scores, float(threshold))
                    margin_rows.append({"threshold": float(threshold), "metrics": policy_metrics(rows, train_group_indices, selected)})
                margin_choice = choose_threshold(margin_rows)
                safe_rows = []
                for threshold in threshold_candidates(np.asarray(safe_values, dtype=np.float32), mode="safe"):
                    selected = select_safe_threshold(rows, train_group_indices, safe_scores, float(threshold))
                    safe_rows.append({"threshold": float(threshold), "metrics": policy_metrics(rows, train_group_indices, selected)})
                safe_choice = choose_threshold(safe_rows)
                val_margin = select_margin_threshold(rows, val_group_indices, oracle_scores, float(margin_choice["threshold"]))
                val_safe = select_safe_threshold(rows, val_group_indices, safe_scores, float(safe_choice["threshold"]))

                val_oracle_labels = oracle_labels[val_rows]
                val_safe_labels = safe_labels[val_rows]
                run = {
                    "feature_family": family_name,
                    "feature_dim": int(features.shape[1]),
                    "val_seed": int(val_seed),
                    "k": int(k),
                    "train_groups": int(len(train_group_indices)),
                    "val_groups": int(len(val_group_indices)),
                    "oracle_auc": roc_auc(val_oracle_scores, val_oracle_labels),
                    "oracle_ap": average_precision(val_oracle_scores, val_oracle_labels),
                    "safe_auc": roc_auc(val_safe_scores, val_safe_labels),
                    "safe_ap": average_precision(val_safe_scores, val_safe_labels),
                    "selected_margin_threshold": margin_choice,
                    "selected_safe_threshold": safe_choice,
                    "support": rank_support(rows, val_group_indices, oracle_scores),
                    "policies": {
                        "oracle_argmax": policy_metrics(rows, val_group_indices, val_argmax),
                        "oracle_margin_train_tuned": policy_metrics(rows, val_group_indices, val_margin),
                        "safe_threshold_train_tuned": policy_metrics(rows, val_group_indices, val_safe),
                        "nearest": policy_metrics(
                            rows,
                            val_group_indices,
                            {
                                group_id: next(idx for idx in indices if int(rows[idx]["is_nearest"]))
                                for group_id, indices in enumerate(val_group_indices)
                            },
                        ),
                        "oracle": policy_metrics(
                            rows,
                            val_group_indices,
                            {
                                group_id: next(idx for idx in indices if int(rows[idx]["label_assignment_oracle_choice"]))
                                for group_id, indices in enumerate(val_group_indices)
                            },
                        ),
                    },
                }
                run["train_oracle_argmax"] = policy_metrics(rows, train_group_indices, train_argmax)
                family_runs.append(run)
                by_k[str(k)].append(run)
        family_result = {
            "name": family_name,
            "feature_dim": int(features.shape[1]),
            "runs": family_runs,
            "aggregate_by_k": {k: aggregate(k_runs) for k, k_runs in by_k.items()},
        }
        result["families"].append(family_result)

    best_rows: list[tuple[float, str, str, dict[str, Any]]] = []
    for family in result["families"]:
        for k, agg in family["aggregate_by_k"].items():
            metrics = agg["policy_metrics"].get("safe_threshold_train_tuned", {})
            value = metrics.get("tesseract_delta_sum", {}).get("mean")
            if value is not None:
                best_rows.append((float(value), family["name"], k, metrics))
    best_rows.sort(key=lambda row: row[0])
    if best_rows:
        best = best_rows[0]
        result["interpretation"].append(
            f"Best train-tuned safe-threshold neighborhood policy is {best[1]} k={best[2]} with mean "
            f"Tesseract delta {best[0]:.3f} over the three seed-held-out top8 diagnostic splits. "
            "This is not the same population as the actual-bitstream 225-row current-best/shortlist evaluation, "
            "where the learned baseline is -8 total Tesseract edits and the shortlist oracle is -20."
        )
        false_mean = best[3].get("false_change_groups", {}).get("mean")
        exact_mean = best[3].get("exact_oracle_groups", {}).get("mean")
        result["interpretation"].append(
            f"Best neighborhood policy exact/false means are {exact_mean}/{false_mean}; if false changes remain high, "
            "feature-neighborhood support is insufficient for promotion without a different verifier or more calibration data."
        )
    result["interpretation"].append(
        "All results that use Tesseract-derived labels are diagnostic only and must not be promoted as deployable OCR-invariant training."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    print(json.dumps({"families": len(result["families"]), "output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
