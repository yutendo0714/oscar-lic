#!/usr/bin/env python3
"""Train a lightweight verifier over assignment shortlist score features.

This diagnostic intentionally uses only deployable first-stage score features
at inference time. OCR deltas are used for labels, train-threshold selection and
evaluation, but never as candidate features.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


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


def load_score_groups(paths: list[Path]) -> dict[tuple[int, int, int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for file_index, path in enumerate(paths):
        for row in read_jsonl(path):
            out = dict(row)
            out["_voter_id"] = f"{file_index}:{int(row['model_seed'])}"
            groups[group_key(out)].append(out)
    return dict(groups)


def _stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def build_group(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_code: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_code[int(row["code_index"])].append(row)
    voter_ids = sorted({str(row["_voter_id"]) for row in raw_rows})
    nearest_code = next(int(row["code_index"]) for row in raw_rows if int(row["is_nearest"]))
    oracle_code = next(int(row["code_index"]) for row in raw_rows if int(row["label_assignment_oracle_choice"]))
    nearest = by_code[nearest_code][0]
    oracle = by_code[oracle_code][0]
    nearest_scores = {
        str(row["_voter_id"]): float(row["score"])
        for row in by_code[nearest_code]
    }

    top_vote_counts: Counter[int] = Counter()
    for voter_id in voter_ids:
        voter_rows = [
            row
            for code, rows in by_code.items()
            if code != nearest_code
            for row in rows
            if str(row["_voter_id"]) == voter_id
        ]
        if voter_rows:
            best = max(voter_rows, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))
            top_vote_counts[int(best["code_index"])] += 1

    candidate_infos: list[dict[str, Any]] = []
    for code, rows in by_code.items():
        if code == nearest_code:
            continue
        template = rows[0]
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        group_probs = np.asarray([float(row.get("group_change_prob", 1.0)) for row in rows], dtype=np.float64)
        margins = np.asarray(
            [float(row["score"]) - nearest_scores.get(str(row["_voter_id"]), 0.0) for row in rows],
            dtype=np.float64,
        )
        candidate_infos.append(
            {
                "code": int(code),
                "row": template,
                "score": _stats(scores),
                "margin": _stats(margins),
                "group_prob": _stats(group_probs),
                "vote_fraction": float(top_vote_counts[int(code)] / max(len(voter_ids), 1)),
                "vote_count": int(top_vote_counts[int(code)]),
                "topk_rank": int(template["topk_rank"]),
                "label": int(template["label_assignment_oracle_choice"]),
                "parseq_delta": int(template["parseq_delta_vs_nearest"]),
                "tesseract_delta": int(template["tesseract_delta_vs_nearest"]),
            }
        )

    score_means = np.asarray([info["score"]["mean"] for info in candidate_infos], dtype=np.float64)
    margin_means = np.asarray([info["margin"]["mean"] for info in candidate_infos], dtype=np.float64)
    if candidate_infos:
        order = np.argsort(-score_means, kind="mergesort")
        ranks = np.empty(len(candidate_infos), dtype=np.int64)
        ranks[order] = np.arange(1, len(candidate_infos) + 1)
        sorted_scores = score_means[order]
        best_score = float(sorted_scores[0])
        second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else best_score
        for index, info in enumerate(candidate_infos):
            info["score_rank"] = int(ranks[index])
            info["score_gap_to_best"] = float(info["score"]["mean"] - best_score)
            info["score_gap_to_second"] = (
                float(best_score - second_score) if ranks[index] == 1 else float(info["score"]["mean"] - best_score)
            )
            info["score_centered"] = float(info["score"]["mean"] - score_means.mean())
            info["margin_centered"] = float(info["margin"]["mean"] - margin_means.mean())

    return {
        "key": group_key(nearest),
        "nearest": nearest,
        "oracle": oracle,
        "candidates": candidate_infos,
        "voter_count": len(voter_ids),
    }


def build_candidate_feature(info: dict[str, Any], voter_count: int, top_m: int) -> np.ndarray:
    rank = int(info["topk_rank"])
    return np.asarray(
        [
            float(info["score"]["mean"]),
            float(info["score"]["min"]),
            float(info["score"]["max"]),
            float(info["score"]["std"]),
            float(info["margin"]["mean"]),
            float(info["margin"]["min"]),
            float(info["margin"]["max"]),
            float(info["margin"]["std"]),
            float(info["group_prob"]["mean"]),
            float(info["group_prob"]["min"]),
            float(info["group_prob"]["std"]),
            float(info["vote_fraction"]),
            float(info["vote_count"] / max(voter_count, 1)),
            float(rank / max(top_m, 1)),
            float(rank == 1),
            float(info.get("score_rank", 0) / max(top_m, 1)),
            float(info.get("score_gap_to_best", 0.0)),
            float(info.get("score_gap_to_second", 0.0)),
            float(info.get("score_centered", 0.0)),
            float(info.get("margin_centered", 0.0)),
        ],
        dtype=np.float32,
    )


def group_feature_from_candidates(candidate_features: np.ndarray, candidate_scores: np.ndarray) -> np.ndarray:
    if candidate_features.size == 0:
        return np.zeros((candidate_features.shape[1] if candidate_features.ndim == 2 else 20) * 2 + 6, dtype=np.float32)
    best_index = int(np.argmax(candidate_scores))
    best = candidate_features[best_index]
    mean = candidate_features.mean(axis=0)
    sorted_scores = np.sort(candidate_scores)[::-1]
    best_score = float(sorted_scores[0])
    second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else best_score
    extra = np.asarray(
        [
            best_score,
            float(second_score),
            float(best_score - second_score),
            float(candidate_scores.mean()),
            float(candidate_scores.std(ddof=0)),
            float(candidate_features.shape[0]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([best, mean, extra], axis=0).astype(np.float32)


def build_arrays(groups: list[dict[str, Any]], top_m: int) -> dict[str, Any]:
    candidate_features: list[np.ndarray] = []
    candidate_labels: list[int] = []
    candidate_group_indices: list[int] = []
    candidate_info_indices: list[int] = []
    group_features: list[np.ndarray] = []
    group_labels: list[int] = []
    per_group_candidates: list[list[int]] = []

    for group_index, group in enumerate(groups):
        local_indices = []
        for info_index, info in enumerate(group["candidates"]):
            if int(info["topk_rank"]) > top_m:
                continue
            local_indices.append(len(candidate_features))
            candidate_features.append(build_candidate_feature(info, int(group["voter_count"]), top_m=top_m))
            candidate_labels.append(int(info["label"]))
            candidate_group_indices.append(group_index)
            candidate_info_indices.append(info_index)
        per_group_candidates.append(local_indices)
        if local_indices:
            feats = np.stack([candidate_features[index] for index in local_indices], axis=0)
            labels = np.asarray([candidate_labels[index] for index in local_indices], dtype=np.int64)
            group_scores = feats[:, 0]
            group_features.append(group_feature_from_candidates(feats, group_scores))
            group_labels.append(int(labels.any()))
        else:
            group_features.append(group_feature_from_candidates(np.zeros((0, 20), dtype=np.float32), np.zeros(0)))
            group_labels.append(0)

    return {
        "candidate_features": np.stack(candidate_features, axis=0).astype(np.float32),
        "candidate_labels": np.asarray(candidate_labels, dtype=np.float32),
        "candidate_group_indices": np.asarray(candidate_group_indices, dtype=np.int64),
        "candidate_info_indices": np.asarray(candidate_info_indices, dtype=np.int64),
        "group_features": np.stack(group_features, axis=0).astype(np.float32),
        "group_labels": np.asarray(group_labels, dtype=np.float32),
        "per_group_candidates": per_group_candidates,
    }


def normalize(train_features: np.ndarray, all_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return ((all_features - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


class SmallVerifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = torch.nn.Linear(in_dim, 1)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def fit_binary_model(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    norm_features, mean, std = normalize(features[train_indices], features)
    x = torch.from_numpy(norm_features).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    train_t = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    positives = float(labels[train_indices].sum())
    negatives = float(len(train_indices) - positives)
    pos_weight = negatives / max(positives, 1.0)
    model = SmallVerifier(features.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = model(x[train_t])
        loss = loss_fn(logits, y[train_t])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).detach().cpu().numpy().astype(np.float64)
    return probs, {
        "positive_count": int(positives),
        "negative_count": int(negatives),
        "pos_weight": float(pos_weight),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
    }


def annotate_selected(group: dict[str, Any], selected: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    nearest_code = int(group["nearest"]["code_index"])
    oracle_code = int(group["oracle"]["code_index"])
    selected_code = int(selected["code_index"])
    selected_changed = selected_code != nearest_code
    oracle_changed = oracle_code != nearest_code
    out = dict(selected)
    out.update(policy)
    out.update(
        {
            "nearest_code": nearest_code,
            "oracle_code": oracle_code,
            "selected_changed": int(selected_changed),
            "oracle_changed": int(oracle_changed),
            "selected_is_oracle": int(selected_code == oracle_code),
        }
    )
    return out


def summarize(selected: list[dict[str, Any]]) -> dict[str, int]:
    errors = Counter(
        policy_error_type(
            bool(row["selected_changed"]),
            bool(row["selected_is_oracle"]),
            bool(row["oracle_changed"]),
        )
        for row in selected
    )
    return {
        "groups": len(selected),
        "changed_groups": sum(int(row["selected_changed"]) for row in selected),
        "parseq_delta_sum": sum(int(row["parseq_delta_vs_nearest"]) for row in selected),
        "tesseract_delta_sum": sum(int(row["tesseract_delta_vs_nearest"]) for row in selected),
        "parseq_worsen_groups": sum(int(row["parseq_delta_vs_nearest"] > 0) for row in selected),
        "tesseract_improve_groups": sum(int(row["tesseract_delta_vs_nearest"] < 0) for row in selected),
        "tesseract_worsen_groups": sum(int(row["tesseract_delta_vs_nearest"] > 0) for row in selected),
        "oracle_change_groups": sum(int(row["oracle_changed"]) for row in selected),
        "exact_oracle_groups": int(errors["exact_oracle"]),
        "missed_oracle_groups": int(errors["missed_oracle"]),
        "false_change_groups": int(errors["false_change"]),
        "wrong_change_groups": int(errors["wrong_change"]),
        "correct_nearest_groups": int(errors["correct_nearest"]),
    }


def select_oracle_upper(groups: list[dict[str, Any]], indices: np.ndarray, top_m: int) -> list[dict[str, Any]]:
    selected = []
    for group_index in indices:
        group = groups[int(group_index)]
        oracle_code = int(group["oracle"]["code_index"])
        oracle_in_top_m = any(int(info["code"]) == oracle_code and int(info["topk_rank"]) <= top_m for info in group["candidates"])
        row = group["oracle"] if oracle_in_top_m else group["nearest"]
        selected.append(annotate_selected(group, row, {"policy": f"top{top_m}_oracle_upper"}))
    return selected


def select_nearest_or_oracle(groups: list[dict[str, Any]], indices: np.ndarray, mode: str) -> list[dict[str, Any]]:
    selected = []
    for group_index in indices:
        group = groups[int(group_index)]
        row = group["nearest"] if mode == "nearest" else group["oracle"]
        selected.append(annotate_selected(group, row, {"policy": mode}))
    return selected


def candidate_policy_rows(
    groups: list[dict[str, Any]],
    arrays: dict[str, Any],
    group_indices: np.ndarray,
    candidate_probs: np.ndarray,
    group_probs: np.ndarray,
    *,
    mode: str,
    candidate_threshold: float,
    group_threshold: float,
) -> list[dict[str, Any]]:
    selected = []
    per_group = arrays["per_group_candidates"]
    for group_index in group_indices:
        group = groups[int(group_index)]
        local_indices = per_group[int(group_index)]
        if not local_indices:
            selected.append(annotate_selected(group, group["nearest"], {"policy": mode, "selected_by_verifier": 0}))
            continue
        local_probs = candidate_probs[local_indices]
        best_local_pos = int(np.argmax(local_probs))
        best_global_index = int(local_indices[best_local_pos])
        best_info_index = int(arrays["candidate_info_indices"][best_global_index])
        best_info = group["candidates"][best_info_index]
        cand_prob = float(candidate_probs[best_global_index])
        grp_prob = float(group_probs[int(group_index)])
        product = cand_prob * grp_prob
        min_prob = min(cand_prob, grp_prob)
        if mode == "candidate_threshold":
            accept = cand_prob >= candidate_threshold
            confidence = cand_prob
        elif mode == "group_threshold":
            accept = grp_prob >= group_threshold
            confidence = grp_prob
        elif mode == "product_threshold":
            accept = product >= candidate_threshold
            confidence = product
        elif mode == "min_threshold":
            accept = min_prob >= candidate_threshold
            confidence = min_prob
        elif mode == "two_threshold":
            accept = cand_prob >= candidate_threshold and grp_prob >= group_threshold
            confidence = min_prob
        else:
            raise ValueError(f"unknown policy mode: {mode}")
        row = best_info["row"] if accept else group["nearest"]
        selected.append(
            annotate_selected(
                group,
                row,
                {
                    "policy": mode,
                    "selected_by_verifier": int(accept),
                    "candidate_probability": cand_prob,
                    "group_probability": grp_prob,
                    "policy_confidence": confidence,
                    "candidate_threshold": float(candidate_threshold),
                    "group_threshold": float(group_threshold),
                },
            )
        )
    return selected


def feasible(metrics: dict[str, int], args: argparse.Namespace, group_count: int) -> bool:
    return (
        metrics["false_change_groups"] <= int(round(group_count * float(args.false_change_rate_max)))
        and metrics["wrong_change_groups"] <= int(round(group_count * float(args.wrong_change_rate_max)))
        and metrics["tesseract_worsen_groups"] <= int(round(group_count * float(args.tesseract_worsen_rate_max)))
        and metrics["parseq_worsen_groups"] <= int(args.parseq_worsen_max)
    )


def threshold_grid(values: np.ndarray, max_points: int) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [float("inf")]
    qs = np.linspace(0.0, 1.0, num=max(2, max_points))
    grid = sorted(set(float(x) for x in np.quantile(finite, qs)))
    grid.extend([0.25, 0.5, 0.75, 0.9, float("inf")])
    return sorted(set(grid))


def tune_policy(
    groups: list[dict[str, Any]],
    arrays: dict[str, Any],
    train_groups: np.ndarray,
    candidate_probs: np.ndarray,
    group_probs: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_key = None
    train_candidate_values = []
    train_group_values = []
    for group_index in train_groups:
        local = arrays["per_group_candidates"][int(group_index)]
        if not local:
            continue
        train_candidate_values.append(float(candidate_probs[local].max()))
        train_group_values.append(float(group_probs[int(group_index)]))
    candidate_grid = threshold_grid(np.asarray(train_candidate_values, dtype=np.float64), args.threshold_grid_points)
    group_grid = threshold_grid(np.asarray(train_group_values, dtype=np.float64), args.threshold_grid_points)
    for mode in args.policy_mode:
        if mode == "two_threshold":
            pairs = [(cand, grp) for cand in candidate_grid for grp in group_grid]
        elif mode == "group_threshold":
            pairs = [(0.0, grp) for grp in group_grid]
        else:
            pairs = [(cand, 0.0) for cand in candidate_grid]
        for candidate_threshold, group_threshold in pairs:
            rows = candidate_policy_rows(
                groups,
                arrays,
                train_groups,
                candidate_probs,
                group_probs,
                mode=mode,
                candidate_threshold=float(candidate_threshold),
                group_threshold=float(group_threshold),
            )
            metrics = summarize(rows)
            if not feasible(metrics, args, len(train_groups)):
                continue
            key = (
                int(metrics["tesseract_delta_sum"]),
                int(metrics["parseq_delta_sum"]),
                int(metrics["false_change_groups"]),
                int(metrics["wrong_change_groups"]),
                int(metrics["tesseract_worsen_groups"]),
                -int(metrics["exact_oracle_groups"]),
                int(metrics["changed_groups"]),
            )
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "policy_mode": mode,
                    "candidate_threshold": float(candidate_threshold),
                    "group_threshold": float(group_threshold),
                    "train_metrics": metrics,
                }
    if best is None:
        best = {
            "policy_mode": "candidate_threshold",
            "candidate_threshold": float("inf"),
            "group_threshold": float("inf"),
            "train_metrics": summarize(select_nearest_or_oracle(groups, train_groups, "nearest")),
            "fallback": "nearest_only_no_feasible_threshold",
        }
    return best


def split_indices(groups: list[dict[str, Any]], split_key_index: int, val_seed: int) -> tuple[np.ndarray, np.ndarray]:
    keys = [group["key"] for group in groups]
    train = [index for index, key in enumerate(keys) if int(key[split_key_index]) != int(val_seed)]
    val = [index for index, key in enumerate(keys) if int(key[split_key_index]) == int(val_seed)]
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def write_policy(path: Path, rows: list[dict[str, Any]], seed: int, tuned: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (int(item["real_seed"]), int(item["source_index"]), int(item["candidate_index"]))):
            out = {
                "source_index": int(row["source_index"]),
                "candidate_index": int(row["candidate_index"]),
                "assignment_code_index": int(row["code_index"]),
                "code_index": int(row["code_index"]),
                "nearest_code": int(row["nearest_code"]),
                "topk_rank": int(row["topk_rank"]),
                "selected_by_policy": bool(int(row["selected_by_verifier"])),
                "selector_score": float(row.get("policy_confidence", 0.0)),
                "candidate_probability": float(row.get("candidate_probability", 0.0)),
                "group_probability": float(row.get("group_probability", 0.0)),
                "policy": str(row.get("policy", "topk_score_verifier")),
                "policy_tuned_config": tuned,
                "verifier_seed": int(seed),
                "seed": int(row["seed"]),
                "real_seed": int(row.get("real_seed", row["seed"])),
                "parseq_delta_vs_nearest": int(row["parseq_delta_vs_nearest"]),
                "tesseract_delta_vs_nearest": int(row["tesseract_delta_vs_nearest"]),
                "label_assignment_oracle_choice": int(row["label_assignment_oracle_choice"]),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def aggregate_seed_results(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in per_seed:
        for section in ["candidate_metrics", "group_metrics"]:
            for split_name, values in row[section].items():
                for key, value in values.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        metrics[f"{section}/{split_name}"][key].append(float(value))
        for split_name, values in row["policy"].items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    metrics[f"policy/{split_name}"][key].append(float(value))
    out: dict[str, dict[str, float]] = {}
    for section, values in metrics.items():
        out[section] = {}
        for key, vals in values.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[section][f"{key}_mean"] = float(arr.mean())
            out[section][f"{key}_std"] = float(arr.std(ddof=0))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-output-prefix", type=Path)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--split-key-index", type=int, default=1, choices=[0, 1])
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--verifier-seed", type=int, action="append", default=[])
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold-grid-points", type=int, default=41)
    parser.add_argument(
        "--policy-mode",
        action="append",
        default=[],
        choices=["candidate_threshold", "group_threshold", "product_threshold", "min_threshold", "two_threshold"],
    )
    parser.add_argument("--false-change-rate-max", type=float, default=0.0)
    parser.add_argument("--wrong-change-rate-max", type=float, default=0.0)
    parser.add_argument("--tesseract-worsen-rate-max", type=float, default=0.0)
    parser.add_argument("--parseq-worsen-max", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.verifier_seed:
        args.verifier_seed = [0, 1, 2]
    if not args.policy_mode:
        args.policy_mode = ["candidate_threshold", "product_threshold", "min_threshold", "two_threshold"]

    raw_groups = load_score_groups(args.score)
    groups = [build_group(rows) for _, rows in sorted(raw_groups.items())]
    train_groups, val_groups = split_indices(groups, split_key_index=args.split_key_index, val_seed=args.val_seed)
    if len(train_groups) == 0 or len(val_groups) == 0:
        raise SystemExit("empty train/val split")

    arrays = build_arrays(groups, top_m=int(args.top_m))
    candidate_train_indices = np.flatnonzero(np.isin(arrays["candidate_group_indices"], train_groups))
    candidate_val_indices = np.flatnonzero(np.isin(arrays["candidate_group_indices"], val_groups))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    per_seed = []
    best_val_rows: list[dict[str, Any]] | None = None
    best_seed = None
    best_seed_key = None
    for seed in args.verifier_seed:
        candidate_probs, candidate_fit = fit_binary_model(
            arrays["candidate_features"],
            arrays["candidate_labels"],
            candidate_train_indices,
            seed=int(seed),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            device=device,
        )
        group_probs, group_fit = fit_binary_model(
            arrays["group_features"],
            arrays["group_labels"],
            train_groups,
            seed=int(seed) + 10_000,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            device=device,
        )
        tuned = tune_policy(groups, arrays, train_groups, candidate_probs, group_probs, args)
        val_rows = candidate_policy_rows(
            groups,
            arrays,
            val_groups,
            candidate_probs,
            group_probs,
            mode=str(tuned["policy_mode"]),
            candidate_threshold=float(tuned["candidate_threshold"]),
            group_threshold=float(tuned["group_threshold"]),
        )
        val_metrics = summarize(val_rows)
        train_rows = candidate_policy_rows(
            groups,
            arrays,
            train_groups,
            candidate_probs,
            group_probs,
            mode=str(tuned["policy_mode"]),
            candidate_threshold=float(tuned["candidate_threshold"]),
            group_threshold=float(tuned["group_threshold"]),
        )
        row = {
            "verifier_seed": int(seed),
            "candidate_fit": {k: v for k, v in candidate_fit.items() if not k.startswith("feature_")},
            "group_fit": {k: v for k, v in group_fit.items() if not k.startswith("feature_")},
            "candidate_metrics": {
                "train": {
                    "auc": roc_auc(candidate_probs[candidate_train_indices], arrays["candidate_labels"][candidate_train_indices]),
                    "ap": average_precision(candidate_probs[candidate_train_indices], arrays["candidate_labels"][candidate_train_indices]),
                },
                "val": {
                    "auc": roc_auc(candidate_probs[candidate_val_indices], arrays["candidate_labels"][candidate_val_indices]),
                    "ap": average_precision(candidate_probs[candidate_val_indices], arrays["candidate_labels"][candidate_val_indices]),
                },
            },
            "group_metrics": {
                "train": {
                    "auc": roc_auc(group_probs[train_groups], arrays["group_labels"][train_groups]),
                    "ap": average_precision(group_probs[train_groups], arrays["group_labels"][train_groups]),
                },
                "val": {
                    "auc": roc_auc(group_probs[val_groups], arrays["group_labels"][val_groups]),
                    "ap": average_precision(group_probs[val_groups], arrays["group_labels"][val_groups]),
                },
            },
            "tuned": tuned,
            "policy": {
                "train_reapplied": summarize(train_rows),
                "val_transfer": val_metrics,
            },
        }
        per_seed.append(row)
        if args.policy_output_prefix is not None:
            write_policy(args.policy_output_prefix.with_name(f"{args.policy_output_prefix.name}_seed{seed}.jsonl"), val_rows, int(seed), tuned)
        key = (
            int(val_metrics["tesseract_delta_sum"]),
            int(val_metrics["parseq_delta_sum"]),
            int(val_metrics["false_change_groups"]),
            int(val_metrics["wrong_change_groups"]),
            int(val_metrics["tesseract_worsen_groups"]),
            -int(val_metrics["exact_oracle_groups"]),
        )
        if best_seed_key is None or key < best_seed_key:
            best_seed_key = key
            best_seed = int(seed)
            best_val_rows = val_rows

    output = {
        "description": "Top-k score verifier diagnostic. Inference features are first-stage scores/ranks/votes only; OCR deltas are labels and evaluation only.",
        "scores": [str(path) for path in args.score],
        "config": {
            "val_seed": int(args.val_seed),
            "split_key_index": int(args.split_key_index),
            "top_m": int(args.top_m),
            "verifier_seeds": [int(seed) for seed in args.verifier_seed],
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "device": str(device),
            "policy_modes": list(args.policy_mode),
            "limits": {
                "false_change_rate_max": float(args.false_change_rate_max),
                "wrong_change_rate_max": float(args.wrong_change_rate_max),
                "tesseract_worsen_rate_max": float(args.tesseract_worsen_rate_max),
                "parseq_worsen_max": int(args.parseq_worsen_max),
            },
            "train_groups": int(len(train_groups)),
            "val_groups": int(len(val_groups)),
            "train_candidate_rows": int(len(candidate_train_indices)),
            "val_candidate_rows": int(len(candidate_val_indices)),
        },
        "baselines": {
            "train_nearest": summarize(select_nearest_or_oracle(groups, train_groups, "nearest")),
            "train_oracle": summarize(select_nearest_or_oracle(groups, train_groups, "oracle")),
            f"train_top{args.top_m}_oracle_upper": summarize(select_oracle_upper(groups, train_groups, int(args.top_m))),
            "val_nearest": summarize(select_nearest_or_oracle(groups, val_groups, "nearest")),
            "val_oracle": summarize(select_nearest_or_oracle(groups, val_groups, "oracle")),
            f"val_top{args.top_m}_oracle_upper": summarize(select_oracle_upper(groups, val_groups, int(args.top_m))),
        },
        "per_seed": per_seed,
        "aggregate": aggregate_seed_results(per_seed),
        "best_seed_by_val_policy": best_seed,
    }
    if best_val_rows is not None:
        output["best_val_policy_metrics"] = summarize(best_val_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"best_seed": best_seed, "best_val_policy_metrics": output.get("best_val_policy_metrics"), "aggregate": output["aggregate"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
