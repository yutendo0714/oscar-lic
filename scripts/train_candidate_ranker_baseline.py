#!/usr/bin/env python3
"""Train a small within-image candidate ranker and evaluate actual-bpp policies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

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
    split_by_group,
    tune_threshold,
)
from train_candidate_utility_baseline import FEATURES, average_precision, roc_auc, topk


RANK_BASE_FEATURES = FEATURES + [
    "candidate_index",
    "gate_payload_bytes",
    "residual_payload_bytes",
]
WITHIN_IMAGE_FEATURES = [
    "actual_total_bpp",
    "psnr_enhanced_db",
    "y_hat_mean_abs",
    "y_hat_rms",
    "y_target_mean_abs",
    "y_target_rms",
    "latent_residual_mean_abs",
    "latent_residual_rms",
    "latent_residual_max_abs",
]
OPTIONAL_FEATURE_PREFIXES = (
    "image_",
    "candidate_pixel_",
    "orig_patch_",
    "base_patch_",
    "patch_base_",
    "full_base_",
    "latent_vec_",
)


def compact_policy(policy: dict, evaluator_labels: list[str]) -> dict:
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
    output = {}
    for key in rows[0].keys():
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return output


def is_numeric_feature(rows: list[dict], name: str) -> bool:
    try:
        for row in rows:
            float(row[name])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def available_optional_features(rows: list[dict]) -> list[str]:
    names = sorted(
        name
        for row in rows[:1]
        for name in row
        if name.startswith(OPTIONAL_FEATURE_PREFIXES)
    )
    return [name for name in names if is_numeric_feature(rows, name)]


def build_features(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["source_index"])].append(index)

    optional_features = available_optional_features(rows)
    base_features = [name for name in RANK_BASE_FEATURES if is_numeric_feature(rows, name)] + optional_features
    within_features = [name for name in WITHIN_IMAGE_FEATURES if is_numeric_feature(rows, name)] + [
        name for name in optional_features if not name.startswith("image_")
    ]
    base = np.asarray([[float(row[name]) for name in base_features] for row in rows], dtype=np.float32)
    feature_blocks = [base]
    names = list(base_features)

    for feature in within_features:
        values = np.asarray([float(row[feature]) for row in rows], dtype=np.float32)
        z = np.zeros_like(values)
        rank_hi = np.zeros_like(values)
        rank_lo = np.zeros_like(values)
        for indices in grouped.values():
            group_values = values[indices]
            std = float(group_values.std())
            if std < 1e-6:
                std = 1.0
            z[indices] = (group_values - float(group_values.mean())) / std
            order_hi = np.argsort(-group_values)
            order_lo = np.argsort(group_values)
            denom = max(len(indices) - 1, 1)
            for rank, local in enumerate(order_hi):
                rank_hi[indices[local]] = 1.0 - rank / denom
            for rank, local in enumerate(order_lo):
                rank_lo[indices[local]] = 1.0 - rank / denom
        feature_blocks.extend([z[:, None], rank_hi[:, None], rank_lo[:, None]])
        names.extend([f"{feature}_image_z", f"{feature}_rank_high", f"{feature}_rank_low"])

    candidate_fraction = np.asarray(
        [
            float(row["candidate_index"]) / max(float(row["candidate_count"]) - 1.0, 1.0)
            for row in rows
        ],
        dtype=np.float32,
    )
    feature_blocks.append(candidate_fraction[:, None])
    names.append("candidate_index_fraction")
    return np.concatenate(feature_blocks, axis=1).astype(np.float32), names


def make_pair_indices(rows: list[dict], train_indices: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    position_by_row = {int(row_index): pos for pos, row_index in enumerate(train_indices)}
    grouped: dict[int, list[int]] = defaultdict(list)
    for row_index in train_indices:
        grouped[int(rows[int(row_index)]["source_index"])].append(int(row_index))
    left = []
    right = []
    for indices in grouped.values():
        positives = [idx for idx in indices if y[idx] > 0.5]
        negatives = [idx for idx in indices if y[idx] <= 0.5]
        for pos in positives:
            for neg in negatives:
                left.append(position_by_row[pos])
                right.append(position_by_row[neg])
    if not left:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)


class Ranker(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def train_ranker(
    rows: list[dict],
    seed: int,
    val_fraction: float,
    epochs: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    pairwise_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    x, feature_names = build_features(rows)
    y = np.asarray([int(row["label_multi_teacher_improves"]) for row in rows], dtype=np.float32)
    groups = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    train_mask, val_mask = split_by_group(groups, val_fraction, seed)
    train_indices = np.flatnonzero(train_mask)
    x_train = x[train_mask]
    x_val = x[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train_n = (x_train - mean) / std
    x_all_n = (x - mean) / std
    model = Ranker(x.shape[1], hidden_dim=hidden_dim, dropout=dropout)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    xt = torch.from_numpy(x_train_n)
    yt = torch.from_numpy(y_train)
    pair_left, pair_right = make_pair_indices(rows, train_indices, y)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(xt)
        loss = bce(logits, yt)
        if len(pair_left):
            pair_loss = torch.nn.functional.softplus(-(logits[pair_left] - logits[pair_right])).mean()
            loss = loss + pairwise_weight * pair_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x_all_n))).numpy()
    val_scores = scores[val_mask]
    model_meta = {
        "features": feature_names,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "pairwise_weight": pairwise_weight,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "train_pair_count": int(len(pair_left)),
        "ranker_train_auc": roc_auc(scores[train_mask], y_train),
        "ranker_val_auc": roc_auc(val_scores, y_val),
        "ranker_val_ap": average_precision(val_scores, y_val),
        "ranker_val_topk": topk(val_scores, y_val, [10, 20, 40]),
    }
    return scores, train_mask, val_mask, model_meta


def run_seed(args: argparse.Namespace, rows: list[dict], analyses: dict, bpp_by_key: dict, seed: int) -> dict:
    scores, train_mask, val_mask, model_meta = train_ranker(
        rows,
        seed=seed,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pairwise_weight=args.pairwise_weight,
    )
    baseline_feature, residual_scores = baseline_score_feature(rows)
    grouped_train = rows_by_image(rows, np.flatnonzero(train_mask))
    grouped_val = rows_by_image(rows, np.flatnonzero(val_mask))
    tuned = tune_threshold(grouped_train, scores)
    policy_specs = [
        ("base", choose_base(grouped_val)),
        ("oracle_single_candidate", choose_oracle(grouped_val)),
        ("ranker_train_tuned", choose_threshold(grouped_val, scores, tuned["threshold"])),
    ]
    for budget in args.image_budget:
        budget = min(int(budget), len(grouped_val))
        policy_specs.append((f"ranker_top{budget}_images", choose_top_images(grouped_val, scores, budget)))
        policy_specs.append((f"residual_rms_top{budget}_images", choose_top_images(grouped_val, residual_scores, budget)))
    policies = [
        evaluate_policy(name, selected, grouped_val, analyses, bpp_by_key, args.profile)
        for name, selected in policy_specs
    ]
    return {
        "seed": seed,
        "model": model_meta,
        "residual_baseline_score_feature": baseline_feature,
        "train_tuned_threshold": tuned,
        "policies": policies,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--candidate-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", action="append", nargs=2, metavar=("LABEL", "ANALYSIS_JSON"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
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
    per_seed = [run_seed(args, rows, analyses, bpp_by_key, seed) for seed in args.seeds]

    by_policy = defaultdict(list)
    for seed_result in per_seed:
        for policy in seed_result["policies"]:
            by_policy[policy["name"]].append(compact_policy(policy, evaluator_labels))

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
