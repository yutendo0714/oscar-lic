#!/usr/bin/env python3
"""Train a tiny candidate utility baseline on JSONL features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch


FEATURES = [
    "slice_index",
    "x0",
    "x1",
    "y0",
    "y1",
    "channel0",
    "channel1",
    "candidate_count",
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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    rank_sum = 0.0
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sorted_labels[i:j].sum()
        i = j
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = int(labels.sum())
    if pos == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    precisions = []
    hits = 0
    for rank, label in enumerate(sorted_labels, 1):
        if label:
            hits += 1
            precisions.append(hits / rank)
    return float(sum(precisions) / pos)


def topk(scores: np.ndarray, labels: np.ndarray, ks: list[int]) -> dict:
    order = np.argsort(-scores)
    result = {}
    for k in ks:
        k = min(k, len(labels))
        top = labels[order[:k]]
        result[str(k)] = {"positives": int(top.sum()), "precision": float(top.mean()) if k else None}
    return result


def make_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[key]) for key in FEATURES] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["label_multi_teacher_improves"]) for row in rows], dtype=np.float32)
    groups = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    return x, y, groups


def split_by_group(groups: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(int(group) for group in groups))
    rng = random.Random(seed)
    rng.shuffle(unique)
    val_count = max(1, round(len(unique) * val_fraction))
    val_groups = set(unique[:val_count])
    val_mask = np.asarray([group in val_groups for group in groups], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = read_jsonl(args.input)
    x, y, groups = make_arrays(rows)
    train_mask, val_mask = split_by_group(groups, args.val_fraction, args.seed)
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_val = x[val_mask]
    y_val = y[val_mask]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    model = torch.nn.Linear(x.shape[1], 1)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    xt = torch.from_numpy(x_train_n)
    yt = torch.from_numpy(y_train).reshape(-1, 1)
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_scores = torch.sigmoid(model(torch.from_numpy(x_train_n))).reshape(-1).numpy()
        val_scores = torch.sigmoid(model(torch.from_numpy(x_val_n))).reshape(-1).numpy()

    residual_score = x_val[:, FEATURES.index("latent_residual_rms")]
    output = {
        "input": str(args.input),
        "features": FEATURES,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "linear_train_auc": roc_auc(train_scores, y_train),
        "linear_val_auc": roc_auc(val_scores, y_val),
        "linear_val_ap": average_precision(val_scores, y_val),
        "residual_rms_val_auc": roc_auc(residual_score, y_val),
        "residual_rms_val_ap": average_precision(residual_score, y_val),
        "linear_val_topk": topk(val_scores, y_val, [10, 20, 40]),
        "residual_rms_val_topk": topk(residual_score, y_val, [10, 20, 40]),
        "weights": {name: float(weight) for name, weight in zip(FEATURES, model.weight.detach().reshape(-1).numpy())},
        "bias": float(model.bias.detach().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
