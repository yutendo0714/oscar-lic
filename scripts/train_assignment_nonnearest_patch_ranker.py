#!/usr/bin/env python3
"""Train a non-nearest-only assignment patch ranker.

The previous patch rankers learned a score over both nearest/no-op and changed
code candidates. N049 showed that this makes oracle-improving changed
candidates consistently lose to nearest. This diagnostic trains only on
non-nearest rows and uses a thresholded abstention rule to fall back to nearest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_assignment_patch_ranker as ranker  # noqa: E402
import train_assignment_topk_policy_selector as base  # noqa: E402


def listwise_positive_targets(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[torch.Tensor], list[int]]:
    pos_by_row = {int(row_index): pos for pos, row_index in enumerate(indices)}
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for row_index in indices:
        row = rows[int(row_index)]
        groups.setdefault(base.group_key(row), []).append(int(row_index))

    positions_out: list[torch.Tensor] = []
    targets_out: list[int] = []
    for group_indices in groups.values():
        ordered = sorted(group_indices, key=lambda idx: int(rows[idx]["topk_rank"]))
        positives = [idx for idx in ordered if labels[idx] > 0.5]
        if not positives:
            continue
        target_row = min(
            positives,
            key=lambda idx: (
                int(rows[idx]["parseq_delta_vs_nearest"]) + int(rows[idx]["tesseract_delta_vs_nearest"]),
                int(rows[idx]["parseq_delta_vs_nearest"]),
                int(rows[idx]["topk_rank"]),
            ),
        )
        positions_out.append(torch.tensor([pos_by_row[idx] for idx in ordered], dtype=torch.long))
        targets_out.append(ordered.index(target_row))
    return positions_out, targets_out


def select_best_nonnearest_threshold(group: list[int], rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> int:
    nearest = next(idx for idx in group if int(rows[idx]["is_nearest"]))
    nonnearest = [idx for idx in group if not int(rows[idx]["is_nearest"])]
    if not nonnearest:
        return nearest
    best = max(nonnearest, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
    return best if float(scores[best]) >= threshold else nearest


def threshold_grid(values: list[float], max_candidates: int = 256) -> list[float]:
    unique = sorted(set(float(value) for value in values))
    if not unique:
        return [1.0]
    if len(unique) > max_candidates:
        positions = np.linspace(0, len(unique) - 1, num=max_candidates).round().astype(np.int64)
        unique = sorted({unique[int(pos)] for pos in positions})
    return [max(unique) + 1e-6] + unique + [min(unique) - 1e-6]


def tune_threshold(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
    *,
    mode: str,
) -> float:
    values = [
        float(scores[idx])
        for group in groups.values()
        for idx in group
        if not int(rows[idx]["is_nearest"])
    ]
    best_threshold = 1.0
    best_key: tuple[float, ...] | None = None
    for threshold in threshold_grid(values):
        selected = [select_best_nonnearest_threshold(group, rows, scores, threshold) for group in groups.values()]
        metrics = base.policy_metrics_by_group(rows, groups, selected)
        if mode == "utility":
            key = (
                float(metrics["parseq_worsen_groups"]),
                float(metrics["tesseract_delta_sum"]),
                float(metrics["tesseract_worsen_groups"]),
                float(metrics["wrong_changes"]),
                float(metrics["changed_groups"]),
            )
        elif mode == "safe":
            key = (
                float(metrics["parseq_worsen_groups"]),
                float(metrics["tesseract_worsen_groups"]),
                float(metrics["wrong_changes"]),
                float(metrics["tesseract_delta_sum"]),
                float(metrics["false_changes"]),
                -float(metrics["exact_oracle_changes"]),
            )
        else:
            raise ValueError(f"unknown threshold tuning mode: {mode}")
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def score_rows_to_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    val_seed: int,
    model_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            out = {
                "row_index": int(row_index),
                "val_seed": int(val_seed),
                "model_seed": int(model_seed),
                "score": float(scores[row_index]),
                "seed": int(row["seed"]),
                "real_seed": int(row.get("real_seed", row["seed"])),
                "source_index": int(row["source_index"]),
                "candidate_index": int(row["candidate_index"]),
                "code_index": int(row["code_index"]),
                "nearest_code": int(row["nearest_code"]),
                "topk_rank": int(row["topk_rank"]),
                "is_nearest": int(row["is_nearest"]),
                "label_assignment_oracle_choice": int(row["label_assignment_oracle_choice"]),
                "label_tesseract_parseq_safe_improves": int(row["label_tesseract_parseq_safe_improves"]),
                "parseq_delta_vs_nearest": int(row["parseq_delta_vs_nearest"]),
                "tesseract_delta_vs_nearest": int(row["tesseract_delta_vs_nearest"]),
                "source_image": row["source_image"],
                "image_path": row["image_path"],
                "score_space": "nonnearest_only",
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def train_one(
    rows: list[dict[str, Any]],
    images: np.ndarray,
    tabular: np.ndarray,
    *,
    val_seed: int,
    model_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed_all(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if args.target == "safe_improve":
        labels = np.asarray([int(row["label_tesseract_parseq_safe_improves"]) for row in rows], dtype=np.float32)
    else:
        labels = np.asarray([int(row["label_assignment_oracle_choice"]) for row in rows], dtype=np.float32)

    seeds = np.asarray([int(row["seed"]) for row in rows], dtype=np.int64)
    nonnearest = np.asarray([not int(row["is_nearest"]) for row in rows], dtype=bool)
    train_mask = (seeds != val_seed) & nonnearest
    val_mask = seeds == val_seed
    train_indices = np.flatnonzero(train_mask)
    if not len(train_indices):
        raise SystemExit("no non-nearest training rows")

    tab_mean = tabular[train_mask].mean(axis=0, keepdims=True)
    tab_std = tabular[train_mask].std(axis=0, keepdims=True)
    tab_std[tab_std < 1e-6] = 1.0
    tab_norm = ((tabular - tab_mean) / tab_std).astype(np.float32)

    model_cls = ranker.AssignmentTabularRanker if args.tabular_only else ranker.AssignmentPatchRanker
    model = model_cls(tab_dim=tabular.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    x_img = torch.from_numpy(images).to(device)
    x_tab = torch.from_numpy(tab_norm).to(device)
    train_labels = torch.from_numpy(labels[train_mask]).to(device)
    positives = float(train_labels.sum().item())
    negatives = float(len(train_labels) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    )
    left, right = ranker.pair_indices(rows, train_indices, labels)
    left = left.to(device)
    right = right.to(device)
    listwise_positions, listwise_targets = listwise_positive_targets(rows, train_indices, labels)
    listwise_positions = [positions.to(device) for positions in listwise_positions]
    listwise_targets_t = [torch.tensor([target], dtype=torch.long, device=device) for target in listwise_targets]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits_all = model(x_img, x_tab)
        logits = logits_all[torch.from_numpy(train_indices).to(device)]
        loss_terms = []
        if args.loss_mode in {"bce", "bce_listwise"}:
            loss_terms.append(loss_fn(logits, train_labels))
        if args.loss_mode in {"listwise", "bce_listwise"} and listwise_positions:
            losses = [
                torch.nn.functional.cross_entropy(logits[pos].unsqueeze(0), target)
                for pos, target in zip(listwise_positions, listwise_targets_t)
            ]
            loss_terms.append(args.listwise_weight * torch.stack(losses).mean())
        loss = torch.stack(loss_terms).sum()
        if len(left) and args.pairwise_weight > 0.0 and args.loss_mode != "listwise":
            loss = loss + args.pairwise_weight * torch.nn.functional.softplus(
                -(logits[left] - logits[right])
            ).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x_img, x_tab)).detach().cpu().numpy()
    scores[~nonnearest] = 0.0

    groups = base.grouped_indices(rows)
    train_groups = {key: value for key, value in groups.items() if int(rows[value[0]]["seed"]) != val_seed}
    val_groups = {key: value for key, value in groups.items() if int(rows[value[0]]["seed"]) == val_seed}
    safe_threshold = tune_threshold(rows, train_groups, scores, mode="safe")
    utility_threshold = tune_threshold(rows, train_groups, scores, mode="utility")
    always_best = [
        select_best_nonnearest_threshold(group, rows, scores, threshold=-1.0) for group in val_groups.values()
    ]
    safe_selected = [
        select_best_nonnearest_threshold(group, rows, scores, threshold=safe_threshold) for group in val_groups.values()
    ]
    utility_selected = [
        select_best_nonnearest_threshold(group, rows, scores, threshold=utility_threshold) for group in val_groups.values()
    ]
    nearest_selected = [next(idx for idx in group if int(rows[idx]["is_nearest"])) for group in val_groups.values()]
    oracle_selected = [
        next((idx for idx in group if int(rows[idx]["label_assignment_oracle_choice"])), nearest_selected[pos])
        for pos, group in enumerate(val_groups.values())
    ]

    if args.score_output is not None:
        score_rows_to_jsonl(args.score_output, rows, scores, val_seed=val_seed, model_seed=model_seed)

    val_scores = scores[val_mask & nonnearest]
    val_labels = labels[val_mask & nonnearest]
    train_scores = scores[train_mask]
    train_labels_np = labels[train_mask]
    return {
        "val_seed": int(val_seed),
        "model_seed": int(model_seed),
        "device": str(device),
        "target": args.target,
        "train_rows": int(train_mask.sum()),
        "train_positive_rate": float(train_labels_np.mean()) if len(train_labels_np) else 0.0,
        "val_nonnearest_rows": int((val_mask & nonnearest).sum()),
        "val_positive_rate": float(val_labels.mean()) if len(val_labels) else 0.0,
        "train_auc": ranker.roc_auc(train_scores, train_labels_np),
        "train_ap": ranker.average_precision(train_scores, train_labels_np),
        "val_auc": ranker.roc_auc(val_scores, val_labels),
        "val_ap": ranker.average_precision(val_scores, val_labels),
        "safe_threshold": float(safe_threshold),
        "utility_threshold": float(utility_threshold),
        "policies": {
            "nearest": base.policy_metrics_by_group(rows, val_groups, nearest_selected),
            "oracle": base.policy_metrics_by_group(rows, val_groups, oracle_selected),
            "always_best_nonnearest": base.policy_metrics_by_group(rows, val_groups, always_best),
            "safe_threshold": base.policy_metrics_by_group(rows, val_groups, safe_selected),
            "utility_threshold": base.policy_metrics_by_group(rows, val_groups, utility_selected),
        },
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"scalar_metrics": {}, "policy_metrics": {}}
    for metric in ["train_auc", "train_ap", "val_auc", "val_ap", "safe_threshold", "utility_threshold"]:
        values = [row[metric] for row in runs if row.get(metric) is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            output["scalar_metrics"][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    policies = sorted({name for run in runs for name in run["policies"]})
    for policy in policies:
        output["policy_metrics"][policy] = {}
        metric_names = sorted({metric for run in runs for metric in run["policies"][policy]})
        for metric in metric_names:
            values = np.asarray([float(run["policies"][policy][metric]) for run in runs], dtype=np.float64)
            output["policy_metrics"][policy][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--loss-mode", choices=["bce", "listwise", "bce_listwise"], default="bce_listwise")
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--target", choices=["safe_improve", "oracle_choice"], default="oracle_choice")
    parser.add_argument("--metadata-features", choices=["none", "source", "source_length", "source_length_difficulty"], default="none")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--image-mode", choices=["full", "diff_crop"], default="diff_crop")
    parser.add_argument("--diff-threshold", type=float, default=0.002)
    parser.add_argument("--crop-pad", type=int, default=4)
    parser.add_argument("--tabular-only", action="store_true")
    parser.add_argument("--include-image-features", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.model_seed is None:
        args.model_seed = [0, 1, 2]
    rows = ranker.read_jsonl(args.table)
    if args.score_output is not None and args.score_output.exists():
        args.score_output.unlink()
    images = (
        np.zeros((len(rows), 3, 1, 1), dtype=np.float32)
        if args.tabular_only
        else ranker.build_image_tensor(
            rows,
            width=args.width,
            height=args.height,
            image_mode=args.image_mode,
            diff_threshold=args.diff_threshold,
            crop_pad=args.crop_pad,
        )
    )
    tabular = ranker.build_tabular(
        rows,
        metadata_features=args.metadata_features,
        include_image_features=args.include_image_features,
    )
    runs = [
        train_one(rows, images, tabular, val_seed=args.val_seed, model_seed=model_seed, args=args)
        for model_seed in args.model_seed
    ]
    result = {
        "description": "Non-nearest-only assignment patch ranker with thresholded nearest fallback.",
        "validity": "diagnostic_no_promotion",
        "track": "A_pure_visual_bitstream_policy_diagnostic",
        "config": {
            "table": str(args.table),
            "val_seed": int(args.val_seed),
            "model_seeds": args.model_seed,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "pairwise_weight": float(args.pairwise_weight),
            "loss_mode": args.loss_mode,
            "listwise_weight": float(args.listwise_weight),
            "target": args.target,
            "metadata_features": args.metadata_features,
            "width": int(args.width),
            "height": int(args.height),
            "image_mode": args.image_mode,
            "diff_threshold": float(args.diff_threshold),
            "crop_pad": int(args.crop_pad),
            "include_image_features": bool(args.include_image_features),
            "cpu": bool(args.cpu),
        },
        "per_seed": runs,
        "aggregate": aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
