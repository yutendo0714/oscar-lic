#!/usr/bin/env python3
"""Generate source-modulo out-of-fold assignment patch-ranker scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def write_score_rows(
    path: Path,
    rows: list[dict[str, Any]],
    original_indices: list[int],
    scores: np.ndarray,
    *,
    model_seed: int,
    fold_remainder: int,
    source_modulo: int,
    target: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for local_index, row in enumerate(rows):
            out = {
                "row_index": int(original_indices[local_index]),
                "model_seed": int(model_seed),
                "score": float(scores[local_index]),
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
                "oof_split": "source_modulo",
                "source_modulo": int(source_modulo),
                "fold_remainder": int(fold_remainder),
                "target": target,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def policy_metrics(rows: list[dict[str, Any]], selected: list[int]) -> dict[str, int]:
    return ranker.policy_metrics(rows, selected)


def select_oracle(rows: list[dict[str, Any]], groups: dict[tuple[int, int, int, int], list[int]]) -> list[int]:
    selected = []
    for group in groups.values():
        oracle = [idx for idx in group if int(rows[idx]["label_assignment_oracle_choice"])]
        if oracle:
            selected.append(oracle[0])
        else:
            selected.append(next(idx for idx in group if int(rows[idx]["is_nearest"])))
    return selected


def train_fold(
    rows: list[dict[str, Any]],
    original_indices: list[int],
    images: np.ndarray,
    tabular: np.ndarray,
    labels: np.ndarray,
    *,
    model_seed: int,
    fold_remainder: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(model_seed * 1000 + fold_remainder)
    np.random.seed(model_seed * 1000 + fold_remainder)
    torch.manual_seed(model_seed * 1000 + fold_remainder)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed_all(model_seed * 1000 + fold_remainder)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    source_mod = np.asarray([int(row["source_index"]) % args.source_modulo for row in rows], dtype=np.int64)
    val_mask = source_mod == fold_remainder
    train_mask = ~val_mask
    if not train_mask.any() or not val_mask.any():
        raise SystemExit(f"empty OOF fold remainder={fold_remainder}: train={train_mask.sum()} val={val_mask.sum()}")

    tab_mean = tabular[train_mask].mean(axis=0, keepdims=True)
    tab_std = tabular[train_mask].std(axis=0, keepdims=True)
    tab_std[tab_std < 1e-6] = 1.0
    tab_norm = ((tabular - tab_mean) / tab_std).astype(np.float32)

    model_cls = ranker.AssignmentTabularRanker if args.tabular_only else ranker.AssignmentPatchRanker
    model = model_cls(tab_dim=tabular.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    train_images = torch.from_numpy(images[train_mask]).to(device)
    train_tab = torch.from_numpy(tab_norm[train_mask].astype(np.float32)).to(device)
    train_labels = torch.from_numpy(labels[train_mask]).to(device)
    positives = float(train_labels.sum().item())
    negatives = float(len(train_labels) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    )

    train_indices = np.flatnonzero(train_mask)
    left, right = ranker.pair_indices(rows, train_indices, labels)
    left = left.to(device)
    right = right.to(device)
    listwise_positions, listwise_targets = ranker.listwise_group_targets(rows, train_indices, labels)
    listwise_positions = [positions.to(device) for positions in listwise_positions]
    listwise_targets = [torch.tensor([target], dtype=torch.long, device=device) for target in listwise_targets]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(train_images, train_tab)
        loss_terms = []
        if args.loss_mode in {"bce", "bce_listwise"}:
            loss_terms.append(loss_fn(logits, train_labels))
        if args.loss_mode in {"listwise", "bce_listwise"} and listwise_positions:
            group_losses = [
                torch.nn.functional.cross_entropy(logits[positions].unsqueeze(0), target)
                for positions, target in zip(listwise_positions, listwise_targets)
            ]
            loss_terms.append(args.listwise_weight * torch.stack(group_losses).mean())
        loss = torch.stack(loss_terms).sum()
        if len(left) and args.pairwise_weight > 0.0 and args.loss_mode != "listwise":
            fold_logits = logits
            pair_loss = torch.nn.functional.softplus(-(fold_logits[left] - fold_logits[right])).mean()
            loss = loss + args.pairwise_weight * pair_loss
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(
            model(
                torch.from_numpy(images).to(device),
                torch.from_numpy(tab_norm.astype(np.float32)).to(device),
            )
        ).detach().cpu().numpy()

    val_scores = scores[val_mask]
    val_labels = labels[val_mask]
    groups = ranker.grouped_indices(rows)
    val_groups = {
        key: value
        for key, value in groups.items()
        if int(rows[value[0]]["source_index"]) % args.source_modulo == fold_remainder
    }
    nearest_selected = [next(idx for idx in group if int(rows[idx]["is_nearest"])) for group in val_groups.values()]
    oracle_selected = select_oracle(rows, val_groups)
    always_best_selected = [
        max(group, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"]))) for group in val_groups.values()
    ]
    threshold = ranker.tune_conservative_threshold(
        rows,
        {key: value for key, value in groups.items() if int(rows[value[0]]["source_index"]) % args.source_modulo != fold_remainder},
        scores,
    )
    threshold_selected = [
        ranker.select_with_threshold(group, rows, scores, threshold) for group in val_groups.values()
    ]
    write_score_rows(
        args.score_output,
        [rows[idx] for idx in np.flatnonzero(val_mask)],
        [original_indices[idx] for idx in np.flatnonzero(val_mask)],
        val_scores,
        model_seed=model_seed,
        fold_remainder=fold_remainder,
        source_modulo=args.source_modulo,
        target=args.target,
    )
    return {
        "model_seed": int(model_seed),
        "fold_remainder": int(fold_remainder),
        "device": str(device),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positive_rate": float(labels[train_mask].mean()),
        "val_positive_rate": float(labels[val_mask].mean()),
        "val_auc": ranker.roc_auc(val_scores, val_labels),
        "val_ap": ranker.average_precision(val_scores, val_labels),
        "threshold": float(threshold),
        "policy_metrics": {
            "nearest": policy_metrics(rows, nearest_selected),
            "oracle": policy_metrics(rows, oracle_selected),
            "always_best_score": policy_metrics(rows, always_best_selected),
            "train_tuned_conservative_threshold": policy_metrics(rows, threshold_selected),
        },
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"policy_metrics": {}}
    for metric in ["val_auc", "val_ap"]:
        values = [row[metric] for row in results if row[metric] is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            output[metric] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "count": int(len(arr))}
    policies = sorted({name for row in results for name in row["policy_metrics"]})
    for policy in policies:
        output["policy_metrics"][policy] = {}
        metric_names = sorted({name for row in results for name in row["policy_metrics"][policy]})
        for metric in metric_names:
            values = np.asarray([float(row["policy_metrics"][policy][metric]) for row in results], dtype=np.float64)
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
    parser.add_argument("--score-output", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=600)
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--source-modulo", type=int, default=5)
    parser.add_argument("--fold-remainder", type=int, action="append")
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

    if not args.model_seed:
        args.model_seed = [2]
    if not args.fold_remainder:
        args.fold_remainder = list(range(args.source_modulo))

    all_rows = ranker.read_jsonl(args.table)
    original_indices = [idx for idx, row in enumerate(all_rows) if int(row["seed"]) == args.train_seed]
    rows = [all_rows[idx] for idx in original_indices]
    if not rows:
        raise SystemExit(f"no rows matched train_seed={args.train_seed}: {args.table}")
    if args.score_output.exists():
        args.score_output.unlink()
    if args.tabular_only:
        images = np.zeros((len(rows), 3, 1, 1), dtype=np.float32)
    else:
        images = ranker.build_image_tensor(
            rows,
            width=args.width,
            height=args.height,
            image_mode=args.image_mode,
            diff_threshold=args.diff_threshold,
            crop_pad=args.crop_pad,
        )
    tabular = ranker.build_tabular(
        rows,
        metadata_features=args.metadata_features,
        include_image_features=args.include_image_features,
    )
    if args.target == "safe_improve":
        labels = np.asarray([int(row["label_tesseract_parseq_safe_improves"]) for row in rows], dtype=np.float32)
    else:
        labels = np.asarray([int(row["label_assignment_oracle_choice"]) for row in rows], dtype=np.float32)

    results = []
    for model_seed in args.model_seed:
        for fold_remainder in args.fold_remainder:
            results.append(
                train_fold(
                    rows,
                    original_indices,
                    images,
                    tabular,
                    labels,
                    model_seed=model_seed,
                    fold_remainder=fold_remainder,
                    args=args,
                )
            )

    output = {
        "description": "Source-modulo out-of-fold assignment patch-ranker scores.",
        "config": {
            "table": str(args.table),
            "train_seed": int(args.train_seed),
            "model_seeds": args.model_seed,
            "source_modulo": int(args.source_modulo),
            "fold_remainders": args.fold_remainder,
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
        "per_fold": results,
        "aggregate": aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
