#!/usr/bin/env python3
"""Train a tiny local patch CNN ranker for candidate utility selection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

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


ROOT = Path(__file__).resolve().parents[1]
LATENT_TO_PIXEL = 16
PATCH_TABULAR_OPTIONAL = [
    "image_width",
    "image_height",
    "image_area",
    "image_aspect",
    "candidate_pixel_area_fraction",
    "candidate_pixel_center_y_fraction",
    "candidate_pixel_center_x_fraction",
    "orig_patch_grad_mean",
    "orig_patch_edge_density",
    "patch_base_abs_error_mean",
    "patch_base_mse",
    "patch_base_grad_abs_error",
    "full_base_abs_error_mean",
    "full_base_mse",
]


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def crop_patch(array: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    height, width = array.shape
    y0 = max(0, min(height, y0))
    y1 = max(y0 + 1, min(height, y1))
    x0 = max(0, min(width, x0))
    x1 = max(x0 + 1, min(width, x1))
    return array[y0:y1, x0:x1]


def resize_patch(channels: np.ndarray, patch_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(channels).float().unsqueeze(0)
    resized = F.interpolate(tensor, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def base_rows_by_source(path: Path) -> dict[int, dict]:
    rows = read_jsonl(path)
    return {index: row for index, row in enumerate(rows)}


def build_patch_tensor(rows: list[dict], base_recon_split: Path, patch_size: int, latent_to_pixel: int) -> torch.Tensor:
    base_rows = base_rows_by_source(base_recon_split)
    cache: dict[str, np.ndarray] = {}
    patches = []
    for row in rows:
        source_index = int(row["source_index"])
        original_path = resolve_path(row["source_image"])
        base_path = resolve_path(base_rows[source_index]["image_path"])
        for path in (original_path, base_path):
            key = str(path)
            if key not in cache:
                cache[key] = load_gray(path)
        original = cache[str(original_path)]
        base = cache[str(base_path)]
        if original.shape != base.shape:
            base = np.asarray(
                Image.open(base_path).convert("L").resize((original.shape[1], original.shape[0])),
                dtype=np.float32,
            ) / 255.0
        y0 = int(row["y0"]) * latent_to_pixel
        y1 = int(row["y1"]) * latent_to_pixel
        x0 = int(row["x0"]) * latent_to_pixel
        x1 = int(row["x1"]) * latent_to_pixel
        orig_patch = crop_patch(original, y0, y1, x0, x1)
        base_patch = crop_patch(base, y0, y1, x0, x1)
        diff_patch = np.abs(orig_patch - base_patch)
        patch = np.stack([orig_patch, base_patch, diff_patch], axis=0)
        patches.append(resize_patch(patch, patch_size))
    return torch.stack(patches, dim=0)


def numeric_feature(rows: list[dict], name: str) -> bool:
    try:
        for row in rows:
            float(row[name])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def build_tabular(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = [name for name in FEATURES + ["candidate_index"] + PATCH_TABULAR_OPTIONAL if numeric_feature(rows, name)]
    x = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float32)
    return x, names


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


class PatchRanker(torch.nn.Module):
    def __init__(self, tab_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.patch = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.tab = torch.nn.Sequential(
            torch.nn.Linear(tab_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64 + hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, patches: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.patch(patches), self.tab(tabular)], dim=1)).reshape(-1)


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


def train_once(
    args: argparse.Namespace,
    rows: list[dict],
    patches: torch.Tensor,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    tabular, tabular_names = build_tabular(rows)
    y = np.asarray([int(row["label_multi_teacher_improves"]) for row in rows], dtype=np.float32)
    raw_benefit = np.asarray([max(0.0, -float(row["multi_teacher_delta_distance"])) for row in rows], dtype=np.float32)
    if args.target == "binary":
        target = y.copy()
    elif args.target == "benefit":
        target = raw_benefit.copy()
    elif args.target == "benefit_per_bpp":
        target = raw_benefit / np.maximum(
            np.asarray([float(row["actual_total_bpp"]) for row in rows], dtype=np.float32),
            1e-6,
        )
    else:
        raise ValueError(f"unsupported target: {args.target}")
    groups = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    train_mask, val_mask = split_by_group(groups, args.val_fraction, seed)
    train_indices = np.flatnonzero(train_mask)
    x_train = tabular[train_mask]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    tab_all = torch.from_numpy((tabular - mean) / std).float().to(args.device)
    patches = patches.to(args.device)
    y_train = y[train_mask]
    target_train = target[train_mask]
    target_scale = float(target_train.max()) if float(target_train.max()) > 0 else 1.0
    target_train = target_train / target_scale
    yt = torch.from_numpy(y_train).float().to(args.device)
    target_t = torch.from_numpy(target_train).float().to(args.device)

    model = PatchRanker(tab_all.shape[1], args.hidden_dim, args.dropout).to(args.device)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=args.device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_tensor_indices = torch.from_numpy(train_indices).long().to(args.device)
    pair_left, pair_right = make_pair_indices(rows, train_indices, y)
    pair_left = pair_left.to(args.device)
    pair_right = pair_right.to(args.device)
    for _ in range(args.epochs):
        opt.zero_grad()
        logits = model(patches[train_tensor_indices], tab_all[train_tensor_indices])
        if args.target == "binary":
            loss = bce(logits, yt)
        else:
            loss = torch.nn.functional.mse_loss(torch.sigmoid(logits), target_t)
        if len(pair_left):
            pair_loss = torch.nn.functional.softplus(-(logits[pair_left] - logits[pair_right])).mean()
            loss = loss + args.pairwise_weight * pair_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(patches, tab_all)).detach().cpu().numpy()
    y_val = y[val_mask]
    model_meta = {
        "seed": seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "pairwise_weight": args.pairwise_weight,
        "weight_decay": args.weight_decay,
        "patch_size": args.patch_size,
        "target": args.target,
        "target_scale": target_scale,
        "tabular_features": tabular_names,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "train_pair_count": int(len(pair_left)),
        "ranker_train_auc": roc_auc(scores[train_mask], y_train),
        "ranker_val_auc": roc_auc(scores[val_mask], y_val),
        "ranker_val_ap": average_precision(scores[val_mask], y_val),
        "ranker_val_topk": topk(scores[val_mask], y_val, [10, 20, 40]),
    }
    return scores, train_mask, val_mask, model_meta


def run_seed(args: argparse.Namespace, rows: list[dict], patches: torch.Tensor, analyses: dict, bpp_by_key: dict, seed: int) -> dict:
    scores, train_mask, val_mask, model_meta = train_once(args, rows, patches, seed)
    baseline_feature, residual_scores = baseline_score_feature(rows)
    grouped_train = rows_by_image(rows, np.flatnonzero(train_mask))
    grouped_val = rows_by_image(rows, np.flatnonzero(val_mask))
    tuned = tune_threshold(grouped_train, scores)
    policy_specs = [
        ("base", choose_base(grouped_val)),
        ("oracle_single_candidate", choose_oracle(grouped_val)),
        ("patch_ranker_train_tuned", choose_threshold(grouped_val, scores, tuned["threshold"])),
    ]
    for budget in args.image_budget:
        budget = min(int(budget), len(grouped_val))
        policy_specs.append((f"patch_ranker_top{budget}_images", choose_top_images(grouped_val, scores, budget)))
        policy_specs.append((f"residual_rms_top{budget}_images", choose_top_images(grouped_val, residual_scores, budget)))
    return {
        "seed": seed,
        "model": model_meta,
        "residual_baseline_score_feature": baseline_feature,
        "train_tuned_threshold": tuned,
        "policies": [
            evaluate_policy(name, selected, grouped_val, analyses, bpp_by_key, args.profile)
            for name, selected in policy_specs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--base-recon-split", type=Path, required=True)
    parser.add_argument("--candidate-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", action="append", nargs=2, metavar=("LABEL", "ANALYSIS_JSON"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--target", choices=["binary", "benefit", "benefit_per_bpp"], default="binary")
    parser.add_argument("--latent-to-pixel", type=int, default=LATENT_TO_PIXEL)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--image-budget", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--base-structural-overhead-bytes", type=int, default=88)
    args = parser.parse_args()

    if not args.evaluator:
        raise SystemExit("provide at least one --evaluator LABEL ANALYSIS_JSON")
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and args.device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    rows = read_jsonl(args.table)
    patches = build_patch_tensor(rows, args.base_recon_split, args.patch_size, args.latent_to_pixel)
    evaluator_labels = [label for label, _ in args.evaluator]
    analyses = {label: read_analysis(Path(path)) for label, path in args.evaluator}
    bpp_by_key = candidate_split_maps(args.candidate_split, args.base_structural_overhead_bytes)
    per_seed = [run_seed(args, rows, patches, analyses, bpp_by_key, seed) for seed in args.seeds]

    by_policy = defaultdict(list)
    for seed_result in per_seed:
        for policy in seed_result["policies"]:
            by_policy[policy["name"]].append(compact_policy(policy, evaluator_labels))

    output = {
        "table": str(args.table),
        "base_recon_split": str(args.base_recon_split),
        "candidate_split": str(args.candidate_split),
        "profile": args.profile,
        "device": args.device,
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
