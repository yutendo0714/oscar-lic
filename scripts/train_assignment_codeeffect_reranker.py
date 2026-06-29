#!/usr/bin/env python3
"""Train a compact tensor-backed assignment reranker over nearest plus top-k codes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch


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


class CodeEffectReranker(torch.nn.Module):
    def __init__(
        self,
        core_dim: int,
        code_dim: int,
        hidden_dim: int,
        dropout: float,
        use_images: bool,
        latent_dim: int = 0,
        latent_mode: str = "branch",
        latent_tile: int = 4,
        latent_window_channels: int = 0,
    ) -> None:
        super().__init__()
        self.use_images = use_images
        self.use_latent_branch = latent_dim > 0
        self.use_latent_window = latent_window_channels > 0
        self.latent_mode = latent_mode
        self.core = torch.nn.Sequential(
            torch.nn.Linear(core_dim, 24),
            torch.nn.GELU(),
            torch.nn.LayerNorm(24),
        )
        self.code = torch.nn.Sequential(
            torch.nn.Linear(code_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim // 2),
        )
        latent_out_dim = 0
        if self.use_latent_branch:
            latent_out_dim = hidden_dim // 2
            if latent_mode == "cnn":
                if latent_dim % (3 * latent_tile * latent_tile) != 0:
                    raise ValueError("latent_dim must be divisible by 3 * latent_tile * latent_tile")
                latent_channels = latent_dim // (3 * latent_tile * latent_tile)
                self.latent = torch.nn.Sequential(
                    torch.nn.Conv2d(3 * latent_channels, hidden_dim, kernel_size=3, padding=1),
                    torch.nn.GELU(),
                    torch.nn.Dropout2d(dropout),
                    torch.nn.Conv2d(hidden_dim, latent_out_dim, kernel_size=3, padding=1),
                    torch.nn.GELU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                    torch.nn.LayerNorm(latent_out_dim),
                )
            else:
                self.latent = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_dim, latent_out_dim),
                    torch.nn.GELU(),
                    torch.nn.LayerNorm(latent_out_dim),
                )
        image_dim = 0
        latent_window_dim = 0
        if self.use_latent_window:
            latent_window_dim = hidden_dim // 2
            self.latent_window = torch.nn.Sequential(
                torch.nn.Conv2d(latent_window_channels, hidden_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.Dropout2d(dropout),
                torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(hidden_dim, latent_window_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(latent_window_dim),
            )
        if use_images:
            image_dim = 48
            self.image = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, image_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
            )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(24 + hidden_dim // 2 + latent_out_dim + latent_window_dim + image_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        core: torch.Tensor,
        code: torch.Tensor,
        image: torch.Tensor | None,
        latent: torch.Tensor | None,
        latent_window: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [self.core(core), self.code(code)]
        if self.use_latent_branch:
            if latent is None:
                raise RuntimeError("latent tensor is required when latent branch is enabled")
            parts.append(self.latent(latent))
        if self.use_latent_window:
            if latent_window is None:
                raise RuntimeError("latent_window tensor is required when latent-window branch is enabled")
            parts.append(self.latent_window(latent_window))
        if self.use_images:
            if image is None:
                raise RuntimeError("image tensor is required when use_images=True")
            parts.append(self.image(image))
        return self.head(torch.cat(parts, dim=-1)).reshape(-1)


def masked_group_logits(row_logits: torch.Tensor, mask: torch.Tensor, group_count: int, max_candidates: int) -> torch.Tensor:
    logits = row_logits.reshape(group_count, max_candidates)
    return logits.masked_fill(~mask, -1.0e9)


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


def tensorize(
    data: dict[str, np.ndarray],
    device: torch.device,
    use_latents: bool,
    latent_mode: str,
    use_latent_windows: bool,
) -> dict[str, torch.Tensor]:
    code_parts = [
        data["code_center"].astype(np.float32),
        data["nearest_center"].astype(np.float32),
        data["delta_center"].astype(np.float32),
    ]
    latent = None
    if use_latents:
        required = ["latent_yhat", "latent_target", "latent_residual"]
        missing = [name for name in required if name not in data]
        if missing:
            raise SystemExit(f"--use-latents requested but missing arrays: {', '.join(missing)}")
        latent = np.concatenate([data[name].astype(np.float32) for name in required], axis=-1)
        if latent_mode == "concat":
            code_parts.append(latent)
    code = np.concatenate(
        code_parts,
        axis=-1,
    )
    tensors = {
        "group_keys": torch.from_numpy(data["group_keys"]).to(device),
        "mask": torch.from_numpy(data["mask"].astype(bool)).to(device),
        "labels": torch.from_numpy(data["labels"].astype(np.float32)).to(device),
        "is_nearest": torch.from_numpy(data["is_nearest"].astype(np.float32)).to(device),
        "parseq_delta": torch.from_numpy(data["parseq_delta"].astype(np.float32)).to(device),
        "tesseract_delta": torch.from_numpy(data["tesseract_delta"].astype(np.float32)).to(device),
        "core": torch.from_numpy(data["core_features"].astype(np.float32)).to(device),
        "code": torch.from_numpy(code.astype(np.float32)).to(device),
    }
    if latent is not None and latent_mode in {"branch", "cnn"}:
        tensors["latent"] = torch.from_numpy(latent.astype(np.float32)).to(device)
    if use_latent_windows:
        required = ["latent_window_yhat", "latent_window_target", "latent_window_residual"]
        missing = [name for name in required if name not in data]
        if missing:
            raise SystemExit(f"--use-latent-windows requested but missing arrays: {', '.join(missing)}")
        tensors["latent_window"] = torch.from_numpy(
            np.concatenate([data[name].astype(np.float32) for name in required], axis=2)
        ).to(device)
    if "image_triplet" in data:
        tensors["image"] = torch.from_numpy(data["image_triplet"].astype(np.float32)).to(device)
    return tensors


def normalize_features(
    tensors: dict[str, torch.Tensor],
    train_groups: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = tensors["mask"][train_groups]
    for name in ["core", "code", "latent"]:
        if name not in tensors:
            continue
        values = tensors[name][train_groups][mask]
        mean = values.mean(dim=0, keepdim=True)
        std = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
        tensors[name] = (tensors[name] - mean.reshape((1, 1, -1))) / std.reshape((1, 1, -1))
    if "latent_window" in tensors:
        values = tensors["latent_window"][train_groups][mask]
        mean = values.mean(dim=(0, 2, 3), keepdim=True)
        std = values.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(1.0e-6)
        tensors["latent_window"] = (tensors["latent_window"] - mean.reshape((1, 1, -1, 1, 1))) / std.reshape(
            (1, 1, -1, 1, 1)
        )
    return tensors["core"], tensors["code"]


def train_one(data_np: dict[str, np.ndarray], args: argparse.Namespace, model_seed: int) -> dict:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tensors = tensorize(
        data_np,
        device,
        use_latents=args.use_latents,
        latent_mode=args.latent_mode,
        use_latent_windows=args.use_latent_windows,
    )
    group_keys_np = data_np["group_keys"]
    train_groups = np.flatnonzero(group_keys_np[:, 1] != args.val_seed)
    val_groups = np.flatnonzero(group_keys_np[:, 1] == args.val_seed)
    normalize_features(tensors, train_groups)

    group_count, max_candidates = data_np["mask"].shape
    row_mask = tensors["mask"].reshape(-1)
    flat_core = tensors["core"].reshape(group_count * max_candidates, -1)[row_mask]
    flat_code = tensors["code"].reshape(group_count * max_candidates, -1)[row_mask]
    flat_image = None
    flat_latent = None
    flat_latent_window = None
    model_latent_dim = 0
    latent_window_channels = 0
    if args.use_images:
        if "image" not in tensors:
            raise SystemExit("--use-images requested but artifact has no image_triplet")
        flat_image = tensors["image"].reshape(group_count * max_candidates, 3, args.image_height, args.image_width)[
            row_mask
        ]
    if "latent" in tensors:
        flat_latent = tensors["latent"].reshape(group_count * max_candidates, -1)[row_mask]
        model_latent_dim = flat_latent.shape[-1]
        if args.latent_mode == "cnn":
            denom = 3 * args.latent_tile * args.latent_tile
            if model_latent_dim % denom != 0:
                raise SystemExit(f"latent dim {model_latent_dim} is not divisible by {denom}")
            latent_channels = model_latent_dim // denom
            flat_latent = flat_latent.reshape(-1, 3 * latent_channels, args.latent_tile, args.latent_tile)
    if "latent_window" in tensors:
        _, _, latent_window_channels, latent_window_height, latent_window_width = tensors["latent_window"].shape
        flat_latent_window = tensors["latent_window"].reshape(
            group_count * max_candidates,
            latent_window_channels,
            latent_window_height,
            latent_window_width,
        )[row_mask]
    row_to_flat = torch.full((group_count * max_candidates,), -1, dtype=torch.long, device=device)
    row_to_flat[row_mask] = torch.arange(int(row_mask.sum().item()), dtype=torch.long, device=device)

    model = CodeEffectReranker(
        core_dim=flat_core.shape[-1],
        code_dim=flat_code.shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_images=args.use_images,
        latent_dim=model_latent_dim,
        latent_mode=args.latent_mode,
        latent_tile=args.latent_tile,
        latent_window_channels=latent_window_channels,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_groups_t = torch.as_tensor(train_groups, dtype=torch.long, device=device)
    train_nearest = tensors["is_nearest"][train_groups_t].argmax(dim=1)
    if args.target_array == "labels":
        target_values = tensors["labels"]
    else:
        if args.target_array not in data_np:
            raise SystemExit(f"--target-array {args.target_array!r} not found in {args.features}")
        target_values = torch.from_numpy(data_np[args.target_array].astype(np.float32)).to(device)
    masked_targets = target_values[train_groups_t].masked_fill(~tensors["mask"][train_groups_t], -1.0e9)
    train_targets = masked_targets.argmax(dim=1)
    target_positive = masked_targets.max(dim=1).values > 0
    train_targets = torch.where(target_positive, train_targets, train_nearest)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        flat_logits = model(flat_core, flat_code, flat_image, flat_latent, flat_latent_window)
        all_logits_flat = torch.full((group_count * max_candidates,), -1.0e9, dtype=torch.float32, device=device)
        all_logits_flat[row_mask] = flat_logits
        logits = masked_group_logits(all_logits_flat, tensors["mask"], group_count, max_candidates)
        train_logits = logits[train_groups_t]
        loss = torch.nn.functional.cross_entropy(train_logits, train_targets)
        if args.no_change_weight > 0:
            target_is_nearest = train_targets == train_nearest
            if bool(target_is_nearest.any().item()):
                no_change_logits = train_logits[target_is_nearest]
                no_change_nearest = train_nearest[target_is_nearest]
                nearest_scores = no_change_logits.gather(1, no_change_nearest.reshape(-1, 1))
                non_nearest = tensors["is_nearest"][train_groups_t][target_is_nearest] < 0.5
                margin_terms = torch.nn.functional.softplus(no_change_logits - nearest_scores)[non_nearest]
                if margin_terms.numel():
                    loss = loss + args.no_change_weight * margin_terms.mean()
        if args.group_change_weight > 0:
            target_changed = (train_targets != train_nearest).float()
            nearest_scores = train_logits.gather(1, train_nearest.reshape(-1, 1)).reshape(-1)
            non_nearest_mask = tensors["is_nearest"][train_groups_t] < 0.5
            non_nearest_logits = train_logits.masked_fill(~non_nearest_mask, -1.0e9)
            best_non_nearest = non_nearest_logits.max(dim=1).values
            change_logits = best_non_nearest - nearest_scores
            loss = loss + args.group_change_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                change_logits,
                target_changed,
            )
        if args.parseq_harm_weight > 0:
            harm = tensors["parseq_delta"][train_groups_t] > 0
            if bool(harm.any().item()):
                loss = loss + args.parseq_harm_weight * torch.nn.functional.softplus(train_logits[harm]).mean()
        if args.tesseract_harm_weight > 0:
            non_nearest = tensors["is_nearest"][train_groups_t] < 0.5
            harm = tensors["tesseract_delta"][train_groups_t] > 0
            suppress = non_nearest & harm
            if bool(suppress.any().item()):
                loss = loss + args.tesseract_harm_weight * torch.nn.functional.softplus(train_logits[suppress]).mean()
        if args.tesseract_nonimprove_weight > 0:
            non_nearest = tensors["is_nearest"][train_groups_t] < 0.5
            non_improve = tensors["tesseract_delta"][train_groups_t] >= 0
            suppress = non_nearest & non_improve
            if bool(suppress.any().item()):
                loss = loss + args.tesseract_nonimprove_weight * torch.nn.functional.softplus(
                    train_logits[suppress]
                ).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        flat_logits = model(flat_core, flat_code, flat_image, flat_latent, flat_latent_window)
        flat_scores = torch.sigmoid(flat_logits)
        all_scores_flat = torch.full((group_count * max_candidates,), float("nan"), dtype=torch.float32, device=device)
        all_scores_flat[row_mask] = flat_scores
        scores = all_scores_flat.reshape(group_count, max_candidates).detach().cpu().numpy()

    valid = data_np["mask"].astype(bool)
    labels_np = data_np["labels"].astype(np.float32)
    train_valid = valid[train_groups]
    val_valid = valid[val_groups]
    train_scores = scores[train_groups][train_valid]
    train_labels = labels_np[train_groups][train_valid]
    val_scores = scores[val_groups][val_valid]
    val_labels = labels_np[val_groups][val_valid]

    selected_model = select_argmax(scores, valid, val_groups)
    selected_nearest = data_np["is_nearest"][val_groups].argmax(axis=1)
    selected_oracle = data_np["labels"][val_groups].argmax(axis=1)
    baseline = data_np.get("baseline_score")
    policies = {
        "nearest": policy_metrics(data_np, val_groups, selected_nearest),
        "oracle": policy_metrics(data_np, val_groups, selected_oracle),
        "model_argmax": policy_metrics(data_np, val_groups, selected_model),
    }
    if baseline is not None and np.isfinite(baseline[val_groups][val_valid]).any():
        selected_baseline = select_argmax(baseline, valid, val_groups)
        policies["baseline_score_argmax"] = policy_metrics(data_np, val_groups, selected_baseline)

    if args.score_output is not None:
        args.score_output.parent.mkdir(parents=True, exist_ok=True)
        with args.score_output.open("a", encoding="utf-8") as handle:
            for group_index in range(group_count):
                real_seed, seed, source_index, candidate_index = [int(value) for value in data_np["group_keys"][group_index]]
                for row_position in range(max_candidates):
                    if not valid[group_index, row_position]:
                        continue
                    handle.write(
                        json.dumps(
                            {
                                "model_seed": model_seed,
                                "group_index": group_index,
                                "row_position": row_position,
                                "real_seed": real_seed,
                                "seed": seed,
                                "source_index": source_index,
                                "candidate_index": candidate_index,
                                "code_index": int(data_np["code_indices"][group_index, row_position]),
                                "nearest_code": int(data_np["nearest_indices"][group_index, row_position]),
                                "topk_rank": int(data_np["topk_ranks"][group_index, row_position]),
                                "is_nearest": int(data_np["is_nearest"][group_index, row_position]),
                                "label_assignment_oracle_choice": int(labels_np[group_index, row_position]),
                                "parseq_delta_vs_nearest": int(data_np["parseq_delta"][group_index, row_position]),
                                "tesseract_delta_vs_nearest": int(data_np["tesseract_delta"][group_index, row_position]),
                                "score": float(scores[group_index, row_position]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    return {
        "model_seed": model_seed,
        "device": str(device),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "train_auc": roc_auc(train_scores, train_labels),
        "train_ap": average_precision(train_scores, train_labels),
        "val_auc": roc_auc(val_scores, val_labels),
        "val_ap": average_precision(val_scores, val_labels),
        "policy_metrics": policies,
    }


def aggregate(results: list[dict]) -> dict:
    output = {}
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
    parser.add_argument(
        "--target-array",
        default="labels",
        help="Group target array in the NPZ. Use labels for assignment oracle choice.",
    )
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-change-weight", type=float, default=0.5)
    parser.add_argument("--group-change-weight", type=float, default=0.0)
    parser.add_argument("--parseq-harm-weight", type=float, default=0.1)
    parser.add_argument("--tesseract-harm-weight", type=float, default=0.0)
    parser.add_argument("--tesseract-nonimprove-weight", type=float, default=0.0)
    parser.add_argument("--use-images", action="store_true")
    parser.add_argument("--use-latents", action="store_true")
    parser.add_argument("--use-latent-windows", action="store_true")
    parser.add_argument("--latent-mode", choices=["concat", "branch", "cnn"], default="concat")
    parser.add_argument("--latent-tile", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.model_seed is None:
        args.model_seed = [0]
    if args.score_output is not None and args.score_output.exists():
        args.score_output.unlink()

    data_np = {key: value for key, value in np.load(args.features).items()}
    results = [train_one(data_np, args, model_seed=seed) for seed in args.model_seed]
    output = {
        "description": "Compact tensor-backed code-effect assignment reranker.",
        "features": str(args.features),
        "config": {
            "val_seed": args.val_seed,
            "target_array": args.target_array,
            "model_seeds": args.model_seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "no_change_weight": args.no_change_weight,
            "group_change_weight": args.group_change_weight,
            "parseq_harm_weight": args.parseq_harm_weight,
            "tesseract_harm_weight": args.tesseract_harm_weight,
            "tesseract_nonimprove_weight": args.tesseract_nonimprove_weight,
            "use_images": args.use_images,
            "use_latents": args.use_latents,
            "use_latent_windows": args.use_latent_windows,
            "latent_mode": args.latent_mode,
            "score_output": str(args.score_output) if args.score_output is not None else None,
        },
        "per_run": results,
        "aggregate": aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
