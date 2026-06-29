#!/usr/bin/env python3
"""Train a group/no-op aware assignment head over nearest-plus-topk tensors."""

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


def select_group_policy(
    row_scores: np.ndarray,
    group_probs: np.ndarray,
    mask: np.ndarray,
    nearest_mask: np.ndarray,
    group_indices: np.ndarray,
    threshold: float,
    parseq_delta: np.ndarray | None = None,
    parseq_safe: bool = False,
) -> np.ndarray:
    selected = []
    for group_index in group_indices:
        valid = mask[group_index] & np.isfinite(row_scores[group_index])
        nearest_positions = np.flatnonzero(nearest_mask[group_index] & valid)
        if nearest_positions.size == 0:
            selected.append(int(np.flatnonzero(valid)[0]))
            continue
        nearest_pos = int(nearest_positions[0])
        if float(group_probs[group_index]) < threshold:
            selected.append(nearest_pos)
            continue
        candidate_valid = valid & (~nearest_mask[group_index])
        if parseq_safe and parseq_delta is not None:
            candidate_valid &= parseq_delta[group_index] <= 0
        if not candidate_valid.any():
            selected.append(nearest_pos)
            continue
        candidate_scores = row_scores[group_index].copy()
        candidate_scores[~candidate_valid] = -np.inf
        selected.append(int(np.argmax(candidate_scores)))
    return np.asarray(selected, dtype=np.int64)


class GroupNoopAssignmentModel(torch.nn.Module):
    def __init__(
        self,
        core_dim: int,
        code_dim: int,
        window_channels: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.use_window = window_channels > 0
        code_out = hidden_dim // 2
        window_out = hidden_dim // 2 if self.use_window else 0
        self.core = torch.nn.Sequential(torch.nn.Linear(core_dim, 24), torch.nn.GELU(), torch.nn.LayerNorm(24))
        self.code = torch.nn.Sequential(
            torch.nn.Linear(code_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, code_out),
            torch.nn.GELU(),
            torch.nn.LayerNorm(code_out),
        )
        if self.use_window:
            self.window = torch.nn.Sequential(
                torch.nn.Conv2d(window_channels, hidden_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.Dropout2d(dropout),
                torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(hidden_dim, window_out, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.LayerNorm(window_out),
            )
        row_in = 24 + code_out + window_out
        self.row_fuse = torch.nn.Sequential(
            torch.nn.Linear(row_in, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.row_head = torch.nn.Linear(hidden_dim, 1)
        self.group_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        core: torch.Tensor,
        code: torch.Tensor,
        window: torch.Tensor | None,
        mask: torch.Tensor,
        nearest_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_count, max_candidates = mask.shape
        flat_core = core.reshape(group_count * max_candidates, -1)
        flat_code = code.reshape(group_count * max_candidates, -1)
        parts = [self.core(flat_core), self.code(flat_code)]
        if self.use_window:
            if window is None:
                raise RuntimeError("window tensor is required when window_channels > 0")
            _, _, channels, height, width = window.shape
            flat_window = window.reshape(group_count * max_candidates, channels, height, width)
            parts.append(self.window(flat_window))
        row_emb = self.row_fuse(torch.cat(parts, dim=-1)).reshape(group_count, max_candidates, -1)
        row_logits = self.row_head(row_emb).reshape(group_count, max_candidates)
        row_logits = row_logits.masked_fill(~mask, -1.0e9)

        nearest_pos = nearest_mask.float().argmax(dim=1)
        batch = torch.arange(group_count, device=mask.device)
        nearest_emb = row_emb[batch, nearest_pos]
        nearest_score = row_logits[batch, nearest_pos]

        non_nearest = mask & (~nearest_mask)
        non_nearest_logits = row_logits.masked_fill(~non_nearest, -1.0e9)
        best_non_nearest_pos = non_nearest_logits.argmax(dim=1)
        best_non_nearest_emb = row_emb[batch, best_non_nearest_pos]
        best_non_nearest_score = non_nearest_logits[batch, best_non_nearest_pos]

        valid_counts = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_emb = (row_emb * mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts
        score_features = torch.stack(
            [
                best_non_nearest_score - nearest_score,
                torch.sigmoid(best_non_nearest_score),
                torch.sigmoid(nearest_score),
            ],
            dim=1,
        )
        group_logits = self.group_head(
            torch.cat([nearest_emb, best_non_nearest_emb, pooled_emb, score_features], dim=1)
        ).reshape(-1)
        return row_logits, group_logits


def tensorize(data: dict[str, np.ndarray], device: torch.device, use_window: bool) -> dict[str, torch.Tensor]:
    code = np.concatenate(
        [
            data["code_center"].astype(np.float32),
            data["nearest_center"].astype(np.float32),
            data["delta_center"].astype(np.float32),
        ],
        axis=-1,
    )
    tensors = {
        "mask": torch.from_numpy(data["mask"].astype(bool)).to(device),
        "is_nearest": torch.from_numpy(data["is_nearest"].astype(bool)).to(device),
        "labels": torch.from_numpy(data["labels"].astype(np.float32)).to(device),
        "parseq_delta": torch.from_numpy(data["parseq_delta"].astype(np.float32)).to(device),
        "tesseract_delta": torch.from_numpy(data["tesseract_delta"].astype(np.float32)).to(device),
        "core": torch.from_numpy(data["core_features"].astype(np.float32)).to(device),
        "code": torch.from_numpy(code.astype(np.float32)).to(device),
    }
    if use_window:
        window = np.concatenate(
            [
                data["latent_window_yhat"].astype(np.float32),
                data["latent_window_target"].astype(np.float32),
                data["latent_window_residual"].astype(np.float32),
            ],
            axis=2,
        )
        tensors["window"] = torch.from_numpy(window.astype(np.float32)).to(device)
    return tensors


def normalize_features(tensors: dict[str, torch.Tensor], train_groups: np.ndarray) -> None:
    mask = tensors["mask"][train_groups]
    for name in ["core", "code"]:
        values = tensors[name][train_groups][mask]
        mean = values.mean(dim=0, keepdim=True)
        std = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
        tensors[name] = (tensors[name] - mean.reshape((1, 1, -1))) / std.reshape((1, 1, -1))
    if "window" in tensors:
        values = tensors["window"][train_groups][mask]
        mean = values.mean(dim=(0, 2, 3), keepdim=True)
        std = values.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(1.0e-6)
        tensors["window"] = (tensors["window"] - mean.reshape((1, 1, -1, 1, 1))) / std.reshape(
            (1, 1, -1, 1, 1)
        )


def threshold_rows(
    data_np: dict[str, np.ndarray],
    row_scores: np.ndarray,
    group_probs: np.ndarray,
    group_indices: np.ndarray,
    thresholds: np.ndarray,
    parseq_safe: bool,
) -> list[dict]:
    rows = []
    mask = data_np["mask"].astype(bool)
    nearest_mask = data_np["is_nearest"].astype(bool)
    for threshold in thresholds:
        selected = select_group_policy(
            row_scores,
            group_probs,
            mask,
            nearest_mask,
            group_indices,
            float(threshold),
            parseq_delta=data_np["parseq_delta"],
            parseq_safe=parseq_safe,
        )
        rows.append({"threshold": float(threshold), "metrics": policy_metrics(data_np, group_indices, selected)})
    return rows


def choose_train_threshold(
    rows: list[dict],
    false_change_max: int | None,
    wrong_change_max: int | None,
    tesseract_worsen_max: int | None,
) -> dict:
    candidates = [row for row in rows if row["metrics"]["parseq_delta_sum"] <= 0]
    if false_change_max is not None:
        strict = [row for row in candidates if row["metrics"]["false_change_groups"] <= false_change_max]
        if strict:
            candidates = strict
    if wrong_change_max is not None:
        strict = [row for row in candidates if row["metrics"]["wrong_change_groups"] <= wrong_change_max]
        if strict:
            candidates = strict
    if tesseract_worsen_max is not None:
        strict = [row for row in candidates if row["metrics"]["tesseract_worsen_groups"] <= tesseract_worsen_max]
        if strict:
            candidates = strict
    if not candidates:
        candidates = rows
    return min(
        candidates,
        key=lambda row: (
            row["metrics"]["tesseract_delta_sum"],
            row["metrics"]["parseq_delta_sum"],
            row["metrics"]["false_change_groups"],
            row["metrics"]["wrong_change_groups"],
            -row["metrics"]["exact_oracle_groups"],
        ),
    )


def build_matched_group_pairs(
    data_np: dict[str, np.ndarray],
    train_groups: np.ndarray,
    *,
    pairs_per_positive: int,
    min_tesseract_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    if pairs_per_positive <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    labels = data_np["labels"]
    nearest = data_np["is_nearest"].astype(bool)
    tesseract = data_np["tesseract_delta"]
    oracle_pos = labels.argmax(axis=1)
    nearest_pos = nearest.argmax(axis=1)
    oracle_changed = oracle_pos != nearest_pos
    oracle_tess = tesseract[np.arange(labels.shape[0]), oracle_pos]
    positives = [
        int(group_index)
        for group_index in train_groups
        if oracle_changed[group_index] and oracle_tess[group_index] <= -float(min_tesseract_gain)
    ]
    negatives = [int(group_index) for group_index in train_groups if not oracle_changed[group_index]]
    if not positives or not negatives:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    # Difficulty features are available in the nearest row core features:
    # reference length, PARSeq nearest distance, and Tesseract nearest distance.
    core = data_np["core_features"].astype(np.float32)
    nearest_core = core[np.arange(core.shape[0]), nearest_pos]
    difficulty = nearest_core[:, [4, 5, 6]].astype(np.float32)
    train_difficulty = difficulty[train_groups]
    mean = train_difficulty.mean(axis=0, keepdims=True)
    std = train_difficulty.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    difficulty = (difficulty - mean) / std

    negative_arr = np.asarray(negatives, dtype=np.int64)
    pair_pos: list[int] = []
    pair_neg: list[int] = []
    for positive in positives:
        distances = np.sum((difficulty[negative_arr] - difficulty[positive]) ** 2, axis=1)
        order = np.argsort(distances, kind="mergesort")[:pairs_per_positive]
        for neg_index in negative_arr[order]:
            pair_pos.append(positive)
            pair_neg.append(int(neg_index))
    return np.asarray(pair_pos, dtype=np.int64), np.asarray(pair_neg, dtype=np.int64)


def write_score_rows(
    path: Path,
    data_np: dict[str, np.ndarray],
    row_scores: np.ndarray,
    group_probs: np.ndarray,
    val_seed: int,
    model_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    valid = data_np["mask"].astype(bool)
    labels = data_np["labels"].astype(np.float32)
    with path.open(mode, encoding="utf-8") as handle:
        for group_index in range(valid.shape[0]):
            real_seed, seed, source_index, candidate_index = [int(value) for value in data_np["group_keys"][group_index]]
            for row_position in range(valid.shape[1]):
                if not valid[group_index, row_position]:
                    continue
                row = {
                    "val_seed": int(val_seed),
                    "model_seed": int(model_seed),
                    "group_index": int(group_index),
                    "row_position": int(row_position),
                    "real_seed": real_seed,
                    "seed": seed,
                    "source_index": source_index,
                    "candidate_index": candidate_index,
                    "code_index": int(data_np["code_indices"][group_index, row_position]),
                    "nearest_code": int(data_np["nearest_indices"][group_index, row_position]),
                    "topk_rank": int(data_np["topk_ranks"][group_index, row_position]),
                    "is_nearest": int(data_np["is_nearest"][group_index, row_position]),
                    "label_assignment_oracle_choice": int(labels[group_index, row_position]),
                    "parseq_delta_vs_nearest": int(data_np["parseq_delta"][group_index, row_position]),
                    "tesseract_delta_vs_nearest": int(data_np["tesseract_delta"][group_index, row_position]),
                    "score": float(row_scores[group_index, row_position]),
                    "group_change_prob": float(group_probs[group_index]),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_policy_rows(
    path: Path,
    data_np: dict[str, np.ndarray],
    row_scores: np.ndarray,
    group_probs: np.ndarray,
    selected: np.ndarray,
    group_indices: np.ndarray,
    val_seed: int,
    model_seed: int,
    policy_name: str,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for group_index, row_position in zip(group_indices, selected):
            real_seed, seed, source_index, candidate_index = [int(value) for value in data_np["group_keys"][group_index]]
            code_index = int(data_np["code_indices"][group_index, row_position])
            nearest_code = int(data_np["nearest_indices"][group_index, row_position])
            row = {
                "source_index": source_index,
                "candidate_index": candidate_index,
                "assignment_code_index": code_index,
                "code_index": code_index,
                "nearest_code": nearest_code,
                "topk_rank": int(data_np["topk_ranks"][group_index, row_position]),
                "selected_by_policy": True,
                "selector_score": float(row_scores[group_index, row_position]),
                "group_change_prob": float(group_probs[group_index]),
                "score_val_seed": int(val_seed),
                "val_seed": int(val_seed),
                "score_model_seed": int(model_seed),
                "score_model_seeds": [int(model_seed)],
                "score_count": 1,
                "score_min": float(row_scores[group_index, row_position]),
                "score_max": float(row_scores[group_index, row_position]),
                "seed": seed,
                "real_seed": real_seed,
                "policy": policy_name,
                "policy_threshold": float(threshold),
                "parseq_delta_vs_nearest": int(data_np["parseq_delta"][group_index, row_position]),
                "tesseract_delta_vs_nearest": int(data_np["tesseract_delta"][group_index, row_position]),
                "label_assignment_oracle_choice": int(data_np["labels"][group_index, row_position]),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_one(data_np: dict[str, np.ndarray], args: argparse.Namespace, model_seed: int) -> dict:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.cuda_required and device.type != "cuda":
        raise SystemExit("CUDA required but unavailable")

    tensors = tensorize(data_np, device, use_window=not args.disable_window)
    group_keys_np = data_np["group_keys"]
    val_key_value = args.val_seed if args.val_key_value is None else args.val_key_value
    if args.split_mode == "key":
        train_groups = np.flatnonzero(group_keys_np[:, args.split_key_index] != val_key_value)
        val_groups = np.flatnonzero(group_keys_np[:, args.split_key_index] == val_key_value)
    elif args.split_mode == "source_modulo":
        if args.val_source_modulo is None or args.val_source_remainder is None:
            raise SystemExit("--split-mode source_modulo requires --val-source-modulo and --val-source-remainder")
        source_ids = group_keys_np[:, 2]
        val_mask = (source_ids % args.val_source_modulo) == args.val_source_remainder
        train_groups = np.flatnonzero(~val_mask)
        val_groups = np.flatnonzero(val_mask)
    else:
        raise ValueError(f"unknown split mode: {args.split_mode}")
    if len(train_groups) == 0 or len(val_groups) == 0:
        raise SystemExit(
            f"empty train/val split for split_mode={args.split_mode}, "
            f"split_key_index={args.split_key_index}, val_key_value={val_key_value}, "
            f"val_source_modulo={args.val_source_modulo}, val_source_remainder={args.val_source_remainder}"
        )
    normalize_features(tensors, train_groups)

    group_count, max_candidates = data_np["mask"].shape
    model = GroupNoopAssignmentModel(
        core_dim=tensors["core"].shape[-1],
        code_dim=tensors["code"].shape[-1],
        window_channels=tensors["window"].shape[2] if "window" in tensors else 0,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_groups_t = torch.as_tensor(train_groups, dtype=torch.long, device=device)
    train_targets = tensors["labels"][train_groups_t].argmax(dim=1)
    train_nearest = tensors["is_nearest"][train_groups_t].float().argmax(dim=1)
    train_changed = (train_targets != train_nearest).float()
    batch = torch.arange(len(train_groups), device=device)
    oracle_tesseract_delta = tensors["tesseract_delta"][train_groups_t][batch, train_targets]
    if args.group_target == "any_change":
        train_group_targets = train_changed
    elif args.group_target == "strong_tesseract":
        train_group_targets = (oracle_tesseract_delta <= args.strong_tesseract_delta).float()
    else:
        raise ValueError(f"unknown group target: {args.group_target}")
    positive = float(train_group_targets.sum().item())
    negative = float(train_group_targets.numel() - positive)
    group_pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32, device=device)
    all_targets = tensors["labels"].argmax(dim=1)
    all_nearest = tensors["is_nearest"].float().argmax(dim=1)
    pair_pos_np, pair_neg_np = build_matched_group_pairs(
        data_np,
        train_groups,
        pairs_per_positive=args.matched_pairwise_pairs_per_positive,
        min_tesseract_gain=args.hard_positive_min_tesseract_gain,
    )
    pair_pos_t = torch.as_tensor(pair_pos_np, dtype=torch.long, device=device)
    pair_neg_t = torch.as_tensor(pair_neg_np, dtype=torch.long, device=device)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        row_logits, group_logits = model(
            tensors["core"],
            tensors["code"],
            tensors.get("window"),
            tensors["mask"],
            tensors["is_nearest"],
        )
        train_logits = row_logits[train_groups_t]
        train_nearest_logits = train_logits[batch, train_nearest]
        train_oracle_logits = train_logits[batch, train_targets]
        if args.candidate_only_row_loss:
            candidate_rows = tensors["mask"][train_groups_t] & (~tensors["is_nearest"][train_groups_t])
            candidate_labels = tensors["labels"][train_groups_t][candidate_rows]
            candidate_logits = train_logits[candidate_rows]
            pos_weight = None
            if args.candidate_row_pos_weight:
                positives = candidate_labels.sum().clamp_min(1.0)
                negatives = (candidate_labels.numel() - candidate_labels.sum()).clamp_min(1.0)
                pos_weight = negatives / positives
            row_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                candidate_logits,
                candidate_labels,
                pos_weight=pos_weight,
            )
        elif args.row_target == "oracle_choice":
            row_loss = torch.nn.functional.cross_entropy(train_logits, train_targets)
        elif args.row_target == "tesseract_reward_softmax":
            valid_train = tensors["mask"][train_groups_t]
            safe_train = tensors["parseq_delta"][train_groups_t] <= 0
            reward = (-tensors["tesseract_delta"][train_groups_t]).clamp_min(0.0)
            reward = reward * valid_train.float() * safe_train.float()
            reward = reward.masked_fill(tensors["is_nearest"][train_groups_t], 0.0)
            reward_sum = reward.sum(dim=1, keepdim=True)
            target_probs = reward / reward_sum.clamp_min(1.0e-6)
            nearest_one_hot = torch.nn.functional.one_hot(train_nearest, num_classes=max_candidates).float()
            target_probs = torch.where(reward_sum > 0, target_probs, nearest_one_hot)
            log_probs = torch.nn.functional.log_softmax(train_logits, dim=1)
            row_loss = -(target_probs * log_probs).sum(dim=1).mean()
        elif args.row_target == "safe_tesseract_soft_bce":
            valid_train = tensors["mask"][train_groups_t]
            nearest_train = tensors["is_nearest"][train_groups_t].bool()
            non_nearest_train = valid_train & (~nearest_train)
            safe_train = tensors["parseq_delta"][train_groups_t] <= 0
            tess_delta = tensors["tesseract_delta"][train_groups_t]
            gain = (-tess_delta / max(args.utility_gain_scale, 1.0e-6)).clamp(0.0, 1.0)
            target_probs = torch.zeros_like(train_logits)
            target_probs = target_probs.masked_fill(nearest_train, args.nearest_soft_label)
            target_probs = torch.where(non_nearest_train & safe_train & (tess_delta < 0), gain, target_probs)
            target_probs = torch.where(
                non_nearest_train & safe_train & (tess_delta == 0),
                torch.full_like(target_probs, args.safe_nonimprove_soft_label),
                target_probs,
            )
            row_weights = torch.ones_like(train_logits)
            row_weights = torch.where(
                non_nearest_train & safe_train & (tess_delta < 0),
                torch.full_like(row_weights, args.soft_positive_weight),
                row_weights,
            )
            row_weights = torch.where(
                nearest_train,
                torch.full_like(row_weights, args.soft_nearest_weight),
                row_weights,
            )
            row_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                train_logits[valid_train],
                target_probs[valid_train],
                weight=row_weights[valid_train],
            )
        else:
            raise ValueError(f"unknown row target: {args.row_target}")
        loss = args.listwise_weight * row_loss
        loss = loss + args.group_change_weight * torch.nn.functional.binary_cross_entropy_with_logits(
            group_logits[train_groups_t],
            train_group_targets,
            pos_weight=group_pos_weight if args.group_pos_weight else None,
        )
        no_headroom = train_group_targets < 0.5
        if args.false_change_margin_weight > 0 and bool(no_headroom.any().item()):
            loss = loss + args.false_change_margin_weight * torch.nn.functional.softplus(
                group_logits[train_groups_t][no_headroom]
            ).mean()
        non_nearest = tensors["is_nearest"][train_groups_t] < 0.5
        if args.parseq_harm_weight > 0:
            harm = non_nearest & (tensors["parseq_delta"][train_groups_t] > 0)
            if bool(harm.any().item()):
                loss = loss + args.parseq_harm_weight * torch.nn.functional.softplus(train_logits[harm]).mean()
        if args.tesseract_nonimprove_weight > 0:
            non_improve = non_nearest & (tensors["tesseract_delta"][train_groups_t] >= 0)
            if bool(non_improve.any().item()):
                loss = loss + args.tesseract_nonimprove_weight * torch.nn.functional.softplus(
                    train_logits[non_improve]
                ).mean()
        if args.rank0_row_margin_weight > 0:
            no_headroom_rows = train_changed < 0.5
            non_nearest_rows = tensors["mask"][train_groups_t] & (~tensors["is_nearest"][train_groups_t])
            best_non_nearest_logits = train_logits.masked_fill(~non_nearest_rows, -1.0e9).max(dim=1).values
            finite_non_nearest = best_non_nearest_logits > -1.0e8
            rank0_margin_rows = no_headroom_rows & finite_non_nearest
            if bool(rank0_margin_rows.any().item()):
                loss = loss + args.rank0_row_margin_weight * torch.nn.functional.softplus(
                    best_non_nearest_logits[rank0_margin_rows]
                    - train_nearest_logits[rank0_margin_rows]
                    + args.tail_margin
                ).mean()
        if args.hard_positive_row_margin_weight > 0 or args.hard_positive_group_margin_weight > 0:
            hard_positive = (train_changed > 0.5) & (
                oracle_tesseract_delta <= -float(args.hard_positive_min_tesseract_gain)
            )
            if bool(hard_positive.any().item()):
                if args.hard_positive_row_margin_weight > 0:
                    loss = loss + args.hard_positive_row_margin_weight * torch.nn.functional.softplus(
                        train_nearest_logits[hard_positive]
                        - train_oracle_logits[hard_positive]
                        + args.tail_margin
                    ).mean()
                if args.hard_positive_group_margin_weight > 0:
                    loss = loss + args.hard_positive_group_margin_weight * torch.nn.functional.softplus(
                        -group_logits[train_groups_t][hard_positive] + args.tail_margin
                    ).mean()
        if (
            pair_pos_t.numel() > 0
            and (args.matched_pairwise_group_weight > 0 or args.matched_pairwise_row_weight > 0)
        ):
            pos_oracle = all_targets[pair_pos_t]
            pos_nearest = all_nearest[pair_pos_t]
            neg_nearest = all_nearest[pair_neg_t]
            pair_range = torch.arange(pair_pos_t.numel(), device=device)
            if args.matched_pairwise_group_weight > 0:
                loss = loss + args.matched_pairwise_group_weight * torch.nn.functional.softplus(
                    group_logits[pair_neg_t] - group_logits[pair_pos_t] + args.matched_pairwise_margin
                ).mean()
            if args.matched_pairwise_row_weight > 0:
                pos_change_score = (
                    row_logits[pair_pos_t, pos_oracle] - row_logits[pair_pos_t, pos_nearest]
                )
                neg_non_nearest = tensors["mask"][pair_neg_t] & (~tensors["is_nearest"][pair_neg_t])
                neg_best = row_logits[pair_neg_t].masked_fill(~neg_non_nearest, -1.0e9).max(dim=1).values
                neg_change_score = neg_best - row_logits[pair_neg_t, neg_nearest]
                finite = neg_best > -1.0e8
                if bool(finite.any().item()):
                    loss = loss + args.matched_pairwise_row_weight * torch.nn.functional.softplus(
                        neg_change_score[finite] - pos_change_score[finite] + args.matched_pairwise_margin
                    ).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        row_logits, group_logits = model(
            tensors["core"],
            tensors["code"],
            tensors.get("window"),
            tensors["mask"],
            tensors["is_nearest"],
        )
        row_scores = torch.sigmoid(row_logits).detach().cpu().numpy()
        group_probs = torch.sigmoid(group_logits).detach().cpu().numpy()
    row_scores[~data_np["mask"].astype(bool)] = np.nan

    valid = data_np["mask"].astype(bool)
    labels_np = data_np["labels"].astype(np.float32)
    train_scores = row_scores[train_groups][valid[train_groups]]
    train_labels = labels_np[train_groups][valid[train_groups]]
    val_scores = row_scores[val_groups][valid[val_groups]]
    val_labels = labels_np[val_groups][valid[val_groups]]

    group_change_labels = (
        data_np["labels"].argmax(axis=1) != data_np["is_nearest"].astype(bool).argmax(axis=1)
    ).astype(np.float32)
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    train_rows = threshold_rows(data_np, row_scores, group_probs, train_groups, thresholds, args.parseq_safe)
    false_change_max = args.train_false_change_max
    if false_change_max is None and args.train_false_change_rate_max is not None:
        false_change_max = max(0, int(round(len(train_groups) * args.train_false_change_rate_max)))
    selected_train_threshold = choose_train_threshold(
        train_rows,
        false_change_max,
        args.train_wrong_change_max,
        args.train_tesseract_worsen_max,
    )
    threshold = float(selected_train_threshold["threshold"])

    selected_nearest = data_np["is_nearest"][val_groups].argmax(axis=1)
    selected_oracle = data_np["labels"][val_groups].argmax(axis=1)
    selected_row_argmax = select_argmax(row_scores, valid, val_groups)
    selected_default = select_group_policy(
        row_scores,
        group_probs,
        valid,
        data_np["is_nearest"].astype(bool),
        val_groups,
        args.default_threshold,
        parseq_delta=data_np["parseq_delta"],
        parseq_safe=args.parseq_safe,
    )
    selected_train_tuned = select_group_policy(
        row_scores,
        group_probs,
        valid,
        data_np["is_nearest"].astype(bool),
        val_groups,
        threshold,
        parseq_delta=data_np["parseq_delta"],
        parseq_safe=args.parseq_safe,
    )
    baseline = data_np.get("baseline_score")
    policies = {
        "nearest": policy_metrics(data_np, val_groups, selected_nearest),
        "oracle": policy_metrics(data_np, val_groups, selected_oracle),
        "row_argmax": policy_metrics(data_np, val_groups, selected_row_argmax),
        f"group_threshold_{args.default_threshold:.3f}": policy_metrics(data_np, val_groups, selected_default),
        "group_train_tuned": policy_metrics(data_np, val_groups, selected_train_tuned),
    }
    if baseline is not None and np.isfinite(baseline[val_groups][valid[val_groups]]).any():
        selected_baseline = select_argmax(baseline, valid, val_groups)
        policies["baseline_score_argmax"] = policy_metrics(data_np, val_groups, selected_baseline)

    if args.score_output is not None:
        write_score_rows(args.score_output, data_np, row_scores, group_probs, args.val_seed, model_seed)
    if args.policy_output_dir is not None:
        out_path = args.policy_output_dir / (
            f"{args.policy_output_prefix}_val{args.val_seed}_model{model_seed}_train_tuned_policy.jsonl"
        )
        write_policy_rows(
            out_path,
            data_np,
            row_scores,
            group_probs,
            selected_train_tuned,
            val_groups,
            args.val_seed,
            model_seed,
            "group_train_tuned",
            threshold,
        )

    return {
        "model_seed": model_seed,
        "device": str(device),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "split_mode": args.split_mode,
        "split_key_index": int(args.split_key_index),
        "val_key_value": int(val_key_value),
        "val_source_modulo": args.val_source_modulo,
        "val_source_remainder": args.val_source_remainder,
        "train_auc": roc_auc(train_scores, train_labels),
        "train_ap": average_precision(train_scores, train_labels),
        "val_auc": roc_auc(val_scores, val_labels),
        "val_ap": average_precision(val_scores, val_labels),
        "train_group_auc": roc_auc(group_probs[train_groups], group_change_labels[train_groups]),
        "val_group_auc": roc_auc(group_probs[val_groups], group_change_labels[val_groups]),
        "matched_pair_count": int(pair_pos_np.size),
        "selected_train_threshold": selected_train_threshold,
        "policy_metrics": policies,
    }


def aggregate(results: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for metric in ["train_auc", "train_ap", "val_auc", "val_ap", "train_group_auc", "val_group_auc"]:
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
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--split-mode", choices=["key", "source_modulo"], default="key")
    parser.add_argument("--split-key-index", type=int, default=1)
    parser.add_argument("--val-key-value", type=int)
    parser.add_argument("--val-source-modulo", type=int)
    parser.add_argument("--val-source-remainder", type=int)
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument(
        "--row-target",
        choices=["oracle_choice", "tesseract_reward_softmax", "safe_tesseract_soft_bce"],
        default="oracle_choice",
    )
    parser.add_argument("--utility-gain-scale", type=float, default=2.0)
    parser.add_argument("--nearest-soft-label", type=float, default=0.5)
    parser.add_argument("--safe-nonimprove-soft-label", type=float, default=0.1)
    parser.add_argument("--soft-positive-weight", type=float, default=4.0)
    parser.add_argument("--soft-nearest-weight", type=float, default=1.0)
    parser.add_argument("--candidate-only-row-loss", action="store_true")
    parser.add_argument("--candidate-row-pos-weight", action="store_true")
    parser.add_argument("--group-change-weight", type=float, default=0.5)
    parser.add_argument("--false-change-margin-weight", type=float, default=1.0)
    parser.add_argument("--parseq-harm-weight", type=float, default=0.2)
    parser.add_argument("--tesseract-nonimprove-weight", type=float, default=0.2)
    parser.add_argument("--rank0-row-margin-weight", type=float, default=0.0)
    parser.add_argument("--hard-positive-row-margin-weight", type=float, default=0.0)
    parser.add_argument("--hard-positive-group-margin-weight", type=float, default=0.0)
    parser.add_argument("--hard-positive-min-tesseract-gain", type=float, default=1.0)
    parser.add_argument("--matched-pairwise-group-weight", type=float, default=0.0)
    parser.add_argument("--matched-pairwise-row-weight", type=float, default=0.0)
    parser.add_argument("--matched-pairwise-margin", type=float, default=1.0)
    parser.add_argument("--matched-pairwise-pairs-per-positive", type=int, default=0)
    parser.add_argument("--tail-margin", type=float, default=1.0)
    parser.add_argument("--group-pos-weight", action="store_true")
    parser.add_argument("--group-target", choices=["any_change", "strong_tesseract"], default="any_change")
    parser.add_argument("--strong-tesseract-delta", type=float, default=-2.0)
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=1.0)
    parser.add_argument("--threshold-steps", type=int, default=101)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--train-false-change-max", type=int)
    parser.add_argument("--train-false-change-rate-max", type=float, default=0.02)
    parser.add_argument("--train-wrong-change-max", type=int)
    parser.add_argument("--train-tesseract-worsen-max", type=int)
    parser.add_argument("--parseq-safe", action="store_true")
    parser.add_argument("--disable-window", action="store_true")
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--policy-output-dir", type=Path)
    parser.add_argument("--policy-output-prefix", default="group_noop_window")
    parser.add_argument("--cuda-required", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.model_seed is None:
        args.model_seed = [0]
    if args.score_output is not None and args.score_output.exists():
        args.score_output.unlink()

    data_np = {key: value for key, value in np.load(args.features).items()}
    results = [train_one(data_np, args, model_seed=seed) for seed in args.model_seed]
    output = {
        "description": "Group/no-op aware assignment head with structured local latent windows.",
        "features": str(args.features),
        "config": {
            "val_seed": args.val_seed,
            "split_mode": args.split_mode,
            "split_key_index": args.split_key_index,
            "val_key_value": args.val_key_value,
            "val_source_modulo": args.val_source_modulo,
            "val_source_remainder": args.val_source_remainder,
            "model_seeds": args.model_seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "listwise_weight": args.listwise_weight,
            "row_target": args.row_target,
            "utility_gain_scale": args.utility_gain_scale,
            "nearest_soft_label": args.nearest_soft_label,
            "safe_nonimprove_soft_label": args.safe_nonimprove_soft_label,
            "soft_positive_weight": args.soft_positive_weight,
            "soft_nearest_weight": args.soft_nearest_weight,
            "candidate_only_row_loss": args.candidate_only_row_loss,
            "candidate_row_pos_weight": args.candidate_row_pos_weight,
            "group_change_weight": args.group_change_weight,
            "false_change_margin_weight": args.false_change_margin_weight,
            "parseq_harm_weight": args.parseq_harm_weight,
            "tesseract_nonimprove_weight": args.tesseract_nonimprove_weight,
            "rank0_row_margin_weight": args.rank0_row_margin_weight,
            "hard_positive_row_margin_weight": args.hard_positive_row_margin_weight,
            "hard_positive_group_margin_weight": args.hard_positive_group_margin_weight,
            "hard_positive_min_tesseract_gain": args.hard_positive_min_tesseract_gain,
            "matched_pairwise_group_weight": args.matched_pairwise_group_weight,
            "matched_pairwise_row_weight": args.matched_pairwise_row_weight,
            "matched_pairwise_margin": args.matched_pairwise_margin,
            "matched_pairwise_pairs_per_positive": args.matched_pairwise_pairs_per_positive,
            "tail_margin": args.tail_margin,
            "group_pos_weight": args.group_pos_weight,
            "group_target": args.group_target,
            "strong_tesseract_delta": args.strong_tesseract_delta,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_steps": args.threshold_steps,
            "default_threshold": args.default_threshold,
            "train_false_change_max": args.train_false_change_max,
            "train_false_change_rate_max": args.train_false_change_rate_max,
            "train_wrong_change_max": args.train_wrong_change_max,
            "train_tesseract_worsen_max": args.train_tesseract_worsen_max,
            "parseq_safe": args.parseq_safe,
            "disable_window": args.disable_window,
            "score_output": str(args.score_output) if args.score_output is not None else None,
            "policy_output_dir": str(args.policy_output_dir) if args.policy_output_dir is not None else None,
            "policy_output_prefix": args.policy_output_prefix,
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
