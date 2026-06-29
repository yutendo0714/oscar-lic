#!/usr/bin/env python3
"""Train a tiny image-difference ranker for top-k code assignment variants."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def grouped_indices(rows: list[dict]) -> dict[tuple[int, int, int, int], list[int]]:
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        real_seed = int(row.get("real_seed", row.get("original_seed", 0)))
        groups[(real_seed, int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))].append(index)
    return dict(groups)


def find_nearest_indices(
    rows: list[dict],
    groups: dict[tuple[int, int, int, int], list[int]],
) -> dict[tuple[int, int, int, int], int]:
    nearest = {}
    for key, indices in groups.items():
        matches = [idx for idx in indices if int(rows[idx]["is_nearest"])]
        if len(matches) != 1:
            raise SystemExit(f"group {key} has {len(matches)} nearest rows")
        nearest[key] = matches[0]
    return nearest


def load_gray(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((width, height), Image.BICUBIC)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_gray_native(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        return np.asarray(image, dtype=np.float32) / 255.0


def resize_gray(array: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
    image = image.resize((width, height), Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def crop_bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if len(xs) == 0:
        return 0, 0, width, height
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(width, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(height, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def build_full_image_tensor(rows: list[dict], width: int, height: int) -> np.ndarray:
    groups = grouped_indices(rows)
    nearest_by_group = find_nearest_indices(rows, groups)
    images = []
    cache: dict[str, np.ndarray] = {}
    for group_key, indices in groups.items():
        nearest_path = str(rows[nearest_by_group[group_key]]["image_path"])
        if nearest_path not in cache:
            cache[nearest_path] = load_gray(Path(nearest_path), width, height)
        nearest_img = cache[nearest_path]
        for idx in indices:
            path = str(rows[idx]["image_path"])
            if path not in cache:
                cache[path] = load_gray(Path(path), width, height)
            variant = cache[path]
            diff = np.abs(variant - nearest_img)
            images.append(np.stack([nearest_img, variant, diff], axis=0))
    # The loop above follows grouped order; restore original row order.
    output = np.empty((len(rows), 3, height, width), dtype=np.float32)
    cursor = 0
    for indices in groups.values():
        for idx in indices:
            output[idx] = images[cursor]
            cursor += 1
    return output


def build_diff_crop_image_tensor(
    rows: list[dict],
    width: int,
    height: int,
    diff_threshold: float,
    crop_pad: int,
) -> np.ndarray:
    groups = grouped_indices(rows)
    nearest_by_group = find_nearest_indices(rows, groups)
    output = np.empty((len(rows), 3, height, width), dtype=np.float32)
    cache: dict[str, np.ndarray] = {}
    for group_key, indices in groups.items():
        nearest_path = str(rows[nearest_by_group[group_key]]["image_path"])
        if nearest_path not in cache:
            cache[nearest_path] = load_gray_native(Path(nearest_path))
        nearest_img = cache[nearest_path]
        union_mask = np.zeros_like(nearest_img, dtype=bool)
        variants = {}
        for idx in indices:
            path = str(rows[idx]["image_path"])
            if path not in cache:
                cache[path] = load_gray_native(Path(path))
            variant = cache[path]
            variants[idx] = variant
            union_mask |= np.abs(variant - nearest_img) > diff_threshold
        x0, y0, x1, y1 = crop_bbox(union_mask, pad=crop_pad)
        nearest_crop = nearest_img[y0:y1, x0:x1]
        for idx in indices:
            variant_crop = variants[idx][y0:y1, x0:x1]
            nearest_resized = resize_gray(nearest_crop, width, height)
            variant_resized = resize_gray(variant_crop, width, height)
            diff_resized = np.abs(variant_resized - nearest_resized)
            output[idx] = np.stack([nearest_resized, variant_resized, diff_resized], axis=0)
    return output


def build_image_tensor(
    rows: list[dict],
    width: int,
    height: int,
    image_mode: str,
    diff_threshold: float,
    crop_pad: int,
) -> np.ndarray:
    if image_mode == "full":
        return build_full_image_tensor(rows, width=width, height=height)
    if image_mode == "diff_crop":
        return build_diff_crop_image_tensor(
            rows,
            width=width,
            height=height,
            diff_threshold=diff_threshold,
            crop_pad=crop_pad,
        )
    raise ValueError(f"unsupported image mode: {image_mode}")


def build_tabular(
    rows: list[dict],
    metadata_features: str = "none",
    *,
    include_image_features: bool = False,
) -> np.ndarray:
    optional_prefixes = ["codebook_", "latent_vec_"]
    if include_image_features:
        optional_prefixes.append("img_")
    optional_features = sorted(
        key
        for key, value in rows[0].items()
        if any(key.startswith(prefix) for prefix in optional_prefixes) and isinstance(value, (int, float))
    )
    sources = sorted({str(row.get("source", "unknown")) for row in rows})
    source_to_index = {source: index for index, source in enumerate(sources)}
    values = []
    for row in rows:
        rank = float(row["topk_rank"])
        rel = float(row["assignment_relative_error"])
        rel_clipped = min(max(rel, 1.0), 10.0)
        base = [
            rank / 7.0,
            rel_clipped,
            np.log1p(rel_clipped - 1.0),
            float(row["is_nearest"]),
            float(row["code_equals_nearest"]),
        ]
        if metadata_features in {"source", "source_length", "source_length_difficulty"}:
            onehot = [0.0] * len(sources)
            onehot[source_to_index[str(row.get("source", "unknown"))]] = 1.0
            base.extend(onehot)
        if metadata_features in {"source_length", "source_length_difficulty"}:
            reference_length = float(row.get("reference_length", 0.0))
            base.extend([reference_length / 16.0, np.log1p(reference_length) / np.log(32.0)])
        if metadata_features == "source_length_difficulty":
            # Diagnostic-only features: these use OCR distances from the generated assignment table.
            base.extend(
                [
                    min(float(row.get("tesseract_nearest_distance", 0.0)), 8.0) / 8.0,
                    min(float(row.get("parseq_nearest_distance", 0.0)), 8.0) / 8.0,
                ]
            )
        base.extend(float(row[key]) for key in optional_features)
        values.append(base)
    return np.asarray(values, dtype=np.float32)


class AssignmentPatchRanker(torch.nn.Module):
    def __init__(self, tab_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 48, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(48 + tab_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        embedding = self.cnn(image)
        return self.head(torch.cat([embedding, tab], dim=1)).reshape(-1)


class AssignmentTabularRanker(torch.nn.Module):
    def __init__(self, tab_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(tab_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return self.head(tab).reshape(-1)


def pair_indices(rows: list[dict], indices: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    pos_by_row = {int(row_index): pos for pos, row_index in enumerate(indices)}
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for row_index in indices:
        row = rows[int(row_index)]
        real_seed = int(row.get("real_seed", row.get("original_seed", 0)))
        groups[(real_seed, int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))].append(int(row_index))
    left = []
    right = []
    for group_indices in groups.values():
        positives = [idx for idx in group_indices if labels[idx] > 0.5]
        negatives = [idx for idx in group_indices if labels[idx] <= 0.5]
        for pos in positives:
            for neg in negatives:
                left.append(pos_by_row[pos])
                right.append(pos_by_row[neg])
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)


def listwise_group_targets(
    rows: list[dict],
    indices: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[torch.Tensor], list[int]]:
    pos_by_row = {int(row_index): pos for pos, row_index in enumerate(indices)}
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for row_index in indices:
        row = rows[int(row_index)]
        real_seed = int(row.get("real_seed", row.get("original_seed", 0)))
        groups[(real_seed, int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))].append(int(row_index))

    group_positions = []
    targets = []
    for group_indices in groups.values():
        ordered = sorted(group_indices, key=lambda idx: int(rows[idx]["topk_rank"]))
        positives = [idx for idx in ordered if labels[idx] > 0.5]
        if len(positives) == 1:
            target_row = positives[0]
        elif len(positives) > 1:
            target_row = min(
                positives,
                key=lambda idx: (
                    int(rows[idx]["parseq_delta_vs_nearest"]) + int(rows[idx]["tesseract_delta_vs_nearest"]),
                    int(rows[idx]["parseq_delta_vs_nearest"]),
                    int(rows[idx]["topk_rank"]),
                ),
            )
        else:
            target_row = next(idx for idx in ordered if int(rows[idx]["is_nearest"]))
        group_positions.append(torch.tensor([pos_by_row[idx] for idx in ordered], dtype=torch.long))
        targets.append(ordered.index(target_row))
    return group_positions, targets


def select_with_threshold(group: list[int], rows: list[dict], scores: np.ndarray, threshold: float) -> int:
    best = max(group, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
    if float(scores[best]) >= threshold:
        return best
    return next(idx for idx in group if int(rows[idx]["is_nearest"]))


def select_with_score_margin_threshold(
    group: list[int],
    rows: list[dict],
    scores: np.ndarray,
    score_threshold: float,
    margin_threshold: float,
) -> int:
    nearest = next(idx for idx in group if int(rows[idx]["is_nearest"]))
    best = max(group, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
    margin = float(scores[best]) - float(scores[nearest])
    if best != nearest and float(scores[best]) >= score_threshold and margin >= margin_threshold:
        return best
    return nearest


def select_top_changed(
    groups: dict[tuple[int, int, int, int], list[int]],
    rows: list[dict],
    scores: np.ndarray,
    budget: int,
) -> list[int]:
    proposals = []
    selected = []
    for group in groups.values():
        nearest = next(idx for idx in group if int(rows[idx]["is_nearest"]))
        best = max(group, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
        margin = float(scores[best]) - float(scores[nearest])
        proposals.append((margin, best, nearest))
    proposals.sort(reverse=True, key=lambda item: item[0])
    for rank, (_, best, nearest) in enumerate(proposals):
        selected.append(best if rank < budget and best != nearest else nearest)
    return selected


def tune_top_changed_budget(
    rows: list[dict],
    groups: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
) -> int:
    best_budget = 0
    best_key = None
    max_budget = len(groups)
    for budget in range(max_budget + 1):
        selected = select_top_changed(groups, rows, scores, budget)
        metrics = policy_metrics(rows, selected)
        key = (
            metrics["parseq_worsen_groups"],
            metrics["tesseract_delta_sum"],
            metrics["tesseract_worsen_groups"],
            metrics["changed_groups"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_budget = budget
    return best_budget


def threshold_grid(values: list[float], max_candidates: int = 128) -> list[float]:
    unique = sorted(set(float(value) for value in values))
    if not unique:
        return []
    if len(unique) > max_candidates:
        positions = np.linspace(0, len(unique) - 1, num=max_candidates).round().astype(np.int64)
        unique = [unique[int(position)] for position in positions]
        unique = sorted(set(unique))
    return [max(unique) + 1e-6] + unique + [min(unique) - 1e-6]


def policy_metrics(rows: list[dict], selected: list[int]) -> dict:
    parseq_delta = int(sum(int(rows[idx]["parseq_delta_vs_nearest"]) for idx in selected))
    tesseract_delta = int(sum(int(rows[idx]["tesseract_delta_vs_nearest"]) for idx in selected))
    changed = int(sum(1 for idx in selected if not int(rows[idx]["is_nearest"])))
    return {
        "groups": len(selected),
        "changed_groups": changed,
        "parseq_delta_sum": parseq_delta,
        "tesseract_delta_sum": tesseract_delta,
        "multi_teacher_delta_sum": parseq_delta + tesseract_delta,
        "parseq_worsen_groups": int(sum(1 for idx in selected if int(rows[idx]["parseq_delta_vs_nearest"]) > 0)),
        "tesseract_improve_groups": int(sum(1 for idx in selected if int(rows[idx]["tesseract_delta_vs_nearest"]) < 0)),
        "tesseract_worsen_groups": int(sum(1 for idx in selected if int(rows[idx]["tesseract_delta_vs_nearest"]) > 0)),
    }


def tune_threshold(rows: list[dict], groups: dict[tuple[int, int, int, int], list[int]], scores: np.ndarray) -> float:
    candidates = threshold_grid([float(scores[idx]) for group in groups.values() for idx in group])
    if not candidates:
        return 1.0
    best_threshold = candidates[0]
    best_key = None
    for threshold in candidates:
        selected = [select_with_threshold(group, rows, scores, threshold) for group in groups.values()]
        metrics = policy_metrics(rows, selected)
        key = (
            metrics["parseq_worsen_groups"],
            metrics["tesseract_delta_sum"],
            metrics["tesseract_worsen_groups"],
            metrics["changed_groups"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def tune_conservative_threshold(
    rows: list[dict],
    groups: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
) -> float:
    candidates = threshold_grid([float(scores[idx]) for group in groups.values() for idx in group])
    if not candidates:
        return 1.0
    best_threshold = candidates[0]
    best_key = None
    for threshold in candidates:
        selected = [select_with_threshold(group, rows, scores, threshold) for group in groups.values()]
        metrics = policy_metrics(rows, selected)
        key = (
            metrics["parseq_worsen_groups"],
            metrics["tesseract_worsen_groups"],
            metrics["tesseract_delta_sum"],
            metrics["changed_groups"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def tune_conservative_score_margin(
    rows: list[dict],
    groups: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
) -> tuple[float, float]:
    score_thresholds = threshold_grid([float(scores[idx]) for group in groups.values() for idx in group])
    if not score_thresholds:
        return 1.0, 1.0
    margin_candidates = []
    for group in groups.values():
        nearest = next(idx for idx in group if int(rows[idx]["is_nearest"]))
        best = max(group, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
        margin_candidates.append(float(scores[best]) - float(scores[nearest]))
    margin_thresholds = threshold_grid(margin_candidates)
    best_thresholds = (score_thresholds[0], margin_thresholds[0])
    best_key = None
    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            selected = [
                select_with_score_margin_threshold(group, rows, scores, score_threshold, margin_threshold)
                for group in groups.values()
            ]
            metrics = policy_metrics(rows, selected)
            key = (
                metrics["parseq_worsen_groups"],
                metrics["tesseract_worsen_groups"],
                metrics["tesseract_delta_sum"],
                metrics["changed_groups"],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_thresholds = (float(score_threshold), float(margin_threshold))
    return best_thresholds


def train_one(
    rows: list[dict],
    images: np.ndarray,
    tabular: np.ndarray,
    val_seed: int,
    model_seed: int,
    args: argparse.Namespace,
) -> dict:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.target == "safe_improve":
        labels = np.asarray([int(row["label_tesseract_parseq_safe_improves"]) for row in rows], dtype=np.float32)
    elif args.target == "oracle_choice":
        labels = np.asarray([int(row["label_assignment_oracle_choice"]) for row in rows], dtype=np.float32)
    else:
        raise ValueError(f"unsupported target: {args.target}")
    seeds = np.asarray([int(row["seed"]) for row in rows], dtype=np.int64)
    train_mask = seeds != val_seed
    val_mask = seeds == val_seed
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)

    tab_mean = tabular[train_mask].mean(axis=0, keepdims=True)
    tab_std = tabular[train_mask].std(axis=0, keepdims=True)
    tab_std[tab_std < 1e-6] = 1.0
    tab_norm = (tabular - tab_mean) / tab_std

    model_cls = AssignmentTabularRanker if args.tabular_only else AssignmentPatchRanker
    model = model_cls(tab_dim=tabular.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    train_images = torch.from_numpy(images[train_mask]).to(device)
    train_tab = torch.from_numpy(tab_norm[train_mask].astype(np.float32)).to(device)
    train_labels = torch.from_numpy(labels[train_mask]).to(device)
    positives = float(train_labels.sum().item())
    negatives = float(len(train_labels) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    )
    left, right = pair_indices(rows, train_indices, labels)
    left = left.to(device)
    right = right.to(device)
    listwise_positions, listwise_targets = listwise_group_targets(rows, train_indices, labels)
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
        if not loss_terms:
            raise RuntimeError("no loss terms configured")
        loss = torch.stack(loss_terms).sum()
        if len(left) and args.pairwise_weight > 0.0 and args.loss_mode != "listwise":
            pair_loss = torch.nn.functional.softplus(-(logits[left] - logits[right])).mean()
            loss = loss + args.pairwise_weight * pair_loss
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        all_scores = torch.sigmoid(
            model(
                torch.from_numpy(images).to(device),
                torch.from_numpy(tab_norm.astype(np.float32)).to(device),
            )
        ).detach().cpu().numpy()
    if args.score_output is not None:
        args.score_output.parent.mkdir(parents=True, exist_ok=True)
        with args.score_output.open("a", encoding="utf-8") as handle:
            for row_index, row in enumerate(rows):
                out = {
                    "row_index": row_index,
                    "val_seed": val_seed,
                    "model_seed": model_seed,
                    "score": float(all_scores[row_index]),
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
                }
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    groups = grouped_indices(rows)
    train_groups = {key: value for key, value in groups.items() if int(rows[value[0]]["seed"]) != val_seed}
    val_groups = {key: value for key, value in groups.items() if int(rows[value[0]]["seed"]) == val_seed}
    threshold = tune_threshold(rows, train_groups, all_scores)
    conservative_threshold = tune_conservative_threshold(rows, train_groups, all_scores)
    conservative_score_threshold, conservative_margin_threshold = tune_conservative_score_margin(
        rows,
        train_groups,
        all_scores,
    )
    top_changed_budget = tune_top_changed_budget(rows, train_groups, all_scores)
    threshold_selected = [select_with_threshold(group, rows, all_scores, threshold) for group in val_groups.values()]
    conservative_threshold_selected = [
        select_with_threshold(group, rows, all_scores, conservative_threshold) for group in val_groups.values()
    ]
    conservative_score_margin_selected = [
        select_with_score_margin_threshold(
            group,
            rows,
            all_scores,
            conservative_score_threshold,
            conservative_margin_threshold,
        )
        for group in val_groups.values()
    ]
    nearest_selected = [next(idx for idx in group if int(rows[idx]["is_nearest"])) for group in val_groups.values()]
    oracle_selected = []
    for group in val_groups.values():
        oracle_matches = [idx for idx in group if int(rows[idx]["label_assignment_oracle_choice"])]
        if oracle_matches:
            oracle_selected.append(oracle_matches[0])
        else:
            oracle_selected.append(next(idx for idx in group if int(rows[idx]["is_nearest"])))
    always_best_selected = [max(group, key=lambda idx: (float(all_scores[idx]), -int(rows[idx]["topk_rank"]))) for group in val_groups.values()]
    policy = {
        "nearest": policy_metrics(rows, nearest_selected),
        "oracle": policy_metrics(rows, oracle_selected),
        "always_best_score": policy_metrics(rows, always_best_selected),
        "train_tuned_threshold": policy_metrics(rows, threshold_selected),
        "train_tuned_conservative_threshold": policy_metrics(rows, conservative_threshold_selected),
        "train_tuned_conservative_score_margin": policy_metrics(rows, conservative_score_margin_selected),
        "train_tuned_top_changed": policy_metrics(
            rows, select_top_changed(val_groups, rows, all_scores, top_changed_budget)
        ),
    }
    for budget in [3, 5, 10, 15]:
        policy[f"top_changed_{budget}"] = policy_metrics(
            rows, select_top_changed(val_groups, rows, all_scores, budget)
        )

    return {
        "val_seed": val_seed,
        "model_seed": model_seed,
        "device": str(device),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positive_rate": float(labels[train_mask].mean()),
        "val_positive_rate": float(labels[val_mask].mean()),
        "train_auc": roc_auc(all_scores[train_mask], labels[train_mask]),
        "train_ap": average_precision(all_scores[train_mask], labels[train_mask]),
        "val_auc": roc_auc(all_scores[val_mask], labels[val_mask]),
        "val_ap": average_precision(all_scores[val_mask], labels[val_mask]),
        "threshold": threshold,
        "conservative_threshold": conservative_threshold,
        "conservative_score_threshold": conservative_score_threshold,
        "conservative_margin_threshold": conservative_margin_threshold,
        "top_changed_budget": top_changed_budget,
        "policy_metrics": policy,
    }


def aggregate(results: list[dict]) -> dict:
    policy_names = sorted({name for row in results for name in row["policy_metrics"]})
    output = {"policy_metrics": {}}
    for metric in ["train_auc", "train_ap", "val_auc", "val_ap"]:
        values = [row[metric] for row in results if row[metric] is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            output[metric] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "count": int(len(arr))}
    for policy in policy_names:
        metric_names = sorted({name for row in results for name in row["policy_metrics"][policy]})
        output["policy_metrics"][policy] = {}
        for metric in metric_names:
            arr = np.asarray([float(row["policy_metrics"][policy][metric]) for row in results], dtype=np.float64)
            output["policy_metrics"][policy][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--val-seed", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--loss-mode", choices=["bce", "listwise", "bce_listwise"], default="bce")
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--target", choices=["safe_improve", "oracle_choice"], default="safe_improve")
    parser.add_argument(
        "--metadata-features",
        choices=["none", "source", "source_length", "source_length_difficulty"],
        default="none",
        help="Optional diagnostic metadata features; source_length_difficulty uses OCR table distances and is not a deployable feature.",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--image-mode", choices=["full", "diff_crop"], default="full")
    parser.add_argument("--diff-threshold", type=float, default=0.002)
    parser.add_argument("--crop-pad", type=int, default=4)
    parser.add_argument("--tabular-only", action="store_true")
    parser.add_argument("--include-image-features", action="store_true")
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.model_seed is None:
        args.model_seed = [0]
    if args.val_seed is None:
        args.val_seed = [0, 1, 2]

    rows = read_jsonl(args.table)
    if args.score_output is not None and args.score_output.exists():
        args.score_output.unlink()
    if args.tabular_only:
        images = np.zeros((len(rows), 3, 1, 1), dtype=np.float32)
    else:
        images = build_image_tensor(
            rows,
            width=args.width,
            height=args.height,
            image_mode=args.image_mode,
            diff_threshold=args.diff_threshold,
            crop_pad=args.crop_pad,
        )
    tabular = build_tabular(
        rows,
        metadata_features=args.metadata_features,
        include_image_features=args.include_image_features,
    )
    results = []
    for val_seed in args.val_seed:
        for model_seed in args.model_seed:
            results.append(train_one(rows, images, tabular, val_seed, model_seed, args))
    output = {
        "description": "Tiny patch-difference ranker for top-k code assignment variants.",
        "table": str(args.table),
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "pairwise_weight": args.pairwise_weight,
            "loss_mode": args.loss_mode,
            "listwise_weight": args.listwise_weight,
            "target": args.target,
            "metadata_features": args.metadata_features,
            "width": args.width,
            "height": args.height,
            "image_mode": args.image_mode,
            "diff_threshold": args.diff_threshold,
            "crop_pad": args.crop_pad,
            "tabular_only": args.tabular_only,
            "include_image_features": args.include_image_features,
            "model_seeds": args.model_seed,
            "val_seeds": args.val_seed,
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
