#!/usr/bin/env python3
"""Train a top-k assignment selector with local image-difference patches.

This diagnostic tests whether candidate-local visual code-effect evidence can
rescue the score-seed top-k shortlist under leave-real-seed-out calibration.
It is intentionally separate from historical trainer scripts to preserve their
manifested hashes.
"""

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

import train_assignment_patch_ranker as patch_ranker  # noqa: E402
import train_assignment_topk_policy_selector as base  # noqa: E402


def row_real_seed(row: dict[str, Any]) -> int:
    return int(row.get("real_seed", row.get("original_seed", row.get("seed", 0))))


def split_groups(
    rows: list[dict[str, Any]],
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    args: argparse.Namespace,
    val_real_seed: int,
) -> tuple[dict[tuple[int, int, int, int], list[int]], dict[tuple[int, int, int, int], list[int]]]:
    train = {
        key: cand
        for key, cand in group_candidates.items()
        if int(rows[cand[0]]["seed"]) == args.val_seed and row_real_seed(rows[cand[0]]) != val_real_seed
    }
    val = {
        key: cand
        for key, cand in group_candidates.items()
        if int(rows[cand[0]]["seed"]) == args.val_seed and row_real_seed(rows[cand[0]]) == val_real_seed
    }
    if not train or not val:
        raise SystemExit(f"empty split for val_real_seed={val_real_seed}: train={len(train)} val={len(val)}")
    return train, val


def build_active_arrays(
    rows: list[dict[str, Any]],
    selector_features: np.ndarray,
    selected_original_indices: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[int, int]]:
    active_rows = [rows[idx] for idx in selected_original_indices]
    local_to_original = {local: original for local, original in enumerate(selected_original_indices)}
    images = patch_ranker.build_image_tensor(
        active_rows,
        width=args.width,
        height=args.height,
        image_mode=args.image_mode,
        diff_threshold=args.diff_threshold,
        crop_pad=args.crop_pad,
    )
    tabular = selector_features[selected_original_indices].astype(np.float32)
    return active_rows, images, tabular, local_to_original


class PatchSelector(torch.nn.Module):
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

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cnn(image), tabular], dim=1)).reshape(-1)


def grouped_local_indices(active_rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], list[int]]:
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(active_rows):
        groups[base.group_key(row)].append(index)
    return dict(groups)


def local_policy_metrics(
    rows: list[dict[str, Any]],
    full_groups: dict[tuple[int, int, int, int], list[int]],
    selected_local: list[int],
    local_to_original: dict[int, int],
) -> dict[str, Any]:
    selected_original = [local_to_original[idx] for idx in selected_local]
    return base.policy_metrics_by_group(rows, full_groups, selected_original)


def local_select_argmax(
    active_rows: list[dict[str, Any]],
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    logits: np.ndarray,
    margin: float | None = None,
) -> list[int]:
    selected = []
    for candidates in group_candidates.values():
        nearest = next(idx for idx in candidates if int(active_rows[idx]["is_nearest"]))
        best = max(candidates, key=lambda idx: (float(logits[idx]), -int(active_rows[idx]["topk_rank"])))
        if margin is not None and best != nearest and float(logits[best]) - float(logits[nearest]) < margin:
            best = nearest
        selected.append(best)
    return selected


def local_oracle_select(active_rows: list[dict[str, Any]], group_candidates: dict[tuple[int, int, int, int], list[int]]) -> list[int]:
    selected = []
    for candidates in group_candidates.values():
        oracle = [idx for idx in candidates if int(active_rows[idx]["label_assignment_oracle_choice"])]
        if oracle:
            selected.append(oracle[0])
        else:
            selected.append(next(idx for idx in candidates if int(active_rows[idx]["is_nearest"])))
    return selected


def tune_margin_local(
    rows: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    local_groups: dict[tuple[int, int, int, int], list[int]],
    full_groups: dict[tuple[int, int, int, int], list[int]],
    local_to_original: dict[int, int],
    logits: np.ndarray,
) -> float:
    margins = []
    for candidates in local_groups.values():
        nearest = next(idx for idx in candidates if int(active_rows[idx]["is_nearest"]))
        best = max(candidates, key=lambda idx: (float(logits[idx]), -int(active_rows[idx]["topk_rank"])))
        margins.append(float(logits[best]) - float(logits[nearest]))
    candidates = sorted(set(margins))
    if not candidates:
        return 1e6
    candidates = [max(candidates) + 1e-6] + candidates + [min(candidates) - 1e-6]
    best_margin = candidates[0]
    best_key = None
    for margin in candidates:
        selected = local_select_argmax(active_rows, local_groups, logits, margin=margin)
        metrics = local_policy_metrics(rows, full_groups, selected, local_to_original)
        key = (
            metrics["parseq_worsen_groups"],
            metrics["tesseract_worsen_groups"],
            metrics["wrong_changes"],
            metrics["tesseract_delta_sum"],
            metrics["false_changes"],
            metrics["changed_groups"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_margin = float(margin)
    return best_margin


def train_one_split(
    rows: list[dict[str, Any]],
    full_groups_original: dict[tuple[int, int, int, int], list[int]],
    selector_features: np.ndarray,
    group_candidates_original: dict[tuple[int, int, int, int], list[int]],
    *,
    score_model_seed: int,
    selector_seed: int,
    val_real_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(selector_seed)
    np.random.seed(selector_seed)
    torch.manual_seed(selector_seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed_all(selector_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_groups_original, val_groups_original = split_groups(rows, group_candidates_original, args, val_real_seed)
    active_original_indices = sorted(
        {idx for candidates in train_groups_original.values() for idx in candidates}
        | {idx for candidates in val_groups_original.values() for idx in candidates}
    )
    original_to_local = {original: local for local, original in enumerate(active_original_indices)}
    active_rows, images_np, tabular_np, local_to_original = build_active_arrays(
        rows,
        selector_features,
        active_original_indices,
        args,
    )
    train_groups = {
        key: [original_to_local[idx] for idx in candidates]
        for key, candidates in train_groups_original.items()
    }
    val_groups = {
        key: [original_to_local[idx] for idx in candidates]
        for key, candidates in val_groups_original.items()
    }
    full_train_groups = {key: full_groups_original[key] for key in train_groups}
    full_val_groups = {key: full_groups_original[key] for key in val_groups}

    train_local_indices = sorted({idx for candidates in train_groups.values() for idx in candidates})
    tab_mean = tabular_np[train_local_indices].mean(axis=0, keepdims=True)
    tab_std = tabular_np[train_local_indices].std(axis=0, keepdims=True)
    tab_std[tab_std < 1e-6] = 1.0
    tab_norm = ((tabular_np - tab_mean) / tab_std).astype(np.float32)

    model = PatchSelector(tab_dim=tab_norm.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    images = torch.from_numpy(images_np).to(device)
    tabular = torch.from_numpy(tab_norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_positions = []
    train_targets = []
    harmful_positions = []
    nearest_positions = []
    for candidates in train_groups.values():
        target_matches = [idx for idx in candidates if int(active_rows[idx]["label_assignment_oracle_choice"])]
        if target_matches:
            target = target_matches[0]
        else:
            target = next(idx for idx in candidates if int(active_rows[idx]["is_nearest"]))
        train_positions.append(torch.tensor(candidates, dtype=torch.long, device=device))
        train_targets.append(torch.tensor([candidates.index(target)], dtype=torch.long, device=device))
        nearest = next(idx for idx in candidates if int(active_rows[idx]["is_nearest"]))
        for idx in candidates:
            if idx != nearest and (
                int(active_rows[idx]["parseq_delta_vs_nearest"]) > 0
                or int(active_rows[idx]["tesseract_delta_vs_nearest"]) > 0
            ):
                harmful_positions.append(idx)
                nearest_positions.append(nearest)
    harmful_positions_t = torch.tensor(harmful_positions, dtype=torch.long, device=device)
    nearest_positions_t = torch.tensor(nearest_positions, dtype=torch.long, device=device)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(images, tabular)
        losses = [
            torch.nn.functional.cross_entropy(logits[pos].unsqueeze(0), target)
            for pos, target in zip(train_positions, train_targets)
        ]
        loss = torch.stack(losses).mean()
        if len(harmful_positions_t) and args.harmful_margin_weight > 0:
            harm_loss = torch.nn.functional.softplus(
                logits[harmful_positions_t] - logits[nearest_positions_t] + args.harmful_margin
            ).mean()
            loss = loss + args.harmful_margin_weight * harm_loss
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits_np = model(images, tabular).detach().cpu().numpy()

    train_margin = tune_margin_local(rows, active_rows, train_groups, full_train_groups, local_to_original, logits_np)
    policies = {
        "nearest": local_policy_metrics(
            rows,
            full_val_groups,
            [next(idx for idx in cand if int(active_rows[idx]["is_nearest"])) for cand in val_groups.values()],
            local_to_original,
        ),
        "shortlist_oracle": local_policy_metrics(
            rows,
            full_val_groups,
            local_oracle_select(active_rows, val_groups),
            local_to_original,
        ),
        "argmax": local_policy_metrics(
            rows,
            full_val_groups,
            local_select_argmax(active_rows, val_groups, logits_np),
            local_to_original,
        ),
        "train_tuned_margin": local_policy_metrics(
            rows,
            full_val_groups,
            local_select_argmax(active_rows, val_groups, logits_np, margin=train_margin),
            local_to_original,
        ),
    }

    return {
        "split_mode": "real_seed_within_seed",
        "val_seed": int(args.val_seed),
        "val_real_seed": int(val_real_seed),
        "score_model_seed": int(score_model_seed),
        "selector_seed": int(selector_seed),
        "device": str(device),
        "feature_dim": int(tab_norm.shape[1]),
        "active_rows": int(len(active_rows)),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "train_margin": float(train_margin),
        "candidate_count_mean": float(np.mean([len(candidates) for candidates in val_groups.values()])),
        "policies": policies,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    policy_names = sorted({name for run in runs for name in run["policies"]})
    output: dict[str, Any] = {"policy_metrics": {}}
    for policy in policy_names:
        metric_names = sorted({metric for run in runs for metric in run["policies"][policy]})
        output["policy_metrics"][policy] = {}
        for metric in metric_names:
            values = np.asarray([float(run["policies"][policy][metric]) for run in runs], dtype=np.float64)
            output["policy_metrics"][policy][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    output["candidate_count_mean"] = float(np.mean([run["candidate_count_mean"] for run in runs])) if runs else 0.0
    return output


def aggregate_by_split(runs: list[dict[str, Any]]) -> dict[str, Any]:
    splits: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        splits.setdefault(f"real_seed_{run['val_real_seed']}", []).append(run)
    return {key: aggregate(value) for key, value in sorted(splits.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--val-real-seed", type=int, action="append")
    parser.add_argument("--score-model-seed", type=int, action="append")
    parser.add_argument("--selector-seed", type=int, action="append")
    parser.add_argument("--shortlist-topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--harmful-margin-weight", type=float, default=1.0)
    parser.add_argument("--harmful-margin", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--image-mode", choices=["full", "diff_crop"], default="diff_crop")
    parser.add_argument("--diff-threshold", type=float, default=0.002)
    parser.add_argument("--crop-pad", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.val_real_seed:
        args.val_real_seed = [0, 1, 2]
    if not args.score_model_seed:
        args.score_model_seed = [2]
    if not args.selector_seed:
        args.selector_seed = [0, 1, 2]

    rows = base.read_jsonl(args.table)
    full_groups = base.grouped_indices(rows)
    base_features = base.build_base_features(rows)
    oracle_by_model = base.load_score_features(args.oracle_score)
    safe_by_model = base.load_score_features(args.safe_score)

    runs: list[dict[str, Any]] = []
    for score_model_seed in args.score_model_seed:
        oracle_scores = np.asarray(
            [oracle_by_model[score_model_seed][idx] for idx in range(len(rows))],
            dtype=np.float32,
        )
        safe_scores = np.asarray(
            [safe_by_model[score_model_seed][idx] for idx in range(len(rows))],
            dtype=np.float32,
        )
        selector_features = base.build_selector_features(rows, full_groups, base_features, oracle_scores, safe_scores)
        group_candidates = {
            key: base.candidate_indices_for_group(rows, indices, oracle_scores, safe_scores, topk=args.shortlist_topk)
            for key, indices in full_groups.items()
        }
        for val_real_seed in args.val_real_seed:
            for selector_seed in args.selector_seed:
                runs.append(
                    train_one_split(
                        rows,
                        full_groups,
                        selector_features,
                        group_candidates,
                        score_model_seed=score_model_seed,
                        selector_seed=selector_seed,
                        val_real_seed=val_real_seed,
                        args=args,
                    )
                )

    result = {
        "description": "Top-k assignment patch selector split diagnostic. Non-promotion diagnostic only.",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "val_seed": int(args.val_seed),
            "val_real_seeds": args.val_real_seed,
            "score_model_seeds": args.score_model_seed,
            "selector_seeds": args.selector_seed,
            "shortlist_topk": int(args.shortlist_topk),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "harmful_margin_weight": float(args.harmful_margin_weight),
            "harmful_margin": float(args.harmful_margin),
            "width": int(args.width),
            "height": int(args.height),
            "image_mode": args.image_mode,
            "diff_threshold": float(args.diff_threshold),
            "crop_pad": int(args.crop_pad),
            "cpu": bool(args.cpu),
        },
        "per_run": runs,
        "aggregate": aggregate(runs),
        "aggregate_by_split": aggregate_by_split(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
