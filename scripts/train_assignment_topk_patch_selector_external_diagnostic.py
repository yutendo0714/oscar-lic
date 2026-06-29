#!/usr/bin/env python3
"""Train a top-k patch selector on external train/validation seed splits.

This companion keeps ``train_assignment_topk_patch_selector_split_diagnostic.py``
unchanged for its existing manifests, but reuses its patch model utilities for
Eval600-to-Eval300 style diagnostics.
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

import train_assignment_topk_patch_selector_split_diagnostic as patch_diag  # noqa: E402
import train_assignment_topk_policy_selector as base  # noqa: E402


def train_one_external(
    rows: list[dict[str, Any]],
    full_groups_original: dict[tuple[int, int, int, int], list[int]],
    selector_features: np.ndarray,
    group_candidates_original: dict[tuple[int, int, int, int], list[int]],
    *,
    score_model_seed: int,
    selector_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(selector_seed)
    np.random.seed(selector_seed)
    torch.manual_seed(selector_seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed_all(selector_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_groups_original = {
        key: cand
        for key, cand in group_candidates_original.items()
        if int(rows[cand[0]]["seed"]) != args.val_seed
    }
    val_groups_original = {
        key: cand
        for key, cand in group_candidates_original.items()
        if int(rows[cand[0]]["seed"]) == args.val_seed
    }
    if not train_groups_original or not val_groups_original:
        raise SystemExit(f"empty split: train={len(train_groups_original)} val={len(val_groups_original)}")

    active_original_indices = sorted(
        {idx for candidates in train_groups_original.values() for idx in candidates}
        | {idx for candidates in val_groups_original.values() for idx in candidates}
    )
    original_to_local = {original: local for local, original in enumerate(active_original_indices)}
    active_rows, images_np, tabular_np, local_to_original = patch_diag.build_active_arrays(
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

    model = patch_diag.PatchSelector(tab_dim=tab_norm.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
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

    train_margin = patch_diag.tune_margin_local(
        rows,
        active_rows,
        train_groups,
        full_train_groups,
        local_to_original,
        logits_np,
    )
    policies = {
        "nearest": patch_diag.local_policy_metrics(
            rows,
            full_val_groups,
            [next(idx for idx in cand if int(active_rows[idx]["is_nearest"])) for cand in val_groups.values()],
            local_to_original,
        ),
        "shortlist_oracle": patch_diag.local_policy_metrics(
            rows,
            full_val_groups,
            patch_diag.local_oracle_select(active_rows, val_groups),
            local_to_original,
        ),
        "argmax": patch_diag.local_policy_metrics(
            rows,
            full_val_groups,
            patch_diag.local_select_argmax(active_rows, val_groups, logits_np),
            local_to_original,
        ),
        "train_tuned_margin": patch_diag.local_policy_metrics(
            rows,
            full_val_groups,
            patch_diag.local_select_argmax(active_rows, val_groups, logits_np, margin=train_margin),
            local_to_original,
        ),
    }

    return {
        "split_mode": "seed",
        "val_seed": int(args.val_seed),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
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
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--image-mode", choices=["full", "diff_crop"], default="diff_crop")
    parser.add_argument("--diff-threshold", type=float, default=0.002)
    parser.add_argument("--crop-pad", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

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
        for selector_seed in args.selector_seed:
            runs.append(
                train_one_external(
                    rows,
                    full_groups,
                    selector_features,
                    group_candidates,
                    score_model_seed=score_model_seed,
                    selector_seed=selector_seed,
                    args=args,
                )
            )

    result = {
        "description": "Top-k assignment patch selector external seed diagnostic. Non-promotion diagnostic only.",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "val_seed": int(args.val_seed),
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
        "aggregate": patch_diag.aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
