#!/usr/bin/env python3
"""Train a top-k selector with explicit hard-positive weighting.

This diagnostic reuses a precomputed feature matrix for the source-OOF top-k
shortlist and changes the objective, not the evidence. It is meant to test
whether the complete no-op collapse in the multi-evidence selector is caused by
nearest-dominant label imbalance.
"""

from __future__ import annotations

import argparse
import hashlib
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

import train_assignment_topk_policy_selector as base  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features(path: Path, expected_rows: int) -> np.ndarray:
    data = np.load(path)
    features = np.asarray(data["features"], dtype=np.float32)
    if features.shape[0] != expected_rows:
        raise SystemExit(f"feature row mismatch: {features.shape[0]} != {expected_rows}")
    return features


def train_one_balanced(
    rows: list[dict],
    groups: dict[tuple[int, int, int, int], list[int]],
    features: np.ndarray,
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    *,
    score_model_seed: int,
    selector_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(selector_seed)
    np.random.seed(selector_seed)
    torch.manual_seed(selector_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_groups = {key: cand for key, cand in group_candidates.items() if int(rows[cand[0]]["seed"]) != args.val_seed}
    val_groups = {key: cand for key, cand in group_candidates.items() if int(rows[cand[0]]["seed"]) == args.val_seed}
    full_train_groups = {key: groups[key] for key in train_groups}
    full_val_groups = {key: groups[key] for key in val_groups}
    train_row_indices = sorted({idx for candidates in train_groups.values() for idx in candidates})
    train_features_np = features[train_row_indices]
    feat_mean = train_features_np.mean(axis=0, keepdims=True)
    feat_std = train_features_np.std(axis=0, keepdims=True)
    feat_std[feat_std < 1e-6] = 1.0
    norm = ((features - feat_mean) / feat_std).astype(np.float32)

    model = base.Selector(features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    x = torch.from_numpy(norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_positions = []
    train_targets = []
    train_weights = []
    oracle_positions = []
    oracle_nearest_positions = []
    noheadroom_positions = []
    noheadroom_nearest_positions = []
    harmful_positions = []
    harmful_nearest_positions = []
    oracle_change_groups = 0
    noheadroom_groups = 0
    for candidates in train_groups.values():
        nearest = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        target_matches = [idx for idx in candidates if int(rows[idx]["label_assignment_oracle_choice"])]
        target = target_matches[0] if target_matches else nearest
        target_is_change = target != nearest
        oracle_change_groups += int(target_is_change)
        noheadroom_groups += int(not target_is_change)
        train_positions.append(torch.tensor(candidates, dtype=torch.long, device=device))
        train_targets.append(torch.tensor([candidates.index(target)], dtype=torch.long, device=device))
        train_weights.append(float(args.oracle_change_weight if target_is_change else args.nochange_group_weight))
        if target_is_change:
            oracle_positions.append(target)
            oracle_nearest_positions.append(nearest)
        else:
            for idx in candidates:
                if idx != nearest:
                    noheadroom_positions.append(idx)
                    noheadroom_nearest_positions.append(nearest)
        for idx in candidates:
            if idx != nearest and (
                int(rows[idx]["parseq_delta_vs_nearest"]) > 0 or int(rows[idx]["tesseract_delta_vs_nearest"]) > 0
            ):
                harmful_positions.append(idx)
                harmful_nearest_positions.append(nearest)

    train_weights_t = torch.tensor(train_weights, dtype=torch.float32, device=device)
    train_weights_t = train_weights_t / train_weights_t.mean().clamp_min(1e-6)
    oracle_positions_t = torch.tensor(oracle_positions, dtype=torch.long, device=device)
    oracle_nearest_positions_t = torch.tensor(oracle_nearest_positions, dtype=torch.long, device=device)
    noheadroom_positions_t = torch.tensor(noheadroom_positions, dtype=torch.long, device=device)
    noheadroom_nearest_positions_t = torch.tensor(noheadroom_nearest_positions, dtype=torch.long, device=device)
    harmful_positions_t = torch.tensor(harmful_positions, dtype=torch.long, device=device)
    harmful_nearest_positions_t = torch.tensor(harmful_nearest_positions, dtype=torch.long, device=device)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x)
        ce_losses = torch.stack(
            [
                torch.nn.functional.cross_entropy(logits[pos].unsqueeze(0), target, reduction="none").reshape(())
                for pos, target in zip(train_positions, train_targets)
            ]
        )
        loss = (ce_losses * train_weights_t).mean()
        if len(oracle_positions_t) and args.positive_margin_weight > 0:
            positive_loss = torch.nn.functional.softplus(
                logits[oracle_nearest_positions_t] - logits[oracle_positions_t] + args.positive_margin
            ).mean()
            loss = loss + args.positive_margin_weight * positive_loss
        if len(noheadroom_positions_t) and args.noheadroom_margin_weight > 0:
            nohead_loss = torch.nn.functional.softplus(
                logits[noheadroom_positions_t] - logits[noheadroom_nearest_positions_t] + args.noheadroom_margin
            ).mean()
            loss = loss + args.noheadroom_margin_weight * nohead_loss
        if len(harmful_positions_t) and args.harmful_margin_weight > 0:
            harm_loss = torch.nn.functional.softplus(
                logits[harmful_positions_t] - logits[harmful_nearest_positions_t] + args.harmful_margin
            ).mean()
            loss = loss + args.harmful_margin_weight * harm_loss
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()

    train_margin = base.tune_margin(rows, train_groups, full_train_groups, logits)
    policies = {
        "nearest": base.policy_metrics_by_group(
            rows,
            full_val_groups,
            [next(idx for idx in cand if int(rows[idx]["is_nearest"])) for cand in val_groups.values()],
        ),
        "shortlist_oracle": base.policy_metrics_by_group(rows, full_val_groups, base.oracle_select(rows, val_groups)),
        "argmax": base.policy_metrics_by_group(rows, full_val_groups, base.select_argmax(rows, val_groups, logits)),
        "train_tuned_margin": base.policy_metrics_by_group(
            rows,
            full_val_groups,
            base.select_argmax(rows, val_groups, logits, margin=train_margin),
        ),
    }

    return {
        "score_model_seed": int(score_model_seed),
        "selector_seed": int(selector_seed),
        "device": str(device),
        "feature_dim": int(features.shape[1]),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "train_oracle_change_groups": int(oracle_change_groups),
        "train_noheadroom_groups": int(noheadroom_groups),
        "train_margin": float(train_margin),
        "candidate_count_mean": float(np.mean([len(candidates) for candidates in val_groups.values()])),
        "policies": policies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--score-model-seed", type=int, action="append")
    parser.add_argument("--selector-seed", type=int, action="append")
    parser.add_argument("--shortlist-topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--oracle-change-weight", type=float, default=8.0)
    parser.add_argument("--nochange-group-weight", type=float, default=1.0)
    parser.add_argument("--positive-margin-weight", type=float, default=1.0)
    parser.add_argument("--positive-margin", type=float, default=0.25)
    parser.add_argument("--noheadroom-margin-weight", type=float, default=0.2)
    parser.add_argument("--noheadroom-margin", type=float, default=0.25)
    parser.add_argument("--harmful-margin-weight", type=float, default=0.2)
    parser.add_argument("--harmful-margin", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.score_model_seed:
        args.score_model_seed = [2]
    if not args.selector_seed:
        args.selector_seed = [0, 1, 2]

    rows = base.read_jsonl(args.table)
    groups = base.grouped_indices(rows)
    features = load_features(args.feature_cache, len(rows))
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
        group_candidates = {
            key: base.candidate_indices_for_group(rows, indices, oracle_scores, safe_scores, topk=args.shortlist_topk)
            for key, indices in groups.items()
        }
        for selector_seed in args.selector_seed:
            runs.append(
                train_one_balanced(
                    rows,
                    groups,
                    features,
                    group_candidates,
                    score_model_seed=score_model_seed,
                    selector_seed=selector_seed,
                    args=args,
                )
            )

    result = {
        "description": "Top-k assignment selector with hard-positive weighted objective. Non-promotion diagnostic only.",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "feature_cache": str(args.feature_cache),
            "feature_cache_sha256": sha256_file(args.feature_cache),
            "val_seed": int(args.val_seed),
            "score_model_seeds": args.score_model_seed,
            "selector_seeds": args.selector_seed,
            "shortlist_topk": int(args.shortlist_topk),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "oracle_change_weight": float(args.oracle_change_weight),
            "nochange_group_weight": float(args.nochange_group_weight),
            "positive_margin_weight": float(args.positive_margin_weight),
            "positive_margin": float(args.positive_margin),
            "noheadroom_margin_weight": float(args.noheadroom_margin_weight),
            "noheadroom_margin": float(args.noheadroom_margin),
            "harmful_margin_weight": float(args.harmful_margin_weight),
            "harmful_margin": float(args.harmful_margin),
            "cpu": bool(args.cpu),
        },
        "per_run": runs,
        "aggregate": base.aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
