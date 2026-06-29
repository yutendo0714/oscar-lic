#!/usr/bin/env python3
"""Train top-k assignment selectors under alternative calibration splits.

This is a diagnostic companion to ``train_assignment_topk_policy_selector.py``.
It deliberately leaves that historical script untouched so old reproducibility
manifests keep their recorded script hashes.
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

import train_assignment_topk_policy_selector as base  # noqa: E402


def row_real_seed(row: dict[str, Any]) -> int:
    return int(row.get("real_seed", row.get("original_seed", row.get("seed", 0))))


def split_groups(
    rows: list[dict[str, Any]],
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    args: argparse.Namespace,
    *,
    val_real_seed: int | None,
    val_source_remainder: int | None,
) -> tuple[dict[tuple[int, int, int, int], list[int]], dict[tuple[int, int, int, int], list[int]]]:
    if args.split_mode == "seed":
        train = {key: cand for key, cand in group_candidates.items() if int(rows[cand[0]]["seed"]) != args.val_seed}
        val = {key: cand for key, cand in group_candidates.items() if int(rows[cand[0]]["seed"]) == args.val_seed}
        return train, val

    if args.split_mode == "real_seed_within_seed":
        if val_real_seed is None:
            raise SystemExit("--val-real-seed is required for real_seed_within_seed")
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
        return train, val

    if args.split_mode == "source_modulo_within_seed":
        if val_source_remainder is None:
            raise SystemExit("--val-source-remainder is required for source_modulo_within_seed")
        train = {
            key: cand
            for key, cand in group_candidates.items()
            if int(rows[cand[0]]["seed"]) == args.val_seed
            and int(rows[cand[0]]["source_index"]) % args.source_modulo != val_source_remainder
        }
        val = {
            key: cand
            for key, cand in group_candidates.items()
            if int(rows[cand[0]]["seed"]) == args.val_seed
            and int(rows[cand[0]]["source_index"]) % args.source_modulo == val_source_remainder
        }
        return train, val

    raise SystemExit(f"unknown split mode: {args.split_mode}")


def train_one_split(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, int, int, int], list[int]],
    features: np.ndarray,
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    *,
    score_model_seed: int,
    selector_seed: int,
    val_real_seed: int | None,
    val_source_remainder: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(selector_seed)
    np.random.seed(selector_seed)
    torch.manual_seed(selector_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_groups, val_groups = split_groups(
        rows,
        group_candidates,
        args,
        val_real_seed=val_real_seed,
        val_source_remainder=val_source_remainder,
    )
    if not train_groups or not val_groups:
        raise SystemExit(
            f"empty split for mode={args.split_mode} val_real_seed={val_real_seed} "
            f"val_source_remainder={val_source_remainder}: train={len(train_groups)} val={len(val_groups)}"
        )

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
    harmful_positions = []
    nearest_positions = []
    for candidates in train_groups.values():
        target_matches = [idx for idx in candidates if int(rows[idx]["label_assignment_oracle_choice"])]
        if target_matches:
            target = target_matches[0]
        else:
            target = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        train_positions.append(torch.tensor(candidates, dtype=torch.long, device=device))
        train_targets.append(torch.tensor([candidates.index(target)], dtype=torch.long, device=device))
        nearest = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        for idx in candidates:
            if idx != nearest and (
                int(rows[idx]["parseq_delta_vs_nearest"]) > 0 or int(rows[idx]["tesseract_delta_vs_nearest"]) > 0
            ):
                harmful_positions.append(idx)
                nearest_positions.append(nearest)

    harmful_positions_t = torch.tensor(harmful_positions, dtype=torch.long, device=device)
    nearest_positions_t = torch.tensor(nearest_positions, dtype=torch.long, device=device)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x)
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
        logits_np = model(x).detach().cpu().numpy()

    train_margin = base.tune_margin(rows, train_groups, full_train_groups, logits_np)
    policies = {
        "nearest": base.policy_metrics_by_group(
            rows,
            full_val_groups,
            [next(idx for idx in cand if int(rows[idx]["is_nearest"])) for cand in val_groups.values()],
        ),
        "shortlist_oracle": base.policy_metrics_by_group(rows, full_val_groups, base.oracle_select(rows, val_groups)),
        "argmax": base.policy_metrics_by_group(rows, full_val_groups, base.select_argmax(rows, val_groups, logits_np)),
        "train_tuned_margin": base.policy_metrics_by_group(
            rows,
            full_val_groups,
            base.select_argmax(rows, val_groups, logits_np, margin=train_margin),
        ),
    }

    return {
        "split_mode": args.split_mode,
        "val_seed": int(args.val_seed),
        "val_real_seed": None if val_real_seed is None else int(val_real_seed),
        "val_source_remainder": None if val_source_remainder is None else int(val_source_remainder),
        "score_model_seed": int(score_model_seed),
        "selector_seed": int(selector_seed),
        "device": str(device),
        "feature_dim": int(features.shape[1]),
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
        key = (
            f"real_seed_{run['val_real_seed']}"
            if run["split_mode"] == "real_seed_within_seed"
            else f"source_rem_{run['val_source_remainder']}"
            if run["split_mode"] == "source_modulo_within_seed"
            else f"seed_{run['val_seed']}"
        )
        splits.setdefault(key, []).append(run)
    return {key: aggregate(value) for key, value in sorted(splits.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--split-mode", choices=["seed", "real_seed_within_seed", "source_modulo_within_seed"], default="seed")
    parser.add_argument("--val-real-seed", type=int, action="append")
    parser.add_argument("--source-modulo", type=int, default=5)
    parser.add_argument("--val-source-remainder", type=int, action="append")
    parser.add_argument("--score-model-seed", type=int, action="append")
    parser.add_argument("--selector-seed", type=int, action="append")
    parser.add_argument("--shortlist-topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--harmful-margin-weight", type=float, default=1.0)
    parser.add_argument("--harmful-margin", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.split_mode == "real_seed_within_seed" and not args.val_real_seed:
        args.val_real_seed = [0, 1, 2]
    if args.split_mode == "source_modulo_within_seed" and not args.val_source_remainder:
        args.val_source_remainder = list(range(args.source_modulo))
    if not args.score_model_seed:
        args.score_model_seed = [0, 1, 2]
    if not args.selector_seed:
        args.selector_seed = [0, 1, 2]
    if args.split_mode == "seed":
        split_values: list[tuple[int | None, int | None]] = [(None, None)]
    elif args.split_mode == "real_seed_within_seed":
        split_values = [(value, None) for value in args.val_real_seed]
    else:
        split_values = [(None, value) for value in args.val_source_remainder]

    rows = base.read_jsonl(args.table)
    groups = base.grouped_indices(rows)
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
        features = base.build_selector_features(rows, groups, base_features, oracle_scores, safe_scores)
        group_candidates = {
            key: base.candidate_indices_for_group(rows, indices, oracle_scores, safe_scores, topk=args.shortlist_topk)
            for key, indices in groups.items()
        }
        for val_real_seed, val_source_remainder in split_values:
            for selector_seed in args.selector_seed:
                runs.append(
                    train_one_split(
                        rows,
                        groups,
                        features,
                        group_candidates,
                        score_model_seed=score_model_seed,
                        selector_seed=selector_seed,
                        val_real_seed=val_real_seed,
                        val_source_remainder=val_source_remainder,
                        args=args,
                    )
                )

    result = {
        "description": "Top-k assignment policy selector split diagnostic. Non-promotion diagnostic only.",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "val_seed": int(args.val_seed),
            "split_mode": args.split_mode,
            "val_real_seeds": args.val_real_seed,
            "source_modulo": int(args.source_modulo),
            "val_source_remainders": args.val_source_remainder,
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
