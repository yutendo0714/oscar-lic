#!/usr/bin/env python3
"""Train a top-k assignment selector with an explicit OCR reward objective.

This is a non-promotion diagnostic. It reuses the existing nearest-plus-top-k
shortlist construction, but changes the learning objective from oracle-choice
classification to a small policy-level reward/cost over train OCR deltas.
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


def candidate_reward(row: dict[str, Any], args: argparse.Namespace) -> float:
    if int(row["is_nearest"]):
        return 0.0

    parseq_delta = int(row["parseq_delta_vs_nearest"])
    tess_delta = int(row["tesseract_delta_vs_nearest"])
    reward = -float(tess_delta)

    if parseq_delta > 0:
        reward -= args.parseq_worsen_penalty * float(parseq_delta)
        reward -= args.false_change_penalty

    if tess_delta >= 0:
        reward -= args.false_change_penalty
    if tess_delta > 0:
        reward -= args.tesseract_worsen_penalty * float(tess_delta)

    return float(np.clip(reward, -args.reward_clip, args.reward_clip))


def reward_oracle_select(
    rows: list[dict[str, Any]],
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    rewards: np.ndarray,
) -> list[int]:
    selected = []
    for candidates in group_candidates.values():
        nearest = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        best = max(candidates, key=lambda idx: (float(rewards[idx]), -int(rows[idx]["topk_rank"])))
        selected.append(best if float(rewards[best]) > 0.0 else nearest)
    return selected


def tune_reward_margin(
    rows: list[dict[str, Any]],
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    metric_groups: dict[tuple[int, int, int, int], list[int]],
    logits: np.ndarray,
) -> float:
    margins = []
    for candidates in group_candidates.values():
        nearest = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        best = max(candidates, key=lambda idx: (float(logits[idx]), -int(rows[idx]["topk_rank"])))
        margins.append(float(logits[best]) - float(logits[nearest]))
    candidates = sorted(set(margins))
    if not candidates:
        return 1e6
    candidates = [max(candidates) + 1e-6] + candidates + [min(candidates) - 1e-6]

    best_margin = candidates[0]
    best_key = None
    for margin in candidates:
        selected = base.select_argmax(rows, group_candidates, logits, margin=margin)
        metrics = base.policy_metrics_by_group(rows, metric_groups, selected)
        key = (
            metrics["parseq_worsen_groups"],
            metrics["tesseract_worsen_groups"],
            metrics["tesseract_delta_sum"],
            metrics["false_changes"],
            metrics["wrong_changes"],
            metrics["changed_groups"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_margin = float(margin)
    return best_margin


def train_one_split(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, int, int, int], list[int]],
    features: np.ndarray,
    rewards: np.ndarray,
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
    train_target_probs = []
    positive_positions = []
    positive_nearest = []
    positive_margins = []
    nonpositive_positions = []
    nonpositive_nearest = []
    for candidates in train_groups.values():
        nearest = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        pos = torch.tensor(candidates, dtype=torch.long, device=device)
        group_rewards = torch.tensor([float(rewards[idx]) for idx in candidates], dtype=torch.float32, device=device)
        if float(group_rewards.max().detach().cpu()) <= 0.0:
            target = torch.zeros_like(group_rewards)
            target[candidates.index(nearest)] = 1.0
        else:
            target = torch.softmax(group_rewards / args.reward_temperature, dim=0)
        train_positions.append(pos)
        train_target_probs.append(target)

        for idx in candidates:
            if idx == nearest:
                continue
            reward = float(rewards[idx])
            if reward > 0:
                positive_positions.append(idx)
                positive_nearest.append(nearest)
                positive_margins.append(min(args.positive_margin_cap, args.positive_margin_scale * reward))
            else:
                nonpositive_positions.append(idx)
                nonpositive_nearest.append(nearest)

    positive_positions_t = torch.tensor(positive_positions, dtype=torch.long, device=device)
    positive_nearest_t = torch.tensor(positive_nearest, dtype=torch.long, device=device)
    positive_margins_t = torch.tensor(positive_margins, dtype=torch.float32, device=device)
    nonpositive_positions_t = torch.tensor(nonpositive_positions, dtype=torch.long, device=device)
    nonpositive_nearest_t = torch.tensor(nonpositive_nearest, dtype=torch.long, device=device)
    reward_t = torch.from_numpy(rewards.astype(np.float32)).to(device)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x)
        losses = []
        for pos, target_probs in zip(train_positions, train_target_probs):
            log_probs = torch.nn.functional.log_softmax(logits[pos], dim=0)
            losses.append(-(target_probs * log_probs).sum())
            if args.expected_reward_weight > 0:
                probs = torch.softmax(logits[pos], dim=0)
                losses.append(-args.expected_reward_weight * (probs * reward_t[pos]).sum())
        loss = torch.stack(losses).mean()
        if len(positive_positions_t) and args.positive_margin_weight > 0:
            pos_loss = torch.nn.functional.softplus(
                logits[positive_nearest_t] - logits[positive_positions_t] + positive_margins_t
            ).mean()
            loss = loss + args.positive_margin_weight * pos_loss
        if len(nonpositive_positions_t) and args.nonpositive_margin_weight > 0:
            nonpos_loss = torch.nn.functional.softplus(
                logits[nonpositive_positions_t] - logits[nonpositive_nearest_t] + args.nonpositive_margin
            ).mean()
            loss = loss + args.nonpositive_margin_weight * nonpos_loss
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits_np = model(x).detach().cpu().numpy()

    train_margin = base.tune_margin(rows, train_groups, full_train_groups, logits_np)
    reward_margin = tune_reward_margin(rows, train_groups, full_train_groups, logits_np)
    policies = {
        "nearest": base.policy_metrics_by_group(
            rows,
            full_val_groups,
            [next(idx for idx in cand if int(rows[idx]["is_nearest"])) for cand in val_groups.values()],
        ),
        "label_oracle": base.policy_metrics_by_group(rows, full_val_groups, base.oracle_select(rows, val_groups)),
        "reward_oracle": base.policy_metrics_by_group(rows, full_val_groups, reward_oracle_select(rows, val_groups, rewards)),
        "argmax": base.policy_metrics_by_group(rows, full_val_groups, base.select_argmax(rows, val_groups, logits_np)),
        "train_tuned_margin": base.policy_metrics_by_group(
            rows,
            full_val_groups,
            base.select_argmax(rows, val_groups, logits_np, margin=train_margin),
        ),
        "train_tuned_reward_margin": base.policy_metrics_by_group(
            rows,
            full_val_groups,
            base.select_argmax(rows, val_groups, logits_np, margin=reward_margin),
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
        "train_reward_margin": float(reward_margin),
        "candidate_count_mean": float(np.mean([len(candidates) for candidates in val_groups.values()])),
        "train_positive_reward_groups": int(
            sum(1 for candidates in train_groups.values() if max(float(rewards[idx]) for idx in candidates) > 0)
        ),
        "val_positive_reward_groups": int(
            sum(1 for candidates in val_groups.values() if max(float(rewards[idx]) for idx in candidates) > 0)
        ),
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
    output["train_positive_reward_groups_mean"] = (
        float(np.mean([run["train_positive_reward_groups"] for run in runs])) if runs else 0.0
    )
    output["val_positive_reward_groups_mean"] = (
        float(np.mean([run["val_positive_reward_groups"] for run in runs])) if runs else 0.0
    )
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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--reward-temperature", type=float, default=0.5)
    parser.add_argument("--reward-clip", type=float, default=8.0)
    parser.add_argument("--false-change-penalty", type=float, default=1.0)
    parser.add_argument("--parseq-worsen-penalty", type=float, default=4.0)
    parser.add_argument("--tesseract-worsen-penalty", type=float, default=1.0)
    parser.add_argument("--expected-reward-weight", type=float, default=0.2)
    parser.add_argument("--positive-margin-weight", type=float, default=0.5)
    parser.add_argument("--positive-margin-scale", type=float, default=0.25)
    parser.add_argument("--positive-margin-cap", type=float, default=1.0)
    parser.add_argument("--nonpositive-margin-weight", type=float, default=1.0)
    parser.add_argument("--nonpositive-margin", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    rewards = np.asarray([candidate_reward(row, args) for row in rows], dtype=np.float32)

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
                        rewards,
                        group_candidates,
                        score_model_seed=score_model_seed,
                        selector_seed=selector_seed,
                        val_real_seed=val_real_seed,
                        val_source_remainder=val_source_remainder,
                        args=args,
                    )
                )

    result = {
        "description": "Top-k assignment reward-policy selector diagnostic. Non-promotion diagnostic only.",
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
            "reward_temperature": float(args.reward_temperature),
            "reward_clip": float(args.reward_clip),
            "false_change_penalty": float(args.false_change_penalty),
            "parseq_worsen_penalty": float(args.parseq_worsen_penalty),
            "tesseract_worsen_penalty": float(args.tesseract_worsen_penalty),
            "expected_reward_weight": float(args.expected_reward_weight),
            "positive_margin_weight": float(args.positive_margin_weight),
            "positive_margin_scale": float(args.positive_margin_scale),
            "positive_margin_cap": float(args.positive_margin_cap),
            "nonpositive_margin_weight": float(args.nonpositive_margin_weight),
            "nonpositive_margin": float(args.nonpositive_margin),
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
