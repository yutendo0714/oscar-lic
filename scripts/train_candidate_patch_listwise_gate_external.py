#!/usr/bin/env python3
"""Train a one-candidate patch gate on one split and evaluate it on another."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
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
    tune_threshold,
)
from train_candidate_patch_listwise_gate import (
    ListwisePatchGate,
    group_indices,
    row_is_positive,
    target_for_group,
    tune_evaluator_constrained_threshold,
)
from train_candidate_patch_ranker import (
    LATENT_TO_PIXEL,
    aggregate,
    build_patch_tensor,
    build_tabular,
    compact_policy,
)
from train_candidate_utility_baseline import average_precision, roc_auc, topk


def tabular_with_names(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float32)


def fit_seed(
    args: argparse.Namespace,
    train_rows: list[dict],
    train_patches: torch.Tensor,
    eval_rows: list[dict],
    eval_patches: torch.Tensor,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    train_tabular, tabular_names = build_tabular(train_rows)
    eval_tabular = tabular_with_names(eval_rows, tabular_names)
    mean = train_tabular.mean(axis=0, keepdims=True)
    std = train_tabular.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    train_tab = torch.from_numpy((train_tabular - mean) / std).float().to(args.device)
    eval_tab = torch.from_numpy((eval_tabular - mean) / std).float().to(args.device)
    train_patches = train_patches.to(args.device)
    eval_patches = eval_patches.to(args.device)

    model = ListwisePatchGate(train_tab.shape[1], args.hidden_dim, args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_groups = group_indices(train_rows)
    group_targets = {
        source_index: target_for_group(train_rows, indices, args.target_policy)
        for source_index, indices in train_groups.items()
    }
    labels = np.asarray([int(row_is_positive(row, args.target_policy)) for row in train_rows], dtype=np.float32)
    train_indices = np.arange(len(train_rows), dtype=np.int64)

    for _ in range(args.epochs):
        opt.zero_grad()
        logits = model(train_patches, train_tab)
        losses = []
        for source_index, indices in train_groups.items():
            tensor_indices = torch.tensor(indices, dtype=torch.long, device=args.device)
            scores = torch.cat([model.noop_logit.reshape(1), logits[tensor_indices]])
            target = torch.tensor([group_targets[source_index]], dtype=torch.long, device=args.device)
            losses.append(F.cross_entropy(scores.reshape(1, -1), target))
        loss = torch.stack(losses).mean()
        if args.aux_bce_weight > 0:
            y_train = torch.from_numpy(labels).float().to(args.device)
            row_logits = logits - model.noop_logit
            positives = float(labels.sum())
            negatives = float(len(labels) - positives)
            bce = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=args.device)
            )
            loss = loss + args.aux_bce_weight * bce(row_logits, y_train)
        if args.harm_logit_penalty > 0:
            row_logits = logits - model.noop_logit
            harm_mask = torch.tensor(
                [int(row["worsening_evaluator_count"]) > 0 for row in train_rows],
                dtype=torch.bool,
                device=args.device,
            )
            if bool(harm_mask.any()):
                loss = loss + args.harm_logit_penalty * F.softplus(row_logits[harm_mask]).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_logits = model(train_patches, train_tab)
        eval_logits = model(eval_patches, eval_tab)
        train_scores = torch.sigmoid(train_logits - model.noop_logit).detach().cpu().numpy()
        eval_scores = torch.sigmoid(eval_logits - model.noop_logit).detach().cpu().numpy()

    eval_labels = np.asarray([int(row_is_positive(row, args.target_policy)) for row in eval_rows], dtype=np.float32)
    eval_multi_labels = np.asarray([int(row["label_multi_teacher_improves"]) for row in eval_rows], dtype=np.float32)
    model_meta = {
        "seed": seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "patch_size": args.patch_size,
        "aux_bce_weight": args.aux_bce_weight,
        "harm_logit_penalty": args.harm_logit_penalty,
        "target_policy": args.target_policy,
        "noop_logit": float(model.noop_logit.detach().cpu().item()),
        "tabular_features": tabular_names,
        "train_rows": int(len(train_rows)),
        "eval_rows": int(len(eval_rows)),
        "train_groups": int(len(train_groups)),
        "eval_groups": int(len(group_indices(eval_rows))),
        "train_positive_group_rate": float(
            np.mean([1 if group_targets[source_index] > 0 else 0 for source_index in train_groups])
        ),
        "train_positive_rate": float(labels.mean()),
        "eval_positive_rate": float(eval_labels.mean()),
        "eval_multi_teacher_positive_rate": float(eval_multi_labels.mean()),
        "ranker_train_auc": roc_auc(train_scores, labels),
        "ranker_train_ap": average_precision(train_scores, labels),
        "ranker_train_topk": topk(train_scores, labels, [10, 20, 40]),
        "ranker_eval_auc": roc_auc(eval_scores, eval_labels),
        "ranker_eval_ap": average_precision(eval_scores, eval_labels),
        "ranker_eval_topk": topk(eval_scores, eval_labels, [10, 20, 40]),
        "ranker_eval_multi_teacher_auc": roc_auc(eval_scores, eval_multi_labels),
        "ranker_eval_multi_teacher_ap": average_precision(eval_scores, eval_multi_labels),
    }
    return train_scores, eval_scores, model_meta


def run_seed(
    args: argparse.Namespace,
    train_rows: list[dict],
    train_patches: torch.Tensor,
    eval_rows: list[dict],
    eval_patches: torch.Tensor,
    train_analyses: dict,
    eval_analyses: dict,
    train_bpp_by_key: dict,
    eval_bpp_by_key: dict,
    seed: int,
) -> dict:
    train_scores, eval_scores, model_meta = fit_seed(args, train_rows, train_patches, eval_rows, eval_patches, seed)
    baseline_feature, eval_residual_scores = baseline_score_feature(eval_rows)
    _, train_residual_scores = baseline_score_feature(train_rows)
    grouped_train = rows_by_image(train_rows, np.arange(len(train_rows), dtype=np.int64))
    grouped_eval = rows_by_image(eval_rows, np.arange(len(eval_rows), dtype=np.int64))
    tuned = tune_threshold(grouped_train, train_scores)
    constrained_tuned = None
    if args.primary_threshold_evaluator and args.protected_threshold_evaluator:
        constrained_tuned = tune_evaluator_constrained_threshold(
            grouped_train,
            train_scores,
            args.primary_threshold_evaluator,
            args.protected_threshold_evaluator,
            args.protected_max_delta,
        )
    train_policy_specs = [
        ("base", choose_base(grouped_train)),
        ("oracle_single_candidate", choose_oracle(grouped_train)),
        ("listwise_gate_threshold_0", choose_threshold(grouped_train, train_scores, 0.5)),
        ("listwise_gate_train_tuned", choose_threshold(grouped_train, train_scores, tuned["threshold"])),
    ]
    eval_policy_specs = [
        ("base", choose_base(grouped_eval)),
        ("oracle_single_candidate", choose_oracle(grouped_eval)),
        ("listwise_gate_threshold_0", choose_threshold(grouped_eval, eval_scores, 0.5)),
        ("listwise_gate_train_tuned", choose_threshold(grouped_eval, eval_scores, tuned["threshold"])),
    ]
    if constrained_tuned is not None:
        name = f"listwise_gate_{args.primary_threshold_evaluator}_{args.protected_threshold_evaluator}_constrained_tuned"
        train_policy_specs.append((name, choose_threshold(grouped_train, train_scores, constrained_tuned["threshold"])))
        eval_policy_specs.append((name, choose_threshold(grouped_eval, eval_scores, constrained_tuned["threshold"])))
    for budget in args.image_budget:
        train_budget = min(int(budget), len(grouped_train))
        eval_budget = min(int(budget), len(grouped_eval))
        eval_policy_specs.append((f"listwise_gate_top{eval_budget}_images", choose_top_images(grouped_eval, eval_scores, eval_budget)))
        eval_policy_specs.append(
            (f"residual_rms_top{eval_budget}_images", choose_top_images(grouped_eval, eval_residual_scores, eval_budget))
        )
        train_policy_specs.append((f"listwise_gate_top{train_budget}_images", choose_top_images(grouped_train, train_scores, train_budget)))
        train_policy_specs.append(
            (f"residual_rms_top{train_budget}_images", choose_top_images(grouped_train, train_residual_scores, train_budget))
        )
    return {
        "seed": seed,
        "model": model_meta,
        "residual_baseline_score_feature": baseline_feature,
        "train_tuned_threshold": tuned,
        "train_constrained_threshold": constrained_tuned,
        "train_policies": [
            evaluate_policy(name, selected, grouped_train, train_analyses, train_bpp_by_key, args.profile)
            for name, selected in train_policy_specs
        ],
        "eval_policies": [
            evaluate_policy(name, selected, grouped_eval, eval_analyses, eval_bpp_by_key, args.profile)
            for name, selected in eval_policy_specs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-table", type=Path, required=True)
    parser.add_argument("--train-base-recon-split", type=Path, required=True)
    parser.add_argument("--train-candidate-split", type=Path, required=True)
    parser.add_argument("--eval-table", type=Path, required=True)
    parser.add_argument("--eval-base-recon-split", type=Path, required=True)
    parser.add_argument("--eval-candidate-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-evaluator", action="append", nargs=2, metavar=("LABEL", "ANALYSIS_JSON"))
    parser.add_argument("--eval-evaluator", action="append", nargs=2, metavar=("LABEL", "ANALYSIS_JSON"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--aux-bce-weight", type=float, default=0.1)
    parser.add_argument("--harm-logit-penalty", type=float, default=0.0)
    parser.add_argument(
        "--target-policy",
        choices=["multi_teacher", "pareto", "tesseract_parseq_safe", "parseq_tesseract_safe"],
        default="multi_teacher",
    )
    parser.add_argument("--primary-threshold-evaluator")
    parser.add_argument("--protected-threshold-evaluator")
    parser.add_argument("--protected-max-delta", type=int, default=0)
    parser.add_argument("--latent-to-pixel", type=int, default=LATENT_TO_PIXEL)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--image-budget", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--base-structural-overhead-bytes", type=int, default=88)
    args = parser.parse_args()

    if not args.train_evaluator or not args.eval_evaluator:
        raise SystemExit("provide --train-evaluator and --eval-evaluator entries")
    train_labels = [label for label, _ in args.train_evaluator]
    eval_labels = [label for label, _ in args.eval_evaluator]
    if train_labels != eval_labels:
        raise SystemExit(f"train/eval evaluator labels must match, got {train_labels} vs {eval_labels}")
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and args.device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    train_rows = read_jsonl(args.train_table)
    eval_rows = read_jsonl(args.eval_table)
    train_patches = build_patch_tensor(train_rows, args.train_base_recon_split, args.patch_size, args.latent_to_pixel)
    eval_patches = build_patch_tensor(eval_rows, args.eval_base_recon_split, args.patch_size, args.latent_to_pixel)
    train_analyses = {label: read_analysis(Path(path)) for label, path in args.train_evaluator}
    eval_analyses = {label: read_analysis(Path(path)) for label, path in args.eval_evaluator}
    train_bpp_by_key = candidate_split_maps(args.train_candidate_split, args.base_structural_overhead_bytes)
    eval_bpp_by_key = candidate_split_maps(args.eval_candidate_split, args.base_structural_overhead_bytes)
    per_seed = [
        run_seed(
            args,
            train_rows,
            train_patches,
            eval_rows,
            eval_patches,
            train_analyses,
            eval_analyses,
            train_bpp_by_key,
            eval_bpp_by_key,
            seed,
        )
        for seed in args.seeds
    ]

    train_by_policy = defaultdict(list)
    eval_by_policy = defaultdict(list)
    for seed_result in per_seed:
        for policy in seed_result["train_policies"]:
            train_by_policy[policy["name"]].append(compact_policy(policy, eval_labels))
        for policy in seed_result["eval_policies"]:
            eval_by_policy[policy["name"]].append(compact_policy(policy, eval_labels))

    output = {
        "train_table": str(args.train_table),
        "train_base_recon_split": str(args.train_base_recon_split),
        "train_candidate_split": str(args.train_candidate_split),
        "eval_table": str(args.eval_table),
        "eval_base_recon_split": str(args.eval_base_recon_split),
        "eval_candidate_split": str(args.eval_candidate_split),
        "profile": args.profile,
        "device": args.device,
        "seeds": args.seeds,
        "evaluators": eval_labels,
        "per_seed": per_seed,
        "aggregate_train_by_policy": {name: aggregate(rows) for name, rows in sorted(train_by_policy.items())},
        "aggregate_eval_by_policy": {name: aggregate(rows) for name, rows in sorted(eval_by_policy.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate_eval_by_policy"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
