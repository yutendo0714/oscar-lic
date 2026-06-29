#!/usr/bin/env python3
"""Train an image-level listwise patch gate for one-candidate utility selection."""

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
    split_by_group,
    tune_threshold,
)
from train_candidate_patch_ranker import (
    LATENT_TO_PIXEL,
    PatchRanker,
    aggregate,
    build_patch_tensor,
    build_tabular,
    compact_policy,
)
from train_candidate_utility_baseline import average_precision, roc_auc, topk


def group_indices(rows: list[dict], mask: np.ndarray | None = None) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if mask is not None and not bool(mask[index]):
            continue
        grouped[int(row["source_index"])].append(index)
    return grouped


def row_is_positive(row: dict, target_policy: str) -> bool:
    if target_policy == "multi_teacher":
        return int(row["label_multi_teacher_improves"]) > 0
    if target_policy == "pareto":
        return int(row["label_pareto_improves"]) > 0
    if target_policy == "tesseract_parseq_safe":
        return int(row["tesseract_delta_distance"]) < 0 and int(row["parseq_delta_distance"]) <= 0
    if target_policy == "parseq_tesseract_safe":
        return int(row["parseq_delta_distance"]) < 0 and int(row["tesseract_delta_distance"]) <= 0
    raise ValueError(f"unsupported target policy: {target_policy}")


def target_for_group(rows: list[dict], indices: list[int], target_policy: str) -> int:
    best_local = None
    best_key = None
    for local, index in enumerate(indices, start=1):
        row = rows[index]
        delta = int(row["multi_teacher_delta_distance"])
        if target_policy == "multi_teacher":
            key = (delta, -int(row["improving_evaluator_count"]))
        elif target_policy == "pareto":
            if int(row["worsening_evaluator_count"]) > 0:
                continue
            key = (delta, -int(row["improving_evaluator_count"]))
        elif target_policy == "tesseract_parseq_safe":
            if int(row["parseq_delta_distance"]) > 0 or int(row["tesseract_delta_distance"]) >= 0:
                continue
            key = (
                int(row["tesseract_delta_distance"]),
                int(row["multi_teacher_delta_distance"]),
                int(row["parseq_delta_distance"]),
            )
        elif target_policy == "parseq_tesseract_safe":
            if int(row["tesseract_delta_distance"]) > 0 or int(row["parseq_delta_distance"]) >= 0:
                continue
            key = (
                int(row["parseq_delta_distance"]),
                int(row["multi_teacher_delta_distance"]),
                int(row["tesseract_delta_distance"]),
            )
        else:
            raise ValueError(f"unsupported target policy: {target_policy}")
        if best_key is None or key < best_key:
            best_key = key
            best_local = local
    if best_key is not None:
        if target_policy in {"multi_teacher", "pareto"} and best_key[0] >= 0:
            return 0
        assert best_local is not None
        return best_local
    return 0


class ListwisePatchGate(torch.nn.Module):
    def __init__(self, tab_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.rank = PatchRanker(tab_dim=tab_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.noop_logit = torch.nn.Parameter(torch.zeros(()))

    def forward(self, patches: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.rank(patches, tabular)


def train_once(
    args: argparse.Namespace,
    rows: list[dict],
    patches: torch.Tensor,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    tabular, tabular_names = build_tabular(rows)
    labels = np.asarray([int(row_is_positive(row, args.target_policy)) for row in rows], dtype=np.float32)
    multi_teacher_labels = np.asarray([int(row["label_multi_teacher_improves"]) for row in rows], dtype=np.float32)
    groups = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    train_mask, val_mask = split_by_group(groups, args.val_fraction, seed)
    train_indices = np.flatnonzero(train_mask)
    x_train = tabular[train_mask]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    tab_all = torch.from_numpy((tabular - mean) / std).float().to(args.device)
    patches = patches.to(args.device)
    model = ListwisePatchGate(tab_all.shape[1], args.hidden_dim, args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_groups = group_indices(rows, train_mask)
    group_targets = {
        source_index: target_for_group(rows, indices, args.target_policy)
        for source_index, indices in train_groups.items()
    }

    for _ in range(args.epochs):
        opt.zero_grad()
        logits = model(patches, tab_all)
        losses = []
        for source_index, indices in train_groups.items():
            tensor_indices = torch.tensor(indices, dtype=torch.long, device=args.device)
            scores = torch.cat([model.noop_logit.reshape(1), logits[tensor_indices]])
            target = torch.tensor([group_targets[source_index]], dtype=torch.long, device=args.device)
            losses.append(F.cross_entropy(scores.reshape(1, -1), target))
        loss = torch.stack(losses).mean()
        if args.aux_bce_weight > 0:
            train_tensor_indices = torch.from_numpy(train_indices).long().to(args.device)
            y_train = torch.from_numpy(labels[train_mask]).float().to(args.device)
            row_logits = logits[train_tensor_indices] - model.noop_logit
            positives = float(labels[train_mask].sum())
            negatives = float(len(train_indices) - positives)
            bce = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=args.device)
            )
            loss = loss + args.aux_bce_weight * bce(row_logits, y_train)
        if args.harm_logit_penalty > 0:
            train_tensor_indices = torch.from_numpy(train_indices).long().to(args.device)
            row_logits = logits[train_tensor_indices] - model.noop_logit
            harm_mask = torch.tensor(
                [int(rows[index]["worsening_evaluator_count"]) > 0 for index in train_indices],
                dtype=torch.bool,
                device=args.device,
            )
            if bool(harm_mask.any()):
                loss = loss + args.harm_logit_penalty * F.softplus(row_logits[harm_mask]).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        candidate_logits = model(patches, tab_all)
        scores = torch.sigmoid(candidate_logits - model.noop_logit).detach().cpu().numpy()

    y_train_np = labels[train_mask]
    y_val_np = labels[val_mask]
    y_val_multi_np = multi_teacher_labels[val_mask]
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
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_groups": int(len(train_groups)),
        "train_positive_group_rate": float(
            np.mean([1 if group_targets[source_index] > 0 else 0 for source_index in train_groups])
        ),
        "train_positive_rate": float(y_train_np.mean()),
        "val_positive_rate": float(y_val_np.mean()),
        "val_multi_teacher_positive_rate": float(y_val_multi_np.mean()),
        "ranker_train_auc": roc_auc(scores[train_mask], y_train_np),
        "ranker_val_auc": roc_auc(scores[val_mask], y_val_np),
        "ranker_val_ap": average_precision(scores[val_mask], y_val_np),
        "ranker_val_topk": topk(scores[val_mask], y_val_np, [10, 20, 40]),
        "ranker_val_multi_teacher_auc": roc_auc(scores[val_mask], y_val_multi_np),
        "ranker_val_multi_teacher_ap": average_precision(scores[val_mask], y_val_multi_np),
    }
    return scores, train_mask, val_mask, model_meta


def tune_evaluator_constrained_threshold(
    grouped: dict[int, list[tuple[int, dict]]],
    scores: np.ndarray,
    primary_evaluator: str,
    protected_evaluator: str,
    protected_max_delta: int,
) -> dict:
    thresholds = sorted({float(scores[row_index]) for entries in grouped.values() for row_index, _ in entries})
    if thresholds:
        thresholds = [max(thresholds) + 1e-6] + thresholds
    else:
        thresholds = [1.0]
    primary_key = f"{primary_evaluator}_delta_distance"
    protected_key = f"{protected_evaluator}_delta_distance"
    best = None
    for threshold in thresholds:
        selected = choose_threshold(grouped, scores, threshold)
        primary_delta = sum(int(item[1][primary_key]) for item in selected.values() if item is not None)
        protected_delta = sum(int(item[1][protected_key]) for item in selected.values() if item is not None)
        multi_delta = sum(int(item[1]["multi_teacher_delta_distance"]) for item in selected.values() if item is not None)
        selected_count = sum(1 for item in selected.values() if item is not None)
        violation = max(0, protected_delta - protected_max_delta)
        candidate = {
            "threshold": float(threshold),
            "primary_evaluator": primary_evaluator,
            "protected_evaluator": protected_evaluator,
            "protected_max_delta": int(protected_max_delta),
            "train_primary_delta_distance_sum": int(primary_delta),
            "train_protected_delta_distance_sum": int(protected_delta),
            "train_multi_teacher_delta_distance_sum": int(multi_delta),
            "train_selected_images": int(selected_count),
            "train_protected_constraint_violation": int(violation),
        }
        key = (
            0 if violation == 0 else 1,
            violation,
            primary_delta,
            multi_delta,
            selected_count,
        )
        if best is None or key < best[0]:
            best = (key, candidate)
    assert best is not None
    return best[1]


def run_seed(args: argparse.Namespace, rows: list[dict], patches: torch.Tensor, analyses: dict, bpp_by_key: dict, seed: int) -> dict:
    scores, train_mask, val_mask, model_meta = train_once(args, rows, patches, seed)
    baseline_feature, residual_scores = baseline_score_feature(rows)
    grouped_train = rows_by_image(rows, np.flatnonzero(train_mask))
    grouped_val = rows_by_image(rows, np.flatnonzero(val_mask))
    tuned = tune_threshold(grouped_train, scores)
    constrained_tuned = None
    if args.primary_threshold_evaluator and args.protected_threshold_evaluator:
        constrained_tuned = tune_evaluator_constrained_threshold(
            grouped_train,
            scores,
            args.primary_threshold_evaluator,
            args.protected_threshold_evaluator,
            args.protected_max_delta,
        )
    policy_specs = [
        ("base", choose_base(grouped_val)),
        ("oracle_single_candidate", choose_oracle(grouped_val)),
        ("listwise_gate_threshold_0", choose_threshold(grouped_val, scores, 0.5)),
        ("listwise_gate_train_tuned", choose_threshold(grouped_val, scores, tuned["threshold"])),
    ]
    if constrained_tuned is not None:
        policy_specs.append(
            (
                f"listwise_gate_{args.primary_threshold_evaluator}_{args.protected_threshold_evaluator}_constrained_tuned",
                choose_threshold(grouped_val, scores, constrained_tuned["threshold"]),
            )
        )
    for budget in args.image_budget:
        budget = min(int(budget), len(grouped_val))
        policy_specs.append((f"listwise_gate_top{budget}_images", choose_top_images(grouped_val, scores, budget)))
        policy_specs.append((f"residual_rms_top{budget}_images", choose_top_images(grouped_val, residual_scores, budget)))
    return {
        "seed": seed,
        "model": model_meta,
        "residual_baseline_score_feature": baseline_feature,
        "train_tuned_threshold": tuned,
        "train_constrained_threshold": constrained_tuned,
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
