#!/usr/bin/env python3
"""Source-robust feature/loss ablation for Eval300 top-8 assignment ranking.

This diagnostic extends N096. It still evaluates candidate ranking only and
does not decide whether to transmit an enhancement candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_trainval_leave_source_out_ranker as lso  # noqa: E402
import train_top8_trainval_tabular_listwise_ranker as ranker  # noqa: E402


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_feature_names(rows: list[dict[str, Any]], family: str) -> list[str]:
    all_names = ranker.numeric_feature_names(rows)
    if family == "all_tabular":
        return all_names
    if family == "continuous_no_onehot":
        return [name for name in all_names if "_onehot_" not in name]
    if family == "codegeom_no_onehot":
        return [
            name
            for name in all_names
            if name in ranker.CORE_FEATURES or (name.startswith("codebook_") and "_onehot_" not in name)
        ]
    if family == "imgdiff_only":
        return [name for name in all_names if name in ranker.CORE_FEATURES or name.startswith("img_")]
    if family == "topk_error_only":
        return [name for name in all_names if name in ranker.CORE_FEATURES]
    raise ValueError(f"unknown feature family: {family}")


def group_source(rows: list[dict[str, Any]], indices: list[int]) -> str:
    return str(rows[indices[0]].get("source", ""))


def source_balanced_loss(
    logits: torch.Tensor,
    train_groups: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    pairwise_weight: float,
    device: torch.device,
) -> torch.Tensor:
    by_source: dict[str, list[torch.Tensor]] = defaultdict(list)
    for group in train_groups:
        idx = torch.tensor(group["indices"], dtype=torch.long, device=device)
        target_local = int(group["target_local"])
        target = torch.tensor([target_local], dtype=torch.long, device=device)
        group_logits = logits[idx].reshape(1, -1)
        loss = F.cross_entropy(group_logits, target)
        if pairwise_weight > 0:
            target_logit = group_logits[0, target_local]
            neg = torch.cat([group_logits[0, :target_local], group_logits[0, target_local + 1 :]])
            if neg.numel():
                loss = loss + pairwise_weight * F.softplus(-(target_logit - neg)).mean()
        by_source[str(rows[group["target_index"]].get("source", ""))].append(loss)
    return torch.stack([torch.stack(losses).mean() for losses in by_source.values()]).mean()


def standard_loss(
    logits: torch.Tensor,
    train_groups: list[dict[str, Any]],
    pairwise_weight: float,
    device: torch.device,
) -> torch.Tensor:
    losses = []
    for group in train_groups:
        idx = torch.tensor(group["indices"], dtype=torch.long, device=device)
        target_local = int(group["target_local"])
        target = torch.tensor([target_local], dtype=torch.long, device=device)
        group_logits = logits[idx].reshape(1, -1)
        losses.append(F.cross_entropy(group_logits, target))
        if pairwise_weight > 0:
            target_logit = group_logits[0, target_local]
            neg = torch.cat([group_logits[0, :target_local], group_logits[0, target_local + 1 :]])
            if neg.numel():
                losses.append(pairwise_weight * F.softplus(-(target_logit - neg)).mean())
    return torch.stack(losses).mean()


def train_model(
    x_all: torch.Tensor,
    rows: list[dict[str, Any]],
    train_groups: list[dict[str, Any]],
    model_seed: int,
    args: argparse.Namespace,
    loss_mode: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
    model = ranker.TabularRanker(x_all.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_all)
        if loss_mode == "standard":
            loss = standard_loss(logits, train_groups, args.pairwise_weight, device)
        elif loss_mode == "source_balanced":
            loss = source_balanced_loss(logits, train_groups, rows, args.pairwise_weight, device)
        else:
            raise ValueError(f"unknown loss mode: {loss_mode}")
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        scores = model(x_all).detach().cpu().numpy()
    train_ranks = []
    for group in train_groups:
        group_scores = scores[group["indices"]]
        target_score = group_scores[int(group["target_local"])]
        train_ranks.append(int(1 + np.sum(group_scores > target_score)))
    return scores, {
        "model_seed": int(model_seed),
        "final_loss": float(last_loss),
        "train_target_rank": ranker.rank_counts(train_ranks),
    }


def aggregate_source_metrics(source_results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    ranks: list[int] = []
    topk: list[int] = []
    for row in source_results:
        metrics = row["metrics"]
        for key in [
            "groups",
            "oracle_change_groups",
            "changed_groups_oracle_change_only",
            "exact",
            "exact_changed_groups",
            "false_change",
            "wrong_change",
            "missed_oracle",
            "tesseract_delta_vs_nearest",
            "parseq_delta_vs_nearest",
        ]:
            counts[key] += int(metrics.get(key, 0))
        for audit in row["audits"]:
            rank_value = audit["oracle_nonnearest_score_rank"]
            if rank_value is not None:
                ranks.append(int(rank_value))
                topk.append(int(audit["oracle_topk_rank"]))
    aggregate = dict(counts)
    aggregate["oracle_nonnearest_score_rank_counts"] = ranker.rank_counts(ranks)
    aggregate["oracle_topk_rank_counts"] = ranker.rank_counts(topk)
    return aggregate


def run_variant(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, str, int, int], list[int]],
    feature_family: str,
    loss_mode: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    feature_names = select_feature_names(rows, feature_family)
    features = ranker.build_features(rows, feature_names)
    sources = sorted({group_source(rows, indices) for indices in groups.values()})
    group_sources = {key: group_source(rows, indices) for key, indices in groups.items()}
    group_nonnearest = {
        key: [idx for idx in indices if int(rows[idx]["code_index"]) != int(rows[idx]["nearest_code"])]
        for key, indices in groups.items()
    }
    source_results = []
    for source_offset, heldout_source in enumerate(sources):
        train_keys = {key for key, source in group_sources.items() if source != heldout_source}
        eval_keys = {key for key, source in group_sources.items() if source == heldout_source}
        train_groups = lso.listwise_groups_for_keys(rows, groups, train_keys)
        train_nonnearest = np.asarray([idx for key in train_keys for idx in group_nonnearest[key]], dtype=np.int64)
        if not len(train_groups) or not len(train_nonnearest):
            continue
        features_norm = ranker.normalize(features[train_nonnearest], features)
        x_all = torch.from_numpy(features_norm).to(device)
        all_scores = []
        model_metas = []
        for offset in range(args.model_seeds):
            seed = args.seed + 100_000 * source_offset + 1_000 * args.feature_families.index(feature_family) + 100 * args.loss_modes.index(loss_mode) + offset
            scores, meta = train_model(x_all, rows, train_groups, seed, args, loss_mode, device)
            all_scores.append(scores)
            model_metas.append(meta)
        scores = np.mean(np.stack(all_scores, axis=0), axis=0)
        metrics, audits = lso.evaluate_source(rows, groups, eval_keys, scores)
        for audit in audits:
            audit["heldout_source"] = heldout_source
        source_results.append(
            {
                "heldout_source": heldout_source,
                "feature_count": len(feature_names),
                "train_groups": len(train_keys),
                "train_listwise_groups": len(train_groups),
                "eval_groups": len(eval_keys),
                "models": model_metas,
                "metrics": metrics,
                "audits": audits,
            }
        )
    return {
        "feature_family": feature_family,
        "loss_mode": loss_mode,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "source_results": source_results,
        "aggregate": aggregate_source_metrics(source_results),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Source leave-out feature/loss ablation for Eval300 top-8 candidate ranking.",
        "This is a diagnostic only: no validation labels are used for promotion and no counted `.oscr` stream is exported.",
        "",
        "## Variant Summary",
        "",
        "| feature family | loss | features | rank<=1 | rank<=4 | exact changed | wrong | Tesseract delta | PARSeq delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in result["variants"]:
        agg = variant["aggregate"]
        ranks = agg["oracle_nonnearest_score_rank_counts"]
        lines.append(
            f"| `{variant['feature_family']}` | `{variant['loss_mode']}` | {variant['feature_count']} | "
            f"{ranks['le1']} | {ranks['le4']} | {agg['exact_changed_groups']} | {agg['wrong_change']} | "
            f"{agg['tesseract_delta_vs_nearest']} | {agg['parseq_delta_vs_nearest']} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_trainval_source_robust_feature_ablation")
    parser.add_argument(
        "--feature-family",
        action="append",
        dest="feature_families",
        default=[],
        choices=["all_tabular", "continuous_no_onehot", "codegeom_no_onehot", "imgdiff_only", "topk_error_only"],
    )
    parser.add_argument(
        "--loss-mode",
        action="append",
        dest="loss_modes",
        default=[],
        choices=["standard", "source_balanced"],
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--model-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.feature_families:
        args.feature_families = ["all_tabular", "continuous_no_onehot", "codegeom_no_onehot", "imgdiff_only", "topk_error_only"]
    if not args.loss_modes:
        args.loss_modes = ["standard", "source_balanced"]
    return args


def main() -> int:
    args = parse_args()
    rows = ranker.read_jsonl(args.table)
    groups = ranker.group_rows(rows)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    variants = []
    for feature_family in args.feature_families:
        for loss_mode in args.loss_modes:
            variants.append(run_variant(rows, groups, feature_family, loss_mode, args, device))
    best_rank1 = max(
        variants,
        key=lambda item: (
            int(item["aggregate"]["oracle_nonnearest_score_rank_counts"]["le1"]),
            int(item["aggregate"]["oracle_nonnearest_score_rank_counts"]["le4"]),
            -int(item["aggregate"]["wrong_change"]),
        ),
    )
    best_rank4 = max(
        variants,
        key=lambda item: (
            int(item["aggregate"]["oracle_nonnearest_score_rank_counts"]["le4"]),
            int(item["aggregate"]["oracle_nonnearest_score_rank_counts"]["le1"]),
            -int(item["aggregate"]["wrong_change"]),
        ),
    )
    interpretation = (
        "If source-balanced loss or removing code one-hot improves leave-source-out rank1/rank4, the next "
        "candidate scorer should reduce code-identity shortcuts and include source-balanced calibration. If image-only "
        "features win, richer OCR-aware image/latent patches are the right evidence path. If all variants remain weak, "
        "the bottleneck is representation depth or label diversity rather than loss weighting."
    )
    scalar_metrics: dict[str, dict[str, float]] = {}
    for variant in variants:
        name = f"{variant['feature_family']}_{variant['loss_mode']}"
        agg = variant["aggregate"]
        ranks = agg["oracle_nonnearest_score_rank_counts"]
        scalar_metrics[f"{name}_rank_le1"] = {"value": float(ranks["le1"])}
        scalar_metrics[f"{name}_rank_le4"] = {"value": float(ranks["le4"])}
        scalar_metrics[f"{name}_exact_changed"] = {"value": float(agg["exact_changed_groups"])}
        scalar_metrics[f"{name}_wrong_change"] = {"value": float(agg["wrong_change"])}
        scalar_metrics[f"{name}_tesseract_delta"] = {"value": float(agg["tesseract_delta_vs_nearest"])}
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_source_robust_feature_loss_ablation_not_promoted_selector",
        "inputs": {"table": {"path": str(args.table), "sha256": sha256_file(args.table)}},
        "config": {
            "feature_families": args.feature_families,
            "loss_modes": args.loss_modes,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pairwise_weight": args.pairwise_weight,
            "model_seeds": args.model_seeds,
            "seed": args.seed,
            "device": str(device),
            "split_policy": "leave one source out; train on all other source groups for diagnostic comparability with N096",
        },
        "variants": variants,
        "best_by_rank1": {
            "feature_family": best_rank1["feature_family"],
            "loss_mode": best_rank1["loss_mode"],
            "aggregate": best_rank1["aggregate"],
        },
        "best_by_rank4": {
            "feature_family": best_rank4["feature_family"],
            "loss_mode": best_rank4["loss_mode"],
            "aggregate": best_rank4["aggregate"],
        },
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "base_ranker_script": sha256_file(SCRIPT_DIR / "train_top8_trainval_tabular_listwise_ranker.py"),
            "lso_script": sha256_file(SCRIPT_DIR / "analyze_top8_trainval_leave_source_out_ranker.py"),
        },
        "aggregate": {"scalar_metrics": scalar_metrics},
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(args.report),
                "best_by_rank1": result["best_by_rank1"],
                "best_by_rank4": result["best_by_rank4"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
