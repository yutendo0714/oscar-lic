#!/usr/bin/env python3
"""Group-OOF calibrated top-k-prior verifier over latent/code tensors.

Candidate choice is fixed to the simple top-k/rate prior. Unlike the earlier
latent verifier, thresholds are tuned on train-group out-of-fold scores rather
than in-sample train scores. This is a table-level diagnostic only.
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

import train_top8_topk_prior_latent_verifier as latent_verifier  # noqa: E402


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_values(
    data: dict[str, np.ndarray],
    table_rows: list[dict[str, Any]],
    group_key: str,
) -> np.ndarray:
    if group_key == "source_index":
        return np.asarray([int(values[2]) for values in data["group_keys"]], dtype=object)
    by_key = {}
    for row in table_rows:
        key = (int(row["real_seed"]), int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))
        by_key[key] = str(row.get(group_key, "unknown"))
    values = []
    for key_values in data["group_keys"]:
        real_seed, seed, source_index, candidate_index = [int(value) for value in key_values]
        values.append(by_key.get((real_seed, seed, source_index, candidate_index), "unknown"))
    return np.asarray(values, dtype=object)


def make_sample_weights(labels: np.ndarray, arrays: dict[str, np.ndarray], mode: str) -> np.ndarray:
    weights = np.ones(labels.shape[0], dtype=np.float32)
    if mode == "standard":
        return weights
    negative = labels < 0.5
    non_improve = arrays["tesseract_delta"] >= 0
    harm = (arrays["tesseract_delta"] > 0) | (arrays["parseq_delta"] > 0)
    if mode == "hardneg4":
        weights[negative & non_improve] = 4.0
        weights[negative & harm] = 8.0
    elif mode == "hardneg8":
        weights[negative & non_improve] = 8.0
        weights[negative & harm] = 16.0
    elif mode == "hardneg16":
        weights[negative & non_improve] = 16.0
        weights[negative & harm] = 32.0
    else:
        raise ValueError(f"unknown loss mode: {mode}")
    return weights


def train_model(
    arrays_raw: dict[str, np.ndarray],
    train_mask: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    model_seed: int,
    loss_mode: str,
    pairwise_weight: float,
) -> tuple[latent_verifier.TopPriorVerifier, dict[str, np.ndarray], str, float]:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    arrays = latent_verifier.normalize(arrays_raw, train_mask)
    x_core = torch.from_numpy(arrays["core"]).to(device)
    x_code = torch.from_numpy(arrays["code"]).to(device)
    x_window = torch.from_numpy(arrays["window"]).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    train_idx_np = np.flatnonzero(train_mask)
    train_idx = torch.from_numpy(train_idx_np).long().to(device)
    model = latent_verifier.TopPriorVerifier(
        core_dim=arrays["core"].shape[1],
        code_dim=arrays["code"].shape[1],
        window_channels=arrays["window"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    positives = float(labels[train_mask].sum())
    negatives = float(train_mask.sum() - positives)
    pos_weight = negatives / max(positives, 1.0)
    sample_weights = make_sample_weights(labels, arrays_raw, loss_mode)
    weights = torch.from_numpy(sample_weights.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_core[train_idx], x_code[train_idx], x_window[train_idx])
        y_train = y[train_idx]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device),
            reduction="none",
        )
        loss = (bce * weights[train_idx]).mean()
        if pairwise_weight > 0:
            pos_logits = logits[y_train > 0.5]
            hard_neg_mask = (y_train < 0.5) & (weights[train_idx] > 1.0)
            neg_logits = logits[hard_neg_mask]
            if pos_logits.numel() and neg_logits.numel():
                pair = torch.nn.functional.softplus(args.margin - pos_logits[:, None] + neg_logits[None, :]).mean()
                loss = loss + pairwise_weight * pair
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())
    return model, arrays, str(device), final_loss


def score_model(
    model: latent_verifier.TopPriorVerifier,
    arrays: dict[str, np.ndarray],
    device_name: str,
) -> np.ndarray:
    device = torch.device(device_name)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(
            model(
                torch.from_numpy(arrays["core"]).to(device),
                torch.from_numpy(arrays["code"]).to(device),
                torch.from_numpy(arrays["window"]).to(device),
            )
        )
    return scores.detach().cpu().numpy().astype(np.float32)


def threshold_candidates(scores: np.ndarray, mask: np.ndarray) -> list[float]:
    values = sorted(set(float(score) for score in scores[mask]))
    if not values:
        return [1.0]
    return [max(values) + 1e-6] + values + [min(values) - 1e-6]


def choose_budget_threshold(
    scores: np.ndarray,
    records: list[dict[str, Any]],
    train_mask: np.ndarray,
    arrays: dict[str, np.ndarray],
    false_wrong_budget: int,
    allow_parseq_worse: int,
    allow_tess_worse: int,
) -> dict[str, Any]:
    rows = []
    for threshold in threshold_candidates(scores, train_mask):
        metrics = latent_verifier.policy_metrics(records, scores >= threshold, train_mask, arrays)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    candidates = [
        row
        for row in rows
        if row["false_change"] + row["wrong_change"] <= false_wrong_budget
        and row["parseq_worsen_groups"] <= allow_parseq_worse
        and row["tesseract_worsen_groups"] <= allow_tess_worse
    ]
    if not candidates:
        candidates = [row for row in rows if row["parseq_worsen_groups"] <= allow_parseq_worse]
    if not candidates:
        candidates = rows
    return min(
        candidates,
        key=lambda row: (
            row["tesseract_delta_vs_nearest"],
            row["tesseract_worsen_groups"],
            row["false_change"] + row["wrong_change"],
            -row["exact_changed_groups"],
            row["changed_groups"],
        ),
    )


def run_one(
    arrays_raw: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    target_name: str,
    model_seed: int,
    loss_mode: str,
    pairwise_weight: float,
) -> dict[str, Any]:
    partition = arrays_raw["partition"]
    train_mask = partition == "train"
    val_mask = partition == "val"
    labels = arrays_raw["labels_exact"] if target_name == "exact_changed" else arrays_raw["labels_safe"]
    group = group_values(args._feature_data, args._table_rows, args.group_key)
    train_groups = sorted(set(value for value in group[train_mask]))
    oof_scores = np.full(labels.shape[0], np.nan, dtype=np.float32)
    val_scores = []
    fold_summaries = []
    for fold_id, group_value in enumerate(train_groups):
        fold_train = train_mask & (group != group_value)
        fold_holdout = train_mask & (group == group_value)
        if labels[fold_train].sum() <= 0 or fold_holdout.sum() <= 0:
            continue
        model, arrays_fold, device_name, final_loss = train_model(
            arrays_raw, fold_train, labels, args, model_seed + fold_id * 1009, loss_mode, pairwise_weight
        )
        scores = score_model(model, arrays_fold, device_name)
        oof_scores[fold_holdout] = scores[fold_holdout]
        val_scores.append(scores[val_mask])
        fold_summaries.append(
            {
                "heldout_group_key": args.group_key,
                "heldout_group_value": str(group_value),
                "fold_train_groups": int(fold_train.sum()),
                "fold_holdout_groups": int(fold_holdout.sum()),
                "fold_train_positives": int(labels[fold_train].sum()),
                "fold_holdout_positives": int(labels[fold_holdout].sum()),
                "final_loss": final_loss,
            }
        )
    oof_mask = train_mask & np.isfinite(oof_scores)
    if not val_scores:
        raise RuntimeError("no OOF folds were trained")
    val_ensemble = np.mean(np.stack(val_scores, axis=0), axis=0).astype(np.float32)
    ensemble_scores = np.full(labels.shape[0], np.nan, dtype=np.float32)
    ensemble_scores[oof_mask] = oof_scores[oof_mask]
    ensemble_scores[val_mask] = val_ensemble
    threshold = choose_budget_threshold(
        ensemble_scores,
        records,
        oof_mask,
        arrays_raw,
        false_wrong_budget=args.false_wrong_budget,
        allow_parseq_worse=args.allow_parseq_worse,
        allow_tess_worse=args.allow_tess_worse,
    )
    accept_ensemble = ensemble_scores >= float(threshold["threshold"])
    oof_metrics = latent_verifier.policy_metrics(records, accept_ensemble, oof_mask, arrays_raw)
    val_ensemble_metrics = latent_verifier.policy_metrics(records, accept_ensemble, val_mask, arrays_raw)

    full_model, arrays_full, full_device, full_loss = train_model(
        arrays_raw, train_mask, labels, args, model_seed + 77777, loss_mode, pairwise_weight
    )
    full_scores = score_model(full_model, arrays_full, full_device)
    full_accept = full_scores >= float(threshold["threshold"])
    full_train_metrics = latent_verifier.policy_metrics(records, full_accept, train_mask, arrays_raw)
    full_val_metrics = latent_verifier.policy_metrics(records, full_accept, val_mask, arrays_raw)
    return {
        "target": target_name,
        "model_seed": int(model_seed),
        "loss_mode": loss_mode,
        "pairwise_weight": float(pairwise_weight),
        "threshold_from_oof": float(threshold["threshold"]),
        "oof_positive_groups": int(labels[oof_mask].sum()),
        "val_positive_groups": int(labels[val_mask].sum()),
        "folds": fold_summaries,
        "oof_train": oof_metrics,
        "val_oof_ensemble": val_ensemble_metrics,
        "full_model_final_loss": full_loss,
        "full_train_with_oof_threshold": full_train_metrics,
        "val_full_model_with_oof_threshold": full_val_metrics,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Group-OOF calibrated top-k-prior verifier over local latent/code tensors.",
        "This is a table-level diagnostic and does not export counted `.oscr` streams.",
        "",
        "## Results",
        "",
        "| target | loss | pair | seed | oof changed | oof exact | oof false+wrong | oof T | val ens changed | val ens exact | val ens false+wrong | val ens T | val full changed | val full exact | val full false+wrong | val full T | val full P |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        oof = row["oof_train"]
        ens = row["val_oof_ensemble"]
        full = row["val_full_model_with_oof_threshold"]
        lines.append(
            f"| `{row['target']}` | `{row['loss_mode']}` | {row['pairwise_weight']:.2f} | {row['model_seed']} | "
            f"{oof['changed_groups']} | {oof['exact_changed_groups']} | {oof['false_change'] + oof['wrong_change']} | {oof['tesseract_delta_vs_nearest']} | "
            f"{ens['changed_groups']} | {ens['exact_changed_groups']} | {ens['false_change'] + ens['wrong_change']} | {ens['tesseract_delta_vs_nearest']} | "
            f"{full['changed_groups']} | {full['exact_changed_groups']} | {full['false_change'] + full['wrong_change']} | {full['tesseract_delta_vs_nearest']} | {full['parseq_delta_vs_nearest']} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_topk_prior_oof_calibrated_verifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--false-wrong-budget", type=int, default=1)
    parser.add_argument("--allow-parseq-worse", type=int, default=0)
    parser.add_argument("--allow-tess-worse", type=int, default=1)
    parser.add_argument("--group-key", default="source", help="Table column for train-only OOF calibration groups.")
    parser.add_argument("--model-seed", type=int, action="append", default=[])
    parser.add_argument("--loss-mode", action="append", default=[])
    parser.add_argument("--pairwise-weight", type=float, action="append", default=[])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    seeds = args.model_seed or [0]
    loss_modes = args.loss_mode or ["standard", "hardneg4", "hardneg8"]
    pairwise_weights = args.pairwise_weight or [0.0, 0.25]
    data = latent_verifier.load_npz(args.features)
    table_rows = latent_verifier.read_jsonl(args.table)
    args._feature_data = data
    args._table_rows = table_rows
    arrays, records = latent_verifier.build_group_arrays(data, table_rows)
    runs = []
    for target in ["exact_changed", "safe_improve"]:
        for loss_mode in loss_modes:
            for pairwise_weight in pairwise_weights:
                for seed in seeds:
                    runs.append(run_one(arrays, records, args, target, int(seed), loss_mode, float(pairwise_weight)))
    best = min(
        runs,
        key=lambda row: (
            row["val_oof_ensemble"]["tesseract_delta_vs_nearest"],
            row["val_oof_ensemble"]["false_change"] + row["val_oof_ensemble"]["wrong_change"],
            -row["val_oof_ensemble"]["exact_changed_groups"],
        ),
    )
    best_full = min(
        runs,
        key=lambda row: (
            row["val_full_model_with_oof_threshold"]["tesseract_delta_vs_nearest"],
            row["val_full_model_with_oof_threshold"]["false_change"] + row["val_full_model_with_oof_threshold"]["wrong_change"],
            -row["val_full_model_with_oof_threshold"]["exact_changed_groups"],
        ),
    )
    interpretation = (
        "This tests whether train-group out-of-fold calibration and explicit hard-negative weighting can make the "
        "top-k/rate prior plus latent/code evidence safer than the in-sample N099 verifier. Promotion would still "
        "require beating the current counted actual-bitstream -8 Tesseract floor with low false/wrong changes."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_topk_prior_oof_calibrated_verifier_not_promoted_selector",
        "inputs": {
            "features": {"path": str(args.features), "sha256": sha256_file(args.features)},
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
        },
        "config": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "margin": args.margin,
            "false_wrong_budget": args.false_wrong_budget,
            "allow_parseq_worse": args.allow_parseq_worse,
            "allow_tess_worse": args.allow_tess_worse,
            "group_key": args.group_key,
            "model_seeds": seeds,
            "loss_modes": loss_modes,
            "pairwise_weights": pairwise_weights,
        },
        "runs": runs,
        "best_oof_ensemble_run": best,
        "best_full_model_run": best_full,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "latent_verifier_script": sha256_file(SCRIPT_DIR / "train_top8_topk_prior_latent_verifier.py"),
            "base_ranker_script": sha256_file(SCRIPT_DIR / "train_top8_trainval_tabular_listwise_ranker.py"),
        },
        "aggregate": {
            "scalar_metrics": {
                "best_ensemble_val_tesseract_delta": {"value": float(best["val_oof_ensemble"]["tesseract_delta_vs_nearest"])},
                "best_ensemble_val_parseq_delta": {"value": float(best["val_oof_ensemble"]["parseq_delta_vs_nearest"])},
                "best_ensemble_val_exact_changed": {"value": float(best["val_oof_ensemble"]["exact_changed_groups"])},
                "best_ensemble_val_false_wrong": {
                    "value": float(best["val_oof_ensemble"]["false_change"] + best["val_oof_ensemble"]["wrong_change"])
                },
                "best_full_val_tesseract_delta": {"value": float(best_full["val_full_model_with_oof_threshold"]["tesseract_delta_vs_nearest"])},
                "best_full_val_exact_changed": {"value": float(best_full["val_full_model_with_oof_threshold"]["exact_changed_groups"])},
                "best_full_val_false_wrong": {
                    "value": float(
                        best_full["val_full_model_with_oof_threshold"]["false_change"]
                        + best_full["val_full_model_with_oof_threshold"]["wrong_change"]
                    )
                },
            }
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "best_oof_ensemble_run": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
