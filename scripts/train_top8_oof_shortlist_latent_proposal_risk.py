#!/usr/bin/env python3
"""Train an OOF-shortlist latent/code proposal model plus risk abstention.

N120 showed that separating non-nearest proposal from no-op risk is the right
shape, but scalar OOF/rate/codebook/image-diff features plateaued. This
diagnostic keeps the N117 source-modulo OOF shortlist and adds candidate-local
latent/code/window evidence to the proposal stage only. A separate shallow risk
gate decides whether to accept the proposed change.

Teacher OCR fields, held-out OCR deltas, strings, predictions, and references
are not used as inference features. OCR labels/deltas are used only for
training/evaluation in this diagnostic. No `.oscr` stream is exported here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402
import train_top8_oof_shortlist_policy_selector as n118  # noqa: E402
import train_top8_oof_shortlist_two_stage_risk as n120  # noqa: E402


FEATURE_GUARD = {
    "used_npz_keys": [
        "core_features",
        "code_center",
        "nearest_center",
        "delta_center",
        "latent_yhat",
        "latent_target",
        "latent_residual",
        "latent_window_yhat",
        "latent_window_target",
        "latent_window_residual",
    ],
    "excluded_label_or_teacher_keys": [
        "labels",
        "parseq_delta",
        "tesseract_delta",
        "teacher_parseq_edit_delta",
        "teacher_parseq_norm_edit_delta",
        "teacher_parseq_confidence_delta",
        "teacher_parseq_edit_gain",
        "teacher_parseq_norm_edit_gain",
        "teacher_parseq_conf_gain",
        "teacher_parseq_combo_gain",
    ],
}


class LatentProposalNet(torch.nn.Module):
    def __init__(self, in_channels: int, code_dim: int, tab_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.window = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        self.code = torch.nn.Sequential(
            torch.nn.Linear(code_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.GELU(),
        )
        self.tab = torch.nn.Sequential(
            torch.nn.Linear(tab_dim, 64),
            torch.nn.LayerNorm(64),
            torch.nn.GELU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(64 + 64 + 64, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, window: torch.Tensor, code: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.window(window), self.code(code), self.tab(tab)], dim=1)
        return self.head(fused).reshape(-1)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def stats(values: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            values.mean(axis=1),
            values.std(axis=1),
            np.mean(np.abs(values), axis=1),
            np.sqrt(np.mean(np.square(values), axis=1)),
            np.max(np.abs(values), axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def tensor_lookup(data: dict[str, np.ndarray]) -> dict[tuple[int, int, int, int, int], tuple[int, int]]:
    lookup: dict[tuple[int, int, int, int, int], tuple[int, int]] = {}
    for group_index in range(data["group_keys"].shape[0]):
        group = tuple(int(value) for value in data["group_keys"][group_index])
        for slot in np.flatnonzero(data["mask"][group_index].astype(bool)).tolist():
            key = (*group, int(data["code_indices"][group_index, slot]))
            lookup[key] = (int(group_index), int(slot))
    return lookup


def row_tensor_positions(
    rows_all: list[dict[str, Any]],
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    lookup = tensor_lookup(data)
    group_pos = np.zeros(len(rows_all), dtype=np.int64)
    slot_pos = np.zeros(len(rows_all), dtype=np.int64)
    missing: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows_all):
        key = bottleneck.candidate_key(row)
        found = lookup.get(key)
        if found is None:
            missing.append({"row_index": row_index, "candidate_key": list(key)})
            continue
        group_pos[row_index], slot_pos[row_index] = found
    return group_pos, slot_pos, missing


def normalize_2d(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    train_values = values[train_mask]
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    out = ((values - mean) / std).astype(np.float32)
    return out, {"dim": int(values.shape[1]), "mean_abs": float(np.mean(np.abs(mean))), "std_min": float(std.min())}


def normalize_window(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    train_values = values[train_mask]
    mean = train_values.mean(axis=(0, 2, 3), keepdims=True)
    std = train_values.std(axis=(0, 2, 3), keepdims=True)
    std[std < 1.0e-6] = 1.0
    out = ((values - mean) / std).astype(np.float32)
    return out, {"channels": int(values.shape[1]), "mean_abs": float(np.mean(np.abs(mean))), "std_min": float(std.min())}


def row_aligned_features(
    data: dict[str, np.ndarray],
    group_pos: np.ndarray,
    slot_pos: np.ndarray,
    rows_all: list[dict[str, Any]],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    yhat_win = data["latent_window_yhat"][group_pos, slot_pos].astype(np.float32)
    target_win = data["latent_window_target"][group_pos, slot_pos].astype(np.float32)
    residual_win = data["latent_window_residual"][group_pos, slot_pos].astype(np.float32)
    window = np.concatenate([yhat_win, target_win, residual_win, target_win - yhat_win], axis=1)

    code_parts = [
        data["code_center"][group_pos, slot_pos].astype(np.float32),
        data["nearest_center"][group_pos, slot_pos].astype(np.float32),
        data["delta_center"][group_pos, slot_pos].astype(np.float32),
        data["latent_yhat"][group_pos, slot_pos].astype(np.float32),
        data["latent_target"][group_pos, slot_pos].astype(np.float32),
        data["latent_residual"][group_pos, slot_pos].astype(np.float32),
    ]
    code = np.concatenate(code_parts, axis=1)

    tab_names = n118.feature_names(rows_all)
    tab_table = n118.feature_matrix(rows_all, tab_names)
    core = data["core_features"][group_pos, slot_pos].astype(np.float32)
    tab = np.concatenate(
        [
            tab_table,
            core,
            *(stats(part) for part in code_parts),
        ],
        axis=1,
    ).astype(np.float32)

    window, window_norm = normalize_window(window, train_mask)
    code, code_norm = normalize_2d(code, train_mask)
    tab, tab_norm = normalize_2d(tab, train_mask)
    return window, code, tab, {
        "tabular_feature_count": int(tab_table.shape[1]),
        "core_feature_count": int(core.shape[1]),
        "latent_stat_count": int(tab.shape[1] - tab_table.shape[1] - core.shape[1]),
        "window_normalizer": window_norm,
        "code_normalizer": code_norm,
        "tab_normalizer": tab_norm,
        "table_features": tab_names,
    }


def proposal_train_groups(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, rows in groups.items():
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        indices = [idx for idx in group_to_indices.get(key, []) if not int(rows_all[idx].get("is_nearest", 0))]
        target_local = None
        for local, idx in enumerate(indices):
            if int(rows_all[idx]["code_index"]) == int(oracle["code_index"]):
                target_local = local
                break
        if target_local is None:
            continue
        out.append({"key": key, "indices": indices, "target_local": int(target_local)})
    return out


def train_seed(
    args: argparse.Namespace,
    seed: int,
    window_np: np.ndarray,
    code_np: np.ndarray,
    tab_np: np.ndarray,
    train_records: list[dict[str, Any]],
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    rows_all: list[dict[str, Any]],
    train_group_indices: dict[tuple[int, int, int, int], list[int]],
    val_group_indices: dict[tuple[int, int, int, int], list[int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda_required and (args.cpu or not torch.cuda.is_available()):
        raise SystemExit("GPU is not visible; stopping before N123 training as requested.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    window = torch.from_numpy(window_np).to(device)
    code = torch.from_numpy(code_np).to(device)
    tab = torch.from_numpy(tab_np).to(device)
    model = LatentProposalNet(window.shape[1], code.shape[1], tab.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    for _ in range(args.epochs):
        random.shuffle(train_records)
        model.train()
        opt.zero_grad()
        logits = model(window, code, tab)
        losses: list[torch.Tensor] = []
        for record in train_records:
            idx = torch.tensor(record["indices"], dtype=torch.long, device=device)
            group_logits = logits[idx].reshape(1, -1)
            target = torch.tensor([record["target_local"]], dtype=torch.long, device=device)
            losses.append(F.cross_entropy(group_logits, target))
            if args.pairwise_weight > 0:
                target_logit = group_logits[0, record["target_local"]]
                negatives = torch.cat(
                    [
                        group_logits[0, : record["target_local"]],
                        group_logits[0, record["target_local"] + 1 :],
                    ]
                )
                if negatives.numel():
                    losses.append(args.pairwise_weight * F.softplus(-(target_logit - negatives)).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.no_grad():
        scores = model(window, code, tab).detach().cpu().numpy().astype(np.float64)
    train_proposals = n120.select_best_nonnearest(groups["train"], train_group_indices, rows_all, scores)
    val_proposals = n120.select_best_nonnearest(groups["val"], val_group_indices, rows_all, scores)
    summary = {
        "seed": int(seed),
        "device": str(device),
        "final_loss": last_loss,
        "train_records": int(len(train_records)),
        "train_proposal_rank_counts": n120.proposal_rank_counts(groups["train"], train_group_indices, rows_all, scores),
        "val_proposal_rank_counts": n120.proposal_rank_counts(groups["val"], val_group_indices, rows_all, scores),
        "val_all_proposals": bottleneck.evaluate_selection(groups["val"], val_proposals),
        "train_all_proposals": bottleneck.evaluate_selection(groups["train"], train_proposals),
    }
    return scores, summary


def proposal_score_stats(
    val_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, Any]:
    oracle_scores = []
    wrong_top_scores = []
    for key, rows in val_groups.items():
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        non_indices = [idx for idx in group_to_indices.get(key, []) if not int(rows_all[idx].get("is_nearest", 0))]
        if not non_indices:
            continue
        best_idx = max(non_indices, key=lambda idx: (float(scores[idx]), -int(rows_all[idx].get("topk_rank", 999))))
        oracle_idx = None
        for idx in non_indices:
            if int(rows_all[idx]["code_index"]) == int(oracle["code_index"]):
                oracle_idx = idx
                break
        if oracle_idx is not None:
            oracle_scores.append(float(scores[oracle_idx]))
        if int(rows_all[best_idx]["code_index"]) != int(oracle["code_index"]):
            wrong_top_scores.append(float(scores[best_idx]))
    return {
        "oracle_score_mean": float(np.mean(oracle_scores)) if oracle_scores else None,
        "wrong_top_score_mean": float(np.mean(wrong_top_scores)) if wrong_top_scores else None,
        "oracle_score_count": int(len(oracle_scores)),
        "wrong_top_score_count": int(len(wrong_top_scores)),
    }


def risk_eval(
    args: argparse.Namespace,
    split_groups: dict[str, dict[tuple[int, int, int, int], list[dict[str, Any]]]],
    rows_all: list[dict[str, Any]],
    train_group_indices: dict[tuple[int, int, int, int], list[int]],
    val_group_indices: dict[tuple[int, int, int, int], list[int]],
    proposal_scores: np.ndarray,
) -> dict[str, Any]:
    train_proposals = n120.select_best_nonnearest(split_groups["train"], train_group_indices, rows_all, proposal_scores)
    val_proposals = n120.select_best_nonnearest(split_groups["val"], val_group_indices, rows_all, proposal_scores)
    train_risk_rows, train_risk_keys = n120.proposal_group_features(
        split_groups["train"], train_group_indices, rows_all, proposal_scores, train_proposals
    )
    val_risk_rows, val_risk_keys = n120.proposal_group_features(
        split_groups["val"], val_group_indices, rows_all, proposal_scores, val_proposals
    )
    risk_names = n120.risk_feature_names(train_risk_rows + val_risk_rows)
    x_train_risk = n120.risk_matrix(train_risk_rows, risk_names)
    y_train_risk = np.asarray([int(row["accept_label"]) for row in train_risk_rows], dtype=np.int32)
    x_val_risk = n120.risk_matrix(val_risk_rows, risk_names)
    risk_models: dict[str, Any] = {}
    for risk_name, risk_model in n120.risk_grid(args.random_state).items():
        if int(y_train_risk.sum()) == 0 or int(y_train_risk.sum()) == len(y_train_risk):
            risk_scores_train = {key: 0.0 for key in train_risk_keys}
            risk_scores_val = {key: 0.0 for key in val_risk_keys}
        else:
            risk_model.fit(x_train_risk, y_train_risk)
            risk_scores_train = n120.score_group_model(risk_model, x_train_risk, train_risk_keys)
            risk_scores_val = n120.score_group_model(risk_model, x_val_risk, val_risk_keys)
        tuned: dict[str, Any] = {}
        for budget in [0, 1, 2]:
            train_tuned = n120.tune_risk_threshold(
                split_groups["train"],
                train_proposals,
                risk_scores_train,
                false_wrong_budget=budget,
            )
            threshold = float(train_tuned["threshold"])
            val_metrics = bottleneck.evaluate_selection(
                split_groups["val"],
                n120.select_with_risk(split_groups["val"], val_proposals, risk_scores_val, threshold),
            )
            tuned[f"val_budget{budget}"] = {
                "threshold": threshold,
                "train_metrics": train_tuned["metrics"],
                "metrics": val_metrics,
            }
        risk_models[risk_name] = tuned
    return {
        "train_risk_groups": int(len(train_risk_rows)),
        "train_risk_positive_groups": int(y_train_risk.sum()),
        "risk_feature_count": int(len(risk_names)),
        "risk_features": risk_names,
        "val_all_proposals": bottleneck.evaluate_selection(split_groups["val"], val_proposals),
        "train_all_proposals": bottleneck.evaluate_selection(split_groups["train"], train_proposals),
        "val_proposal_rank_counts": n120.proposal_rank_counts(
            split_groups["val"], val_group_indices, rows_all, proposal_scores
        ),
        "train_proposal_rank_counts": n120.proposal_rank_counts(
            split_groups["train"], train_group_indices, rows_all, proposal_scores
        ),
        "val_score_stats": proposal_score_stats(split_groups["val"], val_group_indices, rows_all, proposal_scores),
        "risk_models": risk_models,
    }


def compact(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics.get('exact_changed_groups', 0)}/{metrics.get('oracle_change_groups', 0)} exact, "
        f"false {metrics.get('false_change', 0)}, wrong {metrics.get('wrong_change', 0)}, "
        f"miss {metrics.get('missed_oracle', 0)}, "
        f"T {metrics.get('tesseract_delta_vs_nearest', 0)}, P {metrics.get('parseq_delta_vs_nearest', 0)}"
    )


def flatten_numeric(prefix: str, value: object, out: dict[str, float]) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            safe_key = str(key).replace("/", "_")
            flatten_numeric(f"{prefix}/{safe_key}", child, out)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if index >= 16:
                break
            flatten_numeric(f"{prefix}/{index}", child, out)


def start_wandb(args: argparse.Namespace, result: dict[str, Any]) -> Any | None:
    if args.no_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.experiment_id,
        mode=args.wandb_mode,
        tags=["oscar-lic", "N123", "latent-proposal-risk"],
        notes=(
            "N123 OOF-shortlist latent/code/window proposal + separate risk abstention. "
            "Diagnostic only; no .oscr export."
        ),
        config={
            "experiment_id": args.experiment_id,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pairwise_weight": args.pairwise_weight,
            "model_seeds": args.model_seeds,
            "feature_guard": FEATURE_GUARD,
            "inputs": result["inputs"],
        },
    )


def finish_wandb(run: Any | None, result: dict[str, Any], output: Path, report: Path) -> None:
    if run is None:
        return
    import wandb

    metrics: dict[str, float] = {}
    flatten_numeric("n123", result, metrics)
    if metrics:
        wandb.log(metrics)
    artifact = wandb.Artifact(result["experiment_id"].replace("/", "_"), type="result-json")
    artifact.add_file(str(output), name=output.name)
    artifact.add_file(str(report), name=report.name)
    run.log_artifact(artifact)
    run.finish()


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "N117 OOF shortlist with a latent/code/window non-nearest proposal model and separate N120-style risk abstention. Diagnostic only; no `.oscr` stream is exported.",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "## Proposal Seeds",
        "",
        "| seed | loss | rank1 | rank<=4 | all proposals |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in result["proposal_seed_runs"]:
        ranks = row["val_proposal_rank_counts"]
        lines.append(
            f"| {row['seed']} | {row['final_loss']:.4f} | {ranks.get('rank_le1', 0)} | "
            f"{ranks.get('rank_le4', 0)} | {compact(row['val_all_proposals'])} |"
        )
    ens = result["ensemble"]
    lines.extend(
        [
            "",
            "## Ensemble Risk",
            "",
            "| risk | budget0 | budget1 | budget2 | proposal rank1/rank<=4 |",
            "|---|---|---|---|---:|",
        ]
    )
    ranks = ens["val_proposal_rank_counts"]
    for risk_name, risk in ens["risk_models"].items():
        lines.append(
            f"| {risk_name} | {compact(risk['val_budget0']['metrics'])} | "
            f"{compact(risk['val_budget1']['metrics'])} | {compact(risk['val_budget2']['metrics'])} | "
            f"{ranks.get('rank_le1', 0)}/{ranks.get('rank_le4', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Inference features use only deployable table scalars plus latent/code/window tensors listed in `feature_guard.used_npz_keys`.",
            "- `teacher_parseq_*`, OCR deltas, OCR strings, references, predictions, and oracle labels are excluded from inference features.",
            "- Risk thresholds are tuned on train groups and reported on validation groups.",
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-oof-latent-proposal-risk",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
    ]
    if result.get("wandb_run_id"):
        lines.append(f"wandb_run_id: {result['wandb_run_id']}")
    lines.extend(
        [
            "inputs:",
            "  table:",
            f"    path: {result['inputs']['table']['path']}",
            f"    sha256: {result['inputs']['table']['sha256']}",
            "  features:",
            f"    path: {result['inputs']['features']['path']}",
            f"    sha256: {result['inputs']['features']['sha256']}",
        ]
    )
    for item in result["inputs"]["score_files"]:
        lines.extend([f"  score_r{item['remainder']}:", f"    path: {item['path']}", f"    sha256: {item['sha256']}"])
    lines.extend(
        [
            "outputs:",
            "  result_json:",
            f"    path: {output}",
            f"    sha256: {bottleneck.sha256_file(output)}",
            "  report:",
            f"    path: {report}",
            f"    sha256: {bottleneck.sha256_file(report)}",
            "scripts:",
            "  trainer:",
            "    path: scripts/train_top8_oof_shortlist_latent_proposal_risk.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  risk_source:",
            "    path: scripts/train_top8_oof_shortlist_two_stage_risk.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_two_stage_risk.py')}",
            "  selector_source:",
            "    path: scripts/train_top8_oof_shortlist_policy_selector.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_policy_selector.py')}",
            "conclusion: diagnostic latent proposal plus risk; no counted oscr export",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, action="append", required=True)
    parser.add_argument("--source-modulo", type=int, default=5)
    parser.add_argument("--topk-cap", type=int, default=4)
    parser.add_argument("--score-count", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=7.0e-4)
    parser.add_argument("--weight-decay", type=float, default=2.0e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--model-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--random-state", type=int, default=20260626)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda-required", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oscar-lic")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_npz(args.features)
    groups_all, by_candidate = bottleneck.load_table(args.table)
    split_groups = bottleneck.split_groups(groups_all)
    oof_scores, oof_meta = oof_shortlist.build_oof_scores(args.score_file, by_candidate, modulo=args.source_modulo)
    rows_all, row_index = n118.build_dataset(groups_all, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count)
    train_group_indices = n118.group_indices(
        split_groups["train"], rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    val_group_indices = n118.group_indices(
        split_groups["val"], rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    group_pos, slot_pos, missing = row_tensor_positions(rows_all, data)
    if missing:
        raise SystemExit(f"feature tensor missing {len(missing)} shortlist rows; first={missing[:3]}")
    train_non_indices = n120.group_train_indices(split_groups["train"], train_group_indices, rows_all)
    train_mask = np.zeros(len(rows_all), dtype=bool)
    train_mask[train_non_indices] = True
    window_np, code_np, tab_np, feature_meta = row_aligned_features(data, group_pos, slot_pos, rows_all, train_mask)
    train_records = proposal_train_groups(split_groups["train"], train_group_indices, rows_all)
    if not train_records:
        raise SystemExit("no train oracle-change records are covered by the N117 shortlist")

    split_package = {"train": split_groups["train"], "val": split_groups["val"]}
    seed_scores: list[np.ndarray] = []
    seed_summaries: list[dict[str, Any]] = []
    for offset in range(args.model_seeds):
        scores, summary = train_seed(
            args,
            args.seed + offset,
            window_np,
            code_np,
            tab_np,
            train_records,
            split_package,
            rows_all,
            train_group_indices,
            val_group_indices,
        )
        seed_scores.append(scores)
        seed_summaries.append(summary)
    ensemble_scores = np.mean(np.stack(seed_scores, axis=0), axis=0)
    ensemble = risk_eval(args, split_package, rows_all, train_group_indices, val_group_indices, ensemble_scores)

    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_latent_proposal_risk_not_promoted",
        "code_commit": bottleneck.git_commit(),
        "inputs": {
            "table": {"path": str(args.table), "sha256": bottleneck.sha256_file(args.table)},
            "features": {"path": str(args.features), "sha256": bottleneck.sha256_file(args.features)},
            "score_files": oof_meta["score_files"],
        },
        "config": {
            "source_modulo": args.source_modulo,
            "topk_cap": args.topk_cap,
            "score_count": args.score_count,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pairwise_weight": args.pairwise_weight,
            "grad_clip": args.grad_clip,
            "model_seeds": args.model_seeds,
            "seed": args.seed,
            "random_state": args.random_state,
            "cuda_required": bool(args.cuda_required),
        },
        "feature_guard": FEATURE_GUARD,
        "feature_meta": feature_meta,
        "data_summary": {
            "groups": int(len(groups_all)),
            "train_groups": int(len(split_groups["train"])),
            "val_groups": int(len(split_groups["val"])),
            "selector_rows": int(len(rows_all)),
            "train_nonnearest_rows": int(len(train_non_indices)),
            "train_covered_oracle_change_records": int(len(train_records)),
            "window_shape": [int(value) for value in window_np.shape],
            "code_shape": [int(value) for value in code_np.shape],
            "tab_shape": [int(value) for value in tab_np.shape],
            "oof": {key: value for key, value in oof_meta.items() if key != "score_files"},
        },
        "proposal_seed_runs": seed_summaries,
        "ensemble": ensemble,
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "risk_source": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_two_stage_risk.py"),
            "selector_source": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_policy_selector.py"),
            "shortlist_source": bottleneck.sha256_file(ROOT / "scripts/analyze_top8_source_mod_oof_shortlist.py"),
        },
        "interpretation": (
            "This is the first N117/N120-style proposal-risk diagnostic whose proposal stage sees "
            "candidate-local latent/code/window evidence. It is a valid next step after N122 because it does "
            "not sweep scalar sample weights or risk thresholds over the same feature family. It remains "
            "diagnostic-only unless validation table metrics beat the current actual-bitstream floor before "
            "any counted `.oscr` export."
        ),
    }

    run = start_wandb(args, result)
    if run is not None:
        result["wandb_run_id"] = run.id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    finish_wandb(run, result, args.output, args.report)
    write_manifest(args.manifest, result, args.output, args.report)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "wandb_run_id": result.get("wandb_run_id"),
                "ensemble_rank_counts": ensemble["val_proposal_rank_counts"],
                "ensemble_all_proposals": compact(ensemble["val_all_proposals"]),
                "best_budget0": {
                    name: compact(row["val_budget0"]["metrics"])
                    for name, row in ensemble["risk_models"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
