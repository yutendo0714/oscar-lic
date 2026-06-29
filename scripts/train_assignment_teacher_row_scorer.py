#!/usr/bin/env python3
"""Train a deployable row scorer against PARSeq teacher-derived targets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    hits = 0
    total = 0.0
    for rank, label in enumerate(sorted_labels, start=1):
        if label:
            hits += 1
            total += hits / rank
    return total / positives


def target_value(row: dict[str, Any], target: str, conf_weight: float) -> float:
    if int(row.get("is_nearest", 0)):
        return 0.0
    edit_gain = max(0.0, -float(row.get("img_teacher_parseq_edit_delta_vs_nearest", 0.0)))
    norm_edit_gain = max(0.0, -float(row.get("img_teacher_parseq_norm_edit_delta_vs_nearest", 0.0)))
    conf_gain = max(0.0, float(row.get("img_teacher_parseq_confidence_delta_vs_nearest", 0.0)))
    if target == "edit_gain":
        return edit_gain
    if target == "norm_edit_gain":
        return norm_edit_gain
    if target == "conf_gain":
        return conf_gain
    if target == "combo":
        return max(edit_gain, norm_edit_gain) + conf_weight * conf_gain
    raise ValueError(f"unknown target: {target}")


EXCLUDE_EXACT = {
    "seed",
    "real_seed",
    "original_seed",
    "source_index",
    "candidate_index",
    "parseq_distance",
    "tesseract_distance",
    "parseq_nearest_distance",
    "tesseract_nearest_distance",
    "parseq_delta_vs_nearest",
    "tesseract_delta_vs_nearest",
    "multi_teacher_delta_vs_nearest",
    "assignment_oracle_code_index",
    "assignment_oracle_topk_rank",
    "assignment_oracle_relative_error",
    "assignment_oracle_tesseract_delta",
    "assignment_oracle_parseq_delta",
}


def is_feature(name: str, value: Any, include_onehot: bool) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if name in EXCLUDE_EXACT:
        return False
    if name.startswith(("label_", "group_", "parseq_", "tesseract_", "img_teacher_")):
        return False
    if name.startswith("assignment_oracle_"):
        return False
    if "_onehot_" in name and not include_onehot:
        return False
    if name.startswith(("codebook_", "img_")):
        return True
    return name in {"topk_rank", "is_nearest", "code_equals_nearest", "assignment_relative_error", "code_index"}


def feature_names(rows: list[dict[str, Any]], include_onehot: bool) -> list[str]:
    names: set[str] = set()
    for row in rows[: min(len(rows), 64)]:
        for name, value in row.items():
            if is_feature(name, value, include_onehot):
                names.add(name)
    return sorted(names)


def grouped_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return {key: sorted(group, key=lambda item: int(item["topk_rank"])) for key, group in groups.items()}


def build_arrays(
    rows: list[dict[str, Any]],
    names: list[str],
    target: str,
    conf_weight: float,
) -> dict[str, Any]:
    groups = grouped_rows(rows)
    keys = sorted(groups)
    max_candidates = max(len(groups[key]) for key in keys)
    features = np.zeros((len(keys), max_candidates, len(names)), dtype=np.float32)
    targets = np.zeros((len(keys), max_candidates), dtype=np.float32)
    teacher_labels = np.zeros((len(keys), max_candidates), dtype=np.float32)
    mask = np.zeros((len(keys), max_candidates), dtype=bool)
    is_nearest = np.zeros((len(keys), max_candidates), dtype=bool)
    oracle_labels = np.zeros((len(keys), max_candidates), dtype=bool)
    parseq_delta = np.zeros((len(keys), max_candidates), dtype=np.int32)
    tesseract_delta = np.zeros((len(keys), max_candidates), dtype=np.int32)
    for group_index, key in enumerate(keys):
        for row_index, row in enumerate(groups[key]):
            mask[group_index, row_index] = True
            features[group_index, row_index] = np.asarray([float(row.get(name, 0.0)) for name in names], dtype=np.float32)
            value = target_value(row, target, conf_weight)
            targets[group_index, row_index] = value
            teacher_labels[group_index, row_index] = float(value > 0.0)
            is_nearest[group_index, row_index] = bool(int(row["is_nearest"]))
            oracle_labels[group_index, row_index] = bool(int(row.get("label_assignment_oracle_choice", 0)))
            parseq_delta[group_index, row_index] = int(row["parseq_delta_vs_nearest"])
            tesseract_delta[group_index, row_index] = int(row["tesseract_delta_vs_nearest"])
    return {
        "keys": np.asarray(keys, dtype=np.int64),
        "features": features,
        "targets": targets,
        "teacher_labels": teacher_labels,
        "mask": mask,
        "is_nearest": is_nearest,
        "oracle_labels": oracle_labels,
        "parseq_delta": parseq_delta,
        "tesseract_delta": tesseract_delta,
    }


def split_groups(keys: np.ndarray, split_key_index: int, val_seed: int) -> tuple[np.ndarray, np.ndarray]:
    val = keys[:, split_key_index] == val_seed
    return np.flatnonzero(~val), np.flatnonzero(val)


class RowScorer(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def standardize(features: np.ndarray, train_groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = features[train_groups].reshape(-1, features.shape[-1])
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1.0e-6] = 1.0
    return (features - mean) / std, mean, std


def train_one(
    arrays: dict[str, Any],
    train_groups: np.ndarray,
    model_seed: int,
    epochs: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    loss_name: str,
    device: torch.device,
) -> np.ndarray:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    features = torch.from_numpy(arrays["features"][train_groups]).float().to(device)
    mask = torch.from_numpy(arrays["mask"][train_groups].reshape(-1)).bool().to(device)
    targets = torch.from_numpy(arrays["targets"][train_groups].reshape(-1)).float().to(device)
    labels = torch.from_numpy(arrays["teacher_labels"][train_groups].reshape(-1)).float().to(device)
    x = features.reshape(-1, features.shape[-1])
    model = RowScorer(x.shape[-1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = float(labels[mask].sum().item())
    neg = float(mask.sum().item() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        if loss_name == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[mask], labels[mask], pos_weight=pos_weight)
        elif loss_name == "mse":
            loss = torch.nn.functional.mse_loss(torch.sigmoid(pred[mask]), targets[mask])
        else:
            raise ValueError(f"unknown loss: {loss_name}")
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.eval()
        all_x = torch.from_numpy(arrays["features"].reshape(-1, arrays["features"].shape[-1])).float().to(device)
        scores = torch.sigmoid(model(all_x)).cpu().numpy().reshape(arrays["features"].shape[:2])
    scores[~arrays["mask"]] = -np.inf
    return scores


def select_policy(scores: np.ndarray, arrays: dict[str, Any], groups: np.ndarray, margin: float) -> np.ndarray:
    selected = []
    mask = arrays["mask"]
    nearest = arrays["is_nearest"]
    for group_index in groups:
        valid = np.flatnonzero(mask[group_index])
        nearest_pos = int(np.flatnonzero(nearest[group_index])[0])
        non_nearest = [pos for pos in valid if pos != nearest_pos]
        best_non = max(non_nearest, key=lambda pos: float(scores[group_index, pos]), default=nearest_pos)
        if best_non != nearest_pos and float(scores[group_index, best_non] - scores[group_index, nearest_pos]) > margin:
            selected.append(best_non)
        else:
            selected.append(nearest_pos)
    return np.asarray(selected, dtype=np.int64)


def oracle_policy(arrays: dict[str, Any], groups: np.ndarray) -> np.ndarray:
    selected = []
    for group_index in groups:
        oracle = np.flatnonzero(arrays["oracle_labels"][group_index])
        nearest = int(np.flatnonzero(arrays["is_nearest"][group_index])[0])
        selected.append(int(oracle[0]) if len(oracle) else nearest)
    return np.asarray(selected, dtype=np.int64)


def always_best_policy(scores: np.ndarray, arrays: dict[str, Any], groups: np.ndarray) -> np.ndarray:
    selected = []
    for group_index in groups:
        row = scores[group_index].copy()
        row[~arrays["mask"][group_index]] = -np.inf
        selected.append(int(np.argmax(row)))
    return np.asarray(selected, dtype=np.int64)


def policy_error_type(selected_changed: bool, selected_is_oracle: bool, oracle_changed: bool) -> str:
    if selected_is_oracle and oracle_changed:
        return "exact_oracle"
    if selected_is_oracle and not oracle_changed:
        return "correct_nearest"
    if selected_changed and not oracle_changed:
        return "false_change"
    if (not selected_changed) and oracle_changed:
        return "missed_oracle"
    return "wrong_change"


def policy_metrics(arrays: dict[str, Any], groups: np.ndarray, selected: np.ndarray) -> dict[str, int]:
    nearest_pos = arrays["is_nearest"][groups].argmax(axis=1)
    oracle_pos = arrays["oracle_labels"][groups].argmax(axis=1)
    oracle_changed = oracle_pos != nearest_pos
    selected_changed = selected != nearest_pos
    selected_is_oracle = selected == oracle_pos
    selected_parseq = arrays["parseq_delta"][groups, selected]
    selected_tess = arrays["tesseract_delta"][groups, selected]
    errors = Counter(
        policy_error_type(bool(ch), bool(ok), bool(oc))
        for ch, ok, oc in zip(selected_changed, selected_is_oracle, oracle_changed)
    )
    return {
        "groups": int(len(groups)),
        "changed_groups": int(selected_changed.sum()),
        "parseq_delta_sum": int(selected_parseq.sum()),
        "tesseract_delta_sum": int(selected_tess.sum()),
        "parseq_worsen_groups": int((selected_parseq > 0).sum()),
        "tesseract_improve_groups": int((selected_tess < 0).sum()),
        "tesseract_worsen_groups": int((selected_tess > 0).sum()),
        "oracle_change_groups": int(oracle_changed.sum()),
        "exact_oracle_groups": int(errors["exact_oracle"]),
        "missed_oracle_groups": int(errors["missed_oracle"]),
        "false_change_groups": int(errors["false_change"]),
        "wrong_change_groups": int(errors["wrong_change"]),
        "correct_nearest_groups": int(errors["correct_nearest"]),
    }


def summarize(values: list[float | int | None]) -> dict[str, float | int | None]:
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    return {
        "count": len(numeric),
        "mean": float(np.mean(numeric)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
        "std": float(np.std(numeric)),
    }


def aggregate(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"scalar_metrics": {}, "policy_metrics": {}}
    for key in ["train_auc", "train_ap", "val_auc", "val_ap"]:
        out["scalar_metrics"][key] = summarize([run.get(key) for run in per_run])
    policy_names = sorted(per_run[0]["policies"]) if per_run else []
    for name in policy_names:
        metrics = sorted(per_run[0]["policies"][name])
        out["policy_metrics"][name] = {
            metric: summarize([run["policies"][name][metric] for run in per_run]) for metric in metrics
        }
    return out


def margin_grid(scores: np.ndarray, arrays: dict[str, Any], groups: np.ndarray) -> list[float]:
    margins = []
    for group_index in groups:
        nearest = int(np.flatnonzero(arrays["is_nearest"][group_index])[0])
        valid = np.flatnonzero(arrays["mask"][group_index])
        non = [pos for pos in valid if pos != nearest]
        if not non:
            continue
        best = max(non, key=lambda pos: float(scores[group_index, pos]))
        margins.append(float(scores[group_index, best] - scores[group_index, nearest]))
    if not margins:
        return [1.0]
    unique = sorted(set(margins))
    return [max(unique) + 1.0e-6] + unique + [min(unique) - 1.0e-6]


def choose_train_tuned_margin(
    scores: np.ndarray,
    arrays: dict[str, Any],
    train_groups: np.ndarray,
    max_false_changes: int,
    max_parseq_worsen: int,
) -> tuple[float, dict[str, int]]:
    best_margin = 1.0
    best_metrics: dict[str, int] | None = None
    best_key: tuple[int, int, int, int] | None = None
    for margin in margin_grid(scores, arrays, train_groups):
        selected = select_policy(scores, arrays, train_groups, margin=margin)
        metrics = policy_metrics(arrays, train_groups, selected)
        if metrics["false_change_groups"] > max_false_changes:
            continue
        if metrics["parseq_worsen_groups"] > max_parseq_worsen:
            continue
        key = (
            -metrics["tesseract_delta_sum"],
            metrics["exact_oracle_groups"],
            -metrics["false_change_groups"],
            -metrics["changed_groups"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_margin = float(margin)
            best_metrics = metrics
    if best_metrics is None:
        selected = select_policy(scores, arrays, train_groups, margin=1.0)
        best_metrics = policy_metrics(arrays, train_groups, selected)
    return best_margin, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["edit_gain", "norm_edit_gain", "conf_gain", "combo"], default="combo")
    parser.add_argument("--conf-weight", type=float, default=1.0)
    parser.add_argument("--model-seed", type=int, action="append", default=[])
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--split-key-index", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--loss", choices=["bce", "mse"], default="bce")
    parser.add_argument("--include-onehot", action="store_true")
    parser.add_argument("--train-max-false-changes", type=int, default=1)
    parser.add_argument("--train-max-parseq-worsen", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_seeds = args.model_seed or [0]
    rows = read_jsonl(args.table)
    names = feature_names(rows, include_onehot=args.include_onehot)
    arrays = build_arrays(rows, names, target=args.target, conf_weight=args.conf_weight)
    arrays["features"], mean, std = standardize(arrays["features"], split_groups(arrays["keys"], args.split_key_index, args.val_seed)[0])
    train_groups, val_groups = split_groups(arrays["keys"], args.split_key_index, args.val_seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    per_run = []
    for seed in model_seeds:
        scores = train_one(
            arrays,
            train_groups=train_groups,
            model_seed=seed,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            device=device,
        )
        flat_train = arrays["mask"][train_groups].reshape(-1)
        flat_val = arrays["mask"][val_groups].reshape(-1)
        train_scores = scores[train_groups].reshape(-1)[flat_train]
        val_scores = scores[val_groups].reshape(-1)[flat_val]
        train_labels = arrays["teacher_labels"][train_groups].reshape(-1)[flat_train]
        val_labels = arrays["teacher_labels"][val_groups].reshape(-1)[flat_val]
        tuned_margin, tuned_train_metrics = choose_train_tuned_margin(
            scores,
            arrays,
            train_groups=train_groups,
            max_false_changes=args.train_max_false_changes,
            max_parseq_worsen=args.train_max_parseq_worsen,
        )
        policies = {
            "nearest": policy_metrics(arrays, val_groups, select_policy(scores, arrays, val_groups, margin=1.0e9)),
            "oracle": policy_metrics(arrays, val_groups, oracle_policy(arrays, val_groups)),
            "always_best_score": policy_metrics(arrays, val_groups, always_best_policy(scores, arrays, val_groups)),
            "margin_0": policy_metrics(arrays, val_groups, select_policy(scores, arrays, val_groups, margin=0.0)),
            "margin_0_05": policy_metrics(arrays, val_groups, select_policy(scores, arrays, val_groups, margin=0.05)),
            "margin_0_1": policy_metrics(arrays, val_groups, select_policy(scores, arrays, val_groups, margin=0.1)),
            "train_tuned": policy_metrics(arrays, val_groups, select_policy(scores, arrays, val_groups, margin=tuned_margin)),
        }
        per_run.append(
            {
                "model_seed": seed,
                "train_auc": roc_auc(train_scores, train_labels),
                "train_ap": average_precision(train_scores, train_labels),
                "val_auc": roc_auc(val_scores, val_labels),
                "val_ap": average_precision(val_scores, val_labels),
                "train_teacher_positive_rows": int(train_labels.sum()),
                "val_teacher_positive_rows": int(val_labels.sum()),
                "tuned_margin": tuned_margin,
                "tuned_train_metrics": tuned_train_metrics,
                "policies": policies,
            }
        )
    result = {
        "description": "Candidate-local row scorer trained on PARSeq teacher-derived targets with deployable features only.",
        "config": {
            "table": str(args.table),
            "target": args.target,
            "conf_weight": args.conf_weight,
            "model_seeds": model_seeds,
            "val_seed": args.val_seed,
            "split_key_index": args.split_key_index,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "include_onehot": args.include_onehot,
            "train_max_false_changes": args.train_max_false_changes,
            "train_max_parseq_worsen": args.train_max_parseq_worsen,
            "device": str(device),
            "feature_dim": len(names),
            "feature_names": names,
            "feature_mean_mean": float(np.mean(mean)),
            "feature_std_mean": float(np.mean(std)),
        },
        "table": {
            "groups": int(len(arrays["keys"])),
            "train_groups": int(len(train_groups)),
            "val_groups": int(len(val_groups)),
            "max_candidates": int(arrays["features"].shape[1]),
        },
        "per_run": per_run,
        "aggregate": aggregate(per_run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": result["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
