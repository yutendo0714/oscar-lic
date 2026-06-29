#!/usr/bin/env python3
"""Train a small group-level no-op gate for assignment score policies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def group_rows(rows: list[dict]) -> dict[tuple[int, int, int, int], list[dict]]:
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return dict(groups)


def feature_names(
    rows: list[dict],
    include_latent_features: bool,
    include_image_features: bool,
    exclude_teacher_image_features: bool,
) -> list[str]:
    names = []
    for key, value in rows[0].items():
        if not isinstance(value, (int, float)):
            continue
        if key in {
            "seed",
            "source_index",
            "candidate_index",
            "candidate_slot",
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
        }:
            continue
        if key.startswith("label_") or key.startswith("group_"):
            continue
        if key.startswith("codebook_") and "_onehot_" in key:
            continue
        if key.startswith("codebook_") or key in {"topk_rank", "is_nearest", "code_equals_nearest", "assignment_relative_error"}:
            names.append(key)
        elif include_latent_features and key.startswith("latent_vec_"):
            names.append(key)
        elif include_image_features and key.startswith("img_"):
            if exclude_teacher_image_features and key.startswith("img_teacher_"):
                continue
            names.append(key)
    return sorted(set(names))


def build_group_table(
    rows: list[dict],
    target: str,
    difficulty_threshold: int,
    include_latent_features: bool,
    include_image_features: bool,
    exclude_teacher_image_features: bool,
    teacher_confidence_delta_threshold: float,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray, np.ndarray, list[str]]:
    names = feature_names(
        rows,
        include_latent_features=include_latent_features,
        include_image_features=include_image_features,
        exclude_teacher_image_features=exclude_teacher_image_features,
    )
    groups = group_rows(rows)
    keys = []
    values = []
    labels = []
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["topk_rank"]))
        nearest = next(row for row in ordered if int(row["is_nearest"]))
        feats = []
        for name in names:
            arr = np.asarray([float(row.get(name, 0.0)) for row in ordered], dtype=np.float32)
            feats.extend([float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), float(nearest.get(name, 0.0))])
        keys.append(key)
        values.append(feats)
        if target == "oracle_change":
            label = int(nearest["group_oracle_changes_code"])
        elif target == "tesseract_difficulty":
            label = int(int(nearest.get("tesseract_nearest_distance", 0)) >= difficulty_threshold)
        elif target == "parseq_teacher_edit_improve":
            label = int(
                any(
                    (not int(row["is_nearest"]))
                    and int(row.get("img_teacher_parseq_edit_delta_vs_nearest", 0)) < 0
                    for row in ordered
                )
            )
        elif target == "parseq_teacher_norm_edit_improve":
            label = int(
                any(
                    (not int(row["is_nearest"]))
                    and int(row.get("img_teacher_parseq_norm_edit_delta_vs_nearest", 0)) < 0
                    for row in ordered
                )
            )
        elif target == "parseq_teacher_conf_gain":
            label = int(
                any(
                    (not int(row["is_nearest"]))
                    and float(row.get("img_teacher_parseq_confidence_delta_vs_nearest", 0.0))
                    > teacher_confidence_delta_threshold
                    for row in ordered
                )
            )
        else:
            raise ValueError(f"unsupported target: {target}")
        labels.append(label)
    return keys, np.asarray(values, dtype=np.float32), np.asarray(labels, dtype=np.float32), names


def threshold_grid(scores: np.ndarray) -> list[float]:
    unique = sorted(set(float(value) for value in scores))
    if not unique:
        return [1.0]
    return [max(unique) + 1e-6] + unique + [min(unique) - 1e-6]


def load_assignment_scores(paths: list[Path], model_seed: int, val_seed: int | None = None) -> dict[tuple[int, int, int, int], list[dict]]:
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            if int(row.get("model_seed", -1)) != model_seed:
                continue
            if val_seed is not None and int(row.get("seed", -1)) != val_seed:
                continue
            groups[group_key(row)].append(row)
    return dict(groups)


def policy_metrics(selected: list[dict]) -> dict:
    changed = [not int(row["is_nearest"]) for row in selected]
    oracle_changes = [bool(int(row.get("_group_oracle_changes_code", row.get("group_oracle_changes_code", 0)))) for row in selected]
    exact_oracle = [
        bool(int(row.get("label_assignment_oracle_choice", 0))) and (not int(row["is_nearest"]))
        for row in selected
    ]
    return {
        "groups": len(selected),
        "changed_groups": int(sum(changed)),
        "correct_nearest_groups": int(
            sum((not is_changed) and (not oracle_change) for is_changed, oracle_change in zip(changed, oracle_changes))
        ),
        "exact_oracle_groups": int(sum(exact_oracle)),
        "false_change_groups": int(
            sum(is_changed and (not oracle_change) for is_changed, oracle_change in zip(changed, oracle_changes))
        ),
        "missed_oracle_groups": int(
            sum((not is_changed) and oracle_change for is_changed, oracle_change in zip(changed, oracle_changes))
        ),
        "oracle_change_groups": int(sum(oracle_changes)),
        "wrong_change_groups": int(
            sum(
                is_changed and oracle_change and (not is_exact)
                for is_changed, oracle_change, is_exact in zip(changed, oracle_changes, exact_oracle)
            )
        ),
        "parseq_delta_sum": int(sum(int(row["parseq_delta_vs_nearest"]) for row in selected)),
        "tesseract_delta_sum": int(sum(int(row["tesseract_delta_vs_nearest"]) for row in selected)),
        "parseq_worsen_groups": int(sum(int(row["parseq_delta_vs_nearest"]) > 0 for row in selected)),
        "tesseract_improve_groups": int(sum(int(row["tesseract_delta_vs_nearest"]) < 0 for row in selected)),
        "tesseract_worsen_groups": int(sum(int(row["tesseract_delta_vs_nearest"]) > 0 for row in selected)),
    }


def attach_group_metadata(row: dict, group: list[dict]) -> dict:
    out = dict(row)
    out["_group_oracle_changes_code"] = int(
        any(int(item.get("label_assignment_oracle_choice", 0)) and not int(item["is_nearest"]) for item in group)
    )
    return out


def threshold_slug(threshold: float) -> str:
    text = f"{threshold:.6g}".replace("-", "m").replace(".", "_")
    return text


def threshold_metric_name(threshold: float) -> str:
    return f"gate_threshold_{threshold_slug(threshold)}"


def threshold_policy_name(threshold: float) -> str:
    return f"group_gate_threshold_{threshold_slug(threshold)}"


def policy_output_path(
    base: Path,
    model_seed_count: int,
    model_seed: int,
    threshold_count: int,
    threshold: float,
) -> Path:
    path = base
    if model_seed_count > 1:
        path = path.with_name(f"{path.stem}.model{model_seed}{path.suffix}")
    if threshold_count > 1:
        path = path.with_name(f"{path.stem}.thr{threshold_slug(threshold)}{path.suffix}")
    return path


def named_policy_output_path(
    base: Path,
    model_seed_count: int,
    model_seed: int,
    policy_suffix: str,
) -> Path:
    path = base
    if model_seed_count > 1:
        path = path.with_name(f"{path.stem}.model{model_seed}{path.suffix}")
    return path.with_name(f"{path.stem}.{policy_suffix}{path.suffix}")


def write_policy_output(
    path: Path,
    selected: list[dict],
    policy: str,
    model_seed: int,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    parseq_delta = 0
    tesseract_delta = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(selected, key=lambda item: (int(item["source_index"]), int(item["candidate_index"]))):
            code_index = int(row["code_index"])
            nearest_code = int(row["nearest_code"])
            changed += int(code_index != nearest_code)
            parseq_delta += int(row.get("parseq_delta_vs_nearest", 0))
            tesseract_delta += int(row.get("tesseract_delta_vs_nearest", 0))
            out = {
                "source_index": int(row["source_index"]),
                "candidate_index": int(row["candidate_index"]),
                "assignment_code_index": code_index,
                "code_index": code_index,
                "nearest_code": nearest_code,
                "topk_rank": int(row["topk_rank"]),
                "selected_by_policy": True,
                "selector_score": float(row["score"]),
                "gate_threshold": float(threshold),
                "score_val_seed": int(row["val_seed"]),
                "score_model_seed": int(row["model_seed"]),
                "gate_model_seed": int(model_seed),
                "score_model_seeds": [int(row["model_seed"])],
                "score_count": 1,
                "score_min": float(row["score"]),
                "score_max": float(row["score"]),
                "seed": int(row["seed"]),
                "real_seed": int(row.get("real_seed", row["seed"])),
                "policy": policy,
                "parseq_delta_vs_nearest": int(row.get("parseq_delta_vs_nearest", 0)),
                "tesseract_delta_vs_nearest": int(row.get("tesseract_delta_vs_nearest", 0)),
                "label_assignment_oracle_choice": int(row.get("label_assignment_oracle_choice", 0)),
                "group_oracle_changes_code": int(
                    row.get("_group_oracle_changes_code", row.get("group_oracle_changes_code", 0))
                ),
                "source_image": row.get("source_image"),
                "image_path": row.get("image_path"),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    summary = {
        "output": str(path),
        "policy": policy,
        "gate_model_seed": int(model_seed),
        "gate_threshold": float(threshold),
        "groups": len(selected),
        "changed_groups": changed,
        "parseq_delta_sum_vs_nearest": parseq_delta,
        "tesseract_delta_sum_vs_nearest": tesseract_delta,
    }
    path.with_suffix(path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def choose_rows(
    score_groups: dict[tuple[int, int, int, int], list[dict]],
    gate_scores_by_key: dict[tuple[int, int, int, int], float],
    threshold: float,
) -> list[dict]:
    selected = []
    for key, group in sorted(score_groups.items()):
        nearest = next(row for row in group if int(row["is_nearest"]))
        best = max(group, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))
        selected.append(attach_group_metadata(best if gate_scores_by_key[key] >= threshold else nearest, group))
    return selected


def tune_threshold(
    score_groups: dict[tuple[int, int, int, int], list[dict]],
    gate_scores_by_key: dict[tuple[int, int, int, int], float],
) -> float:
    scores = np.asarray([gate_scores_by_key[key] for key in score_groups], dtype=np.float32)
    best_threshold = 1.0
    best_key = None
    for threshold in threshold_grid(scores):
        metrics = policy_metrics(choose_rows(score_groups, gate_scores_by_key, threshold))
        key = (
            metrics["parseq_worsen_groups"],
            metrics["tesseract_delta_sum"],
            metrics["tesseract_worsen_groups"],
            metrics["changed_groups"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


class GateMLP(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = torch.nn.Linear(dim, 1)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def train_one(
    keys: list[tuple[int, int, int, int]],
    features: np.ndarray,
    labels: np.ndarray,
    assignment_score_paths: list[Path],
    val_seed: int,
    model_seed: int,
    args: argparse.Namespace,
) -> dict:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    split_values = np.asarray([key[args.split_key_index] for key in keys], dtype=np.int64)
    train_mask = split_values != val_seed
    val_mask = split_values == val_seed
    if not bool(train_mask.any()) or not bool(val_mask.any()):
        raise ValueError(
            f"empty train/val split for split_key_index={args.split_key_index}, val_seed={val_seed}"
        )
    mean = features[train_mask].mean(axis=0, keepdims=True)
    std = features[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    norm = (features - mean) / std

    model = GateMLP(dim=features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    train_x = torch.from_numpy(norm[train_mask].astype(np.float32)).to(device)
    train_y = torch.from_numpy(labels[train_mask].astype(np.float32)).to(device)
    positives = float(train_y.sum().item())
    negatives = float(len(train_y) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        gate_scores = torch.sigmoid(model(torch.from_numpy(norm.astype(np.float32)).to(device))).cpu().numpy()
    gate_scores_by_key = {key: float(score) for key, score in zip(keys, gate_scores)}

    all_score_groups = load_assignment_scores(assignment_score_paths, model_seed=args.assignment_model_seed)
    train_score_groups = {key: group for key, group in all_score_groups.items() if key[args.split_key_index] != val_seed}
    val_score_groups = {key: group for key, group in all_score_groups.items() if key[args.split_key_index] == val_seed}
    threshold = tune_threshold(train_score_groups, gate_scores_by_key)
    selected = choose_rows(val_score_groups, gate_scores_by_key, threshold)
    threshold_selected = {
        float(policy_threshold): choose_rows(val_score_groups, gate_scores_by_key, threshold=float(policy_threshold))
        for policy_threshold in args.policy_threshold
    }
    assignment_always_selected = [
        attach_group_metadata(max(group, key=lambda row: (float(row["score"]), -int(row["topk_rank"]))), group)
        for group in sorted(val_score_groups.values(), key=lambda group: group_key(group[0]))
    ]
    nearest_selected = [
        attach_group_metadata(next(row for row in group if int(row["is_nearest"])), group)
        for group in sorted(val_score_groups.values(), key=lambda group: group_key(group[0]))
    ]
    oracle_selected = [
        attach_group_metadata(
            next(
                (
                    row
                    for row in group
                    if int(row["label_assignment_oracle_choice"]) and not int(row["is_nearest"])
                ),
                next(row for row in group if int(row["is_nearest"])),
            ),
            group,
        )
        for group in sorted(val_score_groups.values(), key=lambda group: group_key(group[0]))
    ]
    if args.policy_output is not None:
        if args.write_train_tuned_policy:
            out_path = named_policy_output_path(
                args.policy_output,
                len(args.model_seed),
                model_seed,
                "train_tuned",
            )
            write_policy_output(
                out_path,
                selected,
                policy="group_gate_train_tuned",
                model_seed=model_seed,
                threshold=float(threshold),
            )
        for policy_threshold, policy_selected in threshold_selected.items():
            out_path = policy_output_path(
                args.policy_output,
                len(args.model_seed),
                model_seed,
                len(args.policy_threshold),
                float(policy_threshold),
            )
            write_policy_output(
                out_path,
                policy_selected,
                policy=threshold_policy_name(float(policy_threshold)),
                model_seed=model_seed,
                threshold=float(policy_threshold),
            )
    policy_metrics_by_name = {
        "nearest": policy_metrics(nearest_selected),
        "oracle": policy_metrics(oracle_selected),
        "assignment_always_best": policy_metrics(assignment_always_selected),
        "gate_train_tuned": policy_metrics(selected),
    }
    for policy_threshold, policy_selected in threshold_selected.items():
        policy_metrics_by_name[threshold_metric_name(float(policy_threshold))] = policy_metrics(policy_selected)
    return {
        "val_seed": val_seed,
        "split_key_index": int(args.split_key_index),
        "model_seed": model_seed,
        "assignment_model_seed": args.assignment_model_seed,
        "device": str(device),
        "train_groups": int(train_mask.sum()),
        "val_groups": int(val_mask.sum()),
        "train_positive_rate": float(labels[train_mask].mean()),
        "val_positive_rate": float(labels[val_mask].mean()),
        "train_auc": roc_auc(gate_scores[train_mask], labels[train_mask]),
        "train_ap": average_precision(gate_scores[train_mask], labels[train_mask]),
        "val_auc": roc_auc(gate_scores[val_mask], labels[val_mask]),
        "val_ap": average_precision(gate_scores[val_mask], labels[val_mask]),
        "threshold": threshold,
        "policy_metrics": policy_metrics_by_name,
    }


def aggregate(results: list[dict]) -> dict:
    out = {"scalar_metrics": {}, "policy_metrics": {}}
    for metric in ["train_auc", "train_ap", "val_auc", "val_ap"]:
        values = [row[metric] for row in results if row[metric] is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            out["scalar_metrics"][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(len(arr)),
            }
    policy_names = sorted({name for row in results for name in row["policy_metrics"]})
    for policy in policy_names:
        out["policy_metrics"][policy] = {}
        metric_names = sorted({name for row in results for name in row["policy_metrics"][policy]})
        for metric in metric_names:
            arr = np.asarray([float(row["policy_metrics"][policy][metric]) for row in results], dtype=np.float64)
            out["policy_metrics"][policy][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--assignment-score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--split-key-index", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--assignment-model-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--target",
        choices=[
            "oracle_change",
            "tesseract_difficulty",
            "parseq_teacher_edit_improve",
            "parseq_teacher_norm_edit_improve",
            "parseq_teacher_conf_gain",
        ],
        default="oracle_change",
    )
    parser.add_argument("--difficulty-threshold", type=int, default=2)
    parser.add_argument("--teacher-confidence-delta-threshold", type=float, default=0.0)
    parser.add_argument("--include-latent-features", action="store_true")
    parser.add_argument("--include-image-features", action="store_true")
    parser.add_argument("--exclude-teacher-image-features", action="store_true")
    parser.add_argument("--policy-threshold", type=float, action="append")
    parser.add_argument("--policy-output", type=Path)
    parser.add_argument("--write-train-tuned-policy", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.model_seed is None:
        args.model_seed = [0]
    if args.policy_threshold is None:
        args.policy_threshold = [0.5]

    rows = read_jsonl(args.table)
    keys, features, labels, names = build_group_table(
        rows,
        target=args.target,
        difficulty_threshold=args.difficulty_threshold,
        include_latent_features=args.include_latent_features,
        include_image_features=args.include_image_features,
        exclude_teacher_image_features=args.exclude_teacher_image_features,
        teacher_confidence_delta_threshold=args.teacher_confidence_delta_threshold,
    )
    results = [
        train_one(keys, features, labels, args.assignment_score, args.val_seed, model_seed, args)
        for model_seed in args.model_seed
    ]
    output = {
        "description": "Small group-level no-op gate over assignment score policies.",
        "table": str(args.table),
        "assignment_scores": [str(path) for path in args.assignment_score],
        "config": {
            "val_seed": args.val_seed,
            "split_key_index": int(args.split_key_index),
            "model_seeds": args.model_seed,
            "assignment_model_seed": args.assignment_model_seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "feature_names": names,
            "group_feature_dim": int(features.shape[1]),
            "target": args.target,
            "difficulty_threshold": args.difficulty_threshold,
            "teacher_confidence_delta_threshold": args.teacher_confidence_delta_threshold,
            "include_latent_features": bool(args.include_latent_features),
            "include_image_features": bool(args.include_image_features),
            "exclude_teacher_image_features": bool(args.exclude_teacher_image_features),
            "write_train_tuned_policy": bool(args.write_train_tuned_policy),
        },
        "per_run": results,
        "aggregate": aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
