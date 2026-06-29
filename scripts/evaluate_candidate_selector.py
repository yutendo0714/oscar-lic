#!/usr/bin/env python3
"""Evaluate one-candidate OSCAR-LIC utility selectors with actual transmitted bpp."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from train_candidate_utility_baseline import FEATURES, average_precision, roc_auc, topk

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
from oscarlic.text_metrics import evaluate_pairs  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_analysis(path: Path) -> dict[tuple[int, int], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(row["source_index"]), int(row["candidate_index"])): row
        for row in data["details"]
    }


def split_by_group(groups: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(int(group) for group in groups))
    rng = random.Random(seed)
    rng.shuffle(unique)
    val_count = max(1, round(len(unique) * val_fraction))
    val_groups = set(unique[:val_count])
    val_mask = np.asarray([group in val_groups for group in groups], dtype=bool)
    return ~val_mask, val_mask


def is_numeric_feature(rows: list[dict], name: str) -> bool:
    try:
        for row in rows:
            float(row[name])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def available_features(rows: list[dict]) -> list[str]:
    names = [name for name in FEATURES if is_numeric_feature(rows, name)]
    if not names:
        raise ValueError("no numeric selector features are available in the candidate table")
    return names


def make_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_names = available_features(rows)
    x = np.asarray([[float(row[key]) for key in feature_names] for row in rows], dtype=np.float32)
    y = np.asarray([int(row["label_multi_teacher_improves"]) for row in rows], dtype=np.float32)
    groups = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    return x, y, groups, feature_names


def baseline_score_feature(rows: list[dict]) -> tuple[str, np.ndarray]:
    for name in ("latent_residual_rms", "residual_payload_bytes", "actual_total_bpp"):
        if is_numeric_feature(rows, name):
            return name, np.asarray([float(row[name]) for row in rows], dtype=np.float32)
    raise ValueError("no residual/bpp baseline score feature is available in the candidate table")


def train_linear_scores(
    rows: list[dict],
    seed: int,
    val_fraction: float,
    epochs: int,
    lr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    x, y, groups, feature_names = make_arrays(rows)
    train_mask, val_mask = split_by_group(groups, val_fraction, seed)
    x_train = x[train_mask]
    x_val = x[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    model = torch.nn.Linear(x.shape[1], 1)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    xt = torch.from_numpy((x_train - mean) / std)
    yt = torch.from_numpy(y_train).reshape(-1, 1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        all_scores = torch.sigmoid(model(torch.from_numpy((x - mean) / std))).reshape(-1).numpy()
        val_scores = all_scores[val_mask]
    meta = {
        "features": feature_names,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "linear_train_auc": roc_auc(all_scores[train_mask], y_train),
        "linear_val_auc": roc_auc(val_scores, y_val),
        "linear_val_ap": average_precision(val_scores, y_val),
        "linear_val_topk": topk(val_scores, y_val, [10, 20, 40]),
        "weights": {
            name: float(weight)
            for name, weight in zip(feature_names, model.weight.detach().reshape(-1).numpy())
        },
        "bias": float(model.bias.detach().item()),
    }
    return all_scores, train_mask, val_mask, meta


def candidate_split_maps(candidate_split: Path, base_structural_overhead_bytes: int) -> dict:
    rows = read_jsonl(candidate_split)
    by_key = {}
    for row in rows:
        key = (int(row["source_index"]), int(row["candidate_index"]))
        pixels = 8.0 * float(row["actual_total_bytes"]) / float(row["actual_total_bpp"])
        by_key[key] = {
            "pixels": pixels,
            "base_total_bpp": 8.0 * (int(row["base_payload_bytes"]) + base_structural_overhead_bytes) / pixels,
            "candidate_total_bpp": float(row["actual_total_bpp"]),
            "candidate_total_bytes": int(row["actual_total_bytes"]),
        }
    return by_key


def rows_by_image(rows: list[dict], row_indices: np.ndarray) -> dict[int, list[tuple[int, dict]]]:
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for row_index in row_indices:
        row = rows[int(row_index)]
        grouped[int(row["source_index"])].append((int(row_index), row))
    return grouped


def choose_base(grouped: dict[int, list[tuple[int, dict]]]) -> dict[int, tuple[int, dict] | None]:
    return {source_index: None for source_index in grouped}


def choose_oracle(grouped: dict[int, list[tuple[int, dict]]]) -> dict[int, tuple[int, dict] | None]:
    selected = {}
    for source_index, entries in grouped.items():
        best = min(entries, key=lambda item: (int(item[1]["multi_teacher_delta_distance"]), -int(item[1]["improving_evaluator_count"])))
        selected[source_index] = best if int(best[1]["multi_teacher_delta_distance"]) < 0 else None
    return selected


def choose_threshold(
    grouped: dict[int, list[tuple[int, dict]]],
    scores: np.ndarray,
    threshold: float,
) -> dict[int, tuple[int, dict] | None]:
    selected = {}
    for source_index, entries in grouped.items():
        best = max(entries, key=lambda item: float(scores[item[0]]))
        selected[source_index] = best if float(scores[best[0]]) >= threshold else None
    return selected


def tune_threshold(
    grouped: dict[int, list[tuple[int, dict]]],
    scores: np.ndarray,
) -> dict:
    thresholds = sorted({float(scores[row_index]) for entries in grouped.values() for row_index, _ in entries})
    if thresholds:
        thresholds = [max(thresholds) + 1e-6] + thresholds
    else:
        thresholds = [1.0]
    best = None
    for threshold in thresholds:
        selected = choose_threshold(grouped, scores, threshold)
        delta = sum(
            int(item[1]["multi_teacher_delta_distance"])
            for item in selected.values()
            if item is not None
        )
        selected_count = sum(1 for item in selected.values() if item is not None)
        candidate = {
            "threshold": float(threshold),
            "train_multi_teacher_delta_distance_sum": int(delta),
            "train_selected_images": int(selected_count),
        }
        key = (delta, selected_count)
        if best is None or key < best[0]:
            best = (key, candidate)
    assert best is not None
    return best[1]


def choose_top_images(
    grouped: dict[int, list[tuple[int, dict]]],
    scores: np.ndarray,
    image_budget: int,
) -> dict[int, tuple[int, dict] | None]:
    best_per_image = []
    for source_index, entries in grouped.items():
        best = max(entries, key=lambda item: float(scores[item[0]]))
        best_per_image.append((float(scores[best[0]]), source_index, best))
    best_per_image.sort(reverse=True)
    allowed = {source_index for _, source_index, _ in best_per_image[:image_budget]}
    return {
        source_index: best if source_index in allowed else None
        for _, source_index, best in best_per_image
    }


def evaluate_policy(
    name: str,
    selected: dict[int, tuple[int, dict] | None],
    grouped: dict[int, list[tuple[int, dict]]],
    analyses: dict[str, dict[tuple[int, int], dict]],
    bpp_by_key: dict,
    profile: str,
) -> dict:
    selected_count = sum(1 for item in selected.values() if item is not None)
    bpps = []
    multi_teacher_delta_sum = 0
    per_evaluator = {}
    for label, details in analyses.items():
        pairs = []
        for source_index, entries in sorted(grouped.items()):
            chosen = selected[source_index]
            base_key = (source_index, int(entries[0][1]["candidate_index"]))
            base_detail = details[base_key]
            if chosen is None:
                pairs.append((base_detail["reference"], base_detail["base_prediction"]))
            else:
                key = (source_index, int(chosen[1]["candidate_index"]))
                detail = details[key]
                pairs.append((detail["reference"], detail["candidate_prediction"]))
        per_evaluator[label] = evaluate_pairs(pairs, profile=profile).to_dict()

    for source_index, entries in sorted(grouped.items()):
        chosen = selected[source_index]
        if chosen is None:
            first = entries[0][1]
            key = (source_index, int(first["candidate_index"]))
            bpps.append(float(bpp_by_key[key]["base_total_bpp"]))
        else:
            row = chosen[1]
            key = (source_index, int(row["candidate_index"]))
            bpps.append(float(bpp_by_key[key]["candidate_total_bpp"]))
            multi_teacher_delta_sum += int(row["multi_teacher_delta_distance"])

    return {
        "name": name,
        "images": len(grouped),
        "selected_images": selected_count,
        "selected_fraction": selected_count / len(grouped) if grouped else None,
        "avg_actual_total_bpp": float(np.mean(bpps)) if bpps else None,
        "multi_teacher_delta_distance_sum": int(multi_teacher_delta_sum),
        "evaluators": per_evaluator,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--candidate-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", action="append", nargs=2, metavar=("LABEL", "ANALYSIS_JSON"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-budget", action="append", type=int)
    parser.add_argument("--base-structural-overhead-bytes", type=int, default=88)
    args = parser.parse_args()

    if not args.evaluator:
        raise SystemExit("provide at least one --evaluator LABEL ANALYSIS_JSON")

    rows = read_jsonl(args.table)
    scores, train_mask, val_mask, model_meta = train_linear_scores(
        rows,
        seed=args.seed,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        lr=args.lr,
    )
    baseline_feature, residual_scores = baseline_score_feature(rows)
    val_indices = np.flatnonzero(val_mask)
    train_indices = np.flatnonzero(train_mask)
    grouped_val = rows_by_image(rows, val_indices)
    grouped_train = rows_by_image(rows, train_indices)
    analyses = {label: read_analysis(Path(path)) for label, path in args.evaluator}
    bpp_by_key = candidate_split_maps(args.candidate_split, args.base_structural_overhead_bytes)
    budgets = args.image_budget or [5, 10, 20]
    tuned_threshold = tune_threshold(grouped_train, scores)

    policies = []
    policy_specs = [
        ("base", choose_base(grouped_val)),
        ("oracle_single_candidate", choose_oracle(grouped_val)),
        (f"linear_threshold_{args.threshold:g}", choose_threshold(grouped_val, scores, args.threshold)),
        (
            f"linear_train_tuned_threshold_{tuned_threshold['threshold']:.6g}",
            choose_threshold(grouped_val, scores, tuned_threshold["threshold"]),
        ),
    ]
    for budget in budgets:
        budget = min(int(budget), len(grouped_val))
        policy_specs.append((f"linear_top{budget}_images", choose_top_images(grouped_val, scores, budget)))
        policy_specs.append((f"residual_rms_top{budget}_images", choose_top_images(grouped_val, residual_scores, budget)))

    for name, selected in policy_specs:
        policies.append(evaluate_policy(name, selected, grouped_val, analyses, bpp_by_key, args.profile))

    output = {
        "table": str(args.table),
        "candidate_split": str(args.candidate_split),
        "profile": args.profile,
        "model": model_meta,
        "residual_baseline_score_feature": baseline_feature,
        "train_tuned_threshold": tuned_threshold,
        "val_source_indices": sorted(int(group) for group in grouped_val),
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
