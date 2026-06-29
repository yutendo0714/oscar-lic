#!/usr/bin/env python3
"""Feature-based conditional stop/continue diagnostics for subset edge tables."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np


PROFILE = "unicode_strict_v1"
TARGET = "label_tesseract_improve_parseq_safe_unicode_strict_v1"
CANDIDATES = ("a", "b", "c")
TRANSITIONS = ("0_to_1", "1_to_2", "2_to_3")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tuple_members(value: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in value))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def make_categories(rows: list[dict]) -> dict[str, list[str]]:
    return {
        "transition": sorted({str(row["transition"]) for row in rows}),
        "added_candidate": sorted({str(row["added_candidate"]) for row in rows}),
        "source": sorted({str(row.get("source", "unknown")) for row in rows}),
    }


def onehot(value: str, categories: list[str]) -> list[float]:
    return [1.0 if value == category else 0.0 for category in categories]


def build_features(rows: list[dict], mode: str, categories: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    vectors: list[list[float]] = []
    include_candidate = mode in {
        "transition_candidate_rate_psnr",
        "transition_candidate_rate_psnr_source",
        "transition_candidate_rate_psnr_ocr_state_upper",
    }
    include_rate = include_candidate
    include_source = mode in {"transition_candidate_rate_psnr_source"}
    include_ocr_upper = mode in {"transition_candidate_rate_psnr_ocr_state_upper"}

    for row in rows:
        features: list[float] = []
        row_names: list[str] = []

        for name, value in zip(
            [f"transition={item}" for item in categories["transition"]],
            onehot(str(row["transition"]), categories["transition"]),
            strict=True,
        ):
            row_names.append(name)
            features.append(value)

        if include_candidate:
            for name, value in zip(
                [f"added_candidate={item}" for item in categories["added_candidate"]],
                onehot(str(row["added_candidate"]), categories["added_candidate"]),
                strict=True,
            ):
                row_names.append(name)
                features.append(value)
            source_members = set(row.get("source_members", []))
            for candidate in CANDIDATES:
                row_names.append(f"source_has_{candidate}")
                features.append(1.0 if candidate in source_members else 0.0)
            row_names.append("source_cardinality")
            features.append(float(row["source_cardinality"]))

        if include_rate:
            enh = float(row["added_enhancement_bpp"])
            psnr = float(row["added_psnr_delta_db"])
            row_names.extend(
                [
                    "added_enhancement_bpp",
                    "added_psnr_delta_db",
                    "enhancement_bpp_per_psnr",
                ]
            )
            features.extend([enh, psnr, enh / max(abs(psnr), 1e-6)])

        if include_source:
            for name, value in zip(
                [f"source={item}" for item in categories["source"]],
                onehot(str(row.get("source", "unknown")), categories["source"]),
                strict=True,
            ):
                row_names.append(name)
                features.append(value)

        if include_ocr_upper:
            # Non-deployable upper-control features. They use OCR distances from the source state,
            # so they are diagnostic only and must not be promoted as encoder-side features.
            row_names.extend(
                [
                    "tesseract_source_distance",
                    "parseq_source_distance",
                    "ocr_source_distance_gap",
                ]
            )
            tess = float(row[f"tesseract_source_distance_{PROFILE}"])
            parseq = float(row[f"parseq_source_distance_{PROFILE}"])
            features.extend([tess, parseq, tess - parseq])

        if not names:
            names = row_names
        vectors.append(features)

    return np.asarray(vectors, dtype=np.float64), names


def standardize_train_val(
    x_train: np.ndarray, x_eval: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_eval - mean) / std, mean.reshape(-1), std.reshape(-1)


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    l2: float,
    steps: int,
    lr: float,
) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1), dtype=np.float64), x_train], axis=1)
    weights = np.zeros(x_aug.shape[1], dtype=np.float64)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    sample_weights = np.where(y_train > 0.5, 0.5 / pos, 0.5 / neg) * len(y_train)
    for _ in range(steps):
        probs = sigmoid(x_aug @ weights)
        grad = (x_aug.T @ ((probs - y_train) * sample_weights)) / len(y_train)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return weights


def predict_logistic(x_eval: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_eval.shape[0], 1), dtype=np.float64), x_eval], axis=1)
    return sigmoid(x_aug @ weights)


def edge_metrics(rows: list[dict], selected: np.ndarray, target: str, profile: str) -> dict:
    labels = np.asarray([1 if row[target] else 0 for row in rows], dtype=np.int64)
    selected_i = selected.astype(bool)
    tp = int(np.sum((labels == 1) & selected_i))
    fp = int(np.sum((labels == 0) & selected_i))
    tn = int(np.sum((labels == 0) & ~selected_i))
    fn = int(np.sum((labels == 1) & ~selected_i))
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    chosen = [row for row, flag in zip(rows, selected_i, strict=True) if flag]
    by_transition = {
        transition: int(sum(1 for row in chosen if row["transition"] == transition)) for transition in TRANSITIONS
    }
    return {
        "rows": len(rows),
        "positives": int(labels.sum()),
        "selected": int(selected_i.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tesseract_delta_sum": int(sum(row[f"tesseract_delta_{profile}"] for row in chosen)),
        "parseq_delta_sum": int(sum(row[f"parseq_delta_{profile}"] for row in chosen)),
        "mean_added_enhancement_bpp_selected": None
        if not chosen
        else float(np.mean([float(row["added_enhancement_bpp"]) for row in chosen])),
        "selected_by_transition": by_transition,
    }


def score_edge_metric(metric: dict, objective: str) -> tuple:
    precision = -1.0 if metric["precision"] is None else float(metric["precision"])
    recall = -1.0 if metric["recall"] is None else float(metric["recall"])
    f1 = -1.0 if metric["f1"] is None else float(metric["f1"])
    gain = -float(metric["tesseract_delta_sum"])
    if objective == "max_f1":
        return (f1, precision, recall, gain, -metric["fp"], -metric["selected"])
    if objective == "max_gain_precision90":
        if precision < 0.90 or metric["parseq_delta_sum"] > 0:
            return (-1e9, precision, f1, gain, -metric["fp"], -metric["selected"])
        return (gain, precision, f1, recall, -metric["fp"], -metric["selected"])
    raise ValueError(f"unknown edge objective: {objective}")


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 101))
    values = np.unique(np.concatenate([np.asarray([-0.001, 0.001, 0.5, 0.999, 1.001]), quantiles]))
    return np.sort(values)


def choose_edge_threshold(rows: list[dict], scores: np.ndarray, objective: str, target: str, profile: str) -> float:
    best_threshold = 1.001
    best_score = None
    for threshold in threshold_grid(scores):
        metric = edge_metrics(rows, scores >= threshold, target, profile)
        score = score_edge_metric(metric, objective)
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def build_state_outcomes(rows: list[dict], profile: str) -> dict[tuple[int, tuple[str, ...]], dict]:
    outcomes: dict[tuple[int, tuple[str, ...]], dict] = {}
    by_cardinality = sorted(rows, key=lambda row: int(row["source_cardinality"]))
    for row in by_cardinality:
        index = int(row["index"])
        source = tuple_members(row["source_members"])
        target = tuple_members(row["target_members"])
        source_key = (index, source)
        target_key = (index, target)
        if source_key not in outcomes:
            outcomes[source_key] = {
                "members": source,
                "enhancement_bpp": 0.0 if not source else None,
                "psnr_delta_db": 0.0 if not source else None,
                "tesseract_distance": int(row[f"tesseract_source_distance_{profile}"]),
                "parseq_distance": int(row[f"parseq_source_distance_{profile}"]),
            }
        source_outcome = outcomes[source_key]
        target_enh = None
        target_psnr = None
        if source_outcome.get("enhancement_bpp") is not None:
            target_enh = float(source_outcome["enhancement_bpp"]) + float(row["added_enhancement_bpp"])
        if source_outcome.get("psnr_delta_db") is not None:
            target_psnr = float(source_outcome["psnr_delta_db"]) + float(row["added_psnr_delta_db"])
        existing = outcomes.get(target_key, {})
        outcomes[target_key] = {
            "members": target,
            "enhancement_bpp": target_enh if target_enh is not None else existing.get("enhancement_bpp"),
            "psnr_delta_db": target_psnr if target_psnr is not None else existing.get("psnr_delta_db"),
            "tesseract_distance": int(row[f"tesseract_target_distance_{profile}"]),
            "parseq_distance": int(row[f"parseq_target_distance_{profile}"]),
        }
    return outcomes


def policy_metrics(
    rows: list[dict],
    final_members_by_index: dict[int, tuple[str, ...]],
    profile: str,
) -> dict:
    outcomes = build_state_outcomes(rows, profile)
    indices = sorted(final_members_by_index)
    tess_delta = 0
    parseq_delta = 0
    enh_bpps: list[float] = []
    psnrs: list[float] = []
    policy_counts = Counter()
    improved = 0
    worsened = 0
    unchanged = 0
    for index in indices:
        base = outcomes[(index, tuple())]
        final_members = final_members_by_index[index]
        final = outcomes[(index, final_members)]
        delta = int(final["tesseract_distance"] - base["tesseract_distance"])
        pdelta = int(final["parseq_distance"] - base["parseq_distance"])
        tess_delta += delta
        parseq_delta += pdelta
        enh_bpps.append(float(final.get("enhancement_bpp") or 0.0))
        psnrs.append(float(final.get("psnr_delta_db") or 0.0))
        policy_name = "base" if not final_members else "".join(final_members)
        policy_counts[policy_name] += 1
        if delta < 0:
            improved += 1
        elif delta > 0:
            worsened += 1
        else:
            unchanged += 1
    cardinality_counts = Counter(len(members) for members in final_members_by_index.values())
    return {
        "images": len(indices),
        "changed_images": int(sum(1 for members in final_members_by_index.values() if members)),
        "tesseract_delta_sum": int(tess_delta),
        "parseq_delta_sum": int(parseq_delta),
        "improved_worse_unchanged": [int(improved), int(worsened), int(unchanged)],
        "mean_enhancement_bpp": float(np.mean(enh_bpps)) if enh_bpps else 0.0,
        "mean_psnr_delta_db": float(np.mean(psnrs)) if psnrs else 0.0,
        "cardinality_counts": {str(key): int(value) for key, value in sorted(cardinality_counts.items())},
        "policy_counts": {key: int(value) for key, value in sorted(policy_counts.items())},
    }


def all_indices(rows: list[dict]) -> list[int]:
    return sorted({int(row["index"]) for row in rows})


def fixed_policy_members(rows: list[dict], members: tuple[str, ...]) -> dict[int, tuple[str, ...]]:
    return {index: members for index in all_indices(rows)}


def choose_best_fixed_single(rows: list[dict], train_indices: set[int], profile: str) -> tuple[str, ...]:
    candidates = []
    for candidate in CANDIDATES:
        members = (candidate,)
        finals = fixed_policy_members(rows, members)
        train_finals = {index: finals[index] for index in train_indices}
        metric = policy_metrics(rows, train_finals, profile)
        candidates.append(((-metric["tesseract_delta_sum"], -metric["parseq_delta_sum"], -ord(candidate)), members))
    return max(candidates, key=lambda item: item[0])[1]


def subset_oracle_members(rows: list[dict], indices: Iterable[int], profile: str) -> dict[int, tuple[str, ...]]:
    outcomes = build_state_outcomes(rows, profile)
    possible_states = [tuple(), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")]
    out = {}
    for index in indices:
        base = outcomes[(index, tuple())]
        candidates = []
        for members in possible_states:
            state = outcomes[(index, members)]
            tess_delta = int(state["tesseract_distance"] - base["tesseract_distance"])
            parseq_delta = int(state["parseq_distance"] - base["parseq_distance"])
            candidates.append((tess_delta, max(parseq_delta, 0), len(members), str(members), members))
        out[index] = min(candidates, key=lambda item: item[:4])[-1]
    return out


def edge_lookup(rows: list[dict]) -> dict[tuple[int, tuple[str, ...]], list[int]]:
    lookup: dict[tuple[int, tuple[str, ...]], list[int]] = defaultdict(list)
    for row_id, row in enumerate(rows):
        lookup[(int(row["index"]), tuple_members(row["source_members"]))].append(row_id)
    return lookup


def simulate_greedy_policy(
    rows: list[dict],
    scores: np.ndarray,
    threshold_by_index: dict[int, float],
    indices: Iterable[int],
) -> dict[int, tuple[str, ...]]:
    lookup = edge_lookup(rows)
    finals = {}
    for index in sorted(indices):
        members: tuple[str, ...] = tuple()
        threshold = threshold_by_index[index]
        for _ in range(3):
            available = lookup.get((index, members), [])
            if not available:
                break
            best_row_id = max(available, key=lambda row_id: float(scores[row_id]))
            if float(scores[best_row_id]) < threshold:
                break
            members = tuple_members(rows[best_row_id]["target_members"])
        finals[index] = members
    return finals


def choose_policy_threshold(
    rows: list[dict],
    scores: np.ndarray,
    threshold_scores: np.ndarray,
    train_indices: set[int],
    profile: str,
) -> float:
    best_threshold = 1.001
    best_score = None
    for threshold in threshold_grid(threshold_scores):
        finals = simulate_greedy_policy(rows, scores, {index: float(threshold) for index in train_indices}, train_indices)
        metric = policy_metrics(rows, finals, profile)
        # Require non-worse PARSeq on train, then maximize Tesseract improvement with shorter streams as a tie-break.
        if metric["parseq_delta_sum"] > 0:
            score = (-1e9, -metric["changed_images"], -metric["mean_enhancement_bpp"])
        else:
            score = (
                -metric["tesseract_delta_sum"],
                -metric["changed_images"],
                -metric["mean_enhancement_bpp"],
                metric["improved_worse_unchanged"][0],
            )
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def crossfit_model(
    rows: list[dict],
    x: np.ndarray,
    *,
    target: str,
    profile: str,
    folds: int,
    mode: str,
    l2: float,
    steps: int,
    lr: float,
) -> dict:
    labels = np.asarray([1 if row[target] else 0 for row in rows], dtype=np.float64)
    indices = np.asarray([int(row["index"]) for row in rows], dtype=np.int64)
    all_scores = np.zeros(len(rows), dtype=np.float64)
    edge_selected_by_objective = {
        "max_f1": np.zeros(len(rows), dtype=bool),
        "max_gain_precision90": np.zeros(len(rows), dtype=bool),
    }
    policy_threshold_by_index: dict[int, float] = {}
    fold_summaries = []

    for fold in range(folds):
        train_mask = (indices % folds) != fold
        eval_mask = ~train_mask
        train_rows = [row for row, flag in zip(rows, train_mask, strict=True) if flag]
        eval_rows = [row for row, flag in zip(rows, eval_mask, strict=True) if flag]
        x_train, x_eval, mean, std = standardize_train_val(x[train_mask], x[eval_mask])
        weights = fit_logistic(x_train, labels[train_mask], l2=l2, steps=steps, lr=lr)
        train_scores = predict_logistic(x_train, weights)
        eval_scores = predict_logistic(x_eval, weights)
        all_scores[eval_mask] = eval_scores
        for objective in edge_selected_by_objective:
            threshold = choose_edge_threshold(train_rows, train_scores, objective, target, profile)
            edge_selected_by_objective[objective][eval_mask] = eval_scores >= threshold
        train_indices = {int(row["index"]) for row in train_rows}
        eval_indices = {int(row["index"]) for row in eval_rows}
        policy_threshold = choose_policy_threshold(
            rows,
            all_scores_for_fold(rows, train_mask, train_scores, eval_mask, eval_scores),
            train_scores,
            train_indices,
            profile,
        )
        for index in eval_indices:
            policy_threshold_by_index[index] = policy_threshold
        fold_metric = edge_metrics(eval_rows, eval_scores >= choose_edge_threshold(train_rows, train_scores, "max_f1", target, profile), target, profile)
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "eval_rows": int(eval_mask.sum()),
                "train_positive_rate": float(labels[train_mask].mean()),
                "eval_positive_rate": float(labels[eval_mask].mean()),
                "score_mean_eval": float(eval_scores.mean()),
                "score_std_eval": float(eval_scores.std(ddof=0)),
                "max_f1_eval_selected": fold_metric["selected"],
                "policy_threshold": policy_threshold,
                "weight_l2_norm": float(np.linalg.norm(weights[1:])),
                "feature_mean_abs_z_eval": float(np.mean(np.abs((x[eval_mask] - mean.reshape(1, -1)) / std.reshape(1, -1)))),
            }
        )

    edge_policies = {
        objective: edge_metrics(rows, selected, target, profile) for objective, selected in edge_selected_by_objective.items()
    }
    greedy_finals = simulate_greedy_policy(rows, all_scores, policy_threshold_by_index, all_indices(rows))
    greedy_metric = policy_metrics(rows, greedy_finals, profile)
    greedy_metric["thresholds_by_fold_index"] = {
        str(index): float(policy_threshold_by_index[index]) for index in sorted(policy_threshold_by_index)
    }
    greedy_metric["final_members_by_index"] = {
        str(index): "".join(members) if members else "base" for index, members in sorted(greedy_finals.items())
    }
    return {
        "mode": mode,
        "edge_policies": edge_policies,
        "greedy_policy": greedy_metric,
        "folds": fold_summaries,
        "score_summary": {
            "mean": float(all_scores.mean()),
            "std": float(all_scores.std(ddof=0)),
            "min": float(all_scores.min()),
            "max": float(all_scores.max()),
        },
    }


def all_scores_for_fold(
    rows: list[dict],
    train_mask: np.ndarray,
    train_scores: np.ndarray,
    eval_mask: np.ndarray,
    eval_scores: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(rows), dtype=np.float64)
    scores[train_mask] = train_scores
    scores[eval_mask] = eval_scores
    return scores


def scalar_metrics(prefix: str, value: object, out: dict) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            scalar_metrics(f"{prefix}_{key}", child, out)


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(path: Path, result: dict) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "This is a small diagnostic over OCR-derived conditional edge labels.",
        "Feature models are cross-fitted by `index % folds`; threshold selection uses train folds only.",
        "",
        "## Edge Baselines",
        "",
        "| policy | selected | TP/FP/FN | precision | recall | F1 | Tess delta | PARSeq delta | selected transitions |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in result["edge_baselines"].items():
        lines.append(
            f"| {name} | {row['selected']} | {row['tp']}/{row['fp']}/{row['fn']} | "
            f"{format_metric(row['precision'])} | {format_metric(row['recall'])} | {format_metric(row['f1'])} | "
            f"{row['tesseract_delta_sum']} | {row['parseq_delta_sum']} | `{row['selected_by_transition']}` |"
        )

    lines.extend(
        [
            "",
            "## Policy Baselines",
            "",
            "| policy | changed | cardinalities | policies | Tess delta | PARSeq delta | mean enh bpp | improved/worse/unchanged |",
            "|---|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for name, row in result["policy_baselines"].items():
        lines.append(
            f"| {name} | {row['changed_images']} | `{row['cardinality_counts']}` | `{row['policy_counts']}` | "
            f"{row['tesseract_delta_sum']} | {row['parseq_delta_sum']} | {row['mean_enhancement_bpp']:.4f} | "
            f"`{row['improved_worse_unchanged']}` |"
        )

    lines.extend(
        [
            "",
            "## Feature Models",
            "",
            "| mode | edge objective | selected | precision | recall | F1 | Tess delta | PARSeq delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, model in result["feature_models"].items():
        for objective, row in model["edge_policies"].items():
            lines.append(
                f"| {mode} | {objective} | {row['selected']} | {format_metric(row['precision'])} | "
                f"{format_metric(row['recall'])} | {format_metric(row['f1'])} | "
                f"{row['tesseract_delta_sum']} | {row['parseq_delta_sum']} |"
            )

    lines.extend(
        [
            "",
            "## Greedy Stop/Continue Policies",
            "",
            "| mode | changed | cardinalities | policies | Tess delta | PARSeq delta | mean enh bpp | improved/worse/unchanged |",
            "|---|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for mode, model in result["feature_models"].items():
        row = model["greedy_policy"]
        lines.append(
            f"| {mode} | {row['changed_images']} | `{row['cardinality_counts']}` | `{row['policy_counts']}` | "
            f"{row['tesseract_delta_sum']} | {row['parseq_delta_sum']} | {row['mean_enhancement_bpp']:.4f} | "
            f"`{row['improved_worse_unchanged']}` |"
        )

    best_edge = result["summary"]["best_feature_edge_max_f1"]
    best_policy = result["summary"]["best_feature_greedy_policy"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Best feature edge model by F1: `{best_edge['mode']}` / `{best_edge['objective']}` "
            f"with F1 `{best_edge['f1']:.4f}` and Tesseract delta `{best_edge['tesseract_delta_sum']}`.",
            f"Best feature greedy policy by Tesseract delta: `{best_policy['mode']}` "
            f"with delta `{best_policy['tesseract_delta_sum']}` and mean enhancement bpp `{best_policy['mean_enhancement_bpp']:.4f}`.",
            "",
            "Promotion rule: a learned multi-candidate allocator must beat both the N074 edge first-only diagnostic "
            "and the fixed one-candidate subset policy under policy-level OCR, not just row metrics.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_interaction_subset3_smoke40_feature_allocator")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--l2", type=float, default=0.002)
    args = parser.parse_args()

    rows = read_jsonl(args.edge_table)
    if not rows:
        raise SystemExit("empty edge table")
    if args.target not in rows[0]:
        raise SystemExit(f"missing target field: {args.target}")

    categories = make_categories(rows)
    edge_baselines = {
        "none": edge_metrics(rows, np.zeros(len(rows), dtype=bool), args.target, args.profile),
        "all": edge_metrics(rows, np.ones(len(rows), dtype=bool), args.target, args.profile),
        "first_only": edge_metrics(
            rows, np.asarray([row["transition"] == "0_to_1" for row in rows], dtype=bool), args.target, args.profile
        ),
        "first_or_third": edge_metrics(
            rows,
            np.asarray([row["transition"] in {"0_to_1", "2_to_3"} for row in rows], dtype=bool),
            args.target,
            args.profile,
        ),
    }

    indices = all_indices(rows)
    policy_baselines = {
        "base": policy_metrics(rows, fixed_policy_members(rows, tuple()), args.profile),
        "single_a": policy_metrics(rows, fixed_policy_members(rows, ("a",)), args.profile),
        "single_b": policy_metrics(rows, fixed_policy_members(rows, ("b",)), args.profile),
        "single_c": policy_metrics(rows, fixed_policy_members(rows, ("c",)), args.profile),
        "subset_oracle": policy_metrics(rows, subset_oracle_members(rows, indices, args.profile), args.profile),
    }
    best_single_cv: dict[int, tuple[str, ...]] = {}
    for fold in range(args.folds):
        train_indices = {index for index in indices if index % args.folds != fold}
        eval_indices = {index for index in indices if index % args.folds == fold}
        members = choose_best_fixed_single(rows, train_indices, args.profile)
        for index in eval_indices:
            best_single_cv[index] = members
    policy_baselines["best_single_train_cv"] = policy_metrics(rows, best_single_cv, args.profile)

    feature_models = {}
    feature_names = {}
    for mode in [
        "transition",
        "transition_candidate_rate_psnr",
        "transition_candidate_rate_psnr_source",
        "transition_candidate_rate_psnr_ocr_state_upper",
    ]:
        x, names = build_features(rows, mode, categories)
        feature_names[mode] = names
        feature_models[mode] = crossfit_model(
            rows,
            x,
            target=args.target,
            profile=args.profile,
            folds=args.folds,
            mode=mode,
            l2=args.l2,
            steps=args.steps,
            lr=args.lr,
        )

    edge_candidates = []
    for mode, model in feature_models.items():
        for objective, metric in model["edge_policies"].items():
            edge_candidates.append(
                (
                    (
                        -1.0 if metric["f1"] is None else metric["f1"],
                        -metric["fp"],
                        -metric["tesseract_delta_sum"],
                    ),
                    mode,
                    objective,
                    metric,
                )
            )
    best_edge = max(edge_candidates, key=lambda item: item[0])
    policy_candidates = [
        ((-model["greedy_policy"]["tesseract_delta_sum"], -model["greedy_policy"]["changed_images"]), mode, model)
        for mode, model in feature_models.items()
    ]
    best_policy = max(policy_candidates, key=lambda item: item[0])

    result = {
        "experiment_id": args.experiment_id,
        "edge_table": str(args.edge_table),
        "target": args.target,
        "profile": args.profile,
        "folds": args.folds,
        "validity": "valid_diagnostic",
        "feature_note": {
            "transition_candidate_rate_psnr_ocr_state_upper": "Non-deployable upper control: uses OCR source distances.",
            "other_modes": "No held-out OCR source distances; still diagnostic because labels/thresholds are OCR-derived.",
        },
        "feature_names": feature_names,
        "edge_baselines": edge_baselines,
        "policy_baselines": policy_baselines,
        "feature_models": feature_models,
        "summary": {
            "best_feature_edge_max_f1": {
                "mode": best_edge[1],
                "objective": best_edge[2],
                **best_edge[3],
            },
            "best_feature_greedy_policy": {
                "mode": best_policy[1],
                **best_policy[2]["greedy_policy"],
            },
            "beats_n074_first_only_edge_f1": bool(
                (best_edge[3]["f1"] or -1.0) > (edge_baselines["first_only"]["f1"] or -1.0)
            ),
            "beats_n074_first_only_edge_tesseract_delta": bool(
                best_edge[3]["tesseract_delta_sum"] < edge_baselines["first_only"]["tesseract_delta_sum"]
            ),
            "beats_best_fixed_single_policy_tesseract_delta": bool(
                best_policy[2]["greedy_policy"]["tesseract_delta_sum"]
                < min(
                    policy_baselines["single_a"]["tesseract_delta_sum"],
                    policy_baselines["single_b"]["tesseract_delta_sum"],
                    policy_baselines["single_c"]["tesseract_delta_sum"],
                    policy_baselines["best_single_train_cv"]["tesseract_delta_sum"],
                )
            ),
        },
    }
    metrics = {}
    scalar_metrics("edge_baselines", edge_baselines, metrics)
    scalar_metrics("policy_baselines", policy_baselines, metrics)
    for mode, model in feature_models.items():
        scalar_metrics(f"feature_models_{mode}_edge", model["edge_policies"], metrics)
        scalar_metrics(f"feature_models_{mode}_greedy", model["greedy_policy"], metrics)
        scalar_metrics(f"feature_models_{mode}_score_summary", model["score_summary"], metrics)
    scalar_metrics("summary", result["summary"], metrics)
    result["aggregate"] = {"scalar_metrics": {key: {"value": value} for key, value in metrics.items()}}

    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
