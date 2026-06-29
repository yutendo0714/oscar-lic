#!/usr/bin/env python3
"""Audit top-k candidate scoring before training another selector.

This diagnostic separates three failure modes:

1. the oracle candidate is not in the deployable shortlist,
2. a score argmax prefers a false/harmful non-nearest candidate, and
3. a score argmax finds the correct candidate but a margin would suppress it.

It uses OCR deltas only as labels for analysis, not as deployable features.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_assignment_topk_policy_selector as base  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_oracle_idx(rows: list[dict], indices: list[int]) -> int:
    oracle = [idx for idx in indices if int(rows[idx]["label_assignment_oracle_choice"])]
    if oracle:
        return oracle[0]
    return next(idx for idx in indices if int(rows[idx]["is_nearest"]))


def nearest_idx(rows: list[dict], indices: list[int]) -> int:
    return next(idx for idx in indices if int(rows[idx]["is_nearest"]))


def score_ranks(indices: list[int], scores: np.ndarray) -> dict[int, int]:
    ordered = sorted(indices, key=lambda idx: (-float(scores[idx]), int(idx)))
    return {idx: rank + 1 for rank, idx in enumerate(ordered)}


def rank_hist(values: list[int | None]) -> dict[str, int]:
    hist: Counter[str] = Counter()
    for value in values:
        hist["missing" if value is None else str(int(value))] += 1
    return dict(sorted(hist.items(), key=lambda item: (item[0] == "missing", item[0])))


def select_by_scores(
    rows: list[dict],
    candidates_by_group: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
    *,
    margin: float | None = None,
) -> list[int]:
    selected: list[int] = []
    for candidates in candidates_by_group.values():
        nearest = nearest_idx(rows, candidates)
        best = max(candidates, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
        if margin is not None and best != nearest and float(scores[best]) - float(scores[nearest]) < margin:
            best = nearest
        selected.append(best)
    return selected


def select_best_nonnearest_by_margin(
    rows: list[dict],
    candidates_by_group: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
    *,
    threshold: float,
) -> list[int]:
    selected: list[int] = []
    for candidates in candidates_by_group.values():
        nearest = nearest_idx(rows, candidates)
        nonnearest = [idx for idx in candidates if idx != nearest]
        if not nonnearest:
            selected.append(nearest)
            continue
        best_nonnearest = max(nonnearest, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
        margin = float(scores[best_nonnearest]) - float(scores[nearest])
        selected.append(best_nonnearest if margin >= threshold else nearest)
    return selected


def tune_score_margin(
    rows: list[dict],
    train_candidates: dict[tuple[int, int, int, int], list[int]],
    full_train_groups: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
) -> float:
    margins: list[float] = []
    for candidates in train_candidates.values():
        nearest = nearest_idx(rows, candidates)
        best = max(candidates, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
        margins.append(float(scores[best]) - float(scores[nearest]))
    if not margins:
        return 1e6
    candidates = sorted(set(margins))
    candidates = [max(candidates) + 1e-6] + candidates + [min(candidates) - 1e-6]
    best_margin = candidates[0]
    best_key: tuple[int, int, int, int, int, int] | None = None
    for margin in candidates:
        selected = select_by_scores(rows, train_candidates, scores, margin=margin)
        metrics = base.policy_metrics_by_group(rows, full_train_groups, selected)
        key = (
            int(metrics["parseq_worsen_groups"]),
            int(metrics["tesseract_worsen_groups"]),
            int(metrics["wrong_changes"]),
            int(metrics["tesseract_delta_sum"]),
            int(metrics["false_changes"]),
            int(metrics["changed_groups"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_margin = float(margin)
    return best_margin


def tune_nonnearest_promoter_threshold(
    rows: list[dict],
    train_candidates: dict[tuple[int, int, int, int], list[int]],
    full_train_groups: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
) -> float:
    margins: list[float] = []
    for candidates in train_candidates.values():
        nearest = nearest_idx(rows, candidates)
        nonnearest = [idx for idx in candidates if idx != nearest]
        if not nonnearest:
            continue
        best_nonnearest = max(nonnearest, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
        margins.append(float(scores[best_nonnearest]) - float(scores[nearest]))
    if not margins:
        return 1e6
    candidates = sorted(set(margins))
    candidates = [max(candidates) + 1e-6] + candidates + [min(candidates) - 1e-6]
    best_threshold = candidates[0]
    best_key: tuple[int, int, int, int, int, int] | None = None
    for threshold in candidates:
        selected = select_best_nonnearest_by_margin(rows, train_candidates, scores, threshold=threshold)
        metrics = base.policy_metrics_by_group(rows, full_train_groups, selected)
        key = (
            int(metrics["parseq_worsen_groups"]),
            int(metrics["tesseract_worsen_groups"]),
            int(metrics["wrong_changes"]),
            int(metrics["tesseract_delta_sum"]),
            int(metrics["false_changes"]),
            -int(metrics["exact_oracle_changes"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def selection_type(rows: list[dict], full_group: list[int], selected: int) -> str:
    nearest = nearest_idx(rows, full_group)
    oracle = full_oracle_idx(rows, full_group)
    if selected == nearest:
        return "nearest_missed" if oracle != nearest else "nearest_correct"
    if selected == oracle:
        return "exact_oracle"
    if oracle != nearest:
        return "wrong_change"
    if int(rows[selected]["parseq_delta_vs_nearest"]) > 0 or int(rows[selected]["tesseract_delta_vs_nearest"]) > 0:
        return "false_harmful_change"
    if int(rows[selected]["parseq_delta_vs_nearest"]) < 0 or int(rows[selected]["tesseract_delta_vs_nearest"]) < 0:
        return "false_safe_improve_label_gap"
    return "false_neutral_change"


def summarize_selection_types(
    rows: list[dict],
    full_groups: dict[tuple[int, int, int, int], list[int]],
    selected: list[int],
) -> dict[str, int]:
    selected_by_group = {base.group_key(rows[idx]): idx for idx in selected}
    counts: Counter[str] = Counter()
    for key, indices in full_groups.items():
        counts[selection_type(rows, indices, selected_by_group[key])] += 1
    return dict(sorted(counts.items()))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float64).mean())


def policy_summary(
    rows: list[dict],
    full_val_groups: dict[tuple[int, int, int, int], list[int]],
    selected: list[int],
) -> dict[str, Any]:
    metrics = base.policy_metrics_by_group(rows, full_val_groups, selected)
    metrics["selection_types"] = summarize_selection_types(rows, full_val_groups, selected)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--score-model-seed", type=int, default=2)
    parser.add_argument("--shortlist-topk", type=int, default=4)
    args = parser.parse_args()

    rows = base.read_jsonl(args.table)
    groups = base.grouped_indices(rows)
    oracle_by_model = base.load_score_features(args.oracle_score)
    safe_by_model = base.load_score_features(args.safe_score)
    oracle_scores = np.asarray(
        [oracle_by_model[args.score_model_seed][idx] for idx in range(len(rows))],
        dtype=np.float32,
    )
    safe_scores = np.asarray(
        [safe_by_model[args.score_model_seed][idx] for idx in range(len(rows))],
        dtype=np.float32,
    )
    fused_scores = oracle_scores + safe_scores
    conservative_scores = np.minimum(oracle_scores, safe_scores)

    candidates = {
        key: base.candidate_indices_for_group(rows, indices, oracle_scores, safe_scores, topk=args.shortlist_topk)
        for key, indices in groups.items()
    }
    full_train_groups = {key: indices for key, indices in groups.items() if int(rows[indices[0]]["seed"]) != args.val_seed}
    full_val_groups = {key: indices for key, indices in groups.items() if int(rows[indices[0]]["seed"]) == args.val_seed}
    train_candidates = {key: candidates[key] for key in full_train_groups}
    val_candidates = {key: candidates[key] for key in full_val_groups}

    coverage = {
        "train_groups": len(full_train_groups),
        "val_groups": len(full_val_groups),
        "train_full_oracle_changes": 0,
        "val_full_oracle_changes": 0,
        "train_shortlist_oracle_changes": 0,
        "val_shortlist_oracle_changes": 0,
        "train_oracle_missing_from_shortlist": 0,
        "val_oracle_missing_from_shortlist": 0,
        "candidate_count_mean_train": mean_or_none([float(len(v)) for v in train_candidates.values()]),
        "candidate_count_mean_val": mean_or_none([float(len(v)) for v in val_candidates.values()]),
    }

    rank_stats: dict[str, list[int | None]] = defaultdict(list)
    pressure: dict[str, Counter[str]] = {
        "oracle_score_argmax": Counter(),
        "safe_score_argmax": Counter(),
        "fused_score_argmax": Counter(),
        "conservative_score_argmax": Counter(),
    }
    details: list[dict[str, Any]] = []

    for split_name, full_split_groups in [("train", full_train_groups), ("val", full_val_groups)]:
        for key, indices in full_split_groups.items():
            nearest = nearest_idx(rows, indices)
            oracle = full_oracle_idx(rows, indices)
            cand = candidates[key]
            shortlist_oracle = full_oracle_idx(rows, cand)
            oracle_changes = oracle != nearest
            shortlist_changes = shortlist_oracle != nearest
            coverage[f"{split_name}_full_oracle_changes"] += int(oracle_changes)
            coverage[f"{split_name}_shortlist_oracle_changes"] += int(shortlist_changes)
            coverage[f"{split_name}_oracle_missing_from_shortlist"] += int(oracle_changes and oracle not in cand)
            if split_name == "val":
                oracle_rank_all = score_ranks(indices, oracle_scores).get(oracle) if oracle_changes else None
                safe_rank_all = score_ranks(indices, safe_scores).get(oracle) if oracle_changes else None
                oracle_rank_short = score_ranks(cand, oracle_scores).get(oracle) if oracle in cand else None
                safe_rank_short = score_ranks(cand, safe_scores).get(oracle) if oracle in cand else None
                if oracle_changes:
                    rank_stats["oracle_score_rank_all"].append(oracle_rank_all)
                    rank_stats["safe_score_rank_all"].append(safe_rank_all)
                    rank_stats["oracle_score_rank_shortlist"].append(oracle_rank_short)
                    rank_stats["safe_score_rank_shortlist"].append(safe_rank_short)
                selected_by_score = {
                    "oracle_score_argmax": max(cand, key=lambda idx: (float(oracle_scores[idx]), -int(rows[idx]["topk_rank"]))),
                    "safe_score_argmax": max(cand, key=lambda idx: (float(safe_scores[idx]), -int(rows[idx]["topk_rank"]))),
                    "fused_score_argmax": max(cand, key=lambda idx: (float(fused_scores[idx]), -int(rows[idx]["topk_rank"]))),
                    "conservative_score_argmax": max(
                        cand, key=lambda idx: (float(conservative_scores[idx]), -int(rows[idx]["topk_rank"]))
                    ),
                }
                for name, selected in selected_by_score.items():
                    pressure[name][selection_type(rows, indices, selected)] += 1
                details.append(
                    {
                        "key": list(key),
                        "source": rows[nearest].get("source"),
                        "split": rows[nearest].get("split"),
                        "reference": rows[nearest].get("reference"),
                        "nearest_row_index": nearest,
                        "oracle_row_index": oracle,
                        "nearest_code": int(rows[nearest]["code_index"]),
                        "oracle_code": int(rows[oracle]["code_index"]),
                        "oracle_topk_rank": int(rows[oracle]["topk_rank"]),
                        "oracle_in_shortlist": int(oracle in cand),
                        "oracle_tesseract_delta": int(rows[oracle]["tesseract_delta_vs_nearest"]),
                        "oracle_parseq_delta": int(rows[oracle]["parseq_delta_vs_nearest"]),
                        "oracle_score_rank_all": oracle_rank_all,
                        "safe_score_rank_all": safe_rank_all,
                        "oracle_score_rank_shortlist": oracle_rank_short,
                        "safe_score_rank_shortlist": safe_rank_short,
                        "candidate_count": len(cand),
                        "oracle_score_margin": float(oracle_scores[oracle] - oracle_scores[nearest]),
                        "safe_score_margin": float(safe_scores[oracle] - safe_scores[nearest]),
                        "score_selection_types": {
                            name: selection_type(rows, indices, selected)
                            for name, selected in selected_by_score.items()
                        },
                    }
                )

    policies: dict[str, dict[str, Any]] = {}
    nearest_selected = [nearest_idx(rows, cand) for cand in val_candidates.values()]
    policies["nearest"] = policy_summary(rows, full_val_groups, nearest_selected)
    policies["shortlist_oracle"] = policy_summary(rows, full_val_groups, base.oracle_select(rows, val_candidates))
    policies["full_oracle"] = policy_summary(
        rows,
        full_val_groups,
        [full_oracle_idx(rows, full_val_groups[key]) for key in full_val_groups],
    )

    score_sets = {
        "oracle_score": oracle_scores,
        "safe_score": safe_scores,
        "fused_score": fused_scores,
        "conservative_score": conservative_scores,
    }
    margins: dict[str, float] = {}
    promoter_thresholds: dict[str, float] = {}
    for name, scores in score_sets.items():
        selected = select_by_scores(rows, val_candidates, scores)
        policies[f"{name}_argmax"] = policy_summary(rows, full_val_groups, selected)
        margin = tune_score_margin(rows, train_candidates, full_train_groups, scores)
        margins[name] = margin
        selected_margin = select_by_scores(rows, val_candidates, scores, margin=margin)
        policies[f"{name}_train_tuned_margin"] = policy_summary(rows, full_val_groups, selected_margin)
        threshold = tune_nonnearest_promoter_threshold(rows, train_candidates, full_train_groups, scores)
        promoter_thresholds[name] = threshold
        selected_promoter = select_best_nonnearest_by_margin(rows, val_candidates, scores, threshold=threshold)
        policies[f"{name}_nonnearest_promoter"] = policy_summary(rows, full_val_groups, selected_promoter)

    result = {
        "description": "Candidate scoring bottleneck audit for Eval600-to-Eval300 top-4 assignment selection.",
        "validity": "diagnostic_no_promotion",
        "track": "A_pure_visual_bitstream_policy_diagnostic",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "val_seed": args.val_seed,
            "score_model_seed": args.score_model_seed,
            "shortlist_topk": args.shortlist_topk,
        },
        "artifacts": {
            "table_sha256": sha256_file(args.table),
            "oracle_score_sha256": sha256_file(args.oracle_score),
            "safe_score_sha256": sha256_file(args.safe_score),
        },
        "coverage": coverage,
        "rank_histograms_val_oracle_changes": {key: rank_hist(values) for key, values in rank_stats.items()},
        "score_argmax_selection_pressure_val": {key: dict(value) for key, value in pressure.items()},
        "train_tuned_margins": margins,
        "train_tuned_nonnearest_promoter_thresholds": promoter_thresholds,
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        with args.details_output.open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
