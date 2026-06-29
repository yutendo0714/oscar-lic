#!/usr/bin/env python3
"""Train a scrubbed N117 hard-cohort verifier.

This diagnostic uses the N127 train-only hard cohorts as supervision, then
tunes a change/no-op threshold on source-modulo out-of-fold train predictions.
Validation rows are scored only by a full train-cohort model. No OCR strings,
predictions, references, held-out deltas or oracle labels are used as features.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402
import train_top8_oof_shortlist_policy_selector as n118  # noqa: E402


ENGINEERED_FEATURES = {
    "candidate_slot",
    "in_topk4",
    "in_oof_top4",
    "oof_score",
    "oof_rank",
    "oof_margin_to_best",
    "group_oof_best",
    "group_oof_second",
    "group_oof_gap",
    "topk_rank",
    "assignment_relative_error",
    "log_assignment_relative_error",
    "rank_inverse",
    "oof_rank_inverse",
}

BLOCKED_EXACT = {
    "seed",
    "real_seed",
    "original_seed",
    "source",
    "source_index",
    "source_image",
    "image_path",
    "split",
    "assignment_partition",
    "group_key",
    "candidate_index",
    "reference",
    "reference_length",
}

BLOCKED_TOKENS = (
    "tesseract",
    "parseq",
    "prediction",
    "confidence",
    "exact",
    "label",
    "oracle",
    "reference",
    "teacher",
    "ocr",
    "transcription",
    "distance",
    "utility",
)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_source_mod(key: tuple[int, int, int, int], modulo: int) -> int:
    return int(key[2]) % modulo


def cohort_candidate_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    group = row["group_key"]
    if not isinstance(group, list) or len(group) != 4:
        raise ValueError(f"bad group_key in cohort row: {group!r}")
    return (int(group[0]), int(group[1]), int(group[2]), int(group[3]), int(row["code_index"]))


def allowed_feature(name: str) -> bool:
    lower = name.lower()
    if name in ENGINEERED_FEATURES:
        return True
    if name in BLOCKED_EXACT:
        return False
    if name.startswith("codebook_") or name.startswith("img_"):
        return not any(token in lower for token in BLOCKED_TOKENS)
    return False


def all_numeric_columns(rows: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for row in rows[: min(len(rows), 512)]:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                names.add(key)
    return names


def feature_names(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    numeric = all_numeric_columns(rows)
    selected = sorted(name for name in numeric if allowed_feature(name))
    rejected = sorted(name for name in numeric if not allowed_feature(name))
    suspicious_selected = [
        name
        for name in selected
        if name not in ENGINEERED_FEATURES
        and not name.startswith(("codebook_", "img_"))
    ]
    leak_like_selected = [
        name
        for name in selected
        if any(token in name.lower() for token in ("tesseract", "parseq", "label", "oracle", "reference", "prediction"))
    ]
    return selected, {
        "selected_feature_count": len(selected),
        "rejected_numeric_feature_count": len(rejected),
        "selected_features": selected,
        "rejected_numeric_features_sample": rejected[:80],
        "suspicious_selected": suspicious_selected,
        "leak_like_selected": leak_like_selected,
    }


def feature_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    x = np.zeros((len(rows), len(names)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, name in enumerate(names):
            value = finite_float(row.get(name))
            x[row_idx, col_idx] = 0.0 if value is None else value
    x[~np.isfinite(x)] = 0.0
    return x


def candidate_rows_for_group(
    rows: list[dict[str, Any]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> list[dict[str, Any]]:
    out = []
    for row in n118.candidate_rows(rows, scores, topk_cap=topk_cap, score_count=score_count):
        if int(row.get("is_nearest", 0)):
            continue
        row = dict(row)
        rel_error = finite_float(row.get("assignment_relative_error")) or 0.0
        topk_rank = int(row.get("topk_rank", 999))
        oof_rank = int(row.get("oof_rank", 999))
        row["log_assignment_relative_error"] = float(np.log1p(max(0.0, rel_error)))
        row["rank_inverse"] = float(1.0 / max(1, topk_rank))
        row["oof_rank_inverse"] = float(1.0 / max(1, oof_rank if oof_rank < 999 else 999))
        out.append(row)
    return out


def build_candidate_dataset(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, int, int, int, int], int],
    dict[tuple[int, int, int, int], list[int]],
]:
    rows_all: list[dict[str, Any]] = []
    by_candidate: dict[tuple[int, int, int, int, int], int] = {}
    by_group: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for key, group_rows in sorted(groups.items()):
        for row in candidate_rows_for_group(group_rows, scores, topk_cap=topk_cap, score_count=score_count):
            cand_key = bottleneck.candidate_key(row)
            by_candidate[cand_key] = len(rows_all)
            by_group[key].append(len(rows_all))
            rows_all.append(row)
    return rows_all, by_candidate, dict(by_group)


def load_cohort_labels(
    cohort_path: Path,
    row_index: dict[tuple[int, int, int, int, int], int],
    rows_all: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cohort_rows = read_jsonl(cohort_path)
    indices: list[int] = []
    labels: list[int] = []
    missing = 0
    partitions = Counter()
    cohort_types = Counter()
    for row in cohort_rows:
        partitions[str(row.get("assignment_partition", "unknown"))] += 1
        cohort_types[str(row.get("cohort_type", "unknown"))] += 1
        key = cohort_candidate_key(row)
        idx = row_index.get(key)
        if idx is None:
            missing += 1
            continue
        if str(row.get("assignment_partition", "")) != "train":
            raise ValueError(f"N127 cohort contains non-train row: {row}")
        if str(rows_all[idx].get("assignment_partition", rows_all[idx].get("split"))) != "train":
            raise ValueError(f"cohort key resolved outside train partition: {key}")
        indices.append(idx)
        labels.append(int(row["target_label"]))
    labels_arr = np.asarray(labels, dtype=np.int32)
    return (
        np.asarray(indices, dtype=np.int64),
        labels_arr,
        {
            "cohort_rows": len(cohort_rows),
            "matched_rows": len(indices),
            "missing_rows": missing,
            "positive_rows": int(labels_arr.sum()),
            "negative_rows": int(len(labels_arr) - labels_arr.sum()),
            "partitions": {key: int(value) for key, value in sorted(partitions.items())},
            "cohort_types": {key: int(value) for key, value in sorted(cohort_types.items())},
        },
    )


def classifier_grid(random_state: int) -> dict[str, Any]:
    return {
        "logistic_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000, random_state=random_state),
        ),
        "hist_gradient_l2": HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=180,
            max_leaf_nodes=11,
            l2_regularization=0.5,
            random_state=random_state,
        ),
        "random_forest_d5": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=4,
        ),
    }


def score_model(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1].astype(np.float64)
        return prob.reshape(-1).astype(np.float64)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(x), dtype=np.float64)


def fit_or_constant(model: Any, x: np.ndarray, y: np.ndarray) -> tuple[Any | None, float | None]:
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return None, float(positives / max(1, len(y)))
    fitted = clone(model)
    fitted.fit(x, y)
    return fitted, None


def score_or_constant(model: Any | None, constant: float | None, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full((x.shape[0],), 0.0 if constant is None else float(constant), dtype=np.float64)
    return score_model(model, x)


def partition_indices(
    rows_all: list[dict[str, Any]],
    partition: str,
) -> np.ndarray:
    return np.asarray(
        [
            idx
            for idx, row in enumerate(rows_all)
            if str(row.get("assignment_partition", row.get("split", "unknown"))) == partition
        ],
        dtype=np.int64,
    )


def source_mod_indices(
    rows_all: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    modulo: int,
    remainder: int,
) -> np.ndarray:
    selected = []
    for idx in indices:
        key = bottleneck.group_key(rows_all[int(idx)])
        if group_source_mod(key, modulo) == remainder:
            selected.append(int(idx))
    return np.asarray(selected, dtype=np.int64)


def source_oof_scores(
    base_model: Any,
    x: np.ndarray,
    rows_all: list[dict[str, Any]],
    cohort_indices: np.ndarray,
    cohort_labels: np.ndarray,
    target_indices: np.ndarray,
    *,
    modulo: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.full((len(rows_all),), np.nan, dtype=np.float64)
    fold_meta = []
    for remainder in range(modulo):
        train_mask = [
            group_source_mod(bottleneck.group_key(rows_all[int(idx)]), modulo) != remainder
            for idx in cohort_indices
        ]
        fold_train_indices = cohort_indices[np.asarray(train_mask, dtype=bool)]
        fold_labels = cohort_labels[np.asarray(train_mask, dtype=bool)]
        fold_target = source_mod_indices(rows_all, target_indices, modulo=modulo, remainder=remainder)
        model, constant = fit_or_constant(base_model, x[fold_train_indices], fold_labels)
        if len(fold_target):
            scores[fold_target] = score_or_constant(model, constant, x[fold_target])
        fold_meta.append(
            {
                "remainder": remainder,
                "train_rows": int(len(fold_train_indices)),
                "train_positives": int(fold_labels.sum()),
                "target_rows": int(len(fold_target)),
                "constant_score": constant,
            }
        )
    return scores, {"folds": fold_meta}


def best_candidate_for_group(
    key: tuple[int, int, int, int],
    group_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
) -> tuple[int | None, float]:
    best_idx = None
    best_score = float("-inf")
    for idx in group_indices.get(key, []):
        score = float(scores[idx])
        if not np.isfinite(score):
            continue
        row = rows_all[idx]
        order = (score, -int(row.get("topk_rank", 999)), -int(row["code_index"]))
        if best_idx is None:
            best_idx = idx
            best_score = score
            best_order = order
            continue
        if order > best_order:
            best_idx = idx
            best_score = score
            best_order = order
    return best_idx, best_score


def select_by_threshold(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, group_rows in groups.items():
        nearest_code = int(bottleneck.nearest_row(group_rows)["code_index"])
        best_idx, best_score = best_candidate_for_group(key, group_indices, rows_all, scores)
        if best_idx is None or best_score < threshold:
            selected[key] = nearest_code
        else:
            selected[key] = int(rows_all[best_idx]["code_index"])
    return selected


def tune_threshold(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    false_wrong_budget: int,
) -> dict[str, Any]:
    finite_scores = sorted({float(score) for score in scores if np.isfinite(score)})
    thresholds = [float("inf")]
    thresholds.extend(finite_scores)
    if finite_scores:
        thresholds.append(float(min(finite_scores) - 1.0e-6))
    best: tuple[Any, ...] | None = None
    for threshold in thresholds:
        metrics = bottleneck.evaluate_selection(
            groups,
            select_by_threshold(groups, group_indices, rows_all, scores, threshold),
        )
        bad = int(metrics.get("false_change", 0)) + int(metrics.get("wrong_change", 0))
        if bad > false_wrong_budget or int(metrics.get("parseq_delta_vs_nearest", 0)) > 0:
            continue
        candidate = (
            int(metrics.get("tesseract_delta_vs_nearest", 0)),
            -int(metrics.get("exact_changed_groups", 0)),
            bad,
            int(metrics.get("changed_groups", 0)),
            -float(threshold if np.isfinite(threshold) else 1.0e9),
            float(threshold),
            metrics,
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        threshold = float("inf")
        metrics = bottleneck.evaluate_selection(
            groups,
            select_by_threshold(groups, group_indices, rows_all, scores, threshold),
        )
    else:
        threshold = float(best[5])
        metrics = best[6]
    return {"threshold": threshold, "metrics": metrics}


def proposal_rank_counts(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, int]:
    counts = Counter()
    for key, group_rows in groups.items():
        nearest = bottleneck.nearest_row(group_rows)
        oracle = bottleneck.oracle_row(group_rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        counts["oracle_change_groups"] += 1
        ranked = [
            idx
            for idx in group_indices.get(key, [])
            if np.isfinite(scores[idx])
        ]
        ranked.sort(
            key=lambda idx: (
                float(scores[idx]),
                -int(rows_all[idx].get("topk_rank", 999)),
                -int(rows_all[idx]["code_index"]),
            ),
            reverse=True,
        )
        rank = None
        for pos, idx in enumerate(ranked, start=1):
            if int(rows_all[idx]["code_index"]) == int(oracle["code_index"]):
                rank = pos
                break
        if rank is None:
            counts["oracle_missing"] += 1
            continue
        for cap in [1, 2, 4, 8]:
            counts[f"rank_le{cap}"] += int(rank <= cap)
    return {key: int(value) for key, value in sorted(counts.items())}


def compact(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics.get('exact_changed_groups', 0)}/{metrics.get('oracle_change_groups', 0)} exact, "
        f"false {metrics.get('false_change', 0)}, wrong {metrics.get('wrong_change', 0)}, "
        f"miss {metrics.get('missed_oracle', 0)}, "
        f"T {metrics.get('tesseract_delta_vs_nearest', 0)}, P {metrics.get('parseq_delta_vs_nearest', 0)}"
    )


def best_policy_summary(models: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for model_name, model_result in models.items():
        for budget_name, policy in model_result["policies"].items():
            metrics = policy["val_metrics"]
            rows.append(
                {
                    "model": model_name,
                    "budget": budget_name,
                    "threshold": policy["threshold"],
                    "exact_changed_groups": int(metrics.get("exact_changed_groups", 0)),
                    "false_wrong": int(metrics.get("false_change", 0)) + int(metrics.get("wrong_change", 0)),
                    "false_change": int(metrics.get("false_change", 0)),
                    "wrong_change": int(metrics.get("wrong_change", 0)),
                    "tesseract_delta_vs_nearest": int(metrics.get("tesseract_delta_vs_nearest", 0)),
                    "parseq_delta_vs_nearest": int(metrics.get("parseq_delta_vs_nearest", 0)),
                    "metrics": metrics,
                }
            )
    rows.sort(
        key=lambda row: (
            row["tesseract_delta_vs_nearest"],
            -row["exact_changed_groups"],
            row["false_wrong"],
            row["parseq_delta_vs_nearest"],
        )
    )
    return rows[0] if rows else {}


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "Train-only N127 hard-cohort verifier over the N117 OOF shortlist. This is a table-level diagnostic; no counted `.oscr` stream is exported.",
        "",
        "## Feature Scrub",
        "",
        f"- Selected features: `{result['feature_scrub']['selected_feature_count']}`",
        f"- Leak-like selected features: `{result['feature_scrub']['leak_like_selected']}`",
        f"- Suspicious selected features: `{result['feature_scrub']['suspicious_selected']}`",
        "",
        "## Validation Policies",
        "",
        "| model | rank1/rank<=4 | budget | train OOF | validation |",
        "|---|---:|---|---|---|",
    ]
    for model_name, model_result in result["models"].items():
        rank = model_result["val_rank_counts"]
        for budget_name, policy in model_result["policies"].items():
            lines.append(
                f"| {model_name} | {rank.get('rank_le1', 0)}/{rank.get('rank_le4', 0)} | {budget_name} | "
                f"{compact(policy['train_oof_metrics'])} | {compact(policy['val_metrics'])} |"
            )
    best = result.get("best_validation_policy", {})
    lines.extend(
        [
            "",
            "## Best Observed Validation Policy",
            "",
            json.dumps(
                {
                    key: best.get(key)
                    for key in [
                        "model",
                        "budget",
                        "exact_changed_groups",
                        "false_wrong",
                        "tesseract_delta_vs_nearest",
                        "parseq_delta_vs_nearest",
                    ]
                },
                ensure_ascii=False,
            ),
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path, cohort: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-hard-cohort-verifier",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
        "  table:",
        f"    path: {result['inputs']['table']['path']}",
        f"    sha256: {result['inputs']['table']['sha256']}",
        "  hard_cohort:",
        f"    path: {cohort}",
        f"    sha256: {bottleneck.sha256_file(cohort)}",
    ]
    for item in result["inputs"]["oof_scores"]["score_files"]:
        lines.extend(
            [
                f"  score_r{item['remainder']}:",
                f"    path: {item['path']}",
                f"    sha256: {item['sha256']}",
            ]
        )
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
            "    path: scripts/train_top8_hard_cohort_verifier.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  cohort_builder:",
            "    path: scripts/select_top8_verifier_hard_cohorts.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/select_top8_verifier_hard_cohorts.py')}",
            "  shortlist_source:",
            "    path: scripts/train_top8_oof_shortlist_policy_selector.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_policy_selector.py')}",
            "conclusion: table-level verifier diagnostic only; no counted oscr export",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--hard-cohort", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, action="append", required=True)
    parser.add_argument("--source-modulo", type=int, default=5)
    parser.add_argument("--topk-cap", type=int, default=4)
    parser.add_argument("--score-count", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=20260626)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups, by_candidate = bottleneck.load_table(args.table)
    partitions = bottleneck.split_groups(groups)
    train_groups = partitions["train"]
    val_groups = partitions["val"]
    oof_scores, oof_meta = oof_shortlist.build_oof_scores(args.score_file, by_candidate, modulo=args.source_modulo)
    rows_all, row_index, group_indices = build_candidate_dataset(
        groups,
        oof_scores,
        topk_cap=args.topk_cap,
        score_count=args.score_count,
    )
    names, scrub = feature_names(rows_all)
    if scrub["leak_like_selected"] or scrub["suspicious_selected"]:
        raise RuntimeError(f"feature scrub failed: {scrub}")
    x = feature_matrix(rows_all, names)
    cohort_indices, cohort_labels, cohort_meta = load_cohort_labels(args.hard_cohort, row_index, rows_all)
    train_candidate_indices = partition_indices(rows_all, "train")
    val_candidate_indices = partition_indices(rows_all, "val")
    models: dict[str, Any] = {}
    for model_name, base_model in classifier_grid(args.random_state).items():
        full_model, constant = fit_or_constant(base_model, x[cohort_indices], cohort_labels)
        full_scores = np.full((len(rows_all),), np.nan, dtype=np.float64)
        full_scores[val_candidate_indices] = score_or_constant(full_model, constant, x[val_candidate_indices])
        train_oof_scores, oof_train_meta = source_oof_scores(
            base_model,
            x,
            rows_all,
            cohort_indices,
            cohort_labels,
            train_candidate_indices,
            modulo=args.source_modulo,
        )
        policies: dict[str, Any] = {}
        for budget in [0, 1, 3]:
            tuned = tune_threshold(
                train_groups,
                group_indices,
                rows_all,
                train_oof_scores,
                false_wrong_budget=budget,
            )
            threshold = float(tuned["threshold"])
            val_metrics = bottleneck.evaluate_selection(
                val_groups,
                select_by_threshold(val_groups, group_indices, rows_all, full_scores, threshold),
            )
            policies[f"train_oof_budget{budget}"] = {
                "threshold": threshold,
                "train_oof_metrics": tuned["metrics"],
                "val_metrics": val_metrics,
            }
        models[model_name] = {
            "full_constant_score": constant,
            "train_oof": oof_train_meta,
            "val_rank_counts": proposal_rank_counts(val_groups, group_indices, rows_all, full_scores),
            "val_all_change_metrics": bottleneck.evaluate_selection(
                val_groups,
                select_by_threshold(val_groups, group_indices, rows_all, full_scores, float("-inf")),
            ),
            "policies": policies,
        }
    result = {
        "experiment_id": args.experiment_id,
        "validity": "table_level_diagnostic_not_promoted",
        "code_commit": bottleneck.git_commit(),
        "inputs": {
            "table": {"path": str(args.table), "sha256": bottleneck.sha256_file(args.table)},
            "hard_cohort": {"path": str(args.hard_cohort), "sha256": bottleneck.sha256_file(args.hard_cohort)},
            "oof_scores": oof_meta,
        },
        "config": {
            "source_modulo": args.source_modulo,
            "topk_cap": args.topk_cap,
            "score_count": args.score_count,
            "random_state": args.random_state,
        },
        "data_summary": {
            "groups": len(groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "candidate_rows": len(rows_all),
            "train_candidate_rows": int(len(train_candidate_indices)),
            "val_candidate_rows": int(len(val_candidate_indices)),
            "cohort": cohort_meta,
        },
        "feature_scrub": scrub,
        "models": models,
        "best_validation_policy": best_policy_summary(models),
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "cohort_builder": bottleneck.sha256_file(ROOT / "scripts/select_top8_verifier_hard_cohorts.py"),
        },
        "wandb_run_id": "not_logged",
        "interpretation": (
            "N128 tests whether the N127 hard cohorts provide a cleaner verifier objective under strict feature scrubbing "
            "and train-only/source-OOF thresholding. A positive result would still need counted `.oscr` export; a negative "
            "result means hard-cohort supervision alone does not solve N126's no-headroom boundary."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report, args.hard_cohort)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "best_validation_policy": {
                    key: result["best_validation_policy"].get(key)
                    for key in [
                        "model",
                        "budget",
                        "exact_changed_groups",
                        "false_wrong",
                        "tesseract_delta_vs_nearest",
                        "parseq_delta_vs_nearest",
                    ]
                },
                "feature_count": scrub["selected_feature_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
