#!/usr/bin/env python3
"""Train group-level no-op acceptors on top of the N091 tabular ranker."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_top8_trainval_tabular_threshold_policy as threshold_policy  # noqa: E402
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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_log1p(value: Any) -> float:
    return math.log1p(min(max(finite(value), 0.0), 1.0e8))


def group_rows_from_scores(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, str, int, int], list[int]],
    scores: np.ndarray,
    ranker_feature_names: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    source_values = sorted({str(rows[indices[0]].get("source", "")) for indices in groups.values()})
    source_names = [f"source_is_{source}" for source in source_values]
    feature_names = [
        "score_best",
        "score_second",
        "score_margin_1_2",
        "score_mean",
        "score_std",
        "score_range",
        "best_topk_rank",
        "best_assignment_relative_error_log1p",
        "best_candidate_is_rank0",
        "best_candidate_is_rank1",
        "best_candidate_is_rank2",
        "best_candidate_is_rank_ge4",
    ] + [f"best_{name}" for name in ranker_feature_names] + source_names
    records = []
    features = []
    for key, indices in sorted(groups.items()):
        nearest = next(idx for idx in indices if int(rows[idx]["code_index"]) == int(rows[idx]["nearest_code"]))
        oracle = next((idx for idx in indices if int(rows[idx].get("label_assignment_oracle_choice", 0))), nearest)
        nonnearest = ranker.nonnearest_indices(rows, indices)
        sorted_candidates = sorted(nonnearest, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])), reverse=True)
        best = sorted_candidates[0]
        candidate_scores = np.asarray([float(scores[idx]) for idx in sorted_candidates], dtype=np.float64)
        second = float(candidate_scores[1]) if candidate_scores.size > 1 else float(candidate_scores[0] - 1.0)
        best_score = float(candidate_scores[0])
        best_topk = int(rows[best]["topk_rank"])
        source = str(rows[nearest].get("source", ""))
        source_onehot = [1.0 if source == source_value else 0.0 for source_value in source_values]
        row_features = [
            best_score,
            second,
            best_score - second,
            float(candidate_scores.mean()),
            float(candidate_scores.std(ddof=0)),
            float(candidate_scores.max() - candidate_scores.min()),
            float(best_topk),
            safe_log1p(rows[best].get("assignment_relative_error", 0.0)),
            float(best_topk == 0),
            float(best_topk == 1),
            float(best_topk == 2),
            float(best_topk >= 4),
        ]
        row_features.extend(ranker.feature_value(rows[best], name) for name in ranker_feature_names)
        row_features.extend(source_onehot)
        nearest_code = int(rows[nearest]["code_index"])
        oracle_code = int(rows[oracle]["code_index"])
        best_code = int(rows[best]["code_index"])
        best_safe = int(rows[best]["tesseract_delta_vs_nearest"] < 0 and rows[best]["parseq_delta_vs_nearest"] <= 0)
        exact_changed = int(best_code == oracle_code and oracle_code != nearest_code)
        records.append(
            {
                "key": {"real_seed": key[0], "partition": key[1], "source_index": key[2], "candidate_index": key[3]},
                "source": source,
                "reference": rows[nearest].get("reference"),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "best_code": best_code,
                "best_score": best_score,
                "best_topk_rank": best_topk,
                "label_accept_safe_improve": best_safe,
                "label_accept_exact_changed": exact_changed,
                "best_tesseract_delta_vs_nearest": int(rows[best]["tesseract_delta_vs_nearest"]),
                "best_parseq_delta_vs_nearest": int(rows[best]["parseq_delta_vs_nearest"]),
            }
        )
        features.append(row_features)
    return records, np.asarray(features, dtype=np.float32), feature_names


def split_records(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    partitions = np.asarray([row["key"]["partition"] for row in records])
    return partitions == "train", partitions == "val"


def policy_metrics(records: list[dict[str, Any]], accept: np.ndarray, mask: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    counts = Counter()
    tess_delta = 0
    parseq_delta = 0
    for idx, row in enumerate(records):
        if not bool(mask[idx]):
            continue
        nearest_code = int(row["nearest_code"])
        oracle_code = int(row["oracle_code"])
        selected_code = int(row["best_code"]) if bool(accept[idx]) else nearest_code
        status = ranker.classify(selected_code, nearest_code, oracle_code)
        counts[status] += 1
        counts["changed_groups"] += int(selected_code != nearest_code)
        counts["oracle_change_groups"] += int(oracle_code != nearest_code)
        counts["exact_changed_groups"] += int(status == "exact" and oracle_code != nearest_code)
        if bool(accept[idx]):
            t_delta = int(row["best_tesseract_delta_vs_nearest"])
            p_delta = int(row["best_parseq_delta_vs_nearest"])
        else:
            t_delta = 0
            p_delta = 0
        counts["parseq_worsen_groups"] += int(p_delta > 0)
        counts["tesseract_worsen_groups"] += int(t_delta > 0)
        counts["tesseract_improve_groups"] += int(t_delta < 0)
        tess_delta += t_delta
        parseq_delta += p_delta
    accepted_scores = [float(scores[idx]) for idx, flag in enumerate(accept) if bool(mask[idx]) and bool(flag)]
    return {
        "groups": int(mask.sum()),
        "oracle_change_groups": int(counts["oracle_change_groups"]),
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "exact_changed_groups": int(counts["exact_changed_groups"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "parseq_worsen_groups": int(counts["parseq_worsen_groups"]),
        "tesseract_worsen_groups": int(counts["tesseract_worsen_groups"]),
        "tesseract_improve_groups": int(counts["tesseract_improve_groups"]),
        "tesseract_delta_vs_nearest": int(tess_delta),
        "parseq_delta_vs_nearest": int(parseq_delta),
        "accepted_score_min": min(accepted_scores) if accepted_scores else None,
        "accepted_score_max": max(accepted_scores) if accepted_scores else None,
    }


def threshold_grid(scores: np.ndarray, train_mask: np.ndarray) -> list[float]:
    unique = sorted(set(float(score) for score in scores[train_mask]))
    if not unique:
        return [1.0]
    return [max(unique) + 1.0e-6] + unique + [min(unique) - 1.0e-6]


def choose_thresholds(records: list[dict[str, Any]], train_mask: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    rows = []
    for threshold in threshold_grid(scores, train_mask):
        accept = scores >= threshold
        metrics = policy_metrics(records, accept, train_mask, scores)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    policies = {}
    zero_safe = [
        row
        for row in rows
        if row["false_change"] + row["wrong_change"] == 0
        and row["parseq_worsen_groups"] == 0
        and row["tesseract_worsen_groups"] == 0
    ]
    if zero_safe:
        policies["zero_error_safe"] = min(
            zero_safe,
            key=lambda row: (row["tesseract_delta_vs_nearest"], -row["exact_changed_groups"], row["changed_groups"]),
        )
    one_error = [row for row in rows if row["false_change"] + row["wrong_change"] <= 1 and row["parseq_worsen_groups"] == 0]
    if one_error:
        policies["one_error_parseq_safe"] = min(
            one_error,
            key=lambda row: (
                row["tesseract_delta_vs_nearest"],
                row["tesseract_worsen_groups"],
                row["false_change"] + row["wrong_change"],
                -row["exact_changed_groups"],
            ),
        )
    parseq_safe = [row for row in rows if row["parseq_worsen_groups"] == 0]
    if parseq_safe:
        policies["best_train_tesseract_parseq_safe"] = min(
            parseq_safe,
            key=lambda row: (
                row["tesseract_delta_vs_nearest"],
                row["tesseract_worsen_groups"],
                row["false_change"] + row["wrong_change"],
            ),
        )
    return policies


def model_specs(seed: int) -> dict[str, Any]:
    return {
        "logistic_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=0.2, max_iter=2000, random_state=seed),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=160,
            max_leaf_nodes=6,
            learning_rate=0.035,
            l2_regularization=0.2,
            random_state=seed,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    return np.asarray(model.decision_function(x), dtype=np.float64)


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Group-level no-op acceptors over N091 tabular ranker score features.",
        "This is a policy-calibration diagnostic, not a counted `.oscr` promotion.",
        "",
        "## Results",
        "",
        "| model | policy | train changed | train exact changed | train false+wrong | train T | val changed | val exact changed | val false+wrong | val T | val P |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, model_result in result["models"].items():
        for policy_name, row in model_result["policies"].items():
            train = row["train"]
            val = row["val"]
            lines.append(
                f"| {model_name} | {policy_name} | {train['changed_groups']} | {train['exact_changed_groups']} | "
                f"{train['false_change'] + train['wrong_change']} | {train['tesseract_delta_vs_nearest']} | "
                f"{val['changed_groups']} | {val['exact_changed_groups']} | {val['false_change'] + val['wrong_change']} | "
                f"{val['tesseract_delta_vs_nearest']} | {val['parseq_delta_vs_nearest']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Promotion remains blocked unless a val policy beats the counted current `-8` Tesseract floor with low false/wrong changes.",
            "- Inputs exclude OCR deltas, oracle labels and OCR prediction text; labels are used only for train supervision and evaluation.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_trainval_group_acceptor")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--model-seeds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows, groups, scores, ranker_meta = threshold_policy.train_scores(args)
    records, features, group_feature_names = group_rows_from_scores(rows, groups, scores, ranker_meta["feature_names"])
    train_mask, val_mask = split_records(records)
    y_train = np.asarray([int(row["label_accept_safe_improve"]) for row in records], dtype=np.int64)[train_mask]
    models = {}
    for model_name, model in model_specs(args.seed).items():
        model.fit(features[train_mask], y_train)
        accept_scores = predict_scores(model, features)
        selected_thresholds = choose_thresholds(records, train_mask, accept_scores)
        policies = {}
        for policy_name, train_metrics in selected_thresholds.items():
            threshold = float(train_metrics["threshold"])
            accept = accept_scores >= threshold
            val_metrics = policy_metrics(records, accept, val_mask, accept_scores)
            policies[policy_name] = {
                "threshold": threshold,
                "train": train_metrics,
                "val": val_metrics,
            }
        models[model_name] = {
            "train_positive_groups": int(y_train.sum()),
            "train_groups": int(train_mask.sum()),
            "val_groups": int(val_mask.sum()),
            "policies": policies,
        }
    scalar_metrics = {}
    for model_name, model_result in models.items():
        for policy_name, policy in model_result["policies"].items():
            prefix = f"{model_name}_{policy_name}"
            scalar_metrics[f"{prefix}_val_tesseract_delta"] = {"value": float(policy["val"]["tesseract_delta_vs_nearest"])}
            scalar_metrics[f"{prefix}_val_false_wrong"] = {
                "value": float(policy["val"]["false_change"] + policy["val"]["wrong_change"])
            }
            scalar_metrics[f"{prefix}_val_exact_changed"] = {"value": float(policy["val"]["exact_changed_groups"])}
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_group_acceptor_not_promoted_stream",
        "inputs": {
            "table": str(args.table),
            "table_sha256": sha256_file(args.table),
        },
        "config": {
            "ranker_epochs": args.epochs,
            "ranker_hidden_dim": args.hidden_dim,
            "ranker_model_seeds": args.model_seeds,
            "ranker_seed": args.seed,
            "group_feature_count": int(features.shape[1]),
            "group_feature_names": group_feature_names,
            "acceptor_label": "best candidate has Tesseract delta < 0 and PARSeq delta <= 0 on train partition",
        },
        "ranker_training": {
            "device": ranker_meta["device"],
            "train_listwise_groups": ranker_meta["train_listwise_groups"],
        },
        "models": models,
        "aggregate": {"scalar_metrics": scalar_metrics},
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
