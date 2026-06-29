#!/usr/bin/env python3
"""Train group-level proxies for one-candidate safe headroom."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FORBIDDEN_PREFIXES = (
    "label_",
    "parseq_",
    "tesseract_",
    "multi_teacher_",
)
FORBIDDEN_NAMES = {
    "source_index",
    "candidate_index",
    "improving_evaluator_count",
    "worsening_evaluator_count",
}
PARSEQ_BASE_NAMES = [
    "parseq_base_distance",
    "parseq_base_exact",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def safe_tesseract_parseq(row: dict) -> bool:
    return int(row["tesseract_delta_distance"]) < 0 and int(row["parseq_delta_distance"]) <= 0


def candidate_feature_names(rows: list[dict], include_parseq_base: bool) -> list[str]:
    names = []
    for key in sorted(rows[0].keys()):
        if key in FORBIDDEN_NAMES:
            continue
        if any(key.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            if not (include_parseq_base and key in PARSEQ_BASE_NAMES):
                continue
        if all(is_number(row.get(key)) for row in rows):
            names.append(key)
    return names


def group_rows(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_index"])].append(row)
    return dict(sorted(grouped.items()))


def aggregate_group_features(
    rows: list[dict],
    feature_names: list[str],
    source_vocab: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict]]:
    grouped = group_rows(rows)
    output_rows = []
    features = []
    labels = []
    harms = []
    names = []
    for feature in feature_names:
        for stat in ["mean", "std", "min", "max"]:
            names.append(f"{feature}_{stat}")
    if source_vocab is not None:
        names.extend([f"source={source}" for source in source_vocab])
    for source_index, group in grouped.items():
        vector = []
        for feature in feature_names:
            values = np.asarray([float(row[feature]) for row in group], dtype=np.float64)
            vector.extend([float(values.mean()), float(values.std(ddof=0)), float(values.min()), float(values.max())])
        if source_vocab is not None:
            source = str(group[0].get("source", "unknown"))
            vector.extend([1.0 if source == item else 0.0 for item in source_vocab])
        safe_rows = [row for row in group if safe_tesseract_parseq(row)]
        harmful_rows = [row for row in group if int(row["worsening_evaluator_count"]) > 0]
        labels.append(1.0 if safe_rows else 0.0)
        harms.append(1.0 if harmful_rows else 0.0)
        output_rows.append(
            {
                "source_index": source_index,
                "source": str(group[0].get("source", "unknown")),
                "candidate_rows": len(group),
                "has_safe_tess_parseq": int(bool(safe_rows)),
                "has_harm": int(bool(harmful_rows)),
                "best_tess_parseq_safe_delta_or_zero": min(
                    (int(row["tesseract_delta_distance"]) for row in safe_rows),
                    default=0,
                ),
            }
        )
        features.append(vector)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(harms, dtype=np.float32),
        names,
        output_rows,
    )


def ap_or_none(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true, scores))


def auc_or_none(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    return float(roc_auc_score(y_true, scores))


def topk_metrics(scores: np.ndarray, labels: np.ndarray, harms: np.ndarray, ks: list[int]) -> dict:
    order = np.argsort(-scores)
    output = {}
    total_positive = max(float(labels.sum()), 1.0)
    for k in ks:
        k = min(int(k), len(labels))
        chosen = order[:k]
        positives = float(labels[chosen].sum())
        harmful = float(harms[chosen].sum())
        output[str(k)] = {
            "safe_hits": int(positives),
            "precision": positives / k if k else None,
            "recall": positives / total_positive,
            "harmful_groups": int(harmful),
            "harmful_rate": harmful / k if k else None,
        }
    return output


def predict_scores(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int) -> np.ndarray:
    if model_name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "hist_gbdt":
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(x_train, y_train.astype(int))
    return model.predict_proba(x_eval)[:, 1]


def feature_mode_config(mode: str) -> tuple[bool, bool]:
    if mode == "image_only":
        return False, False
    if mode == "image_parseq_base":
        return True, False
    if mode == "image_source":
        return False, True
    if mode == "image_parseq_base_source":
        return True, True
    raise ValueError(f"unsupported feature mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-table", type=Path, required=True)
    parser.add_argument("--eval-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--feature-mode", action="append", default=None)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--topk", type=int, nargs="+", default=[10, 20, 40, 80])
    args = parser.parse_args()
    if args.feature_mode is None:
        args.feature_mode = ["image_only", "image_parseq_base", "image_source"]
    if args.model is None:
        args.model = ["logistic", "extra_trees", "hist_gbdt"]

    train_rows = read_jsonl(args.train_table)
    eval_rows = read_jsonl(args.eval_table)
    results = []
    for mode in args.feature_mode:
        include_parseq_base, include_source = feature_mode_config(mode)
        names = candidate_feature_names(train_rows, include_parseq_base=include_parseq_base)
        source_vocab = sorted({str(row.get("source", "unknown")) for row in train_rows}) if include_source else None
        x_train, y_train, harm_train, feature_names, train_groups = aggregate_group_features(train_rows, names, source_vocab)
        x_eval, y_eval, harm_eval, _, eval_groups = aggregate_group_features(eval_rows, names, source_vocab)
        for model_name in args.model:
            scores = predict_scores(model_name, x_train, y_train, x_eval, args.seed)
            order = np.argsort(-scores)
            examples = []
            for index in order[:20]:
                row = dict(eval_groups[int(index)])
                row["score"] = float(scores[int(index)])
                examples.append(row)
            results.append(
                {
                    "feature_mode": mode,
                    "model": model_name,
                    "train_groups": int(len(y_train)),
                    "eval_groups": int(len(y_eval)),
                    "features": feature_names,
                    "feature_count": int(x_train.shape[1]),
                    "train_safe_rate": float(y_train.mean()),
                    "eval_safe_rate": float(y_eval.mean()),
                    "eval_auc": auc_or_none(y_eval, scores),
                    "eval_ap": ap_or_none(y_eval, scores),
                    "eval_topk": topk_metrics(scores, y_eval, harm_eval, args.topk),
                    "top_eval_examples": examples,
                }
            )
    output = {
        "train_table": str(args.train_table),
        "eval_table": str(args.eval_table),
        "seed": args.seed,
        "target": "group_has_tesseract_improvement_with_parseq_nonworse",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
