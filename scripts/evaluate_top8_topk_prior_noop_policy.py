#!/usr/bin/env python3
"""Train-tuned no-op policies over a source-robust top-k prior candidate.

N097 showed that the first-stage top-k/rate prior is more source-robust than
wide shallow tabular rankers. This diagnostic fixes the candidate to the top
non-nearest prior row per group, then tests whether train-only accept/abstain
calibration can produce a deployable policy.
"""

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names = []
    for name in ranker.numeric_feature_names(rows):
        if "_onehot_" in name:
            continue
        if name.startswith(("parseq_", "tesseract_", "multi_teacher_")):
            continue
        if "oracle" in name or "label" in name:
            continue
        names.append(name)
    return sorted(names)


def top_prior_candidate(rows: list[dict[str, Any]], indices: list[int]) -> int | None:
    nonnearest = ranker.nonnearest_indices(rows, indices)
    if not nonnearest:
        return None
    return min(
        nonnearest,
        key=lambda idx: (
            int(rows[idx]["topk_rank"]),
            safe_log1p(rows[idx].get("assignment_relative_error", 0.0)),
            int(rows[idx]["code_index"]),
        ),
    )


def build_group_records(rows: list[dict[str, Any]], groups: dict[tuple[int, str, int, int], list[int]]) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    names = feature_names(rows)
    extra_names = [
        "top1_rank_is_0",
        "top1_rank_is_1",
        "top1_rank_is_2",
        "top1_rank_ge_4",
        "top1_assignment_relative_error_log1p",
        "top1_code_diff_abs",
    ]
    records = []
    features = []
    for key, indices in sorted(groups.items()):
        nearest = next(idx for idx in indices if int(rows[idx]["code_index"]) == int(rows[idx]["nearest_code"]))
        oracle = next((idx for idx in indices if int(rows[idx].get("label_assignment_oracle_choice", 0))), nearest)
        best = top_prior_candidate(rows, indices)
        if best is None:
            continue
        nearest_code = int(rows[nearest]["code_index"])
        oracle_code = int(rows[oracle]["code_index"])
        best_code = int(rows[best]["code_index"])
        status_if_changed = ranker.classify(best_code, nearest_code, oracle_code)
        topk_rank = int(rows[best]["topk_rank"])
        row_features = [
            float(topk_rank == 0),
            float(topk_rank == 1),
            float(topk_rank == 2),
            float(topk_rank >= 4),
            safe_log1p(rows[best].get("assignment_relative_error", 0.0)),
            float(abs(best_code - nearest_code)) / max(float(rows[best].get("codebook_size", 64)), 1.0),
        ]
        row_features.extend(ranker.feature_value(rows[best], name) for name in names)
        records.append(
            {
                "key": {"real_seed": key[0], "partition": key[1], "source_index": key[2], "candidate_index": key[3]},
                "source": rows[nearest].get("source"),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "top_prior_code": best_code,
                "top_prior_topk_rank": topk_rank,
                "status_if_changed": status_if_changed,
                "label_exact_changed": int(status_if_changed == "exact" and oracle_code != nearest_code),
                "label_safe_improve": int(
                    rows[best]["tesseract_delta_vs_nearest"] < 0 and rows[best]["parseq_delta_vs_nearest"] <= 0
                ),
                "top_prior_tesseract_delta_vs_nearest": int(rows[best]["tesseract_delta_vs_nearest"]),
                "top_prior_parseq_delta_vs_nearest": int(rows[best]["parseq_delta_vs_nearest"]),
            }
        )
        features.append(row_features)
    return records, np.asarray(features, dtype=np.float32), extra_names + names


def split_masks(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    partitions = np.asarray([str(row["key"]["partition"]) for row in records])
    return partitions == "train", partitions == "val"


def policy_metrics(records: list[dict[str, Any]], accept: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    counts = Counter()
    tess_delta = 0
    parseq_delta = 0
    audits = []
    for idx, row in enumerate(records):
        if not bool(mask[idx]):
            continue
        nearest_code = int(row["nearest_code"])
        oracle_code = int(row["oracle_code"])
        selected_code = int(row["top_prior_code"]) if bool(accept[idx]) else nearest_code
        status = ranker.classify(selected_code, nearest_code, oracle_code)
        if bool(accept[idx]):
            t_delta = int(row["top_prior_tesseract_delta_vs_nearest"])
            p_delta = int(row["top_prior_parseq_delta_vs_nearest"])
        else:
            t_delta = 0
            p_delta = 0
        counts[status] += 1
        counts["oracle_change_groups"] += int(oracle_code != nearest_code)
        counts["changed_groups"] += int(selected_code != nearest_code)
        counts["exact_changed_groups"] += int(status == "exact" and oracle_code != nearest_code)
        counts["parseq_worsen_groups"] += int(p_delta > 0)
        counts["tesseract_worsen_groups"] += int(t_delta > 0)
        counts["tesseract_improve_groups"] += int(t_delta < 0)
        tess_delta += t_delta
        parseq_delta += p_delta
        audits.append(
            {
                "key": row["key"],
                "source": row["source"],
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "top_prior_code": int(row["top_prior_code"]),
                "selected_code": selected_code,
                "selected_status": status,
                "accepted": bool(accept[idx]),
                "selected_tesseract_delta_vs_nearest": t_delta,
                "selected_parseq_delta_vs_nearest": p_delta,
            }
        )
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
        "audits": audits,
    }


def thresholds(scores: np.ndarray, train_mask: np.ndarray) -> list[float]:
    values = sorted(set(float(score) for score in scores[train_mask]))
    if not values:
        return [1.0]
    return [max(values) + 1e-6] + values + [min(values) - 1e-6]


def choose_train_policies(records: list[dict[str, Any]], train_mask: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    rows = []
    for threshold in thresholds(scores, train_mask):
        accept = scores >= threshold
        metrics = policy_metrics(records, accept, train_mask)
        metrics.pop("audits")
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
    policies["change_all"] = min(rows, key=lambda row: row["threshold"])
    policies["no_change"] = max(rows, key=lambda row: row["threshold"])
    return policies


def evaluate_policy_bundle(
    records: list[dict[str, Any]],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    scores: np.ndarray,
    policy_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out = {}
    for name, row in policy_rows.items():
        threshold = float(row["threshold"])
        accept = scores >= threshold
        train = policy_metrics(records, accept, train_mask)
        val = policy_metrics(records, accept, val_mask)
        out[name] = {
            "threshold": threshold,
            "train": {k: v for k, v in train.items() if k != "audits"},
            "val": {k: v for k, v in val.items() if k != "audits"},
            "val_audits": val["audits"],
        }
    return out


def train_model_scores(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    models = {
        "logistic_exact": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=0.25, max_iter=2000, random_state=seed),
        ),
        "hgb_exact": HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=6,
            learning_rate=0.04,
            l2_regularization=0.2,
            random_state=seed,
        ),
    }
    scores = {}
    for name, model in models.items():
        model.fit(x[train_mask], y[train_mask])
        scores[name] = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    return scores


def feature_threshold_scores(x: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        values = x[:, idx].astype(np.float64)
        scores[f"feature_le_{name}"] = -values
        scores[f"feature_ge_{name}"] = values
    return scores


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Train-tuned no-op policies over the top non-nearest first-stage prior candidate.",
        "This is a table-level diagnostic and does not export counted `.oscr` streams.",
        "",
        "## Candidate Prior",
        "",
        f"- train top-prior exact changed: `{result['top_prior_baseline']['train']['exact_changed_groups']}`",
        f"- train top-prior false+wrong: `{result['top_prior_baseline']['train']['false_change'] + result['top_prior_baseline']['train']['wrong_change']}`",
        f"- train top-prior Tesseract delta: `{result['top_prior_baseline']['train']['tesseract_delta_vs_nearest']}`",
        f"- val top-prior exact changed: `{result['top_prior_baseline']['val']['exact_changed_groups']}`",
        f"- val top-prior false+wrong: `{result['top_prior_baseline']['val']['false_change'] + result['top_prior_baseline']['val']['wrong_change']}`",
        f"- val top-prior Tesseract delta: `{result['top_prior_baseline']['val']['tesseract_delta_vs_nearest']}`",
        "",
        "## Best Train-Tuned Policies",
        "",
        "| score | policy | train changed | train exact changed | train false+wrong | train T | val changed | val exact changed | val false+wrong | val T | val P |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["selected_policy_summary"]:
        train = row["train"]
        val = row["val"]
        lines.append(
            f"| `{row['score_name']}` | `{row['policy_name']}` | {train['changed_groups']} | {train['exact_changed_groups']} | "
            f"{train['false_change'] + train['wrong_change']} | {train['tesseract_delta_vs_nearest']} | "
            f"{val['changed_groups']} | {val['exact_changed_groups']} | {val['false_change'] + val['wrong_change']} | "
            f"{val['tesseract_delta_vs_nearest']} | {val['parseq_delta_vs_nearest']} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_topk_prior_noop_policy")
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--max-feature-thresholds", type=int, default=40)
    args = parser.parse_args()

    rows = ranker.read_jsonl(args.table)
    groups = ranker.group_rows(rows)
    records, x, names = build_group_records(rows, groups)
    train_mask, val_mask = split_masks(records)
    y_exact = np.asarray([int(row["label_exact_changed"]) for row in records], dtype=np.int64)
    all_accept = np.ones(len(records), dtype=bool)
    no_accept = np.zeros(len(records), dtype=bool)
    top_prior_baseline = {
        "train": {k: v for k, v in policy_metrics(records, all_accept, train_mask).items() if k != "audits"},
        "val": {k: v for k, v in policy_metrics(records, all_accept, val_mask).items() if k != "audits"},
        "no_change_train": {k: v for k, v in policy_metrics(records, no_accept, train_mask).items() if k != "audits"},
        "no_change_val": {k: v for k, v in policy_metrics(records, no_accept, val_mask).items() if k != "audits"},
    }

    all_scores = train_model_scores(x, y_exact, train_mask, args.seed)
    # Cheap one-dimensional controls. Rank features by train AUC-like separation
    # using absolute difference of positive/negative means to avoid a huge table.
    pos = x[train_mask & (y_exact == 1)]
    neg = x[train_mask & (y_exact == 0)]
    if len(pos) and len(neg):
        diffs = np.abs(pos.mean(axis=0) - neg.mean(axis=0)) / np.maximum(x[train_mask].std(axis=0), 1e-6)
        top_feature_indices = np.argsort(diffs)[::-1][: args.max_feature_thresholds]
    else:
        top_feature_indices = np.arange(min(x.shape[1], args.max_feature_thresholds))
    feature_score_names = [names[idx] for idx in top_feature_indices]
    all_scores.update(feature_threshold_scores(x[:, top_feature_indices], feature_score_names))

    policy_results = {}
    selected_policy_summary = []
    for score_name, scores in sorted(all_scores.items()):
        train_policies = choose_train_policies(records, train_mask, scores)
        bundle = evaluate_policy_bundle(records, train_mask, val_mask, scores, train_policies)
        policy_results[score_name] = bundle
        for policy_name, policy in bundle.items():
            if policy_name not in {"zero_error_safe", "one_error_parseq_safe", "best_train_tesseract_parseq_safe"}:
                continue
            selected_policy_summary.append(
                {
                    "score_name": score_name,
                    "policy_name": policy_name,
                    "threshold": policy["threshold"],
                    "train": policy["train"],
                    "val": policy["val"],
                }
            )
    selected_policy_summary.sort(
        key=lambda row: (
            row["val"]["tesseract_delta_vs_nearest"],
            row["val"]["false_change"] + row["val"]["wrong_change"],
            -row["val"]["exact_changed_groups"],
            row["score_name"],
        )
    )
    best = selected_policy_summary[0] if selected_policy_summary else None
    interpretation = (
        "This tests whether the source-robust top-k prior from N097 becomes deployable with train-only no-op calibration. "
        "Promotion still requires beating the current actual-bitstream -8 Tesseract floor with low false/wrong changes and PARSeq no worse."
    )
    scalar_metrics: dict[str, dict[str, float]] = {}
    if best is not None:
        scalar_metrics = {
            "best_val_tesseract_delta": {"value": float(best["val"]["tesseract_delta_vs_nearest"])},
            "best_val_parseq_delta": {"value": float(best["val"]["parseq_delta_vs_nearest"])},
            "best_val_exact_changed": {"value": float(best["val"]["exact_changed_groups"])},
            "best_val_false_wrong": {"value": float(best["val"]["false_change"] + best["val"]["wrong_change"])},
            "top_prior_val_tesseract_delta": {"value": float(top_prior_baseline["val"]["tesseract_delta_vs_nearest"])},
            "top_prior_val_exact_changed": {"value": float(top_prior_baseline["val"]["exact_changed_groups"])},
            "top_prior_val_false_wrong": {
                "value": float(top_prior_baseline["val"]["false_change"] + top_prior_baseline["val"]["wrong_change"])
            },
        }
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_topk_prior_noop_policy_not_promoted_selector",
        "inputs": {"table": {"path": str(args.table), "sha256": sha256_file(args.table)}},
        "config": {
            "seed": args.seed,
            "max_feature_thresholds": args.max_feature_thresholds,
            "feature_count": len(names),
            "feature_rule": "top prior candidate deployable numeric features; OCR/oracle/label/source/reference text excluded from inputs",
        },
        "top_prior_baseline": top_prior_baseline,
        "selected_policy_summary": selected_policy_summary[:20],
        "best_policy": best,
        "policy_results": policy_results,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "base_ranker_script": sha256_file(SCRIPT_DIR / "train_top8_trainval_tabular_listwise_ranker.py"),
        },
        "aggregate": {"scalar_metrics": scalar_metrics},
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "best_policy": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
