#!/usr/bin/env python3
"""Score-space audit for the top-8 glyph-code verifier."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key3(row: dict[str, Any]) -> tuple[int, int, int]:
    key = row["key"]
    return (int(key["real_seed"]), int(key["source_index"]), int(key["candidate_index"]))


def key4_candidate(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"]), int(row["code_index"]))


def feature_names(records: list[dict[str, Any]]) -> list[str]:
    return sorted(k for k in records[0]["features"] if k.startswith("glyph_")) + ["topk_rank_norm", "is_nearest"]


def feature_vector(row: dict[str, Any], names: list[str]) -> list[float]:
    out = []
    for name in names:
        if name == "topk_rank_norm":
            out.append(float(row["topk_rank"]) / 8.0)
        elif name == "is_nearest":
            out.append(float(int(row["code_index"]) == int(row["nearest_code"])))
        else:
            out.append(float(row["features"].get(name, 0.0)))
    return out


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (val - mean) / std


def fit_scores(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    names: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray([feature_vector(row, names) for row in train_rows], dtype=np.float64)
    y_train = np.asarray([int(row["is_oracle_code"]) for row in train_rows], dtype=np.int64)
    x_val = np.asarray([feature_vector(row, names) for row in val_rows], dtype=np.float64)
    x_train, x_val = standardize(x_train, x_val)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(x_train, y_train)
    return model.predict_proba(train_rows and x_train)[:, 1], model.predict_proba(x_val)[:, 1], model.coef_[0]


def add_scores(rows: list[dict[str, Any]], scores: np.ndarray) -> None:
    for row, score in zip(rows, scores, strict=True):
        row["verifier_score"] = float(score)


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key3(row)].append(row)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["topk_rank"]))
    return grouped


def classify(selected: dict[str, Any], group: list[dict[str, Any]]) -> str:
    nearest_code = int(group[0]["nearest_code"])
    oracle_code = int(group[0]["oracle_code"])
    selected_code = int(selected["code_index"])
    if selected_code == oracle_code:
        return "exact"
    if oracle_code == nearest_code and selected_code != nearest_code:
        return "false_change"
    if oracle_code != nearest_code and selected_code == nearest_code:
        return "missed_oracle"
    return "wrong_change"


def select_margin(group: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    nearest = next(row for row in group if int(row["code_index"]) == int(row["nearest_code"]))
    nonnearest = [row for row in group if int(row["code_index"]) != int(row["nearest_code"])]
    if not nonnearest:
        return nearest
    best = max(nonnearest, key=lambda row: (float(row["verifier_score"]), -int(row["topk_rank"])))
    if float(best["verifier_score"]) - float(nearest["verifier_score"]) >= threshold:
        return best
    return nearest


def policy_metrics(
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]],
    candidate_info: dict[tuple[int, int, int, int], dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    counts = Counter()
    tesseract_delta = 0
    parseq_delta = 0
    for group in grouped.values():
        selected = select_margin(group, threshold)
        status = classify(selected, group)
        counts[status] += 1
        counts["changed_groups"] += int(int(selected["code_index"]) != int(selected["nearest_code"]))
        info = candidate_info[key3(selected) + (int(selected["code_index"]),)]
        tesseract_delta += int(info.get("tesseract_delta_vs_nearest", 0))
        parseq_delta += int(info.get("parseq_delta_vs_nearest", 0))
    n = len(grouped)
    return {
        "groups": n,
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "tesseract_delta_vs_nearest": int(tesseract_delta),
        "parseq_delta_vs_nearest": int(parseq_delta),
        "accuracy": float(counts["exact"] / n) if n else 0.0,
    }


def threshold_grid(grouped: dict[tuple[int, int, int], list[dict[str, Any]]]) -> list[float]:
    margins = []
    for group in grouped.values():
        nearest = next(row for row in group if int(row["code_index"]) == int(row["nearest_code"]))
        nonnearest = [row for row in group if int(row["code_index"]) != int(row["nearest_code"])]
        if nonnearest:
            best = max(nonnearest, key=lambda row: (float(row["verifier_score"]), -int(row["topk_rank"])))
            margins.append(float(best["verifier_score"]) - float(nearest["verifier_score"]))
    values = sorted(set(margins))
    if not values:
        return [0.0]
    return [values[0] - 1e-6] + [(a + b) / 2.0 for a, b in zip(values, values[1:], strict=False)] + [values[-1] + 1e-6]


def policy_loss(metrics: dict[str, Any]) -> float:
    return (
        2.0 * float(metrics["false_change"])
        + 2.0 * float(metrics["wrong_change"])
        + float(metrics["missed_oracle"])
        - 0.25 * float(metrics["exact"])
    )


def tune_threshold(
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]],
    candidate_info: dict[tuple[int, int, int, int], dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    best = (0.0, float("inf"), {})
    for threshold in threshold_grid(grouped):
        metrics = policy_metrics(grouped, candidate_info, threshold)
        loss = policy_loss(metrics)
        if loss < best[1]:
            best = (float(threshold), float(loss), metrics)
    return best[0], best[2]


def audit_group(
    group: list[dict[str, Any]],
    threshold: float,
    candidate_info: dict[tuple[int, int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    nearest = next(row for row in group if int(row["code_index"]) == int(row["nearest_code"]))
    oracle = next(row for row in group if int(row["code_index"]) == int(row["oracle_code"]))
    selected = select_margin(group, threshold)
    by_score = sorted(group, key=lambda row: (float(row["verifier_score"]), -int(row["topk_rank"])), reverse=True)
    oracle_score_rank = 1 + next(index for index, row in enumerate(by_score) if int(row["code_index"]) == int(oracle["code_index"]))
    selected_score_rank = 1 + next(index for index, row in enumerate(by_score) if int(row["code_index"]) == int(selected["code_index"]))
    nonnearest = [row for row in group if int(row["code_index"]) != int(row["nearest_code"])]
    best_nonnearest = max(nonnearest, key=lambda row: (float(row["verifier_score"]), -int(row["topk_rank"]))) if nonnearest else nearest
    nearest_score = float(nearest["verifier_score"])
    oracle_score = float(oracle["verifier_score"])
    best_score = float(best_nonnearest["verifier_score"])
    selected_info = candidate_info[key3(selected) + (int(selected["code_index"]),)]
    return {
        "key": group[0]["key"],
        "source": group[0]["source"],
        "reference": group[0]["reference"],
        "next_model_target": group[0]["next_model_target"],
        "nearest_code": int(nearest["code_index"]),
        "oracle_code": int(oracle["code_index"]),
        "selected_code": int(selected["code_index"]),
        "best_nonnearest_code": int(best_nonnearest["code_index"]),
        "oracle_topk_rank": int(oracle["topk_rank"]),
        "oracle_score_rank": int(oracle_score_rank),
        "selected_score_rank": int(selected_score_rank),
        "nearest_score": nearest_score,
        "oracle_score": oracle_score,
        "best_nonnearest_score": best_score,
        "oracle_minus_nearest_score": float(oracle_score - nearest_score),
        "best_nonnearest_minus_nearest_score": float(best_score - nearest_score),
        "selected_status": classify(selected, group),
        "selected_tesseract_delta_vs_nearest": int(selected_info.get("tesseract_delta_vs_nearest", 0)),
        "selected_parseq_delta_vs_nearest": int(selected_info.get("parseq_delta_vs_nearest", 0)),
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def rank_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {
        "le1": sum(1 for row in rows if int(row[field]) <= 1),
        "le2": sum(1 for row in rows if int(row[field]) <= 2),
        "le4": sum(1 for row in rows if int(row[field]) <= 4),
        "le8": sum(1 for row in rows if int(row[field]) <= 8),
    }


def aggregate_audit(group_audits: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_audits:
        by_target[str(row["next_model_target"])].append(row)
    summary: dict[str, Any] = {}
    for target, rows in sorted(by_target.items()):
        summary[target] = {
            "groups": len(rows),
            "selected_status_counts": dict(Counter(str(row["selected_status"]) for row in rows)),
            "oracle_score_rank_counts": rank_counts(rows, "oracle_score_rank"),
            "oracle_topk_rank_counts": rank_counts(rows, "oracle_topk_rank"),
            "oracle_minus_nearest_score": summarize([float(row["oracle_minus_nearest_score"]) for row in rows]),
            "best_nonnearest_minus_nearest_score": summarize([float(row["best_nonnearest_minus_nearest_score"]) for row in rows]),
        }
    return summary


def coefficient_summary(names: list[str], fold_coefficients: list[np.ndarray]) -> dict[str, Any]:
    coef = np.stack(fold_coefficients, axis=0)
    mean = coef.mean(axis=0)
    rows = [{"feature": name, "mean_coef": float(value)} for name, value in zip(names, mean, strict=True)]
    rows.sort(key=lambda item: item["mean_coef"], reverse=True)
    return {
        "top_positive": rows[:12],
        "top_negative": list(reversed(rows[-12:])),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Diagnostic score audit for the N085 glyph-code verifier. This is not a promoted selector.",
        "",
        "## Policy Reproduction",
        "",
        "| val seed | threshold | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        m = row["eval_margin"]
        lines.append(
            f"| {row['val_seed']} | {row['threshold']} | {m['changed_groups']} | {m['exact']} | "
            f"{m['false_change']} | {m['wrong_change']} | {m['missed_oracle']} | "
            f"{m['tesseract_delta_vs_nearest']} | {m['parseq_delta_vs_nearest']} |"
        )
    lines.extend(["", "## Target Score-Rank Audit", ""])
    lines.append("| target | groups | selected status | oracle score <=1/2/4/8 | oracle topk <=1/2/4/8 | oracle-nearest score mean | best-nonnearest margin mean |")
    lines.append("|---|---:|---|---|---|---:|---:|")
    for target, row in result["aggregate"]["target_score_audit"].items():
        status = ", ".join(f"{k}:{v}" for k, v in sorted(row["selected_status_counts"].items()))
        score_counts = row["oracle_score_rank_counts"]
        topk_counts = row["oracle_topk_rank_counts"]
        lines.append(
            f"| {target} | {row['groups']} | {status} | "
            f"{score_counts['le1']}/{score_counts['le2']}/{score_counts['le4']}/{score_counts['le8']} | "
            f"{topk_counts['le1']}/{topk_counts['le2']}/{topk_counts['le4']}/{topk_counts['le8']} | "
            f"{row['oracle_minus_nearest_score']['mean']} | {row['best_nonnearest_minus_nearest_score']['mean']} |"
        )
    lines.extend(["", "## Mean Standardized Coefficients", "", "### Positive", ""])
    for row in result["aggregate"]["coefficient_summary"]["top_positive"]:
        lines.append(f"- `{row['feature']}`: {row['mean_coef']}")
    lines.extend(["", "### Negative", ""])
    for row in result["aggregate"]["coefficient_summary"]["top_negative"]:
        lines.append(f"- `{row['feature']}`: {row['mean_coef']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This audit explains why the direct glyph-code verifier is not promotable: useful recover candidates are often not score-rank 1, while no-op pressure remains close enough that train-fold margin tuning abstains.",
            "- Use glyph-code signals as auxiliary local evidence in a richer verifier; do not retry glyphcode-only thresholding or logistic calibration.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glyph-features", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_glyphcode_verifier_score_audit")
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    glyph = read_json(args.glyph_features)
    records = glyph["candidate_records"]
    names = feature_names(records)
    candidate_info = {key4_candidate(row): row for row in read_jsonl(args.candidate_table)}
    per_seed = []
    group_audits = []
    fold_coefficients = []
    for val_seed in sorted({int(row["key"]["real_seed"]) for row in records}):
        train_rows = [dict(row) for row in records if int(row["key"]["real_seed"]) != val_seed]
        val_rows = [dict(row) for row in records if int(row["key"]["real_seed"]) == val_seed]
        train_scores, val_scores, coef = fit_scores(train_rows, val_rows, names, args.seed + val_seed)
        fold_coefficients.append(coef)
        add_scores(train_rows, train_scores)
        add_scores(val_rows, val_scores)
        train_grouped = group_rows(train_rows)
        val_grouped = group_rows(val_rows)
        threshold, train_margin = tune_threshold(train_grouped, candidate_info)
        eval_margin = policy_metrics(val_grouped, candidate_info, threshold)
        seed_audits = [audit_group(group, threshold, candidate_info) for group in val_grouped.values()]
        group_audits.extend(seed_audits)
        per_seed.append(
            {
                "val_seed": val_seed,
                "threshold": threshold,
                "train_margin": train_margin,
                "eval_margin": eval_margin,
                "group_audits": seed_audits,
            }
        )

    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_score_audit_not_promoted_selector",
        "inputs": {
            "glyph_features": str(args.glyph_features),
            "glyph_features_sha256": sha256_file(args.glyph_features),
            "candidate_table": str(args.candidate_table),
            "candidate_table_sha256": sha256_file(args.candidate_table),
        },
        "config": {
            "seed": args.seed,
            "feature_names": names,
            "model": "sklearn.linear_model.LogisticRegression(class_weight=balanced,C=1.0)",
            "threshold_tuning": "same as N085: train-fold margin threshold minimizing 2*false + 2*wrong + missed - 0.25*exact",
        },
        "per_seed": per_seed,
        "aggregate": {
            "target_score_audit": aggregate_audit(group_audits),
            "coefficient_summary": coefficient_summary(names, fold_coefficients),
        },
    }
    recover = result["aggregate"]["target_score_audit"].get("recover_shortlist_oracle_change") or result["aggregate"]["target_score_audit"].get("recover_top8_oracle_change")
    if recover:
        result["aggregate"]["scalar_metrics"] = {
            "recover_oracle_score_rank_le1": {"value": recover["oracle_score_rank_counts"]["le1"]},
            "recover_oracle_score_rank_le2": {"value": recover["oracle_score_rank_counts"]["le2"]},
            "recover_oracle_score_rank_le4": {"value": recover["oracle_score_rank_counts"]["le4"]},
            "recover_oracle_minus_nearest_score_mean": {"value": recover["oracle_minus_nearest_score"]["mean"]},
        }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
