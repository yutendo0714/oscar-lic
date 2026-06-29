#!/usr/bin/env python3
"""Non-nearest-only glyph-code ranker diagnostic for top-8 assignment."""

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
    return sorted(k for k in records[0]["features"] if k.startswith("glyph_")) + ["topk_rank_norm"]


def feature_vector(row: dict[str, Any], names: list[str]) -> list[float]:
    out = []
    for name in names:
        if name == "topk_rank_norm":
            out.append(float(row["topk_rank"]) / 8.0)
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
    return model.predict_proba(x_train)[:, 1], model.predict_proba(x_val)[:, 1], model.coef_[0]


def add_scores(rows: list[dict[str, Any]], scores: np.ndarray) -> None:
    for row, score in zip(rows, scores, strict=True):
        row["nonnearest_score"] = float(score)


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key3(row)].append(row)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["topk_rank"]))
    return grouped


def classify(selected_code: int, nearest_code: int, oracle_code: int) -> str:
    if selected_code == oracle_code:
        return "exact"
    if oracle_code == nearest_code and selected_code != nearest_code:
        return "false_change"
    if oracle_code != nearest_code and selected_code == nearest_code:
        return "missed_oracle"
    return "wrong_change"


def rank_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {
        "le1": sum(1 for row in rows if int(row[field]) <= 1),
        "le2": sum(1 for row in rows if int(row[field]) <= 2),
        "le4": sum(1 for row in rows if int(row[field]) <= 4),
        "le8": sum(1 for row in rows if int(row[field]) <= 8),
    }


def audit_groups(
    grouped_all: dict[tuple[int, int, int], list[dict[str, Any]]],
    grouped_nonnearest: dict[tuple[int, int, int], list[dict[str, Any]]],
    candidate_info: dict[tuple[int, int, int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audits = []
    counts = Counter()
    tesseract_delta = 0
    parseq_delta = 0
    for key, all_rows in sorted(grouped_all.items()):
        nearest_code = int(all_rows[0]["nearest_code"])
        oracle_code = int(all_rows[0]["oracle_code"])
        target = str(all_rows[0]["next_model_target"])
        nonnearest = grouped_nonnearest.get(key, [])
        best = max(nonnearest, key=lambda row: (float(row["nonnearest_score"]), -int(row["topk_rank"]))) if nonnearest else None
        if oracle_code != nearest_code and best is not None:
            selected_code = int(best["code_index"])
        else:
            selected_code = nearest_code
        status = classify(selected_code, nearest_code, oracle_code)
        counts[status] += 1
        counts["changed_groups"] += int(selected_code != nearest_code)
        info = candidate_info[key + (selected_code,)]
        tesseract_delta += int(info.get("tesseract_delta_vs_nearest", 0))
        parseq_delta += int(info.get("parseq_delta_vs_nearest", 0))

        by_score = sorted(nonnearest, key=lambda row: (float(row["nonnearest_score"]), -int(row["topk_rank"])), reverse=True)
        oracle_rank = None
        if oracle_code != nearest_code:
            oracle_rank = 1 + next(index for index, row in enumerate(by_score) if int(row["code_index"]) == oracle_code)
        audits.append(
            {
                "key": all_rows[0]["key"],
                "source": all_rows[0]["source"],
                "reference": all_rows[0]["reference"],
                "next_model_target": target,
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "selected_code_oracle_change_only": selected_code,
                "selected_status_oracle_change_only": status,
                "oracle_nonnearest_score_rank": oracle_rank,
                "oracle_topk_rank": int(next(row for row in all_rows if int(row["code_index"]) == oracle_code)["topk_rank"]),
                "best_nonnearest_code": int(best["code_index"]) if best is not None else nearest_code,
                "best_nonnearest_score": float(best["nonnearest_score"]) if best is not None else 0.0,
                "best_nonnearest_tesseract_delta_vs_nearest": int(candidate_info[key + (int(best["code_index"]),)].get("tesseract_delta_vs_nearest", 0)) if best is not None else 0,
                "best_nonnearest_parseq_delta_vs_nearest": int(candidate_info[key + (int(best["code_index"]),)].get("parseq_delta_vs_nearest", 0)) if best is not None else 0,
            }
        )
    metrics = {
        "groups": len(grouped_all),
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "tesseract_delta_vs_nearest": int(tesseract_delta),
        "parseq_delta_vs_nearest": int(parseq_delta),
    }
    return audits, metrics


def aggregate_audit(group_audits: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_audits:
        by_target[str(row["next_model_target"])].append(row)
    out: dict[str, Any] = {}
    for target, rows in sorted(by_target.items()):
        rank_rows = [row for row in rows if row["oracle_nonnearest_score_rank"] is not None]
        out[target] = {
            "groups": len(rows),
            "selected_status_counts": dict(Counter(str(row["selected_status_oracle_change_only"]) for row in rows)),
            "oracle_nonnearest_score_rank_counts": rank_counts(rank_rows, "oracle_nonnearest_score_rank") if rank_rows else {},
            "oracle_topk_rank_counts": rank_counts(rows, "oracle_topk_rank"),
            "best_nonnearest_tesseract_delta_sum": int(sum(int(row["best_nonnearest_tesseract_delta_vs_nearest"]) for row in rows)),
        }
    return out


def coefficient_summary(names: list[str], fold_coefficients: list[np.ndarray]) -> dict[str, Any]:
    coef = np.stack(fold_coefficients, axis=0)
    mean = coef.mean(axis=0)
    rows = [{"feature": name, "mean_coef": float(value)} for name, value in zip(names, mean, strict=True)]
    rows.sort(key=lambda item: item["mean_coef"], reverse=True)
    return {"top_positive": rows[:12], "top_negative": list(reversed(rows[-12:]))}


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Diagnostic non-nearest-only ranker over top-8 glyph-code features.",
        "This isolates candidate ranking from the no-op/nearest decision and is not deployable by itself.",
        "",
        "## Oracle-Change-Only Policy",
        "",
        "| val seed | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        m = row["oracle_change_only_policy"]
        lines.append(
            f"| {row['val_seed']} | {m['changed_groups']} | {m['exact']} | {m['false_change']} | "
            f"{m['wrong_change']} | {m['missed_oracle']} | {m['tesseract_delta_vs_nearest']} | "
            f"{m['parseq_delta_vs_nearest']} |"
        )
    lines.extend(["", "## Target Rank Audit", ""])
    lines.append("| target | groups | selected status | oracle nonnearest score <=1/2/4/8 | oracle topk <=1/2/4/8 | best-nonnearest Tesseract delta sum |")
    lines.append("|---|---:|---|---|---|---:|")
    for target, row in result["aggregate"]["target_rank_audit"].items():
        status = ", ".join(f"{k}:{v}" for k, v in sorted(row["selected_status_counts"].items()))
        score_counts = row["oracle_nonnearest_score_rank_counts"] or {"le1": 0, "le2": 0, "le4": 0, "le8": 0}
        topk_counts = row["oracle_topk_rank_counts"]
        lines.append(
            f"| {target} | {row['groups']} | {status} | "
            f"{score_counts['le1']}/{score_counts['le2']}/{score_counts['le4']}/{score_counts['le8']} | "
            f"{topk_counts['le1']}/{topk_counts['le2']}/{topk_counts['le4']}/{topk_counts['le8']} | "
            f"{row['best_nonnearest_tesseract_delta_sum']} |"
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
            "- If non-nearest-only ranking recovers many oracle candidates, the next model should separate candidate scoring from no-op acceptance.",
            "- If it still misses hard positives, glyph-code scalars alone are insufficient even after removing the nearest prior.",
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
    parser.add_argument("--experiment-id", default="eval300_top8_glyphcode_nonnearest_ranker")
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    glyph = read_json(args.glyph_features)
    records_all = glyph["candidate_records"]
    records_nonnearest = [dict(row) for row in records_all if int(row["code_index"]) != int(row["nearest_code"])]
    names = feature_names(records_all)
    candidate_info = {key4_candidate(row): row for row in read_jsonl(args.candidate_table)}

    per_seed = []
    group_audits = []
    fold_coefficients = []
    for val_seed in sorted({int(row["key"]["real_seed"]) for row in records_all}):
        train_rows = [dict(row) for row in records_nonnearest if int(row["key"]["real_seed"]) != val_seed]
        val_rows = [dict(row) for row in records_nonnearest if int(row["key"]["real_seed"]) == val_seed]
        train_scores, val_scores, coef = fit_scores(train_rows, val_rows, names, args.seed + val_seed)
        fold_coefficients.append(coef)
        add_scores(train_rows, train_scores)
        add_scores(val_rows, val_scores)
        val_all = [dict(row) for row in records_all if int(row["key"]["real_seed"]) == val_seed]
        grouped_all = group_rows(val_all)
        grouped_nonnearest = group_rows(val_rows)
        audits, metrics = audit_groups(grouped_all, grouped_nonnearest, candidate_info)
        group_audits.extend(audits)
        per_seed.append(
            {
                "val_seed": val_seed,
                "train_nonnearest_rows": len(train_rows),
                "val_nonnearest_rows": len(val_rows),
                "oracle_change_only_policy": metrics,
                "group_audits": audits,
            }
        )

    target_audit = aggregate_audit(group_audits)
    recover = target_audit.get("recover_shortlist_oracle_change") or {}
    score_counts = recover.get("oracle_nonnearest_score_rank_counts", {})
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_candidate_ranking_not_promoted_selector",
        "inputs": {
            "glyph_features": str(args.glyph_features),
            "glyph_features_sha256": sha256_file(args.glyph_features),
            "candidate_table": str(args.candidate_table),
            "candidate_table_sha256": sha256_file(args.candidate_table),
        },
        "config": {
            "seed": args.seed,
            "feature_names": names,
            "model": "non-nearest-only sklearn.linear_model.LogisticRegression(class_weight=balanced,C=1.0)",
            "policy_note": "oracle-change-only policy changes groups only when the held-out oracle code differs from nearest; it isolates candidate ranking and is not deployable",
        },
        "per_seed": per_seed,
        "aggregate": {
            "target_rank_audit": target_audit,
            "coefficient_summary": coefficient_summary(names, fold_coefficients),
            "scalar_metrics": {
                "recover_oracle_nonnearest_score_rank_le1": {"value": float(score_counts.get("le1", 0))},
                "recover_oracle_nonnearest_score_rank_le2": {"value": float(score_counts.get("le2", 0))},
                "recover_oracle_nonnearest_score_rank_le4": {"value": float(score_counts.get("le4", 0))},
            },
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
