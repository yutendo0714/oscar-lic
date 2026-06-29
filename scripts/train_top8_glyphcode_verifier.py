#!/usr/bin/env python3
"""Cross-fit diagnostic verifier over top-8 glyph-code differential features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]


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


def key3_from_record(row: dict[str, Any]) -> tuple[int, int, int]:
    key = row["key"]
    return (int(key["real_seed"]), int(key["source_index"]), int(key["candidate_index"]))


def key4_from_candidate(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"]), int(row["code_index"]))


def feature_names(records: list[dict[str, Any]], include_rank: bool, include_nearest_flag: bool) -> list[str]:
    names = sorted(k for k in records[0]["features"] if k.startswith("glyph_"))
    if include_rank:
        names.append("topk_rank_norm")
    if include_nearest_flag:
        names.append("is_nearest")
    return names


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    out = []
    for name in names:
        if name == "topk_rank_norm":
            out.append(float(row["topk_rank"]) / 8.0)
        elif name == "is_nearest":
            out.append(float(int(row["code_index"]) == int(row["nearest_code"])))
        else:
            out.append(float(row["features"].get(name, 0.0)))
    return out


def standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (val - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


def fit_model(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], names: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray([vector(row, names) for row in train_rows], dtype=np.float64)
    y_train = np.asarray([int(row["is_oracle_code"]) for row in train_rows], dtype=np.int64)
    x_val = np.asarray([vector(row, names) for row in val_rows], dtype=np.float64)
    x_train, x_val, _ = standardize(x_train, x_val)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_train)[:, 1], model.predict_proba(x_val)[:, 1]


def add_scores(rows: list[dict[str, Any]], probs: np.ndarray) -> None:
    if len(rows) != len(probs):
        raise ValueError("score length mismatch")
    for row, prob in zip(rows, probs, strict=True):
        row["verifier_score"] = float(prob)


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key3_from_record(row)].append(row)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["topk_rank"]))
    return grouped


def classify_selection(selected: dict[str, Any], group: list[dict[str, Any]]) -> str:
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


def select_argmax(group: list[dict[str, Any]]) -> dict[str, Any]:
    return max(group, key=lambda row: (float(row["verifier_score"]), -int(row["topk_rank"])))


def policy_metrics(
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]],
    candidate_info: dict[tuple[int, int, int, int], dict[str, Any]],
    threshold: float | None,
) -> dict[str, Any]:
    counts = Counter()
    tesseract_delta = 0
    parseq_delta = 0
    rows = []
    for key, group in sorted(grouped.items()):
        selected = select_argmax(group) if threshold is None else select_margin(group, threshold)
        status = classify_selection(selected, group)
        counts[status] += 1
        changed = int(selected["code_index"]) != int(selected["nearest_code"])
        counts["changed_groups"] += int(changed)
        info = candidate_info[key3_from_record(selected) + (int(selected["code_index"]),)]
        t_delta = int(info.get("tesseract_delta_vs_nearest", 0))
        p_delta = int(info.get("parseq_delta_vs_nearest", 0))
        tesseract_delta += t_delta
        parseq_delta += p_delta
        rows.append(
            {
                "key": selected["key"],
                "selected_code": int(selected["code_index"]),
                "nearest_code": int(selected["nearest_code"]),
                "oracle_code": int(selected["oracle_code"]),
                "status": status,
                "changed": changed,
                "score": float(selected["verifier_score"]),
                "tesseract_delta_vs_nearest": t_delta,
                "parseq_delta_vs_nearest": p_delta,
            }
        )
    n = len(grouped)
    return {
        "groups": n,
        "threshold": threshold,
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "tesseract_delta_vs_nearest": int(tesseract_delta),
        "parseq_delta_vs_nearest": int(parseq_delta),
        "accuracy": float(counts["exact"] / n) if n else 0.0,
        "rows": rows,
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
    candidates = [min(values) - 1e-6] if values else [0.0]
    candidates.extend((a + b) / 2.0 for a, b in zip(values, values[1:], strict=False))
    candidates.append(max(values) + 1e-6 if values else 1.0)
    return candidates


def policy_loss(metrics: dict[str, Any]) -> float:
    # False/wrong changes are expensive because they break the no-op precision floor.
    return (
        2.0 * float(metrics["false_change"])
        + 2.0 * float(metrics["wrong_change"])
        + 1.0 * float(metrics["missed_oracle"])
        - 0.25 * float(metrics["exact"])
    )


def tune_threshold(
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]],
    candidate_info: dict[tuple[int, int, int, int], dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.0
    best_metrics: dict[str, Any] | None = None
    best_loss = float("inf")
    for threshold in threshold_grid(grouped):
        metrics = policy_metrics(grouped, candidate_info, threshold)
        loss = policy_loss(metrics)
        if loss < best_loss:
            best_loss = loss
            best_threshold = threshold
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=0)) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "changed_groups",
        "exact",
        "false_change",
        "wrong_change",
        "missed_oracle",
        "tesseract_delta_vs_nearest",
        "parseq_delta_vs_nearest",
        "accuracy",
    ]
    return {key: summarize([float(row["eval_margin"][key]) for row in per_seed]) for key in keys}


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Cross-fit diagnostic verifier over top-8 glyph-code differential features.",
        "This uses held-out OCR-derived oracle labels as diagnostic targets only; it is not a promoted selector.",
        "",
        "## Aggregate Margin Policy",
        "",
        "| metric | mean | std | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, stats in result["aggregate"]["eval_margin"].items():
        lines.append(f"| {key} | {stats['mean']} | {stats['std']} | {stats['min']} | {stats['max']} |")
    lines.extend(["", "## Per-Seed", "", "| val seed | threshold | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in result["per_seed"]:
        m = row["eval_margin"]
        lines.append(
            f"| {row['val_seed']} | {row['threshold']} | {m['changed_groups']} | {m['exact']} | "
            f"{m['false_change']} | {m['wrong_change']} | {m['missed_oracle']} | "
            f"{m['tesseract_delta_vs_nearest']} | {m['parseq_delta_vs_nearest']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The useful question is whether glyphcode evidence can improve exact top8 recovery without losing the no-op precision floor.",
            "- If false/wrong changes remain high or Tesseract delta fails to beat the current `-8` actual-bitstream policy, treat this as a representation diagnostic only.",
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
    parser.add_argument("--experiment-id", default="eval300_top8_glyphcode_verifier")
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--include-rank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-nearest-flag", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    glyph = read_json(args.glyph_features)
    records = glyph["candidate_records"]
    names = feature_names(records, include_rank=args.include_rank, include_nearest_flag=args.include_nearest_flag)
    candidate_info = {key4_from_candidate(row): row for row in read_jsonl(args.candidate_table)}
    per_seed = []
    for val_seed in sorted({int(row["key"]["real_seed"]) for row in records}):
        train_rows = [dict(row) for row in records if int(row["key"]["real_seed"]) != val_seed]
        val_rows = [dict(row) for row in records if int(row["key"]["real_seed"]) == val_seed]
        train_probs, val_probs = fit_model(train_rows, val_rows, names, seed=args.seed + val_seed)
        add_scores(train_rows, train_probs)
        add_scores(val_rows, val_probs)
        train_grouped = group_rows(train_rows)
        val_grouped = group_rows(val_rows)
        threshold, train_margin = tune_threshold(train_grouped, candidate_info)
        eval_margin = policy_metrics(val_grouped, candidate_info, threshold)
        eval_argmax = policy_metrics(val_grouped, candidate_info, threshold=None)
        per_seed.append(
            {
                "val_seed": val_seed,
                "train_groups": len(train_grouped),
                "eval_groups": len(val_grouped),
                "threshold": float(threshold),
                "train_margin": {key: value for key, value in train_margin.items() if key != "rows"},
                "eval_margin": {key: value for key, value in eval_margin.items() if key != "rows"},
                "eval_argmax": {key: value for key, value in eval_argmax.items() if key != "rows"},
                "eval_rows": eval_margin["rows"],
            }
        )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_crossfit_not_promoted_selector",
        "inputs": {
            "glyph_features": str(args.glyph_features),
            "glyph_features_sha256": sha256_file(args.glyph_features),
            "candidate_table": str(args.candidate_table),
            "candidate_table_sha256": sha256_file(args.candidate_table),
        },
        "config": {
            "seed": args.seed,
            "feature_names": names,
            "include_rank": args.include_rank,
            "include_nearest_flag": args.include_nearest_flag,
            "model": "sklearn.linear_model.LogisticRegression(class_weight=balanced,C=1.0)",
            "threshold_tuning": "train-fold margin threshold minimizing 2*false + 2*wrong + missed - 0.25*exact",
        },
        "per_seed": per_seed,
        "aggregate": {"eval_margin": aggregate(per_seed)},
    }
    result["aggregate"]["scalar_metrics"] = {
        f"eval_margin_{metric}_{stat}": {"value": float(value)}
        for metric, stats in result["aggregate"]["eval_margin"].items()
        for stat, value in stats.items()
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
