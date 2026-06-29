#!/usr/bin/env python3
"""Audit assignment score calibration gaps between train and validation groups."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    seed = int(row["seed"])
    return (
        int(row.get("real_seed", row.get("original_seed", seed))),
        seed,
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def row_code_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (*row_group_key(row), int(row["code_index"]))


def table_lookup(paths: list[Path]) -> tuple[dict[tuple[int, int, int, int, int], dict[str, Any]], dict[tuple[int, int, int, int], dict[str, Any]]]:
    by_code: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    group_meta: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            code_key = row_code_key(row)
            by_code[code_key] = row
            group_key = row_group_key(row)
            if group_key not in group_meta or int(row.get("is_nearest", 0)):
                group_meta[group_key] = row
    return by_code, group_meta


def load_scores(path: Path, val_seed: int) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        seed = int(row["seed"])
        real_seed = int(row.get("real_seed", seed))
        out = dict(row)
        out["_group_key"] = (real_seed, seed, int(row["source_index"]), int(row["candidate_index"]))
        out["_code_key"] = (*out["_group_key"], int(row["code_index"]))
        out["_voter_id"] = int(row["model_seed"])
        out["_partition"] = "val" if seed == val_seed else "train"
        rows.append(out)
    return rows


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "q10": None, "q50": None, "q90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def dist_bin(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    return "6+"


def len_bin(value: int) -> str:
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    if value <= 10:
        return "7-10"
    return "11+"


def error_type(selected_changed: bool, selected_is_oracle: bool, oracle_changed: bool) -> str:
    if selected_is_oracle and oracle_changed:
        return "exact_oracle"
    if selected_is_oracle and not oracle_changed:
        return "correct_nearest"
    if selected_changed and not oracle_changed:
        return "false_change"
    if (not selected_changed) and oracle_changed:
        return "missed_oracle"
    return "wrong_change"


def summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    errors = Counter(row["always_best_error"] for row in groups)
    return {
        "groups": len(groups),
        "oracle_change_groups": sum(int(row["oracle_changed"]) for row in groups),
        "always_best_changed_groups": sum(int(row["always_best_changed"]) for row in groups),
        "always_best_tesseract_delta_sum": sum(int(row["always_best_tesseract_delta"]) for row in groups),
        "always_best_parseq_delta_sum": sum(int(row["always_best_parseq_delta"]) for row in groups),
        "always_best_false_change_groups": int(errors["false_change"]),
        "always_best_wrong_change_groups": int(errors["wrong_change"]),
        "always_best_exact_oracle_groups": int(errors["exact_oracle"]),
        "always_best_missed_oracle_groups": int(errors["missed_oracle"]),
        "always_best_tesseract_worsen_groups": sum(int(row["always_best_tesseract_delta"] > 0) for row in groups),
        "best_change_score": quantiles([float(row["best_change_score"]) for row in groups if row["best_change_score"] is not None]),
        "best_change_margin": quantiles([float(row["best_change_margin"]) for row in groups if row["best_change_margin"] is not None]),
        "group_change_prob": quantiles([float(row["group_change_prob_mean"]) for row in groups if row["group_change_prob_mean"] is not None]),
    }


def grouped_summary(groups: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in groups:
        buckets[str(row[key_name])].append(row)
    return {key: summarize_groups(rows) for key, rows in sorted(buckets.items())}


def audit_score_file(score_path: Path, table_by_code: dict, group_meta: dict, val_seed: int) -> dict[str, Any]:
    rows = load_scores(score_path, val_seed=val_seed)
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["_group_key"]].append(row)

    group_rows = []
    missing_meta = 0
    for group_key, raw_rows in sorted(grouped.items()):
        by_code: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in raw_rows:
            by_code[int(row["code_index"])].append(row)
        nearest_code = next(int(row["code_index"]) for row in raw_rows if int(row["is_nearest"]))
        oracle_code = next(int(row["code_index"]) for row in raw_rows if int(row["label_assignment_oracle_choice"]))
        nearest_scores = [float(row["score"]) for row in by_code[nearest_code]]
        nearest_score = float(np.mean(nearest_scores))
        candidates = []
        for code, code_rows in by_code.items():
            if code == nearest_code:
                continue
            scores = np.asarray([float(row["score"]) for row in code_rows], dtype=np.float64)
            group_probs = np.asarray([float(row.get("group_change_prob", 1.0)) for row in code_rows], dtype=np.float64)
            template = code_rows[0]
            candidates.append(
                {
                    "code": code,
                    "row": template,
                    "score_mean": float(scores.mean()),
                    "margin_mean": float(scores.mean() - nearest_score),
                    "group_prob_mean": float(group_probs.mean()),
                    "topk_rank": int(template["topk_rank"]),
                }
            )
        best = max(candidates, key=lambda item: (item["score_mean"], item["margin_mean"], -item["topk_rank"])) if candidates else None
        meta = group_meta.get(group_key)
        if meta is None:
            missing_meta += 1
            meta = {}
        selected_row = best["row"] if best is not None else next(row for row in raw_rows if int(row["is_nearest"]))
        selected_code = int(selected_row["code_index"])
        oracle_changed = oracle_code != nearest_code
        always_best_changed = selected_code != nearest_code
        selected_is_oracle = selected_code == oracle_code
        tdist = int(meta.get("tesseract_nearest_distance", selected_row.get("tesseract_delta_vs_nearest", 0)))
        pdist = int(meta.get("parseq_nearest_distance", 0))
        ref_len = int(meta.get("reference_length", 0))
        group_rows.append(
            {
                "group_key": list(group_key),
                "partition": "val" if int(group_key[1]) == val_seed else "train",
                "source": str(meta.get("source", "unknown")),
                "reference_length": ref_len,
                "reference_length_bin": len_bin(ref_len),
                "tesseract_nearest_distance": tdist,
                "tesseract_distance_bin": dist_bin(tdist),
                "parseq_nearest_distance": pdist,
                "parseq_distance_bin": dist_bin(pdist),
                "oracle_changed": int(oracle_changed),
                "oracle_code": oracle_code,
                "nearest_code": nearest_code,
                "best_change_code": int(best["code"]) if best is not None else None,
                "best_change_score": float(best["score_mean"]) if best is not None else None,
                "best_change_margin": float(best["margin_mean"]) if best is not None else None,
                "group_change_prob_mean": float(best["group_prob_mean"]) if best is not None else None,
                "always_best_changed": int(always_best_changed),
                "always_best_is_oracle": int(selected_is_oracle),
                "always_best_error": error_type(always_best_changed, selected_is_oracle, oracle_changed),
                "always_best_parseq_delta": int(selected_row["parseq_delta_vs_nearest"]),
                "always_best_tesseract_delta": int(selected_row["tesseract_delta_vs_nearest"]),
            }
        )
    partitions = {name: [row for row in group_rows if row["partition"] == name] for name in ["train", "val"]}
    output = {
        "score": str(score_path),
        "missing_group_meta": int(missing_meta),
        "overall": summarize_groups(group_rows),
        "by_partition": {name: summarize_groups(rows) for name, rows in partitions.items()},
        "by_partition_and_oracle_changed": {
            name: grouped_summary(rows, "oracle_changed") for name, rows in partitions.items()
        },
        "by_partition_and_source": {
            name: grouped_summary(rows, "source") for name, rows in partitions.items()
        },
        "by_partition_and_reference_length_bin": {
            name: grouped_summary(rows, "reference_length_bin") for name, rows in partitions.items()
        },
        "by_partition_and_tesseract_distance_bin": {
            name: grouped_summary(rows, "tesseract_distance_bin") for name, rows in partitions.items()
        },
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--table", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    args = parser.parse_args()

    table_by_code, group_meta = table_lookup(args.table)
    audits = [audit_score_file(path, table_by_code, group_meta, val_seed=int(args.val_seed)) for path in args.score]
    output = {
        "description": "Score calibration gap audit: join score JSONL with assignment tables and summarize always-best score behavior by partition/source/difficulty.",
        "scores": [str(path) for path in args.score],
        "tables": [str(path) for path in args.table],
        "config": {"val_seed": int(args.val_seed), "table_code_rows": len(table_by_code), "table_groups": len(group_meta)},
        "audits": audits,
        "aggregate": {
            Path(audit["score"]).stem: {
                "train_always_best_tesseract_delta_sum": audit["by_partition"]["train"]["always_best_tesseract_delta_sum"],
                "train_always_best_false_change_groups": audit["by_partition"]["train"]["always_best_false_change_groups"],
                "val_always_best_tesseract_delta_sum": audit["by_partition"]["val"]["always_best_tesseract_delta_sum"],
                "val_always_best_false_change_groups": audit["by_partition"]["val"]["always_best_false_change_groups"],
                "val_oracle_change_groups": audit["by_partition"]["val"]["oracle_change_groups"],
                "missing_group_meta": audit["missing_group_meta"],
            }
            for audit in audits
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
