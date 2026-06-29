#!/usr/bin/env python3
"""Analyze group-level errors for code-assignment score policies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bucket_distance(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def bucket_rank(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    return "5-7"


def group_key(row: dict) -> tuple[int, int, int, int]:
    real_seed = int(row.get("real_seed", row.get("original_seed", 0)))
    return (real_seed, int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))


def summarize_rows(groups: dict[str, list[dict]]) -> dict:
    out = {}
    for name, rows in groups.items():
        if not rows:
            out[name] = {"groups": 0}
            continue
        parseq_delta = sum(int(row["selected_parseq_delta"]) for row in rows)
        tesseract_delta = sum(int(row["selected_tesseract_delta"]) for row in rows)
        out[name] = {
            "groups": len(rows),
            "oracle_change_groups": sum(int(row["oracle_changed"]) for row in rows),
            "predicted_change_groups": sum(int(row["predicted_changed"]) for row in rows),
            "selected_parseq_delta_sum": parseq_delta,
            "selected_tesseract_delta_sum": tesseract_delta,
            "missed_oracle_groups": sum(int(row["error_type"] == "missed_oracle") for row in rows),
            "false_change_groups": sum(int(row["error_type"] == "false_change") for row in rows),
            "wrong_change_groups": sum(int(row["error_type"] == "wrong_change") for row in rows),
            "exact_oracle_groups": sum(int(row["error_type"] == "exact_oracle") for row in rows),
            "correct_nearest_groups": sum(int(row["error_type"] == "correct_nearest") for row in rows),
        }
    return out


def counter_to_dict(counter: Counter) -> dict:
    return {str(key): int(value) for key, value in counter.most_common()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, action="append", required=True)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()

    table_by_variant: dict[tuple[int, int, int, int, int], dict] = {}
    for table_path in args.table:
        rows = read_jsonl(table_path)
        for row in rows:
            key = (*group_key(row), int(row["code_index"]))
            table_by_variant[key] = row

    scores_by_key: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    score_rows = []
    for score_path in args.score:
        for score_row in read_jsonl(score_path):
            if int(score_row.get("model_seed", -1)) != args.model_seed:
                continue
            if int(score_row.get("seed", -1)) != args.val_seed:
                continue
            score_rows.append(score_row)
            variant_key = (
                int(score_row.get("real_seed", score_row.get("seed", 0))),
                int(score_row["seed"]),
                int(score_row["source_index"]),
                int(score_row["candidate_index"]),
                int(score_row["code_index"]),
            )
            score_row["_table_row"] = table_by_variant.get(variant_key, {})
            scores_by_key[
                (
                    int(score_row.get("real_seed", score_row.get("seed", 0))),
                    int(score_row["seed"]),
                    int(score_row["source_index"]),
                    int(score_row["candidate_index"]),
                )
            ].append(score_row)

    group_records = []
    for key, group_scores in sorted(scores_by_key.items()):
        best_score = max(group_scores, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))
        nearest = next(row for row in group_scores if int(row["is_nearest"]))
        oracle = next(row for row in group_scores if int(row["label_assignment_oracle_choice"]))
        best_table = best_score.get("_table_row", {})
        nearest_table = nearest.get("_table_row", {})
        oracle_table = oracle.get("_table_row", {})
        selected_changed = int(best_score["code_index"]) != int(nearest["code_index"])
        oracle_changed = int(oracle["code_index"]) != int(nearest["code_index"])
        selected_is_oracle = int(best_score["code_index"]) == int(oracle["code_index"])
        if selected_is_oracle and oracle_changed:
            error_type = "exact_oracle"
        elif selected_is_oracle and not oracle_changed:
            error_type = "correct_nearest"
        elif selected_changed and not oracle_changed:
            error_type = "false_change"
        elif (not selected_changed) and oracle_changed:
            error_type = "missed_oracle"
        else:
            error_type = "wrong_change"
        record = {
            "real_seed": key[0],
            "seed": key[1],
            "source_index": key[2],
            "candidate_index": key[3],
            "source": str(best_table.get("source", nearest_table.get("source", ""))),
            "source_image": best_score.get("source_image", nearest.get("source_image", "")),
            "reference": str(best_table.get("reference", nearest_table.get("reference", ""))),
            "nearest_code": int(nearest["code_index"]),
            "selected_code": int(best_score["code_index"]),
            "oracle_code": int(oracle["code_index"]),
            "selected_topk_rank": int(best_score["topk_rank"]),
            "oracle_topk_rank": int(oracle["topk_rank"]),
            "selected_score": float(best_score["score"]),
            "nearest_score": float(nearest["score"]),
            "score_margin_vs_nearest": float(best_score["score"]) - float(nearest["score"]),
            "predicted_changed": int(selected_changed),
            "oracle_changed": int(oracle_changed),
            "selected_parseq_delta": int(best_score["parseq_delta_vs_nearest"]),
            "selected_tesseract_delta": int(best_score["tesseract_delta_vs_nearest"]),
            "oracle_parseq_delta": int(oracle["parseq_delta_vs_nearest"]),
            "oracle_tesseract_delta": int(oracle["tesseract_delta_vs_nearest"]),
            "nearest_tesseract_distance": int(nearest_table.get("tesseract_nearest_distance", 0)),
            "nearest_parseq_distance": int(nearest_table.get("parseq_nearest_distance", 0)),
            "error_type": error_type,
        }
        group_records.append(record)

    strata = {
        "by_error_type": defaultdict(list),
        "by_real_seed": defaultdict(list),
        "by_source": defaultdict(list),
        "by_nearest_tesseract_distance": defaultdict(list),
        "by_oracle_rank_bucket": defaultdict(list),
    }
    for row in group_records:
        strata["by_error_type"][row["error_type"]].append(row)
        strata["by_real_seed"][str(row["real_seed"])].append(row)
        strata["by_source"][row["source"] or "unknown"].append(row)
        strata["by_nearest_tesseract_distance"][bucket_distance(row["nearest_tesseract_distance"])].append(row)
        strata["by_oracle_rank_bucket"][bucket_rank(row["oracle_topk_rank"])].append(row)

    missed = [row for row in group_records if row["error_type"] == "missed_oracle"]
    false_changes = [row for row in group_records if row["error_type"] == "false_change"]
    wrong_changes = [row for row in group_records if row["error_type"] == "wrong_change"]

    output = {
        "description": "Group-level error analysis for score-based top-k assignment policies.",
        "tables": [str(path) for path in args.table],
        "scores": [str(path) for path in args.score],
        "config": {"val_seed": args.val_seed, "model_seed": args.model_seed},
        "groups": len(group_records),
        "overall": summarize_rows({"all": group_records})["all"],
        "strata": {name: summarize_rows(dict(values)) for name, values in strata.items()},
        "error_type_counts": counter_to_dict(Counter(row["error_type"] for row in group_records)),
        "missed_oracle_by_source": counter_to_dict(Counter(row["source"] or "unknown" for row in missed)),
        "false_change_by_source": counter_to_dict(Counter(row["source"] or "unknown" for row in false_changes)),
        "wrong_change_by_source": counter_to_dict(Counter(row["source"] or "unknown" for row in wrong_changes)),
        "examples": {
            "missed_oracle": sorted(missed, key=lambda row: row["oracle_tesseract_delta"])[:20],
            "false_change": sorted(false_changes, key=lambda row: -row["selected_tesseract_delta"])[:20],
            "wrong_change": sorted(wrong_changes, key=lambda row: -row["selected_tesseract_delta"])[:20],
        },
    }
    output["aggregate"] = {
        "overall": output["overall"],
        "strata": output["strata"],
        "error_type_counts": output["error_type_counts"],
        "missed_oracle_by_source": output["missed_oracle_by_source"],
        "false_change_by_source": output["false_change_by_source"],
        "wrong_change_by_source": output["wrong_change_by_source"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["overall"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
