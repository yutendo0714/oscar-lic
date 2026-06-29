#!/usr/bin/env python3
"""Stratify code-assignment oracle headroom by source and OCR difficulty."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Callable

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_seed(row: dict) -> int:
    return int(row.get("real_seed", row.get("original_seed", row.get("seed", 0))))


def group_key(row: dict) -> tuple[int, int, int]:
    return (row_seed(row), int(row["source_index"]), int(row["candidate_index"]))


def group_rows(rows: list[dict]) -> dict[tuple[int, int, int], list[dict]]:
    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return dict(groups)


def nearest(group: list[dict]) -> dict:
    matches = [row for row in group if int(row["is_nearest"])]
    if len(matches) != 1:
        raise SystemExit(f"group has {len(matches)} nearest rows")
    return matches[0]


def oracle(group: list[dict]) -> dict:
    matches = [row for row in group if int(row["label_assignment_oracle_choice"])]
    if len(matches) != 1:
        raise SystemExit(f"group has {len(matches)} oracle-choice rows")
    return matches[0]


def length_bin(value: int) -> str:
    if value <= 3:
        return "len_01_03"
    if value <= 6:
        return "len_04_06"
    if value <= 10:
        return "len_07_10"
    return "len_11_plus"


def distance_bin(value: int) -> str:
    if value <= 0:
        return "dist_0"
    if value == 1:
        return "dist_1"
    if value == 2:
        return "dist_2"
    return "dist_3_plus"


def relerr_bin(value: float) -> str:
    if value <= 1.001:
        return "rel_1.000"
    if value <= 1.01:
        return "rel_1.001_1.010"
    if value <= 1.05:
        return "rel_1.010_1.050"
    if value <= 1.20:
        return "rel_1.050_1.200"
    return "rel_gt_1.200"


def group_record(group: list[dict]) -> dict:
    base = nearest(group)
    best = oracle(group)
    changed = int(not int(best["is_nearest"]))
    return {
        "real_seed": row_seed(base),
        "source": str(base.get("source", "unknown")),
        "reference_length": int(base.get("reference_length", 0)),
        "reference_length_bin": length_bin(int(base.get("reference_length", 0))),
        "nearest_tesseract_distance": int(base["tesseract_nearest_distance"]),
        "nearest_tesseract_distance_bin": distance_bin(int(base["tesseract_nearest_distance"])),
        "nearest_parseq_distance": int(base["parseq_nearest_distance"]),
        "nearest_parseq_distance_bin": distance_bin(int(base["parseq_nearest_distance"])),
        "oracle_changed": changed,
        "oracle_topk_rank": int(best["topk_rank"]),
        "oracle_relative_error": float(best["assignment_relative_error"]),
        "oracle_relative_error_bin": relerr_bin(float(best["assignment_relative_error"])),
        "oracle_tesseract_delta": int(best["tesseract_delta_vs_nearest"]),
        "oracle_parseq_delta": int(best["parseq_delta_vs_nearest"]),
        "group_has_safe_tesseract_gain": int(base.get("group_has_safe_tesseract_gain", changed)),
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        return {
            "groups": 0,
            "changed_groups": 0,
            "safe_gain_groups": 0,
            "tesseract_delta_sum": 0,
            "parseq_delta_sum": 0,
        }
    changed = [row for row in records if row["oracle_changed"]]
    rel_errors = [row["oracle_relative_error"] for row in changed]
    return {
        "groups": len(records),
        "changed_groups": int(sum(row["oracle_changed"] for row in records)),
        "changed_fraction": float(np.mean([row["oracle_changed"] for row in records])),
        "safe_gain_groups": int(sum(row["group_has_safe_tesseract_gain"] for row in records)),
        "safe_gain_fraction": float(np.mean([row["group_has_safe_tesseract_gain"] for row in records])),
        "tesseract_delta_sum": int(sum(row["oracle_tesseract_delta"] for row in records)),
        "parseq_delta_sum": int(sum(row["oracle_parseq_delta"] for row in records)),
        "mean_tesseract_delta": float(np.mean([row["oracle_tesseract_delta"] for row in records])),
        "mean_parseq_delta": float(np.mean([row["oracle_parseq_delta"] for row in records])),
        "mean_reference_length": float(np.mean([row["reference_length"] for row in records])),
        "mean_changed_relative_error": float(np.mean(rel_errors)) if rel_errors else None,
        "oracle_topk_rank_counts": dict(sorted(Counter(str(row["oracle_topk_rank"]) for row in records).items())),
        "oracle_relative_error_bin_counts": dict(sorted(Counter(row["oracle_relative_error_bin"] for row in changed).items())),
    }


def summarize_by(records: list[dict], key_fn: Callable[[dict], str]) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[key_fn(record)].append(record)
    return {key: summarize(value) for key, value in sorted(buckets.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=["all", "train", "val"], default="val")
    args = parser.parse_args()

    rows = read_jsonl(args.table)
    if args.partition != "all":
        rows = [row for row in rows if str(row.get("assignment_partition", row.get("split", ""))) == args.partition]
    groups = group_rows(rows)
    records = [group_record(group) for group in groups.values()]
    overall = summarize(records)
    by_real_seed = summarize_by(records, lambda row: str(row["real_seed"]))
    by_source = summarize_by(records, lambda row: row["source"])
    by_reference_length_bin = summarize_by(records, lambda row: row["reference_length_bin"])
    by_nearest_tesseract_distance_bin = summarize_by(records, lambda row: row["nearest_tesseract_distance_bin"])
    by_nearest_parseq_distance_bin = summarize_by(records, lambda row: row["nearest_parseq_distance_bin"])
    by_source_and_tesseract_distance = summarize_by(
        records,
        lambda row: f"{row['source']}|{row['nearest_tesseract_distance_bin']}",
    )
    output = {
        "description": "Stratified top-k code-assignment oracle headroom.",
        "table": str(args.table),
        "partition": args.partition,
        "rows": len(rows),
        "groups": len(groups),
        "overall": overall,
        "by_real_seed": by_real_seed,
        "by_source": by_source,
        "by_reference_length_bin": by_reference_length_bin,
        "by_nearest_tesseract_distance_bin": by_nearest_tesseract_distance_bin,
        "by_nearest_parseq_distance_bin": by_nearest_parseq_distance_bin,
        "by_source_and_tesseract_distance": by_source_and_tesseract_distance,
        "aggregate": {
            "overall": overall,
            "by_real_seed": by_real_seed,
            "by_source": by_source,
            "by_reference_length_bin": by_reference_length_bin,
            "by_nearest_tesseract_distance_bin": by_nearest_tesseract_distance_bin,
            "by_nearest_parseq_distance_bin": by_nearest_parseq_distance_bin,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "groups": len(groups), "overall": output["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
