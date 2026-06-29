#!/usr/bin/env python3
"""Analyze group-level errors for an explicit code-assignment policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


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


def summarize_rows(groups: dict[str, list[dict]]) -> dict:
    out = {}
    for name, rows in groups.items():
        if not rows:
            out[name] = {"groups": 0}
            continue
        out[name] = {
            "groups": len(rows),
            "oracle_change_groups": sum(int(row["oracle_changed"]) for row in rows),
            "policy_change_groups": sum(int(row["policy_changed"]) for row in rows),
            "policy_parseq_delta_sum": sum(int(row["policy_parseq_delta"]) for row in rows),
            "policy_tesseract_delta_sum": sum(int(row["policy_tesseract_delta"]) for row in rows),
            "policy_tesseract_improve_groups": sum(int(row["policy_tesseract_delta"] < 0) for row in rows),
            "policy_tesseract_worsen_groups": sum(int(row["policy_tesseract_delta"] > 0) for row in rows),
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
    parser.add_argument("--policy", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    args = parser.parse_args()

    table_by_group: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    table_by_variant: dict[tuple[int, int, int, int, int], dict] = {}
    for table_path in args.table:
        for row in read_jsonl(table_path):
            if int(row["seed"]) != args.val_seed:
                continue
            key = group_key(row)
            table_by_group[key].append(row)
            table_by_variant[(*key, int(row["code_index"]))] = row

    policy_by_group = {}
    for policy_path in args.policy:
        for row in read_jsonl(policy_path):
            key = group_key(row)
            policy_by_group[key] = row

    records = []
    missing_policy = []
    for key, group in sorted(table_by_group.items()):
        nearest = next(row for row in group if int(row["is_nearest"]))
        oracle = next(row for row in group if int(row["label_assignment_oracle_choice"]))
        policy = policy_by_group.get(key)
        if policy is None:
            missing_policy.append(key)
            continue
        selected_code = int(policy.get("assignment_code_index", policy.get("code_index")))
        selected = table_by_variant.get((*key, selected_code))
        if selected is None:
            raise SystemExit(f"policy selected unknown code {selected_code} for group {key}")

        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        policy_changed = selected_code != nearest_code
        oracle_changed = oracle_code != nearest_code
        selected_is_oracle = selected_code == oracle_code
        if selected_is_oracle and oracle_changed:
            error_type = "exact_oracle"
        elif selected_is_oracle and not oracle_changed:
            error_type = "correct_nearest"
        elif policy_changed and not oracle_changed:
            error_type = "false_change"
        elif (not policy_changed) and oracle_changed:
            error_type = "missed_oracle"
        else:
            error_type = "wrong_change"
        records.append(
            {
                "real_seed": key[0],
                "seed": key[1],
                "source_index": key[2],
                "candidate_index": key[3],
                "source": str(nearest.get("source", "")),
                "source_image": nearest.get("source_image", ""),
                "reference": str(nearest.get("reference", "")),
                "nearest_code": nearest_code,
                "policy_code": selected_code,
                "oracle_code": oracle_code,
                "policy_topk_rank": int(selected["topk_rank"]),
                "oracle_topk_rank": int(oracle["topk_rank"]),
                "policy_changed": int(policy_changed),
                "oracle_changed": int(oracle_changed),
                "policy_parseq_delta": int(selected["parseq_delta_vs_nearest"]),
                "policy_tesseract_delta": int(selected["tesseract_delta_vs_nearest"]),
                "oracle_parseq_delta": int(oracle["parseq_delta_vs_nearest"]),
                "oracle_tesseract_delta": int(oracle["tesseract_delta_vs_nearest"]),
                "nearest_tesseract_distance": int(nearest.get("tesseract_nearest_distance", 0)),
                "nearest_parseq_distance": int(nearest.get("parseq_nearest_distance", 0)),
                "error_type": error_type,
            }
        )

    strata = {
        "by_error_type": defaultdict(list),
        "by_real_seed": defaultdict(list),
        "by_source": defaultdict(list),
        "by_nearest_tesseract_distance": defaultdict(list),
        "by_oracle_rank_bucket": defaultdict(list),
    }
    for row in records:
        strata["by_error_type"][row["error_type"]].append(row)
        strata["by_real_seed"][str(row["real_seed"])].append(row)
        strata["by_source"][row["source"] or "unknown"].append(row)
        strata["by_nearest_tesseract_distance"][bucket_distance(row["nearest_tesseract_distance"])].append(row)
        strata["by_oracle_rank_bucket"][bucket_rank(row["oracle_topk_rank"])].append(row)

    examples = {}
    for error_type in ["missed_oracle", "false_change", "wrong_change", "exact_oracle"]:
        subset = [row for row in records if row["error_type"] == error_type]
        examples[error_type] = sorted(
            subset,
            key=lambda row: (row["policy_tesseract_delta"] - row["oracle_tesseract_delta"], row["oracle_tesseract_delta"]),
            reverse=error_type in {"false_change", "wrong_change"},
        )[:20]

    output = {
        "description": "Group-level error analysis for explicit code-assignment policies.",
        "tables": [str(path) for path in args.table],
        "policies": [str(path) for path in args.policy],
        "config": {"val_seed": args.val_seed},
        "groups": len(records),
        "missing_policy_groups": [list(key) for key in missing_policy],
        "overall": summarize_rows({"all": records})["all"],
        "strata": {name: summarize_rows(dict(values)) for name, values in strata.items()},
        "error_type_counts": counter_to_dict(Counter(row["error_type"] for row in records)),
        "missed_oracle_by_source": counter_to_dict(
            Counter(row["source"] or "unknown" for row in records if row["error_type"] == "missed_oracle")
        ),
        "false_change_by_source": counter_to_dict(
            Counter(row["source"] or "unknown" for row in records if row["error_type"] == "false_change")
        ),
        "wrong_change_by_source": counter_to_dict(
            Counter(row["source"] or "unknown" for row in records if row["error_type"] == "wrong_change")
        ),
        "examples": examples,
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
