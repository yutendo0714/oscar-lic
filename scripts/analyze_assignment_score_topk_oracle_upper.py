#!/usr/bin/env python3
"""Upper-control analysis for oracle recovery within scorer top-M candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def code_key(row: dict) -> tuple[int, int, int, int, int]:
    return (*group_key(row), int(row["code_index"]))


def policy_metrics(selected: list[dict], nearest_by_group: dict[tuple[int, int, int, int], dict]) -> dict:
    changed = 0
    parseq_delta = 0
    tesseract_delta = 0
    for row in selected:
        key = group_key(row)
        nearest_code = int(nearest_by_group[key]["code_index"])
        changed += int(int(row["code_index"]) != nearest_code)
        parseq_delta += int(row["parseq_delta_vs_nearest"])
        tesseract_delta += int(row["tesseract_delta_vs_nearest"])
    return {
        "groups": len(selected),
        "changed_groups": changed,
        "parseq_delta_sum": parseq_delta,
        "tesseract_delta_sum": tesseract_delta,
        "parseq_worsen_groups": sum(int(row["parseq_delta_vs_nearest"]) > 0 for row in selected),
        "tesseract_improve_groups": sum(int(row["tesseract_delta_vs_nearest"]) < 0 for row in selected),
        "tesseract_worsen_groups": sum(int(row["tesseract_delta_vs_nearest"]) > 0 for row in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, action="append", required=True)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-m", type=int, action="append", default=[1, 2, 4, 8])
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--assignment-model-seed", type=int, default=0)
    args = parser.parse_args()

    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    by_code: dict[tuple[int, int, int, int, int], dict] = {}
    for path in args.table:
        for row in read_jsonl(path):
            if int(row["seed"]) != args.val_seed:
                continue
            key = group_key(row)
            groups[key].append(row)
            by_code[(*key, int(row["code_index"]))] = row

    score_by_code: dict[tuple[int, int, int, int, int], float] = {}
    for path in args.score:
        for row in read_jsonl(path):
            if int(row["val_seed"]) != args.val_seed:
                continue
            if int(row["model_seed"]) != args.assignment_model_seed:
                continue
            if int(row["seed"]) != args.val_seed:
                continue
            score_by_code[code_key(row)] = float(row["score"])

    nearest_by_group = {}
    oracle_by_group = {}
    rank_by_group_code = {}
    for key, rows in groups.items():
        nearest_by_group[key] = next(row for row in rows if int(row["is_nearest"]))
        oracle_by_group[key] = next(row for row in rows if int(row["label_assignment_oracle_choice"]))
        ranked = sorted(
            rows,
            key=lambda row: (
                score_by_code.get((*key, int(row["code_index"])), float("-inf")),
                -int(row["topk_rank"]),
            ),
            reverse=True,
        )
        rank_by_group_code[key] = {int(row["code_index"]): rank + 1 for rank, row in enumerate(ranked)}

    policies = {
        "nearest": [nearest_by_group[key] for key in sorted(groups)],
        "oracle": [oracle_by_group[key] for key in sorted(groups)],
    }
    for top_m in sorted(set(args.top_m)):
        selected = []
        selected_oracle_count = 0
        oracle_change_recovered = 0
        oracle_change_total = 0
        for key in sorted(groups):
            nearest = nearest_by_group[key]
            oracle = oracle_by_group[key]
            oracle_rank = rank_by_group_code[key][int(oracle["code_index"])]
            oracle_changed = int(oracle["code_index"]) != int(nearest["code_index"])
            oracle_change_total += int(oracle_changed)
            if oracle_rank <= top_m:
                selected.append(oracle)
                selected_oracle_count += 1
                oracle_change_recovered += int(oracle_changed)
            else:
                selected.append(nearest)
        name = f"oracle_if_score_top_{top_m}"
        metrics = policy_metrics(selected, nearest_by_group)
        metrics.update(
            oracle_selected_groups=selected_oracle_count,
            oracle_change_recovered_groups=oracle_change_recovered,
            oracle_change_groups=oracle_change_total,
        )
        policies[name] = selected
        policies[name + "_metrics"] = metrics

    policy_metric_map = {}
    for name, rows in policies.items():
        if name.endswith("_metrics"):
            continue
        policy_metric_map[name] = policy_metrics(rows, nearest_by_group)
    for name, value in policies.items():
        if name.endswith("_metrics"):
            policy_metric_map[name.removesuffix("_metrics")] = value

    output = {
        "description": "Non-deployable upper control: select the oracle code only when it is within scorer top-M, otherwise keep nearest.",
        "tables": [str(path) for path in args.table],
        "scores": [str(path) for path in args.score],
        "config": {
            "val_seed": args.val_seed,
            "assignment_model_seed": args.assignment_model_seed,
            "top_m": sorted(set(args.top_m)),
        },
        "policy_metrics": policy_metric_map,
        "aggregate": {"policy_metrics": policy_metric_map},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(policy_metric_map, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
