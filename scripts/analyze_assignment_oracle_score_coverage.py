#!/usr/bin/env python3
"""Analyze whether assignment oracle misses come from scorer ranking or gating."""

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


def code_key(row: dict) -> tuple[int, int, int, int, int]:
    return (*group_key(row), int(row["code_index"]))


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"groups": 0}
    return {
        "groups": len(rows),
        "oracle_change_groups": sum(int(row["oracle_changed"]) for row in rows),
        "policy_changed_groups": sum(int(row["policy_changed"]) for row in rows),
        "policy_exact_oracle_groups": sum(int(row["policy_code"] == row["oracle_code"]) for row in rows),
        "best_score_exact_oracle_groups": sum(int(row["best_score_code"] == row["oracle_code"]) for row in rows),
        "best_score_changed_groups": sum(int(row["best_score_code"] != row["nearest_code"]) for row in rows),
        "policy_tesseract_delta_sum": sum(int(row["policy_tesseract_delta"]) for row in rows),
        "best_score_tesseract_delta_sum": sum(int(row["best_score_tesseract_delta"]) for row in rows),
        "oracle_tesseract_delta_sum": sum(int(row["oracle_tesseract_delta"]) for row in rows),
        "policy_missed_oracle_groups": sum(int(row["policy_error_type"] == "missed_oracle") for row in rows),
        "policy_false_change_groups": sum(int(row["policy_error_type"] == "false_change") for row in rows),
        "policy_wrong_change_groups": sum(int(row["policy_error_type"] == "wrong_change") for row in rows),
        "oracle_score_rank_mean": sum(float(row["oracle_score_rank"]) for row in rows) / len(rows),
        "oracle_score_rank_le_1": sum(int(row["oracle_score_rank"] <= 1) for row in rows),
        "oracle_score_rank_le_2": sum(int(row["oracle_score_rank"] <= 2) for row in rows),
        "oracle_score_rank_le_4": sum(int(row["oracle_score_rank"] <= 4) for row in rows),
    }


def policy_error_type(policy_code: int, nearest_code: int, oracle_code: int) -> str:
    if policy_code == oracle_code and oracle_code != nearest_code:
        return "exact_oracle"
    if policy_code == oracle_code and oracle_code == nearest_code:
        return "correct_nearest"
    if policy_code != nearest_code and oracle_code == nearest_code:
        return "false_change"
    if policy_code == nearest_code and oracle_code != nearest_code:
        return "missed_oracle"
    return "wrong_change"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, action="append", required=True)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
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

    score_by_code: dict[tuple[int, int, int, int, int], dict] = {}
    for path in args.score:
        for row in read_jsonl(path):
            if int(row["val_seed"]) != args.val_seed:
                continue
            if int(row["model_seed"]) != args.assignment_model_seed:
                continue
            if int(row["seed"]) != args.val_seed:
                continue
            score_by_code[code_key(row)] = row

    policy_by_group: dict[tuple[int, int, int, int], dict] = {}
    for path in args.policy:
        for row in read_jsonl(path):
            policy_by_group[group_key(row)] = row

    records = []
    missing_scores = []
    for key, group in sorted(groups.items()):
        nearest = next(row for row in group if int(row["is_nearest"]))
        oracle = next(row for row in group if int(row["label_assignment_oracle_choice"]))
        policy = policy_by_group.get(key)
        if policy is None:
            raise SystemExit(f"missing policy for group {key}")
        scored = []
        for row in group:
            score_row = score_by_code.get((*key, int(row["code_index"])))
            if score_row is None:
                missing_scores.append([*key, int(row["code_index"])])
                score = float("-inf")
            else:
                score = float(score_row["score"])
            scored.append((score, int(row["topk_rank"]), row))
        ranked = sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)
        best = ranked[0][2]
        rank_by_code = {int(row["code_index"]): rank + 1 for rank, (_, _, row) in enumerate(ranked)}
        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        policy_code = int(policy.get("assignment_code_index", policy.get("code_index")))
        policy_row = by_code[(*key, policy_code)]
        records.append(
            {
                "real_seed": key[0],
                "seed": key[1],
                "source_index": key[2],
                "candidate_index": key[3],
                "source": str(nearest.get("source", "")),
                "reference": str(nearest.get("reference", "")),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "policy_code": policy_code,
                "best_score_code": int(best["code_index"]),
                "oracle_changed": int(oracle_code != nearest_code),
                "policy_changed": int(policy_code != nearest_code),
                "best_score_changed": int(int(best["code_index"]) != nearest_code),
                "oracle_score_rank": int(rank_by_code[oracle_code]),
                "nearest_score_rank": int(rank_by_code[nearest_code]),
                "policy_score_rank": int(rank_by_code[policy_code]),
                "best_score_tesseract_delta": int(best["tesseract_delta_vs_nearest"]),
                "best_score_parseq_delta": int(best["parseq_delta_vs_nearest"]),
                "policy_tesseract_delta": int(policy_row["tesseract_delta_vs_nearest"]),
                "policy_parseq_delta": int(policy_row["parseq_delta_vs_nearest"]),
                "oracle_tesseract_delta": int(oracle["tesseract_delta_vs_nearest"]),
                "oracle_parseq_delta": int(oracle["parseq_delta_vs_nearest"]),
                "policy_error_type": policy_error_type(policy_code, nearest_code, oracle_code),
                "best_score_error_type": policy_error_type(int(best["code_index"]), nearest_code, oracle_code),
            }
        )

    oracle_change = [row for row in records if row["oracle_changed"]]
    missed = [row for row in records if row["policy_error_type"] == "missed_oracle"]
    missed_scorer_has_oracle_best = [row for row in missed if row["best_score_code"] == row["oracle_code"]]
    missed_scorer_not_oracle_best = [row for row in missed if row["best_score_code"] != row["oracle_code"]]
    strata = {
        "all": records,
        "oracle_change": oracle_change,
        "policy_missed_oracle": missed,
        "missed_because_gate_rejected_best_oracle": missed_scorer_has_oracle_best,
        "missed_because_scorer_ranked_non_oracle_best": missed_scorer_not_oracle_best,
        "policy_exact_oracle": [row for row in records if row["policy_error_type"] == "exact_oracle"],
        "policy_bad_change": [
            row for row in records if row["policy_error_type"] in {"false_change", "wrong_change"}
        ],
    }
    output = {
        "description": "Coverage analysis separating assignment scorer ranking limits from group-gate no-op decisions.",
        "tables": [str(path) for path in args.table],
        "scores": [str(path) for path in args.score],
        "policies": [str(path) for path in args.policy],
        "config": {
            "val_seed": args.val_seed,
            "assignment_model_seed": args.assignment_model_seed,
        },
        "missing_scores": missing_scores,
        "overall": summarize_rows(records),
        "strata": {name: summarize_rows(rows) for name, rows in strata.items()},
        "policy_error_type_counts": dict(Counter(row["policy_error_type"] for row in records)),
        "best_score_error_type_counts": dict(Counter(row["best_score_error_type"] for row in records)),
        "oracle_score_rank_counts_for_oracle_changes": dict(
            Counter(str(row["oracle_score_rank"]) for row in oracle_change)
        ),
        "examples": {
            name: sorted(
                rows,
                key=lambda row: (
                    row["oracle_score_rank"],
                    row["policy_tesseract_delta"] - row["oracle_tesseract_delta"],
                ),
            )[:20]
            for name, rows in strata.items()
            if name != "all"
        },
    }
    output["aggregate"] = {
        "overall": output["overall"],
        "strata": output["strata"],
        "policy_error_type_counts": output["policy_error_type_counts"],
        "best_score_error_type_counts": output["best_score_error_type_counts"],
        "oracle_score_rank_counts_for_oracle_changes": output["oracle_score_rank_counts_for_oracle_changes"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
