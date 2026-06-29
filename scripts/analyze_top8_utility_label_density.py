#!/usr/bin/env python3
"""Audit top-8 assignment utility-label density and oracle target mismatch."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def classify(selected_code: int, nearest_code: int, exact_oracle_code: int) -> str:
    if selected_code == exact_oracle_code:
        return "exact"
    if selected_code == nearest_code and exact_oracle_code != nearest_code:
        return "missed_exact_oracle"
    if selected_code != nearest_code and exact_oracle_code == nearest_code:
        return "false_change_vs_exact"
    return "different_nonnearest"


def choose_best_safe_tesseract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nearest = next(row for row in rows if int(row.get("is_nearest", 0)))
    candidates = [
        row for row in rows
        if not int(row.get("is_nearest", 0))
        and int(row.get("parseq_delta_vs_nearest", 0)) <= 0
        and int(row.get("tesseract_delta_vs_nearest", 0)) < 0
    ]
    if not candidates:
        return nearest
    return min(
        candidates,
        key=lambda row: (
            int(row.get("tesseract_delta_vs_nearest", 0)),
            int(row.get("parseq_delta_vs_nearest", 0)),
            int(row.get("topk_rank", 999)),
        ),
    )


def summarize_partition(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    counts = Counter()
    safe_counts = []
    exact_t_sum = 0
    exact_p_sum = 0
    safe_t_sum = 0
    safe_p_sum = 0
    safe_changed = 0
    safe_t_worse = 0
    exact_changed = 0
    mismatch_rows = []
    for rows in groups:
        first = rows[0]
        nearest_code = int(first["nearest_code"])
        exact_oracle_code = int(first["assignment_oracle_code_index"])
        exact_row = next(row for row in rows if int(row["code_index"]) == exact_oracle_code)
        safe_row = choose_best_safe_tesseract(rows)
        safe_improving_rows = [
            row for row in rows
            if not int(row.get("is_nearest", 0))
            and int(row.get("parseq_delta_vs_nearest", 0)) <= 0
            and int(row.get("tesseract_delta_vs_nearest", 0)) < 0
        ]
        safe_counts.append(len(safe_improving_rows))
        counts["groups"] += 1
        counts["exact_oracle_change_groups"] += int(exact_oracle_code != nearest_code)
        counts["groups_with_safe_tesseract_candidate"] += int(bool(safe_improving_rows))
        counts["groups_with_multiple_safe_tesseract_candidates"] += int(len(safe_improving_rows) > 1)
        exact_t_sum += int(exact_row.get("tesseract_delta_vs_nearest", 0)) if exact_oracle_code != nearest_code else 0
        exact_p_sum += int(exact_row.get("parseq_delta_vs_nearest", 0)) if exact_oracle_code != nearest_code else 0
        safe_changed_now = int(safe_row["code_index"]) != nearest_code
        safe_changed += int(safe_changed_now)
        safe_t = int(safe_row.get("tesseract_delta_vs_nearest", 0)) if safe_changed_now else 0
        safe_p = int(safe_row.get("parseq_delta_vs_nearest", 0)) if safe_changed_now else 0
        safe_t_sum += safe_t
        safe_p_sum += safe_p
        safe_t_worse += int(safe_t > 0)
        exact_changed += int(exact_oracle_code != nearest_code)
        status = classify(int(safe_row["code_index"]), nearest_code, exact_oracle_code)
        counts[f"safe_best_{status}"] += 1
        if status != "exact" and safe_changed_now:
            mismatch_rows.append(
                {
                    "group_key": list(group_key(first)),
                    "source": first.get("source"),
                    "reference": first.get("reference"),
                    "nearest_code": nearest_code,
                    "exact_oracle_code": exact_oracle_code,
                    "safe_best_code": int(safe_row["code_index"]),
                    "safe_best_topk_rank": int(safe_row.get("topk_rank", -1)),
                    "safe_best_tesseract_delta": safe_t,
                    "safe_best_parseq_delta": safe_p,
                    "exact_oracle_tesseract_delta": int(exact_row.get("tesseract_delta_vs_nearest", 0)),
                    "exact_oracle_parseq_delta": int(exact_row.get("parseq_delta_vs_nearest", 0)),
                    "status_vs_exact": status,
                }
            )
    return {
        **{key: int(value) for key, value in counts.items()},
        "safe_tesseract_candidate_rows": int(sum(safe_counts)),
        "safe_tesseract_candidates_per_group_mean": float(mean(safe_counts)) if safe_counts else 0.0,
        "safe_tesseract_candidates_per_group_median": float(median(safe_counts)) if safe_counts else 0.0,
        "exact_oracle_changed_groups": int(exact_changed),
        "exact_oracle_tesseract_delta_vs_nearest": int(exact_t_sum),
        "exact_oracle_parseq_delta_vs_nearest": int(exact_p_sum),
        "safe_best_changed_groups": int(safe_changed),
        "safe_best_tesseract_delta_vs_nearest": int(safe_t_sum),
        "safe_best_parseq_delta_vs_nearest": int(safe_p_sum),
        "safe_best_tesseract_worse_groups": int(safe_t_worse),
        "safe_best_mismatches": mismatch_rows[:30],
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Top-8 assignment label-density audit comparing exact assignment oracle labels with a non-deployable Tesseract/PARSeq-safe utility oracle.",
        "",
        "| partition | groups | exact changes | groups with safe T cand | safe rows | exact T | safe-best T | safe-best changed | safe-best exact | safe-best different | false vs exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for partition, summary in result["partitions"].items():
        lines.append(
            f"| {partition} | {summary['groups']} | {summary['exact_oracle_change_groups']} | "
            f"{summary['groups_with_safe_tesseract_candidate']} | {summary['safe_tesseract_candidate_rows']} | "
            f"{summary['exact_oracle_tesseract_delta_vs_nearest']} | {summary['safe_best_tesseract_delta_vs_nearest']} | "
            f"{summary['safe_best_changed_groups']} | {summary.get('safe_best_exact', 0)} | "
            f"{summary.get('safe_best_different_nonnearest', 0)} | {summary.get('safe_best_false_change_vs_exact', 0)} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.table)
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    by_partition: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group_rows in grouped.values():
        group_rows = sorted(group_rows, key=lambda row: int(row.get("topk_rank", 999)))
        by_partition[str(group_rows[0].get("assignment_partition", group_rows[0].get("split", "")))].append(group_rows)

    partitions = {partition: summarize_partition(groups) for partition, groups in sorted(by_partition.items())}
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_label_density_not_promoted",
        "inputs": {"table": {"path": str(args.table), "sha256": sha256_file(args.table)}},
        "partitions": partitions,
        "hashes": {"script": sha256_file(Path(__file__))},
        "interpretation": (
            "Exact oracle-code supervision and direct OCR-utility supervision are not the same target. "
            "The safe-best oracle is a non-deployable held-out Tesseract/PARSeq upper control, but it "
            "quantifies how often useful utility exists outside the exact assignment oracle label. Use this "
            "to design future marginal-utility targets, not as a selector or publication result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "safe_best_mismatches"} for k, v in partitions.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
