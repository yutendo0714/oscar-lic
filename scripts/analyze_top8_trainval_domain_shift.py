#!/usr/bin/env python3
"""Analyze train/val shift for the Eval300 top-8 assignment ranker."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


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


def group_key(row: dict[str, Any]) -> tuple[int, str, int, int]:
    return (int(row["real_seed"]), str(row["assignment_partition"]), int(row["source_index"]), int(row["candidate_index"]))


def group_table(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    for value in groups.values():
        value.sort(key=lambda row: (int(row["topk_rank"]), int(row["code_index"])))
    return groups


def oracle_row(group: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in group if int(row.get("label_assignment_oracle_choice", 0))), group[0])


def nearest_row(group: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in group if int(row["code_index"]) == int(row["nearest_code"]))


def bucket_topk(rank: int) -> str:
    if rank <= 1:
        return "rank_le1"
    if rank <= 2:
        return "rank_le2"
    if rank <= 4:
        return "rank_le4"
    return "rank_gt4"


def bucket_len(length: int) -> str:
    if length <= 3:
        return "len_le3"
    if length <= 6:
        return "len_4_6"
    if length <= 10:
        return "len_7_10"
    return "len_gt10"


def count_groups(groups: dict[tuple[int, str, int, int], list[dict[str, Any]]]) -> dict[str, Any]:
    by_partition_source = Counter()
    by_partition_topk = Counter()
    by_partition_length = Counter()
    by_partition_seed = Counter()
    references: dict[str, set[str]] = defaultdict(set)
    for key, rows in groups.items():
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        changed = int(oracle["code_index"]) != int(nearest["code_index"])
        partition = key[1]
        source = str(nearest.get("source", ""))
        if changed:
            by_partition_source[(partition, source)] += 1
            by_partition_topk[(partition, bucket_topk(int(oracle["topk_rank"])))] += 1
            by_partition_length[(partition, bucket_len(int(nearest.get("reference_length", len(str(nearest.get("reference", "")))))))] += 1
            by_partition_seed[(partition, int(key[0]))] += 1
            references[partition].add(str(nearest.get("reference", "")))
    overlap = references["train"] & references["val"]
    return {
        "oracle_change_by_partition_source": {"/".join(map(str, key)): value for key, value in sorted(by_partition_source.items())},
        "oracle_change_by_partition_topk_bucket": {"/".join(map(str, key)): value for key, value in sorted(by_partition_topk.items())},
        "oracle_change_by_partition_length_bucket": {"/".join(map(str, key)): value for key, value in sorted(by_partition_length.items())},
        "oracle_change_by_partition_seed": {"/".join(map(str, key)): value for key, value in sorted(by_partition_seed.items())},
        "train_oracle_change_references": len(references["train"]),
        "val_oracle_change_references": len(references["val"]),
        "reference_overlap_count": len(overlap),
        "reference_overlap": sorted(overlap),
    }


def summarize_n091_val(n091: dict[str, Any]) -> dict[str, Any]:
    by_source_status = Counter()
    by_topk_status = Counter()
    by_len_status = Counter()
    records = []
    for row in n091["group_audits"]:
        if int(row["oracle_code"]) == int(row["nearest_code"]):
            continue
        status = str(row["selected_status_oracle_change_only"])
        source = str(row.get("source", ""))
        topk_bucket = bucket_topk(int(row["oracle_topk_rank"]))
        length_bucket = bucket_len(len(str(row.get("reference", ""))))
        by_source_status[(source, status)] += 1
        by_topk_status[(topk_bucket, status)] += 1
        by_len_status[(length_bucket, status)] += 1
        records.append(
            {
                "source": source,
                "reference": row.get("reference"),
                "status": status,
                "oracle_topk_rank": int(row["oracle_topk_rank"]),
                "oracle_score_rank": int(row["oracle_nonnearest_score_rank"]),
                "best_nonnearest_code": int(row["best_nonnearest_code"]),
                "oracle_code": int(row["oracle_code"]),
                "nearest_code": int(row["nearest_code"]),
                "best_tesseract_delta": int(row["selected_tesseract_delta_vs_nearest"]),
            }
        )
    return {
        "by_source_status": {"/".join(map(str, key)): value for key, value in sorted(by_source_status.items())},
        "by_topk_status": {"/".join(map(str, key)): value for key, value in sorted(by_topk_status.items())},
        "by_length_status": {"/".join(map(str, key)): value for key, value in sorted(by_len_status.items())},
        "records": records,
    }


def summarize_n092_false_changes(n092: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for policy_name, policy in n092["policies"].items():
        by_source = Counter()
        by_status = Counter()
        for row in policy.get("val_audits", []):
            status = str(row["selected_status"])
            if status in {"exact", "missed_oracle"}:
                continue
            by_source[(str(row.get("source", "")), status)] += 1
            by_status[status] += 1
        out[policy_name] = {
            "by_status": dict(sorted(by_status.items())),
            "by_source_status": {"/".join(map(str, key)): value for key, value in sorted(by_source.items())},
        }
    return out


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Train/val shift diagnostic for the Eval300 top-8 tabular assignment path.",
        "",
        "## Oracle-Change Distribution",
        "",
        "### Source",
        "",
    ]
    for key, value in result["table_shift"]["oracle_change_by_partition_source"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "### Top-K Bucket", ""])
    for key, value in result["table_shift"]["oracle_change_by_partition_topk_bucket"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## N091 Val Status",
            "",
            "### By Source",
            "",
        ]
    )
    for key, value in result["n091_val"]["by_source_status"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "### By Oracle Top-K Bucket", ""])
    for key, value in result["n091_val"]["by_topk_status"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Reference Overlap",
            "",
            f"- train oracle-change unique references: `{result['table_shift']['train_oracle_change_references']}`",
            f"- val oracle-change unique references: `{result['table_shift']['val_oracle_change_references']}`",
            f"- overlap count: `{result['table_shift']['reference_overlap_count']}`",
            "",
            "## Interpretation",
            "",
            "- The diagnostic is descriptive: it explains N091-N093 behavior but is not a selector.",
            "- If failures cluster by source or deep top-k rank, next evidence should target that stratum directly rather than sweeping thresholds.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--n091", type=Path, required=True)
    parser.add_argument("--n092", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_trainval_domain_shift")
    args = parser.parse_args()

    table_rows = read_jsonl(args.table)
    table_groups = group_table(table_rows)
    n091 = read_json(args.n091)
    n092 = read_json(args.n092)
    table_shift = count_groups(table_groups)
    n091_val = summarize_n091_val(n091)
    n092_false = summarize_n092_false_changes(n092)
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_domain_shift_analysis_not_selector",
        "inputs": {
            "table": str(args.table),
            "table_sha256": sha256_file(args.table),
            "n091_result": str(args.n091),
            "n091_result_sha256": sha256_file(args.n091),
            "n092_result": str(args.n092),
            "n092_result_sha256": sha256_file(args.n092),
        },
        "table_shift": table_shift,
        "n091_val": n091_val,
        "n092_false_changes": n092_false,
        "aggregate": {
            "scalar_metrics": {
                "n091_val_exact_changes": {"value": float(sum(1 for row in n091_val["records"] if row["status"] == "exact"))},
                "n091_val_wrong_changes": {"value": float(sum(1 for row in n091_val["records"] if row["status"] == "wrong_change"))},
                "reference_overlap_count": {"value": float(table_shift["reference_overlap_count"])},
            }
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
