#!/usr/bin/env python3
"""Build small explicit multi-candidate policies for interaction smoke tests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def candidate_score(row: dict) -> tuple[int, int, int]:
    return (
        int(row.get("multi_teacher_delta_distance", 0)),
        int(row.get("tesseract_delta_distance", 0)),
        int(row.get("candidate_index", 0)),
    )


def policy_row(source_index: int, source_row: dict, candidate_row: dict, policy: str) -> dict:
    return {
        "source_index": source_index,
        "original_source_index": int(source_row["source_index"]),
        "candidate_index": int(candidate_row["candidate_index"]),
        "selected_by_policy": True,
        "policy": policy,
        "expected_parseq_delta_distance": int(candidate_row.get("parseq_delta_distance", 0)),
        "expected_tesseract_delta_distance": int(candidate_row.get("tesseract_delta_distance", 0)),
        "expected_multi_teacher_delta_distance": int(candidate_row.get("multi_teacher_delta_distance", 0)),
        "source_image": source_row.get("source_image"),
        "reference": source_row.get("reference"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--output-split", type=Path, required=True)
    parser.add_argument("--output-policy-prefix", type=Path, required=True)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--require-no-evaluator-worsens", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    split_rows = read_jsonl(args.source_split)
    candidates_by_source: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.candidate_table):
        if int(row.get("multi_teacher_delta_distance", 0)) >= 0:
            continue
        if args.require_no_evaluator_worsens and not bool(int(row.get("label_no_evaluator_worsens", 0))):
            continue
        candidates_by_source[int(row["source_index"])].append(row)

    selected_sources = []
    for source_index, rows in sorted(candidates_by_source.items()):
        distinct: dict[int, dict] = {}
        for row in sorted(rows, key=candidate_score):
            distinct.setdefault(int(row["candidate_index"]), row)
        if len(distinct) >= 2:
            selected_sources.append((source_index, list(distinct.values())[:2]))
        if len(selected_sources) == args.max_sources:
            break
    if not selected_sources:
        raise SystemExit("no sources with at least two distinct safe improving candidates")

    split_out = []
    policies = {"single_a": [], "single_b": [], "pair_ab": []}
    selected_summary = []
    for new_index, (old_index, pair) in enumerate(selected_sources):
        source_record = split_rows[old_index]
        split_out.append(
            {
                "image_path": source_record["image_path"],
                "text": source_record["text"],
                "source": source_record.get("source"),
                "split": source_record.get("split"),
                "lang": source_record.get("lang"),
                "width": source_record.get("width"),
                "height": source_record.get("height"),
                "original_source_index": old_index,
            }
        )
        policies["single_a"].append(policy_row(new_index, pair[0], pair[0], "single_a"))
        policies["single_b"].append(policy_row(new_index, pair[1], pair[1], "single_b"))
        policies["pair_ab"].append(policy_row(new_index, pair[0], pair[0], "pair_ab"))
        policies["pair_ab"].append(policy_row(new_index, pair[1], pair[1], "pair_ab"))
        selected_summary.append(
            {
                "new_source_index": new_index,
                "original_source_index": old_index,
                "reference": source_record["text"],
                "candidates": [
                    {
                        "candidate_index": int(row["candidate_index"]),
                        "parseq_delta_distance": int(row.get("parseq_delta_distance", 0)),
                        "tesseract_delta_distance": int(row.get("tesseract_delta_distance", 0)),
                        "multi_teacher_delta_distance": int(row.get("multi_teacher_delta_distance", 0)),
                    }
                    for row in pair
                ],
            }
        )

    write_jsonl(args.output_split, split_out)
    policy_paths = {}
    for name, rows in policies.items():
        path = args.output_policy_prefix.with_name(f"{args.output_policy_prefix.name}_{name}.jsonl")
        write_jsonl(path, rows)
        policy_paths[name] = str(path)

    summary = {
        "candidate_table": str(args.candidate_table),
        "source_split": str(args.source_split),
        "output_split": str(args.output_split),
        "policy_paths": policy_paths,
        "sources": selected_summary,
    }
    summary_path = args.output_policy_prefix.with_name(f"{args.output_policy_prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
