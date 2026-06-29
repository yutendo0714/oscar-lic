#!/usr/bin/env python3
"""Build explicit subset policies for multi-candidate interaction diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path


LETTERS = "abcdefghijklmnopqrstuvwxyz"


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


def subset_name(indices: tuple[int, ...]) -> str:
    if len(indices) == 1:
        return f"single_{LETTERS[indices[0]]}"
    if len(indices) == 2:
        return "pair_" + "".join(LETTERS[index] for index in indices)
    return "subset" + str(len(indices)) + "_" + "".join(LETTERS[index] for index in indices)


def policy_row(new_source_index: int, source_row: dict, candidate_row: dict, policy: str) -> dict:
    return {
        "source_index": new_source_index,
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
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--subset-size", type=int, default=3)
    parser.add_argument("--require-no-evaluator-worsens", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not 1 <= args.subset_size <= len(LETTERS):
        raise SystemExit(f"--subset-size must be in [1, {len(LETTERS)}]")

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
        if len(distinct) >= args.subset_size:
            selected_sources.append((source_index, list(distinct.values())[: args.subset_size]))
        if len(selected_sources) == args.max_sources:
            break
    if not selected_sources:
        raise SystemExit(f"no sources with at least {args.subset_size} distinct safe improving candidates")

    split_out = []
    policies: dict[str, list[dict]] = {}
    selected_summary = []
    subset_indices = [
        subset
        for size in range(1, args.subset_size + 1)
        for subset in combinations(range(args.subset_size), size)
    ]
    for subset in subset_indices:
        policies[subset_name(subset)] = []

    for new_index, (old_index, candidates) in enumerate(selected_sources):
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
        for subset in subset_indices:
            name = subset_name(subset)
            for candidate_slot in subset:
                policies[name].append(policy_row(new_index, candidates[candidate_slot], candidates[candidate_slot], name))
        selected_summary.append(
            {
                "new_source_index": new_index,
                "original_source_index": old_index,
                "reference": source_record["text"],
                "candidates": [
                    {
                        "slot": slot,
                        "candidate_index": int(row["candidate_index"]),
                        "parseq_delta_distance": int(row.get("parseq_delta_distance", 0)),
                        "tesseract_delta_distance": int(row.get("tesseract_delta_distance", 0)),
                        "multi_teacher_delta_distance": int(row.get("multi_teacher_delta_distance", 0)),
                    }
                    for slot, row in enumerate(candidates)
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
        "subset_size": args.subset_size,
        "max_sources": args.max_sources,
        "policy_paths": policy_paths,
        "sources": selected_summary,
    }
    summary_path = args.output_policy_prefix.with_name(f"{args.output_policy_prefix.name}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
