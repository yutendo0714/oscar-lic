#!/usr/bin/env python3
"""Build paired OCR-image splits for rows whose OCR output changed in a comparison JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    output_rows: list[dict] = []
    summary_rows: list[dict] = []
    for pair in comparison["pairs"]:
        profile = pair["profiles"][args.profile]
        if not profile["changed_examples"]:
            continue
        baseline_rows = read_jsonl(Path(pair["baseline_results"]))
        candidate_rows = read_jsonl(Path(pair["candidate_results"]))
        for example in profile["changed_examples"]:
            index = int(example["index"])
            baseline = baseline_rows[index]
            candidate = candidate_rows[index]
            common = {
                "comparison": str(args.comparison),
                "profile": args.profile,
                "pair_label": pair["label"],
                "row_index": index,
                "reference": str(baseline["reference"]),
                "text": str(baseline["reference"]),
                "source": baseline.get("source"),
                "split": baseline.get("split"),
                "baseline_prediction": example["baseline_prediction"],
                "candidate_prediction": example["candidate_prediction"],
                "baseline_distance": example["baseline_distance"],
                "candidate_distance": example["candidate_distance"],
                "delta_distance": example["delta_distance"],
                "delta_exact": example["delta_exact"],
            }
            output_rows.append(
                {
                    **common,
                    "image_path": baseline["image"],
                    "variant": "baseline",
                    "method_id": baseline.get("method_id"),
                }
            )
            output_rows.append(
                {
                    **common,
                    "image_path": candidate["image"],
                    "variant": "candidate",
                    "method_id": candidate.get("method_id"),
                }
            )
            summary_rows.append(common)

    write_jsonl(args.output, output_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "comparison": str(args.comparison),
        "profile": args.profile,
        "changed_pairs": len(summary_rows),
        "rows": len(output_rows),
        "delta_distance_sum": sum(row["delta_distance"] for row in summary_rows),
        "changed": summary_rows,
        "output": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
