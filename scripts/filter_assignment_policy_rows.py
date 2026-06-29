#!/usr/bin/env python3
"""Filter explicit assignment policy rows for clean split-compatible evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def matches(row: dict, args: argparse.Namespace) -> bool:
    if args.seed is not None and int(row.get("seed", -1)) != args.seed:
        return False
    if args.real_seed is not None and int(row.get("real_seed", row.get("seed", -1))) != args.real_seed:
        return False
    if args.policy is not None and str(row.get("policy")) != args.policy:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--real-seed", type=int)
    parser.add_argument("--policy")
    args = parser.parse_args()

    selected = [row for row in read_jsonl(args.input) if matches(row, args)]
    if not selected:
        raise SystemExit("no policy rows matched the requested filters")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    parseq_delta = 0
    tesseract_delta = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in sorted(selected, key=lambda item: (int(item["source_index"]), int(item["candidate_index"]))):
            changed += int(int(row["code_index"]) != int(row["nearest_code"]))
            parseq_delta += int(row.get("parseq_delta_vs_nearest", 0))
            tesseract_delta += int(row.get("tesseract_delta_vs_nearest", 0))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "filters": {
            "seed": args.seed,
            "real_seed": args.real_seed,
            "policy": args.policy,
        },
        "groups": len(selected),
        "changed_groups": changed,
        "parseq_delta_sum_vs_nearest": parseq_delta,
        "tesseract_delta_sum_vs_nearest": tesseract_delta,
        "gate_thresholds": sorted({float(row["gate_threshold"]) for row in selected if "gate_threshold" in row}),
        "gate_model_seeds": sorted({int(row["gate_model_seed"]) for row in selected if "gate_model_seed" in row}),
        "score_model_seeds": sorted({str(row.get("score_model_seed")) for row in selected}),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
