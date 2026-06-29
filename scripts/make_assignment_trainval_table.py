#!/usr/bin/env python3
"""Create a two-partition assignment table for explicit train->val ranker runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-table", type=Path, required=True)
    parser.add_argument("--val-table", type=Path, required=True)
    parser.add_argument("--real-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for source, internal_seed, partition in [
        (args.train_table, 0, "train"),
        (args.val_table, 1, "val"),
    ]:
        for row in read_jsonl(source):
            if int(row["seed"]) != args.real_seed:
                continue
            out = dict(row)
            out["real_seed"] = args.real_seed
            out["original_seed"] = int(row["seed"])
            out["seed"] = internal_seed
            out["assignment_partition"] = partition
            rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "description": "Assignment train/val table with train mapped to seed 0 and val mapped to seed 1.",
        "real_seed": args.real_seed,
        "train_table": str(args.train_table),
        "val_table": str(args.val_table),
        "output": str(args.output),
        "rows": len(rows),
        "train_rows": sum(1 for row in rows if row["assignment_partition"] == "train"),
        "val_rows": sum(1 for row in rows if row["assignment_partition"] == "val"),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
