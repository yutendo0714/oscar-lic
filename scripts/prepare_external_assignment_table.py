#!/usr/bin/env python3
"""Prepare a score-key-compatible external assignment table.

Eval600 assignment tables were generated before `real_seed` was required by the
external transfer score files. This utility adds that key and filters the
Eval300 table to a chosen held-out seed so downstream gates can join by
`(real_seed, seed, source_index, candidate_index)`.
"""

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
    parser.add_argument("--eval600-table", type=Path, action="append", required=True)
    parser.add_argument("--eval300-table", type=Path, required=True)
    parser.add_argument("--eval300-seed", type=int, default=1)
    parser.add_argument("--eval600-real-seed", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    counts = {}
    for table in args.eval600_table:
        table_rows = []
        for row in read_jsonl(table):
            out = dict(row)
            out.setdefault("real_seed", int(args.eval600_real_seed))
            out.setdefault("original_seed", int(args.eval600_real_seed))
            out["table_source"] = table.name
            table_rows.append(out)
        counts[str(table)] = len(table_rows)
        rows.extend(table_rows)

    eval300_rows = []
    for row in read_jsonl(args.eval300_table):
        if int(row["seed"]) != int(args.eval300_seed):
            continue
        out = dict(row)
        out.setdefault("real_seed", int(out.get("original_seed", out["seed"])))
        out.setdefault("original_seed", int(out["real_seed"]))
        out["table_source"] = args.eval300_table.name
        eval300_rows.append(out)
    counts[str(args.eval300_table)] = len(eval300_rows)
    rows.extend(eval300_rows)

    write_jsonl(args.output, rows)
    groups = {
        (
            int(row.get("real_seed", row.get("original_seed", 0))),
            int(row["seed"]),
            int(row["source_index"]),
            int(row["candidate_index"]),
        )
        for row in rows
    }
    summary = {
        "description": "Merged Eval600 train tables plus filtered Eval300 validation assignment table.",
        "eval600_tables": [str(path) for path in args.eval600_table],
        "eval300_table": str(args.eval300_table),
        "eval300_seed": int(args.eval300_seed),
        "eval600_real_seed": int(args.eval600_real_seed),
        "output": str(args.output),
        "rows": len(rows),
        "groups": len(groups),
        "input_row_counts": counts,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
