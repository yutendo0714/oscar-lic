#!/usr/bin/env python3
"""Merge OOF train scores with existing external validation scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--base-score", type=Path, required=True, help="Existing full score JSONL used for non-OOF rows.")
    parser.add_argument("--oof-score", type=Path, required=True, help="OOF score JSONL for train rows.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=2)
    parser.add_argument("--oof-seed", type=int, default=600)
    parser.add_argument("--val-seed", type=int, default=1)
    args = parser.parse_args()

    table_rows = read_jsonl(args.table)
    expected = set(range(len(table_rows)))
    base_by_row = {
        int(row["row_index"]): row
        for row in read_jsonl(args.base_score)
        if int(row["model_seed"]) == int(args.model_seed)
    }
    oof_by_row = {
        int(row["row_index"]): row
        for row in read_jsonl(args.oof_score)
        if int(row["model_seed"]) == int(args.model_seed)
    }
    missing_base = expected - set(base_by_row)
    if missing_base:
        raise SystemExit(f"base score missing {len(missing_base)} row indices; first={min(missing_base)}")

    merged = []
    replaced = 0
    for row_index, table_row in enumerate(table_rows):
        seed = int(table_row["seed"])
        if seed == int(args.oof_seed):
            if row_index not in oof_by_row:
                raise SystemExit(f"OOF score missing row_index={row_index}")
            out = dict(oof_by_row[row_index])
            out["score_source"] = "oof"
            replaced += 1
        else:
            out = dict(base_by_row[row_index])
            out["score_source"] = "base_external" if seed == int(args.val_seed) else "base"
        out["model_seed"] = int(args.model_seed)
        merged.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "table": str(args.table),
        "base_score": str(args.base_score),
        "oof_score": str(args.oof_score),
        "output": str(args.output),
        "model_seed": int(args.model_seed),
        "rows": len(merged),
        "replaced_oof_rows": int(replaced),
        "base_external_rows": int(sum(1 for row in merged if row.get("score_source") == "base_external")),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
