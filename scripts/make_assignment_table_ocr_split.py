#!/usr/bin/env python3
"""Create an OCR split from assignment-table reconstruction rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-field", default="image_path")
    args = parser.parse_args()

    rows = read_jsonl(args.table)
    records = []
    seen = set()
    for row in rows:
        image = str(row[args.image_field])
        key = (
            image,
            int(row.get("real_seed", row.get("original_seed", row.get("seed", 0)))),
            int(row["seed"]),
            int(row["source_index"]),
            int(row["candidate_index"]),
            int(row["code_index"]),
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "image_path": image,
                "text": str(row.get("reference", "")),
                "source": row.get("source"),
                "split": "assignment_table",
                "real_seed": int(row.get("real_seed", row.get("original_seed", row.get("seed", 0)))),
                "seed": int(row["seed"]),
                "source_index": int(row["source_index"]),
                "candidate_index": int(row["candidate_index"]),
                "code_index": int(row["code_index"]),
                "nearest_code": int(row["nearest_code"]),
                "topk_rank": int(row["topk_rank"]),
                "is_nearest": int(row["is_nearest"]),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "description": "OCR split generated from assignment-table reconstruction rows.",
        "table": str(args.table),
        "output": str(args.output),
        "image_field": args.image_field,
        "records": len(records),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
