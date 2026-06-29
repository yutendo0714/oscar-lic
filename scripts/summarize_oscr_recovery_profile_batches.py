#!/usr/bin/env python3
"""Summarize per-seed OSCR recovery-profile batch audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COUNT_KEYS = (
    "stream_count",
    "text_main_stream_count",
    "default_rejects_text_main_corruption",
    "recovery_recovers_text_main_corruption",
    "default_rejects_base_main_corruption",
    "recovery_rejects_base_main_corruption",
)


def load_batch(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in data:
        raise ValueError(f"{path} has no summary")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batches = [load_batch(path) for path in args.batch]
    totals = {key: sum(int(batch["summary"][key]) for batch in batches) for key in COUNT_KEYS}
    totals["all_text_main_corruptions_recovered"] = (
        totals["recovery_recovers_text_main_corruption"] == totals["text_main_stream_count"]
    )
    totals["all_base_main_corruptions_rejected"] = (
        totals["recovery_rejects_base_main_corruption"] == totals["stream_count"]
    )
    output = {
        "batch_paths": [str(path) for path in args.batch],
        "batch_count": len(batches),
        "summary": totals,
        "per_batch": [
            {
                "path": str(path),
                "stream_dir": batch.get("stream_dir"),
                "summary": batch["summary"],
            }
            for path, batch in zip(args.batch, batches, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
