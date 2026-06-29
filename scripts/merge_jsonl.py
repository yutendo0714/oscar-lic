#!/usr/bin/env python3
"""Merge JSONL files while preserving line order."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with args.output.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    out.write(line if line.endswith("\n") else line + "\n")
                    rows += 1
    print(f"{args.output}\t{rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
