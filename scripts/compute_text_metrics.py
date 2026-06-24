#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oscarlic.text_metrics import evaluate_pairs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute corpus CER/WER/NED from JSONL")
    parser.add_argument("input", type=Path, help="JSONL with reference and hypothesis fields")
    parser.add_argument("--reference-key", default="reference")
    parser.add_argument("--hypothesis-key", default="hypothesis")
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pairs = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                pairs.append((row[args.reference_key], row[args.hypothesis_key]))
            except KeyError as exc:
                raise SystemExit(f"line {line_number}: missing key {exc}")
    result = evaluate_pairs(pairs, args.profile).to_dict()
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
