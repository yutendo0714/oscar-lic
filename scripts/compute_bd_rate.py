#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oscarlic.bd import bd_rate_linear  # noqa: E402


def load(path: Path) -> tuple[list[float], list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["rate"], data["metric"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute no-extrapolation linear BD-rate")
    parser.add_argument("anchor", type=Path, help='JSON: {"rate": [...], "metric": [...]}')
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    ar, am = load(args.anchor)
    cr, cm = load(args.candidate)
    print(json.dumps(bd_rate_linear(ar, am, cr, cm, samples=args.samples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
