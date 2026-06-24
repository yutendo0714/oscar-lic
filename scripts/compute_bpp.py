#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oscarlic.rate import file_bpp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute actual bpp from complete file size")
    parser.add_argument("path", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(file_bpp(args.path, args.width, args.height), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
