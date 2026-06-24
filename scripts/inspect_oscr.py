#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oscarlic.bitstream import read_container  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and inspect an OSCAR .oscr container")
    parser.add_argument("path", type=Path)
    parser.add_argument("--no-crc", action="store_true")
    args = parser.parse_args()
    parsed = read_container(args.path, verify_crc=not args.no_crc)
    result = {
        "path": str(args.path), "width": parsed.width, "height": parsed.height,
        "total_bytes": parsed.total_bytes, "bpp": parsed.bpp,
        "model_id": parsed.model_id, "model_version": parsed.model_version,
        "sections": [
            {"type": s.section_type, "codec": s.codec, "flags": s.flags,
             "offset": s.offset, "length": s.length,
             "unprotected_length": s.unprotected_length,
             "dependency_mask": s.dependency_mask}
            for s in parsed.sections
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
