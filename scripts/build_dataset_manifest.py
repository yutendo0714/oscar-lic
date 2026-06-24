#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build immutable JSONL image manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash", action="store_true", help="Compute SHA256 for every file")
    args = parser.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"not a directory: {args.root}")
    files = sorted(p for p in args.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for path in files:
            row = {
                "dataset_id": args.dataset_id,
                "split": args.split,
                "relative_path": path.relative_to(args.root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hash_file(path) if args.hash else None,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(files)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
