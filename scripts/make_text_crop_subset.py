#!/usr/bin/env python3
"""Create a deterministic text-crop JSONL subset from existing manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import random
import re


ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_key(record: dict) -> tuple[str, str]:
    return (str(record.get("image_path", "")), str(record.get("text", "")))


def keep_record(record: dict, args) -> bool:
    text = str(record.get("text", ""))
    image_path = Path(record.get("image_path", ""))
    if not image_path.is_file():
        return False
    if len(text) < args.min_text_len or len(text) > args.max_text_len:
        return False
    if args.require_alnum and not ALNUM_RE.search(text):
        return False
    width = int(record.get("width", 0) or 0)
    height = int(record.get("height", 0) or 0)
    if width < args.min_width or height < args.min_height:
        return False
    if args.sources and record.get("source") not in args.sources:
        return False
    return True


def balanced_sample(records: list[dict], n: int, seed: int, excluded: set[tuple[str, str]]) -> list[dict]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record_key(record) in excluded:
            continue
        by_source[str(record.get("source", "unknown"))].append(record)
    queues = []
    for source in sorted(by_source):
        values = by_source[source]
        rng.shuffle(values)
        queues.append((source, deque(values)))

    selected = []
    seen = set()
    while len(selected) < n and any(queue for _, queue in queues):
        for _, queue in queues:
            if len(selected) >= n:
                break
            while queue:
                record = queue.popleft()
                key = record_key(record)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(record)
                break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--min-width", type=int, default=24)
    parser.add_argument("--min-height", type=int, default=24)
    parser.add_argument("--min-text-len", type=int, default=2)
    parser.add_argument("--max-text-len", type=int, default=24)
    parser.add_argument("--require-alnum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()

    manifests = []
    candidates = []
    excluded: set[tuple[str, str]] = set()
    exclude_manifests = []
    for exclude_path in args.exclude:
        exclude_rows = read_jsonl(exclude_path)
        keys = {record_key(record) for record in exclude_rows}
        excluded.update(keys)
        exclude_manifests.append(
            {
                "path": str(exclude_path),
                "sha256": sha256_file(exclude_path),
                "records": len(exclude_rows),
                "unique_excluded_keys": len(keys),
            }
        )
    for manifest in args.manifest:
        rows = read_jsonl(manifest)
        kept = [record for record in rows if keep_record(record, args)]
        manifests.append(
            {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "records": len(rows),
                "kept_after_filter": len(kept),
            }
        )
        candidates.extend(kept)
    selected = balanced_sample(candidates, args.samples, args.seed, excluded)
    if len(selected) < args.samples:
        raise SystemExit(f"only selected {len(selected)} records, requested {args.samples}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(
                {
                    "image_path": record["image_path"],
                    "text": str(record["text"]),
                    "source": record.get("source"),
                    "split": record.get("split"),
                    "lang": record.get("lang"),
                    "width": record.get("width"),
                    "height": record.get("height"),
                },
                ensure_ascii=False,
            )
            + "\n"
            for record in selected
        ),
        encoding="utf-8",
    )
    meta_output = args.meta_output or args.output.with_suffix(args.output.suffix + ".meta.json")
    meta = {
        "output": str(args.output),
        "samples": len(selected),
        "seed": args.seed,
        "filters": {
            "min_width": args.min_width,
            "min_height": args.min_height,
            "min_text_len": args.min_text_len,
            "max_text_len": args.max_text_len,
            "require_alnum": args.require_alnum,
            "sources": args.sources,
        },
        "excluded_keys": len(excluded),
        "exclude_manifests": exclude_manifests,
        "source_manifests": manifests,
        "source_counts": dict(sorted((source, sum(1 for row in selected if row.get("source") == source)) for source in {row.get("source") for row in selected})),
    }
    meta_output.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "meta": str(meta_output), "samples": len(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
