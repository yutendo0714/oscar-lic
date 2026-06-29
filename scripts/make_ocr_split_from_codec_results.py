#!/usr/bin/env python3
"""Build an OCR JSONL split over codec reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_reference_split(path: Path) -> dict[str, dict]:
    by_path: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = Path(record["image_path"])
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            record["image_path"] = str(image_path)
            by_path[str(image_path)] = record
            by_path[project_path(image_path)] = record
    return by_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-split", type=Path, required=True)
    parser.add_argument("--codec-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--reconstruction-field", default="reconstruction_path")
    args = parser.parse_args()

    references = read_reference_split(args.reference_split)
    rows = []
    with args.codec_results.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            codec_row = json.loads(line)
            source = codec_row["image"]
            ref = references.get(source)
            if ref is None:
                ref = references.get(str(Path(source).resolve()))
            if ref is None:
                raise KeyError(f"{args.codec_results}:{line_no}: no reference for {source}")
            rec_path_value = codec_row.get(args.reconstruction_field)
            if rec_path_value is None:
                raise KeyError(
                    f"{args.codec_results}:{line_no}: missing reconstruction field {args.reconstruction_field}"
                )
            rec_path = ROOT / rec_path_value
            if not rec_path.is_file():
                raise FileNotFoundError(rec_path)
            rows.append(
                {
                    "image_path": project_path(rec_path),
                    "text": ref["text"],
                    "source": ref.get("source"),
                    "split": ref.get("split"),
                    "method_id": args.method_id,
                    "source_image": source,
                    "actual_total_bpp": codec_row.get("actual_total_bpp"),
                    "actual_total_bytes": codec_row.get("actual_total_bytes"),
                    "psnr_db": codec_row.get("psnr_db"),
                    "ms_ssim": codec_row.get("ms_ssim"),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
