#!/usr/bin/env python3
"""Summarize codec result JSONL files and optionally recompute MS-SSIM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from pytorch_msssim import ms_ssim


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    return candidate if candidate.exists() else path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def image_tensor(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB")).copy()
    return torch.from_numpy(array).float().unsqueeze(0).permute(0, 3, 1, 2)


def recompute_ms_ssim(row: dict) -> float | None:
    if not row.get("reconstruction_path"):
        return row.get("ms_ssim")
    original_path = resolve_path(row["image"])
    reconstruction_path = resolve_path(row["reconstruction_path"])
    if not original_path.is_file() or not reconstruction_path.is_file():
        return row.get("ms_ssim")
    original = image_tensor(original_path)
    reconstruction = image_tensor(reconstruction_path)
    if original.shape != reconstruction.shape:
        return None
    if min(original.shape[-2:]) <= 160:
        return None
    return float(ms_ssim(reconstruction, original, data_range=255.0).item())


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recompute-ms-ssim", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    if not rows:
        raise SystemExit(f"no rows in {args.results}")

    if args.recompute_ms_ssim:
        for row in rows:
            row["ms_ssim"] = recompute_ms_ssim(row)

    numeric_metrics = {}
    for key in ["actual_total_bpp", "psnr_db", "ms_ssim", "encode_seconds", "decode_seconds"]:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            numeric_metrics[key] = stats(values)

    total_pixels = sum(int(row["width"]) * int(row["height"]) for row in rows)
    total_bytes = sum(int(row["actual_total_bytes"]) for row in rows if "actual_total_bytes" in row)
    aggregate = {
        "images": len(rows),
        "total_pixels": total_pixels,
        "total_actual_bytes": total_bytes,
        "pixel_weighted_actual_bpp": 8.0 * total_bytes / total_pixels if total_pixels else None,
        "numeric_metrics": numeric_metrics,
    }

    prior_summary = None
    if args.summary and args.summary.is_file():
        prior_summary = json.loads(args.summary.read_text(encoding="utf-8"))

    output = {
        "schema_version": 1,
        "results_path": str(args.results),
        "summary_path": str(args.summary) if args.summary else None,
        "prior_summary": prior_summary,
        "aggregate": aggregate,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
