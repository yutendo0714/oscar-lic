#!/usr/bin/env python3
"""Add image/base-reconstruction patch features to candidate utility rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LATENT_TO_PIXEL = 16


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def load_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def crop(array: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    height, width = array.shape
    y0 = max(0, min(height, y0))
    y1 = max(y0 + 1, min(height, y1))
    x0 = max(0, min(width, x0))
    x1 = max(x0 + 1, min(width, x1))
    return array[y0:y1, x0:x1]


def grad_stats(array: np.ndarray) -> tuple[float, float]:
    if array.size <= 1:
        return 0.0, 0.0
    dx = np.abs(np.diff(array, axis=1)) if array.shape[1] > 1 else np.empty((0,), dtype=np.float32)
    dy = np.abs(np.diff(array, axis=0)) if array.shape[0] > 1 else np.empty((0,), dtype=np.float32)
    parts = [part.reshape(-1) for part in (dx, dy) if part.size]
    if not parts:
        return 0.0, 0.0
    values = np.concatenate(parts)
    return float(values.mean()), float((values > 0.10).mean())


def patch_features(prefix: str, patch: np.ndarray) -> dict:
    grad_mean, edge_density = grad_stats(patch)
    return {
        f"{prefix}_mean": float(patch.mean()),
        f"{prefix}_std": float(patch.std()),
        f"{prefix}_min": float(patch.min()),
        f"{prefix}_max": float(patch.max()),
        f"{prefix}_grad_mean": grad_mean,
        f"{prefix}_edge_density": edge_density,
    }


def diff_features(prefix: str, a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    abs_diff = np.abs(diff)
    grad_a, edge_a = grad_stats(a)
    grad_b, edge_b = grad_stats(b)
    return {
        f"{prefix}_abs_error_mean": float(abs_diff.mean()),
        f"{prefix}_abs_error_std": float(abs_diff.std()),
        f"{prefix}_mse": float(np.mean(diff * diff)),
        f"{prefix}_max_abs_error": float(abs_diff.max()),
        f"{prefix}_grad_abs_error": float(abs(grad_a - grad_b)),
        f"{prefix}_edge_density_delta": float(edge_a - edge_b),
    }


def base_rows_by_source(path: Path) -> dict[int, dict]:
    rows = read_jsonl(path)
    return {index: row for index, row in enumerate(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-recon-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-to-pixel", type=int, default=LATENT_TO_PIXEL)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    base_rows = base_rows_by_source(args.base_recon_split)
    image_cache: dict[str, np.ndarray] = {}
    output_rows = []
    for row in rows:
        source_index = int(row["source_index"])
        source_path = resolve_path(row["source_image"])
        base_path = resolve_path(base_rows[source_index]["image_path"])
        for path in (source_path, base_path):
            key = str(path)
            if key not in image_cache:
                image_cache[key] = load_gray(path)
        original = image_cache[str(source_path)]
        base = image_cache[str(base_path)]
        if original.shape != base.shape:
            base = np.asarray(Image.open(base_path).convert("L").resize((original.shape[1], original.shape[0])), dtype=np.float32) / 255.0

        height, width = original.shape
        y0 = int(row["y0"]) * args.latent_to_pixel
        y1 = int(row["y1"]) * args.latent_to_pixel
        x0 = int(row["x0"]) * args.latent_to_pixel
        x1 = int(row["x1"]) * args.latent_to_pixel
        original_patch = crop(original, y0, y1, x0, x1)
        base_patch = crop(base, y0, y1, x0, x1)

        out = dict(row)
        clipped_y0 = max(0, min(height, y0))
        clipped_y1 = max(clipped_y0 + 1, min(height, y1))
        clipped_x0 = max(0, min(width, x0))
        clipped_x1 = max(clipped_x0 + 1, min(width, x1))
        patch_area = float((clipped_y1 - clipped_y0) * (clipped_x1 - clipped_x0))
        out.update(
            {
                "image_width": float(width),
                "image_height": float(height),
                "image_area": float(width * height),
                "image_aspect": float(width / max(height, 1)),
                "candidate_pixel_y0": float(clipped_y0),
                "candidate_pixel_y1": float(clipped_y1),
                "candidate_pixel_x0": float(clipped_x0),
                "candidate_pixel_x1": float(clipped_x1),
                "candidate_pixel_area": patch_area,
                "candidate_pixel_area_fraction": patch_area / float(width * height),
                "candidate_pixel_center_y_fraction": float((clipped_y0 + clipped_y1) / (2 * max(height, 1))),
                "candidate_pixel_center_x_fraction": float((clipped_x0 + clipped_x1) / (2 * max(width, 1))),
            }
        )
        out.update(patch_features("orig_patch", original_patch))
        out.update(patch_features("base_patch", base_patch))
        out.update(diff_features("patch_base", original_patch, base_patch))
        out.update(diff_features("full_base", original, base))
        output_rows.append(out)
        if (len(output_rows) % 250) == 0:
            print(json.dumps({"rows": len(output_rows)}, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "input": str(args.input),
        "base_recon_split": str(args.base_recon_split),
        "output": str(args.output),
        "rows": len(output_rows),
        "latent_to_pixel": args.latent_to_pixel,
        "features_added": [
            "image_*",
            "candidate_pixel_*",
            "orig_patch_*",
            "base_patch_*",
            "patch_base_*",
            "full_base_*",
        ],
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
