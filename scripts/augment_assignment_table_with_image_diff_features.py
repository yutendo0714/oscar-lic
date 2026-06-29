#!/usr/bin/env python3
"""Add lightweight reconstructed-image/code-effect features to assignment rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape == reference.shape:
        return image
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L")
    resized = pil.resize((reference.shape[1], reference.shape[0]), Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def grad_stats(array: np.ndarray) -> tuple[float, float]:
    if array.size <= 1:
        return 0.0, 0.0
    parts = []
    if array.shape[1] > 1:
        parts.append(np.abs(np.diff(array, axis=1)).reshape(-1))
    if array.shape[0] > 1:
        parts.append(np.abs(np.diff(array, axis=0)).reshape(-1))
    if not parts:
        return 0.0, 0.0
    values = np.concatenate(parts)
    return float(values.mean()), float((values > 0.10).mean())


def gradient_map(array: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(array, dtype=np.float32)
    gy = np.zeros_like(array, dtype=np.float32)
    if array.shape[1] > 1:
        gx[:, 1:] = np.abs(np.diff(array, axis=1))
    if array.shape[0] > 1:
        gy[1:, :] = np.abs(np.diff(array, axis=0))
    return gx + gy


def safe_region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].mean())


def projection_delta_stats(prefix: str, a: np.ndarray, b: np.ndarray) -> dict:
    row_delta = a.mean(axis=1) - b.mean(axis=1)
    col_delta = a.mean(axis=0) - b.mean(axis=0)
    a_dark = (a < 0.5).astype(np.float32)
    b_dark = (b < 0.5).astype(np.float32)
    row_dark_delta = a_dark.mean(axis=1) - b_dark.mean(axis=1)
    col_dark_delta = a_dark.mean(axis=0) - b_dark.mean(axis=0)
    return {
        f"{prefix}_row_projection_abs_mean": float(np.abs(row_delta).mean()),
        f"{prefix}_row_projection_max_abs": float(np.abs(row_delta).max(initial=0.0)),
        f"{prefix}_col_projection_abs_mean": float(np.abs(col_delta).mean()),
        f"{prefix}_col_projection_max_abs": float(np.abs(col_delta).max(initial=0.0)),
        f"{prefix}_row_dark_projection_abs_mean": float(np.abs(row_dark_delta).mean()),
        f"{prefix}_row_dark_projection_max_abs": float(np.abs(row_dark_delta).max(initial=0.0)),
        f"{prefix}_col_dark_projection_abs_mean": float(np.abs(col_dark_delta).mean()),
        f"{prefix}_col_dark_projection_max_abs": float(np.abs(col_dark_delta).max(initial=0.0)),
    }


def image_stats(prefix: str, image: np.ndarray) -> dict:
    grad_mean, edge_density = grad_stats(image)
    dark_025 = image < 0.25
    dark_050 = image < 0.50
    dark_075 = image < 0.75
    if bool(dark_050.any()):
        ys, xs = np.nonzero(dark_050)
        ink_center_y = float(ys.mean() / max(image.shape[0] - 1, 1))
        ink_center_x = float(xs.mean() / max(image.shape[1] - 1, 1))
    else:
        ink_center_y = 0.5
        ink_center_x = 0.5
    return {
        f"{prefix}_mean": float(image.mean()),
        f"{prefix}_std": float(image.std()),
        f"{prefix}_min": float(image.min()),
        f"{prefix}_max": float(image.max()),
        f"{prefix}_grad_mean": grad_mean,
        f"{prefix}_edge_density": edge_density,
        f"{prefix}_dark025_fraction": float(dark_025.mean()),
        f"{prefix}_dark050_fraction": float(dark_050.mean()),
        f"{prefix}_dark075_fraction": float(dark_075.mean()),
        f"{prefix}_ink_center_y_fraction": ink_center_y,
        f"{prefix}_ink_center_x_fraction": ink_center_x,
    }


def diff_bbox(abs_diff: np.ndarray, threshold: float) -> dict:
    mask = abs_diff > threshold
    height, width = abs_diff.shape
    if not bool(mask.any()):
        return {
            "bbox_area_fraction": 0.0,
            "bbox_height_fraction": 0.0,
            "bbox_width_fraction": 0.0,
            "bbox_center_y_fraction": 0.5,
            "bbox_center_x_fraction": 0.5,
        }
    ys, xs = np.nonzero(mask)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    bbox_h = max(y1 - y0, 1)
    bbox_w = max(x1 - x0, 1)
    return {
        "bbox_area_fraction": float((bbox_h * bbox_w) / max(height * width, 1)),
        "bbox_height_fraction": float(bbox_h / max(height, 1)),
        "bbox_width_fraction": float(bbox_w / max(width, 1)),
        "bbox_center_y_fraction": float((y0 + y1) / (2 * max(height, 1))),
        "bbox_center_x_fraction": float((x0 + x1) / (2 * max(width, 1))),
    }


def diff_stats(prefix: str, a: np.ndarray, b: np.ndarray, threshold: float) -> dict:
    diff = a - b
    abs_diff = np.abs(diff)
    grad_a, edge_a = grad_stats(a)
    grad_b, edge_b = grad_stats(b)
    a_dark = a < 0.5
    b_dark = b < 0.5
    a_edge = gradient_map(a) > 0.10
    b_edge = gradient_map(b) > 0.10
    bbox = diff_bbox(abs_diff, threshold)
    out = {
        f"{prefix}_abs_mean": float(abs_diff.mean()),
        f"{prefix}_abs_std": float(abs_diff.std()),
        f"{prefix}_mse": float(np.mean(diff * diff)),
        f"{prefix}_max_abs": float(abs_diff.max()),
        f"{prefix}_changed_fraction": float((abs_diff > threshold).mean()),
        f"{prefix}_signed_mean": float(diff.mean()),
        f"{prefix}_signed_abs_ratio": float(abs(diff.mean()) / max(float(abs_diff.mean()), 1e-8)),
        f"{prefix}_grad_abs_delta": float(abs(grad_a - grad_b)),
        f"{prefix}_edge_density_delta": float(edge_a - edge_b),
        f"{prefix}_a_dark_abs_mean": safe_region_mean(abs_diff, a_dark),
        f"{prefix}_b_dark_abs_mean": safe_region_mean(abs_diff, b_dark),
        f"{prefix}_dark_union_abs_mean": safe_region_mean(abs_diff, a_dark | b_dark),
        f"{prefix}_light_intersection_abs_mean": safe_region_mean(abs_diff, (~a_dark) & (~b_dark)),
        f"{prefix}_a_edge_abs_mean": safe_region_mean(abs_diff, a_edge),
        f"{prefix}_b_edge_abs_mean": safe_region_mean(abs_diff, b_edge),
        f"{prefix}_edge_union_abs_mean": safe_region_mean(abs_diff, a_edge | b_edge),
        f"{prefix}_dark_changed_fraction": safe_region_mean((abs_diff > threshold).astype(np.float32), a_dark | b_dark),
        f"{prefix}_edge_changed_fraction": safe_region_mean((abs_diff > threshold).astype(np.float32), a_edge | b_edge),
    }
    out.update({f"{prefix}_{key}": value for key, value in bbox.items()})
    out.update(projection_delta_stats(prefix, a, b))
    return out


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def group_rows(rows: list[dict]) -> dict[tuple[int, int, int, int], list[dict]]:
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return dict(groups)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diff-threshold", type=float, default=1.0 / 255.0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    image_cache: dict[str, np.ndarray] = {}
    output_rows = []
    for group_index, group in enumerate(group_rows(rows).values(), start=1):
        ordered = sorted(group, key=lambda row: int(row["topk_rank"]))
        nearest_row = next(row for row in ordered if int(row["is_nearest"]))
        source_path = resolve_path(str(nearest_row["source_image"]))
        nearest_path = resolve_path(str(nearest_row["image_path"]))
        for path in (source_path, nearest_path):
            key = str(path)
            if key not in image_cache:
                image_cache[key] = load_gray(path)
        source = image_cache[str(source_path)]
        nearest = resize_like(image_cache[str(nearest_path)], source)
        base_features = {
            "img_width": float(source.shape[1]),
            "img_height": float(source.shape[0]),
            "img_area": float(source.shape[0] * source.shape[1]),
            "img_aspect": float(source.shape[1] / max(source.shape[0], 1)),
        }
        base_features.update(image_stats("img_source", source))
        base_features.update(image_stats("img_nearest", nearest))
        base_features.update(diff_stats("img_source_nearest", source, nearest, args.diff_threshold))
        for row in ordered:
            variant_path = resolve_path(str(row["image_path"]))
            key = str(variant_path)
            if key not in image_cache:
                image_cache[key] = load_gray(variant_path)
            variant = resize_like(image_cache[key], source)
            out = dict(row)
            out.update(base_features)
            out.update(image_stats("img_variant", variant))
            out.update(diff_stats("img_variant_nearest", variant, nearest, args.diff_threshold))
            out.update(diff_stats("img_source_variant", source, variant, args.diff_threshold))
            output_rows.append(out)
        if (group_index % 25) == 0:
            print(json.dumps({"groups": group_index, "rows": len(output_rows)}, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "description": "Assignment table augmented with lightweight reconstructed-image difference features.",
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "groups": len(group_rows(rows)),
        "diff_threshold": float(args.diff_threshold),
        "features_added": ["img_*"],
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
