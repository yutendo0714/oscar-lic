#!/usr/bin/env python3
"""Extract deployable-ish text-topology features for source-OOF top-k rows.

This diagnostic avoids OCR model features. It summarizes connected components,
holes, dark-pixel projections and candidate-vs-nearest dark-pixel changes from
the reconstructed word crops, then appends those features to the existing
source-OOF top-k selector features.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_assignment_topk_policy_selector as base  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape == reference.shape:
        return image
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L")
    resized = pil.resize((reference.shape[1], reference.shape[0]), Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def crop_bbox(mask: np.ndarray, pad: int = 2) -> tuple[slice, slice]:
    if not bool(mask.any()):
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    ys, xs = np.nonzero(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)


def count_holes(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    background = ~mask
    labels, count = ndimage.label(background)
    if count == 0:
        return 0
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_labels = set(int(v) for v in np.unique(labels[border]) if int(v) != 0)
    return int(sum(1 for label in range(1, count + 1) if label not in border_labels))


def run_count(values: np.ndarray) -> int:
    active = values.astype(bool)
    if active.size == 0:
        return 0
    padded = np.pad(active.astype(np.int8), (1, 1), mode="constant")
    starts = np.diff(padded) == 1
    return int(starts.sum())


def safe_stats(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(values.mean()), float(values.std()), float(values.max(initial=0.0))


def component_features(prefix: str, image: np.ndarray, threshold: float) -> tuple[list[str], list[float]]:
    if image.size == 0:
        image = np.zeros((1, 1), dtype=np.float32)
    mask = image < threshold
    area = float(mask.size)
    dark = float(mask.sum())
    labels, count = ndimage.label(mask)
    comp_sizes = np.bincount(labels.reshape(-1), minlength=count + 1)[1:].astype(np.float32)
    comp_sizes_sorted = np.sort(comp_sizes)[::-1] if comp_sizes.size else np.asarray([], dtype=np.float32)
    largest = float(comp_sizes_sorted[0]) if comp_sizes_sorted.size else 0.0
    second = float(comp_sizes_sorted[1]) if comp_sizes_sorted.size > 1 else 0.0
    small = comp_sizes[comp_sizes <= max(2.0, 0.002 * area)] if comp_sizes.size else np.asarray([], dtype=np.float32)
    holes = count_holes(mask)
    if bool(mask.any()):
        ys, xs = np.nonzero(mask)
        bbox_h = int(ys.max()) - int(ys.min()) + 1
        bbox_w = int(xs.max()) - int(xs.min()) + 1
        bbox_area = float(bbox_h * bbox_w)
        bbox_fill = dark / max(bbox_area, 1.0)
    else:
        bbox_h = bbox_w = 0
        bbox_area = 0.0
        bbox_fill = 0.0
    row_occ = mask.mean(axis=1)
    col_occ = mask.mean(axis=0)
    row_mean, row_std, row_max = safe_stats(row_occ)
    col_mean, col_std, col_max = safe_stats(col_occ)
    row_active = row_occ > 0.0
    col_active = col_occ > 0.0
    if dark > 0:
        dist = ndimage.distance_transform_edt(mask)
        stroke_mean = float((2.0 * dist[mask]).mean())
        stroke_max = float((2.0 * dist[mask]).max(initial=0.0))
        eroded = ndimage.binary_erosion(mask)
        boundary_fraction = float((mask & ~eroded).sum() / dark)
    else:
        stroke_mean = 0.0
        stroke_max = 0.0
        boundary_fraction = 0.0
    sobel_x = ndimage.sobel(image, axis=1, mode="nearest")
    sobel_y = ndimage.sobel(image, axis=0, mode="nearest")
    edge = np.hypot(sobel_x, sobel_y)
    edge_mean = float(edge.mean())
    edge_dark_mean = float(edge[mask].mean()) if dark > 0 else 0.0
    names = [
        f"{prefix}_dark_fraction",
        f"{prefix}_component_count_per_kpix",
        f"{prefix}_largest_component_fraction",
        f"{prefix}_second_component_fraction",
        f"{prefix}_small_component_count_per_kpix",
        f"{prefix}_small_component_area_fraction",
        f"{prefix}_hole_count_per_kpix",
        f"{prefix}_euler_per_kpix",
        f"{prefix}_bbox_area_fraction",
        f"{prefix}_bbox_height_fraction",
        f"{prefix}_bbox_width_fraction",
        f"{prefix}_bbox_fill_fraction",
        f"{prefix}_row_occ_mean",
        f"{prefix}_row_occ_std",
        f"{prefix}_row_occ_max",
        f"{prefix}_col_occ_mean",
        f"{prefix}_col_occ_std",
        f"{prefix}_col_occ_max",
        f"{prefix}_active_row_fraction",
        f"{prefix}_active_col_fraction",
        f"{prefix}_row_run_count_per_kpix",
        f"{prefix}_col_run_count_per_kpix",
        f"{prefix}_stroke_width_mean",
        f"{prefix}_stroke_width_max",
        f"{prefix}_boundary_fraction",
        f"{prefix}_sobel_mean",
        f"{prefix}_sobel_dark_mean",
    ]
    values = [
        dark / area,
        1000.0 * float(count) / area,
        largest / area,
        second / area,
        1000.0 * float(small.size) / area,
        float(small.sum()) / area if small.size else 0.0,
        1000.0 * float(holes) / area,
        1000.0 * float(count - holes) / area,
        bbox_area / area,
        float(bbox_h) / max(float(mask.shape[0]), 1.0),
        float(bbox_w) / max(float(mask.shape[1]), 1.0),
        bbox_fill,
        row_mean,
        row_std,
        row_max,
        col_mean,
        col_std,
        col_max,
        float(row_active.mean()),
        float(col_active.mean()),
        1000.0 * float(run_count(row_active)) / area,
        1000.0 * float(run_count(col_active)) / area,
        stroke_mean,
        stroke_max,
        boundary_fraction,
        edge_mean,
        edge_dark_mean,
    ]
    return names, values


def pair_features(prefix: str, a: np.ndarray, b: np.ndarray, threshold: float) -> tuple[list[str], list[float]]:
    if a.shape != b.shape:
        b = resize_like(b, a)
    ma = a < threshold
    mb = b < threshold
    added = ma & ~mb
    removed = mb & ~ma
    sym = ma ^ mb
    added_labels, added_count = ndimage.label(added)
    removed_labels, removed_count = ndimage.label(removed)
    _ = added_labels, removed_labels
    area = float(ma.size)
    row_sym = sym.mean(axis=1)
    col_sym = sym.mean(axis=0)
    row_mean, row_std, row_max = safe_stats(row_sym)
    col_mean, col_std, col_max = safe_stats(col_sym)
    abs_diff = np.abs(a - b)
    dark_union = ma | mb
    names = [
        f"{prefix}_added_dark_fraction",
        f"{prefix}_removed_dark_fraction",
        f"{prefix}_sym_dark_fraction",
        f"{prefix}_added_component_count_per_kpix",
        f"{prefix}_removed_component_count_per_kpix",
        f"{prefix}_row_sym_mean",
        f"{prefix}_row_sym_std",
        f"{prefix}_row_sym_max",
        f"{prefix}_col_sym_mean",
        f"{prefix}_col_sym_std",
        f"{prefix}_col_sym_max",
        f"{prefix}_absdiff_mean",
        f"{prefix}_absdiff_dark_union_mean",
        f"{prefix}_absdiff_max",
    ]
    values = [
        float(added.mean()),
        float(removed.mean()),
        float(sym.mean()),
        1000.0 * float(added_count) / area,
        1000.0 * float(removed_count) / area,
        row_mean,
        row_std,
        row_max,
        col_mean,
        col_std,
        col_max,
        float(abs_diff.mean()),
        float(abs_diff[dark_union].mean()) if bool(dark_union.any()) else 0.0,
        float(abs_diff.max(initial=0.0)),
    ]
    return names, values


def delta_features(prefix: str, names_a: list[str], values_a: list[float], values_b: list[float]) -> tuple[list[str], list[float]]:
    suffixes = [name.split("_", 1)[1] for name in names_a]
    names = []
    values = []
    for suffix, va, vb in zip(suffixes, values_a, values_b):
        names.append(f"{prefix}_{suffix}_delta")
        values.append(float(va - vb))
        names.append(f"{prefix}_{suffix}_abs_delta")
        values.append(float(abs(va - vb)))
    return names, values


def topology_for_row(
    source: np.ndarray,
    nearest: np.ndarray,
    variant: np.ndarray,
    threshold: float,
) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    src_names, src_values = component_features("topo_source", source, threshold)
    near_names, near_values = component_features("topo_nearest", nearest, threshold)
    var_names, var_values = component_features("topo_variant", variant, threshold)
    for n, v in [(src_names, src_values), (near_names, near_values), (var_names, var_values)]:
        names.extend(n)
        values.extend(v)
    n, v = delta_features("topo_variant_nearest", var_names, var_values, near_values)
    names.extend(n)
    values.extend(v)
    n, v = delta_features("topo_source_variant", src_names, src_values, var_values)
    names.extend(n)
    values.extend(v)
    for prefix, a, b in [
        ("topo_pair_variant_nearest", variant, nearest),
        ("topo_pair_source_variant", source, variant),
        ("topo_pair_source_nearest", source, nearest),
    ]:
        n, v = pair_features(prefix, a, b, threshold)
        names.extend(n)
        values.extend(v)
    change_mask = (np.abs(variant - nearest) > (1.0 / 255.0)) | ((variant < threshold) ^ (nearest < threshold))
    sy, sx = crop_bbox(change_mask, pad=2)
    for prefix, image in [("topo_local_nearest", nearest[sy, sx]), ("topo_local_variant", variant[sy, sx])]:
        n, v = component_features(prefix, image, threshold)
        names.extend(n)
        values.extend(v)
    n, v = pair_features("topo_local_pair_variant_nearest", variant[sy, sx], nearest[sy, sx], threshold)
    names.extend(n)
    values.extend(v)
    return names, values


def build_topology_features(rows: list[dict[str, Any]], groups: dict[tuple[int, int, int, int], list[int]], threshold: float) -> tuple[np.ndarray, list[str]]:
    cache: dict[str, np.ndarray] = {}
    feature_rows: list[list[float] | None] = [None] * len(rows)
    feature_names: list[str] | None = None
    for group_number, indices in enumerate(groups.values(), start=1):
        nearest_idx = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
        nearest_row = rows[nearest_idx]
        source_path = resolve_path(str(nearest_row["source_image"]))
        nearest_path = resolve_path(str(nearest_row["image_path"]))
        for path in [source_path, nearest_path]:
            key = str(path)
            if key not in cache:
                cache[key] = load_gray(path)
        source = cache[str(source_path)]
        nearest = resize_like(cache[str(nearest_path)], source)
        for idx in indices:
            variant_path = resolve_path(str(rows[idx]["image_path"]))
            key = str(variant_path)
            if key not in cache:
                cache[key] = load_gray(variant_path)
            variant = resize_like(cache[key], source)
            names, values = topology_for_row(source, nearest, variant, threshold)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise RuntimeError("inconsistent topology feature names")
            feature_rows[idx] = values
        if group_number % 50 == 0:
            print(json.dumps({"groups": group_number, "rows": sum(v is not None for v in feature_rows)}), flush=True)
    if any(row is None for row in feature_rows):
        missing = sum(row is None for row in feature_rows)
        raise RuntimeError(f"missing topology rows: {missing}")
    assert feature_names is not None
    return np.asarray(feature_rows, dtype=np.float32), feature_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-model-seed", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows = base.read_jsonl(args.table)
    groups = base.grouped_indices(rows)
    base_features = base.build_base_features(rows)
    oracle_by_model = base.load_score_features(args.oracle_score)
    safe_by_model = base.load_score_features(args.safe_score)
    oracle_scores = np.asarray([oracle_by_model[args.score_model_seed][idx] for idx in range(len(rows))], dtype=np.float32)
    safe_scores = np.asarray([safe_by_model[args.score_model_seed][idx] for idx in range(len(rows))], dtype=np.float32)
    selector_features = base.build_selector_features(rows, groups, base_features, oracle_scores, safe_scores)
    topology_features, topology_names = build_topology_features(rows, groups, args.threshold)
    features = np.concatenate([selector_features.astype(np.float32), topology_features], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise SystemExit("non-finite feature cache")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_family": "text_topology",
        "role": "deployable_image_processing_diagnostic",
        "threshold": args.threshold,
        "topology_feature_dim": int(topology_features.shape[1]),
        "selector_feature_dim": int(selector_features.shape[1]),
        "feature_dim": int(features.shape[1]),
    }
    np.savez_compressed(
        args.output,
        features=features,
        topology_features=topology_features,
        topology_feature_names=np.asarray(topology_names),
        score_model_seed=np.asarray([args.score_model_seed], dtype=np.int64),
        metadata=np.asarray([json.dumps(metadata, ensure_ascii=False)]),
    )
    result = {
        "output": str(args.output),
        "rows": len(rows),
        "selector_feature_dim": int(selector_features.shape[1]),
        "topology_feature_dim": int(topology_features.shape[1]),
        "feature_dim": int(features.shape[1]),
        "sha256": sha256_file(args.output),
        "metadata": metadata,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
