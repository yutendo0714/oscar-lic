#!/usr/bin/env python3
"""Diagnostic glyph-code differential features for top-8 assignment candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT / value


def group_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"]))


def load_gray(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize(size, resampling)
    return np.asarray(image, dtype=np.float32) / 255.0


def otsu_threshold(gray: np.ndarray) -> float:
    hist, _ = np.histogram(np.clip(gray, 0.0, 1.0), bins=256, range=(0.0, 1.0))
    total = float(gray.size)
    if total <= 0:
        return 0.5
    prob = hist.astype(np.float64) / total
    centers = (np.arange(256, dtype=np.float64) + 0.5) / 256.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    valid = denom > 1e-12
    sigma = np.zeros_like(denom)
    sigma[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return float(centers[int(np.argmax(sigma))])


def ink_mask(gray: np.ndarray) -> np.ndarray:
    threshold = otsu_threshold(gray)
    # Avoid classifying bright background texture as ink on nearly blank crops.
    threshold = min(threshold, float(np.quantile(gray, 0.65)))
    return gray < threshold


def shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    ys = slice(max(0, dy), min(h, h + dy))
    xs = slice(max(0, dx), min(w, w + dx))
    yt = slice(max(0, -dy), min(h, h - dy))
    xt = slice(max(0, -dx), min(w, w - dx))
    out[yt, xt] = mask[ys, xs]
    return out


def neighbor_any(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out |= shift(mask, dy, dx)
    return out


def connected_components(mask: np.ndarray) -> list[dict[str, float]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, float]] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=False):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        count = 0
        min_y = max_y = start_y
        min_x = max_x = start_x
        while stack:
            y, x = stack.pop()
            count += 1
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        bbox_area = float((max_y - min_y + 1) * (max_x - min_x + 1))
        components.append(
            {
                "area": float(count),
                "bbox_area": bbox_area,
                "bbox_width": float(max_x - min_x + 1),
                "bbox_height": float(max_y - min_y + 1),
            }
        )
    components.sort(key=lambda item: item["area"], reverse=True)
    return components


def entropy_from_projection(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    prob = values.astype(np.float64) / total
    prob = prob[prob > 0.0]
    denom = math.log(max(len(values), 2))
    return float(-(prob * np.log(prob)).sum() / denom)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


class ImageCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[int, int] | None], np.ndarray] = {}

    def gray(self, path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
        key = (str(path), size)
        if key not in self._cache:
            self._cache[key] = load_gray(resolve_path(path), size=size)
        return self._cache[key]


def glyph_features(source: np.ndarray, nearest: np.ndarray, variant: np.ndarray) -> dict[str, float]:
    eps = 1e-8
    change = variant - nearest
    desired = source - nearest
    abs_change = np.abs(change)
    threshold = max(1.0 / 255.0, float(np.quantile(abs_change, 0.995)) * 0.25)
    changed = abs_change > threshold
    changed_count = float(changed.sum())
    pixels = float(changed.size)
    added_dark = change < -threshold
    removed_dark = change > threshold
    source_ink = ink_mask(source)
    nearest_ink = ink_mask(nearest)
    variant_ink = ink_mask(variant)

    nearest_error = (source - nearest) ** 2
    variant_error = (source - variant) ** 2
    source_mse_delta = float(nearest_error.mean() - variant_error.mean())
    changed_source_mse_delta = float((nearest_error[changed].mean() - variant_error[changed].mean())) if changed_count else 0.0
    changed_toward = (np.abs(source - variant) + eps) < (np.abs(source - nearest) - eps)
    sign_aligned = (change * desired) > 0.0

    dark_action_correct = (added_dark & source_ink) | (removed_dark & ~source_ink)
    added_count = float(added_dark.sum())
    removed_count = float(removed_dark.sum())
    component_rows = connected_components(changed)
    largest_area = component_rows[0]["area"] if component_rows else 0.0
    largest_bbox_area = component_rows[0]["bbox_area"] if component_rows else 0.0
    column_energy = abs_change.sum(axis=0)
    row_energy = abs_change.sum(axis=1)
    nearest_neighbors = neighbor_any(nearest_ink)
    variant_neighbors = neighbor_any(variant_ink)
    bridge_horizontal = added_dark & shift(nearest_ink, 0, -1) & shift(nearest_ink, 0, 1)
    bridge_vertical = added_dark & shift(nearest_ink, -1, 0) & shift(nearest_ink, 1, 0)
    nearest_components = connected_components(nearest_ink)
    variant_components = connected_components(variant_ink)

    return {
        "glyph_changed_fraction": safe_ratio(changed_count, pixels),
        "glyph_abs_change_mean": float(abs_change.mean()),
        "glyph_abs_change_changed_mean": float(abs_change[changed].mean()) if changed_count else 0.0,
        "glyph_source_mse_delta": source_mse_delta,
        "glyph_source_mse_delta_changed": changed_source_mse_delta,
        "glyph_toward_source_fraction_changed": safe_ratio(float((changed & changed_toward).sum()), changed_count),
        "glyph_sign_alignment_fraction_changed": safe_ratio(float((changed & sign_aligned).sum()), changed_count),
        "glyph_added_dark_fraction": safe_ratio(added_count, pixels),
        "glyph_removed_dark_fraction": safe_ratio(removed_count, pixels),
        "glyph_added_dark_source_ink_fraction": safe_ratio(float((added_dark & source_ink).sum()), added_count),
        "glyph_removed_dark_source_light_fraction": safe_ratio(float((removed_dark & ~source_ink).sum()), removed_count),
        "glyph_dark_action_source_consistency": safe_ratio(float(dark_action_correct.sum()), added_count + removed_count),
        "glyph_changed_adjacent_to_nearest_ink_fraction": safe_ratio(float((changed & nearest_neighbors).sum()), changed_count),
        "glyph_changed_adjacent_to_variant_ink_fraction": safe_ratio(float((changed & variant_neighbors).sum()), changed_count),
        "glyph_added_bridge_horizontal_fraction": safe_ratio(float(bridge_horizontal.sum()), added_count),
        "glyph_added_bridge_vertical_fraction": safe_ratio(float(bridge_vertical.sum()), added_count),
        "glyph_diff_component_count": float(len(component_rows)),
        "glyph_diff_component_largest_fraction": safe_ratio(largest_area, changed_count),
        "glyph_diff_component_largest_bbox_fraction": safe_ratio(largest_bbox_area, pixels),
        "glyph_diff_component_largest_fill": safe_ratio(largest_area, largest_bbox_area),
        "glyph_x_projection_entropy": entropy_from_projection(column_energy),
        "glyph_y_projection_entropy": entropy_from_projection(row_energy),
        "glyph_x_projection_max_fraction": safe_ratio(float(column_energy.max()) if column_energy.size else 0.0, float(column_energy.sum())),
        "glyph_y_projection_max_fraction": safe_ratio(float(row_energy.max()) if row_energy.size else 0.0, float(row_energy.sum())),
        "glyph_ink_component_delta": float(len(variant_components) - len(nearest_components)),
        "glyph_ink_area_delta_fraction": safe_ratio(float(variant_ink.sum() - nearest_ink.sum()), pixels),
    }


SCORES: dict[str, str] = {
    "source_mse_delta": "glyph_source_mse_delta",
    "source_mse_delta_changed": "glyph_source_mse_delta_changed",
    "toward_source": "glyph_toward_source_fraction_changed",
    "sign_alignment": "glyph_sign_alignment_fraction_changed",
    "dark_action_consistency": "glyph_dark_action_source_consistency",
    "adjacent_nearest_ink": "glyph_changed_adjacent_to_nearest_ink_fraction",
    "bridge_horizontal": "glyph_added_bridge_horizontal_fraction",
    "diff_largest_component": "glyph_diff_component_largest_fraction",
}


def rank_desc(rows: list[dict[str, Any]], score_key: str, code: int) -> int | None:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row["features"].get(score_key, 0.0)), int(row["topk_rank"]), int(row["code_index"])),
    )
    for index, row in enumerate(ordered, start=1):
        if int(row["code_index"]) == code:
            return index
    return None


def summarize_ranks(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for score_name in SCORES:
        target_ranks = [row["score_ranks"][score_name]["oracle_rank"] for row in records if row["score_ranks"][score_name]["oracle_rank"] is not None]
        bad_ranks = [row["score_ranks"][score_name].get("current_bad_rank") for row in records if row["score_ranks"][score_name].get("current_bad_rank") is not None]
        out[score_name] = {
            "oracle_rank_count": len(target_ranks),
            "oracle_rank_mean": float(np.mean(target_ranks)) if target_ranks else None,
            "oracle_rank_le_1": int(sum(rank <= 1 for rank in target_ranks)),
            "oracle_rank_le_2": int(sum(rank <= 2 for rank in target_ranks)),
            "oracle_rank_le_4": int(sum(rank <= 4 for rank in target_ranks)),
            "current_bad_rank_mean": float(np.mean(bad_ranks)) if bad_ranks else None,
            "current_bad_rank_le_1": int(sum(rank <= 1 for rank in bad_ranks)),
            "current_bad_rank_le_4": int(sum(rank <= 4 for rank in bad_ranks)),
        }
    return out


def feature_stats(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row["features"][key]) for row in rows if key in row["features"]]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            out[key] = {
                "count": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return out


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Diagnostic glyph-code differential features for top-8 assignment candidates.",
        "The labels come from held-out OCR failure-bank strata; these OCR strings are not deployable features.",
        "",
        "## Summary",
        "",
        f"- Groups: `{summary['groups']}`",
        f"- Candidate rows: `{summary['candidate_rows']}`",
        f"- Target counts: `{summary['target_counts']}`",
        "",
        "## Oracle Rank By Glyph Score",
        "",
        "Ranks are shown by target because keep-noop groups use the nearest code as the oracle target.",
        "",
        "| target | score | oracle rank mean | rank<=1 | rank<=2 | rank<=4 | bad rank mean | bad rank<=1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target, rank_summary in summary["rank_summary_by_target"].items():
        for score_name, values in rank_summary.items():
            lines.append(
                f"| {target} | `{score_name}` | {values['oracle_rank_mean']} | {values['oracle_rank_le_1']} | "
                f"{values['oracle_rank_le_2']} | {values['oracle_rank_le_4']} | "
                f"{values['current_bad_rank_mean']} | {values['current_bad_rank_le_1']} |"
            )
    lines.extend(["", "## Target Feature Means", ""])
    stat_keys = [
        "glyph_source_mse_delta",
        "glyph_source_mse_delta_changed",
        "glyph_toward_source_fraction_changed",
        "glyph_sign_alignment_fraction_changed",
        "glyph_dark_action_source_consistency",
        "glyph_changed_adjacent_to_nearest_ink_fraction",
        "glyph_diff_component_largest_fraction",
        "glyph_diff_component_largest_bbox_fraction",
    ]
    lines.append("| target | feature | mean | min | max |")
    lines.append("|---|---|---:|---:|---:|")
    for target, stats in summary["feature_stats_by_target"].items():
        for key in stat_keys:
            item = stats.get(key)
            if item:
                lines.append(f"| {target} | `{key}` | {item['mean']} | {item['min']} | {item['max']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a representation diagnostic, not a promoted selector.",
            "- A useful next feature family should rank recover targets high without also ranking the current bad-change candidate high.",
            "- If the simple glyph scores still rank the bad change near the top, the next step should combine them with stricter abstention or richer OCR/text-aware evidence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_glyphcode_differential_features")
    args = parser.parse_args()

    bank = read_json(args.bank)
    cases = {group_key(row["key"]): row for row in bank["cases"]}
    rows_by_group: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.candidate_table):
        key = group_key(row)
        if key in cases:
            rows_by_group[key].append(row)
    missing_groups = sorted(set(cases) - set(rows_by_group))
    if missing_groups:
        raise SystemExit(f"missing candidate groups: {missing_groups[:5]}")

    cache = ImageCache()
    candidate_records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    for key in sorted(cases):
        case = cases[key]
        rows = sorted(rows_by_group[key], key=lambda item: int(item["topk_rank"]))
        nearest_rows = [row for row in rows if int(row["code_index"]) == int(case["nearest_code"])]
        if not nearest_rows:
            raise SystemExit(f"missing nearest row for {key}")
        nearest_row = nearest_rows[0]
        nearest = cache.gray(nearest_row["image_path"])
        size = (nearest.shape[1], nearest.shape[0])
        source = cache.gray(nearest_row["source_image"], size=size)
        enriched_rows = []
        for row in rows:
            variant = cache.gray(row["image_path"], size=size)
            features = glyph_features(source, nearest, variant)
            out = {
                "key": {"real_seed": key[0], "source_index": key[1], "candidate_index": key[2]},
                "source": case["source"],
                "reference": case["reference"],
                "next_model_target": case["next_model_target"],
                "code_index": int(row["code_index"]),
                "nearest_code": int(case["nearest_code"]),
                "oracle_code": int(case["oracle_code"]),
                "current_code": int(case["current_code"]),
                "topk_rank": int(row["topk_rank"]),
                "is_oracle_code": int(int(row["code_index"]) == int(case["oracle_code"])),
                "is_current_bad_code": int(
                    case["next_model_target"] == "reject_current_bad_change"
                    and int(row["code_index"]) == int(case["current_code"])
                ),
                "features": features,
            }
            enriched_rows.append(out)
            candidate_records.append(out)
        score_ranks: dict[str, dict[str, int | None]] = {}
        for score_name, feature_key in SCORES.items():
            score_ranks[score_name] = {
                "oracle_rank": rank_desc(enriched_rows, feature_key, int(case["oracle_code"])),
                "current_bad_rank": (
                    rank_desc(enriched_rows, feature_key, int(case["current_code"]))
                    if case["next_model_target"] == "reject_current_bad_change"
                    else None
                ),
            }
        group_records.append(
            {
                "key": {"real_seed": key[0], "source_index": key[1], "candidate_index": key[2]},
                "source": case["source"],
                "reference": case["reference"],
                "next_model_target": case["next_model_target"],
                "nearest_code": int(case["nearest_code"]),
                "oracle_code": int(case["oracle_code"]),
                "current_code": int(case["current_code"]),
                "score_ranks": score_ranks,
            }
        )

    target_counts = Counter(row["next_model_target"] for row in group_records)
    feature_keys = sorted(next(iter(candidate_records))["features"].keys())
    by_target_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_records:
        if row["is_oracle_code"]:
            by_target_records[row["next_model_target"]].append(row)
    by_target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_records:
        by_target_groups[row["next_model_target"]].append(row)
    summary = {
        "groups": len(group_records),
        "candidate_rows": len(candidate_records),
        "target_counts": dict(sorted(target_counts.items())),
        "rank_summary": summarize_ranks(group_records),
        "rank_summary_by_target": {
            target: summarize_ranks(rows) for target, rows in sorted(by_target_groups.items())
        },
        "feature_stats_by_target": {
            target: feature_stats(rows, feature_keys) for target, rows in sorted(by_target_records.items())
        },
    }
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_only_heldout_ocr_labels_not_features",
        "inputs": {
            "bank": str(args.bank),
            "candidate_table": str(args.candidate_table),
            "bank_sha256": sha256_file(args.bank),
            "candidate_table_sha256": sha256_file(args.candidate_table),
        },
        "score_definitions": SCORES,
        "summary": summary,
        "group_records": group_records,
        "candidate_records": candidate_records,
    }
    result["aggregate"] = {
        "scalar_metrics": {
            "groups": {"value": float(summary["groups"])},
            "candidate_rows": {"value": float(summary["candidate_rows"])},
            **{
                f"target_{key}": {"value": float(value)}
                for key, value in summary["target_counts"].items()
            },
            **{
                f"rank_{score}_{metric}": {"value": float(value)}
                for score, metrics in summary["rank_summary"].items()
                for metric, value in metrics.items()
                if isinstance(value, (int, float))
            },
        }
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
