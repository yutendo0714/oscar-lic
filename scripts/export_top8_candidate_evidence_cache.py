#!/usr/bin/env python3
"""Export candidate-local evidence tensors for top-8 assignment diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
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


def resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT / value


def key3_from_glyph(row: dict[str, Any]) -> tuple[int, int, int]:
    key = row["key"]
    return (int(key["real_seed"]), int(key["source_index"]), int(key["candidate_index"]))


def key4_from_table(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"]), int(row["code_index"]))


def load_gray(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(resolve(path)).convert("L")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize(size, resampling)
    return np.asarray(image, dtype=np.float32) / 255.0


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def code_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names = []
    for key, value in rows[0].items():
        if not key.startswith("codebook_"):
            continue
        if "onehot" in key:
            continue
        if isinstance(value, (int, float)):
            names.append(key)
    return sorted(names)


def target_to_id(target: str) -> int:
    mapping = {
        "keep_noop": 0,
        "keep_current_good_change": 1,
        "recover_shortlist_oracle_change": 2,
        "reject_current_bad_change": 3,
    }
    return mapping.get(target, -1)


def build_cache(
    glyph: dict[str, Any],
    table_rows: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    table_by_key4 = {key4_from_table(row): row for row in table_rows}
    table_by_group: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in table_rows:
        table_by_group.setdefault((int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"])), []).append(row)
    for rows in table_by_group.values():
        rows.sort(key=lambda row: int(row["topk_rank"]))
    code_names = code_feature_names(table_rows)
    channel_names = [
        "source",
        "nearest",
        "candidate",
        "candidate_minus_nearest",
        "source_minus_nearest",
        "source_minus_candidate",
    ]
    records = glyph["candidate_records"]
    n = len(records)
    h, w = image_size[1], image_size[0]
    patches = np.zeros((n, len(channel_names), h, w), dtype=np.float16)
    code_features = np.zeros((n, len(code_names)), dtype=np.float32)
    glyph_features = np.zeros((n, len(sorted(records[0]["features"]))), dtype=np.float32)
    glyph_names = sorted(records[0]["features"])
    keys = np.zeros((n, 4), dtype=np.int32)
    label_is_oracle = np.zeros((n,), dtype=np.int8)
    target_ids = np.zeros((n,), dtype=np.int8)
    deltas = np.zeros((n, 2), dtype=np.int16)
    targets: list[str] = []
    sources: list[str] = []
    references: list[str] = []
    image_paths: list[str] = []
    for index, record in enumerate(records):
        group_key = key3_from_glyph(record)
        code_index = int(record["code_index"])
        table_row = table_by_key4[group_key + (code_index,)]
        nearest_code = int(record["nearest_code"])
        nearest_row = table_by_key4[group_key + (nearest_code,)]
        source = load_gray(table_row["source_image"], image_size)
        nearest = load_gray(nearest_row["image_path"], image_size)
        candidate = load_gray(table_row["image_path"], image_size)
        patches[index, 0] = source.astype(np.float16)
        patches[index, 1] = nearest.astype(np.float16)
        patches[index, 2] = candidate.astype(np.float16)
        patches[index, 3] = (candidate - nearest).astype(np.float16)
        patches[index, 4] = (source - nearest).astype(np.float16)
        patches[index, 5] = (source - candidate).astype(np.float16)
        code_features[index] = np.asarray([finite_float(table_row.get(name)) for name in code_names], dtype=np.float32)
        glyph_features[index] = np.asarray([finite_float(record["features"].get(name)) for name in glyph_names], dtype=np.float32)
        keys[index] = np.asarray([group_key[0], group_key[1], group_key[2], code_index], dtype=np.int32)
        label_is_oracle[index] = int(record["is_oracle_code"])
        target = str(record["next_model_target"])
        target_ids[index] = target_to_id(target)
        deltas[index] = np.asarray(
            [int(table_row.get("tesseract_delta_vs_nearest", 0)), int(table_row.get("parseq_delta_vs_nearest", 0))],
            dtype=np.int16,
        )
        targets.append(target)
        sources.append(str(record["source"]))
        references.append(str(record["reference"]))
        image_paths.append(str(table_row["image_path"]))
    arrays = {
        "patches": patches,
        "keys": keys,
        "label_is_oracle": label_is_oracle,
        "target_ids": target_ids,
        "deltas_tesseract_parseq": deltas,
        "code_features": code_features,
        "glyph_features": glyph_features,
        "channel_names": np.asarray(channel_names),
        "code_feature_names": np.asarray(code_names),
        "glyph_feature_names": np.asarray(glyph_names),
        "targets": np.asarray(targets),
        "sources": np.asarray(sources),
        "references": np.asarray(references),
        "image_paths": np.asarray(image_paths),
    }
    summary = {
        "rows": n,
        "patch_shape": list(patches.shape),
        "patch_dtype": str(patches.dtype),
        "image_size_width_height": [image_size[0], image_size[1]],
        "channel_names": channel_names,
        "code_feature_count": len(code_names),
        "glyph_feature_count": len(glyph_names),
        "target_counts": dict(Counter(targets)),
        "source_counts": dict(Counter(sources)),
        "oracle_rows": int(label_is_oracle.sum()),
        "tesseract_delta_sum": int(deltas[:, 0].sum()),
        "parseq_delta_sum": int(deltas[:, 1].sum()),
    }
    return arrays, summary


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Candidate-local evidence cache for top-8 assignment diagnostics.",
        "This is infrastructure for future rankers, not a compression result.",
        "",
        "## Summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- patch shape: `{summary['patch_shape']}`",
        f"- patch dtype: `{summary['patch_dtype']}`",
        f"- image size `[width, height]`: `{summary['image_size_width_height']}`",
        f"- channels: `{', '.join(summary['channel_names'])}`",
        f"- code feature count: `{summary['code_feature_count']}`",
        f"- glyph feature count: `{summary['glyph_feature_count']}`",
        f"- oracle rows: `{summary['oracle_rows']}`",
        "",
        "## Target Counts",
        "",
    ]
    for key, value in sorted(summary["target_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Train a candidate scorer over non-nearest rows using patch channels plus code/glyph auxiliary features.",
            "- Keep no-op acceptance as a separate head; this cache is not sufficient by itself for policy promotion.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glyph-features", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_candidate_evidence_cache")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=64)
    args = parser.parse_args()

    glyph = read_json(args.glyph_features)
    table_rows = read_jsonl(args.candidate_table)
    arrays, summary = build_cache(glyph, table_rows, image_size=(args.width, args.height))
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    result = {
        "experiment_id": args.experiment_id,
        "validity": "infrastructure_cache_not_promoted_selector",
        "inputs": {
            "glyph_features": str(args.glyph_features),
            "glyph_features_sha256": sha256_file(args.glyph_features),
            "candidate_table": str(args.candidate_table),
            "candidate_table_sha256": sha256_file(args.candidate_table),
        },
        "outputs": {
            "npz": str(args.output_npz),
            "npz_sha256": sha256_file(args.output_npz),
        },
        "summary": summary,
        "aggregate": {
            "scalar_metrics": {
                "rows": {"value": float(summary["rows"])},
                "oracle_rows": {"value": float(summary["oracle_rows"])},
                "code_feature_count": {"value": float(summary["code_feature_count"])},
                "glyph_feature_count": {"value": float(summary["glyph_feature_count"])},
            }
        },
    }
    write_json(args.summary_json, result)
    write_report(args.report, result)
    print(json.dumps({"npz": str(args.output_npz), "summary": str(args.summary_json), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
