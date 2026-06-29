#!/usr/bin/env python3
"""Summarize one-candidate utility tables by source and OCR-difficulty strata."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("empty label")
    return label, Path(path)


def tesseract_bucket(distance: int) -> str:
    if distance <= 0:
        return "0_exact"
    if distance == 1:
        return "1_char"
    if distance <= 3:
        return "2_3_chars"
    return "4plus_chars"


def parseq_bucket(distance: int) -> str:
    if distance <= 0:
        return "0_exact"
    if distance == 1:
        return "1_char"
    return "2plus_chars"


def area_bucket(area: float) -> str:
    if area < 2048:
        return "lt2k"
    if area < 4096:
        return "2k_4k"
    if area < 8192:
        return "4k_8k"
    return "8k_plus"


def safe_tesseract_improves(row: dict) -> bool:
    return int(row["tesseract_delta_distance"]) < 0 and int(row["parseq_delta_distance"]) <= 0


def summarize_records(records: list[dict]) -> dict:
    if not records:
        return {}
    images = len(records)
    candidate_rows = sum(int(row["candidate_rows"]) for row in records)
    safe_images = sum(int(row["has_tess_parseq_safe"]) for row in records)
    multi_images = sum(int(row["has_multi_teacher"]) for row in records)
    harmful_images = sum(int(row["has_harm"]) for row in records)
    return {
        "images": images,
        "candidate_rows": candidate_rows,
        "mean_candidates_per_image": candidate_rows / images,
        "safe_tess_parseq_images": safe_images,
        "safe_tess_parseq_image_rate": safe_images / images,
        "multi_teacher_images": multi_images,
        "multi_teacher_image_rate": multi_images / images,
        "harmful_images": harmful_images,
        "harmful_image_rate": harmful_images / images,
        "safe_tess_parseq_candidate_rows": sum(int(row["safe_tess_parseq_candidate_rows"]) for row in records),
        "multi_teacher_candidate_rows": sum(int(row["multi_teacher_candidate_rows"]) for row in records),
        "harmful_candidate_rows": sum(int(row["harmful_candidate_rows"]) for row in records),
        "tess_safe_oracle_delta_sum": sum(int(row["best_tess_parseq_safe_delta_or_zero"]) for row in records),
        "multi_teacher_oracle_delta_sum": sum(int(row["best_multi_teacher_delta_or_zero"]) for row in records),
        "mean_best_tess_parseq_safe_delta_or_zero": mean(
            float(row["best_tess_parseq_safe_delta_or_zero"]) for row in records
        ),
        "mean_best_multi_teacher_delta_or_zero": mean(float(row["best_multi_teacher_delta_or_zero"]) for row in records),
        "mean_parseq_base_distance": mean(float(row["parseq_base_distance"]) for row in records),
        "mean_tesseract_base_distance": mean(float(row["tesseract_base_distance"]) for row in records),
        "mean_image_area": mean(float(row["image_area"]) for row in records),
    }


def image_records(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_index"])].append(row)
    records = []
    for source_index, group in sorted(grouped.items()):
        first = group[0]
        safe_rows = [row for row in group if safe_tesseract_improves(row)]
        multi_rows = [row for row in group if int(row["label_multi_teacher_improves"]) > 0]
        harmful_rows = [row for row in group if int(row["worsening_evaluator_count"]) > 0]
        best_safe = min((int(row["tesseract_delta_distance"]) for row in safe_rows), default=0)
        best_multi = min((int(row["multi_teacher_delta_distance"]) for row in multi_rows), default=0)
        record = {
            "source_index": source_index,
            "source": str(first.get("source", "unknown")),
            "candidate_rows": len(group),
            "parseq_base_distance": int(first.get("parseq_base_distance", 0)),
            "tesseract_base_distance": int(first.get("tesseract_base_distance", 0)),
            "parseq_bucket": parseq_bucket(int(first.get("parseq_base_distance", 0))),
            "tesseract_bucket": tesseract_bucket(int(first.get("tesseract_base_distance", 0))),
            "image_area_bucket": area_bucket(float(first.get("image_area", 0.0))),
            "image_area": float(first.get("image_area", 0.0)),
            "has_tess_parseq_safe": int(bool(safe_rows)),
            "has_multi_teacher": int(bool(multi_rows)),
            "has_harm": int(bool(harmful_rows)),
            "safe_tess_parseq_candidate_rows": len(safe_rows),
            "multi_teacher_candidate_rows": len(multi_rows),
            "harmful_candidate_rows": len(harmful_rows),
            "best_tess_parseq_safe_delta_or_zero": best_safe,
            "best_multi_teacher_delta_or_zero": best_multi,
        }
        records.append(record)
    return records


def grouped_summary(records: list[dict], field: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        buckets[str(record[field])].append(record)
    return {key: summarize_records(value) for key, value in sorted(buckets.items())}


def summarize_table(label: str, path: Path) -> dict:
    rows = read_jsonl(path)
    records = image_records(rows)
    return {
        "label": label,
        "path": str(path),
        "rows": len(rows),
        "images": len(records),
        "overall": summarize_records(records),
        "by_source": grouped_summary(records, "source"),
        "by_tesseract_base_distance": grouped_summary(records, "tesseract_bucket"),
        "by_parseq_base_distance": grouped_summary(records, "parseq_bucket"),
        "by_image_area": grouped_summary(records, "image_area_bucket"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=parse_named_path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = {"tables": [summarize_table(label, path) for label, path in args.table]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tables": len(output["tables"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
