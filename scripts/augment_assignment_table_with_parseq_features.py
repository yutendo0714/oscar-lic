#!/usr/bin/env python3
"""Add PARSeq teacher OCR confidence/edit-distance features to assignment rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + int(ca != cb)))
        prev = cur
    return prev[-1]


def norm_text(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", row.get("seed", 0)))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def code_key(row: dict) -> tuple[int, int, int, int, int]:
    return (*group_key(row), int(row["code_index"]))


def load_parseq_results(path: Path) -> dict[tuple[int, int, int, int, int], dict]:
    out = {}
    for row in read_jsonl(path):
        key = None
        if {"seed", "source_index", "candidate_index", "code_index"} <= set(row):
            key = (
                int(row.get("real_seed", row.get("seed", 0))),
                int(row["seed"]),
                int(row["source_index"]),
                int(row["candidate_index"]),
                int(row["code_index"]),
            )
        reference = str(row.get("reference", ""))
        prediction = str(row.get("prediction", ""))
        dist = levenshtein(reference, prediction)
        ref_norm = norm_text(reference)
        pred_norm = norm_text(prediction)
        norm_dist = levenshtein(ref_norm, pred_norm)
        features = {
            "prediction": prediction,
            "confidence": float(row.get("confidence", 0.0)),
            "token_count": int(row.get("token_count", 0)),
            "pred_length": len(prediction),
            "edit_distance_ref": dist,
            "ned_ref": float(dist / max(len(reference), 1)),
            "norm_edit_distance_ref": norm_dist,
            "norm_ned_ref": float(norm_dist / max(len(ref_norm), 1)),
        }
        if key is not None:
            out[key] = features
        image = row.get("image")
        if image is not None:
            out[("image", normalize_image_path(str(image)))] = features
    return out


def normalize_image_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path.resolve())
    return str(path)


def group_rows(rows: list[dict]) -> dict[tuple[int, int, int, int], list[dict]]:
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return dict(groups)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--parseq-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.table)
    parseq = load_parseq_results(args.parseq_results)
    output_rows = []
    missing = 0
    for group in group_rows(rows).values():
        nearest_row = next(row for row in group if int(row["is_nearest"]))
        nearest = parseq.get(code_key(nearest_row))
        if nearest is None:
            nearest = parseq.get(("image", normalize_image_path(str(nearest_row["image_path"]))))
        if nearest is None:
            missing += 1
            nearest = {
                "confidence": 0.0,
                "token_count": 0,
                "pred_length": 0,
                "edit_distance_ref": 0,
                "ned_ref": 0.0,
                "norm_edit_distance_ref": 0,
                "norm_ned_ref": 0.0,
            }
        for row in group:
            features = parseq.get(code_key(row))
            if features is None:
                features = parseq.get(("image", normalize_image_path(str(row["image_path"]))))
            if features is None:
                missing += 1
                features = nearest
            out = dict(row)
            for key, value in features.items():
                if key == "prediction":
                    continue
                out[f"img_teacher_parseq_{key}"] = value
            out["img_teacher_parseq_confidence_delta_vs_nearest"] = float(features["confidence"] - nearest["confidence"])
            out["img_teacher_parseq_edit_delta_vs_nearest"] = int(
                features["edit_distance_ref"] - nearest["edit_distance_ref"]
            )
            out["img_teacher_parseq_norm_edit_delta_vs_nearest"] = int(
                features["norm_edit_distance_ref"] - nearest["norm_edit_distance_ref"]
            )
            out["img_teacher_parseq_ned_delta_vs_nearest"] = float(features["ned_ref"] - nearest["ned_ref"])
            out["img_teacher_parseq_norm_ned_delta_vs_nearest"] = float(features["norm_ned_ref"] - nearest["norm_ned_ref"])
            output_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "description": "Assignment table augmented with PARSeq teacher confidence/edit-distance features.",
        "table": str(args.table),
        "parseq_results": str(args.parseq_results),
        "output": str(args.output),
        "rows": len(output_rows),
        "groups": len(group_rows(rows)),
        "missing_parseq_rows": int(missing),
        "features_added": ["img_teacher_parseq_*"],
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
