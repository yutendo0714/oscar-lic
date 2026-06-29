#!/usr/bin/env python3
"""Compare single-candidate OCR utility against base OCR results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, evaluate_pairs, normalize_text  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def distance(reference: str, prediction: str, profile: str) -> int:
    return character_counts(reference, prediction, profile).distance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-split", type=Path, required=True)
    parser.add_argument("--candidate-ocr", type=Path, required=True)
    parser.add_argument("--base-ocr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="unicode_strict_v1")
    args = parser.parse_args()

    split_rows = read_jsonl(args.candidate_split)
    cand_rows = read_jsonl(args.candidate_ocr)
    base_rows = read_jsonl(args.base_ocr)
    if len(split_rows) != len(cand_rows):
        raise SystemExit("candidate split and OCR result lengths differ")

    details = []
    by_image = defaultdict(list)
    for split_row, cand_row in zip(split_rows, cand_rows):
        source_index = int(split_row["source_index"])
        base_row = base_rows[source_index]
        reference = cand_row["reference"]
        base_distance = distance(reference, base_row["prediction"], args.profile)
        candidate_distance = distance(reference, cand_row["prediction"], args.profile)
        ref_norm = normalize_text(reference, args.profile)
        base_exact = int(ref_norm == normalize_text(base_row["prediction"], args.profile))
        candidate_exact = int(ref_norm == normalize_text(cand_row["prediction"], args.profile))
        row = {
            **{key: split_row[key] for key in [
                "source_index",
                "candidate_index",
                "slice_index",
                "y0",
                "y1",
                "x0",
                "x1",
                "channel0",
                "channel1",
                "actual_total_bpp",
                "psnr_enhanced_db",
            ]},
            "reference": reference,
            "base_prediction": base_row["prediction"],
            "candidate_prediction": cand_row["prediction"],
            "base_distance": base_distance,
            "candidate_distance": candidate_distance,
            "delta_distance_candidate_minus_base": candidate_distance - base_distance,
            "base_exact": base_exact,
            "candidate_exact": candidate_exact,
            "delta_exact_candidate_minus_base": candidate_exact - base_exact,
            "candidate_confidence": cand_row.get("confidence"),
            "source": split_row.get("source"),
        }
        details.append(row)
        by_image[source_index].append(row)

    improvement_counter = Counter()
    slice_delta = defaultdict(list)
    best_rows = []
    base_pairs = []
    best_pairs = []
    for row in details:
        delta = row["delta_distance_candidate_minus_base"]
        if delta < 0:
            improvement_counter["improved"] += 1
        elif delta > 0:
            improvement_counter["worsened"] += 1
        else:
            improvement_counter["same"] += 1
        slice_delta[row["slice_index"]].append(delta)

    for source_index, rows in sorted(by_image.items()):
        best = min(rows, key=lambda row: (row["candidate_distance"], -(row["candidate_confidence"] or 0.0)))
        base_distance = rows[0]["base_distance"]
        best_rows.append(
            {
                "source_index": source_index,
                "source": rows[0].get("source"),
                "candidate_count": len(rows),
                "reference": best["reference"],
                "base_prediction": best["base_prediction"],
                "best_prediction": best["candidate_prediction"],
                "base_distance": base_distance,
                "best_candidate_index": best["candidate_index"],
                "best_slice_index": best["slice_index"],
                "best_distance": best["candidate_distance"],
                "best_delta_distance": best["candidate_distance"] - base_distance,
                "base_exact": rows[0]["base_exact"],
                "best_exact": best["candidate_exact"],
                "best_delta_exact": best["candidate_exact"] - rows[0]["base_exact"],
            }
        )
        base_pairs.append((best["reference"], best["base_prediction"]))
        best_pairs.append((best["reference"], best["candidate_prediction"]))

    summary = {
        "profile": args.profile,
        "candidate_rows": len(details),
        "images": len(by_image),
        "candidate_delta_counts": dict(improvement_counter),
        "images_with_any_improving_candidate": sum(1 for row in best_rows if row["best_delta_distance"] < 0),
        "images_with_any_worsening_best": sum(1 for row in best_rows if row["best_delta_distance"] > 0),
        "mean_best_delta_distance": sum(row["best_delta_distance"] for row in best_rows) / len(best_rows),
        "base_corpus_metrics": evaluate_pairs(base_pairs, profile=args.profile).to_dict(),
        "best_single_candidate_oracle_metrics": evaluate_pairs(best_pairs, profile=args.profile).to_dict(),
        "slice_mean_delta_distance": {
            str(slice_index): sum(values) / len(values) for slice_index, values in sorted(slice_delta.items())
        },
        "best_slice_counts": dict(Counter(row["best_slice_index"] for row in best_rows)),
        "best_rows": best_rows,
    }
    output = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
