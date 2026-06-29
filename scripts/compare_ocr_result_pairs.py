#!/usr/bin/env python3
"""Compare OCR result JSONL files row-by-row under text normalization profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, evaluate_pairs, normalize_text  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compare_rows(label: str, baseline_path: Path, candidate_path: Path, profiles: list[str]) -> dict:
    baseline = read_jsonl(baseline_path)
    candidate = read_jsonl(candidate_path)
    if len(baseline) != len(candidate):
        raise SystemExit(f"{label}: row count mismatch {len(baseline)} != {len(candidate)}")
    output = {
        "label": label,
        "baseline_results": str(baseline_path),
        "candidate_results": str(candidate_path),
        "samples": len(baseline),
        "profiles": {},
    }
    for profile in profiles:
        deltas = []
        exact_deltas = []
        base_pairs = []
        cand_pairs = []
        examples = []
        for index, (base_row, cand_row) in enumerate(zip(baseline, candidate)):
            if str(base_row["reference"]) != str(cand_row["reference"]):
                raise SystemExit(f"{label}: reference mismatch at row {index}")
            reference = str(base_row["reference"])
            base_pred = str(base_row["prediction"])
            cand_pred = str(cand_row["prediction"])
            base_counts = character_counts(reference, base_pred, profile)
            cand_counts = character_counts(reference, cand_pred, profile)
            delta = cand_counts.distance - base_counts.distance
            base_exact = normalize_text(reference, profile) == normalize_text(base_pred, profile)
            cand_exact = normalize_text(reference, profile) == normalize_text(cand_pred, profile)
            deltas.append(delta)
            exact_deltas.append(int(cand_exact) - int(base_exact))
            base_pairs.append((reference, base_pred))
            cand_pairs.append((reference, cand_pred))
            if delta != 0 or exact_deltas[-1] != 0:
                examples.append(
                    {
                        "index": index,
                        "reference": reference,
                        "baseline_prediction": base_pred,
                        "candidate_prediction": cand_pred,
                        "baseline_distance": base_counts.distance,
                        "candidate_distance": cand_counts.distance,
                        "delta_distance": delta,
                        "delta_exact": exact_deltas[-1],
                    }
                )
        base_metrics = evaluate_pairs(base_pairs, profile).to_dict()
        cand_metrics = evaluate_pairs(cand_pairs, profile).to_dict()
        output["profiles"][profile] = {
            "baseline": base_metrics,
            "candidate": cand_metrics,
            "delta_char_errors": int(sum(deltas)),
            "delta_cer_micro": (
                None
                if base_metrics["cer_micro"] is None or cand_metrics["cer_micro"] is None
                else float(cand_metrics["cer_micro"] - base_metrics["cer_micro"])
            ),
            "improved_samples": int(sum(1 for value in deltas if value < 0)),
            "worsened_samples": int(sum(1 for value in deltas if value > 0)),
            "unchanged_samples": int(sum(1 for value in deltas if value == 0)),
            "delta_exact_matches": int(sum(exact_deltas)),
            "changed_examples": examples,
        }
    return output


def aggregate(pairs: list[dict], profiles: list[str]) -> dict:
    output = {"profiles": {}}
    for profile in profiles:
        output["profiles"][profile] = {
            "pairs": len(pairs),
            "samples": int(sum(pair["samples"] for pair in pairs)),
            "delta_char_errors": int(sum(pair["profiles"][profile]["delta_char_errors"] for pair in pairs)),
            "improved_samples": int(sum(pair["profiles"][profile]["improved_samples"] for pair in pairs)),
            "worsened_samples": int(sum(pair["profiles"][profile]["worsened_samples"] for pair in pairs)),
            "unchanged_samples": int(sum(pair["profiles"][profile]["unchanged_samples"] for pair in pairs)),
            "delta_exact_matches": int(sum(pair["profiles"][profile]["delta_exact_matches"] for pair in pairs)),
        }
        base_errors = sum(
            pair["profiles"][profile]["baseline"]["char_substitutions"]
            + pair["profiles"][profile]["baseline"]["char_deletions"]
            + pair["profiles"][profile]["baseline"]["char_insertions"]
            for pair in pairs
        )
        cand_errors = sum(
            pair["profiles"][profile]["candidate"]["char_substitutions"]
            + pair["profiles"][profile]["candidate"]["char_deletions"]
            + pair["profiles"][profile]["candidate"]["char_insertions"]
            for pair in pairs
        )
        ref_chars = sum(pair["profiles"][profile]["baseline"]["reference_characters"] for pair in pairs)
        output["profiles"][profile].update(
            baseline_char_errors=int(base_errors),
            candidate_char_errors=int(cand_errors),
            reference_characters=int(ref_chars),
            baseline_cer_micro=None if ref_chars == 0 else float(base_errors / ref_chars),
            candidate_cer_micro=None if ref_chars == 0 else float(cand_errors / ref_chars),
            delta_cer_micro=None if ref_chars == 0 else float((cand_errors - base_errors) / ref_chars),
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=3, action="append", metavar=("LABEL", "BASELINE_RESULTS", "CANDIDATE_RESULTS"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["unicode_strict_v1", "latin_alnum_ci_v1", "raw_exact_v1"],
    )
    args = parser.parse_args()
    if not args.pair:
        raise SystemExit("at least one --pair is required")

    pairs = [compare_rows(label, Path(base), Path(candidate), args.profiles) for label, base, candidate in args.pair]
    result = {
        "description": "Row-wise OCR comparison; negative deltas mean the candidate has fewer edit errors.",
        "pairs": pairs,
        "aggregate": aggregate(pairs, args.profiles),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
