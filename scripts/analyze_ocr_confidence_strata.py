#!/usr/bin/env python3
"""Analyze OCR candidate behavior by confidence strata from a source teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import evaluate_pairs, normalize_text  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be label=path")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("candidate label is empty")
    return label, Path(path)


def confidence_bin(confidence: float, low_threshold: float, high_threshold: float) -> str:
    if confidence < low_threshold:
        return "low"
    if confidence < high_threshold:
        return "mid"
    return "high"


def metrics_or_none(pairs: list[tuple[str, str]], profile: str) -> dict | None:
    return evaluate_pairs(pairs, profile).to_dict() if pairs else None


def summarize_candidate(
    *,
    label: str,
    strata_source: list[dict],
    evaluator_original: list[dict],
    candidate: list[dict],
    profile: str,
    low_threshold: float,
    high_threshold: float,
) -> list[dict]:
    if len(strata_source) != len(evaluator_original) or len(strata_source) != len(candidate):
        raise ValueError(
            f"{label}: row count mismatch: strata={len(strata_source)}, "
            f"original={len(evaluator_original)}, candidate={len(candidate)}"
        )

    bins = ["low", "mid", "high"]
    groups: dict[str, dict] = {
        bin_name: {
            "confidences": [],
            "original_pairs": [],
            "candidate_pairs": [],
            "original_exact_pairs": [],
            "original_error_pairs": [],
            "changed_original_exact": 0,
            "worsened_original_exact": 0,
            "fixed_original_error": 0,
        }
        for bin_name in bins
    }

    for source, original, cand in zip(strata_source, evaluator_original, candidate):
        ref = original["reference"]
        if source["reference"] != ref or cand["reference"] != ref:
            raise ValueError(f"{label}: reference mismatch for {ref!r}")
        confidence = float(source["confidence"])
        bin_name = confidence_bin(confidence, low_threshold, high_threshold)
        group = groups[bin_name]
        orig_pred = original["prediction"]
        cand_pred = cand["prediction"]
        ref_n = normalize_text(ref, profile)
        orig_n = normalize_text(orig_pred, profile)
        cand_n = normalize_text(cand_pred, profile)

        group["confidences"].append(confidence)
        group["original_pairs"].append((ref, orig_pred))
        group["candidate_pairs"].append((ref, cand_pred))
        if orig_n == ref_n:
            group["original_exact_pairs"].append((ref, cand_pred))
            if cand_n != orig_n:
                group["changed_original_exact"] += 1
            if cand_n != ref_n:
                group["worsened_original_exact"] += 1
        else:
            group["original_error_pairs"].append((ref, cand_pred))
            if cand_n == ref_n:
                group["fixed_original_error"] += 1

    rows = []
    for bin_name in bins:
        group = groups[bin_name]
        confidences = group["confidences"]
        rows.append(
            {
                "label": label,
                "profile": profile,
                "stratum_source": "confidence",
                "stratum": bin_name,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "samples": len(confidences),
                "mean_source_confidence": statistics.fmean(confidences) if confidences else None,
                "min_source_confidence": min(confidences) if confidences else None,
                "max_source_confidence": max(confidences) if confidences else None,
                "original_exact_samples": len(group["original_exact_pairs"]),
                "original_error_samples": len(group["original_error_pairs"]),
                "changed_original_exact": group["changed_original_exact"],
                "worsened_original_exact": group["worsened_original_exact"],
                "fixed_original_error": group["fixed_original_error"],
                "original_all": metrics_or_none(group["original_pairs"], profile),
                "candidate_all": metrics_or_none(group["candidate_pairs"], profile),
                "candidate_on_original_exact": metrics_or_none(group["original_exact_pairs"], profile),
                "candidate_on_original_error": metrics_or_none(group["original_error_pairs"], profile),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata-results", type=Path, required=True)
    parser.add_argument("--evaluator-original-results", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--profile", action="append", default=["unicode_strict_v1", "latin_alnum_ci_v1"])
    parser.add_argument("--low-threshold", type=float, default=0.75)
    parser.add_argument("--high-threshold", type=float, default=0.98)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    strata_source = read_jsonl(args.strata_results)
    if any("confidence" not in row for row in strata_source):
        raise ValueError("strata-results must include a confidence field in every row")
    evaluator_original = read_jsonl(args.evaluator_original_results)

    rows = []
    for label, path in args.candidate:
        candidate = read_jsonl(path)
        for profile in args.profile:
            rows.extend(
                summarize_candidate(
                    label=label,
                    strata_source=strata_source,
                    evaluator_original=evaluator_original,
                    candidate=candidate,
                    profile=profile,
                    low_threshold=args.low_threshold,
                    high_threshold=args.high_threshold,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
