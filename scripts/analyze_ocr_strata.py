#!/usr/bin/env python3
"""Analyze OCR deltas stratified by original recognizability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def metrics_or_none(pairs: list[tuple[str, str]], profile: str) -> dict | None:
    return evaluate_pairs(pairs, profile).to_dict() if pairs else None


def summarize_candidate(label: str, original: list[dict], candidate: list[dict], profile: str) -> dict:
    if len(original) != len(candidate):
        raise ValueError(f"{label}: row count mismatch: {len(original)} vs {len(candidate)}")

    all_pairs = []
    original_exact_pairs = []
    original_error_pairs = []
    changed_original_exact = 0
    fixed_original_error = 0
    worsened_original_exact = 0
    for orig, cand in zip(original, candidate):
        ref = orig["reference"]
        orig_pred = orig["prediction"]
        cand_pred = cand["prediction"]
        ref_n = normalize_text(ref, profile)
        orig_n = normalize_text(orig_pred, profile)
        cand_n = normalize_text(cand_pred, profile)
        all_pairs.append((ref, cand_pred))
        if orig_n == ref_n:
            original_exact_pairs.append((ref, cand_pred))
            if cand_n != orig_n:
                changed_original_exact += 1
            if cand_n != ref_n:
                worsened_original_exact += 1
        else:
            original_error_pairs.append((ref, cand_pred))
            if cand_n == ref_n:
                fixed_original_error += 1

    return {
        "label": label,
        "profile": profile,
        "samples": len(original),
        "original_exact_samples": len(original_exact_pairs),
        "original_error_samples": len(original_error_pairs),
        "changed_original_exact": changed_original_exact,
        "worsened_original_exact": worsened_original_exact,
        "fixed_original_error": fixed_original_error,
        "candidate_all": metrics_or_none(all_pairs, profile),
        "candidate_on_original_exact": metrics_or_none(original_exact_pairs, profile),
        "candidate_on_original_error": metrics_or_none(original_error_pairs, profile),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-results", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--profile", action="append", default=["unicode_strict_v1", "latin_alnum_ci_v1"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = read_jsonl(args.original_results)
    rows = []
    for label, path in args.candidate:
        candidate = read_jsonl(path)
        for profile in args.profile:
            rows.append(summarize_candidate(label, original, candidate, profile))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
