#!/usr/bin/env python3
"""Aggregate PaddleOCR E2E OCR summaries and row-wise comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sum(int(row["samples"]) for row in rows)
    out: dict[str, Any] = {
        "runs": len(rows),
        "samples": samples,
        "zero_detection_samples": sum(int(row.get("zero_detection_samples", 0)) for row in rows),
        "multi_detection_samples": sum(int(row.get("multi_detection_samples", 0)) for row in rows),
        "mean_infer_seconds": (
            sum(float(row.get("mean_infer_seconds", 0.0)) * int(row["samples"]) for row in rows) / samples
            if samples
            else None
        ),
        "profiles": {},
    }
    profiles = sorted({profile for row in rows for profile in row.get("metrics", {})})
    for profile in profiles:
        metrics = [row["metrics"][profile] for row in rows]
        ref_chars = sum(int(metric["reference_characters"]) for metric in metrics)
        ref_words = sum(int(metric["reference_words"]) for metric in metrics)
        char_errors = sum(
            int(metric["char_substitutions"]) + int(metric["char_deletions"]) + int(metric["char_insertions"])
            for metric in metrics
        )
        word_errors = sum(
            int(metric["word_substitutions"]) + int(metric["word_deletions"]) + int(metric["word_insertions"])
            for metric in metrics
        )
        exact = sum(int(metric["exact_matches"]) for metric in metrics)
        out["profiles"][profile] = {
            "reference_characters": ref_chars,
            "reference_words": ref_words,
            "char_errors": char_errors,
            "word_errors": word_errors,
            "cer_micro": None if ref_chars == 0 else char_errors / ref_chars,
            "wer_micro": None if ref_words == 0 else word_errors / ref_words,
            "exact_rate": None if samples == 0 else exact / samples,
        }
    return out


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# PaddleOCR E2E Eval300 Actual OSCR Evaluation",
        "",
        "Evaluator role: held-out PaddleOCR detector+recognizer; not used for selection, training, utility labels, feature extraction or threshold tuning.",
        "",
        "## Conditions",
        "",
    ]
    for label, row in result["conditions"].items():
        unicode_metrics = row["profiles"]["unicode_strict_v1"]
        lines.append(
            f"- `{label}`: samples={row['samples']}, unicode CER={unicode_metrics['cer_micro']:.6f}, "
            f"exact={unicode_metrics['exact_rate']:.6f}, zero-det={row['zero_detection_samples']}, "
            f"multi-det={row['multi_detection_samples']}, mean_s={row['mean_infer_seconds']:.4f}"
        )
    lines.extend(["", "## Comparisons", ""])
    for label, row in result["comparisons"].items():
        unicode_cmp = row["aggregate"]["profiles"]["unicode_strict_v1"]
        lines.append(
            f"- `{label}`: delta_char_errors={unicode_cmp['delta_char_errors']}, "
            f"improved={unicode_cmp['improved_samples']}, worsened={unicode_cmp['worsened_samples']}, "
            f"unchanged={unicode_cmp['unchanged_samples']}"
        )
    lines.extend(["", "## Notes", ""])
    lines.append(
        "PaddleOCR E2E is stricter than recognizer-only on these word crops because detection can miss the crop; "
        "the current-best and shortlist-oracle actual reconstructions remain exactly neutral versus nearest under this evaluator."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", nargs=2, action="append", metavar=("LABEL", "SUMMARY_JSON"), required=True)
    parser.add_argument("--comparison", nargs=2, action="append", metavar=("LABEL", "COMPARE_JSON"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    condition_groups: dict[str, list[dict[str, Any]]] = {}
    condition_inputs: dict[str, list[dict[str, str]]] = {}
    for label, summary_path in args.condition:
        path = Path(summary_path)
        condition_groups.setdefault(label, []).append(load_json(path))
        condition_inputs.setdefault(label, []).append({"path": str(path), "sha256": sha256_file(path)})

    comparisons: dict[str, Any] = {}
    comparison_inputs: dict[str, dict[str, str]] = {}
    for label, compare_path in args.comparison:
        path = Path(compare_path)
        comparisons[label] = load_json(path)
        comparison_inputs[label] = {"path": str(path), "sha256": sha256_file(path)}

    result = {
        "description": "Held-out PaddleOCR detector+recognizer evaluation on actual decoded Eval300 OSCR reconstructions.",
        "status": "completed_evaluation_only",
        "conditions": {label: aggregate_condition(rows) for label, rows in condition_groups.items()},
        "comparisons": comparisons,
        "condition_inputs": condition_inputs,
        "comparison_inputs": comparison_inputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
