#!/usr/bin/env python3
"""Aggregate PaddleOCR E2E original-image upper-bound summaries.

This is held-out evaluation only. It compares original crops against actual
decoded OSCR reconstructions, but must not be used for selector training or
threshold tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aggregate_paddleocr_e2e_eval import aggregate_condition, load_json, sha256_file


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# PaddleOCR E2E Eval300 Original Upper Bound",
        "",
        "Evaluator role: held-out PaddleOCR detector+recognizer; not used for selection, training, utility labels, feature extraction or threshold tuning.",
        "",
        "## Conditions",
        "",
    ]
    for label, row in result["conditions"].items():
        unicode_metrics = row["profiles"]["unicode_strict_v1"]
        latin_metrics = row["profiles"]["latin_alnum_ci_v1"]
        lines.append(
            f"- `{label}`: samples={row['samples']}, unicode CER={unicode_metrics['cer_micro']:.6f}, "
            f"unicode exact={unicode_metrics['exact_rate']:.6f}, latin CER={latin_metrics['cer_micro']:.6f}, "
            f"zero-det={row['zero_detection_samples']}, multi-det={row['multi_detection_samples']}, "
            f"mean_s={row['mean_infer_seconds']:.4f}"
        )

    lines.extend(["", "## Original To Reconstruction", ""])
    for label, row in result["comparisons"].items():
        unicode_cmp = row["aggregate"]["profiles"]["unicode_strict_v1"]
        latin_cmp = row["aggregate"]["profiles"]["latin_alnum_ci_v1"]
        lines.append(
            f"- `{label}`: unicode_delta_chars={unicode_cmp['delta_char_errors']}, "
            f"unicode_improved={unicode_cmp['improved_samples']}, "
            f"unicode_worsened={unicode_cmp['worsened_samples']}, "
            f"latin_delta_chars={latin_cmp['delta_char_errors']}"
        )

    lines.extend(["", "## Notes", ""])
    lines.append(
        "All actual OSCR reconstruction conditions are identical under PaddleOCR E2E, so the measured degradation "
        "relative to original crops is attributable to the frozen base/reconstruction path rather than the current "
        "OSCAR assignment variants. This is no-regression evidence only, not a deployable selection signal."
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
        "description": "Held-out PaddleOCR detector+recognizer original-image upper bound and actual OSCR comparison on Eval300.",
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
