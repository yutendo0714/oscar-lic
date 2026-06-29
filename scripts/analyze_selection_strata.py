#!/usr/bin/env python3
"""Analyze quality-selection OCR gains by source/domain strata."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import evaluate_pairs  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_q_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected QUALITY=PATH")
    quality, path = value.split("=", 1)
    return int(quality), Path(path)


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_quality_results(items: list[tuple[int, Path]]) -> dict[int, list[dict]]:
    result = {}
    for quality, path in items:
        if quality in result:
            raise ValueError(f"duplicate quality {quality}")
        result[quality] = read_jsonl(path)
    return result


def metric_for_rows(rows: list[dict], profile: str) -> dict:
    metrics = evaluate_pairs(((row["reference"], row["prediction"]) for row in rows), profile=profile)
    return metrics.to_dict()


def selected_rows(results_by_quality: dict[int, list[dict]], selected_qualities: list[int]) -> list[dict]:
    rows = []
    for index, quality in enumerate(selected_qualities):
        rows.append(results_by_quality[quality][index])
    return rows


def group_indices(split_rows: list[dict], field: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(split_rows):
        groups[str(row.get(field, "unknown"))].append(index)
    return dict(sorted(groups.items()))


def subset(rows: list[dict], indices: list[int]) -> list[dict]:
    return [rows[index] for index in indices]


def analyze_evaluator(
    *,
    label: str,
    results_by_quality: dict[int, list[dict]],
    selected_qualities: list[int],
    uniform_quality: int,
    groups: dict[str, list[int]],
    profile: str,
) -> dict:
    uniform_rows = results_by_quality[uniform_quality]
    oracle_rows = selected_rows(results_by_quality, selected_qualities)
    overall_uniform = metric_for_rows(uniform_rows, profile)
    overall_oracle = metric_for_rows(oracle_rows, profile)
    strata = []
    for group, indices in groups.items():
        uniform_metrics = metric_for_rows(subset(uniform_rows, indices), profile)
        oracle_metrics = metric_for_rows(subset(oracle_rows, indices), profile)
        strata.append(
            {
                "group": group,
                "samples": len(indices),
                "uniform": uniform_metrics,
                "selected": oracle_metrics,
                "delta_cer_selected_minus_uniform": (
                    oracle_metrics["cer_micro"] - uniform_metrics["cer_micro"]
                    if oracle_metrics["cer_micro"] is not None and uniform_metrics["cer_micro"] is not None
                    else None
                ),
                "delta_exact_selected_minus_uniform": (
                    oracle_metrics["exact_rate"] - uniform_metrics["exact_rate"]
                    if oracle_metrics["exact_rate"] is not None and uniform_metrics["exact_rate"] is not None
                    else None
                ),
            }
        )
    return {
        "label": label,
        "uniform_quality": uniform_quality,
        "overall": {
            "uniform": overall_uniform,
            "selected": overall_oracle,
            "delta_cer_selected_minus_uniform": overall_oracle["cer_micro"] - overall_uniform["cer_micro"],
            "delta_exact_selected_minus_uniform": overall_oracle["exact_rate"] - overall_uniform["exact_rate"],
        },
        "strata": strata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--uniform-quality", type=int, required=True)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--group-field", default="source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluator",
        action="append",
        nargs="+",
        metavar=("LABEL", "QUALITY=PATH"),
        help="Evaluator label followed by one or more QUALITY=PATH result files.",
    )
    args = parser.parse_args()

    if not args.evaluator:
        raise SystemExit("provide at least one --evaluator LABEL Q=PATH ...")

    split_rows = read_jsonl(args.split)
    selection_rows = read_jsonl(args.selection)
    if len(selection_rows) != len(split_rows):
        raise SystemExit("selection length does not match split length")
    selected_qualities = [int(row["selected_quality"]) for row in selection_rows]
    groups = group_indices(split_rows, args.group_field)

    evaluators = []
    for spec in args.evaluator:
        label = spec[0]
        quality_paths = [parse_q_path(item) for item in spec[1:]]
        results_by_quality = load_quality_results(quality_paths)
        missing = sorted(set(selected_qualities + [args.uniform_quality]) - set(results_by_quality))
        if missing:
            raise SystemExit(f"{label} is missing quality results: {missing}")
        lengths = {quality: len(rows) for quality, rows in results_by_quality.items()}
        if any(length != len(split_rows) for length in lengths.values()):
            raise SystemExit(f"{label} result length mismatch: {lengths}")
        evaluators.append(
            analyze_evaluator(
                label=label,
                results_by_quality=results_by_quality,
                selected_qualities=selected_qualities,
                uniform_quality=args.uniform_quality,
                groups=groups,
                profile=args.profile,
            )
        )

    quality_counts: dict[str, int] = defaultdict(int)
    for quality in selected_qualities:
        quality_counts[str(quality)] += 1
    output = {
        "split": project_path(args.split),
        "selection": project_path(args.selection),
        "samples": len(split_rows),
        "profile": args.profile,
        "group_field": args.group_field,
        "quality_counts": dict(sorted(quality_counts.items(), key=lambda item: int(item[0]))),
        "groups": {group: len(indices) for group, indices in groups.items()},
        "evaluators": evaluators,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "evaluators": len(evaluators), "groups": len(groups)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
