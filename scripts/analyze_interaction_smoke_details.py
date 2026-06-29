#!/usr/bin/env python3
"""Analyze per-sample OCR interactions in multi-candidate smoke runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts  # noqa: E402


PROFILES = ("unicode_strict_v1", "latin_alnum_ci_v1")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_ocr_set(value: str) -> tuple[str, Path, Path, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("OCR set must be label=base,single_a,single_b,pair_ab")
    label, paths = parts
    path_parts = paths.split(",")
    if len(path_parts) != 4:
        raise argparse.ArgumentTypeError("OCR set must contain four comma-separated paths")
    return label, *(Path(part) for part in path_parts)


def row_distance(row: dict, profile: str) -> int:
    return character_counts(str(row["reference"]), str(row["prediction"]), profile).distance


def summarize_values(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def safe_div(num: float, den: float) -> float | None:
    if abs(den) < 1e-12:
        return None
    return num / den


def load_stream_rows(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    rows = read_jsonl(path)
    return rows


def candidate_summary_by_index(policy_summary: dict) -> dict[int, dict]:
    by_index = {}
    for item in policy_summary.get("sources", []):
        by_index[int(item["new_source_index"])] = item
    return by_index


def analyze_one_ocr(
    label: str,
    base_path: Path,
    single_a_path: Path,
    single_b_path: Path,
    pair_path: Path,
    policy_by_index: dict[int, dict],
    stream_rows: dict[str, list[dict] | None],
) -> dict:
    base_rows = read_jsonl(base_path)
    a_rows = read_jsonl(single_a_path)
    b_rows = read_jsonl(single_b_path)
    pair_rows = read_jsonl(pair_path)
    lengths = {len(base_rows), len(a_rows), len(b_rows), len(pair_rows)}
    if len(lengths) != 1:
        raise SystemExit(f"{label}: OCR result lengths differ: {sorted(lengths)}")

    out_profiles = {}
    for profile in PROFILES:
        sample_rows = []
        for index, (base, a, b, pair) in enumerate(zip(base_rows, a_rows, b_rows, pair_rows, strict=True)):
            refs = {base["reference"], a["reference"], b["reference"], pair["reference"]}
            if len(refs) != 1:
                raise SystemExit(f"{label}: reference mismatch at row {index}: {refs}")
            base_d = row_distance(base, profile)
            a_d = row_distance(a, profile)
            b_d = row_distance(b, profile)
            pair_d = row_distance(pair, profile)
            delta_a = a_d - base_d
            delta_b = b_d - base_d
            pair_delta = pair_d - base_d
            additive = delta_a + delta_b
            interaction_residual = pair_delta - additive
            best_single_delta = min(delta_a, delta_b)
            worst_single_delta = max(delta_a, delta_b)
            pair_vs_best = pair_delta - best_single_delta
            pair_vs_worst = pair_delta - worst_single_delta
            cond_b_after_a = pair_delta - delta_a
            cond_a_after_b = pair_delta - delta_b

            row = {
                "index": index,
                "reference": base["reference"],
                "source": base.get("source"),
                "base_prediction": base["prediction"],
                "single_a_prediction": a["prediction"],
                "single_b_prediction": b["prediction"],
                "pair_prediction": pair["prediction"],
                "base_distance": base_d,
                "single_a_distance": a_d,
                "single_b_distance": b_d,
                "pair_distance": pair_d,
                "single_a_delta": delta_a,
                "single_b_delta": delta_b,
                "pair_delta": pair_delta,
                "additive_expected_delta": additive,
                "interaction_residual": interaction_residual,
                "best_single_delta": best_single_delta,
                "pair_vs_best_single_delta": pair_vs_best,
                "pair_vs_worst_single_delta": pair_vs_worst,
                "conditional_delta_b_after_a": cond_b_after_a,
                "conditional_delta_a_after_b": cond_a_after_b,
            }
            if index in policy_by_index:
                item = policy_by_index[index]
                row["original_source_index"] = int(item["original_source_index"])
                row["candidate_indices"] = [int(c["candidate_index"]) for c in item.get("candidates", [])]
                row["table_expected_parseq_delta_sum"] = int(
                    sum(int(c.get("parseq_delta_distance", 0)) for c in item.get("candidates", []))
                )
                row["table_expected_tesseract_delta_sum"] = int(
                    sum(int(c.get("tesseract_delta_distance", 0)) for c in item.get("candidates", []))
                )
            for stream_name, rows in stream_rows.items():
                if rows is None:
                    continue
                row[f"{stream_name}_actual_total_bpp"] = float(rows[index]["actual_total_bpp"])
                row[f"{stream_name}_enhancement_payload_bpp"] = float(rows[index]["enhancement_payload_bpp"])
                row[f"{stream_name}_psnr_delta_db"] = float(rows[index]["psnr_delta_db"])
            if stream_rows.get("pair_ab") is not None and stream_rows.get("single_a") is not None:
                row["extra_total_bpp_pair_vs_single_a"] = (
                    row["pair_ab_actual_total_bpp"] - row["single_a_actual_total_bpp"]
                )
                row["conditional_improvement_per_extra_bpp_b_after_a"] = safe_div(
                    -cond_b_after_a, row["extra_total_bpp_pair_vs_single_a"]
                )
            if stream_rows.get("pair_ab") is not None and stream_rows.get("single_b") is not None:
                row["extra_total_bpp_pair_vs_single_b"] = (
                    row["pair_ab_actual_total_bpp"] - row["single_b_actual_total_bpp"]
                )
                row["conditional_improvement_per_extra_bpp_a_after_b"] = safe_div(
                    -cond_a_after_b, row["extra_total_bpp_pair_vs_single_b"]
                )
            sample_rows.append(row)

        sum_a = int(sum(row["single_a_delta"] for row in sample_rows))
        sum_b = int(sum(row["single_b_delta"] for row in sample_rows))
        sum_pair = int(sum(row["pair_delta"] for row in sample_rows))
        sum_add = int(sum(row["additive_expected_delta"] for row in sample_rows))
        sum_residual = int(sum(row["interaction_residual"] for row in sample_rows))
        sum_best = int(sum(row["best_single_delta"] for row in sample_rows))
        pair_vs_best = int(sum(row["pair_vs_best_single_delta"] for row in sample_rows))
        cond_b_after_a = int(sum(row["conditional_delta_b_after_a"] for row in sample_rows))
        cond_a_after_b = int(sum(row["conditional_delta_a_after_b"] for row in sample_rows))

        category = Counter()
        for row in sample_rows:
            if row["interaction_residual"] > 0:
                category["diminishing_or_conflict"] += 1
            elif row["interaction_residual"] < 0:
                category["superadditive"] += 1
            else:
                category["additive_exact"] += 1
            if row["pair_vs_best_single_delta"] < 0:
                category["pair_beats_best_single"] += 1
            elif row["pair_vs_best_single_delta"] == 0:
                category["pair_ties_best_single"] += 1
            else:
                category["pair_worse_than_best_single"] += 1
            if row["conditional_delta_b_after_a"] < 0:
                category["b_after_a_improves"] += 1
            elif row["conditional_delta_b_after_a"] == 0:
                category["b_after_a_neutral"] += 1
            else:
                category["b_after_a_worsens"] += 1
            if row["conditional_delta_a_after_b"] < 0:
                category["a_after_b_improves"] += 1
            elif row["conditional_delta_a_after_b"] == 0:
                category["a_after_b_neutral"] += 1
            else:
                category["a_after_b_worsens"] += 1

        additive_gain = -sum_add
        observed_pair_gain = -sum_pair
        best_single_gain = -sum_best
        out_profiles[profile] = {
            "samples": len(sample_rows),
            "sum_single_a_delta": sum_a,
            "sum_single_b_delta": sum_b,
            "sum_additive_expected_delta": sum_add,
            "sum_pair_delta": sum_pair,
            "sum_interaction_residual": sum_residual,
            "sum_best_single_delta": sum_best,
            "sum_pair_vs_best_single_delta": pair_vs_best,
            "sum_conditional_delta_b_after_a": cond_b_after_a,
            "sum_conditional_delta_a_after_b": cond_a_after_b,
            "observed_pair_gain_over_additive_gain_ratio": safe_div(observed_pair_gain, additive_gain),
            "observed_pair_gain_over_best_single_gain_ratio": safe_div(observed_pair_gain, best_single_gain),
            "categories": dict(category),
            "interaction_residual_distribution": summarize_values(row["interaction_residual"] for row in sample_rows),
            "pair_vs_best_single_distribution": summarize_values(
                row["pair_vs_best_single_delta"] for row in sample_rows
            ),
            "conditional_improvement_per_extra_bpp_b_after_a": summarize_values(
                value
                for row in sample_rows
                for value in [row.get("conditional_improvement_per_extra_bpp_b_after_a")]
                if value is not None
            ),
            "conditional_improvement_per_extra_bpp_a_after_b": summarize_values(
                value
                for row in sample_rows
                for value in [row.get("conditional_improvement_per_extra_bpp_a_after_b")]
                if value is not None
            ),
            "top_diminishing_examples": sorted(
                sample_rows,
                key=lambda row: (row["interaction_residual"], row["pair_vs_best_single_delta"]),
                reverse=True,
            )[:5],
            "top_superadditive_examples": sorted(
                sample_rows,
                key=lambda row: (row["interaction_residual"], row["pair_vs_best_single_delta"]),
            )[:5],
            "per_sample": sample_rows,
        }
    return {
        "label": label,
        "paths": {
            "base": str(base_path),
            "single_a": str(single_a_path),
            "single_b": str(single_b_path),
            "pair_ab": str(pair_path),
        },
        "profiles": out_profiles,
    }


def write_report(path: Path, result: dict) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "This is a diagnostic re-analysis of existing counted `.oscr` interaction smoke outputs.",
        "It does not train or promote a selector.",
        "",
        "## Summary",
        "",
    ]
    for ocr in result["ocr_results"]:
        lines.append(f"### {ocr['label']}")
        lines.append("")
        lines.append(
            "| profile | additive delta | pair delta | residual | pair vs best single | "
            "pair/additive gain | pair/best-single gain | pair beats/ties/worse best |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for profile, stats in ocr["profiles"].items():
            cats = stats["categories"]
            ratio_add = stats["observed_pair_gain_over_additive_gain_ratio"]
            ratio_best = stats["observed_pair_gain_over_best_single_gain_ratio"]
            ratio_add_s = "n/a" if ratio_add is None else f"{ratio_add:.4f}"
            ratio_best_s = "n/a" if ratio_best is None else f"{ratio_best:.4f}"
            beat = (
                f"{cats.get('pair_beats_best_single', 0)}/"
                f"{cats.get('pair_ties_best_single', 0)}/"
                f"{cats.get('pair_worse_than_best_single', 0)}"
            )
            lines.append(
                f"| {profile} | {stats['sum_additive_expected_delta']} | {stats['sum_pair_delta']} | "
                f"{stats['sum_interaction_residual']} | {stats['sum_pair_vs_best_single_delta']} | "
                f"{ratio_add_s} | {ratio_best_s} | {beat} |"
            )
        lines.append("")
        lines.append("Conditional second-candidate totals:")
        lines.append("")
        lines.append("| profile | B after A delta | A after B delta |")
        lines.append("|---|---:|---:|")
        for profile, stats in ocr["profiles"].items():
            lines.append(
                f"| {profile} | {stats['sum_conditional_delta_b_after_a']} | "
                f"{stats['sum_conditional_delta_a_after_b']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Negative deltas mean fewer OCR edit errors. A positive interaction residual means the pair "
            "underperforms the additive single-candidate utility estimate.",
            "",
            "The important allocator signal is `pair vs best single`: if this is positive, sending two "
            "individually useful candidates is worse than sending only the better single candidate. "
            "That directly argues for conditional marginal utility or subset-aware allocation rather "
            "than independent score summation.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_aggregate(result: dict) -> dict:
    scalar_metrics = {}
    for ocr in result["ocr_results"]:
        label = ocr["label"]
        for profile, stats in ocr["profiles"].items():
            prefix = f"{label}_{profile}"
            for key in [
                "sum_single_a_delta",
                "sum_single_b_delta",
                "sum_additive_expected_delta",
                "sum_pair_delta",
                "sum_interaction_residual",
                "sum_best_single_delta",
                "sum_pair_vs_best_single_delta",
                "sum_conditional_delta_b_after_a",
                "sum_conditional_delta_a_after_b",
            ]:
                scalar_metrics[f"{prefix}_{key}"] = stats[key]
            for key in [
                "observed_pair_gain_over_additive_gain_ratio",
                "observed_pair_gain_over_best_single_gain_ratio",
            ]:
                value = stats[key]
                if value is not None:
                    scalar_metrics[f"{prefix}_{key}"] = value
            for key, value in stats["categories"].items():
                scalar_metrics[f"{prefix}_{key}"] = value
    return {"scalar_metrics": {key: {"value": value} for key, value in scalar_metrics.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-summary", type=Path, required=True)
    parser.add_argument("--ocr-set", type=parse_ocr_set, action="append", required=True)
    parser.add_argument("--single-a-stream-results", type=Path)
    parser.add_argument("--single-b-stream-results", type=Path)
    parser.add_argument("--pair-stream-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_interaction_smoke20_detail_audit_2026_06_26")
    args = parser.parse_args()

    policy_summary = read_json(args.policy_summary)
    policy_by_index = candidate_summary_by_index(policy_summary)
    stream_rows = {
        "single_a": load_stream_rows(args.single_a_stream_results),
        "single_b": load_stream_rows(args.single_b_stream_results),
        "pair_ab": load_stream_rows(args.pair_stream_results),
    }
    result = {
        "experiment_id": args.experiment_id,
        "description": "Per-sample conditional utility audit for existing multi-candidate interaction smoke.",
        "validity": "diagnostic_only_not_selector_training",
        "policy_summary": str(args.policy_summary),
        "stream_results": {
            key: str(path)
            for key, path in {
                "single_a": args.single_a_stream_results,
                "single_b": args.single_b_stream_results,
                "pair_ab": args.pair_stream_results,
            }.items()
            if path is not None
        },
        "ocr_results": [
            analyze_one_ocr(label, base, single_a, single_b, pair, policy_by_index, stream_rows)
            for label, base, single_a, single_b, pair in args.ocr_set
        ],
    }
    result["aggregate"] = build_aggregate(result)
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
