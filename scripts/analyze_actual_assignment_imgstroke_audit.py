#!/usr/bin/env python3
"""Audit imgstroke candidate features on the actual assignment failure bank.

This is diagnostic-only. It uses N076 held-out OCR-derived strata to inspect
whether existing stroke/projection features could be auxiliary evidence. It does
not train or promote a selector.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


TARGET_ORDER = [
    "recover_shortlist_oracle_change",
    "reject_current_bad_change",
    "improve_first_stage_shortlist_recall",
    "keep_current_good_change",
    "keep_noop",
]


REPORT_FEATURES = [
    "img_variant_nearest_changed_fraction",
    "img_variant_nearest_mse",
    "img_variant_nearest_dark_changed_fraction",
    "img_variant_nearest_edge_changed_fraction",
    "img_variant_nearest_row_projection_abs_mean",
    "img_variant_nearest_col_projection_abs_mean",
    "img_variant_nearest_row_dark_projection_abs_mean",
    "img_variant_nearest_col_dark_projection_abs_mean",
    "img_variant_nearest_bbox_area_fraction",
    "img_source_variant_dark_union_abs_mean",
    "img_source_variant_edge_union_abs_mean",
    "img_source_variant_row_dark_projection_abs_mean",
    "img_source_variant_col_dark_projection_abs_mean",
    "img_source_edge_density",
    "img_source_dark050_fraction",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def key3_from_case(case: dict[str, Any]) -> tuple[int, int, int]:
    key = case["key"]
    return (int(key["real_seed"]), int(key["source_index"]), int(key["candidate_index"]))


def key3_from_row(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"]))


def key4_from_row(row: dict[str, Any]) -> tuple[int, int, int, int]:
    a, b, c = key3_from_row(row)
    return (a, b, c, int(row["code_index"]))


def choose_target_role(case: dict[str, Any]) -> tuple[str, int]:
    target = str(case["next_model_target"])
    if target == "recover_shortlist_oracle_change":
        return "shortlist", int(case["shortlist_code"])
    if target == "keep_current_good_change":
        return "current", int(case["current_code"])
    if target == "reject_current_bad_change":
        return "current_bad", int(case["current_code"])
    if target == "improve_first_stage_shortlist_recall":
        return "oracle_missing", int(case["oracle_code"])
    return "nearest", int(case["nearest_code"])


def collect_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows[:256]:
        for key, value in row.items():
            if key.startswith("img_") and finite_float(value) is not None:
                names.add(key)
    return sorted(names)


def build_rows(
    bank_cases: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    by_key = {key4_from_row(row): row for row in feature_rows}
    out: list[dict[str, Any]] = []
    for case in bank_cases:
        key3 = key3_from_case(case)
        role, code = choose_target_role(case)
        row = by_key.get((*key3, code))
        values = {name: finite_float(row.get(name)) if row is not None else None for name in feature_names}
        out.append(
            {
                "key": case["key"],
                "source": case.get("source"),
                "reference": case.get("reference"),
                "target": case["next_model_target"],
                "target_role": role,
                "target_code": code,
                "codes": {
                    "nearest": int(case["nearest_code"]),
                    "current": int(case["current_code"]),
                    "shortlist": int(case["shortlist_code"]),
                    "oracle": int(case["oracle_code"]),
                },
                "ocr_deltas": {
                    "current_vs_nearest": int(case["ocr"]["current_delta_vs_nearest"]),
                    "shortlist_vs_nearest": int(case["ocr"]["shortlist_delta_vs_nearest"]),
                    "shortlist_vs_current": int(case["ocr"]["shortlist_delta_vs_current"]),
                },
                "feature_row_present": row is not None,
                "features": values,
            }
        )
    return out


def target_values(rows: list[dict[str, Any]], target: str, feature: str) -> list[float]:
    vals = [row["features"].get(feature) for row in rows if row["target"] == target]
    return [float(v) for v in vals if v is not None]


def separation_records(rows: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    records = []
    reject_rows = [row for row in rows if row["target"] == "reject_current_bad_change"]
    if len(reject_rows) != 1:
        return records
    for feature in feature_names:
        reject = reject_rows[0]["features"].get(feature)
        if reject is None:
            continue
        recover = target_values(rows, "recover_shortlist_oracle_change", feature)
        good = target_values(rows, "keep_current_good_change", feature)
        first_stage = target_values(rows, "improve_first_stage_shortlist_recall", feature)
        if not recover:
            continue
        reject_f = float(reject)
        high_separates_recover = reject_f > max(recover)
        low_separates_recover = reject_f < min(recover)
        high_also_drops_good = int(sum(1 for value in good if value >= reject_f))
        low_also_drops_good = int(sum(1 for value in good if value <= reject_f))
        recover_mean = float(np.mean(recover))
        recover_std = float(np.std(recover)) if len(recover) > 1 else 0.0
        z_vs_recover = (reject_f - recover_mean) / (recover_std + 1e-12)
        records.append(
            {
                "feature": feature,
                "reject_value": reject_f,
                "recover_mean": recover_mean,
                "recover_min": float(min(recover)),
                "recover_max": float(max(recover)),
                "good_min": float(min(good)) if good else None,
                "good_max": float(max(good)) if good else None,
                "first_stage_mean": float(np.mean(first_stage)) if first_stage else None,
                "reject_above_all_recover": bool(high_separates_recover),
                "reject_below_all_recover": bool(low_separates_recover),
                "good_at_or_above_reject": high_also_drops_good,
                "good_at_or_below_reject": low_also_drops_good,
                "abs_z_vs_recover": float(abs(z_vs_recover)),
                "z_vs_recover": float(z_vs_recover),
            }
        )
    records.sort(
        key=lambda row: (
            not (row["reject_above_all_recover"] or row["reject_below_all_recover"]),
            min(row["good_at_or_above_reject"], row["good_at_or_below_reject"]),
            -row["abs_z_vs_recover"],
        )
    )
    return records


def summarize_all(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target in TARGET_ORDER:
        target_rows = [row for row in rows if row["target"] == target]
        if not target_rows:
            continue
        feature_stats = {feature: summarize(target_values(rows, target, feature)) for feature in feature_names}
        targets[target] = {
            "count": len(target_rows),
            "feature_row_present": int(sum(row["feature_row_present"] for row in target_rows)),
            "target_role_counts": dict(sorted(Counter(row["target_role"] for row in target_rows).items())),
            "feature_stats": feature_stats,
        }
    separations = separation_records(rows, feature_names)
    standalone = [
        row
        for row in separations
        if (row["reject_above_all_recover"] and row["good_at_or_above_reject"] == 0)
        or (row["reject_below_all_recover"] and row["good_at_or_below_reject"] == 0)
    ]
    return {
        "cases": len(rows),
        "by_target": dict(sorted(Counter(row["target"] for row in rows).items())),
        "feature_count": len(feature_names),
        "targets": targets,
        "separation_records": separations[:40],
        "standalone_separation_feature_count": len(standalone),
        "standalone_separation_features": [row["feature"] for row in standalone[:40]],
    }


def scalar_metrics(prefix: str, value: object, out: dict[str, Any]) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            scalar_metrics(f"{prefix}_{key}", child, out)


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Diagnostic-only audit of existing imgstroke/projection features on the N076 actual assignment failure bank.",
        "Held-out OCR categories define strata only; no selector is trained or promoted.",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['cases']}`",
        f"- Target counts: `{summary['by_target']}`",
        f"- Feature count: `{summary['feature_count']}`",
        f"- Standalone reject-separation features: `{summary['standalone_separation_feature_count']}`",
        "",
        "## Selected Feature Means",
        "",
        "| target | count | feature | mean | min | max |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for target in TARGET_ORDER:
        target_data = summary["targets"].get(target)
        if not target_data:
            continue
        for feature in REPORT_FEATURES:
            stats = target_data["feature_stats"].get(feature, {"mean": None, "min": None, "max": None})
            lines.append(
                f"| {target} | {target_data['count']} | `{feature}` | "
                f"{stats['mean'] if stats['mean'] is not None else 'null'} | "
                f"{stats['min'] if stats['min'] is not None else 'null'} | "
                f"{stats['max'] if stats['max'] is not None else 'null'} |"
            )
    lines.extend(
        [
            "",
            "## Reject Separation Probe",
            "",
            "| feature | reject | recover min/mean/max | good min/max | first-stage mean | direction | good collisions | abs z |",
            "|---|---:|---|---|---:|---|---:|---:|",
        ]
    )
    for row in summary["separation_records"][:24]:
        direction = "above" if row["reject_above_all_recover"] else "below" if row["reject_below_all_recover"] else "overlap"
        collisions = (
            row["good_at_or_above_reject"]
            if row["reject_above_all_recover"]
            else row["good_at_or_below_reject"]
            if row["reject_below_all_recover"]
            else min(row["good_at_or_above_reject"], row["good_at_or_below_reject"])
        )
        lines.append(
            f"| `{row['feature']}` | {row['reject_value']} | "
            f"{row['recover_min']}/{row['recover_mean']}/{row['recover_max']} | "
            f"{row['good_min']}/{row['good_max']} | {row['first_stage_mean']} | "
            f"{direction} | {collisions} | {row['abs_z_vs_recover']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This closes only the standalone imgstroke/projection-veto question.",
            "- Features that separate the single reject from recover rows must still be treated as weak evidence because there is only one reject sample and good changes may collide.",
            "- Use these features only as auxiliary inputs inside a stricter verifier with external calibration, not as a direct threshold rule.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--imgstroke-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_actual_assignment_imgstroke_audit")
    args = parser.parse_args()

    bank = read_json(args.bank)
    feature_rows = read_jsonl(args.imgstroke_table)
    feature_names = collect_feature_names(feature_rows)
    rows = build_rows(bank["cases"], feature_rows, feature_names)
    summary = summarize_all(rows, feature_names)
    result = {
        "experiment_id": args.experiment_id,
        "validity": "valid_diagnostic",
        "inputs": {
            "bank": str(args.bank),
            "imgstroke_table": str(args.imgstroke_table),
        },
        "feature_names": feature_names,
        "summary": summary,
        "rows": rows,
    }
    metrics: dict[str, Any] = {}
    scalar_metrics("summary", summary, metrics)
    result["aggregate"] = {"scalar_metrics": {key: {"value": value} for key, value in metrics.items()}}
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
