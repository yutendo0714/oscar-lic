#!/usr/bin/env python3
"""Join compact actual-byte rates with existing OCR bootstrap comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ORDER = ["unicode_strict_v1", "latin_alnum_ci_v1", "raw_exact_v1"]
POLICIES = ["nearest", "current", "top4", "top8"]
BOOTSTRAP_SUFFIXES = [
    "_nearest_current",
    "_nearest_shortlist",
    "_current_shortlist",
    "_nearest_top8",
    "_current_top8",
    "_top4_top8",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def discover_ocrs(*bootstraps: dict[str, Any]) -> list[str]:
    ocrs = set()
    for bootstrap in bootstraps:
        for key in bootstrap.get("comparisons", {}):
            for suffix in BOOTSTRAP_SUFFIXES:
                if key.endswith(suffix):
                    ocrs.add(key[: -len(suffix)])
    return sorted(ocrs)


def get_profile(comparison: dict[str, Any], profile: str) -> dict[str, Any] | None:
    return comparison.get("profiles", {}).get(profile)


def policy_row_from_nearest(
    *,
    policy: str,
    comparison: dict[str, Any] | None,
    profile: str,
) -> dict[str, Any]:
    if policy == "nearest":
        if comparison is None:
            return {}
        profile_row = get_profile(comparison, profile)
        if not profile_row:
            return {}
        observed = profile_row["observed"]
        return {
            "char_errors": observed.get("baseline_char_errors"),
            "cer_micro": observed.get("baseline_cer_micro"),
            "exact_rate": observed.get("baseline_exact_rate"),
            "delta_char_errors_vs_nearest": 0.0,
            "delta_cer_micro_vs_nearest": 0.0,
            "delta_exact_rate_vs_nearest": 0.0,
            "delta_char_errors_ci95": [0.0, 0.0],
            "delta_cer_micro_ci95": [0.0, 0.0],
        }
    if comparison is None:
        return {}
    profile_row = get_profile(comparison, profile)
    if not profile_row:
        return {}
    observed = profile_row["observed"]
    ci95 = profile_row.get("ci95", {})
    return {
        "char_errors": observed.get("candidate_char_errors"),
        "cer_micro": observed.get("candidate_cer_micro"),
        "exact_rate": observed.get("candidate_exact_rate"),
        "delta_char_errors_vs_nearest": observed.get("delta_char_errors"),
        "delta_cer_micro_vs_nearest": observed.get("delta_cer_micro"),
        "delta_exact_rate_vs_nearest": observed.get("delta_exact_rate"),
        "delta_char_errors_ci95": ci95.get("delta_char_errors"),
        "delta_cer_micro_ci95": ci95.get("delta_cer_micro"),
        "improved_rate_vs_nearest": observed.get("improved_rate"),
        "worsened_rate_vs_nearest": observed.get("worsened_rate"),
    }


def policy_row_from_pairwise_baseline(
    *,
    comparison: dict[str, Any] | None,
    nearest_row: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    if comparison is None:
        return {}
    profile_row = get_profile(comparison, profile)
    if not profile_row:
        return {}
    observed = profile_row["observed"]
    char_errors = observed.get("baseline_char_errors")
    cer = observed.get("baseline_cer_micro")
    exact = observed.get("baseline_exact_rate")
    nearest_errors = nearest_row.get("char_errors")
    nearest_cer = nearest_row.get("cer_micro")
    nearest_exact = nearest_row.get("exact_rate")
    out = {
        "char_errors": char_errors,
        "cer_micro": cer,
        "exact_rate": exact,
        "delta_char_errors_ci95": None,
        "delta_cer_micro_ci95": None,
    }
    if isinstance(char_errors, (int, float)) and isinstance(nearest_errors, (int, float)):
        out["delta_char_errors_vs_nearest"] = float(char_errors) - float(nearest_errors)
    if isinstance(cer, (int, float)) and isinstance(nearest_cer, (int, float)):
        out["delta_cer_micro_vs_nearest"] = float(cer) - float(nearest_cer)
    if isinstance(exact, (int, float)) and isinstance(nearest_exact, (int, float)):
        out["delta_exact_rate_vs_nearest"] = float(exact) - float(nearest_exact)
    return out


def pairwise_row(comparison: dict[str, Any] | None, profile: str) -> dict[str, Any]:
    if comparison is None:
        return {}
    profile_row = get_profile(comparison, profile)
    if not profile_row:
        return {}
    observed = profile_row["observed"]
    ci95 = profile_row.get("ci95", {})
    return {
        "delta_char_errors": observed.get("delta_char_errors"),
        "delta_char_errors_ci95": ci95.get("delta_char_errors"),
        "delta_cer_micro": observed.get("delta_cer_micro"),
        "delta_cer_micro_ci95": ci95.get("delta_cer_micro"),
        "improved_rate": observed.get("improved_rate"),
        "worsened_rate": observed.get("worsened_rate"),
    }


def compact_rates(accounting: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    sets = accounting["sets"]
    for policy in POLICIES:
        key = f"{policy}_compact"
        if key not in sets:
            continue
        metrics = sets[key]["metrics"]
        selected = sets[key]["selected_only"]
        out[policy] = {
            "streams": metrics["actual_total_bpp"]["count"],
            "selected_streams": sets[key]["selected_streams"],
            "mean_total_bpp": metrics["actual_total_bpp"]["mean"],
            "mean_enhancement_payload_bpp": metrics["enhancement_payload_bpp"]["mean"],
            "mean_optional_total_bpp": metrics["optional_total_bpp"]["mean"],
            "selected_optional_bytes": selected["optional_total_bytes"]["mean"],
            "actual_total_bytes_sum": metrics["actual_total_bytes"]["sum"],
        }
    return out


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    top4 = load_json(args.bootstrap_top4)
    top8 = load_json(args.bootstrap_top8)
    accounting = load_json(args.accounting)
    rates = compact_rates(accounting)
    ocrs = discover_ocrs(top4, top8)

    comparisons4 = top4["comparisons"]
    comparisons8 = top8["comparisons"]
    ocr_policy_table: dict[str, dict[str, Any]] = {}
    pairwise: dict[str, dict[str, Any]] = {}
    for ocr in ocrs:
        nearest_current = comparisons4.get(f"{ocr}_nearest_current") or comparisons8.get(f"{ocr}_nearest_top8")
        nearest_top4 = comparisons4.get(f"{ocr}_nearest_shortlist")
        nearest_top8 = comparisons8.get(f"{ocr}_nearest_top8")
        current_top8 = comparisons8.get(f"{ocr}_current_top8")
        top4_top8 = comparisons8.get(f"{ocr}_top4_top8")
        ocr_policy_table[ocr] = {}
        pairwise[ocr] = {}
        for profile in PROFILE_ORDER:
            nearest_row = policy_row_from_nearest(policy="nearest", comparison=nearest_current, profile=profile)
            current_row = policy_row_from_nearest(
                policy="current", comparison=comparisons4.get(f"{ocr}_nearest_current"), profile=profile
            )
            if not current_row:
                current_row = policy_row_from_pairwise_baseline(
                    comparison=current_top8, nearest_row=nearest_row, profile=profile
                )
            top4_row = policy_row_from_nearest(policy="top4", comparison=nearest_top4, profile=profile)
            if not top4_row:
                top4_row = policy_row_from_pairwise_baseline(
                    comparison=top4_top8, nearest_row=nearest_row, profile=profile
                )
            ocr_policy_table[ocr][profile] = {
                "nearest": nearest_row,
                "current": current_row,
                "top4": top4_row,
                "top8": policy_row_from_nearest(policy="top8", comparison=nearest_top8, profile=profile),
            }
            pairwise[ocr][profile] = {
                "current_vs_top4": pairwise_row(comparisons4.get(f"{ocr}_current_shortlist"), profile),
                "current_vs_top8": pairwise_row(comparisons8.get(f"{ocr}_current_top8"), profile),
                "top4_vs_top8": pairwise_row(comparisons8.get(f"{ocr}_top4_top8"), profile),
            }

    return {
        "experiment_id": args.experiment_id,
        "validity": "summary_existing_ocr_and_actual_compact_rate",
        "code_commit": git_commit(),
        "accounting": rel(args.accounting),
        "bootstrap_top4": rel(args.bootstrap_top4),
        "bootstrap_top8": rel(args.bootstrap_top8),
        "compact_rates": rates,
        "ocr_policy_table": ocr_policy_table,
        "pairwise_comparisons": pairwise,
        "wandb_run_id": "not_logged",
        "interpretation": (
            "All four policies use matched compact TEXT_SYMBOLS actual bpp; OCR differences in this table come from "
            "selector/assignment choices, not from extra transmitted enhancement budget."
        ),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(item) for item in value) + "]"
    return str(value)


def row(values: list[Any]) -> str:
    return "| " + " | ".join(fmt(value) for value in values) + " |"


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "Compact actual-byte OCR summary built from existing OCR bootstrap JSONs and N134 `.oscr` section accounting.",
        "",
        "## Compact Rates",
        "",
        row(["policy", "streams", "selected", "mean total bpp", "enh payload bpp", "optional bpp", "selected optional bytes"]),
        row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for policy in POLICIES:
        rate = result["compact_rates"].get(policy, {})
        lines.append(
            row(
                [
                    policy,
                    rate.get("streams"),
                    rate.get("selected_streams"),
                    rate.get("mean_total_bpp"),
                    rate.get("mean_enhancement_payload_bpp"),
                    rate.get("mean_optional_total_bpp"),
                    rate.get("selected_optional_bytes"),
                ]
            )
        )
    lines.extend(["", "## Unicode OCR vs Nearest", ""])
    lines.append(row(["OCR", "policy", "CER", "char errors", "delta chars", "delta chars CI95", "exact rate"]))
    lines.append(row(["---", "---", "---:", "---:", "---:", "---:", "---:"]))
    for ocr in sorted(result["ocr_policy_table"]):
        table = result["ocr_policy_table"][ocr]["unicode_strict_v1"]
        for policy in POLICIES:
            item = table.get(policy, {})
            lines.append(
                row(
                    [
                        ocr,
                        policy,
                        item.get("cer_micro"),
                        item.get("char_errors"),
                        item.get("delta_char_errors_vs_nearest"),
                        item.get("delta_char_errors_ci95"),
                        item.get("exact_rate"),
                    ]
                )
            )
    lines.extend(["", "## Key Pairwise Unicode Deltas", ""])
    lines.append(row(["OCR", "comparison", "delta chars", "delta chars CI95", "delta CER"]))
    lines.append(row(["---", "---", "---:", "---:", "---:"]))
    for ocr in sorted(result["pairwise_comparisons"]):
        pairs = result["pairwise_comparisons"][ocr]["unicode_strict_v1"]
        for label in ["current_vs_top4", "current_vs_top8", "top4_vs_top8"]:
            item = pairs.get(label, {})
            lines.append(
                row(
                    [
                        ocr,
                        label,
                        item.get("delta_char_errors"),
                        item.get("delta_char_errors_ci95"),
                        item.get("delta_cer_micro"),
                    ]
                )
            )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path, args: argparse.Namespace) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-compact-rate-ocr-summary",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        "inputs:",
        "  accounting:",
        f"    path: {rel(args.accounting)}",
        f"    sha256: {sha256_file(args.accounting)}",
        "  bootstrap_top4:",
        f"    path: {rel(args.bootstrap_top4)}",
        f"    sha256: {sha256_file(args.bootstrap_top4)}",
        "  bootstrap_top8:",
        f"    path: {rel(args.bootstrap_top8)}",
        f"    sha256: {sha256_file(args.bootstrap_top8)}",
        "outputs:",
        "  result_json:",
        f"    path: {rel(output)}",
        f"    sha256: {sha256_file(output)}",
        "  report:",
        f"    path: {rel(report)}",
        f"    sha256: {sha256_file(report)}",
        "scripts:",
        "  summarizer:",
        "    path: scripts/summarize_compact_rate_ocr_table.py",
        f"    sha256: {sha256_file(Path(__file__))}",
        "conclusion: compact actual-byte rate table joined with existing OCR bootstrap evidence",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--bootstrap-top4", type=Path, required=True)
    parser.add_argument("--bootstrap-top8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_result(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report, args)
    print(json.dumps({"experiment_id": args.experiment_id, "output": rel(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
