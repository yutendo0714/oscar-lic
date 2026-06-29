#!/usr/bin/env python3
"""Summarize an actual counted-OSCR shortlist-oracle evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


PROFILE_ORDER = ["unicode_strict_v1", "latin_alnum_ci_v1", "raw_exact_v1"]
POLICY_KEYS = [
    "groups",
    "changed_groups",
    "exact_oracle_changes",
    "missed_oracle_changes",
    "false_changes",
    "wrong_changes",
    "tesseract_delta_sum_vs_nearest",
    "parseq_delta_sum_vs_nearest",
    "tesseract_improve_groups",
    "tesseract_worsen_groups",
]
STREAM_KEYS = [
    "avg_actual_total_bpp",
    "avg_enhancement_payload_bpp",
    "avg_assignment_changed_fraction",
    "avg_assignment_changed_count",
    "avg_psnr_delta_db",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    if not label:
        raise SystemExit(f"empty label in {value!r}")
    return label, Path(path)


def table_row(values: list[Any]) -> str:
    return "| " + " | ".join(format_cell(value) for value in values) + " |"


def format_cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summarize_streams(stream_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted(stream_summaries)
    aggregate: dict[str, Any] = {"seeds": len(seeds)}
    for key in STREAM_KEYS:
        values = [float(stream_summaries[seed][key]) for seed in seeds if key in stream_summaries[seed]]
        if values:
            aggregate[key] = {
                "mean": float(statistics.fmean(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }
    return aggregate


def compare_policy(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    current_agg = current.get("aggregate", {})
    baseline_agg = baseline.get("aggregate", {})
    return {
        key: current_agg.get(key) - baseline_agg.get(key)
        for key in POLICY_KEYS
        if isinstance(current_agg.get(key), (int, float)) and isinstance(baseline_agg.get(key), (int, float))
    }


def extract_compare_aggregate(compare: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    profiles = compare.get("aggregate", {}).get("profiles", {})
    for profile in PROFILE_ORDER:
        if profile in profiles:
            profile_data = profiles[profile]
            output[profile] = {
                "delta_char_errors": profile_data.get("delta_char_errors"),
                "improved_samples": profile_data.get("improved_samples"),
                "worsened_samples": profile_data.get("worsened_samples"),
                "delta_exact_matches": profile_data.get("delta_exact_matches"),
                "baseline_cer_micro": profile_data.get("baseline_cer_micro"),
                "candidate_cer_micro": profile_data.get("candidate_cer_micro"),
                "delta_cer_micro": profile_data.get("delta_cer_micro"),
            }
    return output


def write_report(
    path: Path,
    result: dict[str, Any],
    stream_summaries: dict[str, dict[str, Any]],
    compare_summaries: dict[str, dict[str, dict[str, Any]]],
) -> None:
    policy = result["policy"]
    baseline_delta = result.get("baseline_policy_delta")
    stream_agg = result["stream_aggregate"]
    hashes = result["hashes"]

    lines: list[str] = []
    lines.append(f"# {result['title']}")
    lines.append("")
    lines.append(f"Date: {result['date']}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(result["description"])
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append(table_row(["Split", "Groups", "Changed", "Exact oracle", "Missed oracle", "Tess table delta", "PARSeq table delta"]))
    lines.append(table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for seed in sorted(policy["by_real_seed"], key=lambda value: int(value)):
        row = policy["by_real_seed"][seed]
        lines.append(
            table_row(
                [
                    f"seed{seed}",
                    row.get("groups"),
                    row.get("changed_groups"),
                    row.get("exact_oracle_changes"),
                    row.get("missed_oracle_changes"),
                    row.get("tesseract_delta_sum_vs_nearest"),
                    row.get("parseq_delta_sum_vs_nearest"),
                ]
            )
        )
    agg = policy["aggregate"]
    lines.append(
        table_row(
            [
                "total",
                agg.get("groups"),
                agg.get("changed_groups"),
                agg.get("exact_oracle_changes"),
                agg.get("missed_oracle_changes"),
                agg.get("tesseract_delta_sum_vs_nearest"),
                agg.get("parseq_delta_sum_vs_nearest"),
            ]
        )
    )
    if baseline_delta is not None:
        lines.append("")
        lines.append(
            "Against the baseline shortlist policy, the current policy delta is: "
            + ", ".join(f"{key} {value:+}" for key, value in baseline_delta.items())
            + "."
        )
    lines.append("")
    lines.append("## Actual Bitstream")
    lines.append("")
    lines.append(table_row(["Split", "Actual bpp", "Enhancement bpp", "Assignment changed fraction", "PSNR delta dB"]))
    lines.append(table_row(["---", "---:", "---:", "---:", "---:"]))
    for seed in sorted(stream_summaries):
        stream = stream_summaries[seed]
        lines.append(
            table_row(
                [
                    seed,
                    stream.get("avg_actual_total_bpp"),
                    stream.get("avg_enhancement_payload_bpp"),
                    stream.get("avg_assignment_changed_fraction"),
                    stream.get("avg_psnr_delta_db"),
                ]
            )
        )
    lines.append(
        table_row(
            [
                "mean",
                stream_agg["avg_actual_total_bpp"]["mean"],
                stream_agg["avg_enhancement_payload_bpp"]["mean"],
                stream_agg["avg_assignment_changed_fraction"]["mean"],
                stream_agg["avg_psnr_delta_db"]["mean"],
            ]
        )
    )
    lines.append("")
    lines.append("Actual bpp is counted from serialized `.oscr` files, including base, enhancement, gate, code, metadata, and container bytes.")
    lines.append("")
    lines.append("## OCR Results")
    lines.append("")
    lines.append(table_row(["Comparison", "Profile", "Delta chars", "Improved", "Worsened", "Exact delta", "CER delta"]))
    lines.append(table_row(["---", "---", "---:", "---:", "---:", "---:", "---:"]))
    for label in sorted(compare_summaries):
        for profile in ["unicode_strict_v1", "latin_alnum_ci_v1"]:
            profile_data = compare_summaries[label].get(profile)
            if not profile_data:
                continue
            lines.append(
                table_row(
                    [
                        label,
                        profile,
                        profile_data.get("delta_char_errors"),
                        profile_data.get("improved_samples"),
                        profile_data.get("worsened_samples"),
                        profile_data.get("delta_exact_matches"),
                        profile_data.get("delta_cer_micro"),
                    ]
                )
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(result["interpretation"])
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Summary JSON: `{result['output_json']}`")
    for label, path_string in result["input_paths"].items():
        lines.append(f"- {label}: `{path_string}`")
    lines.append("")
    lines.append("## Hashes")
    lines.append("")
    for label, digest in sorted(hashes.items()):
        lines.append(f"- {label}: `{digest}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--interpretation", required=True)
    parser.add_argument("--policy-summary", type=Path, required=True)
    parser.add_argument("--baseline-policy-summary", type=Path)
    parser.add_argument("--stream-summary", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--ocr-summary", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--compare", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--extra-input", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    policy = load_json(args.policy_summary)
    baseline_policy = load_json(args.baseline_policy_summary) if args.baseline_policy_summary else None

    stream_paths = dict(parse_labeled_path(item) for item in args.stream_summary)
    stream_summaries = {label: load_json(path) for label, path in stream_paths.items()}
    ocr_paths = dict(parse_labeled_path(item) for item in args.ocr_summary)
    ocr_summaries = {label: load_json(path) for label, path in ocr_paths.items()}
    compare_paths = dict(parse_labeled_path(item) for item in args.compare)
    compare_raw = {label: load_json(path) for label, path in compare_paths.items()}
    compare_summaries = {label: extract_compare_aggregate(data) for label, data in compare_raw.items()}
    extra_paths = dict(parse_labeled_path(item) for item in args.extra_input)

    input_paths: dict[str, str] = {
        "policy_summary": str(args.policy_summary),
        **{f"stream_summary_{label}": str(path) for label, path in stream_paths.items()},
        **{f"ocr_summary_{label}": str(path) for label, path in ocr_paths.items()},
        **{f"compare_{label}": str(path) for label, path in compare_paths.items()},
        **{label: str(path) for label, path in extra_paths.items()},
    }
    if args.baseline_policy_summary:
        input_paths["baseline_policy_summary"] = str(args.baseline_policy_summary)

    hash_paths: dict[str, Path] = {
        "script": Path(__file__),
        "policy_summary": args.policy_summary,
        **{f"stream_summary_{label}": path for label, path in stream_paths.items()},
        **{f"ocr_summary_{label}": path for label, path in ocr_paths.items()},
        **{f"compare_{label}": path for label, path in compare_paths.items()},
        **{label: path for label, path in extra_paths.items()},
    }
    if args.baseline_policy_summary:
        hash_paths["baseline_policy_summary"] = args.baseline_policy_summary

    result = {
        "title": args.title,
        "date": args.date,
        "description": args.description,
        "interpretation": args.interpretation,
        "validity": "oracle_upper_bound_not_deployable",
        "track": "A_pure_visual_bitstream_policy_diagnostic",
        "policy": policy,
        "baseline_policy_delta": compare_policy(policy, baseline_policy),
        "stream_summaries": stream_summaries,
        "stream_aggregate": summarize_streams(stream_summaries),
        "ocr_summaries": ocr_summaries,
        "ocr_comparisons": compare_summaries,
        "input_paths": input_paths,
        "output_json": str(args.output),
        "report": str(args.report),
        "hashes": {label: sha256(path) for label, path in hash_paths.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result, stream_summaries, compare_summaries)

    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
