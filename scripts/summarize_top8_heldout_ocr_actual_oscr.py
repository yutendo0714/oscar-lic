#!/usr/bin/env python3
"""Summarize held-out OCR checks for an actual-OSCR top8 oracle run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROFILES = ["unicode_strict_v1", "latin_alnum_ci_v1", "raw_exact_v1"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"expected LABEL=PATH, got {raw!r}")
    label, value = raw.split("=", 1)
    if not label:
        raise SystemExit(f"empty label in {raw!r}")
    return label, Path(value)


def compare_summary(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = data.get("aggregate", {}).get("profiles", {})
    out: dict[str, dict[str, Any]] = {}
    for profile in PROFILES:
        if profile not in profiles:
            continue
        row = profiles[profile]
        out[profile] = {
            "delta_char_errors": row.get("delta_char_errors"),
            "improved_samples": row.get("improved_samples"),
            "worsened_samples": row.get("worsened_samples"),
            "unchanged_samples": row.get("unchanged_samples"),
            "delta_exact_matches": row.get("delta_exact_matches"),
            "baseline_cer_micro": row.get("baseline_cer_micro"),
            "candidate_cer_micro": row.get("candidate_cer_micro"),
            "delta_cer_micro": row.get("delta_cer_micro"),
        }
    return out


def summary_metric(data: dict[str, Any], profile: str) -> dict[str, Any]:
    metrics = data.get("metrics", {}).get(profile, {})
    return {
        "cer_micro": metrics.get("cer_micro"),
        "exact_rate": metrics.get("exact_rate"),
        "samples": metrics.get("samples"),
    }


def table_row(cells: list[Any]) -> str:
    return "| " + " | ".join(format_cell(cell) for cell in cells) + " |"


def format_cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# {result['title']}")
    lines.append("")
    lines.append(f"Date: {result['date']}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(result["description"])
    lines.append("")
    lines.append("## Row-Wise Comparisons")
    lines.append("")
    lines.append(table_row(["Comparison", "Profile", "Delta chars", "Improved", "Worsened", "Exact delta", "CER delta"]))
    lines.append(table_row(["---", "---", "---:", "---:", "---:", "---:", "---:"]))
    for label in sorted(result["comparisons"]):
        for profile in ["unicode_strict_v1", "latin_alnum_ci_v1"]:
            row = result["comparisons"][label].get(profile)
            if row:
                lines.append(
                    table_row(
                        [
                            label,
                            profile,
                            row.get("delta_char_errors"),
                            row.get("improved_samples"),
                            row.get("worsened_samples"),
                            row.get("delta_exact_matches"),
                            row.get("delta_cer_micro"),
                        ]
                    )
                )
    lines.append("")
    lines.append("## Top-8 Metrics")
    lines.append("")
    lines.append(table_row(["OCR", "Seed", "Unicode CER", "Unicode exact", "Latin CER", "Latin exact", "W&B"]))
    lines.append(table_row(["---", "---", "---:", "---:", "---:", "---:", "---"]))
    for label in sorted(result["summaries"]):
        row = result["summaries"][label]
        lines.append(
            table_row(
                [
                    row["ocr"],
                    row["seed"],
                    row["metrics"].get("unicode_strict_v1", {}).get("cer_micro"),
                    row["metrics"].get("unicode_strict_v1", {}).get("exact_rate"),
                    row["metrics"].get("latin_alnum_ci_v1", {}).get("cer_micro"),
                    row["metrics"].get("latin_alnum_ci_v1", {}).get("exact_rate"),
                    row.get("wandb_run_id"),
                ]
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(result["interpretation"])
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Summary JSON: `{result['output_json']}`")
    for label, path_string in result["input_paths"].items():
        lines.append(f"- {label}: `{path_string}`")
    lines.append("")
    lines.append("## Hashes")
    lines.append("")
    for label, digest in sorted(result["hashes"].items()):
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
    parser.add_argument("--summary", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--compare", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--extra-input", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    summary_paths = dict(parse_labeled_path(item) for item in args.summary)
    compare_paths = dict(parse_labeled_path(item) for item in args.compare)
    extra_paths = dict(parse_labeled_path(item) for item in args.extra_input)

    summaries: dict[str, dict[str, Any]] = {}
    for label, path in summary_paths.items():
        data = load_json(path)
        ocr = str(data.get("model", label.split("_")[0]))
        seed = label.rsplit("_", 1)[-1]
        summaries[label] = {
            "path": str(path),
            "ocr": ocr,
            "seed": seed,
            "samples": data.get("samples"),
            "wandb_run_id": data.get("wandb_run_id"),
            "metrics": {profile: summary_metric(data, profile) for profile in PROFILES},
        }

    comparisons = {label: compare_summary(load_json(path)) for label, path in compare_paths.items()}
    input_paths = {
        **{f"summary_{label}": str(path) for label, path in summary_paths.items()},
        **{f"compare_{label}": str(path) for label, path in compare_paths.items()},
        **{label: str(path) for label, path in extra_paths.items()},
    }
    hash_paths = {
        "script": Path(__file__),
        **{f"summary_{label}": path for label, path in summary_paths.items()},
        **{f"compare_{label}": path for label, path in compare_paths.items()},
        **extra_paths,
    }
    result = {
        "title": args.title,
        "date": args.date,
        "description": args.description,
        "interpretation": args.interpretation,
        "validity": "held_out_ocr_evaluation_only",
        "track": "A_pure_visual_bitstream_evaluation_diagnostic",
        "summaries": summaries,
        "comparisons": comparisons,
        "input_paths": input_paths,
        "output_json": str(args.output),
        "report": str(args.report),
        "hashes": {label: sha256(path) for label, path in hash_paths.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
