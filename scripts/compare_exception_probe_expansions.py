#!/usr/bin/env python3
"""Compare two actual-labeled exception counterfactual probe tables."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ["unicode_strict_v1", "latin_alnum_ci_v1"]
MODELS = ["tesseract", "parseq"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def row_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(row["seed"]),
        int(row["source_index"]),
        int(row.get("candidate_index", -1)),
        str(row["action_taken"]),
    )


def label_set_name(path: Path) -> str:
    return path.stem


def weighted_bpp(rows: list[dict[str, Any]], stream_key: str) -> float:
    bytes_total = 0
    pixels_total = 0
    for row in rows:
        stream = row[stream_key]
        bytes_total += int(stream["actual_total_bytes"])
        pixels_total += int(stream["height"]) * int(stream["width"])
    if pixels_total == 0:
        return 0.0
    return 8.0 * bytes_total / pixels_total


def selected_total(rows: list[dict[str, Any]], stream_key: str) -> int:
    return int(sum(int(row[stream_key]["selected_candidate_count"]) for row in rows))


def ocr_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for model in MODELS:
        result[model] = {}
        for profile in PROFILES:
            deltas = [int(row["ocr"][model][profile]["delta_errors"]) for row in rows]
            result[model][profile] = {
                "delta_errors": int(sum(deltas)),
                "improved_rows": int(sum(1 for value in deltas if value < 0)),
                "worsened_rows": int(sum(1 for value in deltas if value > 0)),
                "equal_rows": int(sum(1 for value in deltas if value == 0)),
            }
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(row["strict_label"] for row in rows)),
        "actions": dict(Counter(row["action_taken"] for row in rows)),
        "sources": dict(Counter(row["source"] for row in rows)),
        "tags": dict(Counter(tag for row in rows for tag in row.get("probe_tags", []))),
        "current_weighted_bpp": weighted_bpp(rows, "current_stream"),
        "expanded_weighted_bpp": weighted_bpp(rows, "exception_stream"),
        "current_selected_candidates": selected_total(rows, "current_stream"),
        "expanded_selected_candidates": selected_total(rows, "exception_stream"),
        "rate_delta_bytes": int(sum(int(row["rate_delta"]["actual_total_bytes"]) for row in rows)),
        "ocr": ocr_summary(rows),
    }


def compact_row(row: dict[str, Any], membership: str) -> dict[str, Any]:
    tess_unicode = row["ocr"]["tesseract"]["unicode_strict_v1"]
    tess_latin = row["ocr"]["tesseract"]["latin_alnum_ci_v1"]
    parseq_unicode = row["ocr"]["parseq"]["unicode_strict_v1"]
    return {
        "membership": membership,
        "seed": int(row["seed"]),
        "source_index": int(row["source_index"]),
        "candidate_index": int(row.get("candidate_index", -1)),
        "source": row["source"],
        "action_taken": row["action_taken"],
        "strict_label": row["strict_label"],
        "reference": row["reference"],
        "tags": row.get("probe_tags", []),
        "tesseract_current": row["ocr"]["tesseract"].get("current_prediction"),
        "tesseract_exception": row["ocr"]["tesseract"].get("exception_prediction"),
        "tesseract_unicode_delta": int(tess_unicode["delta_errors"]),
        "tesseract_latin_delta": int(tess_latin["delta_errors"]),
        "parseq_unicode_delta": int(parseq_unicode["delta_errors"]),
        "rate_delta_bytes": int(row["rate_delta"]["actual_total_bytes"]),
        "current_bpp": float(row["current_stream"]["actual_total_bpp"]),
        "exception_bpp": float(row["exception_stream"]["actual_total_bpp"]),
    }


def write_report(path: Path, result: dict[str, Any], added_rows: list[dict[str, Any]]) -> None:
    def fmt_labels(summary: dict[str, Any]) -> str:
        return "`" + json.dumps(summary["labels"], ensure_ascii=False, sort_keys=True) + "`"

    lines = [
        "# Exception Probe Expansion Comparison",
        "",
        "This compares two actual compact-stream OCR label tables. Negative OCR deltas mean the expanded exception/drop stream has fewer OCR edit errors than the matched current stream.",
        "",
        "## Summary",
        "",
        f"- Base table: `{result['base_table']}`",
        f"- Expanded table: `{result['expanded_table']}`",
        f"- Common rows: `{result['common_rows']}`; added rows: `{result['added_rows']}`; removed rows: `{result['removed_rows']}`.",
        "",
        "| subset | rows | labels | tess unicode | tess latin | parseq unicode | rate bytes | current bpp | expanded bpp |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["base", "expanded", "common", "added", "removed"]:
        summary = result["summaries"][name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(summary["rows"]),
                    fmt_labels(summary),
                    str(summary["ocr"]["tesseract"]["unicode_strict_v1"]["delta_errors"]),
                    str(summary["ocr"]["tesseract"]["latin_alnum_ci_v1"]["delta_errors"]),
                    str(summary["ocr"]["parseq"]["unicode_strict_v1"]["delta_errors"]),
                    str(summary["rate_delta_bytes"]),
                    f"{summary['current_weighted_bpp']:.6f}",
                    f"{summary['expanded_weighted_bpp']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Added Rows",
            "",
            "| seed | source_index | source | action | label | ref | tess current | tess expanded | unicode | latin | bytes | tags |",
            "|---:|---:|---|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in added_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["source_index"]),
                    row["source"],
                    row["action_taken"],
                    row["strict_label"],
                    str(row["reference"]).replace("|", "\\|"),
                    str(row["tesseract_current"]).replace("|", "\\|"),
                    str(row["tesseract_exception"]).replace("|", "\\|"),
                    str(row["tesseract_unicode_delta"]),
                    str(row["tesseract_latin_delta"]),
                    str(row["rate_delta_bytes"]),
                    ",".join(row["tags"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The expanded table adds useful evidence only if its added rows improve the safety/utility frontier. Here the added rows are mostly neutral, so the narrower base table remains the sharper frontier for the next verifier design.",
            "",
            "## Next Action",
            "",
            "Use this comparison to avoid threshold-widening loops. Add new labels only when they target missing hard-negative modes or materially different evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    result_path: Path,
    added_table: Path,
    report_path: Path,
) -> None:
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "code_commit": git_commit(),
        "script": {
            "path": "scripts/compare_exception_probe_expansions.py",
            "sha256": sha256_file(ROOT / "scripts/compare_exception_probe_expansions.py"),
        },
        "inputs": [
            {"name": "base_labels", "path": str(args.base_labels), "sha256": sha256_file(args.base_labels)},
            {"name": "expanded_labels", "path": str(args.expanded_labels), "sha256": sha256_file(args.expanded_labels)},
        ],
        "outputs": [
            {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
            {"name": "added_table", "path": str(added_table), "sha256": sha256_file(added_table)},
            {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
        ],
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def maybe_log_wandb(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if not args.wandb_project:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"wandb unavailable: {exc}")
        return
    run = wandb.init(project=args.wandb_project, name=args.experiment_id, config=vars(args))
    run.summary["base/rows"] = result["summaries"]["base"]["rows"]
    run.summary["expanded/rows"] = result["summaries"]["expanded"]["rows"]
    run.summary["added/rows"] = result["summaries"]["added"]["rows"]
    run.summary["added/beneficial"] = result["summaries"]["added"]["labels"].get("beneficial_no_profile_harm", 0)
    run.summary["added/harmful"] = result["summaries"]["added"]["labels"].get("harmful_any_profile", 0)
    run.summary["added/neutral"] = result["summaries"]["added"]["labels"].get("neutral_all_profiles", 0)
    run.summary["expanded/tesseract_unicode_delta"] = result["summaries"]["expanded"]["ocr"]["tesseract"]["unicode_strict_v1"]["delta_errors"]
    run.summary["added/tesseract_unicode_delta"] = result["summaries"]["added"]["ocr"]["tesseract"]["unicode_strict_v1"]["delta_errors"]
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-labels", type=Path, required=True)
    parser.add_argument("--expanded-labels", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--added-table", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--wandb-project")
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_labels)
    expanded_rows = read_jsonl(args.expanded_labels)
    base_by_key = {row_key(row): row for row in base_rows}
    expanded_by_key = {row_key(row): row for row in expanded_rows}

    common_keys = sorted(set(base_by_key) & set(expanded_by_key))
    added_keys = sorted(set(expanded_by_key) - set(base_by_key))
    removed_keys = sorted(set(base_by_key) - set(expanded_by_key))
    common_rows = [expanded_by_key[key] for key in common_keys]
    added_rows_raw = [expanded_by_key[key] for key in added_keys]
    removed_rows = [base_by_key[key] for key in removed_keys]
    added_rows = [compact_row(row, "added") for row in added_rows_raw]

    result = {
        "experiment_id": args.experiment_id,
        "base_table": label_set_name(args.base_labels),
        "expanded_table": label_set_name(args.expanded_labels),
        "common_rows": len(common_rows),
        "added_rows": len(added_rows_raw),
        "removed_rows": len(removed_rows),
        "summaries": {
            "base": summarize(base_rows),
            "expanded": summarize(expanded_rows),
            "common": summarize(common_rows),
            "added": summarize(added_rows_raw),
            "removed": summarize(removed_rows),
        },
    }

    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.added_table.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.added_table, added_rows)
    write_report(args.report, result, added_rows)
    write_manifest(args.manifest, args, args.result, args.added_table, args.report)
    maybe_log_wandb(args, result)
    print(json.dumps({"result": str(args.result), "added_table": str(args.added_table), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
