#!/usr/bin/env python3
"""Audit harmful/beneficial evidence in actual exception probe labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]


NUMERIC_FEATURES = [
    "reference_length",
    "width",
    "height",
    "aspect_ratio",
    "current_luma_mean",
    "current_luma_std",
    "current_dark_fraction",
    "current_bright_fraction",
    "exception_luma_mean",
    "exception_luma_std",
    "exception_dark_fraction",
    "exception_bright_fraction",
    "exception_minus_current_dark_fraction",
    "exception_minus_current_bright_fraction",
    "current_exception_mad",
    "current_exception_mse",
    "current_exception_max_abs",
    "rate_delta_bytes",
    "rate_delta_bpp",
    "current_actual_bpp",
    "exception_actual_bpp",
    "current_psnr_db",
    "exception_psnr_db",
    "psnr_delta_db",
]


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


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def image_luma(path: str) -> np.ndarray:
    image = Image.open(resolve_path(path)).convert("L")
    return np.asarray(image, dtype=np.float32)


def luma_stats(data: np.ndarray) -> dict[str, float]:
    return {
        "luma_mean": float(data.mean()),
        "luma_std": float(data.std()),
        "dark_fraction": float((data < 80.0).mean()),
        "bright_fraction": float((data > 200.0).mean()),
    }


def quantiles(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not clean:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    data = sorted(clean)

    def pick(frac: float) -> float:
        index = round(frac * (len(data) - 1))
        return float(data[index])

    return {
        "min": float(data[0]),
        "q25": pick(0.25),
        "median": pick(0.5),
        "q75": pick(0.75),
        "max": float(data[-1]),
        "mean": float(sum(data) / len(data)),
    }


def summarize_ocr(rows: list[dict[str, Any]], model: str, profile: str) -> dict[str, int]:
    deltas = [int(row[f"{model}_{profile}_delta"]) for row in rows]
    return {
        "delta_errors": int(sum(deltas)),
        "improved_rows": int(sum(1 for value in deltas if value < 0)),
        "worsened_rows": int(sum(1 for value in deltas if value > 0)),
        "equal_rows": int(sum(1 for value in deltas if value == 0)),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": len(rows),
        "labels": dict(Counter(row["strict_label"] for row in rows)),
        "actions": dict(Counter(row["action_taken"] for row in rows)),
        "sources": dict(Counter(row["source"] for row in rows)),
        "tags": dict(Counter(tag for row in rows for tag in row["probe_tags"])),
        "ocr": {
            model: {
                profile: summarize_ocr(rows, model, profile)
                for profile in ["unicode_strict_v1", "latin_alnum_ci_v1"]
            }
            for model in ["tesseract", "parseq"]
        },
        "features": {},
    }
    for feature in NUMERIC_FEATURES:
        result["features"][feature] = quantiles([row.get(feature) for row in rows])
    return result


def build_feature_row(label_set: str, row: dict[str, Any]) -> dict[str, Any]:
    current = row["current_stream"]
    exception = row["exception_stream"]
    cur = image_luma(current["reconstruction_path"])
    exc = image_luma(exception["reconstruction_path"])
    if cur.shape != exc.shape:
        raise ValueError(f"shape mismatch for {current['reconstruction_path']} and {exception['reconstruction_path']}")
    cur_stats = luma_stats(cur)
    exc_stats = luma_stats(exc)
    diff = exc - cur
    out: dict[str, Any] = {
        "label_set": label_set,
        "probe_index": row.get("probe_index"),
        "seed": int(row["seed"]),
        "source_index": int(row["source_index"]),
        "candidate_index": int(row["candidate_index"]),
        "source": str(row["source"]),
        "reference": str(row["reference"]),
        "reference_length": len(str(row["reference"])),
        "probe_tags": list(row.get("probe_tags", [])),
        "policy_case": row.get("policy_case"),
        "action_taken": row["action_taken"],
        "strict_label": row["strict_label"],
        "width": int(cur.shape[1]),
        "height": int(cur.shape[0]),
        "aspect_ratio": float(cur.shape[1] / max(cur.shape[0], 1)),
        "current_actual_bpp": float(current["actual_total_bpp"]),
        "exception_actual_bpp": float(exception["actual_total_bpp"]),
        "current_psnr_db": float(current.get("psnr_db", 0.0)),
        "exception_psnr_db": float(exception.get("psnr_db", 0.0)),
        "psnr_delta_db": float(exception.get("psnr_db", 0.0)) - float(current.get("psnr_db", 0.0)),
        "rate_delta_bytes": int(row["rate_delta"]["actual_total_bytes"]),
        "rate_delta_bpp": float(row["rate_delta"]["actual_total_bpp"]),
        "current_exception_mad": float(np.abs(diff).mean()),
        "current_exception_mse": float(np.square(diff).mean()),
        "current_exception_max_abs": float(np.abs(diff).max()),
    }
    for prefix, stats in [("current", cur_stats), ("exception", exc_stats)]:
        out[f"{prefix}_luma_mean"] = stats["luma_mean"]
        out[f"{prefix}_luma_std"] = stats["luma_std"]
        out[f"{prefix}_dark_fraction"] = stats["dark_fraction"]
        out[f"{prefix}_bright_fraction"] = stats["bright_fraction"]
    out["exception_minus_current_dark_fraction"] = out["exception_dark_fraction"] - out["current_dark_fraction"]
    out["exception_minus_current_bright_fraction"] = out["exception_bright_fraction"] - out["current_bright_fraction"]
    for model in ["tesseract", "parseq"]:
        out[f"{model}_current_prediction"] = row["ocr"][model]["current_prediction"]
        out[f"{model}_exception_prediction"] = row["ocr"][model]["exception_prediction"]
        for profile in ["unicode_strict_v1", "latin_alnum_ci_v1"]:
            out[f"{model}_{profile}_delta"] = int(row["ocr"][model][profile]["delta_errors"])
            out[f"{model}_{profile}_current_distance"] = int(row["ocr"][model][profile]["current_distance"])
            out[f"{model}_{profile}_exception_distance"] = int(row["ocr"][model][profile]["exception_distance"])
    return out


def top_harmful(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    harmful = [row for row in rows if row["strict_label"] in {"harmful_any_profile", "mixed_harm_and_benefit"}]

    def harm_score(row: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(row["tesseract_unicode_strict_v1_delta"]) + int(row["tesseract_latin_alnum_ci_v1_delta"]),
            int(row["parseq_unicode_strict_v1_delta"]) + int(row["parseq_latin_alnum_ci_v1_delta"]),
            int(row["reference_length"]),
        )

    selected = sorted(harmful, key=harm_score, reverse=True)
    slim = []
    for row in selected:
        slim.append(
            {
                "label_set": row["label_set"],
                "probe_index": row["probe_index"],
                "seed": row["seed"],
                "source_index": row["source_index"],
                "source": row["source"],
                "reference": row["reference"],
                "action_taken": row["action_taken"],
                "probe_tags": row["probe_tags"],
                "tesseract_current": row["tesseract_current_prediction"],
                "tesseract_exception": row["tesseract_exception_prediction"],
                "tesseract_unicode_delta": row["tesseract_unicode_strict_v1_delta"],
                "tesseract_latin_delta": row["tesseract_latin_alnum_ci_v1_delta"],
                "rate_delta_bytes": row["rate_delta_bytes"],
                "current_exception_mad": row["current_exception_mad"],
                "current_dark_fraction": row["current_dark_fraction"],
                "exception_dark_fraction": row["exception_dark_fraction"],
            }
        )
    return slim


def build_result(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for table in args.label_table:
        label_set = table.stem
        for row in read_jsonl(table):
            rows.append(build_feature_row(label_set, row))
    unique_by_key: dict[tuple[int, int, int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["source_index"]),
            int(row["candidate_index"]),
            str(row["reference"]),
            str(row["action_taken"]),
        )
        unique_by_key[key] = row
    unique_rows = list(unique_by_key.values())
    by_label = defaultdict(list)
    by_action = defaultdict(list)
    by_source = defaultdict(list)
    by_label_set = defaultdict(list)
    for row in unique_rows:
        by_label[row["strict_label"]].append(row)
        by_action[row["action_taken"]].append(row)
        by_source[row["source"]].append(row)
    for row in rows:
        by_label_set[row["label_set"]].append(row)
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-harmful-evidence-audit",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "inputs": [str(path) for path in args.label_table],
        "summary": {
            "all_unique": summarize_group(unique_rows),
            "all_with_label_set_duplicates": summarize_group(rows),
            "by_label": {name: summarize_group(group) for name, group in sorted(by_label.items())},
            "by_action": {name: summarize_group(group) for name, group in sorted(by_action.items())},
            "by_source": {name: summarize_group(group) for name, group in sorted(by_source.items())},
            "by_label_set": {name: summarize_group(group) for name, group in sorted(by_label_set.items())},
            "top_harmful": top_harmful(unique_rows),
        },
        "conclusion": (
            "Actual exception probe labels now show a usable positive replacement signal, but harmful rows remain "
            "concentrated in small text, handwriting and dark/background-sensitive cases. These rows should become "
            "explicit hard negatives or veto targets before replace/drop verifier training."
        ),
        "next_action": (
            "Join this evidence table to action/codebook features and train only a diagnostic verifier with strict "
            "held-out thresholding; do not promote it until actual compact OCR beats current and harmful rows are rejected."
        ),
    }
    return rows, result


def write_report(path: Path, result: dict[str, Any]) -> None:
    unique = result["summary"]["all_unique"]
    duplicate = result["summary"]["all_with_label_set_duplicates"]
    lines = [
        "# Exception Probe Harmful Evidence Audit",
        "",
        "This diagnostic uses actual compact-stream OCR labels only; it is not a deployable verifier.",
        "",
        "## Summary",
        "",
        f"Unique rows: `{unique['rows']}`; labels: `{json.dumps(unique['labels'], sort_keys=True)}`.",
        f"Rows with label-set duplicates retained for comparison: `{duplicate['rows']}`.",
        "",
        "| subset | rows | labels | tess unicode delta | tess latin delta |",
        "|---|---:|---|---:|---:|",
    ]
    for section_name in ["by_label_set", "by_action", "by_source"]:
        lines.extend(["", f"## {section_name}", ""])
        lines.append("| name | rows | labels | tess unicode delta | tess latin delta |")
        lines.append("|---|---:|---|---:|---:|")
        for name, data in sorted(result["summary"][section_name].items()):
            lines.append(
                f"| {name} | {data['rows']} | `{json.dumps(data['labels'], sort_keys=True)}` | "
                f"{data['ocr']['tesseract']['unicode_strict_v1']['delta_errors']} | "
                f"{data['ocr']['tesseract']['latin_alnum_ci_v1']['delta_errors']} |"
            )
    lines.extend(["", "## Harmful Rows", "", "| set | seed | source_index | source | action | ref | cur | exc | unicode | latin | bytes |"])
    lines.append("|---|---:|---:|---|---|---|---|---|---:|---:|---:|")
    for row in result["summary"]["top_harmful"]:
        lines.append(
            f"| {row['label_set']} | {row['seed']} | {row['source_index']} | {row['source']} | "
            f"{row['action_taken']} | {row['reference']} | {row['tesseract_current']} | "
            f"{row['tesseract_exception']} | {row['tesseract_unicode_delta']} | "
            f"{row['tesseract_latin_delta']} | {row['rate_delta_bytes']} |"
        )
    lines.extend(["", "## Conclusion", "", result["conclusion"], "", "## Next Action", "", result["next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def log_wandb(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any] | None:
    if not args.wandb_project:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_id,
            tags=["oscar-lic", "exception-centers", "harmful-evidence", "actual-ocr"],
            config={"experiment_id": args.experiment_id, "label_tables": [str(path) for path in args.label_table]},
        )
        summary = result["summary"]["all_unique"]
        wandb.log(
            {
                "rows": summary["rows"],
                "labels/beneficial": summary["labels"].get("beneficial_no_profile_harm", 0),
                "labels/harmful": summary["labels"].get("harmful_any_profile", 0),
                "labels/mixed": summary["labels"].get("mixed_harm_and_benefit", 0),
                "tesseract/unicode_delta": summary["ocr"]["tesseract"]["unicode_strict_v1"]["delta_errors"],
                "tesseract/latin_delta": summary["ocr"]["tesseract"]["latin_alnum_ci_v1"]["delta_errors"],
                "parseq/unicode_delta": summary["ocr"]["parseq"]["unicode_strict_v1"]["delta_errors"],
            }
        )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, table_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "feature_table", "path": str(table_path), "sha256": sha256_file(table_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for index, table in enumerate(args.label_table):
        files.append({"name": f"label_table_{index}", "path": str(table), "sha256": sha256_file(table)})
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic",
        "command": " ".join(["scripts/analyze_exception_probe_harmful_evidence.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-table", type=Path, action="append", required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_probe_harmful_evidence_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, result = build_result(args)
    result["wandb"] = log_wandb(args, result)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_table, rows)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.output_table, args.report)
    print(json.dumps({"rows": len(rows), "result": str(args.result), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
