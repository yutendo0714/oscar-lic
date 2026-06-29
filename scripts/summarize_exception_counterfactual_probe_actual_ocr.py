#!/usr/bin/env python3
"""Summarize actual OCR labels for exception counterfactual probe streams."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, normalize_text  # noqa: E402


SEED_RE = re.compile(r"seed(?P<seed>\d+)")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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


def infer_seed(path: Path) -> int:
    match = SEED_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group("seed"))


def distance(reference: str, prediction: str, profile: str) -> int:
    return int(character_counts(reference, prediction, profile).distance)


def exact(reference: str, prediction: str, profile: str) -> bool:
    return normalize_text(reference, profile) == normalize_text(prediction, profile)


def load_seed_ordered_rows(paths: list[Path], file_name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths, key=infer_seed):
        rows.extend(read_jsonl(path / file_name))
    return rows


def aggregate_rate(stream_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pixels = sum(int(row["width"]) * int(row["height"]) for row in stream_rows)
    bytes_total = sum(int(row["actual_total_bytes"]) for row in stream_rows)
    return {
        "images": len(stream_rows),
        "actual_total_bytes": int(bytes_total),
        "actual_total_bpp_weighted": None if pixels == 0 else float(8.0 * bytes_total / pixels),
        "actual_total_bpp_mean": None if not stream_rows else float(sum(float(row["actual_total_bpp"]) for row in stream_rows) / len(stream_rows)),
        "enhancement_payload_bpp_mean": None
        if not stream_rows
        else float(sum(float(row["enhancement_payload_bpp"]) for row in stream_rows) / len(stream_rows)),
        "selected_candidates": int(sum(int(row["selected_candidate_count"]) for row in stream_rows)),
    }


def summarize_ocr(rows: list[dict[str, Any]], model: str, profile: str) -> dict[str, Any]:
    deltas = [int(row["ocr"][model][profile]["delta_errors"]) for row in rows]
    return {
        "rows": len(rows),
        "delta_errors": int(sum(deltas)),
        "improved_rows": int(sum(1 for value in deltas if value < 0)),
        "worsened_rows": int(sum(1 for value in deltas if value > 0)),
        "equal_rows": int(sum(1 for value in deltas if value == 0)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[row["action_taken"]].append(row)
        by_source[str(row["source"])].append(row)
        for tag in row["probe_tags"]:
            by_tag[tag].append(row)
    result: dict[str, Any] = {
        "rows": len(rows),
        "strict_labels": dict(Counter(row["strict_label"] for row in rows)),
        "actions": dict(Counter(row["action_taken"] for row in rows)),
        "rate": {
            "current": aggregate_rate([row["current_stream"] for row in rows]),
            "exception": aggregate_rate([row["exception_stream"] for row in rows]),
        },
        "ocr": {
            model: {
                profile: summarize_ocr(rows, model, profile)
                for profile in ["unicode_strict_v1", "latin_alnum_ci_v1"]
            }
            for model in ["tesseract", "parseq"]
        },
        "by_action": {},
        "by_source": {},
        "by_tag": {},
    }
    for name, subset in by_action.items():
        result["by_action"][name] = mini_summary(subset)
    for name, subset in by_source.items():
        result["by_source"][name] = mini_summary(subset)
    for name, subset in by_tag.items():
        result["by_tag"][name] = mini_summary(subset)
    return result


def mini_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "strict_labels": dict(Counter(row["strict_label"] for row in rows)),
        "tesseract_unicode": summarize_ocr(rows, "tesseract", "unicode_strict_v1"),
        "tesseract_latin": summarize_ocr(rows, "tesseract", "latin_alnum_ci_v1"),
        "parseq_unicode": summarize_ocr(rows, "parseq", "unicode_strict_v1"),
        "parseq_latin": summarize_ocr(rows, "parseq", "latin_alnum_ci_v1"),
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    probes = [row for row in read_jsonl(args.probe_table) if bool(row["training_use"])]
    current_stream = load_seed_ordered_rows(args.current_stream_dir, "results.jsonl")
    exception_stream = load_seed_ordered_rows(args.exception_stream_dir, "results.jsonl")
    current_tess = read_jsonl(args.current_tesseract_dir / "results.jsonl")
    exception_tess = read_jsonl(args.exception_tesseract_dir / "results.jsonl")
    current_parseq = read_jsonl(args.current_parseq_dir / "results.jsonl")
    exception_parseq = read_jsonl(args.exception_parseq_dir / "results.jsonl")
    expected = len(probes)
    for name, rows in [
        ("current_stream", current_stream),
        ("exception_stream", exception_stream),
        ("current_tesseract", current_tess),
        ("exception_tesseract", exception_tess),
        ("current_parseq", current_parseq),
        ("exception_parseq", exception_parseq),
    ]:
        if len(rows) != expected:
            raise ValueError(f"{name} rows {len(rows)} != probes {expected}")

    out = []
    for index, probe in enumerate(probes):
        cur_s = current_stream[index]
        exc_s = exception_stream[index]
        action = "replace_with_exception" if int(exc_s["selected_candidate_count"]) > 0 else "drop_current"
        row: dict[str, Any] = {
            "probe_index": int(probe["probe_index"]),
            "seed": int(probe["seed"]),
            "source_index": int(probe["source_index"]),
            "candidate_index": int(probe["candidate_index"]),
            "source": probe["source"],
            "reference": probe["reference"],
            "probe_tags": probe["probe_tags"],
            "policy_case": probe["policy_case"],
            "action_taken": action,
            "current_stream": cur_s,
            "exception_stream": exc_s,
            "rate_delta": {
                "actual_total_bytes": int(exc_s["actual_total_bytes"]) - int(cur_s["actual_total_bytes"]),
                "actual_total_bpp": float(exc_s["actual_total_bpp"]) - float(cur_s["actual_total_bpp"]),
                "enhancement_payload_bpp": float(exc_s["enhancement_payload_bpp"]) - float(cur_s["enhancement_payload_bpp"]),
            },
            "ocr": {},
        }
        deltas = []
        for model, current_rows, exception_rows in [
            ("tesseract", current_tess, exception_tess),
            ("parseq", current_parseq, exception_parseq),
        ]:
            cur_o = current_rows[index]
            exc_o = exception_rows[index]
            if str(cur_o["reference"]) != str(exc_o["reference"]):
                raise ValueError(f"{model} reference mismatch at probe index {index}")
            row["ocr"][model] = {
                "current_prediction": str(cur_o["prediction"]),
                "exception_prediction": str(exc_o["prediction"]),
            }
            for profile in ["unicode_strict_v1", "latin_alnum_ci_v1"]:
                cur_d = distance(str(cur_o["reference"]), str(cur_o["prediction"]), profile)
                exc_d = distance(str(exc_o["reference"]), str(exc_o["prediction"]), profile)
                cur_exact = exact(str(cur_o["reference"]), str(cur_o["prediction"]), profile)
                exc_exact = exact(str(exc_o["reference"]), str(exc_o["prediction"]), profile)
                delta = exc_d - cur_d
                deltas.append(delta)
                row["ocr"][model][profile] = {
                    "current_distance": cur_d,
                    "exception_distance": exc_d,
                    "delta_errors": delta,
                    "current_exact": cur_exact,
                    "exception_exact": exc_exact,
                    "delta_exact": int(exc_exact) - int(cur_exact),
                }
        harmful = any(delta > 0 for delta in deltas)
        beneficial = any(delta < 0 for delta in deltas)
        if harmful and beneficial:
            label = "mixed_harm_and_benefit"
        elif harmful:
            label = "harmful_any_profile"
        elif beneficial:
            label = "beneficial_no_profile_harm"
        else:
            label = "neutral_all_profiles"
        row["strict_label"] = label
        out.append(row)
    return out


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception Counterfactual Probe Actual OCR",
        "",
        "Negative deltas mean the exception/drop stream has fewer OCR edit errors than the matched current stream.",
        "",
        "## Summary",
        "",
        f"Rows: `{result['summary']['rows']}`; labels: `{json.dumps(result['summary']['strict_labels'], sort_keys=True)}`.",
        "",
        "| model | profile | delta | improved | worsened | equal |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, profiles in result["summary"]["ocr"].items():
        for profile, metrics in profiles.items():
            lines.append(
                f"| {model} | {profile} | {metrics['delta_errors']} | {metrics['improved_rows']} | "
                f"{metrics['worsened_rows']} | {metrics['equal_rows']} |"
            )
    lines.extend(["", "## By Action", "", "| action | rows | labels | tess unicode delta | tess latin delta |", "|---|---:|---|---:|---:|"])
    for action, data in sorted(result["summary"]["by_action"].items()):
        lines.append(
            f"| {action} | {data['rows']} | `{json.dumps(data['strict_labels'], sort_keys=True)}` | "
            f"{data['tesseract_unicode']['delta_errors']} | {data['tesseract_latin']['delta_errors']} |"
        )
    lines.extend(["", "## By Tag", "", "| tag | rows | labels | tess unicode delta | tess latin delta |", "|---|---:|---|---:|---:|"])
    for tag, data in sorted(result["summary"]["by_tag"].items()):
        lines.append(
            f"| {tag} | {data['rows']} | `{json.dumps(data['strict_labels'], sort_keys=True)}` | "
            f"{data['tesseract_unicode']['delta_errors']} | {data['tesseract_latin']['delta_errors']} |"
        )
    lines.extend(["", "## Rate", ""])
    for name, rate in result["summary"]["rate"].items():
        lines.append(
            f"- `{name}`: weighted bpp `{rate['actual_total_bpp_weighted']:.6f}`, "
            f"mean bpp `{rate['actual_total_bpp_mean']:.6f}`, selected candidates `{rate['selected_candidates']}`."
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
            tags=["oscar-lic", "exception-centers", "counterfactual-probe", "actual-ocr"],
            config={"experiment_id": args.experiment_id},
        )
        summary = result["summary"]
        wandb.log(
            {
                "rows": summary["rows"],
                "labels/beneficial": summary["strict_labels"].get("beneficial_no_profile_harm", 0),
                "labels/harmful": summary["strict_labels"].get("harmful_any_profile", 0),
                "labels/mixed": summary["strict_labels"].get("mixed_harm_and_benefit", 0),
                "tesseract/unicode_delta": summary["ocr"]["tesseract"]["unicode_strict_v1"]["delta_errors"],
                "tesseract/latin_delta": summary["ocr"]["tesseract"]["latin_alnum_ci_v1"]["delta_errors"],
                "parseq/unicode_delta": summary["ocr"]["parseq"]["unicode_strict_v1"]["delta_errors"],
                "rate/current_bpp": summary["rate"]["current"]["actual_total_bpp_weighted"],
                "rate/exception_bpp": summary["rate"]["exception"]["actual_total_bpp_weighted"],
            }
        )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover - wandb availability is environment-dependent.
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, table_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "probe_table", "path": str(args.probe_table), "sha256": sha256_file(args.probe_table)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "label_table", "path": str(table_path), "sha256": sha256_file(table_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for prefix, paths in [("current_stream", args.current_stream_dir), ("exception_stream", args.exception_stream_dir)]:
        for directory in paths:
            files.append({"name": f"{prefix}_{directory.name}_summary", "path": str(directory / "summary.json"), "sha256": sha256_file(directory / "summary.json")})
            files.append({"name": f"{prefix}_{directory.name}_results", "path": str(directory / "results.jsonl"), "sha256": sha256_file(directory / "results.jsonl")})
    for name, directory in [
        ("current_tesseract", args.current_tesseract_dir),
        ("exception_tesseract", args.exception_tesseract_dir),
        ("current_parseq", args.current_parseq_dir),
        ("exception_parseq", args.exception_parseq_dir),
    ]:
        files.append({"name": f"{name}_summary", "path": str(directory / "summary.json"), "sha256": sha256_file(directory / "summary.json")})
        files.append({"name": f"{name}_results", "path": str(directory / "results.jsonl"), "sha256": sha256_file(directory / "results.jsonl")})
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_actual_ocr",
        "command": " ".join(["scripts/summarize_exception_counterfactual_probe_actual_ocr.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-table", type=Path, required=True)
    parser.add_argument("--current-stream-dir", type=Path, action="append", required=True)
    parser.add_argument("--exception-stream-dir", type=Path, action="append", required=True)
    parser.add_argument("--current-tesseract-dir", type=Path, required=True)
    parser.add_argument("--exception-tesseract-dir", type=Path, required=True)
    parser.add_argument("--current-parseq-dir", type=Path, required=True)
    parser.add_argument("--exception-parseq-dir", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_counterfactual_probe_actual_ocr_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-counterfactual-hard-negative-labels",
        "status": "completed_actual_ocr",
        "code_commit": git_commit(),
        "summary": summarize_rows(rows),
        "conclusion": (
            "The mined probe set contains useful current-relative supervision: drop-current can help on some rows, "
            "but it also creates hard harmful cases. These labels are suitable as hard-negative diagnostics, not a "
            "standalone training distribution yet."
        ),
        "next_action": (
            "Use the harmful/mixed probe rows as hard negatives and collect more close current-only probes before "
            "training a replace/drop verifier."
        ),
    }
    result["wandb"] = log_wandb(args, result)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_table, rows)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.output_table, args.report)
    print(json.dumps({"table": str(args.output_table), "result": str(args.result), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
