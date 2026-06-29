#!/usr/bin/env python3
"""Summarize actual OCR for explicit exception-center streams."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def total_char_errors(summary: dict[str, Any], profile: str) -> int:
    metrics = summary["metrics"][profile]
    return int(metrics["char_substitutions"] + metrics["char_deletions"] + metrics["char_insertions"])


def row_distance(row: dict[str, Any], profile: str) -> int:
    return int(character_counts(str(row["reference"]), str(row["prediction"]), profile).distance)


def selected_indices(policy_dir: Path, seed_offsets: dict[int, int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted(policy_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            seed = int(row["real_seed"])
            global_index = seed_offsets[seed] + int(row["source_index"])
            result[global_index] = row
    return result


def summarize_rows(
    *,
    current_rows: list[dict[str, Any]],
    exception_rows: list[dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    profiles: list[str],
) -> dict[str, Any]:
    if len(current_rows) != len(exception_rows):
        raise ValueError("current and exception OCR rows differ in length")
    result = {}
    subsets = {
        "all": list(range(len(current_rows))),
        "selected": sorted(selected),
        "selected_proxy_oracle_change": sorted(index for index, row in selected.items() if bool(row.get("oracle_change_proxy"))),
        "selected_proxy_noheadroom": sorted(index for index, row in selected.items() if not bool(row.get("oracle_change_proxy"))),
    }
    for name, indices in subsets.items():
        profile_result = {}
        for profile in profiles:
            deltas = []
            examples = []
            for index in indices:
                cur = row_distance(current_rows[index], profile)
                exc = row_distance(exception_rows[index], profile)
                delta = exc - cur
                deltas.append(delta)
                if delta != 0 and len(examples) < 20:
                    examples.append(
                        {
                            "index": index,
                            "delta": delta,
                            "reference": current_rows[index]["reference"],
                            "current_prediction": current_rows[index]["prediction"],
                            "exception_prediction": exception_rows[index]["prediction"],
                            "policy": selected.get(index),
                        }
                    )
            profile_result[profile] = {
                "rows": len(indices),
                "delta_errors": int(sum(deltas)),
                "improved_rows": sum(1 for value in deltas if value < 0),
                "worsened_rows": sum(1 for value in deltas if value > 0),
                "equal_rows": sum(1 for value in deltas if value == 0),
                "examples": examples,
            }
        result[name] = profile_result
    return result


def stream_summary(paths: list[Path]) -> dict[str, Any]:
    summaries = [read_json(path / "summary.json") for path in paths]
    return {
        "runs": [str(path) for path in paths],
        "images": sum(int(item["images"]) for item in summaries),
        "selected_candidates": sum(float(item["avg_selected_candidate_count"]) * int(item["images"]) for item in summaries),
        "mean_actual_total_bpp": sum(float(item["avg_actual_total_bpp"]) for item in summaries) / len(summaries),
        "mean_enhancement_payload_bpp": sum(float(item["avg_enhancement_payload_bpp"]) for item in summaries) / len(summaries),
        "mean_psnr_delta_db": sum(float(item["avg_psnr_delta_db"]) for item in summaries) / len(summaries),
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    selected = selected_indices(args.policy_dir, {1: 0, 2: 75})
    current_tess_summary = read_json(args.current_tesseract_dir / "summary.json")
    exception_tess_summary = read_json(args.exception_tesseract_dir / "summary.json")
    current_parseq_summary = read_json(args.current_parseq_dir / "summary.json")
    exception_parseq_summary = read_json(args.exception_parseq_dir / "summary.json")
    row_comparisons = {
        "tesseract": summarize_rows(
            current_rows=read_jsonl(args.current_tesseract_dir / "results.jsonl"),
            exception_rows=read_jsonl(args.exception_tesseract_dir / "results.jsonl"),
            selected=selected,
            profiles=args.profile,
        ),
        "parseq": summarize_rows(
            current_rows=read_jsonl(args.current_parseq_dir / "results.jsonl"),
            exception_rows=read_jsonl(args.exception_parseq_dir / "results.jsonl"),
            selected=selected,
            profiles=args.profile,
        ),
    }
    aggregate = {}
    for model, current_summary, exception_summary in [
        ("tesseract", current_tess_summary, exception_tess_summary),
        ("parseq", current_parseq_summary, exception_parseq_summary),
    ]:
        aggregate[model] = {}
        for profile in args.profile:
            aggregate[model][profile] = {
                "current_char_errors": total_char_errors(current_summary, profile),
                "exception_char_errors": total_char_errors(exception_summary, profile),
                "delta_errors_exception_minus_current": total_char_errors(exception_summary, profile)
                - total_char_errors(current_summary, profile),
                "current_cer": current_summary["metrics"][profile]["cer_micro"],
                "exception_cer": exception_summary["metrics"][profile]["cer_micro"],
            }
    return {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-center-actual-ocr-diagnostic",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "selected_policy": {
            "policy_dir": str(args.policy_dir),
            "selected_rows": len(selected),
            "oracle_change_proxy_rows": sum(1 for row in selected.values() if bool(row.get("oracle_change_proxy"))),
            "noheadroom_proxy_rows": sum(1 for row in selected.values() if not bool(row.get("oracle_change_proxy"))),
        },
        "rate": {
            "current": stream_summary(args.current_stream_dir),
            "exception": stream_summary(args.exception_stream_dir),
        },
        "aggregate_ocr": aggregate,
        "row_comparisons": row_comparisons,
        "wandb_runs": {
            "exception_tesseract": exception_tess_summary.get("wandb_run_id"),
            "exception_parseq": exception_parseq_summary.get("wandb_run_id"),
            "current_tesseract": current_tess_summary.get("wandb_run_id"),
            "current_parseq": current_parseq_summary.get("wandb_run_id"),
        },
        "conclusion": (
            "The rel<=1.0 explicit exception-center policy lowers enhancement rate by selecting fewer candidates, "
            "but it is not an OCR improvement: PARSeq is unchanged and Tesseract worsens versus the matched current "
            "compact seed1/2 subset."
        ),
        "next_action": (
            "Do not run OCR for more simple exception-distance thresholds. Add a stronger no-headroom verifier or "
            "candidate-local visual/code-effect features before another actual-stream exception-center smoke."
        ),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception-Center Actual OCR Diagnostic",
        "",
        f"Selected rows: `{result['selected_policy']['selected_rows']}` "
        f"(proxy oracle-change `{result['selected_policy']['oracle_change_proxy_rows']}`, "
        f"proxy no-headroom `{result['selected_policy']['noheadroom_proxy_rows']}`).",
        "",
        "## Rate",
        "",
        "| policy | images | selected candidates | mean bpp | enh bpp | PSNR delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, data in result["rate"].items():
        lines.append(
            f"| {name} | {data['images']} | {data['selected_candidates']:.0f} | "
            f"{data['mean_actual_total_bpp']:.6f} | {data['mean_enhancement_payload_bpp']:.6f} | "
            f"{data['mean_psnr_delta_db']:.6f} |"
        )
    lines.extend(["", "## OCR Aggregate", "", "| model | profile | current errors | exception errors | delta |", "|---|---|---:|---:|---:|"])
    for model, profiles in result["aggregate_ocr"].items():
        for profile, data in profiles.items():
            lines.append(
                f"| {model} | {profile} | {data['current_char_errors']} | "
                f"{data['exception_char_errors']} | {data['delta_errors_exception_minus_current']} |"
            )
    lines.extend(["", "## Selected-Row Delta", "", "| model | profile | selected delta | improved | worsened | equal |"])
    lines.append("|---|---|---:|---:|---:|---:|")
    for model, subsets in result["row_comparisons"].items():
        for profile, data in subsets["selected"].items():
            lines.append(
                f"| {model} | {profile} | {data['delta_errors']} | {data['improved_rows']} | "
                f"{data['worsened_rows']} | {data['equal_rows']} |"
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
            tags=["oscar-lic", "exception-centers", "actual-ocr", "diagnostic"],
            config={"experiment_id": args.experiment_id},
        )
        for model, profiles in result["aggregate_ocr"].items():
            for profile, data in profiles.items():
                wandb.log({f"{model}/{profile}/delta_errors": data["delta_errors_exception_minus_current"]})
        wandb.log(
            {
                "rate/current_bpp": result["rate"]["current"]["mean_actual_total_bpp"],
                "rate/exception_bpp": result["rate"]["exception"]["mean_actual_total_bpp"],
                "policy/selected_rows": result["selected_policy"]["selected_rows"],
            }
        )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover - wandb availability is environment-dependent.
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for name, directory in [
        ("policy_dir_result", args.policy_result),
        ("current_tesseract_summary", args.current_tesseract_dir / "summary.json"),
        ("current_tesseract_results", args.current_tesseract_dir / "results.jsonl"),
        ("exception_tesseract_summary", args.exception_tesseract_dir / "summary.json"),
        ("exception_tesseract_results", args.exception_tesseract_dir / "results.jsonl"),
        ("current_parseq_summary", args.current_parseq_dir / "summary.json"),
        ("current_parseq_results", args.current_parseq_dir / "results.jsonl"),
        ("exception_parseq_summary", args.exception_parseq_dir / "summary.json"),
        ("exception_parseq_results", args.exception_parseq_dir / "results.jsonl"),
    ]:
        files.append({"name": name, "path": str(directory), "sha256": sha256_file(directory)})
    for directory in args.current_stream_dir + args.exception_stream_dir:
        files.append({"name": f"{directory.name}_summary", "path": str(directory / "summary.json"), "sha256": sha256_file(directory / "summary.json")})
        files.append({"name": f"{directory.name}_results", "path": str(directory / "results.jsonl"), "sha256": sha256_file(directory / "results.jsonl")})
    data = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "command": " ".join(["scripts/summarize_exception_center_actual_ocr.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--policy-result", type=Path, required=True)
    parser.add_argument("--current-stream-dir", type=Path, action="append", required=True)
    parser.add_argument("--exception-stream-dir", type=Path, action="append", required=True)
    parser.add_argument("--current-tesseract-dir", type=Path, required=True)
    parser.add_argument("--exception-tesseract-dir", type=Path, required=True)
    parser.add_argument("--current-parseq-dir", type=Path, required=True)
    parser.add_argument("--exception-parseq-dir", type=Path, required=True)
    parser.add_argument("--profile", action="append", default=["unicode_strict_v1", "latin_alnum_ci_v1"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_center_rel100_actual_ocr_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_result(args)
    result["wandb"] = log_wandb(args, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.output, args.report)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
