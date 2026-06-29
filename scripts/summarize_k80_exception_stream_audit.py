#!/usr/bin/env python3
"""Summarize actual compact streams from the K80 utility-exception audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


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


def parse_seed_dir(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=RUN_DIR")
    seed_text, path_text = value.split("=", 1)
    return int(seed_text), Path(path_text)


def stream_digest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    missing = []
    total_bytes = 0
    count = 0
    for row in rows:
        stream_path = Path(row["stream_path"])
        if not stream_path.exists():
            missing.append(str(stream_path))
            continue
        payload = stream_path.read_bytes()
        digest.update(str(stream_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        count += 1
        total_bytes += len(payload)
    return {
        "stream_count": count,
        "stream_total_bytes": total_bytes,
        "stream_manifest_sha256": digest.hexdigest(),
        "missing_streams": missing,
    }


def summarize_seed(seed: int, run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.jsonl"
    codebook_path = run_dir / "latent_residual_codebook.npz"
    summary_blob = read_json(summary_path)
    summary = summary_blob.get("summary", summary_blob)
    rows = read_jsonl(results_path)
    streams = stream_digest(rows)
    selected_total = sum(int(row["selected_candidate_count"]) for row in rows)
    changed_total = sum(int(row["assignment_changed_count"]) for row in rows)
    compact_rows = sum(1 for row in rows if int(row.get("compact_payload_bytes", 0)) > 0)
    actual_total_bytes = sum(int(row["actual_total_bytes"]) for row in rows)
    actual_stream_bytes_match = actual_total_bytes == streams["stream_total_bytes"]
    codebook_sizes = Counter(int(row["codebook_size"]) for row in rows)
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "summary_sha256": sha256_file(summary_path),
        "results_sha256": sha256_file(results_path),
        "codebook_sha256": sha256_file(codebook_path),
        "rows": len(rows),
        "selected_candidate_total": selected_total,
        "assignment_changed_total": changed_total,
        "assignment_changed_fraction": changed_total / selected_total if selected_total else 0.0,
        "compact_payload_rows": compact_rows,
        "actual_total_bytes_from_rows": actual_total_bytes,
        "actual_stream_bytes_match": actual_stream_bytes_match,
        "codebook_sizes": {str(key): int(value) for key, value in sorted(codebook_sizes.items())},
        "avg_actual_total_bpp": float(summary["avg_actual_total_bpp"]),
        "avg_base_payload_bpp": float(summary["avg_base_payload_bpp"]),
        "avg_enhancement_payload_bpp": float(summary["avg_enhancement_payload_bpp"]),
        "avg_selected_candidate_count": float(summary["avg_selected_candidate_count"]),
        "avg_assignment_changed_count": float(summary["avg_assignment_changed_count"]),
        "avg_assignment_changed_fraction": float(summary["avg_assignment_changed_fraction"]),
        "avg_assignment_mean_relative_error": float(summary["avg_assignment_mean_relative_error"]),
        "avg_psnr_delta_db": float(summary["avg_psnr_delta_db"]),
        "center_utility_min": float(summary.get("center_utility_min", 0.0)),
        "center_utility_max": float(summary.get("center_utility_max", 0.0)),
        "center_utility_mean": float(summary.get("center_utility_mean", 0.0)),
        "candidate_selection_table": summary.get("candidate_selection_table"),
        "codebook_source": summary.get("codebook_source"),
        "bitstream_format": summary.get("bitstream_format"),
        **streams,
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def log_wandb(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any] | None:
    if not args.wandb_project:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_id,
            tags=["oscar-lic", "mlicpp", "compact-bitstream", "exception-codebook", "diagnostic"],
            config={
                "experiment_id": args.experiment_id,
                "run_dirs": {str(seed): str(path) for seed, path in args.run_dir},
                "hypothesis_id": result["hypothesis_id"],
            },
        )
        aggregate = result["aggregate"]
        wandb.log(
            {
                "aggregate/mean_actual_total_bpp": aggregate["mean_actual_total_bpp"],
                "aggregate/mean_enhancement_payload_bpp": aggregate["mean_enhancement_payload_bpp"],
                "aggregate/total_selected_candidate_count": aggregate["total_selected_candidate_count"],
                "aggregate/total_assignment_changed_count": aggregate["total_assignment_changed_count"],
                "aggregate/any_assignment_changed": int(aggregate["any_assignment_changed"]),
                "aggregate/all_stream_bytes_match": int(aggregate["all_stream_bytes_match"]),
            }
        )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover - wandb availability is environment-dependent.
        return {"error": repr(exc)}


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [summarize_seed(seed, path) for seed, path in sorted(args.run_dir)]
    aggregate = {
        "seed_count": len(seeds),
        "row_count": sum(seed["rows"] for seed in seeds),
        "total_selected_candidate_count": sum(seed["selected_candidate_total"] for seed in seeds),
        "total_assignment_changed_count": sum(seed["assignment_changed_total"] for seed in seeds),
        "mean_actual_total_bpp": mean([seed["avg_actual_total_bpp"] for seed in seeds]),
        "mean_enhancement_payload_bpp": mean([seed["avg_enhancement_payload_bpp"] for seed in seeds]),
        "mean_assignment_changed_fraction": mean([seed["assignment_changed_fraction"] for seed in seeds]),
        "mean_psnr_delta_db": mean([seed["avg_psnr_delta_db"] for seed in seeds]),
        "all_stream_bytes_match": all(seed["actual_stream_bytes_match"] for seed in seeds),
    }
    aggregate["any_assignment_changed"] = aggregate["total_assignment_changed_count"] > 0
    conclusion = (
        "Actual compact .oscr streams were generated for the K80=K64+16 utility-exception codebooks, "
        "but nearest assignment selected no non-nearest/exception code on the 3-seed validation smoke. "
        "Therefore appending exception centers alone is a no-op for the current selected candidates; OCR "
        "evaluation is not informative until an explicit exception selector or no-headroom verifier can select them."
    )
    return {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-utility-exception-symbol-actual-stream",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "inputs": {"run_dirs": {str(seed): str(path) for seed, path in sorted(args.run_dir)}},
        "seeds": seeds,
        "aggregate": aggregate,
        "conclusion": conclusion,
        "next_action": (
            "Do not run more nearest-assignment K80 OCR sweeps. Build a small policy that can explicitly propose "
            "exception centers with a no-headroom guard, then rerun actual compact stream/OCR only if assignment changes."
        ),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    agg = result["aggregate"]
    lines = [
        "# K80 Utility-Exception Compact Stream Audit",
        "",
        "This audit checks actual `.oscr` streams for the K80=K64+16 exception-center codebooks.",
        "The key question is whether appended exception centers are selected by the existing nearest assignment path.",
        "",
        "## Summary",
        "",
        f"- Seeds: `{agg['seed_count']}`; rows: `{agg['row_count']}`.",
        f"- Total selected candidates: `{agg['total_selected_candidate_count']}`.",
        f"- Total assignment changes: `{agg['total_assignment_changed_count']}`.",
        f"- Mean actual total bpp: `{agg['mean_actual_total_bpp']:.6f}`.",
        f"- Mean enhancement payload bpp: `{agg['mean_enhancement_payload_bpp']:.6f}`.",
        f"- All stream byte checks matched results rows: `{agg['all_stream_bytes_match']}`.",
        "",
        "| seed | rows | selected | changed | avg bpp | enh bpp | PSNR delta | stream bytes ok |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["seeds"]:
        lines.append(
            f"| {item['seed']} | {item['rows']} | {item['selected_candidate_total']} | "
            f"{item['assignment_changed_total']} | {item['avg_actual_total_bpp']:.6f} | "
            f"{item['avg_enhancement_payload_bpp']:.6f} | {item['avg_psnr_delta_db']:.6f} | "
            f"{item['actual_stream_bytes_match']} |"
        )
    lines.extend(["", "## Conclusion", "", result["conclusion"], "", "## Next Action", "", result["next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for seed, run_dir in sorted(args.run_dir):
        files.extend(
            [
                {"name": f"seed{seed}_summary", "path": str(run_dir / "summary.json"), "sha256": sha256_file(run_dir / "summary.json")},
                {"name": f"seed{seed}_results", "path": str(run_dir / "results.jsonl"), "sha256": sha256_file(run_dir / "results.jsonl")},
                {
                    "name": f"seed{seed}_codebook",
                    "path": str(run_dir / "latent_residual_codebook.npz"),
                    "sha256": sha256_file(run_dir / "latent_residual_codebook.npz"),
                },
            ]
        )
    data = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "command": " ".join(["scripts/summarize_k80_exception_stream_audit.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=parse_seed_dir, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_k80_exception16_compact_stream_audit_2026_06_26")
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
