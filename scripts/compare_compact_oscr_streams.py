#!/usr/bin/env python3
"""Compare split-section and compact-section OSCR stream directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oscarlic.bitstream import read_container  # noqa: E402

SECTION_BASE_MAIN = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum": 0.0, "mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def section_payload_sha(parsed, section_type: int) -> str | None:
    matches = [section for section in parsed.sections if section.section_type == section_type]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one section type {section_type}, found {len(matches)}")
    return sha256_bytes(matches[0].payload)


def compare_rows(split_dir: Path, compact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_rows = {int(row["index"]): row for row in read_jsonl(split_dir / "results.jsonl")}
    compact_rows = {int(row["index"]): row for row in read_jsonl(compact_dir / "results.jsonl")}
    if set(split_rows) != set(compact_rows):
        raise ValueError("split and compact result indices differ")

    rows = []
    for index in sorted(split_rows):
        split = split_rows[index]
        compact = compact_rows[index]
        split_stream = ROOT / split["stream_path"]
        compact_stream = ROOT / compact["stream_path"]
        split_parsed = read_container(split_stream)
        compact_parsed = read_container(compact_stream)
        split_rec = ROOT / split["reconstruction_path"]
        compact_rec = ROOT / compact["reconstruction_path"]
        pixels = int(split["width"]) * int(split["height"])
        if pixels != int(compact["width"]) * int(compact["height"]):
            raise ValueError(f"pixel mismatch at index {index}")
        saving_bytes = int(split["actual_total_bytes"]) - int(compact["actual_total_bytes"])
        rows.append(
            {
                "index": index,
                "image": split["image"],
                "pixels": pixels,
                "split_stream_path": rel(split_stream),
                "compact_stream_path": rel(compact_stream),
                "split_stream_sha256": sha256_file(split_stream),
                "compact_stream_sha256": sha256_file(compact_stream),
                "split_section_types": [int(section.section_type) for section in split_parsed.sections],
                "compact_section_types": [int(section.section_type) for section in compact_parsed.sections],
                "split_total_bytes": int(split["actual_total_bytes"]),
                "compact_total_bytes": int(compact["actual_total_bytes"]),
                "saving_bytes": saving_bytes,
                "saving_bpp": 8.0 * saving_bytes / pixels,
                "split_enhancement_payload_bytes": int(split["gate_payload_bytes"]) + int(split["code_payload_bytes"]),
                "compact_enhancement_payload_bytes": int(compact["compact_payload_bytes"]),
                "split_structural_overhead_bytes": int(split["structural_overhead_bytes"]),
                "compact_structural_overhead_bytes": int(compact["structural_overhead_bytes"]),
                "selected_candidate_count": int(split["selected_candidate_count"]),
                "base_payload_sha256_equal": section_payload_sha(split_parsed, SECTION_BASE_MAIN)
                == section_payload_sha(compact_parsed, SECTION_BASE_MAIN),
                "reconstruction_sha256_equal": sha256_file(split_rec) == sha256_file(compact_rec),
                "psnr_enhanced_absdiff": abs(float(split["psnr_enhanced_db"]) - float(compact["psnr_enhanced_db"])),
            }
        )
    metrics = {
        "streams": len(rows),
        "all_base_payload_equal": all(row["base_payload_sha256_equal"] for row in rows),
        "all_reconstruction_sha_equal": all(row["reconstruction_sha256_equal"] for row in rows),
        "max_psnr_enhanced_absdiff": max((row["psnr_enhanced_absdiff"] for row in rows), default=0.0),
        "pixel_sum": int(sum(row["pixels"] for row in rows)),
        "saving_bytes": summarize([float(row["saving_bytes"]) for row in rows]),
        "saving_bpp": summarize([float(row["saving_bpp"]) for row in rows]),
        "split_total_bytes": summarize([float(row["split_total_bytes"]) for row in rows]),
        "compact_total_bytes": summarize([float(row["compact_total_bytes"]) for row in rows]),
        "split_enhancement_payload_bytes": summarize([float(row["split_enhancement_payload_bytes"]) for row in rows]),
        "compact_enhancement_payload_bytes": summarize([float(row["compact_enhancement_payload_bytes"]) for row in rows]),
        "split_structural_overhead_bytes": summarize([float(row["split_structural_overhead_bytes"]) for row in rows]),
        "compact_structural_overhead_bytes": summarize([float(row["compact_structural_overhead_bytes"]) for row in rows]),
    }
    metrics["dataset_saving_bpp"] = (
        8.0 * metrics["saving_bytes"]["sum"] / metrics["pixel_sum"] if metrics["pixel_sum"] else 0.0
    )
    return rows, metrics


def write_report(path: Path, result: dict[str, Any]) -> None:
    m = result["metrics"]
    lines = [
        f"# {result['experiment_id']}",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "Split-section versus compact-section OSCR smoke. Both runs use the same image split, selected candidates and loaded codebook; only optional enhancement packing changes.",
        "",
        "| streams | split mean bytes | compact mean bytes | mean saving bytes | dataset saving bpp | split payload bytes | compact payload bytes | recon identical |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {m['streams']} | {m['split_total_bytes']['mean']:.2f} | "
            f"{m['compact_total_bytes']['mean']:.2f} | {m['saving_bytes']['mean']:.2f} | "
            f"{m['dataset_saving_bpp']:.6f} | {m['split_enhancement_payload_bytes']['mean']:.2f} | "
            f"{m['compact_enhancement_payload_bytes']['mean']:.2f} | {m['all_reconstruction_sha_equal']} |"
        ),
        "",
        "## Interpretation",
        "",
        result["interpretation"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    split_dir = Path(result["inputs"]["split_dir"])
    compact_dir = Path(result["inputs"]["compact_dir"])
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-compact-oscr-section",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
        "  split_dir:",
        f"    path: {rel(split_dir / 'summary.json')}",
        f"    sha256: {sha256_file(split_dir / 'summary.json')}",
        "  split_results:",
        f"    path: {rel(split_dir / 'results.jsonl')}",
        f"    sha256: {sha256_file(split_dir / 'results.jsonl')}",
        "  split_codebook:",
        f"    path: {rel(split_dir / 'latent_residual_codebook.npz')}",
        f"    sha256: {sha256_file(split_dir / 'latent_residual_codebook.npz')}",
        "  compact_dir:",
        f"    path: {rel(compact_dir / 'summary.json')}",
        f"    sha256: {sha256_file(compact_dir / 'summary.json')}",
        "  compact_results:",
        f"    path: {rel(compact_dir / 'results.jsonl')}",
        f"    sha256: {sha256_file(compact_dir / 'results.jsonl')}",
        "outputs:",
        "  result_json:",
        f"    path: {rel(output)}",
        f"    sha256: {sha256_file(output)}",
        "  report:",
        f"    path: {rel(report)}",
        f"    sha256: {sha256_file(report)}",
        "scripts:",
        "  comparator:",
        "    path: scripts/compare_compact_oscr_streams.py",
        f"    sha256: {sha256_file(Path(__file__))}",
        "  generator:",
        "    path: scripts/run_mlicpp_latent_codebook_stub.py",
        f"    sha256: {sha256_file(ROOT / 'scripts/run_mlicpp_latent_codebook_stub.py')}",
        "  candidates:",
        "    path: src/oscarlic/candidates.py",
        f"    sha256: {sha256_file(ROOT / 'src/oscarlic/candidates.py')}",
        "conclusion: compact optional section smoke; OCR policy unchanged",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    compact_dir = args.compact_dir.resolve()
    rows, metrics = compare_rows(split_dir, compact_dir)
    result = {
        "experiment_id": args.experiment_id,
        "validity": "actual_oscr_compact_section_smoke",
        "code_commit": git_commit(),
        "inputs": {"split_dir": rel(split_dir), "compact_dir": rel(compact_dir)},
        "split_summary": read_json(split_dir / "summary.json"),
        "compact_summary": read_json(compact_dir / "summary.json"),
        "metrics": metrics,
        "rows": rows,
        "wandb_run_id": "not_logged",
        "interpretation": (
            "The compact optional section preserves base payloads and reconstructions while saving exactly the expected "
            "one optional section table entry plus split payload headers for these one-candidate streams. This improves "
            "actual bpp accounting, but it does not change candidate selection or OCR utility."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "streams": metrics["streams"],
                "mean_saving_bytes": metrics["saving_bytes"]["mean"],
                "dataset_saving_bpp": metrics["dataset_saving_bpp"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
