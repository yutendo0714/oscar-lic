#!/usr/bin/env python3
"""Audit actual OSCR section accounting for existing codebook streams."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

BASE_STRUCTURAL_BYTES = 44 + 40 + 4
OPTIONAL_SECTION_TABLE_BYTES = 40
GATE_HEADER_BYTES = 14
CODE_HEADER_BYTES = 6
SECTION_TEXT_SYMBOLS = 14


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


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "sum": 0.0, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def parse_stream_set_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--stream-set must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("stream-set label must be nonempty")
    return label, Path(path)


def stream_row(row: dict[str, Any], *, label: str, root_dir: Path) -> dict[str, Any]:
    stream_path = ROOT / row["stream_path"]
    parsed = read_container(stream_path)
    section_lengths = {int(section.section_type): int(section.length) for section in parsed.sections}
    selected = int(row.get("selected_candidate_count", 0))
    has_optional = selected > 0
    structural = int(row.get("structural_overhead_bytes", parsed.total_bytes - sum(section_lengths.values())))
    optional_table = max(0, structural - BASE_STRUCTURAL_BYTES)
    gate_payload = int(row.get("gate_payload_bytes", section_lengths.get(10, 0)))
    code_payload = int(row.get("code_payload_bytes", section_lengths.get(12, 0)))
    compact_payload = int(row.get("compact_payload_bytes", section_lengths.get(SECTION_TEXT_SYMBOLS, 0)))
    gate_body = max(0, gate_payload - (GATE_HEADER_BYTES if has_optional else 0))
    code_body = max(0, code_payload - (CODE_HEADER_BYTES if has_optional else 0))
    logical_bytes = gate_body + code_body + compact_payload
    optional_payload = gate_payload + code_payload + compact_payload
    optional_total = optional_payload + optional_table
    pixels = int(row["width"]) * int(row["height"])
    return {
        "label": label,
        "root_dir": rel(root_dir),
        "index": int(row["index"]),
        "stream_path": rel(stream_path),
        "stream_sha256": sha256_file(stream_path),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "pixels": pixels,
        "section_count": len(parsed.sections),
        "selected_candidate_count": selected,
        "has_optional": has_optional,
        "actual_total_bytes": int(row["actual_total_bytes"]),
        "parsed_total_bytes": int(parsed.total_bytes),
        "base_payload_bytes": int(row["base_payload_bytes"]),
        "gate_payload_bytes": gate_payload,
        "code_payload_bytes": code_payload,
        "compact_payload_bytes": compact_payload,
        "gate_body_bytes": gate_body,
        "code_index_bytes": code_body,
        "logical_enhancement_bytes": logical_bytes,
        "optional_payload_bytes": optional_payload,
        "optional_section_table_bytes": optional_table,
        "optional_total_bytes": optional_total,
        "structural_overhead_bytes": structural,
        "actual_total_bpp": float(row["actual_total_bpp"]),
        "base_payload_bpp": float(row["base_payload_bpp"]),
        "enhancement_payload_bpp": float(row["enhancement_payload_bpp"]),
        "optional_table_bpp": 8.0 * optional_table / pixels,
        "optional_total_bpp": 8.0 * optional_total / pixels,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "actual_total_bytes",
        "base_payload_bytes",
        "gate_payload_bytes",
        "code_payload_bytes",
        "compact_payload_bytes",
        "gate_body_bytes",
        "code_index_bytes",
        "logical_enhancement_bytes",
        "optional_payload_bytes",
        "optional_section_table_bytes",
        "optional_total_bytes",
        "structural_overhead_bytes",
        "actual_total_bpp",
        "base_payload_bpp",
        "enhancement_payload_bpp",
        "optional_table_bpp",
        "optional_total_bpp",
        "selected_candidate_count",
    ]
    selected_rows = [row for row in rows if row["has_optional"]]
    selected_streams = len(selected_rows)
    split_optional_rows = [row for row in selected_rows if row["gate_payload_bytes"] > 0 and row["code_payload_bytes"] > 0]
    save_merge_optional = len(split_optional_rows) * OPTIONAL_SECTION_TABLE_BYTES
    save_drop_payload_headers = len(split_optional_rows) * (GATE_HEADER_BYTES + CODE_HEADER_BYTES)
    pixels_sum = sum(int(row["pixels"]) for row in rows)
    return {
        "streams": len(rows),
        "selected_streams": selected_streams,
        "selected_stream_fraction": selected_streams / len(rows) if rows else 0.0,
        "selected_candidate_total": int(sum(row["selected_candidate_count"] for row in rows)),
        "split_optional_streams": len(split_optional_rows),
        "pixel_sum": int(pixels_sum),
        "metrics": {key: summarize([float(row[key]) for row in rows]) for key in numeric_keys},
        "selected_only": {key: summarize([float(row[key]) for row in selected_rows]) for key in numeric_keys},
        "theoretical_savings": {
            "merge_gate_and_code_optional_sections_bytes": int(save_merge_optional),
            "merge_gate_and_code_optional_sections_bpp_over_dataset": 8.0 * save_merge_optional / pixels_sum
            if pixels_sum
            else 0.0,
            "drop_gate_and_code_payload_headers_bytes": int(save_drop_payload_headers),
            "drop_gate_and_code_payload_headers_bpp_over_dataset": 8.0 * save_drop_payload_headers / pixels_sum
            if pixels_sum
            else 0.0,
            "merge_plus_drop_headers_bytes": int(save_merge_optional + save_drop_payload_headers),
            "merge_plus_drop_headers_bpp_over_dataset": 8.0 * (save_merge_optional + save_drop_payload_headers) / pixels_sum
            if pixels_sum
            else 0.0,
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "Actual `.oscr` section accounting over existing MLIC++ latent-codebook streams. This is an accounting diagnostic only; no streams are regenerated.",
        "",
        "| set | streams | selected streams | mean total bpp | mean enhancement payload bpp | mean optional table bpp | selected optional bytes | merge+drop-header saving bpp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in result["sets"].items():
        metrics = summary["metrics"]
        selected = summary["selected_only"]
        saving = summary["theoretical_savings"]["merge_plus_drop_headers_bpp_over_dataset"]
        lines.append(
            f"| {label} | {summary['streams']} | {summary['selected_streams']} | "
            f"{metrics['actual_total_bpp']['mean']:.6f} | "
            f"{metrics['enhancement_payload_bpp']['mean']:.6f} | "
            f"{metrics['optional_table_bpp']['mean']:.6f} | "
            f"{selected['optional_total_bytes']['mean'] if selected['optional_total_bytes']['mean'] is not None else 0:.2f} | "
            f"{saving:.6f} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path, stream_manifest: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-oscr-section-accounting",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
    ]
    for label, meta in result["inputs"].items():
        lines.extend([f"  {label}:", "    roots:"])
        for root_meta in meta["roots"]:
            lines.extend(
                [
                    f"      - root: {root_meta['root']}",
                    "        summary_json:",
                    f"          path: {root_meta['summary_json']}",
                    f"          sha256: {root_meta['summary_sha256']}",
                    "        results_jsonl:",
                    f"          path: {root_meta['results_jsonl']}",
                    f"          sha256: {root_meta['results_sha256']}",
                ]
            )
    lines.extend(
        [
            "outputs:",
            "  result_json:",
            f"    path: {rel(output)}",
            f"    sha256: {sha256_file(output)}",
            "  stream_manifest:",
            f"    path: {rel(stream_manifest)}",
            f"    sha256: {sha256_file(stream_manifest)}",
            "  report:",
            f"    path: {rel(report)}",
            f"    sha256: {sha256_file(report)}",
            "scripts:",
            "  analyzer:",
            "    path: scripts/analyze_oscr_section_accounting.py",
            f"    sha256: {sha256_file(Path(__file__))}",
            "conclusion: accounting diagnostic; no streams regenerated",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_interpretation(sets: dict[str, dict[str, Any]]) -> str:
    labels = set(sets)
    if any("compact" in label for label in labels) and any("split" in label for label in labels):
        return (
            "Split and compact rows use the same selected-candidate decisions and base payloads, so any difference here is "
            "pure transmitted-format accounting rather than model or selector quality. The compact TEXT_SYMBOLS section "
            "removes fixed per-stream gate/code payload headers and one optional section-table entry for selected streams; "
            "the selector problem is unchanged, but actual bpp is lower and should be used for future reported OSCAR-LIC points."
        )
    if labels == {"current", "top4_oracle", "top8_oracle"} or {"current", "top4_oracle", "top8_oracle"}.issubset(labels):
        return (
            "The current/top4/top8 streams use the same selected-candidate budget, so OCR gains are not from extra payload. "
            "For selected word-crop streams, optional section-table and fixed gate/code headers dominate the few logical code/index bytes. "
            "A compact combined optional section would not change the selector problem, but it can reduce actual bpp and should be part of the next compact-symbol implementation."
        )
    return (
        "This diagnostic reports actual .oscr container bytes and payload bytes separately. Interpret selector or OCR quality "
        "only together with the corresponding decode/OCR metrics; these rows are rate-accounting evidence for existing streams."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-set", type=parse_stream_set_arg, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inputs: dict[str, Any] = {}
    manifest_rows = []
    for label, root_dir in args.stream_set:
        root_dir = root_dir.resolve()
        summary_path = root_dir / "summary.json"
        results_path = root_dir / "results.jsonl"
        rows = [stream_row(row, label=label, root_dir=root_dir) for row in read_jsonl(results_path)]
        all_rows_by_label[label].extend(rows)
        inputs.setdefault(label, {"roots": []})["roots"].append(
            {
                "root": rel(root_dir),
                "summary_json": rel(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "results_jsonl": rel(results_path),
                "results_sha256": sha256_file(results_path),
            }
        )
        manifest_rows.extend(rows)
    args.stream_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.stream_manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(manifest_rows, key=lambda item: (item["label"], item["index"], item["stream_path"])):
            handle.write(
                json.dumps(
                    {
                        "label": row["label"],
                        "index": row["index"],
                        "stream_path": row["stream_path"],
                        "stream_sha256": row["stream_sha256"],
                        "actual_total_bytes": row["actual_total_bytes"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    sets = {label: aggregate(rows) for label, rows in all_rows_by_label.items()}
    result = {
        "experiment_id": args.experiment_id,
        "validity": "accounting_diagnostic_existing_streams",
        "code_commit": git_commit(),
        "inputs": inputs,
        "sets": sets,
        "stream_manifest": rel(args.stream_manifest),
        "wandb_run_id": "not_logged",
        "interpretation": make_interpretation(sets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report, args.stream_manifest)
    print(json.dumps({"experiment_id": args.experiment_id, "sets": list(result["sets"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
