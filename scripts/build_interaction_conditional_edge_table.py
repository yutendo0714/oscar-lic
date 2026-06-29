#!/usr/bin/env python3
"""Build a JSONL table of conditional add-one OCR utility edges."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts  # noqa: E402


PROFILES = ("unicode_strict_v1", "latin_alnum_ci_v1")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_policy(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("policy must be name=tesseract_ocr:parseq_ocr:stream_results")
    name, rest = parts
    fields = rest.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("policy must be name=tesseract_ocr:parseq_ocr:stream_results")
    return name, Path(fields[0]), Path(fields[1]), Path(fields[2])


def policy_members(name: str) -> tuple[str, ...]:
    if name.startswith("single_"):
        return (name.removeprefix("single_"),)
    if name.startswith("pair_"):
        return tuple(name.removeprefix("pair_"))
    if name.startswith("subset3_"):
        return tuple(name.removeprefix("subset3_"))
    raise ValueError(f"Unsupported policy name: {name}")


def distance(row: dict, profile: str) -> int:
    return character_counts(str(row["reference"]), str(row["prediction"]), profile).distance


def summarize_rows(rows: list[dict], transition: str) -> dict:
    subset = [row for row in rows if row["transition"] == transition]
    if not subset:
        return {}
    out = {"rows": len(subset)}
    for profile in PROFILES:
        tess_key = f"tesseract_delta_{profile}"
        parseq_key = f"parseq_delta_{profile}"
        safe_key = f"label_tesseract_improve_parseq_safe_{profile}"
        out[f"tesseract_delta_sum_{profile}"] = int(sum(row[tess_key] for row in subset))
        out[f"parseq_delta_sum_{profile}"] = int(sum(row[parseq_key] for row in subset))
        out[f"tesseract_improved_{profile}"] = int(sum(1 for row in subset if row[tess_key] < 0))
        out[f"tesseract_worsened_{profile}"] = int(sum(1 for row in subset if row[tess_key] > 0))
        out[f"parseq_worsened_{profile}"] = int(sum(1 for row in subset if row[parseq_key] > 0))
        out[f"safe_tess_improve_parseq_nonworse_{profile}"] = int(sum(1 for row in subset if row[safe_key]))
    out["mean_added_enhancement_bpp"] = float(np.mean([row["added_enhancement_bpp"] for row in subset]))
    out["mean_added_psnr_delta_db"] = float(np.mean([row["added_psnr_delta_db"] for row in subset]))
    return out


def aggregate_metric(prefix: str, value: object, out: dict) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            aggregate_metric(f"{prefix}_{key}", child, out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tesseract-base-ocr", type=Path, required=True)
    parser.add_argument("--parseq-base-ocr", type=Path, required=True)
    parser.add_argument("--base-stream-results", type=Path, required=True)
    parser.add_argument("--policy", type=parse_policy, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="interaction_conditional_edge_table")
    args = parser.parse_args()

    base_tess = read_jsonl(args.tesseract_base_ocr)
    base_parseq = read_jsonl(args.parseq_base_ocr)
    base_stream = read_jsonl(args.base_stream_results)
    if not (len(base_tess) == len(base_parseq) == len(base_stream)):
        raise SystemExit("base row count mismatch")

    policies: dict[tuple[str, ...], dict] = {}
    for name, tess_path, parseq_path, stream_path in args.policy:
        members = tuple(sorted(policy_members(name)))
        tess_rows = read_jsonl(tess_path)
        parseq_rows = read_jsonl(parseq_path)
        stream_rows = read_jsonl(stream_path)
        if not (len(tess_rows) == len(base_tess) == len(parseq_rows) == len(stream_rows)):
            raise SystemExit(f"{name}: row count mismatch")
        policies[members] = {
            "name": name,
            "members": members,
            "tesseract_rows": tess_rows,
            "parseq_rows": parseq_rows,
            "stream_rows": stream_rows,
        }

    states: dict[tuple[str, ...], dict] = {
        tuple(): {
            "name": "base",
            "members": tuple(),
            "tesseract_rows": base_tess,
            "parseq_rows": base_parseq,
            "stream_rows": base_stream,
        }
    }
    states.update(policies)

    required = {
        ("a",),
        ("b",),
        ("c",),
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "b", "c"),
    }
    missing = sorted(required - set(policies))
    if missing:
        raise SystemExit(f"missing policies: {missing}")

    distances: dict[str, dict[tuple[str, ...], dict[str, list[int]]]] = {
        "tesseract": defaultdict(dict),
        "parseq": defaultdict(dict),
    }
    for members, state in states.items():
        for profile in PROFILES:
            distances["tesseract"][members][profile] = [distance(row, profile) for row in state["tesseract_rows"]]
            distances["parseq"][members][profile] = [distance(row, profile) for row in state["parseq_rows"]]

    edge_specs: list[tuple[tuple[str, ...], tuple[str, ...], str]] = []
    candidates = ("a", "b", "c")
    for source in [tuple(), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]:
        source_set = set(source)
        for candidate in candidates:
            if candidate in source_set:
                continue
            target = tuple(sorted([*source, candidate]))
            if target in states:
                edge_specs.append((source, target, candidate))

    rows = []
    for source, target, added_candidate in edge_specs:
        source_state = states[source]
        target_state = states[target]
        transition = f"{len(source)}_to_{len(target)}"
        for index in range(len(base_tess)):
            source_stream = source_state["stream_rows"][index]
            target_stream = target_state["stream_rows"][index]
            source_enh_bpp = 0.0 if not source else float(source_stream["enhancement_payload_bpp"])
            source_psnr_delta = 0.0 if not source else float(source_stream["psnr_delta_db"])
            row = {
                "index": index,
                "source": base_tess[index].get("source"),
                "reference": base_tess[index]["reference"],
                "source_policy": source_state["name"],
                "target_policy": target_state["name"],
                "source_members": list(source),
                "target_members": list(target),
                "added_candidate": added_candidate,
                "transition": transition,
                "source_cardinality": len(source),
                "target_cardinality": len(target),
                "added_actual_bpp": float(target_stream["actual_total_bpp"] - source_stream["actual_total_bpp"]),
                "added_enhancement_bpp": float(target_stream["enhancement_payload_bpp"] - source_enh_bpp),
                "added_psnr_delta_db": float(target_stream["psnr_delta_db"] - source_psnr_delta),
            }
            for profile in PROFILES:
                tess_source = distances["tesseract"][source][profile][index]
                tess_target = distances["tesseract"][target][profile][index]
                parseq_source = distances["parseq"][source][profile][index]
                parseq_target = distances["parseq"][target][profile][index]
                tess_delta = int(tess_target - tess_source)
                parseq_delta = int(parseq_target - parseq_source)
                row[f"tesseract_source_distance_{profile}"] = int(tess_source)
                row[f"tesseract_target_distance_{profile}"] = int(tess_target)
                row[f"tesseract_delta_{profile}"] = tess_delta
                row[f"parseq_source_distance_{profile}"] = int(parseq_source)
                row[f"parseq_target_distance_{profile}"] = int(parseq_target)
                row[f"parseq_delta_{profile}"] = parseq_delta
                row[f"label_tesseract_improve_parseq_safe_{profile}"] = bool(tess_delta < 0 and parseq_delta <= 0)
                row[f"label_any_ocr_improve_no_ocr_worse_{profile}"] = bool(
                    (tess_delta < 0 or parseq_delta < 0) and tess_delta <= 0 and parseq_delta <= 0
                )
            rows.append(row)

    summary = {
        "experiment_id": args.experiment_id,
        "rows": len(rows),
        "transitions": {
            transition: summarize_rows(rows, transition) for transition in ("0_to_1", "1_to_2", "2_to_3")
        },
        "inputs": {
            "tesseract_base_ocr": str(args.tesseract_base_ocr),
            "parseq_base_ocr": str(args.parseq_base_ocr),
            "base_stream_results": str(args.base_stream_results),
        },
    }
    scalar_metrics = {}
    aggregate_metric("transitions", summary["transitions"], scalar_metrics)
    summary["aggregate"] = {"scalar_metrics": {key: {"value": value} for key, value in scalar_metrics.items()}}

    write_jsonl(args.output_jsonl, rows)
    write_json(args.summary, summary)
    write_report(args.report, summary)
    print(json.dumps({"rows": len(rows), "output_jsonl": str(args.output_jsonl), "summary": str(args.summary)}, indent=2))
    return 0


def write_report(path: Path, summary: dict) -> None:
    lines = [
        f"# {summary['experiment_id']}",
        "",
        "This table is a non-deployable diagnostic label substrate for conditional stop/continue modeling.",
        "Labels use OCR outcomes and must not be treated as inference-time features.",
        "",
        "| transition | rows | Tess unicode delta | Tess improved/worse | PARSeq worse | safe Tess-improve/PARSeq-safe | added enh bpp | added PSNR |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for transition, row in summary["transitions"].items():
        lines.append(
            f"| {transition} | {row['rows']} | {row['tesseract_delta_sum_unicode_strict_v1']} | "
            f"{row['tesseract_improved_unicode_strict_v1']}/{row['tesseract_worsened_unicode_strict_v1']} | "
            f"{row['parseq_worsened_unicode_strict_v1']} | "
            f"{row['safe_tess_improve_parseq_nonworse_unicode_strict_v1']} | "
            f"{row['mean_added_enhancement_bpp']:.4f} | {row['mean_added_psnr_delta_db']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: a useful stop/continue target must separate `0_to_1` from `1_to_2`; "
            "the same visual residual evidence has very different OCR effect once a first candidate has already been sent.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
