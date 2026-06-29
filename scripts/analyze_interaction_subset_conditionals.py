#!/usr/bin/env python3
"""Analyze conditional utility edges in all-subset interaction smokes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Iterable

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


def parse_policy(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("policy must be name=ocr_results:stream_results")
    name, rest = parts
    fields = rest.split(":")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("policy must be name=ocr_results:stream_results")
    return name, Path(fields[0]), Path(fields[1])


def policy_members(name: str) -> tuple[str, ...]:
    if name.startswith("single_"):
        return (name.removeprefix("single_"),)
    if name.startswith("pair_"):
        return tuple(name.removeprefix("pair_"))
    if name.startswith("subset3_"):
        return tuple(name.removeprefix("subset3_"))
    raise ValueError(f"Unsupported policy name for subset parsing: {name}")


def distance(row: dict, profile: str) -> int:
    return character_counts(str(row["reference"]), str(row["prediction"]), profile).distance


def summarize_values(values: Iterable[float]) -> dict:
    values = list(values)
    if not values:
        return {"mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def aggregate_edges(rows: list[dict]) -> dict:
    total_delta = int(sum(row["delta_distance"] for row in rows))
    total_added_bpp = float(sum(row["added_actual_bpp"] for row in rows))
    total_added_enhancement_bpp = float(sum(row["added_enhancement_bpp"] for row in rows))
    return {
        "edge_rows": len(rows),
        "improved_edges": int(sum(1 for row in rows if row["delta_distance"] < 0)),
        "worsened_edges": int(sum(1 for row in rows if row["delta_distance"] > 0)),
        "unchanged_edges": int(sum(1 for row in rows if row["delta_distance"] == 0)),
        "total_delta_distance": total_delta,
        "mean_delta_distance": float(np.mean([row["delta_distance"] for row in rows])) if rows else None,
        "mean_added_actual_bpp": float(np.mean([row["added_actual_bpp"] for row in rows])) if rows else None,
        "mean_added_enhancement_bpp": float(np.mean([row["added_enhancement_bpp"] for row in rows])) if rows else None,
        "mean_added_psnr_delta_db": float(np.mean([row["added_psnr_delta_db"] for row in rows])) if rows else None,
        "gain_per_added_actual_bpp": None if total_added_bpp <= 0 else float((-total_delta) / total_added_bpp),
        "gain_per_added_enhancement_bpp": None
        if total_added_enhancement_bpp <= 0
        else float((-total_delta) / total_added_enhancement_bpp),
    }


def aggregate_metric(prefix: str, value: object, out: dict) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            aggregate_metric(f"{prefix}_{key}", child, out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ocr", type=Path, required=True)
    parser.add_argument("--base-stream-results", type=Path, required=True)
    parser.add_argument("--policy", type=parse_policy, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="interaction_subset_conditionals")
    args = parser.parse_args()

    base_ocr = read_jsonl(args.base_ocr)
    base_stream = read_jsonl(args.base_stream_results)
    if len(base_ocr) != len(base_stream):
        raise SystemExit("base OCR and stream row count mismatch")

    policies: dict[tuple[str, ...], dict] = {}
    for name, ocr_path, stream_path in args.policy:
        members = tuple(sorted(policy_members(name)))
        ocr_rows = read_jsonl(ocr_path)
        stream_rows = read_jsonl(stream_path)
        if len(ocr_rows) != len(base_ocr) or len(stream_rows) != len(base_ocr):
            raise SystemExit(f"{name}: row count mismatch")
        policies[members] = {
            "name": name,
            "members": members,
            "ocr_path": str(ocr_path),
            "stream_path": str(stream_path),
            "ocr_rows": ocr_rows,
            "stream_rows": stream_rows,
        }

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
        raise SystemExit(f"missing required subset policies: {missing}")

    states: dict[tuple[str, ...], dict] = {
        tuple(): {
            "name": "base",
            "members": tuple(),
            "ocr_path": str(args.base_ocr),
            "stream_path": str(args.base_stream_results),
            "ocr_rows": base_ocr,
            "stream_rows": base_stream,
        }
    }
    states.update(policies)

    edge_specs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    candidates = ("a", "b", "c")
    for source in [tuple(), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]:
        source_set = set(source)
        for candidate in candidates:
            if candidate in source_set:
                continue
            target = tuple(sorted([*source, candidate]))
            if target in states:
                edge_specs.append((source, target))

    results = {
        "experiment_id": args.experiment_id,
        "base_ocr": str(args.base_ocr),
        "base_stream_results": str(args.base_stream_results),
        "policies": [
            {
                "name": item["name"],
                "members": list(members),
                "ocr_path": item["ocr_path"],
                "stream_path": item["stream_path"],
            }
            for members, item in sorted(policies.items(), key=lambda kv: (len(kv[0]), kv[0]))
        ],
        "profiles": {},
    }

    for profile in PROFILES:
        distances: dict[tuple[str, ...], list[int]] = {}
        for members, state in states.items():
            distances[members] = [distance(row, profile) for row in state["ocr_rows"]]

        edge_rows = []
        edge_groups: dict[str, list[dict]] = defaultdict(list)
        edge_names: dict[str, list[dict]] = defaultdict(list)
        for source, target in edge_specs:
            source_state = states[source]
            target_state = states[target]
            transition = f"{len(source)}_to_{len(target)}"
            edge_name = f"{source_state['name']}_to_{target_state['name']}"
            for index in range(len(base_ocr)):
                source_stream = source_state["stream_rows"][index]
                target_stream = target_state["stream_rows"][index]
                source_psnr = 0.0 if not source else float(source_stream["psnr_delta_db"])
                target_psnr = float(target_stream["psnr_delta_db"])
                row = {
                    "index": index,
                    "source_policy": source_state["name"],
                    "target_policy": target_state["name"],
                    "transition": transition,
                    "source_distance": int(distances[source][index]),
                    "target_distance": int(distances[target][index]),
                    "delta_distance": int(distances[target][index] - distances[source][index]),
                    "added_actual_bpp": float(target_stream["actual_total_bpp"] - source_stream["actual_total_bpp"]),
                    "added_enhancement_bpp": float(
                        target_stream["enhancement_payload_bpp"] - (0.0 if not source else source_stream["enhancement_payload_bpp"])
                    ),
                    "added_psnr_delta_db": float(target_psnr - source_psnr),
                }
                edge_rows.append(row)
                edge_groups[transition].append(row)
                edge_names[edge_name].append(row)

        per_sample = []
        for index, base_row in enumerate(base_ocr):
            single_choices = [(distances[members][index], members) for members in [("a",), ("b",), ("c",)]]
            pair_choices = [
                (distances[members][index], members) for members in [("a", "b"), ("a", "c"), ("b", "c")]
            ]
            best_single_dist, best_single = min(single_choices, key=lambda item: (item[0], item[1]))
            best_pair_dist, best_pair = min(pair_choices, key=lambda item: (item[0], item[1]))
            triple = ("a", "b", "c")
            triple_dist = distances[triple][index]
            all_choices = single_choices + pair_choices + [(triple_dist, triple)]
            best_any_dist, best_any = min(all_choices, key=lambda item: (item[0], len(item[1]), item[1]))
            per_sample.append(
                {
                    "index": index,
                    "reference": base_row["reference"],
                    "source": base_row.get("source"),
                    "base_distance": int(distances[tuple()][index]),
                    "best_single_policy": states[best_single]["name"],
                    "best_single_distance": int(best_single_dist),
                    "best_pair_policy": states[best_pair]["name"],
                    "best_pair_distance": int(best_pair_dist),
                    "triple_distance": int(triple_dist),
                    "best_any_policy": states[best_any]["name"],
                    "best_any_cardinality": len(best_any),
                    "best_any_distance": int(best_any_dist),
                    "best_pair_minus_best_single_delta": int(best_pair_dist - best_single_dist),
                    "triple_minus_best_single_delta": int(triple_dist - best_single_dist),
                    "best_single_regret_to_best_any": int(best_single_dist - best_any_dist),
                }
            )

        profile_result = {
            "edge_transition_summary": {
                transition: aggregate_edges(rows) for transition, rows in sorted(edge_groups.items())
            },
            "edge_name_summary": {name: aggregate_edges(rows) for name, rows in sorted(edge_names.items())},
            "best_subset_summary": {
                "samples": len(per_sample),
                "best_any_cardinality_counts": {
                    str(card): int(sum(1 for row in per_sample if row["best_any_cardinality"] == card))
                    for card in (1, 2, 3)
                },
                "best_pair_better_than_best_single_samples": int(
                    sum(1 for row in per_sample if row["best_pair_minus_best_single_delta"] < 0)
                ),
                "triple_better_than_best_single_samples": int(
                    sum(1 for row in per_sample if row["triple_minus_best_single_delta"] < 0)
                ),
                "total_best_single_regret_to_best_any": int(
                    sum(row["best_single_regret_to_best_any"] for row in per_sample)
                ),
                "best_pair_minus_best_single_delta": summarize_values(
                    row["best_pair_minus_best_single_delta"] for row in per_sample
                ),
                "triple_minus_best_single_delta": summarize_values(
                    row["triple_minus_best_single_delta"] for row in per_sample
                ),
            },
            "per_sample": per_sample,
        }
        results["profiles"][profile] = profile_result

    scalar_metrics = {}
    for profile, profile_result in results["profiles"].items():
        aggregate_metric(f"{profile}_edge_transition_summary", profile_result["edge_transition_summary"], scalar_metrics)
        aggregate_metric(f"{profile}_best_subset_summary", profile_result["best_subset_summary"], scalar_metrics)
    results["aggregate"] = {"scalar_metrics": {key: {"value": value} for key, value in scalar_metrics.items()}}

    write_json(args.output, results)
    write_report(args.report, results)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


def write_report(path: Path, results: dict) -> None:
    def fmt(value: object, digits: int = 4) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"

    lines = [
        f"# {results['experiment_id']}",
        "",
        "This report converts all-subset OCR outcomes into conditional add-one-candidate utility edges.",
        "Negative delta means the added candidate reduced OCR edit distance.",
        "",
    ]
    for profile, profile_result in results["profiles"].items():
        lines.append(f"## {profile}")
        lines.append("")
        lines.append("### Transition Summary")
        lines.append("")
        lines.append(
            "| transition | edge rows | delta chars | improved/worse/unchanged | added actual bpp | added enh bpp | added PSNR | gain/add-enh-bpp |"
        )
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
        for transition, row in profile_result["edge_transition_summary"].items():
            lines.append(
                f"| {transition} | {row['edge_rows']} | {row['total_delta_distance']} | "
                f"{row['improved_edges']}/{row['worsened_edges']}/{row['unchanged_edges']} | "
                f"{fmt(row['mean_added_actual_bpp'])} | {fmt(row['mean_added_enhancement_bpp'])} | "
                f"{fmt(row['mean_added_psnr_delta_db'])} | {fmt(row['gain_per_added_enhancement_bpp'])} |"
            )
        lines.append("")
        summary = profile_result["best_subset_summary"]
        lines.append("### Best Subset")
        lines.append("")
        lines.append(f"Best-any cardinality counts: `{summary['best_any_cardinality_counts']}`")
        lines.append(
            f"Best pair beats best single on `{summary['best_pair_better_than_best_single_samples']}/"
            f"{summary['samples']}` samples."
        )
        lines.append(
            f"Triple beats best single on `{summary['triple_better_than_best_single_samples']}/"
            f"{summary['samples']}` samples."
        )
        lines.append(
            f"Total regret from stopping after the best single instead of best-any subset: "
            f"`{summary['total_best_single_regret_to_best_any']}` edit errors."
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
