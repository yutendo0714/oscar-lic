#!/usr/bin/env python3
"""Summarize all-subset interaction smokes and oracle subset choices."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, evaluate_pairs  # noqa: E402


PROFILES = ("unicode_strict_v1", "latin_alnum_ci_v1")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_policy(value: str) -> tuple[str, int, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("policy must be name=count:ocr_results:stream_results")
    name, rest = parts
    fields = rest.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("policy must be name=count:ocr_results:stream_results")
    return name, int(fields[0]), Path(fields[1]), Path(fields[2])


def distance(row: dict, profile: str) -> int:
    return character_counts(str(row["reference"]), str(row["prediction"]), profile).distance


def policy_label_sort_key(name: str, count: int, avg_bpp: float) -> tuple[int, float, str]:
    return count, avg_bpp, name


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
    parser.add_argument("--experiment-id", default="eval300_interaction_subset3_smoke10_subset_oracle_2026_06_26")
    args = parser.parse_args()

    base_ocr = read_jsonl(args.base_ocr)
    base_stream = read_jsonl(args.base_stream_results)
    policies = []
    for name, count, ocr_path, stream_path in args.policy:
        ocr_rows = read_jsonl(ocr_path)
        stream_rows = read_jsonl(stream_path)
        if len(ocr_rows) != len(base_ocr) or len(stream_rows) != len(base_ocr):
            raise SystemExit(f"{name}: row count mismatch")
        policies.append(
            {
                "name": name,
                "selected_candidate_count": count,
                "ocr_path": str(ocr_path),
                "stream_path": str(stream_path),
                "ocr_rows": ocr_rows,
                "stream_rows": stream_rows,
                "avg_actual_total_bpp": float(np.mean([row["actual_total_bpp"] for row in stream_rows])),
                "avg_enhancement_payload_bpp": float(np.mean([row["enhancement_payload_bpp"] for row in stream_rows])),
                "avg_psnr_delta_db": float(np.mean([row["psnr_delta_db"] for row in stream_rows])),
            }
        )

    results = {"experiment_id": args.experiment_id, "profiles": {}, "policies": []}
    for policy in policies:
        results["policies"].append(
            {
                "name": policy["name"],
                "selected_candidate_count": policy["selected_candidate_count"],
                "avg_actual_total_bpp": policy["avg_actual_total_bpp"],
                "avg_enhancement_payload_bpp": policy["avg_enhancement_payload_bpp"],
                "avg_psnr_delta_db": policy["avg_psnr_delta_db"],
                "ocr_path": policy["ocr_path"],
                "stream_path": policy["stream_path"],
            }
        )

    for profile in PROFILES:
        base_distances = [distance(row, profile) for row in base_ocr]
        base_pairs = [(row["reference"], row["prediction"]) for row in base_ocr]
        profile_result = {
            "base": {
                "ocr_path": str(args.base_ocr),
                "stream_path": str(args.base_stream_results),
                "char_errors": int(sum(base_distances)),
                "metrics": evaluate_pairs(base_pairs, profile).to_dict(),
                "avg_actual_total_bpp": float(np.mean([row["actual_total_bpp"] for row in base_stream])),
                "avg_enhancement_payload_bpp": 0.0,
            },
            "fixed_policies": [],
        }
        policy_distances = {}
        for policy in policies:
            pairs = [(row["reference"], row["prediction"]) for row in policy["ocr_rows"]]
            distances = [distance(row, profile) for row in policy["ocr_rows"]]
            deltas = [dist - base for dist, base in zip(distances, base_distances, strict=True)]
            policy_distances[policy["name"]] = distances
            profile_result["fixed_policies"].append(
                {
                    "name": policy["name"],
                    "selected_candidate_count": policy["selected_candidate_count"],
                    "char_errors": int(sum(distances)),
                    "delta_char_errors": int(sum(deltas)),
                    "improved_samples": int(sum(1 for delta in deltas if delta < 0)),
                    "worsened_samples": int(sum(1 for delta in deltas if delta > 0)),
                    "unchanged_samples": int(sum(1 for delta in deltas if delta == 0)),
                    "metrics": evaluate_pairs(pairs, profile).to_dict(),
                    "avg_actual_total_bpp": policy["avg_actual_total_bpp"],
                    "avg_enhancement_payload_bpp": policy["avg_enhancement_payload_bpp"],
                    "avg_psnr_delta_db": policy["avg_psnr_delta_db"],
                }
            )

        profile_result["fixed_policies"] = sorted(
            profile_result["fixed_policies"],
            key=lambda row: (row["char_errors"], row["selected_candidate_count"], row["avg_actual_total_bpp"], row["name"]),
        )

        oracle_rows = []
        oracle_pairs = []
        for index, base_row in enumerate(base_ocr):
            choices = []
            for policy in policies:
                dist = policy_distances[policy["name"]][index]
                choices.append(
                    (
                        dist,
                        policy_label_sort_key(
                            policy["name"],
                            policy["selected_candidate_count"],
                            float(policy["stream_rows"][index]["actual_total_bpp"]),
                        ),
                        policy,
                    )
                )
            best_dist, _, best_policy = min(choices, key=lambda item: (item[0], item[1]))
            chosen_ocr = best_policy["ocr_rows"][index]
            chosen_stream = best_policy["stream_rows"][index]
            oracle_rows.append(
                {
                    "index": index,
                    "reference": base_row["reference"],
                    "source": base_row.get("source"),
                    "base_prediction": base_row["prediction"],
                    "chosen_policy": best_policy["name"],
                    "chosen_candidate_count": best_policy["selected_candidate_count"],
                    "chosen_prediction": chosen_ocr["prediction"],
                    "base_distance": base_distances[index],
                    "chosen_distance": best_dist,
                    "delta_distance": best_dist - base_distances[index],
                    "actual_total_bpp": float(chosen_stream["actual_total_bpp"]),
                    "enhancement_payload_bpp": float(chosen_stream["enhancement_payload_bpp"]),
                    "psnr_delta_db": float(chosen_stream["psnr_delta_db"]),
                }
            )
            oracle_pairs.append((base_row["reference"], chosen_ocr["prediction"]))

        count_by_policy = Counter(row["chosen_policy"] for row in oracle_rows)
        count_by_cardinality = Counter(str(row["chosen_candidate_count"]) for row in oracle_rows)
        profile_result["subset_oracle"] = {
            "char_errors": int(sum(row["chosen_distance"] for row in oracle_rows)),
            "delta_char_errors": int(sum(row["delta_distance"] for row in oracle_rows)),
            "metrics": evaluate_pairs(oracle_pairs, profile).to_dict(),
            "avg_actual_total_bpp": float(np.mean([row["actual_total_bpp"] for row in oracle_rows])),
            "avg_enhancement_payload_bpp": float(np.mean([row["enhancement_payload_bpp"] for row in oracle_rows])),
            "avg_psnr_delta_db": float(np.mean([row["psnr_delta_db"] for row in oracle_rows])),
            "chosen_policy_counts": dict(count_by_policy),
            "chosen_cardinality_counts": dict(count_by_cardinality),
            "improved_samples": int(sum(1 for row in oracle_rows if row["delta_distance"] < 0)),
            "worsened_samples": int(sum(1 for row in oracle_rows if row["delta_distance"] > 0)),
            "unchanged_samples": int(sum(1 for row in oracle_rows if row["delta_distance"] == 0)),
            "per_sample": oracle_rows,
        }

        singles = [row for row in profile_result["fixed_policies"] if row["selected_candidate_count"] == 1]
        pairs = [row for row in profile_result["fixed_policies"] if row["selected_candidate_count"] == 2]
        triples = [row for row in profile_result["fixed_policies"] if row["selected_candidate_count"] == 3]
        profile_result["best_by_cardinality"] = {
            "1": singles[0] if singles else None,
            "2": pairs[0] if pairs else None,
            "3": triples[0] if triples else None,
        }
        if len(singles) >= 3 and triples:
            additive_delta = int(sum(row["delta_char_errors"] for row in singles[:3]))
            triple_delta = int(triples[0]["delta_char_errors"])
            profile_result["triple_vs_additive_singles"] = {
                "additive_single_delta_sum": additive_delta,
                "observed_triple_delta": triple_delta,
                "interaction_residual": triple_delta - additive_delta,
                "observed_gain_over_additive_gain_ratio": None
                if additive_delta >= 0
                else float((-triple_delta) / (-additive_delta)),
            }
        results["profiles"][profile] = profile_result

    aggregate_metrics = {}
    for profile, profile_result in results["profiles"].items():
        prefix = profile
        aggregate_metric(f"{prefix}_base", profile_result["base"], aggregate_metrics)
        aggregate_metric(f"{prefix}_subset_oracle", profile_result["subset_oracle"], aggregate_metrics)
        if profile_result.get("triple_vs_additive_singles"):
            aggregate_metric(
                f"{prefix}_triple_vs_additive_singles",
                profile_result["triple_vs_additive_singles"],
                aggregate_metrics,
            )
        for card, row in profile_result["best_by_cardinality"].items():
            if row:
                aggregate_metric(f"{prefix}_best_cardinality_{card}", row, aggregate_metrics)
    results["aggregate"] = {"scalar_metrics": {key: {"value": value} for key, value in aggregate_metrics.items()}}

    write_json(args.output, results)
    write_report(args.report, results)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


def write_report(path: Path, results: dict) -> None:
    lines = [
        f"# {results['experiment_id']}",
        "",
        "This is a diagnostic all-subset oracle over counted raw-residual `.oscr` streams.",
        "It is not a deployable selector because it chooses subsets using OCR outcomes.",
        "",
    ]
    for profile, profile_result in results["profiles"].items():
        lines.append(f"## {profile}")
        lines.append("")
        lines.append("| policy | candidates | delta chars | total bpp | enh bpp | PSNR delta | improved/worse/unchanged |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in profile_result["fixed_policies"]:
            lines.append(
                f"| {row['name']} | {row['selected_candidate_count']} | {row['delta_char_errors']} | "
                f"{row['avg_actual_total_bpp']:.4f} | {row['avg_enhancement_payload_bpp']:.4f} | "
                f"{row['avg_psnr_delta_db']:.4f} | "
                f"{row['improved_samples']}/{row['worsened_samples']}/{row['unchanged_samples']} |"
            )
        oracle = profile_result["subset_oracle"]
        lines.append(
            f"| subset_oracle | mixed | {oracle['delta_char_errors']} | {oracle['avg_actual_total_bpp']:.4f} | "
            f"{oracle['avg_enhancement_payload_bpp']:.4f} | {oracle['avg_psnr_delta_db']:.4f} | "
            f"{oracle['improved_samples']}/{oracle['worsened_samples']}/{oracle['unchanged_samples']} |"
        )
        lines.append("")
        lines.append("Best fixed policy by cardinality:")
        lines.append("")
        lines.append("| cardinality | policy | delta chars | total bpp |")
        lines.append("|---:|---|---:|---:|")
        for card, row in profile_result["best_by_cardinality"].items():
            if row is None:
                continue
            lines.append(f"| {card} | {row['name']} | {row['delta_char_errors']} | {row['avg_actual_total_bpp']:.4f} |")
        lines.append("")
        if profile_result.get("triple_vs_additive_singles"):
            item = profile_result["triple_vs_additive_singles"]
            ratio = item["observed_gain_over_additive_gain_ratio"]
            ratio_s = "n/a" if ratio is None else f"{ratio:.4f}"
            lines.append(
                f"Triple vs additive singles: expected `{item['additive_single_delta_sum']}`, "
                f"observed `{item['observed_triple_delta']}`, residual `{item['interaction_residual']}`, "
                f"gain ratio `{ratio_s}`."
            )
            lines.append("")
        lines.append(f"Subset oracle policy counts: `{oracle['chosen_policy_counts']}`")
        lines.append(f"Subset oracle cardinality counts: `{oracle['chosen_cardinality_counts']}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Negative deltas mean fewer OCR edit errors than the frozen MLIC++ base. The all-subset oracle "
            "is an upper-control for a conditional allocator: it asks whether a policy should keep adding "
            "candidates or stop after one candidate.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
