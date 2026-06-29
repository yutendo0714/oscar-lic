#!/usr/bin/env python3
"""Analyze whether StrHub train-teacher OCR signals transfer to held-out utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_key(row: dict) -> tuple[int, int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
        int(row["code_index"]),
    )


def group_key(key: tuple[int, int, int, int, int]) -> tuple[int, int, int, int]:
    return key[:4]


def distance(row: dict, profile: str) -> int:
    return character_counts(str(row["reference"]), str(row["prediction"]), profile).distance


def select_argmax(keys: list[tuple[int, int, int, int, int]], scores: dict, table: dict, nearest: tuple[int, int, int, int, int]):
    best = max(keys, key=lambda key: (float(scores[key]), int(table[key]["is_nearest"])))
    return best if float(scores[best]) > 0 else nearest


def split_name(group: tuple[int, int, int, int]) -> str:
    real_seed, seed, _, _ = group
    if real_seed == 600:
        return "train_eval600"
    if real_seed != 600 and seed == 1:
        return "val_eval300_seed1"
    return "other"


def init_metrics() -> dict:
    return {
        "groups": 0,
        "changed": 0,
        "false_change": 0,
        "exact_oracle": 0,
        "tesseract_delta": 0,
        "parseq_delta": 0,
        "crnn_delta": 0,
        "abinet_delta": 0,
        "tesseract_worse_groups": 0,
    }


def update_metrics(metrics: dict, selected, nearest, oracle, table: dict, teacher_deltas: dict) -> None:
    changed = selected != nearest
    oracle_changed = oracle != nearest
    tesseract_delta = int(table[selected]["tesseract_delta_vs_nearest"])
    metrics["groups"] += 1
    metrics["changed"] += int(changed)
    metrics["false_change"] += int(changed and not oracle_changed)
    metrics["exact_oracle"] += int(selected == oracle and oracle_changed)
    metrics["tesseract_delta"] += tesseract_delta
    metrics["parseq_delta"] += int(table[selected]["parseq_delta_vs_nearest"])
    metrics["crnn_delta"] += int(teacher_deltas["crnn"][selected])
    metrics["abinet_delta"] += int(teacher_deltas["abinet"][selected])
    metrics["tesseract_worse_groups"] += int(tesseract_delta > 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--crnn-results", type=Path, required=True)
    parser.add_argument("--abinet-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="unicode_strict_v1")
    args = parser.parse_args()

    table = {row_key(row): row for row in read_jsonl(args.table)}
    ocr_results = {
        "crnn": {row_key(row): row for row in read_jsonl(args.crnn_results)},
        "abinet": {row_key(row): row for row in read_jsonl(args.abinet_results)},
    }
    missing = {name: sorted(set(table) - set(rows))[:5] for name, rows in ocr_results.items() if set(table) - set(rows)}
    if missing:
        raise SystemExit(f"missing OCR rows: {missing}")

    groups: dict[tuple[int, int, int, int], list[tuple[int, int, int, int, int]]] = {}
    for key in table:
        groups.setdefault(group_key(key), []).append(key)
    for key in groups:
        groups[key].sort(key=lambda row_key_: int(table[row_key_]["topk_rank"]))

    policy_names = [
        "nearest",
        "heldout_tesseract_oracle",
        "crnn_confidence",
        "abinet_confidence",
        "mean_confidence",
        "crnn_distance_confidence",
        "abinet_distance_confidence",
        "mean_distance_confidence",
        "trainteacher_no_worse_heldout_oracle",
    ]
    metrics = {
        split: {policy: init_metrics() for policy in policy_names}
        for split in ["train_eval600", "val_eval300_seed1", "all"]
    }
    summary_counts = {
        split: {
            "groups": 0,
            "crnn_improve_groups": 0,
            "abinet_improve_groups": 0,
            "either_train_teacher_improve_groups": 0,
            "tesseract_oracle_change_groups": 0,
        }
        for split in ["train_eval600", "val_eval300_seed1", "all"]
    }

    for group, keys in groups.items():
        nearest_candidates = [key for key in keys if int(table[key]["is_nearest"]) == 1]
        if not nearest_candidates:
            raise SystemExit(f"group has no nearest row: {group}")
        nearest = nearest_candidates[0]
        nearest_stats = {
            name: {
                "distance": distance(ocr_results[name][nearest], args.profile),
                "confidence": float(ocr_results[name][nearest]["confidence"]),
            }
            for name in ["crnn", "abinet"]
        }
        teacher_deltas = {name: {} for name in ["crnn", "abinet"]}
        teacher_conf_deltas = {name: {} for name in ["crnn", "abinet"]}
        for key in keys:
            for name in ["crnn", "abinet"]:
                teacher_deltas[name][key] = distance(ocr_results[name][key], args.profile) - nearest_stats[name]["distance"]
                teacher_conf_deltas[name][key] = float(ocr_results[name][key]["confidence"]) - nearest_stats[name]["confidence"]

        oracle = min(keys, key=lambda key: (int(table[key]["tesseract_delta_vs_nearest"]), -int(table[key]["is_nearest"])))
        score_maps = {
            "crnn_confidence": {key: teacher_conf_deltas["crnn"][key] for key in keys},
            "abinet_confidence": {key: teacher_conf_deltas["abinet"][key] for key in keys},
            "mean_confidence": {
                key: 0.5 * (teacher_conf_deltas["crnn"][key] + teacher_conf_deltas["abinet"][key]) for key in keys
            },
            "crnn_distance_confidence": {
                key: -teacher_deltas["crnn"][key] + 0.25 * teacher_conf_deltas["crnn"][key] for key in keys
            },
            "abinet_distance_confidence": {
                key: -teacher_deltas["abinet"][key] + 0.25 * teacher_conf_deltas["abinet"][key] for key in keys
            },
            "mean_distance_confidence": {
                key: -(teacher_deltas["crnn"][key] + teacher_deltas["abinet"][key])
                + 0.25 * (teacher_conf_deltas["crnn"][key] + teacher_conf_deltas["abinet"][key])
                for key in keys
            },
        }
        no_worse_keys = [
            key
            for key in keys
            if teacher_deltas["crnn"][key] <= 0 and teacher_deltas["abinet"][key] <= 0
        ]
        selected = {
            "nearest": nearest,
            "heldout_tesseract_oracle": oracle,
            "trainteacher_no_worse_heldout_oracle": min(
                no_worse_keys or [nearest],
                key=lambda key: (int(table[key]["tesseract_delta_vs_nearest"]), -int(table[key]["is_nearest"])),
            ),
        }
        for policy in [
            "crnn_confidence",
            "abinet_confidence",
            "mean_confidence",
            "crnn_distance_confidence",
            "abinet_distance_confidence",
            "mean_distance_confidence",
        ]:
            selected[policy] = select_argmax(keys, score_maps[policy], table, nearest)

        split = split_name(group)
        for bucket in [split, "all"]:
            if bucket == "other":
                continue
            summary_counts[bucket]["groups"] += 1
            summary_counts[bucket]["crnn_improve_groups"] += int(
                any(teacher_deltas["crnn"][key] < 0 and key != nearest for key in keys)
            )
            summary_counts[bucket]["abinet_improve_groups"] += int(
                any(teacher_deltas["abinet"][key] < 0 and key != nearest for key in keys)
            )
            summary_counts[bucket]["either_train_teacher_improve_groups"] += int(
                any(
                    (teacher_deltas["crnn"][key] < 0 or teacher_deltas["abinet"][key] < 0) and key != nearest
                    for key in keys
                )
            )
            summary_counts[bucket]["tesseract_oracle_change_groups"] += int(oracle != nearest)
            for policy, key in selected.items():
                update_metrics(metrics[bucket][policy], key, nearest, oracle, table, teacher_deltas)

    result = {
        "description": "Train-teacher CRNN/ABINet transfer diagnostic. Policies using held-out Tesseract are upper bounds only and must not be used for training or selection.",
        "table": str(args.table),
        "crnn_results": str(args.crnn_results),
        "abinet_results": str(args.abinet_results),
        "profile": args.profile,
        "summary_counts": summary_counts,
        "policies": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary_counts": summary_counts, "val_policies": metrics["val_eval300_seed1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
