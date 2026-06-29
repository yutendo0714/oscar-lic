#!/usr/bin/env python3
"""Analyze whether StrHub per-sample teacher losses transfer to held-out utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts  # noqa: E402


Key = tuple[int, int, int, int, int]
GroupKey = tuple[int, int, int, int]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_key(row: dict) -> Key:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
        int(row["code_index"]),
    )


def group_key(key: Key) -> GroupKey:
    return key[:4]


def split_name(group: GroupKey) -> str:
    real_seed, seed, _, _ = group
    if real_seed == 600:
        return "train_eval600"
    if real_seed != 600 and seed == 1:
        return "val_eval300_seed1"
    return "other"


def distance(row: dict, profile: str) -> int:
    return character_counts(str(row["reference"]), str(row["prediction"]), profile).distance


def init_metrics() -> dict:
    return {
        "groups": 0,
        "changed": 0,
        "false_change": 0,
        "exact_oracle": 0,
        "tesseract_delta": 0,
        "parseq_delta": 0,
        "crnn_edit_delta": 0,
        "abinet_edit_delta": 0,
        "crnn_loss_mean_delta": 0.0,
        "abinet_loss_mean_delta": 0.0,
        "tesseract_worse_groups": 0,
    }


def update_metrics(metrics: dict, selected: Key, nearest: Key, oracle: Key, table: dict, teacher: dict) -> None:
    changed = selected != nearest
    oracle_changed = oracle != nearest
    tesseract_delta = int(table[selected]["tesseract_delta_vs_nearest"])
    metrics["groups"] += 1
    metrics["changed"] += int(changed)
    metrics["false_change"] += int(changed and not oracle_changed)
    metrics["exact_oracle"] += int(selected == oracle and oracle_changed)
    metrics["tesseract_delta"] += tesseract_delta
    metrics["parseq_delta"] += int(table[selected]["parseq_delta_vs_nearest"])
    metrics["crnn_edit_delta"] += int(teacher["edit_delta"]["crnn"][selected])
    metrics["abinet_edit_delta"] += int(teacher["edit_delta"]["abinet"][selected])
    metrics["crnn_loss_mean_delta"] += float(teacher["loss_delta_mean"]["crnn"][selected])
    metrics["abinet_loss_mean_delta"] += float(teacher["loss_delta_mean"]["abinet"][selected])
    metrics["tesseract_worse_groups"] += int(tesseract_delta > 0)


def select_argmax(keys: list[Key], scores: dict[Key, float], table: dict, nearest: Key) -> Key:
    best = max(keys, key=lambda key: (float(scores[key]), int(table[key]["is_nearest"])))
    return best if float(scores[best]) > 0 else nearest


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

    groups: dict[GroupKey, list[Key]] = {}
    for key in table:
        groups.setdefault(group_key(key), []).append(key)
    for key in groups:
        groups[key].sort(key=lambda row_key_: int(table[row_key_]["topk_rank"]))

    policy_names = [
        "nearest",
        "heldout_tesseract_oracle",
        "crnn_loss_mean",
        "abinet_loss_mean",
        "mean_loss_mean",
        "crnn_loss_sum",
        "abinet_loss_sum",
        "mean_loss_sum",
        "trainteacher_loss_no_worse_heldout_oracle",
    ]
    metrics = {
        split: {policy: init_metrics() for policy in policy_names}
        for split in ["train_eval600", "val_eval300_seed1", "all"]
    }
    summary_counts = {
        split: {
            "groups": 0,
            "crnn_loss_improve_groups": 0,
            "abinet_loss_improve_groups": 0,
            "either_train_teacher_loss_improve_groups": 0,
            "both_train_teacher_loss_no_worse_groups": 0,
            "tesseract_oracle_change_groups": 0,
        }
        for split in ["train_eval600", "val_eval300_seed1", "all"]
    }

    for group, keys in groups.items():
        nearest_candidates = [key for key in keys if int(table[key]["is_nearest"]) == 1]
        if not nearest_candidates:
            raise SystemExit(f"group has no nearest row: {group}")
        nearest = nearest_candidates[0]
        teacher = {
            "edit_delta": {"crnn": {}, "abinet": {}},
            "loss_delta_mean": {"crnn": {}, "abinet": {}},
            "loss_delta_sum": {"crnn": {}, "abinet": {}},
        }
        nearest_stats = {
            name: {
                "distance": distance(ocr_results[name][nearest], args.profile),
                "loss_mean": float(ocr_results[name][nearest]["teacher_loss_mean"]),
                "loss_sum": float(ocr_results[name][nearest]["teacher_loss_sum"]),
            }
            for name in ["crnn", "abinet"]
        }
        for key in keys:
            for name in ["crnn", "abinet"]:
                row = ocr_results[name][key]
                teacher["edit_delta"][name][key] = distance(row, args.profile) - nearest_stats[name]["distance"]
                teacher["loss_delta_mean"][name][key] = float(row["teacher_loss_mean"]) - nearest_stats[name]["loss_mean"]
                teacher["loss_delta_sum"][name][key] = float(row["teacher_loss_sum"]) - nearest_stats[name]["loss_sum"]

        oracle = min(keys, key=lambda key: (int(table[key]["tesseract_delta_vs_nearest"]), -int(table[key]["is_nearest"])))
        score_maps = {
            "crnn_loss_mean": {key: -teacher["loss_delta_mean"]["crnn"][key] for key in keys},
            "abinet_loss_mean": {key: -teacher["loss_delta_mean"]["abinet"][key] for key in keys},
            "mean_loss_mean": {
                key: -0.5 * (teacher["loss_delta_mean"]["crnn"][key] + teacher["loss_delta_mean"]["abinet"][key])
                for key in keys
            },
            "crnn_loss_sum": {key: -teacher["loss_delta_sum"]["crnn"][key] for key in keys},
            "abinet_loss_sum": {key: -teacher["loss_delta_sum"]["abinet"][key] for key in keys},
            "mean_loss_sum": {
                key: -0.5 * (teacher["loss_delta_sum"]["crnn"][key] + teacher["loss_delta_sum"]["abinet"][key])
                for key in keys
            },
        }
        loss_no_worse_keys = [
            key
            for key in keys
            if teacher["loss_delta_mean"]["crnn"][key] <= 0.0 and teacher["loss_delta_mean"]["abinet"][key] <= 0.0
        ]
        selected = {
            "nearest": nearest,
            "heldout_tesseract_oracle": oracle,
            "trainteacher_loss_no_worse_heldout_oracle": min(
                loss_no_worse_keys or [nearest],
                key=lambda key: (int(table[key]["tesseract_delta_vs_nearest"]), -int(table[key]["is_nearest"])),
            ),
        }
        for policy in [
            "crnn_loss_mean",
            "abinet_loss_mean",
            "mean_loss_mean",
            "crnn_loss_sum",
            "abinet_loss_sum",
            "mean_loss_sum",
        ]:
            selected[policy] = select_argmax(keys, score_maps[policy], table, nearest)

        split = split_name(group)
        for bucket in [split, "all"]:
            if bucket == "other":
                continue
            summary_counts[bucket]["groups"] += 1
            summary_counts[bucket]["crnn_loss_improve_groups"] += int(
                any(teacher["loss_delta_mean"]["crnn"][key] < 0 and key != nearest for key in keys)
            )
            summary_counts[bucket]["abinet_loss_improve_groups"] += int(
                any(teacher["loss_delta_mean"]["abinet"][key] < 0 and key != nearest for key in keys)
            )
            summary_counts[bucket]["either_train_teacher_loss_improve_groups"] += int(
                any(
                    (teacher["loss_delta_mean"]["crnn"][key] < 0 or teacher["loss_delta_mean"]["abinet"][key] < 0)
                    and key != nearest
                    for key in keys
                )
            )
            summary_counts[bucket]["both_train_teacher_loss_no_worse_groups"] += int(
                any(
                    teacher["loss_delta_mean"]["crnn"][key] <= 0 and teacher["loss_delta_mean"]["abinet"][key] <= 0
                    and key != nearest
                    for key in keys
                )
            )
            summary_counts[bucket]["tesseract_oracle_change_groups"] += int(oracle != nearest)
            for policy, key in selected.items():
                update_metrics(metrics[bucket][policy], key, nearest, oracle, table, teacher)

    result = {
        "description": "Train-teacher CRNN/ABINet per-sample loss transfer diagnostic. Policies using held-out Tesseract are upper bounds only and must not be used for training or selection.",
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
