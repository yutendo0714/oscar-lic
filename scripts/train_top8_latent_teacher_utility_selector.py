#!/usr/bin/env python3
"""Train a no-op-inclusive latent/code selector on CRNN/ABINet teacher-loss utility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_top8_latent_teacher_aux_ranker as base  # noqa: E402
import train_top8_latent_policy_utility_selector as policy  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_utility_values(arrays: dict[str, np.ndarray], loss_weight: float, string_weight: float) -> np.ndarray:
    utility = loss_weight * arrays["teacher_loss_utility"].astype(np.float32)
    utility += string_weight * arrays["teacher_string_utility"].astype(np.float32)
    utility[~arrays["mask"]] = -1.0e6
    utility[arrays["is_nearest"].astype(bool)] = 0.0
    return utility.astype(np.float32)


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "No-op-inclusive latent/code selector trained on CRNN/ABINet teacher-loss utility.",
        "Teacher signals are train targets, not selector inputs; no counted `.oscr` streams are exported.",
        "",
        "## Validation Policy",
        "",
        "| min gain | seed | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | T worse |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        m = row["val_metrics"]
        lines.append(
            f"| {row['min_gain']:.3f} | {row['model_seed']} | {m['changed_groups']} | "
            f"{m['exact_changed_groups']} | {m['false_change']} | {m['wrong_change']} | "
            f"{m['missed_oracle']} | {m['tesseract_delta_vs_nearest']} | {m['parseq_delta_vs_nearest']} | "
            f"{m['tesseract_worse_groups']} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--crnn", type=Path, required=True)
    parser.add_argument("--abinet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_latent_teacher_utility_selector")
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--loss-weight", type=float, default=1.0)
    parser.add_argument("--string-weight", type=float, default=0.0)
    parser.add_argument("--min-gain", type=float, action="append", default=[])
    parser.add_argument("--model-seed", type=int, action="append", default=[])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    min_gains = args.min_gain or [0.0, 0.05, 0.10]
    seeds = args.model_seed or [0, 1, 2]

    data = base.load_npz(args.features)
    table_rows = read_jsonl(args.table)
    crnn_rows = base.build_teacher_lookup(args.crnn)
    abinet_rows = base.build_teacher_lookup(args.abinet)
    arrays, group_records, _row_records = base.build_arrays(data, table_rows, crnn_rows, abinet_rows, args.profile)
    utilities = teacher_utility_values(arrays, args.loss_weight, args.string_weight)
    runs = []
    for min_gain in min_gains:
        args.min_gain = float(min_gain)
        train_targets = policy.policy_groups(arrays, "train", utilities, args.min_gain)
        for seed in seeds:
            scores, meta = policy.train_one(arrays, utilities, args, int(seed + round(min_gain * 1000)))
            val_audits, val_metrics = policy.evaluate_policy(arrays, group_records, scores, "val")
            train_audits, train_metrics = policy.evaluate_policy(arrays, group_records, scores, "train")
            runs.append(
                {
                    "min_gain": float(min_gain),
                    "model_seed": int(seed),
                    "train_targets_changed": int(
                        sum(
                            group["target_pos"]
                            != int(np.flatnonzero(arrays["is_nearest"][group["group_index"]])[0])
                            for group in train_targets
                        )
                    ),
                    "train_meta": meta,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "val_audits": val_audits,
                    "train_audits": train_audits,
                }
            )
    best = min(
        runs,
        key=lambda row: (
            row["val_metrics"]["tesseract_delta_vs_nearest"],
            row["val_metrics"]["false_change"] + row["val_metrics"]["wrong_change"],
            row["val_metrics"]["parseq_worse_groups"],
            -row["val_metrics"]["exact_changed_groups"],
        ),
    )
    interpretation = (
        "This is the teacher-separated counterpart to N104. It tests whether CRNN/ABINet loss utility can define "
        "a no-op-inclusive target that transfers to held-out Tesseract/PARSeq evaluation without using teacher "
        "signals as selector inputs."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_teacher_utility_policy_selector_not_promoted",
        "inputs": {
            "features": {"path": str(args.features), "sha256": sha256_file(args.features)},
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "crnn": {"path": str(args.crnn), "sha256": sha256_file(args.crnn)},
            "abinet": {"path": str(args.abinet), "sha256": sha256_file(args.abinet)},
        },
        "config": {
            "profile": args.profile,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pairwise_weight": args.pairwise_weight,
            "margin": args.margin,
            "loss_weight": args.loss_weight,
            "string_weight": args.string_weight,
            "min_gains": min_gains,
            "model_seeds": seeds,
        },
        "runs": runs,
        "best_run": best,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "base_teacher_aux_script": sha256_file(ROOT / "scripts/train_top8_latent_teacher_aux_ranker.py"),
            "policy_utility_script": sha256_file(ROOT / "scripts/train_top8_latent_policy_utility_selector.py"),
        },
        "aggregate": {
            "scalar_metrics": {
                "best_val_tesseract_delta": {"value": float(best["val_metrics"]["tesseract_delta_vs_nearest"])},
                "best_val_parseq_delta": {"value": float(best["val_metrics"]["parseq_delta_vs_nearest"])},
                "best_val_changed": {"value": float(best["val_metrics"]["changed_groups"])},
                "best_val_exact_changed": {"value": float(best["val_metrics"]["exact_changed_groups"])},
                "best_val_false_wrong": {
                    "value": float(best["val_metrics"]["false_change"] + best["val_metrics"]["wrong_change"])
                },
            }
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "best_run": best["val_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
