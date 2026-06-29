#!/usr/bin/env python3
"""Train-tune threshold policies for the Eval300 top-8 tabular ranker."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_top8_trainval_tabular_listwise_ranker as ranker  # noqa: E402


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_best_nonnearest(rows: list[dict[str, Any]], indices: list[int], scores: np.ndarray) -> int | None:
    nonnearest = ranker.nonnearest_indices(rows, indices)
    if not nonnearest:
        return None
    return max(nonnearest, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))


def evaluate_threshold(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, str, int, int], list[int]],
    scores: np.ndarray,
    *,
    partition: str,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = Counter()
    tess_delta = 0
    parseq_delta = 0
    audits = []
    for key, indices in sorted(groups.items()):
        if key[1] != partition:
            continue
        nearest = next(idx for idx in indices if int(rows[idx]["code_index"]) == int(rows[idx]["nearest_code"]))
        oracle = next((idx for idx in indices if int(rows[idx].get("label_assignment_oracle_choice", 0))), nearest)
        best = group_best_nonnearest(rows, indices, scores)
        best_score = float(scores[best]) if best is not None else float("-inf")
        selected = best if best is not None and best_score >= threshold else nearest
        nearest_code = int(rows[nearest]["code_index"])
        oracle_code = int(rows[oracle]["code_index"])
        selected_code = int(rows[selected]["code_index"])
        status = ranker.classify(selected_code, nearest_code, oracle_code)
        counts[status] += 1
        changed = selected_code != nearest_code
        counts["changed_groups"] += int(changed)
        counts["oracle_change_groups"] += int(oracle_code != nearest_code)
        counts["exact_changed_groups"] += int(status == "exact" and oracle_code != nearest_code)
        counts["parseq_worsen_groups"] += int(rows[selected]["parseq_delta_vs_nearest"] > 0)
        counts["tesseract_worsen_groups"] += int(rows[selected]["tesseract_delta_vs_nearest"] > 0)
        counts["tesseract_improve_groups"] += int(rows[selected]["tesseract_delta_vs_nearest"] < 0)
        tess_delta += int(rows[selected]["tesseract_delta_vs_nearest"])
        parseq_delta += int(rows[selected]["parseq_delta_vs_nearest"])
        audits.append(
            {
                "key": {"real_seed": key[0], "partition": key[1], "source_index": key[2], "candidate_index": key[3]},
                "source": rows[nearest].get("source"),
                "reference": rows[nearest].get("reference"),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "best_nonnearest_code": int(rows[best]["code_index"]) if best is not None else nearest_code,
                "best_nonnearest_score": best_score,
                "selected_code": selected_code,
                "selected_status": status,
                "selected_tesseract_delta_vs_nearest": int(rows[selected]["tesseract_delta_vs_nearest"]),
                "selected_parseq_delta_vs_nearest": int(rows[selected]["parseq_delta_vs_nearest"]),
            }
        )
    metrics = {
        "partition": partition,
        "threshold": float(threshold),
        "groups": int(sum(1 for key in groups if key[1] == partition)),
        "oracle_change_groups": int(counts["oracle_change_groups"]),
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "exact_changed_groups": int(counts["exact_changed_groups"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "parseq_worsen_groups": int(counts["parseq_worsen_groups"]),
        "tesseract_worsen_groups": int(counts["tesseract_worsen_groups"]),
        "tesseract_improve_groups": int(counts["tesseract_improve_groups"]),
        "tesseract_delta_vs_nearest": int(tess_delta),
        "parseq_delta_vs_nearest": int(parseq_delta),
    }
    return metrics, audits


def threshold_grid(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, str, int, int], list[int]],
    scores: np.ndarray,
    partition: str,
) -> list[float]:
    values = []
    for key, indices in groups.items():
        if key[1] != partition:
            continue
        best = group_best_nonnearest(rows, indices, scores)
        if best is not None:
            values.append(float(scores[best]))
    unique = sorted(set(values))
    if not unique:
        return [1.0]
    return [max(unique) + 1.0e-6] + unique + [min(unique) - 1.0e-6]


def choose_policies(train_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    policies = {}

    zero_safe = [
        row
        for row in train_metrics
        if row["false_change"] + row["wrong_change"] == 0
        and row["parseq_worsen_groups"] == 0
        and row["tesseract_worsen_groups"] == 0
    ]
    if zero_safe:
        policies["zero_error_safe"] = min(
            zero_safe,
            key=lambda row: (row["tesseract_delta_vs_nearest"], -row["exact_changed_groups"], row["changed_groups"]),
        )

    one_error = [
        row
        for row in train_metrics
        if row["false_change"] + row["wrong_change"] <= 1 and row["parseq_worsen_groups"] == 0
    ]
    if one_error:
        policies["one_error_parseq_safe"] = min(
            one_error,
            key=lambda row: (
                row["tesseract_delta_vs_nearest"],
                row["tesseract_worsen_groups"],
                row["false_change"] + row["wrong_change"],
                -row["exact_changed_groups"],
            ),
        )

    parseq_safe = [row for row in train_metrics if row["parseq_worsen_groups"] == 0]
    if parseq_safe:
        policies["best_train_tesseract_parseq_safe"] = min(
            parseq_safe,
            key=lambda row: (
                row["tesseract_delta_vs_nearest"],
                row["tesseract_worsen_groups"],
                row["false_change"] + row["wrong_change"],
            ),
        )

    policies["lowest_threshold"] = min(train_metrics, key=lambda row: row["threshold"])
    policies["highest_threshold"] = max(train_metrics, key=lambda row: row["threshold"])
    return policies


def train_scores(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict, np.ndarray, dict[str, Any]]:
    rows = ranker.read_jsonl(args.table)
    groups = ranker.group_rows(rows)
    feature_names = ranker.numeric_feature_names(rows)
    features = ranker.build_features(rows, feature_names)
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if str(row.get("assignment_partition", row.get("split"))) == "train"],
        dtype=np.int64,
    )
    train_nonnearest = np.asarray(
        [idx for idx in train_indices if int(rows[idx]["code_index"]) != int(rows[idx]["nearest_code"])],
        dtype=np.int64,
    )
    train_groups = ranker.build_listwise_groups(rows, groups, "train")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_norm = ranker.normalize(features[train_nonnearest], features)
    x_all = torch.from_numpy(x_norm).to(device)
    all_scores = []
    model_metas = []
    for offset in range(args.model_seeds):
        scores, meta = ranker.train_model(x_all, train_groups, args.seed + offset, args, device)
        all_scores.append(scores)
        model_metas.append(meta)
    scores = np.mean(np.stack(all_scores, axis=0), axis=0)
    meta = {
        "device": str(device),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "train_listwise_groups": int(len(train_groups)),
        "models": model_metas,
    }
    return rows, groups, scores, meta


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Train-tuned threshold policies for the Eval300 top-8 tabular ranker.",
        "This is a deployable-style policy diagnostic, not a counted `.oscr` promotion.",
        "",
        "## Policy Results",
        "",
        "| policy | train changed | train exact changed | train false+wrong | train T | val changed | val exact changed | val false+wrong | val T | val P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["policies"].items():
        train = row["train"]
        val = row["val"]
        lines.append(
            f"| {name} | {train['changed_groups']} | {train['exact_changed_groups']} | "
            f"{train['false_change'] + train['wrong_change']} | {train['tesseract_delta_vs_nearest']} | "
            f"{val['changed_groups']} | {val['exact_changed_groups']} | "
            f"{val['false_change'] + val['wrong_change']} | {val['tesseract_delta_vs_nearest']} | "
            f"{val['parseq_delta_vs_nearest']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A deployable policy must beat the current actual-bitstream `-8` Tesseract floor with low false/wrong changes before any `.oscr` promotion attempt.",
            "- This diagnostic reuses N091's ranker family and tunes thresholds only on train-partition groups.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_trainval_tabular_threshold_policy")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--model-seeds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows, groups, scores, meta = train_scores(args)
    train_rows = []
    for threshold in threshold_grid(rows, groups, scores, "train"):
        metrics, _ = evaluate_threshold(rows, groups, scores, partition="train", threshold=threshold)
        train_rows.append(metrics)
    selected = choose_policies(train_rows)
    policies = {}
    for name, train_metrics in selected.items():
        val_metrics, val_audits = evaluate_threshold(rows, groups, scores, partition="val", threshold=float(train_metrics["threshold"]))
        policies[name] = {
            "threshold": float(train_metrics["threshold"]),
            "train": train_metrics,
            "val": val_metrics,
            "val_audits": val_audits,
        }
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_train_tuned_threshold_policy_not_promoted_stream",
        "inputs": {
            "table": str(args.table),
            "table_sha256": sha256_file(args.table),
        },
        "config": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pairwise_weight": args.pairwise_weight,
            "model_seeds": args.model_seeds,
            "seed": args.seed,
            "feature_count": meta["feature_count"],
            "feature_rule": "topk_rank, assignment_relative_error, codebook_* and img_* numeric columns; OCR/label/oracle columns excluded",
        },
        "training": {
            "device": meta["device"],
            "train_listwise_groups": meta["train_listwise_groups"],
            "models": meta["models"],
        },
        "policies": policies,
        "aggregate": {
            "scalar_metrics": {
                f"{name}_val_tesseract_delta": {"value": float(policy["val"]["tesseract_delta_vs_nearest"])}
                for name, policy in policies.items()
            }
        },
    }
    for name, policy in policies.items():
        result["aggregate"]["scalar_metrics"][f"{name}_val_changed_groups"] = {
            "value": float(policy["val"]["changed_groups"])
        }
        result["aggregate"]["scalar_metrics"][f"{name}_val_false_wrong"] = {
            "value": float(policy["val"]["false_change"] + policy["val"]["wrong_change"])
        }
        result["aggregate"]["scalar_metrics"][f"{name}_val_exact_changed"] = {
            "value": float(policy["val"]["exact_changed_groups"])
        }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
