#!/usr/bin/env python3
"""Train top-8 selectors from binned CRNN/ABINet sequence-profile features."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_top8_strhub_logit_feature_selector import (  # noqa: E402
    best_nonnearest,
    classify,
    evaluate,
    load_feature_cache,
    predict_scores,
    read_jsonl,
    sha256_file,
    train_models,
    tune_threshold,
    write_json,
)


def row_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
        int(row["code_index"]),
    )


def build_dataset(
    table_rows: list[dict[str, Any]],
    crnn_lookup: dict[tuple[int, int, int, int, int], np.ndarray],
    abinet_lookup: dict[tuple[int, int, int, int, int], np.ndarray],
    crnn_dim: int,
    abinet_dim: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int, int], list[int]], np.ndarray, list[str]]:
    rows = []
    grouped: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    feature_rows = []
    feature_names = [
        "topk_rank",
        "log1p_assignment_relative_error",
        "is_nearest",
        "code_equals_nearest",
        "code_index_norm",
    ]
    for prefix, dim in (("crnn", crnn_dim), ("abinet", abinet_dim)):
        feature_names.extend([f"{prefix}_seq_raw_{idx:03d}" for idx in range(dim)])
        feature_names.extend([f"{prefix}_seq_delta_{idx:03d}" for idx in range(dim)])
        feature_names.extend([f"{prefix}_seq_abs_delta_{idx:03d}" for idx in range(dim)])

    nearest_feature_by_group: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for row in table_rows:
        if int(row.get("code_equals_nearest", int(row["code_index"]) == int(row["nearest_code"]))):
            key = row_key(row)
            nearest_feature_by_group[key[:4]] = (crnn_lookup[key], abinet_lookup[key])

    for row in table_rows:
        key = row_key(row)
        gkey = key[:4]
        if key not in crnn_lookup or key not in abinet_lookup or gkey not in nearest_feature_by_group:
            raise RuntimeError(f"missing sequence feature for row {key}")
        crnn = crnn_lookup[key]
        abinet = abinet_lookup[key]
        crnn_nearest, abinet_nearest = nearest_feature_by_group[gkey]
        assignment_error = min(float(row.get("assignment_relative_error", 0.0)), 1.0e6)
        base = np.asarray(
            [
                float(row.get("topk_rank", 0)),
                float(np.log1p(max(assignment_error, 0.0))),
                float(row.get("is_nearest", int(row["code_index"]) == int(row["nearest_code"]))),
                float(row.get("code_equals_nearest", int(row["code_index"]) == int(row["nearest_code"]))),
                float(row["code_index"]) / max(float(row.get("codebook_size", 64)) - 1.0, 1.0),
            ],
            dtype=np.float32,
        )
        features = np.concatenate(
            [
                base,
                crnn,
                crnn - crnn_nearest,
                np.abs(crnn - crnn_nearest),
                abinet,
                abinet - abinet_nearest,
                np.abs(abinet - abinet_nearest),
            ]
        ).astype(np.float32)
        meta = {
            "group_key": gkey,
            "row_key": key,
            "partition": str(row.get("assignment_partition", row.get("split", ""))),
            "source": str(row.get("source", "unknown")),
            "code_index": int(row["code_index"]),
            "nearest_code": int(row["nearest_code"]),
            "oracle_code": int(row["assignment_oracle_code_index"]),
            "is_nearest": bool(int(row.get("is_nearest", int(row["code_index"]) == int(row["nearest_code"])))),
            "topk_rank": int(row.get("topk_rank", 0)),
            "tesseract_delta": int(row.get("tesseract_delta_vs_nearest", 0)),
            "parseq_delta": int(row.get("parseq_delta_vs_nearest", 0)),
            "label_exact": int(row.get("label_assignment_oracle_choice", 0)),
            "label_safe": int(row.get("label_tesseract_parseq_safe_improves", 0)),
        }
        grouped[gkey].append(len(rows))
        rows.append(meta)
        feature_rows.append(features)
    return rows, grouped, np.vstack(feature_rows).astype(np.float32), feature_names


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "CRNN/ABINet binned sequence-profile feature selector over Eval300 top-8 assignment candidates.",
        "This is an OCR-aware diagnostic and does not export counted `.oscr` streams.",
        "",
        "## Validation Policies",
        "",
        "| target | model | budget | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | rank1 | rank<=4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        for policy in run["policies"]:
            m = policy["val_metrics"]
            lines.append(
                f"| {run['target']} | {run['model_name']} | {policy['max_false_wrong']} | "
                f"{m['changed_groups']} | {m['exact_changed_groups']} | {m['false_change']} | "
                f"{m['wrong_change']} | {m['missed_oracle']} | {m['tesseract_delta_vs_nearest']} | "
                f"{m['parseq_delta_vs_nearest']} | {m['oracle_rank_le1']} | {m['oracle_rank_le4']} |"
            )
    lines.extend(["", "## Best Policy", "", "```json", json.dumps(result["best_policy"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--crnn-features", type=Path, required=True)
    parser.add_argument("--abinet-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--max-false-wrong", type=int, action="append")
    args = parser.parse_args()
    if args.max_false_wrong is None:
        args.max_false_wrong = [0, 1, 2]

    table_rows = read_jsonl(args.table)
    crnn_lookup, crnn_names, crnn_meta = load_feature_cache(args.crnn_features)
    abinet_lookup, abinet_names, abinet_meta = load_feature_cache(args.abinet_features)
    rows, grouped, x, feature_names = build_dataset(
        table_rows,
        crnn_lookup,
        abinet_lookup,
        len(crnn_names),
        len(abinet_names),
    )
    partitions = np.asarray([row["partition"] for row in rows])
    nonnearest = np.asarray([not row["is_nearest"] for row in rows])
    train_mask = (partitions == "train") & nonnearest
    val_mask = (partitions == "val") & nonnearest

    runs = []
    for target_name, target_field in (("exact_oracle", "label_exact"), ("safe_improve", "label_safe")):
        y = np.asarray([row[target_field] for row in rows], dtype=np.int64)
        models = train_models(x[train_mask], y[train_mask], args.model_seed)
        for model_name, model in models.items():
            scores = predict_scores(model, x)
            oracle_val = evaluate(rows, grouped, scores, "val", None, True)
            policies = []
            for budget in args.max_false_wrong:
                threshold, train_metrics = tune_threshold(rows, grouped, scores, int(budget))
                val_metrics = evaluate(rows, grouped, scores, "val", threshold, False)
                policies.append(
                    {
                        "max_false_wrong": int(budget),
                        "threshold": float(threshold),
                        "train_metrics": train_metrics,
                        "val_metrics": {k: v for k, v in val_metrics.items() if k != "audits"},
                        "val_audits": val_metrics["audits"],
                    }
                )
            runs.append(
                {
                    "target": target_name,
                    "model_name": model_name,
                    "train_positive_rows": int(y[train_mask].sum()),
                    "val_positive_rows": int(y[val_mask].sum()),
                    "oracle_change_only_val_metrics": {k: v for k, v in oracle_val.items() if k != "audits"},
                    "policies": policies,
                }
            )

    candidates = [(run, policy) for run in runs for policy in run["policies"]]
    best_run, best_policy = sorted(
        candidates,
        key=lambda item: (
            item[1]["val_metrics"]["tesseract_delta_vs_nearest"],
            item[1]["val_metrics"]["false_change"] + item[1]["val_metrics"]["wrong_change"],
            item[1]["val_metrics"]["tesseract_worse_groups"],
            -item[1]["val_metrics"]["exact_changed_groups"],
        ),
    )[0]
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_strhub_sequence_feature_selector_not_promoted",
        "inputs": {
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "crnn_features": crnn_meta,
            "abinet_features": abinet_meta,
        },
        "config": {
            "model_seed": args.model_seed,
            "max_false_wrong": args.max_false_wrong,
            "feature_dim": int(x.shape[1]),
            "feature_blocks": ["base_rank_error", "crnn_sequence_raw_delta_absdelta", "abinet_sequence_raw_delta_absdelta"],
        },
        "data_summary": {
            "rows": int(len(rows)),
            "groups": int(len(grouped)),
            "train_rows": int(train_mask.sum()),
            "val_rows": int(val_mask.sum()),
        },
        "runs": runs,
        "best_policy": {
            "target": best_run["target"],
            "model_name": best_run["model_name"],
            **best_policy,
        },
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "crnn_feature_names": crnn_names,
            "abinet_feature_names": abinet_names,
        },
        "interpretation": (
            "This diagnostic tests whether time-positioned CRNN/ABINet confidence profiles "
            "carry more useful OCR-aware evidence than global logit summaries. It is not "
            "promoted unless a train-tuned table policy exceeds the current actual-bitstream "
            "-8 Tesseract floor with low false/wrong changes."
        ),
    }
    write_json(args.output, result)
    write_report(args.report, result)


if __name__ == "__main__":
    main()
