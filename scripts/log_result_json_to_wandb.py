#!/usr/bin/env python3
"""Log JSON experiment summaries and artifacts to an offline/online W&B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wandb


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(path: Path) -> str:
    return path.stem.replace(".", "_").replace("-", "_")


def safe_component(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def flatten_numeric(prefix: str, value: object, metrics: dict) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        metrics[prefix] = float(value)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}/{safe_component(key)}", child, metrics)


def flatten_aggregate(prefix: str, data: dict) -> dict:
    metrics = {}
    aggregate_root = data.get("aggregate")
    if isinstance(aggregate_root, dict):
        scalar_metrics = aggregate_root.get("scalar_metrics")
        if isinstance(scalar_metrics, dict):
            for metric, stats in scalar_metrics.items():
                if isinstance(stats, dict):
                    for stat_name, value in stats.items():
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            metrics[f"{prefix}/scalar/{metric}/{stat_name}"] = float(value)
        policy_metrics = aggregate_root.get("policy_metrics")
        if isinstance(policy_metrics, dict):
            for policy, policy_values in policy_metrics.items():
                if not isinstance(policy_values, dict):
                    continue
                for metric, stats in policy_values.items():
                    if isinstance(stats, dict):
                        for stat_name, value in stats.items():
                            if isinstance(value, (int, float)) and np.isfinite(value):
                                metrics[f"{prefix}/policy/{policy}/{metric}/{stat_name}"] = float(value)
                    elif isinstance(stats, (int, float)) and np.isfinite(stats):
                        metrics[f"{prefix}/policy/{policy}/{metric}"] = float(stats)
    for aggregate_name in ["aggregate_by_policy", "aggregate_train_by_policy", "aggregate_eval_by_policy"]:
        aggregate = data.get(aggregate_name)
        if isinstance(aggregate, dict):
            aggregate_prefix = prefix if aggregate_name == "aggregate_by_policy" else f"{prefix}/{aggregate_name}"
            for policy, policy_metrics in aggregate.items():
                for metric, stats in policy_metrics.items():
                    if isinstance(stats, dict):
                        for stat_name, value in stats.items():
                            if isinstance(value, (int, float)) and np.isfinite(value):
                                metrics[f"{aggregate_prefix}/{policy}/{metric}/{stat_name}"] = float(value)
                    elif isinstance(stats, (int, float)) and np.isfinite(stats):
                        metrics[f"{aggregate_prefix}/{policy}/{metric}"] = float(stats)
    per_seed = data.get("per_seed")
    if isinstance(per_seed, list) and per_seed:
        model_keys = sorted(
            {
                key
                for row in per_seed
                for key, value in row.get("model", {}).items()
                if isinstance(value, (int, float))
            }
        )
        for key in model_keys:
            values = np.asarray([float(row["model"][key]) for row in per_seed], dtype=np.float64)
            metrics[f"{prefix}/model/{key}/mean"] = float(values.mean())
            metrics[f"{prefix}/model/{key}/std"] = float(values.std(ddof=0))
    tables = data.get("tables")
    if isinstance(tables, list):
        for index, table in enumerate(tables):
            if not isinstance(table, dict):
                continue
            table_label = safe_component(table.get("label", index))
            flatten_numeric(f"{prefix}/table/{table_label}/overall", table.get("overall"), metrics)
            for group_name in ["by_source", "by_tesseract_base_distance", "by_parseq_base_distance", "by_image_area"]:
                group = table.get(group_name)
                if isinstance(group, dict):
                    flatten_numeric(f"{prefix}/table/{table_label}/{group_name}", group, metrics)
    results = data.get("results")
    if isinstance(results, list):
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                continue
            row_label = safe_component(
                "_".join(str(row.get(key, "")) for key in ["feature_mode", "model"]).strip("_") or index
            )
            for key in ["train_groups", "eval_groups", "feature_count", "train_safe_rate", "eval_safe_rate", "eval_auc", "eval_ap"]:
                value = row.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    metrics[f"{prefix}/result/{row_label}/{key}"] = float(value)
            flatten_numeric(f"{prefix}/result/{row_label}/eval_topk", row.get("eval_topk"), metrics)
    if not metrics and isinstance(aggregate_root, dict):
        flatten_numeric(f"{prefix}/aggregate", aggregate_root, metrics)
    if not metrics:
        for root_name in ["train_metrics", "val_metrics"]:
            root = data.get(root_name)
            if isinstance(root, dict):
                flatten_numeric(f"{prefix}/{root_name}", root, metrics)
    if not metrics:
        flatten_numeric(f"{prefix}/root", data, metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="oscar-lic")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--mode", default="offline", choices=["offline", "online", "disabled"])
    args = parser.parse_args()

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        mode=args.mode,
        tags=args.tag,
        notes=args.notes,
        config={"result_files": [str(path) for path in args.result]},
    )
    all_metrics = {}
    artifact = wandb.Artifact(args.run_name.replace("/", "_"), type="result-json")
    for path in args.result:
        data = load_json(path)
        all_metrics.update(flatten_aggregate(safe_name(path), data))
        artifact.add_file(str(path), name=f"{safe_name(path.parent)}_{path.name}")
    if all_metrics:
        wandb.log(all_metrics)
    run.log_artifact(artifact)
    run.finish()
    print(json.dumps({"run_id": run.id, "run_name": args.run_name, "metrics": len(all_metrics)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
