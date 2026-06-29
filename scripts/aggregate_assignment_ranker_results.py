#!/usr/bin/env python3
"""Aggregate assignment ranker JSON results across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate_runs(runs: list[dict]) -> dict:
    output: dict[str, dict] = {"scalar_metrics": {}, "policy_metrics": {}}
    for metric in ["train_auc", "train_ap", "val_auc", "val_ap"]:
        values = [float(run[metric]) for run in runs if run.get(metric) is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            output["scalar_metrics"][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(len(arr)),
            }
    policy_names = sorted({name for run in runs for name in run["policy_metrics"]})
    for policy in policy_names:
        output["policy_metrics"][policy] = {}
        metric_names = sorted({name for run in runs for name in run["policy_metrics"][policy]})
        for metric in metric_names:
            values = np.asarray([float(run["policy_metrics"][policy][metric]) for run in runs], dtype=np.float64)
            output["policy_metrics"][policy][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    inputs = []
    for path in args.result:
        data = json.loads(path.read_text(encoding="utf-8"))
        inputs.append(str(path))
        for run in data["per_run"]:
            run = dict(run)
            run["source_result"] = str(path)
            runs.append(run)
    output = {
        "description": "Aggregate assignment patch ranker results across result files.",
        "inputs": inputs,
        "runs": len(runs),
        "aggregate": aggregate_runs(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
