#!/usr/bin/env python3
"""Source leave-out diagnostic for the latent teacher-auxiliary ranker."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_top8_latent_teacher_aux_ranker as teacher_aux  # noqa: E402


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


def group_sources(table_rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], str]:
    out = {}
    for row in table_rows:
        out[teacher_aux.table_group_key(row)] = str(row.get("source", "unknown"))
    return out


def source_array(group_records: list[dict[str, Any]], source_by_key: dict[tuple[int, int, int, int], str]) -> np.ndarray:
    values = []
    for record in group_records:
        key = record["key"]
        tuple_key = (int(key["real_seed"]), int(key["seed"]), int(key["source_index"]), int(key["candidate_index"]))
        values.append(source_by_key.get(tuple_key, "unknown"))
    return np.asarray(values, dtype=object)


def sum_metrics(metrics: dict[str, Any]) -> tuple[int, int, int, int, int]:
    ranks = metrics["oracle_nonnearest_score_rank_counts"]
    return (
        int(metrics["tesseract_delta_vs_nearest"]),
        int(metrics["exact_changed_groups"]),
        int(metrics["wrong_change"]),
        int(ranks["le1"]),
        int(ranks["le4"]),
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for row in rows:
        metrics = row["metrics"]
        ranks = metrics["oracle_nonnearest_score_rank_counts"]
        totals["groups"] += int(metrics["groups"])
        totals["oracle_change_groups"] += int(metrics["oracle_change_groups"])
        totals["changed_groups"] += int(metrics["changed_groups"])
        totals["exact_changed_groups"] += int(metrics["exact_changed_groups"])
        totals["wrong_change"] += int(metrics["wrong_change"])
        totals["tesseract_delta_vs_nearest"] += int(metrics["tesseract_delta_vs_nearest"])
        totals["parseq_delta_vs_nearest"] += int(metrics["parseq_delta_vs_nearest"])
        totals["rank_le1"] += int(ranks["le1"])
        totals["rank_le4"] += int(ranks["le4"])
    return dict(totals)


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Source leave-out diagnostic for the latent/code teacher-auxiliary ranker.",
        "This is not a deployable selector and does not export counted `.oscr` streams.",
        "",
        "## Aggregate",
        "",
        "| teacher weight | exact changed | wrong | Tesseract | PARSeq | rank<=1 | rank<=4 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, metrics in result["aggregate_by_teacher_weight"].items():
        lines.append(
            f"| {float(key):.3f} | {metrics['exact_changed_groups']} | {metrics['wrong_change']} | "
            f"{metrics['tesseract_delta_vs_nearest']} | {metrics['parseq_delta_vs_nearest']} | "
            f"{metrics['rank_le1']} | {metrics['rank_le4']} |"
        )
    lines.extend(
        [
            "",
            "## By Held-Out Source",
            "",
            "| source | teacher weight | oracle changes | exact changed | wrong | Tesseract | rank<=1 | rank<=4 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["folds"]:
        m = row["metrics"]
        ranks = m["oracle_nonnearest_score_rank_counts"]
        lines.append(
            f"| `{row['heldout_source']}` | {row['teacher_weight']:.3f} | {m['oracle_change_groups']} | "
            f"{m['exact_changed_groups']} | {m['wrong_change']} | {m['tesseract_delta_vs_nearest']} | "
            f"{ranks['le1']} | {ranks['le4']} |"
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
    parser.add_argument("--experiment-id", default="eval300_top8_latent_teacher_aux_source_loo")
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, action="append", default=[])
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    teacher_weights = args.teacher_weight or [0.0, 0.2]

    data = teacher_aux.load_npz(args.features)
    table_rows = read_jsonl(args.table)
    crnn_rows = teacher_aux.build_teacher_lookup(args.crnn)
    abinet_rows = teacher_aux.build_teacher_lookup(args.abinet)
    arrays, group_records, _row_records = teacher_aux.build_arrays(
        data, table_rows, crnn_rows, abinet_rows, args.profile
    )
    sources = source_array(group_records, group_sources(table_rows))
    heldout_sources = sorted(set(str(value) for value in sources if str(value) != "unknown"))
    folds = []
    for source_index, source in enumerate(heldout_sources):
        for teacher_weight in teacher_weights:
            fold_arrays = dict(arrays)
            fold_arrays["partition"] = np.asarray(
                ["val" if str(value) == source else "train" for value in sources],
                dtype=object,
            )
            scores, teacher_pred, train_meta = teacher_aux.train_one(
                fold_arrays,
                args,
                int(args.model_seed + source_index * 1009 + round(float(teacher_weight) * 1000)),
                float(teacher_weight),
            )
            audits, metrics = teacher_aux.evaluate_scores(fold_arrays, group_records, scores, teacher_pred, "val")
            folds.append(
                {
                    "heldout_source": source,
                    "teacher_weight": float(teacher_weight),
                    "model_seed": int(args.model_seed),
                    "train_meta": train_meta,
                    "metrics": metrics,
                    "audits": audits,
                }
            )
    by_weight: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in folds:
        by_weight[f"{row['teacher_weight']:.6f}"].append(row)
    aggregate_by_weight = {key: aggregate(value) for key, value in sorted(by_weight.items())}
    best_key = min(
        aggregate_by_weight,
        key=lambda key: (
            aggregate_by_weight[key]["tesseract_delta_vs_nearest"],
            aggregate_by_weight[key]["wrong_change"],
            -aggregate_by_weight[key]["exact_changed_groups"],
        ),
    )
    interpretation = (
        "This checks whether the N102 teacher-loss auxiliary signal is source-robust when each source domain is "
        "held out. It is a diagnostic split over the existing Eval300 groups and must not be used as a publication "
        "selector or threshold."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_source_leave_out_teacher_aux_ranker_not_promoted_selector",
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
            "teacher_weights": teacher_weights,
            "model_seed": args.model_seed,
        },
        "heldout_sources": heldout_sources,
        "folds": folds,
        "aggregate_by_teacher_weight": aggregate_by_weight,
        "best_teacher_weight_key": best_key,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "teacher_aux_script": sha256_file(ROOT / "scripts/train_top8_latent_teacher_aux_ranker.py"),
        },
        "aggregate": {
            "scalar_metrics": {
                f"teacher_weight_{key}_exact_changed": {"value": float(value["exact_changed_groups"])}
                for key, value in aggregate_by_weight.items()
            }
            | {
                f"teacher_weight_{key}_tesseract_delta": {"value": float(value["tesseract_delta_vs_nearest"])}
                for key, value in aggregate_by_weight.items()
            }
            | {
                f"teacher_weight_{key}_wrong_change": {"value": float(value["wrong_change"])}
                for key, value in aggregate_by_weight.items()
            },
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "best_teacher_weight_key": best_key}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
