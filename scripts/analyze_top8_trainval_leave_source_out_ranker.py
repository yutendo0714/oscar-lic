#!/usr/bin/env python3
"""Leave-source-out diagnostic for the Eval300 top-8 tabular ranker."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
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


def group_source(rows: list[dict[str, Any]], indices: list[int]) -> str:
    return str(rows[indices[0]].get("source", ""))


def listwise_groups_for_keys(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, str, int, int], list[int]],
    allowed_keys: set[tuple[int, str, int, int]],
) -> list[dict[str, Any]]:
    out = []
    for key, indices in sorted(groups.items()):
        if key not in allowed_keys:
            continue
        target_idx = ranker.target_index_for_group(rows, indices)
        if target_idx is None:
            continue
        choices = ranker.nonnearest_indices(rows, indices)
        if target_idx not in choices or len(choices) < 2:
            continue
        out.append(
            {
                "key": key,
                "indices": choices,
                "target_local": choices.index(target_idx),
                "target_index": target_idx,
            }
        )
    return out


def score_models(
    features_norm: np.ndarray,
    train_groups: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    source_offset: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x_all = torch.from_numpy(features_norm).to(device)
    all_scores = []
    metas = []
    for offset in range(args.model_seeds):
        seed = args.seed + source_offset * 1000 + offset
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        scores, meta = ranker.train_model(x_all, train_groups, seed, args, device)
        all_scores.append(scores)
        metas.append(meta)
    return np.mean(np.stack(all_scores, axis=0), axis=0), metas


def evaluate_source(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, str, int, int], list[int]],
    eval_keys: set[tuple[int, str, int, int]],
    scores: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = Counter()
    rank_values = []
    topk_values = []
    tess_delta = 0
    parseq_delta = 0
    audits = []
    for key in sorted(eval_keys):
        indices = groups[key]
        nearest = next(idx for idx in indices if int(rows[idx]["code_index"]) == int(rows[idx]["nearest_code"]))
        oracle = next((idx for idx in indices if int(rows[idx].get("label_assignment_oracle_choice", 0))), nearest)
        nearest_code = int(rows[nearest]["code_index"])
        oracle_code = int(rows[oracle]["code_index"])
        nonnearest = ranker.nonnearest_indices(rows, indices)
        by_score = sorted(nonnearest, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])), reverse=True)
        best_idx = by_score[0] if by_score else nearest
        selected_idx = best_idx if oracle_code != nearest_code else nearest
        selected_code = int(rows[selected_idx]["code_index"])
        status = ranker.classify(selected_code, nearest_code, oracle_code)
        counts[status] += 1
        counts["changed_groups"] += int(selected_code != nearest_code)
        counts["exact_changed_groups"] += int(status == "exact" and oracle_code != nearest_code)
        if oracle_code != nearest_code:
            oracle_rank = 1 + next(rank for rank, idx in enumerate(by_score) if int(rows[idx]["code_index"]) == oracle_code)
            rank_values.append(int(oracle_rank))
            topk_values.append(int(rows[oracle]["topk_rank"]))
        else:
            oracle_rank = None
        tess_delta += int(rows[selected_idx]["tesseract_delta_vs_nearest"]) if selected_code != nearest_code else 0
        parseq_delta += int(rows[selected_idx]["parseq_delta_vs_nearest"]) if selected_code != nearest_code else 0
        audits.append(
            {
                "key": {"real_seed": key[0], "partition": key[1], "source_index": key[2], "candidate_index": key[3]},
                "source": rows[nearest].get("source"),
                "reference": rows[nearest].get("reference"),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "selected_code_oracle_change_only": selected_code,
                "selected_status_oracle_change_only": status,
                "oracle_nonnearest_score_rank": oracle_rank,
                "oracle_topk_rank": int(rows[oracle]["topk_rank"]),
                "best_nonnearest_code": int(rows[best_idx]["code_index"]) if nonnearest else nearest_code,
                "best_nonnearest_score": float(scores[best_idx]) if nonnearest else 0.0,
                "selected_tesseract_delta_vs_nearest": int(rows[selected_idx]["tesseract_delta_vs_nearest"])
                if selected_code != nearest_code
                else 0,
                "selected_parseq_delta_vs_nearest": int(rows[selected_idx]["parseq_delta_vs_nearest"])
                if selected_code != nearest_code
                else 0,
            }
        )
    metrics = {
        "groups": int(len(eval_keys)),
        "oracle_change_groups": int(len(rank_values)),
        "changed_groups_oracle_change_only": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "exact_changed_groups": int(counts["exact_changed_groups"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "tesseract_delta_vs_nearest": int(tess_delta),
        "parseq_delta_vs_nearest": int(parseq_delta),
        "oracle_nonnearest_score_rank_counts": ranker.rank_counts(rank_values),
        "oracle_topk_rank_counts": ranker.rank_counts(topk_values),
    }
    return metrics, audits


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Leave-source-out candidate-ranking diagnostic for Eval300 top-8 assignment.",
        "This is not a deployment selector and does not export counted `.oscr` streams.",
        "",
        "## Source Results",
        "",
        "| held-out source | groups | oracle changes | rank<=1 | rank<=4 | exact changed | wrong | Tesseract delta | PARSeq delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["source_results"]:
        metrics = row["metrics"]
        ranks = metrics["oracle_nonnearest_score_rank_counts"]
        lines.append(
            f"| `{row['heldout_source']}` | {metrics['groups']} | {metrics['oracle_change_groups']} | "
            f"{ranks['le1']} | {ranks['le4']} | {metrics['exact_changed_groups']} | {metrics['wrong_change']} | "
            f"{metrics['tesseract_delta_vs_nearest']} | {metrics['parseq_delta_vs_nearest']} |"
        )
    agg = result["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- held-out oracle-change groups: `{agg['oracle_change_groups']}`",
            f"- learned score rank<=1: `{agg['oracle_nonnearest_score_rank_counts']['le1']}`",
            f"- learned score rank<=4: `{agg['oracle_nonnearest_score_rank_counts']['le4']}`",
            f"- oracle-change-only Tesseract delta: `{agg['tesseract_delta_vs_nearest']}`",
            f"- oracle-change-only PARSeq delta: `{agg['parseq_delta_vs_nearest']}`",
            "",
            "## Interpretation",
            "",
            result["interpretation"],
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
    parser.add_argument("--experiment-id", default="eval300_top8_trainval_leave_source_out_ranker")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--model-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = ranker.read_jsonl(args.table)
    groups = ranker.group_rows(rows)
    feature_names = ranker.numeric_feature_names(rows)
    features = ranker.build_features(rows, feature_names)
    sources = sorted({group_source(rows, indices) for indices in groups.values()})
    group_sources = {key: group_source(rows, indices) for key, indices in groups.items()}
    group_nonnearest = {
        key: [idx for idx in indices if int(rows[idx]["code_index"]) != int(rows[idx]["nearest_code"])]
        for key, indices in groups.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    source_results = []
    aggregate_counts = Counter()
    aggregate_ranks: list[int] = []
    aggregate_topk: list[int] = []
    all_audits = []
    for source_offset, heldout_source in enumerate(sources):
        train_keys = {key for key, source in group_sources.items() if source != heldout_source}
        eval_keys = {key for key, source in group_sources.items() if source == heldout_source}
        train_groups = listwise_groups_for_keys(rows, groups, train_keys)
        train_nonnearest = np.asarray(
            [idx for key in train_keys for idx in group_nonnearest[key]],
            dtype=np.int64,
        )
        if not len(train_groups) or not len(train_nonnearest):
            continue
        features_norm = ranker.normalize(features[train_nonnearest], features)
        scores, model_metas = score_models(features_norm, train_groups, args, device, source_offset)
        metrics, audits = evaluate_source(rows, groups, eval_keys, scores)
        for audit in audits:
            audit["heldout_source"] = heldout_source
        all_audits.extend(audits)
        source_results.append(
            {
                "heldout_source": heldout_source,
                "train_groups": len(train_keys),
                "train_listwise_groups": len(train_groups),
                "eval_groups": len(eval_keys),
                "feature_count": len(feature_names),
                "models": model_metas,
                "metrics": metrics,
            }
        )
        aggregate_counts["groups"] += metrics["groups"]
        aggregate_counts["oracle_change_groups"] += metrics["oracle_change_groups"]
        aggregate_counts["changed_groups_oracle_change_only"] += metrics["changed_groups_oracle_change_only"]
        aggregate_counts["exact"] += metrics["exact"]
        aggregate_counts["exact_changed_groups"] += metrics["exact_changed_groups"]
        aggregate_counts["false_change"] += metrics["false_change"]
        aggregate_counts["wrong_change"] += metrics["wrong_change"]
        aggregate_counts["missed_oracle"] += metrics["missed_oracle"]
        aggregate_counts["tesseract_delta_vs_nearest"] += metrics["tesseract_delta_vs_nearest"]
        aggregate_counts["parseq_delta_vs_nearest"] += metrics["parseq_delta_vs_nearest"]
        for audit in audits:
            rank_value = audit["oracle_nonnearest_score_rank"]
            if rank_value is not None:
                aggregate_ranks.append(int(rank_value))
                aggregate_topk.append(int(audit["oracle_topk_rank"]))
    aggregate = dict(aggregate_counts)
    aggregate["oracle_nonnearest_score_rank_counts"] = ranker.rank_counts(aggregate_ranks)
    aggregate["oracle_topk_rank_counts"] = ranker.rank_counts(aggregate_topk)
    interpretation = (
        "Leave-source-out ranking tests whether the N091 trainval signal is source-local. "
        "If rank<=1/rank<=4 collapse relative to N091, the next useful work is source/domain-balanced "
        "calibration or richer representation; if ranking remains comparable, no-op acceptor evidence is the main bottleneck."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_leave_source_out_ranker_not_promoted_selector",
        "inputs": {
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
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
            "feature_names": feature_names,
            "feature_rule": "same deployable tabular features as N091: topk_rank, assignment_relative_error, codebook_* and img_* numeric columns",
            "device": str(device),
        },
        "source_results": source_results,
        "aggregate": aggregate,
        "group_audits": all_audits,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "ranker_script": sha256_file(SCRIPT_DIR / "train_top8_trainval_tabular_listwise_ranker.py"),
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
