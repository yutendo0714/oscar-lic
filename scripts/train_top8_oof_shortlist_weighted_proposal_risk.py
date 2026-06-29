#!/usr/bin/env python3
"""Hard-positive weighted proposal diagnostic for the N117 OOF shortlist.

N121 showed that risk thresholding is capped by proposal misses: the best N120
hist-gradient proposer has 6 exact proposals, 9 wrong proposals on oracle-change
groups, and 2 shortlist-missing groups. This script changes only the
non-nearest proposal training weights, then reuses the N120 risk-abstention
stage. It remains diagnostic-only and exports no `.oscr` streams.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402
import train_top8_oof_shortlist_policy_selector as n118  # noqa: E402
import train_top8_oof_shortlist_two_stage_risk as n120  # noqa: E402


def json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_sanitize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_sanitize(child) for child in value]
    return value


def proposal_model(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=220,
        max_leaf_nodes=15,
        l2_regularization=0.2,
        random_state=random_state,
    )


def risk_models(random_state: int) -> dict[str, Any]:
    return {
        "risk_hist_gradient": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=7,
            l2_regularization=0.4,
            random_state=random_state,
        ),
        "risk_random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=4,
        ),
    }


def sample_weights(
    rows_all: list[dict[str, Any]],
    indices: np.ndarray,
    y: np.ndarray,
    *,
    pos_weight: float,
    hard_pos_weight: float,
    noop_neg_weight: float,
) -> np.ndarray:
    weights = np.ones(len(indices), dtype=np.float64)
    for out_pos, idx in enumerate(indices):
        row = rows_all[int(idx)]
        if int(y[int(idx)]):
            weights[out_pos] = pos_weight
            if int(row.get("topk_rank", 999)) > 4 or int(row.get("oof_rank", 999)) > 4:
                weights[out_pos] *= hard_pos_weight
        elif not int(row.get("group_oracle_changes_code", 0)):
            weights[out_pos] = noop_neg_weight
    return weights


def score_risk_policy(
    risk_model: Any,
    train_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    val_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    train_group_indices: dict[tuple[int, int, int, int], list[int]],
    val_group_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    proposal_scores: np.ndarray,
) -> dict[str, Any]:
    train_proposals = n120.select_best_nonnearest(train_groups, train_group_indices, rows_all, proposal_scores)
    val_proposals = n120.select_best_nonnearest(val_groups, val_group_indices, rows_all, proposal_scores)
    train_rows, train_keys = n120.proposal_group_features(
        train_groups, train_group_indices, rows_all, proposal_scores, train_proposals
    )
    val_rows, val_keys = n120.proposal_group_features(
        val_groups, val_group_indices, rows_all, proposal_scores, val_proposals
    )
    names = n120.risk_feature_names(train_rows + val_rows)
    x_train = n120.risk_matrix(train_rows, names)
    y_train = np.asarray([int(row["accept_label"]) for row in train_rows], dtype=np.int32)
    x_val = n120.risk_matrix(val_rows, names)
    if int(y_train.sum()) == 0 or int(y_train.sum()) == len(y_train):
        train_scores = {key: 0.0 for key in train_keys}
        val_scores = {key: 0.0 for key in val_keys}
    else:
        risk_model.fit(x_train, y_train)
        train_scores = n120.score_group_model(risk_model, x_train, train_keys)
        val_scores = n120.score_group_model(risk_model, x_val, val_keys)
    out: dict[str, Any] = {
        "risk_feature_count": len(names),
        "train_risk_positive_groups": int(y_train.sum()),
        "train_risk_groups": len(train_rows),
    }
    for budget in [0, 1, 2]:
        tuned = n120.tune_risk_threshold(train_groups, train_proposals, train_scores, false_wrong_budget=budget)
        threshold = float(tuned["threshold"])
        out[f"val_budget{budget}"] = {
            "threshold": threshold,
            "train_metrics": tuned["metrics"],
            "metrics": bottleneck.evaluate_selection(
                val_groups,
                n120.select_with_risk(val_groups, val_proposals, val_scores, threshold),
            ),
        }
    return out


def compact(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics.get('exact_changed_groups', 0)}/{metrics.get('oracle_change_groups', 0)} exact, "
        f"false {metrics.get('false_change', 0)}, wrong {metrics.get('wrong_change', 0)}, "
        f"miss {metrics.get('missed_oracle', 0)}, "
        f"T {metrics.get('tesseract_delta_vs_nearest', 0)}, P {metrics.get('parseq_delta_vs_nearest', 0)}"
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Hard-positive weighted non-nearest proposal diagnostic over the N117 OOF shortlist. No `.oscr` stream is exported.",
        "",
        "## Validation",
        "",
        "| pos_w | hard_pos_w | noop_neg_w | all proposals | risk | budget0 | budget1 | rank1/rank<=4 |",
        "|---:|---:|---:|---|---|---|---|---:|",
    ]
    for row in result["runs"]:
        rank = row["val_proposal_rank_counts"]
        for risk_name, risk in row["risk_models"].items():
            lines.append(
                f"| {row['weights']['pos_weight']} | {row['weights']['hard_pos_weight']} | {row['weights']['noop_neg_weight']} | "
                f"{compact(row['val_all_proposals'])} | {risk_name} | "
                f"{compact(risk['val_budget0']['metrics'])} | {compact(risk['val_budget1']['metrics'])} | "
                f"{rank.get('rank_le1', 0)}/{rank.get('rank_le4', 0)} |"
            )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-oof-weighted-proposal-risk",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
        "  table:",
        f"    path: {result['inputs']['table']['path']}",
        f"    sha256: {result['inputs']['table']['sha256']}",
    ]
    for item in result["inputs"]["score_files"]:
        lines.extend([f"  score_r{item['remainder']}:", f"    path: {item['path']}", f"    sha256: {item['sha256']}"])
    lines.extend(
        [
            "outputs:",
            "  result_json:",
            f"    path: {output}",
            f"    sha256: {bottleneck.sha256_file(output)}",
            "  report:",
            f"    path: {report}",
            f"    sha256: {bottleneck.sha256_file(report)}",
            "scripts:",
            "  trainer:",
            "    path: scripts/train_top8_oof_shortlist_weighted_proposal_risk.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  risk_source:",
            "    path: scripts/train_top8_oof_shortlist_two_stage_risk.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_two_stage_risk.py')}",
            "conclusion: diagnostic weighted proposal; no counted oscr export",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, action="append", required=True)
    parser.add_argument("--source-modulo", type=int, default=5)
    parser.add_argument("--topk-cap", type=int, default=4)
    parser.add_argument("--score-count", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=20260626)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups, by_candidate = bottleneck.load_table(args.table)
    partitions = bottleneck.split_groups(groups)
    train_groups = partitions["train"]
    val_groups = partitions["val"]
    oof_scores, oof_meta = oof_shortlist.build_oof_scores(args.score_file, by_candidate, modulo=args.source_modulo)
    rows_all, row_index = n118.build_dataset(groups, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count)
    names = n118.feature_names(rows_all)
    x = n118.feature_matrix(rows_all, names)
    y_prop = n120.proposal_labels(rows_all)
    train_group_indices = n118.group_indices(
        train_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    val_group_indices = n118.group_indices(
        val_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    train_non_indices = n120.group_train_indices(train_groups, train_group_indices, rows_all)
    grid = []
    for pos_weight in [1.0, 2.0, 4.0, 8.0]:
        for hard_pos_weight in [1.0, 2.0, 4.0]:
            for noop_neg_weight in [0.25, 0.5, 1.0]:
                grid.append((pos_weight, hard_pos_weight, noop_neg_weight))
    runs: list[dict[str, Any]] = []
    for pos_weight, hard_pos_weight, noop_neg_weight in grid:
        weights = sample_weights(
            rows_all,
            train_non_indices,
            y_prop,
            pos_weight=pos_weight,
            hard_pos_weight=hard_pos_weight,
            noop_neg_weight=noop_neg_weight,
        )
        model = proposal_model(args.random_state)
        model.fit(x[train_non_indices], y_prop[train_non_indices], sample_weight=weights)
        proposal_scores = n118.score_model(model, x)
        val_proposals = n120.select_best_nonnearest(val_groups, val_group_indices, rows_all, proposal_scores)
        risk_rows: dict[str, Any] = {}
        for risk_name, risk_model in risk_models(args.random_state).items():
            risk_rows[risk_name] = score_risk_policy(
                risk_model,
                train_groups,
                val_groups,
                train_group_indices,
                val_group_indices,
                rows_all,
                proposal_scores,
            )
        runs.append(
            {
                "weights": {
                    "pos_weight": pos_weight,
                    "hard_pos_weight": hard_pos_weight,
                    "noop_neg_weight": noop_neg_weight,
                },
                "train_weight_stats": {
                    "min": float(weights.min()),
                    "max": float(weights.max()),
                    "mean": float(weights.mean()),
                },
                "val_all_proposals": bottleneck.evaluate_selection(val_groups, val_proposals),
                "val_proposal_rank_counts": n120.proposal_rank_counts(
                    val_groups, val_group_indices, rows_all, proposal_scores
                ),
                "risk_models": risk_rows,
            }
        )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_weighted_proposal_not_promoted",
        "code_commit": bottleneck.git_commit(),
        "inputs": {
            "table": {"path": str(args.table), "sha256": bottleneck.sha256_file(args.table)},
            "score_files": oof_meta["score_files"],
        },
        "config": {
            "source_modulo": args.source_modulo,
            "topk_cap": args.topk_cap,
            "score_count": args.score_count,
            "random_state": args.random_state,
            "proposal_feature_count": len(names),
            "grid_size": len(grid),
        },
        "data_summary": {
            "groups": len(groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "train_non_nearest_rows": int(len(train_non_indices)),
            "train_positive_rows": int(y_prop[train_non_indices].sum()),
            "oof": {key: value for key, value in oof_meta.items() if key != "score_files"},
        },
        "runs": runs,
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "risk_source": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_two_stage_risk.py"),
        },
        "interpretation": (
            "This diagnostic tests whether simple hard-positive/no-op-negative weighting improves the N120 "
            "non-nearest proposal stage. A promotable result would still need to beat the current actual-bitstream "
            "floor before any counted `.oscr` export."
        ),
    }
    result = json_sanitize(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report)
    best = sorted(
        runs,
        key=lambda row: min(
            int(risk["val_budget0"]["metrics"].get("tesseract_delta_vs_nearest", 999))
            for risk in row["risk_models"].values()
        ),
    )[:5]
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "best": [
                    {
                        "weights": row["weights"],
                        "all_proposals": compact(row["val_all_proposals"]),
                        "rank1": row["val_proposal_rank_counts"].get("rank_le1", 0),
                        "rank4": row["val_proposal_rank_counts"].get("rank_le4", 0),
                        "risk": {
                            name: compact(risk["val_budget0"]["metrics"])
                            for name, risk in row["risk_models"].items()
                        },
                    }
                    for row in best
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
