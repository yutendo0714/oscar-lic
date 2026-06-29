#!/usr/bin/env python3
"""Evaluate deterministic N117 proposal anchors plus N120-style risk.

N123 showed that a high-capacity latent/code/window proposer overfits the small
clean-positive set. This diagnostic pivots in the opposite direction: keep the
N117 OOF shortlist, use low-variance deterministic proposal orders, and ask
whether the existing separate risk-abstention stage can safely accept any of
those proposals.

No OCR strings, predictions, references, held-out deltas, or oracle labels are
used as inference features. OCR deltas/oracle labels are evaluation labels only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402
import train_top8_oof_shortlist_policy_selector as n118  # noqa: E402
import train_top8_oof_shortlist_two_stage_risk as n120  # noqa: E402


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def log1p_pos(value: Any) -> float:
    return math.log1p(max(finite(value), 0.0))


def zscore(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64)
    mask = np.isfinite(out)
    if not mask.any():
        return np.zeros_like(out)
    mean = float(out[mask].mean())
    std = float(out[mask].std())
    if std < 1.0e-8:
        std = 1.0
    out[mask] = (out[mask] - mean) / std
    out[~mask] = 0.0
    return out


def raw_score_matrix(rows_all: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    topk = np.asarray([finite(row.get("topk_rank", 999), 999.0) for row in rows_all], dtype=np.float64)
    relerr = np.asarray([log1p_pos(row.get("assignment_relative_error", 0.0)) for row in rows_all], dtype=np.float64)
    oof_score = np.asarray([finite(row.get("oof_score", 0.0), 0.0) for row in rows_all], dtype=np.float64)
    oof_rank = np.asarray([finite(row.get("oof_rank", 999), 999.0) for row in rows_all], dtype=np.float64)
    code = np.asarray([finite(row.get("code_index", 0), 0.0) for row in rows_all], dtype=np.float64)
    in_topk4 = np.asarray([finite(row.get("in_topk4", 0), 0.0) for row in rows_all], dtype=np.float64)
    in_oof_top4 = np.asarray([finite(row.get("in_oof_top4", 0), 0.0) for row in rows_all], dtype=np.float64)
    return {
        "topk_rank_anchor": -topk - 1.0e-3 * relerr - 1.0e-6 * code,
        "relative_error_anchor": -relerr - 0.05 * topk - 1.0e-6 * code,
        "oof_score_anchor": oof_score - 1.0e-3 * topk - 1.0e-6 * code,
        "oof_rank_anchor": -oof_rank - 1.0e-3 * topk - 1.0e-6 * code,
        "topk_oof_rrf_anchor": 1.0 / (topk + 1.0) + 1.0 / (oof_rank + 1.0) - 1.0e-4 * relerr,
        "union_membership_anchor": in_topk4 + in_oof_top4 + 1.0 / (topk + 1.0) + 1.0 / (oof_rank + 1.0),
        "z_fused_anchor": zscore(-topk) + zscore(-relerr) + zscore(oof_score),
    }


def risk_eval(
    args: argparse.Namespace,
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
    risk_names = n120.risk_feature_names(train_rows + val_rows)
    x_train = n120.risk_matrix(train_rows, risk_names)
    y_train = np.asarray([int(row["accept_label"]) for row in train_rows], dtype=np.int32)
    x_val = n120.risk_matrix(val_rows, risk_names)
    risk_models: dict[str, Any] = {}
    for risk_name, risk_model in n120.risk_grid(args.random_state).items():
        if int(y_train.sum()) == 0 or int(y_train.sum()) == len(y_train):
            train_risk_scores = {key: 0.0 for key in train_keys}
            val_risk_scores = {key: 0.0 for key in val_keys}
        else:
            risk_model.fit(x_train, y_train)
            train_risk_scores = n120.score_group_model(risk_model, x_train, train_keys)
            val_risk_scores = n120.score_group_model(risk_model, x_val, val_keys)
        budgets = {}
        for budget in [0, 1, 2]:
            tuned = n120.tune_risk_threshold(
                train_groups,
                train_proposals,
                train_risk_scores,
                false_wrong_budget=budget,
            )
            threshold = float(tuned["threshold"])
            val_metrics = bottleneck.evaluate_selection(
                val_groups,
                n120.select_with_risk(val_groups, val_proposals, val_risk_scores, threshold),
            )
            budgets[f"val_budget{budget}"] = {
                "threshold": threshold,
                "train_metrics": tuned["metrics"],
                "metrics": val_metrics,
            }
        risk_models[risk_name] = budgets
    return {
        "train_risk_groups": int(len(train_rows)),
        "train_risk_positive_groups": int(y_train.sum()),
        "risk_feature_count": int(len(risk_names)),
        "val_all_proposals": bottleneck.evaluate_selection(val_groups, val_proposals),
        "train_all_proposals": bottleneck.evaluate_selection(train_groups, train_proposals),
        "val_proposal_rank_counts": n120.proposal_rank_counts(val_groups, val_group_indices, rows_all, proposal_scores),
        "train_proposal_rank_counts": n120.proposal_rank_counts(train_groups, train_group_indices, rows_all, proposal_scores),
        "risk_models": risk_models,
    }


def compact(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics.get('exact_changed_groups', 0)}/{metrics.get('oracle_change_groups', 0)} exact, "
        f"false {metrics.get('false_change', 0)}, wrong {metrics.get('wrong_change', 0)}, "
        f"miss {metrics.get('missed_oracle', 0)}, "
        f"T {metrics.get('tesseract_delta_vs_nearest', 0)}, P {metrics.get('parseq_delta_vs_nearest', 0)}"
    )


def flatten_numeric(prefix: str, value: object, out: dict[str, float]) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}/{str(key).replace('/', '_')}", child, out)
    elif isinstance(value, list):
        for index, child in enumerate(value[:16]):
            flatten_numeric(f"{prefix}/{index}", child, out)


def maybe_start_wandb(args: argparse.Namespace, result: dict[str, Any]) -> Any | None:
    if args.no_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.experiment_id,
        mode=args.wandb_mode,
        tags=["oscar-lic", "N124", "deterministic-proposal-risk"],
        notes="Deterministic N117 proposal anchors plus N120-style risk. Diagnostic only.",
        config={"experiment_id": args.experiment_id, "inputs": result["inputs"], "config": result["config"]},
    )


def finish_wandb(run: Any | None, result: dict[str, Any], output: Path, report: Path) -> None:
    if run is None:
        return
    import wandb

    metrics: dict[str, float] = {}
    flatten_numeric("n124", result, metrics)
    if metrics:
        wandb.log(metrics)
    artifact = wandb.Artifact(result["experiment_id"].replace("/", "_"), type="result-json")
    artifact.add_file(str(output), name=output.name)
    artifact.add_file(str(report), name=report.name)
    run.log_artifact(artifact)
    run.finish()


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Deterministic proposal anchors over the N117 source-modulo OOF shortlist plus N120-style risk abstention. Diagnostic only; no `.oscr` stream is exported.",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "| anchor | all proposals | best budget0 | best budget1 | rank1/rank<=4 |",
        "|---|---|---|---|---:|",
    ]
    for name, row in result["anchors"].items():
        ranks = row["val_proposal_rank_counts"]
        best_b0 = min(
            (risk["val_budget0"]["metrics"] for risk in row["risk_models"].values()),
            key=lambda metrics: (
                int(metrics.get("tesseract_delta_vs_nearest", 0)),
                -int(metrics.get("exact_changed_groups", 0)),
                int(metrics.get("false_change", 0)) + int(metrics.get("wrong_change", 0)),
            ),
        )
        best_b1 = min(
            (risk["val_budget1"]["metrics"] for risk in row["risk_models"].values()),
            key=lambda metrics: (
                int(metrics.get("tesseract_delta_vs_nearest", 0)),
                -int(metrics.get("exact_changed_groups", 0)),
                int(metrics.get("false_change", 0)) + int(metrics.get("wrong_change", 0)),
            ),
        )
        lines.append(
            f"| {name} | {compact(row['val_all_proposals'])} | {compact(best_b0)} | "
            f"{compact(best_b1)} | {ranks.get('rank_le1', 0)}/{ranks.get('rank_le4', 0)} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-oof-deterministic-proposal-risk",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
    ]
    if result.get("wandb_run_id"):
        lines.append(f"wandb_run_id: {result['wandb_run_id']}")
    lines.extend(
        [
            "inputs:",
            "  table:",
            f"    path: {result['inputs']['table']['path']}",
            f"    sha256: {result['inputs']['table']['sha256']}",
        ]
    )
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
            "  analyzer:",
            "    path: scripts/analyze_top8_oof_deterministic_proposal_risk.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  risk_source:",
            "    path: scripts/train_top8_oof_shortlist_two_stage_risk.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_two_stage_risk.py')}",
            "conclusion: diagnostic deterministic proposal anchors; no counted oscr export",
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
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oscar-lic")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups, by_candidate = bottleneck.load_table(args.table)
    split = bottleneck.split_groups(groups)
    train_groups = split["train"]
    val_groups = split["val"]
    oof_scores, oof_meta = oof_shortlist.build_oof_scores(args.score_file, by_candidate, modulo=args.source_modulo)
    rows_all, row_index = n118.build_dataset(groups, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count)
    train_group_indices = n118.group_indices(
        train_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    val_group_indices = n118.group_indices(
        val_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    anchors = {}
    for name, scores in raw_score_matrix(rows_all).items():
        anchors[name] = risk_eval(
            args,
            train_groups,
            val_groups,
            train_group_indices,
            val_group_indices,
            rows_all,
            scores,
        )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_deterministic_proposal_risk_not_promoted",
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
            "anchors": sorted(anchors),
        },
        "data_summary": {
            "groups": len(groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "selector_rows": len(rows_all),
            "oof": {key: value for key, value in oof_meta.items() if key != "score_files"},
        },
        "anchors": anchors,
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "risk_source": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_two_stage_risk.py"),
            "selector_source": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_policy_selector.py"),
        },
        "interpretation": (
            "This low-variance diagnostic checks whether N123's failure was merely high-capacity overfit. "
            "A promotable result would still need to beat the current actual-bitstream floor before `.oscr` export; "
            "otherwise deterministic anchors should serve only as simple controls for future verifier objectives."
        ),
    }
    run = maybe_start_wandb(args, result)
    if run is not None:
        result["wandb_run_id"] = run.id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    finish_wandb(run, result, args.output, args.report)
    write_manifest(args.manifest, result, args.output, args.report)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "wandb_run_id": result.get("wandb_run_id"),
                "anchors": {
                    name: {
                        "all": compact(row["val_all_proposals"]),
                        "rank1": row["val_proposal_rank_counts"].get("rank_le1", 0),
                        "rank4": row["val_proposal_rank_counts"].get("rank_le4", 0),
                    }
                    for name, row in anchors.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
