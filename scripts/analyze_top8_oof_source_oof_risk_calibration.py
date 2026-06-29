#!/usr/bin/env python3
"""Source-OOF risk calibration for deterministic N117 proposal anchors.

N124 showed that deterministic proposal anchors can match N120 proposal rank1
but train-tuned risk gates still leak false/wrong validation changes or collapse
to no-op. This diagnostic changes only the calibration protocol: risk thresholds
are tuned on source-held-out train scores, then applied to a full-train risk
model on validation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_oof_deterministic_proposal_risk as n124  # noqa: E402
import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402
import train_top8_oof_shortlist_policy_selector as n118  # noqa: E402
import train_top8_oof_shortlist_two_stage_risk as n120  # noqa: E402


def compact(metrics: dict[str, Any]) -> str:
    return n120.compact(metrics)


def risk_sources(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    keys: list[tuple[int, int, int, int]],
) -> np.ndarray:
    return np.asarray([str(groups[key][0].get("source", "unknown")) for key in keys], dtype=object)


def model_scores(model: Any, x: np.ndarray) -> np.ndarray:
    return n118.score_model(model, x).astype(np.float64)


def source_oof_scores(
    base_model: Any,
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = np.full(y.shape[0], np.nan, dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    for source in sorted(set(sources.tolist())):
        holdout = sources == source
        train = ~holdout
        if int(train.sum()) == 0 or int(holdout.sum()) == 0:
            continue
        if int(y[train].sum()) == 0 or int(y[train].sum()) == int(train.sum()):
            fill = float(y[train].mean()) if int(train.sum()) else 0.0
            out[holdout] = fill
            trained = False
        else:
            model = copy.deepcopy(base_model)
            model.fit(x[train], y[train])
            out[holdout] = model_scores(model, x[holdout])
            trained = True
        summaries.append(
            {
                "heldout_source": source,
                "trained_model": bool(trained),
                "train_rows": int(train.sum()),
                "holdout_rows": int(holdout.sum()),
                "train_positive": int(y[train].sum()),
                "holdout_positive": int(y[holdout].sum()),
            }
        )
    if not np.isfinite(out).all():
        missing = ~np.isfinite(out)
        out[missing] = float(y.mean()) if len(y) else 0.0
    return out, summaries


def evaluate_anchor(
    args: argparse.Namespace,
    anchor_name: str,
    proposal_scores: np.ndarray,
    train_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    val_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    train_group_indices: dict[tuple[int, int, int, int], list[int]],
    val_group_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
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
    sources = risk_sources(train_groups, train_keys)
    models = {}
    for risk_name, risk_model in n120.risk_grid(args.random_state).items():
        oof_score_values, fold_summaries = source_oof_scores(risk_model, x_train, y_train, sources)
        oof_scores = {key: float(score) for key, score in zip(train_keys, oof_score_values, strict=True)}
        if int(y_train.sum()) == 0 or int(y_train.sum()) == len(y_train):
            val_scores = {key: 0.0 for key in val_keys}
        else:
            full_model = copy.deepcopy(risk_model)
            full_model.fit(x_train, y_train)
            val_scores = n120.score_group_model(full_model, x_val, val_keys)
        budgets = {}
        for budget in [0, 1, 2]:
            tuned = n120.tune_risk_threshold(
                train_groups,
                train_proposals,
                oof_scores,
                false_wrong_budget=budget,
            )
            threshold = float(tuned["threshold"])
            val_metrics = bottleneck.evaluate_selection(
                val_groups,
                n120.select_with_risk(val_groups, val_proposals, val_scores, threshold),
            )
            budgets[f"val_budget{budget}"] = {
                "threshold": threshold,
                "oof_train_metrics": tuned["metrics"],
                "metrics": val_metrics,
            }
        models[risk_name] = {
            "source_oof_folds": fold_summaries,
            "budgets": budgets,
        }
    return {
        "anchor_name": anchor_name,
        "train_risk_groups": int(len(train_rows)),
        "train_risk_positive_groups": int(y_train.sum()),
        "risk_feature_count": int(len(risk_names)),
        "risk_sources": sorted(set(sources.tolist())),
        "val_all_proposals": bottleneck.evaluate_selection(val_groups, val_proposals),
        "train_all_proposals": bottleneck.evaluate_selection(train_groups, train_proposals),
        "val_proposal_rank_counts": n120.proposal_rank_counts(val_groups, val_group_indices, rows_all, proposal_scores),
        "train_proposal_rank_counts": n120.proposal_rank_counts(train_groups, train_group_indices, rows_all, proposal_scores),
        "risk_models": models,
    }


def flatten_numeric(prefix: str, value: object, out: dict[str, float]) -> None:
    if prefix.endswith("/audits") or prefix.endswith("/source_oof_folds"):
        return
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in {"audits", "source_oof_folds"}:
                continue
            flatten_numeric(f"{prefix}/{str(key).replace('/', '_')}", child, out)
    elif isinstance(value, list):
        for index, child in enumerate(value[:8]):
            flatten_numeric(f"{prefix}/{index}", child, out)


def maybe_start_wandb(args: argparse.Namespace, result: dict[str, Any]) -> Any | None:
    if args.no_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.experiment_id,
        mode=args.wandb_mode,
        tags=["oscar-lic", "N125", "source-oof-risk"],
        notes="Source-OOF risk calibration over deterministic N117 proposal anchors. Diagnostic only.",
        config={"experiment_id": args.experiment_id, "inputs": result["inputs"], "config": result["config"]},
    )


def finish_wandb(run: Any | None, result: dict[str, Any], output: Path, report: Path) -> None:
    if run is None:
        return
    import wandb

    metrics: dict[str, float] = {}
    flatten_numeric("n125", result, metrics)
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
        "Source-OOF calibrated risk thresholds over deterministic N117 proposal anchors. Diagnostic only; no `.oscr` stream is exported.",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "| anchor | all proposals | best budget0 | best budget1 | rank1/rank<=4 |",
        "|---|---|---|---|---:|",
    ]
    for name, row in result["anchors"].items():
        ranks = row["val_proposal_rank_counts"]
        budget0 = []
        budget1 = []
        for risk in row["risk_models"].values():
            budget0.append(risk["budgets"]["val_budget0"]["metrics"])
            budget1.append(risk["budgets"]["val_budget1"]["metrics"])
        best0 = min(
            budget0,
            key=lambda metrics: (
                int(metrics.get("tesseract_delta_vs_nearest", 0)),
                int(metrics.get("false_change", 0)) + int(metrics.get("wrong_change", 0)),
                -int(metrics.get("exact_changed_groups", 0)),
            ),
        )
        best1 = min(
            budget1,
            key=lambda metrics: (
                int(metrics.get("tesseract_delta_vs_nearest", 0)),
                int(metrics.get("false_change", 0)) + int(metrics.get("wrong_change", 0)),
                -int(metrics.get("exact_changed_groups", 0)),
            ),
        )
        lines.append(
            f"| {name} | {compact(row['val_all_proposals'])} | {compact(best0)} | "
            f"{compact(best1)} | {ranks.get('rank_le1', 0)}/{ranks.get('rank_le4', 0)} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-oof-source-oof-risk-calibration",
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
            "    path: scripts/analyze_top8_oof_source_oof_risk_calibration.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  anchor_source:",
            "    path: scripts/analyze_top8_oof_deterministic_proposal_risk.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/analyze_top8_oof_deterministic_proposal_risk.py')}",
            "  risk_source:",
            "    path: scripts/train_top8_oof_shortlist_two_stage_risk.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_two_stage_risk.py')}",
            "conclusion: diagnostic source-oof risk calibration; no counted oscr export",
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
    parser.add_argument("--anchor", action="append", default=["topk_rank_anchor", "oof_rank_anchor", "z_fused_anchor"])
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
    oof_scores, oof_meta = oof_shortlist.build_oof_scores(args.score_file, by_candidate, modulo=args.source_modulo)
    rows_all, row_index = n118.build_dataset(groups, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count)
    train_group_indices = n118.group_indices(
        split["train"], rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    val_group_indices = n118.group_indices(
        split["val"], rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    raw_scores = n124.raw_score_matrix(rows_all)
    anchors = {}
    for anchor in args.anchor:
        if anchor not in raw_scores:
            raise SystemExit(f"unknown anchor {anchor}; available={sorted(raw_scores)}")
        anchors[anchor] = evaluate_anchor(
            args,
            anchor,
            raw_scores[anchor],
            split["train"],
            split["val"],
            train_group_indices,
            val_group_indices,
            rows_all,
        )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_source_oof_risk_not_promoted",
        "code_commit": bottleneck.git_commit(),
        "inputs": {
            "table": {"path": str(args.table), "sha256": bottleneck.sha256_file(args.table)},
            "score_files": oof_meta["score_files"],
        },
        "config": {
            "source_modulo": args.source_modulo,
            "topk_cap": args.topk_cap,
            "score_count": args.score_count,
            "anchors": args.anchor,
            "random_state": args.random_state,
        },
        "data_summary": {
            "groups": len(groups),
            "train_groups": len(split["train"]),
            "val_groups": len(split["val"]),
            "selector_rows": len(rows_all),
            "oof": {key: value for key, value in oof_meta.items() if key != "score_files"},
        },
        "anchors": anchors,
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "anchor_source": bottleneck.sha256_file(ROOT / "scripts/analyze_top8_oof_deterministic_proposal_risk.py"),
            "risk_source": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_two_stage_risk.py"),
        },
        "interpretation": (
            "This diagnostic tests whether train-in-sample risk thresholding caused N124's false/wrong leakage. "
            "It is not promotable unless source-OOF thresholds beat the current actual-bitstream floor before counted `.oscr` export."
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
