#!/usr/bin/env python3
"""Train a pairwise N127 hard-cohort utility verifier.

Unlike N128's row classifier, this diagnostic uses the matched structure in the
N127 cohort file directly: covered oracle-positive rows should score above their
matched no-headroom negatives. Thresholds are still tuned only on train
source-modulo OOF predictions, and no counted `.oscr` streams are exported.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402
import train_top8_hard_cohort_verifier as n128  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cohort_pairs(
    cohort_path: Path,
    row_index: dict[tuple[int, int, int, int, int], int],
    rows_all: list[dict[str, Any]],
    *,
    source_modulo: int,
    heldout_mod: int | None = None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    positives: dict[str, int] = {}
    negatives: dict[str, list[int]] = defaultdict(list)
    missing = 0
    skipped_heldout = 0
    cohort_rows = read_jsonl(cohort_path)
    for row in cohort_rows:
        if str(row.get("assignment_partition", "")) != "train":
            raise ValueError(f"non-train cohort row: {row}")
        key = n128.cohort_candidate_key(row)
        idx = row_index.get(key)
        if idx is None:
            missing += 1
            continue
        row_mod = n128.group_source_mod(bottleneck.group_key(rows_all[idx]), source_modulo)
        if heldout_mod is not None and row_mod == heldout_mod:
            skipped_heldout += 1
            continue
        match_id = str(row["matched_positive_id"])
        if int(row["target_label"]):
            positives[match_id] = idx
        else:
            negatives[match_id].append(idx)
    pairs = []
    for match_id, pos_idx in positives.items():
        for neg_idx in negatives.get(match_id, []):
            pairs.append((pos_idx, neg_idx))
    return pairs, {
        "cohort_rows": len(cohort_rows),
        "missing_rows": missing,
        "skipped_heldout_rows": skipped_heldout,
        "positive_groups": len(positives),
        "pairs": len(pairs),
    }


def fit_pairwise_model(
    x: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    c_value: float,
    random_state: int,
) -> tuple[StandardScaler | None, LogisticRegression | None, dict[str, Any]]:
    if not pairs:
        return None, None, {"pairs": 0, "status": "empty"}
    cohort_indices = sorted({idx for pair in pairs for idx in pair})
    scaler = StandardScaler()
    scaler.fit(x[cohort_indices])
    xz = scaler.transform(x)
    samples = []
    labels = []
    for pos_idx, neg_idx in pairs:
        diff = xz[pos_idx] - xz[neg_idx]
        samples.append(diff)
        labels.append(1)
        samples.append(-diff)
        labels.append(0)
    xp = np.asarray(samples, dtype=np.float32)
    yp = np.asarray(labels, dtype=np.int32)
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        fit_intercept=False,
        max_iter=2000,
        random_state=random_state,
    )
    model.fit(xp, yp)
    train_scores = model.decision_function(xp)
    accuracy = float(((train_scores >= 0) == yp.astype(bool)).mean())
    return scaler, model, {
        "pairs": len(pairs),
        "samples": int(len(yp)),
        "pairwise_train_accuracy": accuracy,
        "coef_l2": float(np.linalg.norm(model.coef_)),
    }


def score_pairwise_model(
    scaler: StandardScaler | None,
    model: LogisticRegression | None,
    x: np.ndarray,
    indices: np.ndarray,
    *,
    total_rows: int,
) -> np.ndarray:
    scores = np.full((total_rows,), np.nan, dtype=np.float64)
    if scaler is None or model is None or not len(indices):
        return scores
    xz = scaler.transform(x[indices])
    scores[indices] = model.decision_function(xz).astype(np.float64)
    return scores


def source_oof_pairwise_scores(
    x: np.ndarray,
    rows_all: list[dict[str, Any]],
    row_index: dict[tuple[int, int, int, int, int], int],
    cohort_path: Path,
    target_indices: np.ndarray,
    *,
    c_value: float,
    random_state: int,
    source_modulo: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.full((len(rows_all),), np.nan, dtype=np.float64)
    folds = []
    for remainder in range(source_modulo):
        pairs, meta = cohort_pairs(
            cohort_path,
            row_index,
            rows_all,
            source_modulo=source_modulo,
            heldout_mod=remainder,
        )
        scaler, model, fit_meta = fit_pairwise_model(x, pairs, c_value=c_value, random_state=random_state)
        fold_target = n128.source_mod_indices(rows_all, target_indices, modulo=source_modulo, remainder=remainder)
        fold_scores = score_pairwise_model(scaler, model, x, fold_target, total_rows=len(rows_all))
        scores[fold_target] = fold_scores[fold_target]
        folds.append({"remainder": remainder, **meta, **fit_meta, "target_rows": int(len(fold_target))})
    return scores, {"folds": folds}


def compact(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics.get('exact_changed_groups', 0)}/{metrics.get('oracle_change_groups', 0)} exact, "
        f"false {metrics.get('false_change', 0)}, wrong {metrics.get('wrong_change', 0)}, "
        f"miss {metrics.get('missed_oracle', 0)}, "
        f"T {metrics.get('tesseract_delta_vs_nearest', 0)}, P {metrics.get('parseq_delta_vs_nearest', 0)}"
    )


def best_policy_summary(models: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for model_name, model_result in models.items():
        for budget_name, policy in model_result["policies"].items():
            metrics = policy["val_metrics"]
            rows.append(
                {
                    "model": model_name,
                    "budget": budget_name,
                    "threshold": policy["threshold"],
                    "exact_changed_groups": int(metrics.get("exact_changed_groups", 0)),
                    "false_wrong": int(metrics.get("false_change", 0)) + int(metrics.get("wrong_change", 0)),
                    "false_change": int(metrics.get("false_change", 0)),
                    "wrong_change": int(metrics.get("wrong_change", 0)),
                    "tesseract_delta_vs_nearest": int(metrics.get("tesseract_delta_vs_nearest", 0)),
                    "parseq_delta_vs_nearest": int(metrics.get("parseq_delta_vs_nearest", 0)),
                    "metrics": metrics,
                }
            )
    rows.sort(
        key=lambda row: (
            row["tesseract_delta_vs_nearest"],
            -row["exact_changed_groups"],
            row["false_wrong"],
            row["parseq_delta_vs_nearest"],
        )
    )
    return rows[0] if rows else {}


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "Pairwise verifier trained from N127 matched hard cohorts. This is a table-level diagnostic; no counted `.oscr` stream is exported.",
        "",
        "## Validation Policies",
        "",
        "| model | pair acc | rank1/rank<=4 | budget | train OOF | validation |",
        "|---|---:|---:|---|---|---|",
    ]
    for model_name, model_result in result["models"].items():
        rank = model_result["val_rank_counts"]
        pair_acc = model_result["full_fit"].get("pairwise_train_accuracy")
        for budget_name, policy in model_result["policies"].items():
            lines.append(
                f"| {model_name} | {pair_acc:.3f} | {rank.get('rank_le1', 0)}/{rank.get('rank_le4', 0)} | "
                f"{budget_name} | {compact(policy['train_oof_metrics'])} | {compact(policy['val_metrics'])} |"
            )
    best = result.get("best_validation_policy", {})
    lines.extend(
        [
            "",
            "## Best Observed Validation Policy",
            "",
            json.dumps(
                {
                    key: best.get(key)
                    for key in [
                        "model",
                        "budget",
                        "exact_changed_groups",
                        "false_wrong",
                        "tesseract_delta_vs_nearest",
                        "parseq_delta_vs_nearest",
                    ]
                },
                ensure_ascii=False,
            ),
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path, cohort: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-hard-cohort-pairwise-verifier",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
        "  table:",
        f"    path: {result['inputs']['table']['path']}",
        f"    sha256: {result['inputs']['table']['sha256']}",
        "  hard_cohort:",
        f"    path: {cohort}",
        f"    sha256: {bottleneck.sha256_file(cohort)}",
    ]
    for item in result["inputs"]["oof_scores"]["score_files"]:
        lines.extend(
            [
                f"  score_r{item['remainder']}:",
                f"    path: {item['path']}",
                f"    sha256: {item['sha256']}",
            ]
        )
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
            "    path: scripts/train_top8_hard_cohort_pairwise_verifier.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  row_verifier:",
            "    path: scripts/train_top8_hard_cohort_verifier.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_hard_cohort_verifier.py')}",
            "  cohort_builder:",
            "    path: scripts/select_top8_verifier_hard_cohorts.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/select_top8_verifier_hard_cohorts.py')}",
            "conclusion: table-level pairwise verifier diagnostic only; no counted oscr export",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--hard-cohort", type=Path, required=True)
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
    rows_all, row_index, group_indices = n128.build_candidate_dataset(
        groups,
        oof_scores,
        topk_cap=args.topk_cap,
        score_count=args.score_count,
    )
    names, scrub = n128.feature_names(rows_all)
    if scrub["leak_like_selected"] or scrub["suspicious_selected"]:
        raise RuntimeError(f"feature scrub failed: {scrub}")
    x = n128.feature_matrix(rows_all, names)
    train_candidate_indices = n128.partition_indices(rows_all, "train")
    val_candidate_indices = n128.partition_indices(rows_all, "val")
    models: dict[str, Any] = {}
    for c_value in [0.05, 0.2, 1.0]:
        model_name = f"pairwise_logistic_c{str(c_value).replace('.', 'p')}"
        pairs, pair_meta = cohort_pairs(
            args.hard_cohort,
            row_index,
            rows_all,
            source_modulo=args.source_modulo,
        )
        scaler, model, fit_meta = fit_pairwise_model(x, pairs, c_value=c_value, random_state=args.random_state)
        full_scores = score_pairwise_model(scaler, model, x, val_candidate_indices, total_rows=len(rows_all))
        train_oof_scores, train_oof_meta = source_oof_pairwise_scores(
            x,
            rows_all,
            row_index,
            args.hard_cohort,
            train_candidate_indices,
            c_value=c_value,
            random_state=args.random_state,
            source_modulo=args.source_modulo,
        )
        policies = {}
        for budget in [0, 1, 3]:
            tuned = n128.tune_threshold(
                train_groups,
                group_indices,
                rows_all,
                train_oof_scores,
                false_wrong_budget=budget,
            )
            threshold = float(tuned["threshold"])
            val_metrics = bottleneck.evaluate_selection(
                val_groups,
                n128.select_by_threshold(val_groups, group_indices, rows_all, full_scores, threshold),
            )
            policies[f"train_oof_budget{budget}"] = {
                "threshold": threshold,
                "train_oof_metrics": tuned["metrics"],
                "val_metrics": val_metrics,
            }
        models[model_name] = {
            "pair_meta": pair_meta,
            "full_fit": fit_meta,
            "train_oof": train_oof_meta,
            "val_rank_counts": n128.proposal_rank_counts(val_groups, group_indices, rows_all, full_scores),
            "val_all_change_metrics": bottleneck.evaluate_selection(
                val_groups,
                n128.select_by_threshold(val_groups, group_indices, rows_all, full_scores, float("-inf")),
            ),
            "policies": policies,
        }
    result = {
        "experiment_id": args.experiment_id,
        "validity": "table_level_pairwise_diagnostic_not_promoted",
        "code_commit": bottleneck.git_commit(),
        "inputs": {
            "table": {"path": str(args.table), "sha256": bottleneck.sha256_file(args.table)},
            "hard_cohort": {"path": str(args.hard_cohort), "sha256": bottleneck.sha256_file(args.hard_cohort)},
            "oof_scores": oof_meta,
        },
        "config": {
            "source_modulo": args.source_modulo,
            "topk_cap": args.topk_cap,
            "score_count": args.score_count,
            "random_state": args.random_state,
            "c_values": [0.05, 0.2, 1.0],
        },
        "data_summary": {
            "groups": len(groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "candidate_rows": len(rows_all),
            "train_candidate_rows": int(len(train_candidate_indices)),
            "val_candidate_rows": int(len(val_candidate_indices)),
        },
        "feature_scrub": scrub,
        "models": models,
        "best_validation_policy": best_policy_summary(models),
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "row_verifier": bottleneck.sha256_file(ROOT / "scripts/train_top8_hard_cohort_verifier.py"),
        },
        "wandb_run_id": "not_logged",
        "interpretation": (
            "N129 tests whether the matched structure in N127 is better used as pairwise utility ranking than row classification. "
            "A positive result would still need counted `.oscr` export; a negative result closes pairwise linear scoring over the same scrubbed feature family."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report, args.hard_cohort)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "best_validation_policy": {
                    key: result["best_validation_policy"].get(key)
                    for key in [
                        "model",
                        "budget",
                        "exact_changed_groups",
                        "false_wrong",
                        "tesseract_delta_vs_nearest",
                        "parseq_delta_vs_nearest",
                    ]
                },
                "feature_count": scrub["selected_feature_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
