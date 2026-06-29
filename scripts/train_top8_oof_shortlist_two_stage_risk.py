#!/usr/bin/env python3
"""Train a two-stage non-nearest proposal plus risk-abstention policy.

This follows N119: the N117 OOF shortlist has useful non-nearest candidates, but
N118's single nearest/no-op-inclusive selector pushes all covered oracle
candidates below nearest. This diagnostic removes nearest from the proposal
softmax, then trains a separate group-level accept/abstain model.

No OCR strings, predictions, references, held-out deltas, or oracle labels are
used as features. OCR deltas and oracle labels are used only for train
supervision/evaluation in this diagnostic.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402
import train_top8_oof_shortlist_policy_selector as n118  # noqa: E402


def proposer_grid(random_state: int) -> dict[str, Any]:
    return {
        "proposal_logistic_c05": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=random_state),
        ),
        "proposal_hist_gradient": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=0.2,
            random_state=random_state,
        ),
        "proposal_random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=4,
        ),
    }


def risk_grid(random_state: int) -> dict[str, Any]:
    return {
        "risk_logistic_c05": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=random_state),
        ),
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


def group_train_indices(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
) -> np.ndarray:
    indices: list[int] = []
    for key in groups:
        for idx in group_to_indices.get(key, []):
            if not int(rows_all[idx].get("is_nearest", 0)):
                indices.append(idx)
    return np.asarray(indices, dtype=np.int64)


def proposal_labels(rows_all: list[dict[str, Any]]) -> np.ndarray:
    labels = []
    for row in rows_all:
        labels.append(
            int(
                (not int(row.get("is_nearest", 0)))
                and int(row.get("group_oracle_changes_code", 0))
                and int(row.get("label_assignment_oracle_choice", 0))
            )
        )
    return np.asarray(labels, dtype=np.int32)


def select_best_nonnearest(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    proposal_scores: np.ndarray,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        best_idx = None
        for idx in group_to_indices.get(key, []):
            row = rows_all[idx]
            if int(row.get("is_nearest", 0)):
                continue
            if best_idx is None or (
                float(proposal_scores[idx]),
                -int(row.get("topk_rank", 999)),
                -int(row["code_index"]),
            ) > (
                float(proposal_scores[best_idx]),
                -int(rows_all[best_idx].get("topk_rank", 999)),
                -int(rows_all[best_idx]["code_index"]),
            ):
                best_idx = idx
        if best_idx is None:
            selected[key] = int(bottleneck.nearest_row(rows)["code_index"])
        else:
            selected[key] = int(rows_all[best_idx]["code_index"])
    return selected


def proposal_rank_counts(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    proposal_scores: np.ndarray,
) -> dict[str, Any]:
    counts = Counter()
    for key, rows in groups.items():
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        counts["oracle_change_groups"] += 1
        ranked = [
            idx
            for idx in group_to_indices.get(key, [])
            if not int(rows_all[idx].get("is_nearest", 0))
        ]
        ranked.sort(
            key=lambda idx: (
                float(proposal_scores[idx]),
                -int(rows_all[idx].get("topk_rank", 999)),
                -int(rows_all[idx]["code_index"]),
            ),
            reverse=True,
        )
        rank = None
        for pos, idx in enumerate(ranked, start=1):
            if int(rows_all[idx]["code_index"]) == int(oracle["code_index"]):
                rank = pos
                break
        if rank is None:
            counts["oracle_missing"] += 1
            continue
        for cap in [1, 2, 4, 8]:
            counts[f"rank_le{cap}"] += int(rank <= cap)
    return {key: int(value) for key, value in counts.items()}


def selected_row_for_group(
    key: tuple[int, int, int, int],
    selected_code: int,
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
) -> tuple[int | None, dict[str, Any] | None]:
    for idx in group_to_indices.get(key, []):
        if int(rows_all[idx]["code_index"]) == int(selected_code):
            return idx, rows_all[idx]
    return None, None


def proposal_group_features(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    proposal_scores: np.ndarray,
    selected_codes: dict[tuple[int, int, int, int], int],
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int, int]]]:
    out: list[dict[str, Any]] = []
    keys: list[tuple[int, int, int, int]] = []
    names = n118.feature_names(rows_all)
    for key, rows in groups.items():
        nearest = bottleneck.nearest_row(rows)
        nearest_code = int(nearest["code_index"])
        selected_code = int(selected_codes.get(key, nearest_code))
        idx, row = selected_row_for_group(key, selected_code, group_to_indices, rows_all)
        if idx is None or row is None or selected_code == nearest_code:
            continue
        non_indices = [
            item
            for item in group_to_indices.get(key, [])
            if not int(rows_all[item].get("is_nearest", 0))
        ]
        non_scores = sorted((float(proposal_scores[item]) for item in non_indices), reverse=True)
        best = non_scores[0] if non_scores else 0.0
        second = non_scores[1] if len(non_scores) > 1 else best
        mean = float(np.mean(non_scores)) if non_scores else 0.0
        std = float(np.std(non_scores)) if non_scores else 0.0
        feat = {name: row.get(name, 0.0) for name in names}
        feat.update(
            {
                "proposal_score": float(proposal_scores[idx]),
                "proposal_score_gap": float(best - second),
                "proposal_score_mean": mean,
                "proposal_score_std": std,
                "proposal_score_minus_mean": float(proposal_scores[idx] - mean),
                "proposal_candidate_count": len(non_indices),
                "proposal_selected_topk_rank": int(row.get("topk_rank", 999)),
                "proposal_selected_oof_rank": int(row.get("oof_rank", 999)),
                "proposal_selected_in_oof_top4": int(row.get("in_oof_top4", 0)),
                "proposal_selected_in_topk4": int(row.get("in_topk4", 0)),
                "accept_label": int(
                    int(row.get("group_oracle_changes_code", 0))
                    and int(row.get("label_assignment_oracle_choice", 0))
                ),
            }
        )
        out.append(feat)
        keys.append(key)
    return out, keys


def risk_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names = set()
    for row in rows:
        for key, value in row.items():
            if key == "accept_label":
                continue
            if key.startswith("proposal_") or n118.allowed_table_feature(key) or key in {
                "in_topk4",
                "in_oof_top4",
                "oof_score",
                "oof_rank",
                "oof_margin_to_best",
                "group_oof_best",
                "group_oof_second",
                "group_oof_gap",
                "candidate_slot",
            }:
                if isinstance(value, (int, float)):
                    names.add(key)
    return sorted(names)


def risk_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return n118.feature_matrix(rows, names)


def select_with_risk(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    proposal_codes: dict[tuple[int, int, int, int], int],
    risk_scores: dict[tuple[int, int, int, int], float],
    threshold: float,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        nearest = int(bottleneck.nearest_row(rows)["code_index"])
        code = int(proposal_codes.get(key, nearest))
        selected[key] = code if code != nearest and risk_scores.get(key, float("-inf")) >= threshold else nearest
    return selected


def tune_risk_threshold(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    proposal_codes: dict[tuple[int, int, int, int], int],
    risk_scores: dict[tuple[int, int, int, int], float],
    *,
    false_wrong_budget: int,
) -> dict[str, Any]:
    thresholds = sorted(set(risk_scores.values()))
    thresholds = [float("inf")] + thresholds + [min(thresholds) - 1.0 if thresholds else -1.0]
    best = None
    for threshold in thresholds:
        metrics = bottleneck.evaluate_selection(groups, select_with_risk(groups, proposal_codes, risk_scores, threshold))
        bad = int(metrics.get("false_change", 0)) + int(metrics.get("wrong_change", 0))
        if bad > false_wrong_budget or int(metrics.get("parseq_delta_vs_nearest", 0)) > 0:
            continue
        candidate = (
            int(metrics.get("tesseract_delta_vs_nearest", 0)),
            -int(metrics.get("exact_changed_groups", 0)),
            int(metrics.get("changed_groups", 0)),
            -float(threshold if np.isfinite(threshold) else 1.0e9),
            threshold,
            metrics,
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        threshold = float("inf")
        metrics = bottleneck.evaluate_selection(groups, select_with_risk(groups, proposal_codes, risk_scores, threshold))
    else:
        threshold = float(best[4])
        metrics = best[5]
    return {"threshold": threshold, "metrics": metrics}


def score_group_model(model: Any, x: np.ndarray, keys: list[tuple[int, int, int, int]]) -> dict[tuple[int, int, int, int], float]:
    scores = n118.score_model(model, x)
    return {key: float(score) for key, score in zip(keys, scores, strict=True)}


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
        "Two-stage diagnostic: non-nearest proposal followed by separate risk-abstention over the N117 OOF shortlist. No `.oscr` stream is exported.",
        "",
        "## Validation",
        "",
        "| proposer | risk | all proposals | budget0 | budget1 | budget2 | nonnearest rank1/rank<=4 |",
        "|---|---|---|---|---|---|---:|",
    ]
    for name, prop in result["proposers"].items():
        rank = prop["val_proposal_rank_counts"]
        for risk_name, risk in prop["risk_models"].items():
            lines.append(
                f"| {name} | {risk_name} | {compact(prop['val_all_proposals'])} | "
                f"{compact(risk['val_budget0']['metrics'])} | "
                f"{compact(risk['val_budget1']['metrics'])} | "
                f"{compact(risk['val_budget2']['metrics'])} | "
                f"{rank.get('rank_le1', 0)}/{rank.get('rank_le4', 0)} |"
            )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-oof-two-stage-risk",
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
            "    path: scripts/train_top8_oof_shortlist_two_stage_risk.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  selector_source:",
            "    path: scripts/train_top8_oof_shortlist_policy_selector.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_policy_selector.py')}",
            "conclusion: diagnostic two-stage risk policy; no counted oscr export",
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
    y_prop = proposal_labels(rows_all)
    train_group_indices = n118.group_indices(
        train_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    val_group_indices = n118.group_indices(
        val_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    train_non_indices = group_train_indices(train_groups, train_group_indices, rows_all)
    proposers: dict[str, Any] = {}
    for prop_name, prop_model in proposer_grid(args.random_state).items():
        prop_model.fit(x[train_non_indices], y_prop[train_non_indices])
        prop_scores = n118.score_model(prop_model, x)
        train_proposals = select_best_nonnearest(train_groups, train_group_indices, rows_all, prop_scores)
        val_proposals = select_best_nonnearest(val_groups, val_group_indices, rows_all, prop_scores)
        train_risk_rows, train_risk_keys = proposal_group_features(
            train_groups, train_group_indices, rows_all, prop_scores, train_proposals
        )
        val_risk_rows, val_risk_keys = proposal_group_features(
            val_groups, val_group_indices, rows_all, prop_scores, val_proposals
        )
        risk_names = risk_feature_names(train_risk_rows + val_risk_rows)
        x_train_risk = risk_matrix(train_risk_rows, risk_names)
        y_train_risk = np.asarray([int(row["accept_label"]) for row in train_risk_rows], dtype=np.int32)
        x_val_risk = risk_matrix(val_risk_rows, risk_names)
        risk_models: dict[str, Any] = {}
        for risk_name, risk_model in risk_grid(args.random_state).items():
            if int(y_train_risk.sum()) == 0 or int(y_train_risk.sum()) == len(y_train_risk):
                risk_scores_train = {key: 0.0 for key in train_risk_keys}
                risk_scores_val = {key: 0.0 for key in val_risk_keys}
            else:
                risk_model.fit(x_train_risk, y_train_risk)
                risk_scores_train = score_group_model(risk_model, x_train_risk, train_risk_keys)
                risk_scores_val = score_group_model(risk_model, x_val_risk, val_risk_keys)
            tuned: dict[str, Any] = {}
            for budget in [0, 1, 2]:
                train_tuned = tune_risk_threshold(
                    train_groups,
                    train_proposals,
                    risk_scores_train,
                    false_wrong_budget=budget,
                )
                threshold = float(train_tuned["threshold"])
                val_metrics = bottleneck.evaluate_selection(
                    val_groups,
                    select_with_risk(val_groups, val_proposals, risk_scores_val, threshold),
                )
                tuned[f"val_budget{budget}"] = {
                    "threshold": threshold,
                    "train_metrics": train_tuned["metrics"],
                    "metrics": val_metrics,
                }
            risk_models[risk_name] = tuned
        proposers[prop_name] = {
            "train_positive_rows": int(y_prop[train_non_indices].sum()),
            "train_non_nearest_rows": int(len(train_non_indices)),
            "risk_feature_count": len(risk_names),
            "train_risk_groups": len(train_risk_rows),
            "train_risk_positive_groups": int(y_train_risk.sum()),
            "val_all_proposals": bottleneck.evaluate_selection(val_groups, val_proposals),
            "val_proposal_rank_counts": proposal_rank_counts(
                val_groups, val_group_indices, rows_all, prop_scores
            ),
            "risk_models": risk_models,
        }
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_two_stage_risk_not_promoted",
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
        },
        "data_summary": {
            "groups": len(groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "selector_rows": len(rows_all),
            "oof": {key: value for key, value in oof_meta.items() if key != "score_files"},
        },
        "proposers": proposers,
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "selector_source": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_policy_selector.py"),
        },
        "interpretation": (
            "This diagnostic tests the N119 recommendation by separating non-nearest proposal from no-op risk. "
            "It is not promotable unless a validation policy beats the current actual-bitstream floor before counted `.oscr` export."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "proposers": {
                    name: {
                        "all_proposals": compact(row["val_all_proposals"]),
                        "rank1": row["val_proposal_rank_counts"].get("rank_le1", 0),
                        "rank4": row["val_proposal_rank_counts"].get("rank_le4", 0),
                        "risk": {
                            risk_name: {
                                key: compact(value["metrics"])
                                for key, value in risk_row.items()
                                if key.startswith("val_budget")
                            }
                            for risk_name, risk_row in row["risk_models"].items()
                        },
                    }
                    for name, row in proposers.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
