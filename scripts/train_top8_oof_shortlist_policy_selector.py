#!/usr/bin/env python3
"""Train a no-op-inclusive selector over the N117 source-modulo OOF shortlist.

This diagnostic uses a leakage-controlled first-stage shortlist:

* nearest/no-op is always included;
* non-nearest candidates are `topk_rank <= 4` union OOF-score top-4;
* OOF scores are filtered by source modulo so every score is from a model that
  held out that candidate's source-index slice.

No OCR strings, OCR predictions, held-out OCR deltas, references, or oracle
labels are used as features. OCR deltas are evaluation-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def allowed_table_feature(name: str) -> bool:
    if name in {"topk_rank", "assignment_relative_error"}:
        return True
    if name.startswith("codebook_") or name.startswith("img_"):
        blocked = [
            "oracle",
            "label",
            "tesseract",
            "parseq",
            "prediction",
            "reference",
            "source_image",
            "image_path",
            "nearest_distance",
            "distance",
            "exact",
        ]
        return not any(token in name for token in blocked)
    return False


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names = {
        "is_nearest",
        "in_topk4",
        "in_oof_top4",
        "oof_score",
        "oof_rank",
        "oof_margin_to_best",
        "group_oof_best",
        "group_oof_second",
        "group_oof_gap",
        "candidate_slot",
    }
    for row in rows[: min(len(rows), 256)]:
        for key, value in row.items():
            if allowed_table_feature(key) and isinstance(value, (int, float)):
                names.add(key)
    return sorted(names)


def feature_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    x = np.zeros((len(rows), len(names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            value = finite_float(row.get(name))
            x[i, j] = 0.0 if value is None else float(value)
    x[~np.isfinite(x)] = 0.0
    return x


def candidate_rows(
    rows: list[dict[str, Any]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> list[dict[str, Any]]:
    nearest = bottleneck.nearest_row(rows)
    code_to_row = {int(nearest["code_index"]): dict(nearest)}
    for row in rows:
        if int(row.get("is_nearest", 0)):
            continue
        if int(row.get("topk_rank", 999)) <= topk_cap:
            code_to_row[int(row["code_index"])] = dict(row)
    for row in oof_shortlist.nonnearest_ranked(rows, scores)[:score_count]:
        code_to_row[int(row["code_index"])] = dict(row)

    ranked = oof_shortlist.nonnearest_ranked(rows, scores)
    best = float(scores[oof_shortlist.candidate_key(ranked[0])]) if ranked else 0.0
    second = float(scores[oof_shortlist.candidate_key(ranked[1])]) if len(ranked) > 1 else best
    rank_by_code = {int(row["code_index"]): rank for rank, row in enumerate(ranked, start=1)}
    out = []
    for slot, row in enumerate(sorted(code_to_row.values(), key=lambda item: (int(item.get("topk_rank", 999)), int(item["code_index"])))):
        key = oof_shortlist.candidate_key(row)
        score = float(scores.get(key, 0.0))
        rank = int(rank_by_code.get(int(row["code_index"]), 999))
        row["candidate_slot"] = slot
        row["in_topk4"] = int((not int(row.get("is_nearest", 0))) and int(row.get("topk_rank", 999)) <= topk_cap)
        row["in_oof_top4"] = int(rank <= score_count)
        row["oof_score"] = score
        row["oof_rank"] = rank
        row["oof_margin_to_best"] = score - best
        row["group_oof_best"] = best
        row["group_oof_second"] = second
        row["group_oof_gap"] = best - second
        out.append(row)
    return out


def build_dataset(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int, int, int], int]]:
    rows_out: list[dict[str, Any]] = []
    row_index: dict[tuple[int, int, int, int, int], int] = {}
    for group_rows in groups.values():
        shortlist_rows = candidate_rows(group_rows, scores, topk_cap=topk_cap, score_count=score_count)
        nearest = bottleneck.nearest_row(group_rows)
        oracle = bottleneck.oracle_row(group_rows)
        shortlist_codes = {int(row["code_index"]) for row in shortlist_rows}
        positive_code = int(oracle["code_index"]) if int(oracle["code_index"]) in shortlist_codes else int(nearest["code_index"])
        for row in shortlist_rows:
            key = bottleneck.candidate_key(row)
            row["selector_label"] = int(int(row["code_index"]) == positive_code)
            row_index[key] = len(rows_out)
            rows_out.append(row)
    return rows_out, row_index


def group_indices(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    rows_all: list[dict[str, Any]],
    row_index: dict[tuple[int, int, int, int, int], int],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> dict[tuple[int, int, int, int], list[int]]:
    out: dict[tuple[int, int, int, int], list[int]] = {}
    for key, group_rows in groups.items():
        indices: list[int] = []
        for row in candidate_rows(group_rows, scores, topk_cap=topk_cap, score_count=score_count):
            idx = row_index.get(bottleneck.candidate_key(row))
            if idx is not None:
                indices.append(idx)
        out[key] = indices
    return out


def classifier_grid(random_state: int) -> dict[str, Any]:
    return {
        "logistic_l2_c02": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=random_state),
        ),
        "hist_gradient_l2": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=0.2,
            random_state=random_state,
        ),
        "random_forest_d5": RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=4,
        ),
    }


def score_model(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1].astype(np.float64)
        return prob.reshape(-1).astype(np.float64)
    if hasattr(model, "decision_function"):
        return model.decision_function(x).astype(np.float64)
    return model.predict(x).astype(np.float64)


def select_argmax(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, group_rows in groups.items():
        indices = group_to_indices.get(key, [])
        if not indices:
            selected[key] = int(bottleneck.nearest_row(group_rows)["code_index"])
            continue
        best = max(indices, key=lambda idx: (float(scores[idx]), -int(rows_all[idx].get("topk_rank", 999)), -int(rows_all[idx]["code_index"])))
        selected[key] = int(rows_all[best]["code_index"])
    return selected


def margins_for_groups(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[tuple[int, int, int, int], tuple[float, int]]:
    out: dict[tuple[int, int, int, int], tuple[float, int]] = {}
    for key, group_rows in groups.items():
        nearest_code = int(bottleneck.nearest_row(group_rows)["code_index"])
        nearest_score = None
        best_non = None
        for idx in group_to_indices.get(key, []):
            row = rows_all[idx]
            score = float(scores[idx])
            if int(row["code_index"]) == nearest_code:
                nearest_score = score
            elif best_non is None or (score, -int(row.get("topk_rank", 999)), -int(row["code_index"])) > best_non[:3]:
                best_non = (score, -int(row.get("topk_rank", 999)), -int(row["code_index"]), int(row["code_index"]))
        if nearest_score is None:
            nearest_score = 0.0
        if best_non is None:
            out[key] = (float("-inf"), nearest_code)
        else:
            out[key] = (float(best_non[0] - nearest_score), int(best_non[3]))
    return out


def select_by_margin(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    margins: dict[tuple[int, int, int, int], tuple[float, int]],
    threshold: float,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        nearest_code = int(bottleneck.nearest_row(rows)["code_index"])
        margin, code = margins.get(key, (float("-inf"), nearest_code))
        selected[key] = int(code) if margin >= threshold else nearest_code
    return selected


def tune_threshold(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    margins: dict[tuple[int, int, int, int], tuple[float, int]],
    *,
    false_wrong_budget: int,
) -> dict[str, Any]:
    thresholds = sorted({margin for margin, _ in margins.values() if np.isfinite(margin)})
    thresholds = [float("inf")] + thresholds + [min(thresholds) - 1.0 if thresholds else -1.0]
    best = None
    for threshold in thresholds:
        metrics = bottleneck.evaluate_selection(groups, select_by_margin(groups, margins, threshold))
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
        metrics = bottleneck.evaluate_selection(groups, select_by_margin(groups, margins, threshold))
    else:
        threshold = float(best[4])
        metrics = best[5]
    return {"threshold": threshold, "metrics": metrics}


def rank_counts(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, Any]:
    counts = Counter()
    for key, group_rows in groups.items():
        nearest = bottleneck.nearest_row(group_rows)
        oracle = bottleneck.oracle_row(group_rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        counts["oracle_change_groups"] += 1
        ranked = sorted(
            group_to_indices.get(key, []),
            key=lambda idx: (float(scores[idx]), -int(rows_all[idx].get("topk_rank", 999)), -int(rows_all[idx]["code_index"])),
            reverse=True,
        )
        rank = None
        for pos, idx in enumerate(ranked, start=1):
            if int(rows_all[idx]["code_index"]) == int(oracle["code_index"]):
                rank = pos
                break
        if rank is None:
            counts["oracle_missing"] += 1
        else:
            for cap in [1, 2, 4, 8]:
                counts[f"rank_le{cap}"] += int(rank <= cap)
    return {key: int(value) for key, value in counts.items()}


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
        "No-op-inclusive selector over the N117 source-modulo OOF shortlist. Diagnostic only; no `.oscr` stream is exported.",
        "",
        "## Validation",
        "",
        "| model | argmax | budget0 margin | budget1 margin | rank1/rank<=4 |",
        "|---|---|---|---|---:|",
    ]
    for name, row in result["models"].items():
        ranks = row["val_rank_counts"]
        b0 = row["val_margin_policies"]["budget0"]["metrics"]
        b1 = row["val_margin_policies"]["budget1"]["metrics"]
        lines.append(
            f"| {name} | {compact(row['val_argmax_policy'])} | {compact(b0)} | {compact(b1)} | "
            f"{ranks.get('rank_le1', 0)}/{ranks.get('rank_le4', 0)} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-oof-shortlist-policy",
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
            f"    sha256: {sha256_file(output)}",
            "  report:",
            f"    path: {report}",
            f"    sha256: {sha256_file(report)}",
            "scripts:",
            "  selector:",
            "    path: scripts/train_top8_oof_shortlist_policy_selector.py",
            f"    sha256: {sha256_file(Path(__file__))}",
            "conclusion: diagnostic selector; no counted oscr export",
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
    rows_all, row_index = build_dataset(groups, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count)
    names = feature_names(rows_all)
    x = feature_matrix(rows_all, names)
    y = np.asarray([int(row["selector_label"]) for row in rows_all], dtype=np.int32)
    train_group_indices = group_indices(train_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count)
    val_group_indices = group_indices(val_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count)
    train_indices = np.asarray([idx for indices in train_group_indices.values() for idx in indices], dtype=np.int64)
    models: dict[str, Any] = {}
    for name, clf in classifier_grid(args.random_state).items():
        clf.fit(x[train_indices], y[train_indices])
        scores = score_model(clf, x)
        train_margins = margins_for_groups(train_groups, train_group_indices, rows_all, scores)
        val_margins = margins_for_groups(val_groups, val_group_indices, rows_all, scores)
        margin_policies: dict[str, Any] = {}
        for budget in [0, 1, 2]:
            tuned = tune_threshold(train_groups, train_margins, false_wrong_budget=budget)
            val_metrics = bottleneck.evaluate_selection(
                val_groups, select_by_margin(val_groups, val_margins, float(tuned["threshold"]))
            )
            margin_policies[f"budget{budget}"] = {
                "train": tuned,
                "val": {"threshold": tuned["threshold"], "metrics": val_metrics},
            }
        models[name] = {
            "train_argmax_policy": bottleneck.evaluate_selection(
                train_groups, select_argmax(train_groups, train_group_indices, rows_all, scores)
            ),
            "val_argmax_policy": bottleneck.evaluate_selection(
                val_groups, select_argmax(val_groups, val_group_indices, rows_all, scores)
            ),
            "train_rank_counts": rank_counts(train_groups, train_group_indices, rows_all, scores),
            "val_rank_counts": rank_counts(val_groups, val_group_indices, rows_all, scores),
            "val_margin_policies": {
                key: value["val"] for key, value in margin_policies.items()
            },
            "train_margin_policies": {
                key: value["train"] for key, value in margin_policies.items()
            },
        }
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_selector_not_promoted",
        "code_commit": bottleneck.git_commit(),
        "inputs": {
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "score_files": oof_meta["score_files"],
        },
        "config": {
            "source_modulo": args.source_modulo,
            "topk_cap": args.topk_cap,
            "score_count": args.score_count,
            "random_state": args.random_state,
            "feature_count": len(names),
            "features": names,
        },
        "data_summary": {
            "groups": len(groups),
            "rows": len(rows_all),
            "train_rows": int(len(train_indices)),
            "positive_rows": int(y.sum()),
            "oof": {k: v for k, v in oof_meta.items() if k != "score_files"},
        },
        "models": models,
        "hashes": {"script": sha256_file(Path(__file__))},
        "interpretation": (
            "The selector is diagnostic-only. A useful result must beat the current actual-bitstream floor at "
            "table level before any counted `.oscr` stream export. The N117 shortlist improves candidate coverage, "
            "but this experiment tests whether ordinary train/val no-op supervision can use that headroom safely."
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
                "models": {
                    name: {
                        "argmax": compact(row["val_argmax_policy"]),
                        "budget0": compact(row["val_margin_policies"]["budget0"]["metrics"]),
                        "budget1": compact(row["val_margin_policies"]["budget1"]["metrics"]),
                        "rank1": row["val_rank_counts"].get("rank_le1", 0),
                        "rank4": row["val_rank_counts"].get("rank_le4", 0),
                    }
                    for name, row in models.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
