#!/usr/bin/env python3
"""Audit why the N118 no-op selector misses the N117 OOF shortlist headroom.

This is diagnostic only. It rebuilds the N118 shortlist and classifiers, then
records per-group margins/ranks for oracle candidates, nearest/no-op rows, and
best wrong non-nearest candidates. No policy rows or `.oscr` streams are
exported.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
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


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def row_status(rows: list[dict[str, Any]], selected_code: int) -> str:
    nearest = bottleneck.nearest_row(rows)
    oracle = bottleneck.oracle_row(rows)
    nearest_code = int(nearest["code_index"])
    oracle_code = int(oracle["code_index"])
    if int(selected_code) == oracle_code:
        return "exact"
    if int(selected_code) == nearest_code and oracle_code != nearest_code:
        return "missed_oracle"
    if int(selected_code) != nearest_code and oracle_code == nearest_code:
        return "false_change"
    return "wrong_change"


def code_to_row(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["code_index"]): row for row in rows}


def rank_lookup(
    indices: list[int],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    include_nearest: bool,
) -> dict[int, int]:
    ranked = [
        idx
        for idx in indices
        if include_nearest or not int(rows_all[idx].get("is_nearest", 0))
    ]
    ranked.sort(
        key=lambda idx: (
            float(scores[idx]),
            -int(rows_all[idx].get("topk_rank", 999)),
            -int(rows_all[idx]["code_index"]),
        ),
        reverse=True,
    )
    return {int(rows_all[idx]["code_index"]): rank for rank, idx in enumerate(ranked, start=1)}


def best_non_nearest(
    indices: list[int],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
) -> tuple[int | None, float | None]:
    best_idx = None
    for idx in indices:
        if int(rows_all[idx].get("is_nearest", 0)):
            continue
        if best_idx is None or (
            float(scores[idx]),
            -int(rows_all[idx].get("topk_rank", 999)),
            -int(rows_all[idx]["code_index"]),
        ) > (
            float(scores[best_idx]),
            -int(rows_all[best_idx].get("topk_rank", 999)),
            -int(rows_all[best_idx]["code_index"]),
        ):
            best_idx = idx
    if best_idx is None:
        return None, None
    return int(rows_all[best_idx]["code_index"]), float(scores[best_idx])


def shortlist_membership(
    rows: list[dict[str, Any]],
    oof_scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> dict[int, dict[str, Any]]:
    ranked = oof_shortlist.nonnearest_ranked(rows, oof_scores)
    oof_rank = {int(row["code_index"]): rank for rank, row in enumerate(ranked, start=1)}
    oof_top = {int(row["code_index"]) for row in ranked[:score_count]}
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        code = int(row["code_index"])
        in_topk = (not int(row.get("is_nearest", 0))) and int(row.get("topk_rank", 999)) <= topk_cap
        in_oof = code in oof_top
        if int(row.get("is_nearest", 0)) or in_topk or in_oof:
            key = oof_shortlist.candidate_key(row)
            out[code] = {
                "in_topk_cap": int(in_topk),
                "in_oof_top": int(in_oof),
                "oof_rank": int(oof_rank.get(code, 999)),
                "oof_score": float(oof_scores.get(key, 0.0)),
            }
    return out


def summarize_shortlist(
    val_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    oof_scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = Counter()
    oracle_oof_ranks: list[float] = []
    oracle_topk_ranks: list[float] = []
    details: list[dict[str, Any]] = []
    for key, rows in sorted(val_groups.items()):
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        counts["oracle_change_groups"] += 1
        membership = shortlist_membership(rows, oof_scores, topk_cap=topk_cap, score_count=score_count)
        oracle_code = int(oracle["code_index"])
        member = membership.get(oracle_code)
        in_shortlist = member is not None
        counts["covered_oracle_changes"] += int(in_shortlist)
        counts["missing_oracle_changes"] += int(not in_shortlist)
        if in_shortlist:
            counts["covered_by_topk_cap"] += int(member["in_topk_cap"])
            counts["covered_by_oof_top"] += int(member["in_oof_top"])
            counts["covered_by_both"] += int(member["in_topk_cap"] and member["in_oof_top"])
            oracle_oof_ranks.append(float(member["oof_rank"]))
        oracle_topk_ranks.append(float(oracle.get("topk_rank", 999)))
        details.append(
            {
                "group_key": list(key),
                "source": rows[0].get("source"),
                "reference": rows[0].get("reference"),
                "reference_length": int(rows[0].get("reference_length", len(str(rows[0].get("reference", ""))))),
                "nearest_code": int(nearest["code_index"]),
                "oracle_code": oracle_code,
                "oracle_topk_rank": int(oracle.get("topk_rank", 999)),
                "oracle_tesseract_delta_vs_nearest": int(oracle.get("tesseract_delta_vs_nearest", 0)),
                "oracle_parseq_delta_vs_nearest": int(oracle.get("parseq_delta_vs_nearest", 0)),
                "in_shortlist": int(in_shortlist),
                "membership": member,
            }
        )
    return (
        {
            **{key: int(value) for key, value in counts.items()},
            "oracle_topk_rank_stats": stats(oracle_topk_ranks),
            "covered_oracle_oof_rank_stats": stats(oracle_oof_ranks),
        },
        details,
    )


def model_audit(
    model_name: str,
    scores: np.ndarray,
    val_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    val_group_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    val_margins: dict[tuple[int, int, int, int], tuple[float, int]],
    margin_thresholds: dict[str, float],
) -> dict[str, Any]:
    counts = Counter()
    oracle_minus_nearest: list[float] = []
    oracle_minus_best_wrong: list[float] = []
    best_non_minus_nearest_all: list[float] = []
    best_non_minus_nearest_oracle: list[float] = []
    rank_all_counts = Counter()
    rank_non_counts = Counter()
    group_details: list[dict[str, Any]] = []
    for key, rows in sorted(val_groups.items()):
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        oracle_changes = oracle_code != nearest_code
        indices = val_group_indices.get(key, [])
        by_code = code_to_row(rows)
        score_by_code = {int(rows_all[idx]["code_index"]): float(scores[idx]) for idx in indices}
        nearest_score = float(score_by_code.get(nearest_code, 0.0))
        oracle_score = score_by_code.get(oracle_code)
        best_non_code, best_non_score = best_non_nearest(indices, rows_all, scores)
        if best_non_score is not None:
            best_non_minus_nearest_all.append(float(best_non_score - nearest_score))
            if oracle_changes:
                best_non_minus_nearest_oracle.append(float(best_non_score - nearest_score))
        rank_all = rank_lookup(indices, rows_all, scores, include_nearest=True)
        rank_non = rank_lookup(indices, rows_all, scores, include_nearest=False)
        if oracle_changes:
            counts["oracle_change_groups"] += 1
            if oracle_score is None:
                counts["oracle_missing_from_shortlist"] += 1
            else:
                counts["oracle_covered"] += 1
                all_rank = int(rank_all.get(oracle_code, 999))
                non_rank = int(rank_non.get(oracle_code, 999))
                for cap in [1, 2, 4, 8]:
                    rank_all_counts[f"rank_all_le{cap}"] += int(all_rank <= cap)
                    rank_non_counts[f"rank_non_le{cap}"] += int(non_rank <= cap)
                oracle_minus_nearest.append(float(oracle_score - nearest_score))
                wrong_scores = [
                    float(score_by_code[code])
                    for code in score_by_code
                    if code not in {nearest_code, oracle_code}
                ]
                if wrong_scores:
                    oracle_minus_best_wrong.append(float(oracle_score - max(wrong_scores)))
        argmax_selection = n118.select_argmax(val_groups, val_group_indices, rows_all, scores).get(key, nearest_code)
        threshold_selections = {
            name: n118.select_by_margin(val_groups, val_margins, threshold).get(key, nearest_code)
            for name, threshold in margin_thresholds.items()
        }
        argmax_status = row_status(rows, int(argmax_selection))
        counts[f"argmax_{argmax_status}"] += 1
        counts["argmax_changed"] += int(int(argmax_selection) != nearest_code)
        for name, code in threshold_selections.items():
            status = row_status(rows, int(code))
            counts[f"{name}_{status}"] += 1
            counts[f"{name}_changed"] += int(int(code) != nearest_code)
        if oracle_changes:
            detail: dict[str, Any] = {
                "group_key": list(key),
                "source": rows[0].get("source"),
                "reference": rows[0].get("reference"),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "oracle_topk_rank": int(oracle.get("topk_rank", 999)),
                "nearest_score": nearest_score,
                "oracle_score": oracle_score,
                "oracle_minus_nearest": None if oracle_score is None else float(oracle_score - nearest_score),
                "oracle_rank_all": int(rank_all.get(oracle_code, 999)),
                "oracle_rank_nonnearest": int(rank_non.get(oracle_code, 999)),
                "best_nonnearest_code": best_non_code,
                "best_nonnearest_score": best_non_score,
                "best_nonnearest_minus_nearest": None if best_non_score is None else float(best_non_score - nearest_score),
                "argmax_code": int(argmax_selection),
                "argmax_status": argmax_status,
                "argmax_tesseract_delta_vs_nearest": int(by_code[int(argmax_selection)].get("tesseract_delta_vs_nearest", 0))
                if int(argmax_selection) != nearest_code
                else 0,
            }
            for name, code in threshold_selections.items():
                detail[f"{name}_code"] = int(code)
                detail[f"{name}_status"] = row_status(rows, int(code))
            group_details.append(detail)
    return {
        "model_name": model_name,
        "counts": {key: int(value) for key, value in counts.items()},
        "rank_all_counts": {key: int(value) for key, value in rank_all_counts.items()},
        "rank_nonnearest_counts": {key: int(value) for key, value in rank_non_counts.items()},
        "oracle_minus_nearest": {
            "stats": stats(oracle_minus_nearest),
            "positive_count": int(sum(value > 0.0 for value in oracle_minus_nearest)),
        },
        "oracle_minus_best_wrong_nonnearest": {
            "stats": stats(oracle_minus_best_wrong),
            "positive_count": int(sum(value > 0.0 for value in oracle_minus_best_wrong)),
        },
        "best_nonnearest_minus_nearest_all_val": {
            "stats": stats(best_non_minus_nearest_all),
            "positive_count": int(sum(value > 0.0 for value in best_non_minus_nearest_all)),
        },
        "best_nonnearest_minus_nearest_oracle_groups": {
            "stats": stats(best_non_minus_nearest_oracle),
            "positive_count": int(sum(value > 0.0 for value in best_non_minus_nearest_oracle)),
        },
        "oracle_group_details": group_details,
    }


def compact_policy(metrics: dict[str, Any]) -> str:
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
        "Diagnostic audit of the N118 selector failure over the N117 source-modulo OOF shortlist. No `.oscr` stream is exported.",
        "",
        "## Shortlist Coverage",
        "",
        json.dumps(result["shortlist_summary"], indent=2, ensure_ascii=False),
        "",
        "## Model Margins",
        "",
        "| model | N118 argmax | oracle>nearest | oracle>wrong nonnearest | best nonnearest>nearest | oracle rank<=1/4 all | oracle rank<=1/4 nonnearest |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["models"].items():
        policy = row["policies"]["argmax"]
        rank_all = row["diagnostics"]["rank_all_counts"]
        rank_non = row["diagnostics"]["rank_nonnearest_counts"]
        lines.append(
            f"| {name} | {compact_policy(policy)} | "
            f"{row['diagnostics']['oracle_minus_nearest']['positive_count']}/"
            f"{row['diagnostics']['oracle_minus_nearest']['stats']['count']} | "
            f"{row['diagnostics']['oracle_minus_best_wrong_nonnearest']['positive_count']}/"
            f"{row['diagnostics']['oracle_minus_best_wrong_nonnearest']['stats']['count']} | "
            f"{row['diagnostics']['best_nonnearest_minus_nearest_all_val']['positive_count']}/"
            f"{row['diagnostics']['best_nonnearest_minus_nearest_all_val']['stats']['count']} | "
            f"{rank_all.get('rank_all_le1', 0)}/{rank_all.get('rank_all_le4', 0)} | "
            f"{rank_non.get('rank_non_le1', 0)}/{rank_non.get('rank_non_le4', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-oof-shortlist-policy-failure-audit",
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
            "  audit:",
            "    path: scripts/analyze_top8_oof_shortlist_policy_failures.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  selector_rebuild:",
            "    path: scripts/train_top8_oof_shortlist_policy_selector.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_policy_selector.py')}",
            "  shortlist:",
            "    path: scripts/analyze_top8_source_mod_oof_shortlist.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/analyze_top8_source_mod_oof_shortlist.py')}",
            "conclusion: diagnostic failure audit; no counted oscr export",
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
    y = np.asarray([int(row["selector_label"]) for row in rows_all], dtype=np.int32)
    train_group_indices = n118.group_indices(
        train_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    val_group_indices = n118.group_indices(
        val_groups, rows_all, row_index, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )
    train_indices = np.asarray([idx for indices in train_group_indices.values() for idx in indices], dtype=np.int64)
    shortlist_summary, oracle_details = summarize_shortlist(
        val_groups, oof_scores, topk_cap=args.topk_cap, score_count=args.score_count
    )

    models: dict[str, Any] = {}
    for name, clf in n118.classifier_grid(args.random_state).items():
        clf.fit(x[train_indices], y[train_indices])
        scores = n118.score_model(clf, x)
        train_margins = n118.margins_for_groups(train_groups, train_group_indices, rows_all, scores)
        val_margins = n118.margins_for_groups(val_groups, val_group_indices, rows_all, scores)
        thresholds: dict[str, float] = {}
        policies: dict[str, Any] = {
            "argmax": bottleneck.evaluate_selection(
                val_groups, n118.select_argmax(val_groups, val_group_indices, rows_all, scores)
            )
        }
        for budget in [0, 1, 2]:
            tuned = n118.tune_threshold(train_groups, train_margins, false_wrong_budget=budget)
            threshold = float(tuned["threshold"])
            thresholds[f"budget{budget}"] = threshold
            policies[f"budget{budget}"] = bottleneck.evaluate_selection(
                val_groups, n118.select_by_margin(val_groups, val_margins, threshold)
            )
        models[name] = {
            "thresholds": thresholds,
            "policies": policies,
            "diagnostics": model_audit(
                name,
                scores,
                val_groups,
                val_group_indices,
                rows_all,
                val_margins,
                thresholds,
            ),
        }

    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_failure_audit_not_promoted",
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
            "feature_count": len(names),
        },
        "data_summary": {
            "groups": len(groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "selector_rows": len(rows_all),
            "positive_rows": int(y.sum()),
            "oof": {k: v for k, v in oof_meta.items() if k != "score_files"},
        },
        "shortlist_summary": shortlist_summary,
        "oracle_change_details": oracle_details,
        "models": models,
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "selector_rebuild_script": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_policy_selector.py"),
            "shortlist_script": bottleneck.sha256_file(ROOT / "scripts/analyze_top8_source_mod_oof_shortlist.py"),
        },
        "interpretation": (
            "N117's union shortlist is a real first-stage improvement, but N118 fails mostly at the no-op/risk "
            "boundary and at ranking the useful non-nearest above nearby wrong non-nearest codes. This audit should "
            "be used to design a materially different risk objective or richer candidate-local evidence, not to "
            "seed/capacity sweep the shallow selector family."
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
                "shortlist": shortlist_summary,
                "models": {
                    name: {
                        "argmax": compact_policy(row["policies"]["argmax"]),
                        "oracle_gt_nearest": row["diagnostics"]["oracle_minus_nearest"]["positive_count"],
                        "oracle_gt_wrong": row["diagnostics"]["oracle_minus_best_wrong_nonnearest"]["positive_count"],
                        "best_non_gt_nearest": row["diagnostics"]["best_nonnearest_minus_nearest_all_val"]["positive_count"],
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
