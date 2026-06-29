#!/usr/bin/env python3
"""Audit missed positives for the N120 two-stage OOF-shortlist risk policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
import train_top8_oof_shortlist_two_stage_risk as n120  # noqa: E402


def stats(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def json_float(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def proposal_reason(
    rows: list[dict[str, Any]],
    proposal_code: int,
    oof_scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> str:
    nearest = bottleneck.nearest_row(rows)
    oracle = bottleneck.oracle_row(rows)
    nearest_code = int(nearest["code_index"])
    oracle_code = int(oracle["code_index"])
    if oracle_code == nearest_code:
        return "no_oracle_change"
    ranked = oof_shortlist.nonnearest_ranked(rows, oof_scores)
    oof_top = {int(row["code_index"]) for row in ranked[:score_count]}
    oracle_in_topk = int(oracle.get("topk_rank", 999)) <= topk_cap
    oracle_in_oof = oracle_code in oof_top
    if not oracle_in_topk and not oracle_in_oof:
        return "oracle_missing_shortlist"
    if int(proposal_code) == oracle_code:
        return "proposal_exact"
    return "proposal_wrong"


def oracle_rank_for_proposal(
    key: tuple[int, int, int, int],
    rows: list[dict[str, Any]],
    group_to_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    scores: np.ndarray,
) -> int | None:
    oracle = bottleneck.oracle_row(rows)
    nearest = bottleneck.nearest_row(rows)
    if int(oracle["code_index"]) == int(nearest["code_index"]):
        return None
    ranked = [
        idx
        for idx in group_to_indices.get(key, [])
        if not int(rows_all[idx].get("is_nearest", 0))
    ]
    ranked.sort(
        key=lambda idx: (
            float(scores[idx]),
            -int(rows_all[idx].get("topk_rank", 999)),
            -int(rows_all[idx]["code_index"]),
        ),
        reverse=True,
    )
    for rank, idx in enumerate(ranked, start=1):
        if int(rows_all[idx]["code_index"]) == int(oracle["code_index"]):
            return rank
    return None


def best_zero_bad_threshold(
    labels: list[str],
    scores: list[float],
) -> dict[str, Any]:
    thresholds = sorted(set(scores), reverse=True)
    thresholds = [float("inf")] + thresholds + [min(thresholds) - 1.0 if thresholds else -1.0]
    best = None
    for threshold in thresholds:
        accepted = [label for label, score in zip(labels, scores, strict=True) if score >= threshold]
        bad = sum(label in {"false", "wrong"} for label in accepted)
        exact = sum(label == "exact" for label in accepted)
        if bad > 0:
            continue
        candidate = (-exact, -len(accepted), threshold)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return {"threshold": None, "accepted_exact": 0, "accepted_total": 0}
    threshold = float(best[2])
    accepted = [label for label, score in zip(labels, scores, strict=True) if score >= threshold]
    return {
        "threshold": json_float(threshold),
        "accepted_exact": int(sum(label == "exact" for label in accepted)),
        "accepted_total": int(len(accepted)),
    }


def audit_pair(
    proposer_name: str,
    risk_name: str,
    risk_model: Any,
    proposal_scores: np.ndarray,
    train_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    val_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    train_group_indices: dict[tuple[int, int, int, int], list[int]],
    val_group_indices: dict[tuple[int, int, int, int], list[int]],
    rows_all: list[dict[str, Any]],
    oof_scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
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
    tuned = n120.tune_risk_threshold(train_groups, train_proposals, train_scores, false_wrong_budget=0)
    threshold = float(tuned["threshold"])
    selected = n120.select_with_risk(val_groups, val_proposals, val_scores, threshold)
    metrics = bottleneck.evaluate_selection(val_groups, selected)

    counts = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    labels: list[str] = []
    score_values: list[float] = []
    proposal_exact_scores: list[float] = []
    proposal_wrong_scores: list[float] = []
    no_oracle_scores: list[float] = []
    case_details: list[dict[str, Any]] = []
    for key, rows in sorted(val_groups.items()):
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        proposal_code = int(val_proposals.get(key, nearest_code))
        accepted = int(selected.get(key, nearest_code)) != nearest_code
        score = float(val_scores.get(key, float("-inf")))
        reason = proposal_reason(rows, proposal_code, oof_scores, topk_cap=topk_cap, score_count=score_count)
        source = str(rows[0].get("source", "unknown"))
        if reason == "proposal_exact":
            proposal_exact_scores.append(score)
            label = "exact"
            if accepted:
                counts["accepted_exact"] += 1
            else:
                counts["risk_abstained_exact"] += 1
        elif reason in {"proposal_wrong", "oracle_missing_shortlist"}:
            proposal_wrong_scores.append(score)
            label = "wrong" if oracle_code != nearest_code else "false"
            if oracle_code != nearest_code:
                counts[f"{reason}_oracle_change"] += 1
            if accepted:
                counts["accepted_wrong"] += int(oracle_code != nearest_code)
        else:
            no_oracle_scores.append(score)
            label = "false"
            if accepted:
                counts["accepted_false"] += 1
        labels.append(label)
        score_values.append(score)
        counts[reason] += 1
        counts["accepted"] += int(accepted)
        by_source[source][reason] += 1
        by_source[source]["accepted"] += int(accepted)
        if (oracle_code != nearest_code or accepted) and len(case_details) < 80:
            case_details.append(
                {
                    "group_key": list(key),
                    "source": source,
                    "reference": rows[0].get("reference"),
                    "nearest_code": nearest_code,
                    "oracle_code": oracle_code,
                    "proposal_code": proposal_code,
                    "selected_code": int(selected.get(key, nearest_code)),
                    "reason": reason,
                    "accepted": int(accepted),
                    "risk_score": json_float(score),
                    "threshold": json_float(threshold),
                    "oracle_rank_nonnearest": oracle_rank_for_proposal(key, rows, val_group_indices, rows_all, proposal_scores),
                    "oracle_topk_rank": int(oracle.get("topk_rank", 999)),
                    "proposal_tesseract_delta_vs_nearest": int(
                        bottleneck.row_for_code(rows, proposal_code).get("tesseract_delta_vs_nearest", 0)
                    )
                    if proposal_code != nearest_code
                    else 0,
                    "oracle_tesseract_delta_vs_nearest": int(oracle.get("tesseract_delta_vs_nearest", 0)),
                }
            )
    return {
        "proposer": proposer_name,
        "risk": risk_name,
        "threshold": json_float(threshold),
        "policy_metrics": metrics,
        "miss_breakdown": {key: int(value) for key, value in counts.items()},
        "by_source": {name: {k: int(v) for k, v in counter.items()} for name, counter in sorted(by_source.items())},
        "score_stats": {
            "proposal_exact": stats(proposal_exact_scores),
            "proposal_wrong_or_missing": stats(proposal_wrong_scores),
            "no_oracle": stats(no_oracle_scores),
        },
        "best_zero_bad_threshold_on_val_scores": best_zero_bad_threshold(labels, score_values),
        "case_details": case_details,
    }


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
        "Diagnostic audit of N120 two-stage risk misses. No `.oscr` stream is exported.",
        "",
        "## Summary",
        "",
        "| proposer | risk | policy | accepted exact | risk-abstained exact | proposal wrong | oracle missing | val zero-bad exact ceiling |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["audits"]:
        miss = row["miss_breakdown"]
        ceiling = row["best_zero_bad_threshold_on_val_scores"]
        lines.append(
            f"| {row['proposer']} | {row['risk']} | {compact(row['policy_metrics'])} | "
            f"{miss.get('accepted_exact', 0)} | {miss.get('risk_abstained_exact', 0)} | "
            f"{miss.get('proposal_wrong_oracle_change', 0)} | {miss.get('oracle_missing_shortlist_oracle_change', 0)} | "
            f"{ceiling.get('accepted_exact', 0)} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-oof-two-stage-risk-miss-audit",
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
            "    path: scripts/analyze_top8_oof_two_stage_risk_misses.py",
            f"    sha256: {bottleneck.sha256_file(Path(__file__))}",
            "  trainer_source:",
            "    path: scripts/train_top8_oof_shortlist_two_stage_risk.py",
            f"    sha256: {bottleneck.sha256_file(ROOT / 'scripts/train_top8_oof_shortlist_two_stage_risk.py')}",
            "conclusion: diagnostic miss audit; no counted oscr export",
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
    audits: list[dict[str, Any]] = []
    for prop_name, prop_model in n120.proposer_grid(args.random_state).items():
        prop_model.fit(x[train_non_indices], y_prop[train_non_indices])
        prop_scores = n118.score_model(prop_model, x)
        for risk_name, risk_model in n120.risk_grid(args.random_state).items():
            audits.append(
                audit_pair(
                    prop_name,
                    risk_name,
                    risk_model,
                    prop_scores,
                    train_groups,
                    val_groups,
                    train_group_indices,
                    val_group_indices,
                    rows_all,
                    oof_scores,
                    topk_cap=args.topk_cap,
                    score_count=args.score_count,
                )
            )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_two_stage_miss_audit_not_promoted",
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
        },
        "audits": audits,
        "hashes": {
            "script": bottleneck.sha256_file(Path(__file__)),
            "trainer_source": bottleneck.sha256_file(ROOT / "scripts/train_top8_oof_shortlist_two_stage_risk.py"),
        },
        "interpretation": (
            "N120's safe gain is limited by two bottlenecks: the proposal stage still fails many oracle groups, "
            "and the risk gate abstains some exact proposals to keep false/wrong at zero. Use this audit before "
            "adding hard-positive mining, richer proposal evidence, or a monotone/conformal abstention rule."
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
                "summary": [
                    {
                        "proposer": row["proposer"],
                        "risk": row["risk"],
                        "policy": compact(row["policy_metrics"]),
                        "miss_breakdown": row["miss_breakdown"],
                        "zero_bad_val_ceiling": row["best_zero_bad_threshold_on_val_scores"],
                    }
                    for row in audits
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
