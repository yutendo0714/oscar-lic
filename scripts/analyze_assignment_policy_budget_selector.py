#!/usr/bin/env python3
"""Tune a budgeted top-k/no-op assignment selector on score JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def load_score_rows(paths: list[Path]) -> dict[tuple[int, int, int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for file_index, path in enumerate(paths):
        for row in read_jsonl(path):
            out = dict(row)
            out["_voter_id"] = f"{file_index}:{int(row['model_seed'])}"
            groups[group_key(out)].append(out)
    return dict(groups)


def error_type(selected_changed: bool, selected_is_oracle: bool, oracle_changed: bool) -> str:
    if selected_is_oracle and oracle_changed:
        return "exact_oracle"
    if selected_is_oracle and not oracle_changed:
        return "correct_nearest"
    if selected_changed and not oracle_changed:
        return "false_change"
    if (not selected_changed) and oracle_changed:
        return "missed_oracle"
    return "wrong_change"


def summarize(selected: list[dict[str, Any]]) -> dict[str, int]:
    errors = Counter(
        error_type(
            bool(row["selected_changed"]),
            bool(row["selected_is_oracle"]),
            bool(row["oracle_changed"]),
        )
        for row in selected
    )
    return {
        "groups": len(selected),
        "changed_groups": sum(int(row["selected_changed"]) for row in selected),
        "parseq_delta_sum": sum(int(row["parseq_delta_vs_nearest"]) for row in selected),
        "tesseract_delta_sum": sum(int(row["tesseract_delta_vs_nearest"]) for row in selected),
        "parseq_worsen_groups": sum(int(row["parseq_delta_vs_nearest"] > 0) for row in selected),
        "tesseract_improve_groups": sum(int(row["tesseract_delta_vs_nearest"] < 0) for row in selected),
        "tesseract_worsen_groups": sum(int(row["tesseract_delta_vs_nearest"] > 0) for row in selected),
        "oracle_change_groups": sum(int(row["oracle_changed"]) for row in selected),
        "exact_oracle_groups": int(errors["exact_oracle"]),
        "missed_oracle_groups": int(errors["missed_oracle"]),
        "false_change_groups": int(errors["false_change"]),
        "wrong_change_groups": int(errors["wrong_change"]),
        "correct_nearest_groups": int(errors["correct_nearest"]),
    }


def annotate(group: dict[str, Any], selected: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    nearest_code = int(group["nearest"]["code_index"])
    oracle_code = int(group["oracle"]["code_index"])
    selected_code = int(selected["code_index"])
    selected_changed = selected_code != nearest_code
    oracle_changed = oracle_code != nearest_code
    out = dict(selected)
    out.update(policy)
    out.update(
        {
            "nearest_code": nearest_code,
            "oracle_code": oracle_code,
            "selected_changed": int(selected_changed),
            "oracle_changed": int(oracle_changed),
            "selected_is_oracle": int(selected_code == oracle_code),
        }
    )
    return out


def scalar_score(info: dict[str, Any], metric: str) -> float:
    if metric == "score_mean":
        return float(info["score_mean"])
    if metric == "score_min":
        return float(info["score_min"])
    if metric == "margin_mean":
        return float(info["margin_mean"])
    if metric == "margin_min":
        return float(info["margin_min"])
    if metric == "score_x_group":
        return float(info["score_mean"]) * float(info["group_prob_mean"])
    if metric == "margin_x_group":
        return float(info["margin_mean"]) * float(info["group_prob_mean"])
    if metric == "vote_then_score":
        return float(info["vote_fraction"]) + 0.001 * float(info["score_mean"])
    raise ValueError(f"unknown rank metric: {metric}")


def build_group(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_code: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_code[int(row["code_index"])].append(row)
    voter_ids = sorted({str(row["_voter_id"]) for row in raw_rows})
    nearest_code = next(int(row["code_index"]) for row in raw_rows if int(row["is_nearest"]))
    oracle_code = next(int(row["code_index"]) for row in raw_rows if int(row["label_assignment_oracle_choice"]))
    nearest = by_code[nearest_code][0]
    oracle = by_code[oracle_code][0]
    nearest_scores = {
        str(row["_voter_id"]): float(row["score"])
        for row in by_code[nearest_code]
    }
    top_vote_counts: Counter[int] = Counter()
    for voter_id in voter_ids:
        voter_rows = [
            row
            for code, rows in by_code.items()
            if code != nearest_code
            for row in rows
            if str(row["_voter_id"]) == voter_id
        ]
        if voter_rows:
            best = max(voter_rows, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))
            top_vote_counts[int(best["code_index"])] += 1

    candidate_infos: list[dict[str, Any]] = []
    for code, rows in by_code.items():
        if code == nearest_code:
            continue
        template = rows[0]
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        group_probs = np.asarray([float(row.get("group_change_prob", 1.0)) for row in rows], dtype=np.float64)
        margins = np.asarray(
            [float(row["score"]) - nearest_scores.get(str(row["_voter_id"]), 0.0) for row in rows],
            dtype=np.float64,
        )
        candidate_infos.append(
            {
                "code": int(code),
                "row": template,
                "score_mean": float(scores.mean()),
                "score_min": float(scores.min()),
                "score_std": float(scores.std(ddof=0)),
                "margin_mean": float(margins.mean()),
                "margin_min": float(margins.min()),
                "group_prob_mean": float(group_probs.mean()),
                "group_prob_min": float(group_probs.min()),
                "vote_fraction": float(top_vote_counts[int(code)] / max(len(voter_ids), 1)),
                "vote_count": int(top_vote_counts[int(code)]),
                "topk_rank": int(template["topk_rank"]),
            }
        )
    return {
        "key": group_key(nearest),
        "nearest": nearest,
        "oracle": oracle,
        "candidates": candidate_infos,
        "voter_count": len(voter_ids),
    }


def best_candidate(group: dict[str, Any], top_m: int, metric: str, parseq_safe: bool) -> dict[str, Any] | None:
    candidates = []
    for info in group["candidates"]:
        row = info["row"]
        if int(row["topk_rank"]) > top_m:
            continue
        if parseq_safe and int(row["parseq_delta_vs_nearest"]) > 0:
            continue
        out = dict(info)
        out["policy_score"] = scalar_score(info, metric)
        candidates.append(out)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["policy_score"]),
            float(item["vote_fraction"]),
            float(item["margin_mean"]),
            -int(item["topk_rank"]),
        ),
    )


def rank_group_candidates(
    groups: list[dict[str, Any]], top_m: int, metric: str, parseq_safe: bool
) -> list[dict[str, Any]]:
    ranked = []
    for group in groups:
        best = best_candidate(group, top_m=top_m, metric=metric, parseq_safe=parseq_safe)
        if best is None:
            continue
        ranked.append({"group": group, "best": best})
    ranked.sort(
        key=lambda item: (
            float(item["best"]["policy_score"]),
            float(item["best"]["vote_fraction"]),
            float(item["best"]["margin_mean"]),
            -int(item["best"]["topk_rank"]),
        ),
        reverse=True,
    )
    return ranked


def select_groups(
    groups: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    budget_count: int | None = None,
    threshold: float | None = None,
    policy_name: str = "budgeted",
) -> list[dict[str, Any]]:
    selected_by_key: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    if threshold is not None:
        for item in ranked:
            if float(item["best"]["policy_score"]) >= threshold:
                selected_by_key[item["group"]["key"]] = item["best"]
    elif budget_count is not None and budget_count > 0:
        for item in ranked[:budget_count]:
            selected_by_key[item["group"]["key"]] = item["best"]

    selected = []
    for group in groups:
        best = selected_by_key.get(group["key"])
        if best is None:
            selected.append(
                annotate(
                    group,
                    group["nearest"],
                    {
                        "selected_by_budget": 0,
                        "policy": policy_name,
                        "policy_score": 0.0,
                    },
                )
            )
            continue
        selected.append(
            annotate(
                group,
                best["row"],
                {
                    "selected_by_budget": 1,
                    "policy": policy_name,
                    "policy_score": float(best["policy_score"]),
                    "score_mean": float(best["score_mean"]),
                    "margin_mean": float(best["margin_mean"]),
                    "vote_fraction": float(best["vote_fraction"]),
                    "group_prob_mean": float(best["group_prob_mean"]),
                },
            )
        )
    return selected


def select_all(groups: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    out = []
    for group in groups:
        if mode == "nearest":
            row = group["nearest"]
        elif mode == "oracle":
            row = group["oracle"]
        else:
            raise ValueError(mode)
        out.append(annotate(group, row, {"selected_by_budget": int(mode == "oracle"), "policy": mode}))
    return out


def feasible(metrics: dict[str, int], limits: dict[str, int]) -> bool:
    return all(int(metrics[name]) <= int(limit) for name, limit in limits.items())


def tune_budget(
    train_groups: list[dict[str, Any]],
    top_m_values: list[int],
    metrics: list[str],
    parseq_safe: bool,
    limits: dict[str, int],
    max_budget_count: int | None,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_key = None
    for top_m in top_m_values:
        for metric in metrics:
            ranked = rank_group_candidates(train_groups, top_m=top_m, metric=metric, parseq_safe=parseq_safe)
            max_count = len(ranked) if max_budget_count is None else min(max_budget_count, len(ranked))
            for budget_count in range(max_count + 1):
                selected = select_groups(
                    train_groups,
                    ranked,
                    budget_count=budget_count,
                    policy_name="train_budget_count",
                )
                metrics_map = summarize(selected)
                if not feasible(metrics_map, limits):
                    continue
                key = (
                    int(metrics_map["parseq_worsen_groups"]),
                    int(metrics_map["tesseract_delta_sum"]),
                    int(metrics_map["tesseract_worsen_groups"]),
                    int(metrics_map["wrong_change_groups"]),
                    int(metrics_map["false_change_groups"]),
                    int(metrics_map["missed_oracle_groups"]),
                    int(metrics_map["changed_groups"]),
                )
                if best_key is None or key < best_key:
                    threshold = float("inf")
                    if budget_count > 0:
                        threshold = float(ranked[budget_count - 1]["best"]["policy_score"])
                    best_key = key
                    best = {
                        "top_m": int(top_m),
                        "rank_metric": metric,
                        "budget_count": int(budget_count),
                        "budget_fraction": float(budget_count / max(len(train_groups), 1)),
                        "threshold": threshold,
                        "train_metrics": metrics_map,
                    }
    if best is None:
        nearest_metrics = summarize(select_all(train_groups, "nearest"))
        best = {
            "top_m": int(top_m_values[0]),
            "rank_metric": metrics[0],
            "budget_count": 0,
            "budget_fraction": 0.0,
            "threshold": float("inf"),
            "train_metrics": nearest_metrics,
            "fallback": "nearest_only_no_feasible_budget",
        }
    return best


def apply_tuned(
    groups: list[dict[str, Any]], tuned: dict[str, Any], parseq_safe: bool, mode: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ranked = rank_group_candidates(
        groups,
        top_m=int(tuned["top_m"]),
        metric=str(tuned["rank_metric"]),
        parseq_safe=parseq_safe,
    )
    if mode == "threshold":
        selected = select_groups(
            groups,
            ranked,
            threshold=float(tuned["threshold"]),
            policy_name="budget_threshold_transfer",
        )
    elif mode == "fraction":
        budget_count = int(round(float(tuned["budget_fraction"]) * len(groups)))
        selected = select_groups(
            groups,
            ranked,
            budget_count=budget_count,
            policy_name="budget_fraction_transfer",
        )
    else:
        raise ValueError(mode)
    return selected, summarize(selected)


def write_policy(path: Path, rows: list[dict[str, Any]], tuned: dict[str, Any], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (int(item["real_seed"]), int(item["source_index"]), int(item["candidate_index"]))):
            out = {
                "source_index": int(row["source_index"]),
                "candidate_index": int(row["candidate_index"]),
                "assignment_code_index": int(row["code_index"]),
                "code_index": int(row["code_index"]),
                "nearest_code": int(row["nearest_code"]),
                "topk_rank": int(row["topk_rank"]),
                "selected_by_policy": bool(int(row["selected_by_budget"])),
                "selector_score": float(row.get("policy_score", 0.0)),
                "policy": str(row.get("policy", mode)),
                "policy_transfer_mode": mode,
                "policy_tuned_config": tuned,
                "seed": int(row["seed"]),
                "real_seed": int(row.get("real_seed", row["seed"])),
                "parseq_delta_vs_nearest": int(row["parseq_delta_vs_nearest"]),
                "tesseract_delta_vs_nearest": int(row["tesseract_delta_vs_nearest"]),
                "label_assignment_oracle_choice": int(row["label_assignment_oracle_choice"]),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-output-prefix", type=Path)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--split-key-index", type=int, default=1, choices=[0, 1])
    parser.add_argument("--top-m", type=int, action="append")
    parser.add_argument(
        "--rank-metric",
        action="append",
    )
    parser.add_argument("--parseq-safe", action="store_true")
    parser.add_argument("--false-change-rate-max", type=float, default=0.0)
    parser.add_argument("--wrong-change-rate-max", type=float, default=0.0)
    parser.add_argument("--tesseract-worsen-rate-max", type=float, default=0.0)
    parser.add_argument("--parseq-worsen-max", type=int, default=0)
    parser.add_argument("--max-budget-rate", type=float, default=1.0)
    args = parser.parse_args()

    raw_groups = load_score_rows(args.score)
    groups = [build_group(rows) for _, rows in sorted(raw_groups.items())]
    train_groups = [group for group in groups if int(group["key"][args.split_key_index]) != int(args.val_seed)]
    val_groups = [group for group in groups if int(group["key"][args.split_key_index]) == int(args.val_seed)]
    limits = {
        "false_change_groups": max(0, int(round(len(train_groups) * float(args.false_change_rate_max)))),
        "wrong_change_groups": max(0, int(round(len(train_groups) * float(args.wrong_change_rate_max)))),
        "tesseract_worsen_groups": max(0, int(round(len(train_groups) * float(args.tesseract_worsen_rate_max)))),
        "parseq_worsen_groups": int(args.parseq_worsen_max),
    }
    max_budget_count = int(round(len(train_groups) * float(args.max_budget_rate)))
    tuned = tune_budget(
        train_groups,
        top_m_values=sorted(set(int(value) for value in (args.top_m or [1, 2, 4, 8]))),
        metrics=list(
            dict.fromkeys(
                args.rank_metric
                or ["score_mean", "margin_mean", "score_x_group", "margin_x_group", "vote_then_score"]
            )
        ),
        parseq_safe=bool(args.parseq_safe),
        limits=limits,
        max_budget_count=max_budget_count,
    )
    threshold_rows, threshold_metrics = apply_tuned(val_groups, tuned, parseq_safe=bool(args.parseq_safe), mode="threshold")
    fraction_rows, fraction_metrics = apply_tuned(val_groups, tuned, parseq_safe=bool(args.parseq_safe), mode="fraction")
    if args.policy_output_prefix is not None:
        write_policy(args.policy_output_prefix.with_suffix(".threshold.jsonl"), threshold_rows, tuned, "threshold")
        write_policy(args.policy_output_prefix.with_suffix(".fraction.jsonl"), fraction_rows, tuned, "fraction")

    output = {
        "description": "Budgeted assignment selector: tune top-k candidate metric and change budget on train groups, then transfer by score threshold or budget fraction to val groups.",
        "scores": [str(path) for path in args.score],
        "config": {
            "val_seed": int(args.val_seed),
            "split_key_index": int(args.split_key_index),
            "parseq_safe": bool(args.parseq_safe),
            "top_m": sorted(set(int(value) for value in (args.top_m or [1, 2, 4, 8]))),
            "rank_metric": list(
                dict.fromkeys(
                    args.rank_metric
                    or ["score_mean", "margin_mean", "score_x_group", "margin_x_group", "vote_then_score"]
                )
            ),
            "limits": limits,
            "max_budget_rate": float(args.max_budget_rate),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
        },
        "tuned": tuned,
        "train": {
            "nearest": summarize(select_all(train_groups, "nearest")),
            "oracle": summarize(select_all(train_groups, "oracle")),
            "budget_train_tuned": tuned["train_metrics"],
        },
        "val": {
            "nearest": summarize(select_all(val_groups, "nearest")),
            "oracle": summarize(select_all(val_groups, "oracle")),
            "budget_threshold_transfer": threshold_metrics,
            "budget_fraction_transfer": fraction_metrics,
        },
        "aggregate": {
            "train_tesseract_delta_sum": int(tuned["train_metrics"]["tesseract_delta_sum"]),
            "train_false_change_groups": int(tuned["train_metrics"]["false_change_groups"]),
            "val_threshold_tesseract_delta_sum": int(threshold_metrics["tesseract_delta_sum"]),
            "val_threshold_changed_groups": int(threshold_metrics["changed_groups"]),
            "val_threshold_false_change_groups": int(threshold_metrics["false_change_groups"]),
            "val_threshold_wrong_change_groups": int(threshold_metrics["wrong_change_groups"]),
            "val_threshold_tesseract_worsen_groups": int(threshold_metrics["tesseract_worsen_groups"]),
            "val_fraction_tesseract_delta_sum": int(fraction_metrics["tesseract_delta_sum"]),
            "val_fraction_changed_groups": int(fraction_metrics["changed_groups"]),
            "val_fraction_false_change_groups": int(fraction_metrics["false_change_groups"]),
            "val_fraction_wrong_change_groups": int(fraction_metrics["wrong_change_groups"]),
            "val_fraction_tesseract_worsen_groups": int(fraction_metrics["tesseract_worsen_groups"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tuned": tuned, "val": output["val"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
