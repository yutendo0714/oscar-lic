#!/usr/bin/env python3
"""Tune a score-consensus assignment selector on train groups and evaluate on clean val groups."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def load_score_rows(paths: list[Path]) -> dict[tuple[int, int, int, int], list[dict]]:
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
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


def annotate(group: dict, selected_code: int, selected_row: dict, policy_features: dict) -> dict:
    nearest = group["nearest"]
    oracle = group["oracle"]
    selected_changed = selected_code != int(nearest["code_index"])
    oracle_changed = int(oracle["code_index"]) != int(nearest["code_index"])
    out = dict(selected_row)
    out.update(policy_features)
    out.update(
        {
            "nearest_code": int(nearest["code_index"]),
            "oracle_code": int(oracle["code_index"]),
            "selected_changed": int(selected_changed),
            "oracle_changed": int(oracle_changed),
            "selected_is_oracle": int(selected_code == int(oracle["code_index"])),
        }
    )
    return out


def summarize(selected: list[dict]) -> dict:
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


def build_group(raw_rows: list[dict], parseq_safe: bool) -> dict:
    by_code: dict[int, list[dict]] = defaultdict(list)
    for row in raw_rows:
        by_code[int(row["code_index"])].append(row)
    voter_ids = sorted({row["_voter_id"] for row in raw_rows})
    nearest_code = next(int(row["code_index"]) for row in raw_rows if int(row["is_nearest"]))
    oracle_code = next(int(row["code_index"]) for row in raw_rows if int(row["label_assignment_oracle_choice"]))
    nearest_row = by_code[nearest_code][0]
    oracle_row = by_code[oracle_code][0]
    nearest_scores = {
        row["_voter_id"]: float(row["score"])
        for row in by_code[nearest_code]
    }
    candidate_infos = []
    top_vote_counts = Counter()
    for voter_id in voter_ids:
        voter_rows = []
        for code, rows in by_code.items():
            row = next((item for item in rows if item["_voter_id"] == voter_id), None)
            if row is None or code == nearest_code:
                continue
            if parseq_safe and int(row["parseq_delta_vs_nearest"]) > 0:
                continue
            voter_rows.append(row)
        if voter_rows:
            best = max(voter_rows, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))
            top_vote_counts[int(best["code_index"])] += 1
    for code, rows in by_code.items():
        if code == nearest_code:
            continue
        template = rows[0]
        if parseq_safe and int(template["parseq_delta_vs_nearest"]) > 0:
            continue
        scores = []
        margins = []
        for voter_id in voter_ids:
            row = next((item for item in rows if item["_voter_id"] == voter_id), None)
            if row is None:
                continue
            score = float(row["score"])
            scores.append(score)
            margins.append(score - nearest_scores.get(voter_id, 0.0))
        if not scores:
            continue
        arr = np.asarray(scores, dtype=np.float64)
        marr = np.asarray(margins, dtype=np.float64)
        candidate_infos.append(
            {
                "code": code,
                "row": template,
                "score_mean": float(arr.mean()),
                "score_min": float(arr.min()),
                "score_max": float(arr.max()),
                "score_std": float(arr.std(ddof=0)),
                "margin_mean": float(marr.mean()),
                "margin_min": float(marr.min()),
                "vote_fraction": float(top_vote_counts[code] / max(len(voter_ids), 1)),
                "vote_count": int(top_vote_counts[code]),
            }
        )
    best_info = None
    if candidate_infos:
        best_info = max(
            candidate_infos,
            key=lambda item: (
                item["score_mean"],
                item["vote_fraction"],
                item["margin_mean"],
                -int(item["row"]["topk_rank"]),
            ),
        )
    return {
        "key": group_key(nearest_row),
        "nearest": nearest_row,
        "oracle": oracle_row,
        "best": best_info,
        "voter_count": len(voter_ids),
    }


def select_group(group: dict, thresholds: dict) -> dict:
    best = group["best"]
    if best is None:
        return annotate(group, int(group["nearest"]["code_index"]), group["nearest"], {"selected_by_consensus": 0})
    take_change = (
        best["score_mean"] >= thresholds["score_mean"]
        and best["score_min"] >= thresholds["score_min"]
        and best["margin_mean"] >= thresholds["margin_mean"]
        and best["margin_min"] >= thresholds["margin_min"]
        and best["vote_fraction"] >= thresholds["vote_fraction"]
    )
    selected = best["row"] if take_change else group["nearest"]
    return annotate(
        group,
        int(selected["code_index"]),
        selected,
        {
            "selected_by_consensus": int(take_change),
            "consensus_score_mean": float(best["score_mean"]),
            "consensus_score_min": float(best["score_min"]),
            "consensus_margin_mean": float(best["margin_mean"]),
            "consensus_margin_min": float(best["margin_min"]),
            "consensus_vote_fraction": float(best["vote_fraction"]),
            "consensus_vote_count": int(best["vote_count"]),
        },
    )


def evaluate(groups: list[dict], thresholds: dict) -> dict:
    return summarize([select_group(group, thresholds) for group in groups])


def threshold_values(groups: list[dict], name: str, default_min: float) -> list[float]:
    values = sorted({float(group["best"][name]) for group in groups if group["best"] is not None})
    if not values:
        return [default_min]
    quantiles = np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0.0, 1.0, 11))
    candidates = sorted(set(float(value) for value in quantiles) | {default_min, min(values) - 1.0e-6, max(values) + 1.0e-6})
    return candidates


def tune_thresholds(groups: list[dict], false_change_max: int | None) -> tuple[dict, dict]:
    grids = {
        "score_mean": threshold_values(groups, "score_mean", 0.0),
        "score_min": threshold_values(groups, "score_min", 0.0),
        "margin_mean": threshold_values(groups, "margin_mean", -1.0),
        "margin_min": threshold_values(groups, "margin_min", -1.0),
        "vote_fraction": sorted({0.0, 0.25, 0.5, 0.75, 1.0}),
    }
    best_thresholds = None
    best_metrics = None
    best_key = None
    for score_mean in grids["score_mean"]:
        for score_min in grids["score_min"]:
            for margin_mean in grids["margin_mean"]:
                for margin_min in grids["margin_min"]:
                    for vote_fraction in grids["vote_fraction"]:
                        thresholds = {
                            "score_mean": float(score_mean),
                            "score_min": float(score_min),
                            "margin_mean": float(margin_mean),
                            "margin_min": float(margin_min),
                            "vote_fraction": float(vote_fraction),
                        }
                        metrics = evaluate(groups, thresholds)
                        if false_change_max is not None and metrics["false_change_groups"] > false_change_max:
                            continue
                        key = (
                            metrics["parseq_worsen_groups"],
                            metrics["tesseract_delta_sum"],
                            metrics["tesseract_worsen_groups"],
                            metrics["wrong_change_groups"],
                            metrics["false_change_groups"],
                            metrics["changed_groups"],
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_thresholds = thresholds
                            best_metrics = metrics
    if best_thresholds is None:
        # Fall back to strictest parseq-safe selector if the false-change constraint is impossible.
        best_thresholds = {
            "score_mean": max(grids["score_mean"]),
            "score_min": max(grids["score_min"]),
            "margin_mean": max(grids["margin_mean"]),
            "margin_min": max(grids["margin_min"]),
            "vote_fraction": 1.0,
        }
        best_metrics = evaluate(groups, best_thresholds)
    return best_thresholds, best_metrics


def select_all(groups: list[dict], mode: str) -> list[dict]:
    rows = []
    for group in groups:
        if mode == "nearest":
            row = group["nearest"]
        elif mode == "oracle":
            row = group["oracle"]
        elif mode == "best_mean":
            row = group["best"]["row"] if group["best"] is not None else group["nearest"]
        else:
            raise ValueError(mode)
        rows.append(annotate(group, int(row["code_index"]), row, {"selected_by_consensus": int(mode == "best_mean")}))
    return rows


def write_policy(path: Path, selected: list[dict], thresholds: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(selected, key=lambda item: (int(item["real_seed"]), int(item["source_index"]), int(item["candidate_index"]))):
            out = {
                "source_index": int(row["source_index"]),
                "candidate_index": int(row["candidate_index"]),
                "assignment_code_index": int(row["code_index"]),
                "code_index": int(row["code_index"]),
                "nearest_code": int(row["nearest_code"]),
                "topk_rank": int(row["topk_rank"]),
                "selected_by_policy": True,
                "selected_by_consensus": int(row.get("selected_by_consensus", 0)),
                "selector_score": float(row.get("consensus_score_mean", row.get("score", 0.0))),
                "policy": "score_consensus_train_tuned",
                "policy_thresholds": thresholds,
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
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--split-key-index", type=int, default=1, choices=[0, 1])
    parser.add_argument("--parseq-safe", action="store_true")
    parser.add_argument("--false-change-rate-max", type=float, default=0.02)
    parser.add_argument("--policy-output", type=Path)
    args = parser.parse_args()

    raw_groups = load_score_rows(args.score)
    groups = [build_group(rows, parseq_safe=args.parseq_safe) for _, rows in sorted(raw_groups.items())]
    train_groups = [group for group in groups if group["key"][args.split_key_index] != args.val_seed]
    val_groups = [group for group in groups if group["key"][args.split_key_index] == args.val_seed]
    false_change_max = max(0, int(round(len(train_groups) * args.false_change_rate_max)))
    thresholds, train_metrics = tune_thresholds(train_groups, false_change_max=false_change_max)
    selected_val = [select_group(group, thresholds) for group in val_groups]
    if args.policy_output is not None:
        write_policy(args.policy_output, selected_val, thresholds)
    output = {
        "description": "Score-consensus selector: tune score/margin/vote thresholds on train groups and evaluate on clean val groups.",
        "scores": [str(path) for path in args.score],
        "config": {
            "val_seed": int(args.val_seed),
            "split_key_index": int(args.split_key_index),
            "parseq_safe": bool(args.parseq_safe),
            "false_change_rate_max": float(args.false_change_rate_max),
            "false_change_max": int(false_change_max),
        },
        "thresholds": thresholds,
        "train": {
            "nearest": summarize(select_all(train_groups, "nearest")),
            "best_mean": summarize(select_all(train_groups, "best_mean")),
            "oracle": summarize(select_all(train_groups, "oracle")),
            "consensus_train_tuned": train_metrics,
        },
        "val": {
            "nearest": summarize(select_all(val_groups, "nearest")),
            "best_mean": summarize(select_all(val_groups, "best_mean")),
            "oracle": summarize(select_all(val_groups, "oracle")),
            "consensus_train_tuned": summarize(selected_val),
        },
        "aggregate": {
            "train_tesseract_delta_sum": int(train_metrics["tesseract_delta_sum"]),
            "train_false_change_groups": int(train_metrics["false_change_groups"]),
            "val_tesseract_delta_sum": int(summarize(selected_val)["tesseract_delta_sum"]),
            "val_changed_groups": int(summarize(selected_val)["changed_groups"]),
            "val_false_change_groups": int(summarize(selected_val)["false_change_groups"]),
            "val_wrong_change_groups": int(summarize(selected_val)["wrong_change_groups"]),
            "val_tesseract_worsen_groups": int(summarize(selected_val)["tesseract_worsen_groups"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"thresholds": thresholds, "val": output["val"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
