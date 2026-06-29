#!/usr/bin/env python3
"""Evaluate a high-precision change gate paired with alternate code reranker scores."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


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


def summarize_selected(selected_rows: list[dict], *, description: str) -> dict:
    errors = Counter()
    for row in selected_rows:
        errors[
            error_type(
                bool(row["selected_changed"]),
                bool(row["selected_is_oracle"]),
                bool(row["oracle_changed"]),
            )
        ] += 1
    return {
        "description": description,
        "groups": len(selected_rows),
        "changed_groups": sum(int(row["selected_changed"]) for row in selected_rows),
        "parseq_delta_sum": sum(int(row["parseq_delta_vs_nearest"]) for row in selected_rows),
        "tesseract_delta_sum": sum(int(row["tesseract_delta_vs_nearest"]) for row in selected_rows),
        "parseq_worsen_groups": sum(int(row["parseq_delta_vs_nearest"] > 0) for row in selected_rows),
        "tesseract_improve_groups": sum(int(row["tesseract_delta_vs_nearest"] < 0) for row in selected_rows),
        "tesseract_worsen_groups": sum(int(row["tesseract_delta_vs_nearest"] > 0) for row in selected_rows),
        "oracle_change_groups": sum(int(row["oracle_changed"]) for row in selected_rows),
        "exact_oracle_groups": int(errors["exact_oracle"]),
        "missed_oracle_groups": int(errors["missed_oracle"]),
        "false_change_groups": int(errors["false_change"]),
        "wrong_change_groups": int(errors["wrong_change"]),
        "correct_nearest_groups": int(errors["correct_nearest"]),
    }


def annotate_choice(group_rows: list[dict], selected: dict) -> dict:
    nearest = next(row for row in group_rows if int(row["is_nearest"]))
    oracle = next(row for row in group_rows if int(row["label_assignment_oracle_choice"]))
    selected_changed = int(selected["code_index"]) != int(nearest["code_index"])
    oracle_changed = int(oracle["code_index"]) != int(nearest["code_index"])
    out = dict(selected)
    out.update(
        {
            "nearest_code": int(nearest["code_index"]),
            "oracle_code": int(oracle["code_index"]),
            "selected_changed": int(selected_changed),
            "oracle_changed": int(oracle_changed),
            "selected_is_oracle": int(int(selected["code_index"]) == int(oracle["code_index"])),
        }
    )
    return out


def choose_best_score(group_rows: list[dict], *, force_change: bool, parseq_safe: bool) -> dict:
    candidates = []
    for row in group_rows:
        if force_change and int(row["is_nearest"]):
            continue
        if parseq_safe and int(row["parseq_delta_vs_nearest"]) > 0:
            continue
        candidates.append(row)
    if not candidates:
        candidates = [row for row in group_rows if int(row["is_nearest"])]
    return max(candidates, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))


def build_model_groups(score_rows: list[dict]) -> dict[int, dict[tuple[int, int, int, int], list[dict]]]:
    grouped: dict[int, dict[tuple[int, int, int, int], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in score_rows:
        grouped[int(row["model_seed"])][group_key(row)].append(row)
    return grouped


def build_ensemble_groups(score_rows: list[dict]) -> dict[tuple[int, int, int, int], list[dict]]:
    values: dict[tuple[tuple[int, int, int, int], int], list[dict]] = defaultdict(list)
    for row in score_rows:
        values[(group_key(row), int(row["code_index"]))].append(row)
    grouped: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for (key, _code), rows in values.items():
        base = dict(rows[0])
        base["score"] = mean(float(row["score"]) for row in rows)
        base["score_model_seeds"] = sorted({int(row["model_seed"]) for row in rows})
        base["score_count"] = len(rows)
        grouped[key].append(base)
    return grouped


def evaluate(
    gate_rows: list[dict],
    score_groups: dict[tuple[int, int, int, int], list[dict]],
    *,
    policy_name: str,
    parseq_safe: bool,
) -> dict:
    selected_include_nearest = []
    selected_force_change = []
    selected_gate_oracle = []
    selected_full_oracle = []
    missing = []
    gate_changed = {group_key(row): int(row["code_index"]) != int(row["nearest_code"]) for row in gate_rows}
    for gate_row in gate_rows:
        key = group_key(gate_row)
        rows = score_groups.get(key)
        if not rows:
            missing.append(key)
            continue
        rows = sorted(rows, key=lambda row: int(row["topk_rank"]))
        nearest = next(row for row in rows if int(row["is_nearest"]))
        oracle = next(row for row in rows if int(row["label_assignment_oracle_choice"]))
        if not gate_changed[key]:
            selected_include_nearest.append(annotate_choice(rows, nearest))
            selected_force_change.append(annotate_choice(rows, nearest))
            selected_gate_oracle.append(annotate_choice(rows, nearest))
        else:
            selected_include_nearest.append(
                annotate_choice(rows, choose_best_score(rows, force_change=False, parseq_safe=parseq_safe))
            )
            selected_force_change.append(
                annotate_choice(rows, choose_best_score(rows, force_change=True, parseq_safe=parseq_safe))
            )
            selected_gate_oracle.append(annotate_choice(rows, oracle))
        selected_full_oracle.append(annotate_choice(rows, oracle))
    return {
        "policy_name": policy_name,
        "parseq_safe": bool(parseq_safe),
        "missing_groups": [list(key) for key in missing],
        "gate_changed_groups": int(sum(gate_changed.values())),
        "include_nearest": summarize_selected(
            selected_include_nearest,
            description="If gate changes, take the reranker argmax over nearest plus candidates.",
        ),
        "force_change": summarize_selected(
            selected_force_change,
            description="If gate changes, force a non-nearest reranker choice when available.",
        ),
        "gate_conditioned_top4_oracle": summarize_selected(
            selected_gate_oracle,
            description="If gate changes, use the top4-label oracle; otherwise keep nearest. Diagnostic only.",
        ),
        "full_top4_oracle": summarize_selected(
            selected_full_oracle,
            description="Top4-label oracle over all groups. Diagnostic upper control only.",
        ),
    }


def summarize_gate_rows(gate_rows: list[dict]) -> dict:
    return {
        "groups": len(gate_rows),
        "changed_groups": sum(int(int(row["code_index"]) != int(row["nearest_code"])) for row in gate_rows),
        "parseq_delta_sum": sum(int(row["parseq_delta_vs_nearest"]) for row in gate_rows),
        "tesseract_delta_sum": sum(int(row["tesseract_delta_vs_nearest"]) for row in gate_rows),
        "parseq_worsen_groups": sum(int(row["parseq_delta_vs_nearest"]) > 0 for row in gate_rows),
        "tesseract_worsen_groups": sum(int(row["tesseract_delta_vs_nearest"]) > 0 for row in gate_rows),
    }


def aggregate_results(per_model: list[dict], ensemble: dict) -> dict:
    output = {}
    for mode in ["include_nearest", "force_change", "gate_conditioned_top4_oracle", "full_top4_oracle"]:
        rows = [row[mode] for row in per_model if mode in row]
        if rows:
            best = min(
                rows,
                key=lambda row: (
                    row["tesseract_delta_sum"],
                    row["parseq_delta_sum"],
                    row["tesseract_worsen_groups"],
                    row["false_change_groups"] + row["wrong_change_groups"],
                ),
            )
            output[f"{mode}_per_model_best"] = {
                "tesseract_delta_sum": int(best["tesseract_delta_sum"]),
                "parseq_delta_sum": int(best["parseq_delta_sum"]),
                "changed_groups": int(best["changed_groups"]),
                "false_change_groups": int(best["false_change_groups"]),
                "wrong_change_groups": int(best["wrong_change_groups"]),
                "tesseract_worsen_groups": int(best["tesseract_worsen_groups"]),
            }
        if mode in ensemble:
            output[f"{mode}_ensemble"] = {
                "tesseract_delta_sum": int(ensemble[mode]["tesseract_delta_sum"]),
                "parseq_delta_sum": int(ensemble[mode]["parseq_delta_sum"]),
                "changed_groups": int(ensemble[mode]["changed_groups"]),
                "false_change_groups": int(ensemble[mode]["false_change_groups"]),
                "wrong_change_groups": int(ensemble[mode]["wrong_change_groups"]),
                "tesseract_worsen_groups": int(ensemble[mode]["tesseract_worsen_groups"]),
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-policy", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parseq-safe", action="store_true")
    args = parser.parse_args()

    gate_rows = read_jsonl(args.gate_policy)
    score_rows = read_jsonl(args.score)
    model_groups = build_model_groups(score_rows)
    per_model = [
        evaluate(gate_rows, groups, policy_name=f"model_seed_{model_seed}", parseq_safe=args.parseq_safe)
        for model_seed, groups in sorted(model_groups.items())
    ]
    ensemble = evaluate(
        gate_rows,
        build_ensemble_groups(score_rows),
        policy_name="score_mean_ensemble",
        parseq_safe=args.parseq_safe,
    )
    output = {
        "description": "Hybrid table diagnostic: keep an existing high-precision group gate and swap only code ranking.",
        "gate_policy": str(args.gate_policy),
        "score": str(args.score),
        "config": {"parseq_safe": bool(args.parseq_safe)},
        "gate_policy_metrics_from_rows": summarize_gate_rows(gate_rows),
        "per_model": per_model,
        "ensemble": ensemble,
        "aggregate": aggregate_results(per_model, ensemble),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gate": output["gate_policy_metrics_from_rows"], "ensemble": ensemble}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
