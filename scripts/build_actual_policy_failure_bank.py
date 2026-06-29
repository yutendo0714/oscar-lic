#!/usr/bin/env python3
"""Build an OCR/string-aware case bank for actual assignment policies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts  # noqa: E402


PROFILE = "unicode_strict_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def group_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"]))


def category(selected_code: int, nearest_code: int, oracle_code: int) -> str:
    changed = selected_code != nearest_code
    oracle_changed = oracle_code != nearest_code
    if changed and oracle_changed and selected_code == oracle_code:
        return "exact_oracle_change"
    if changed and oracle_changed and selected_code != oracle_code:
        return "wrong_change"
    if changed and not oracle_changed:
        return "false_change"
    if not changed and oracle_changed:
        return "missed_oracle"
    return "correct_nearest"


def load_compare_predictions(path: Path) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    data = read_json(path)
    by_seed: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for pair in data["pairs"]:
        seed = str(pair["label"]).removeprefix("seed")
        baseline = read_jsonl(ROOT / pair["baseline_results"])
        candidate = read_jsonl(ROOT / pair["candidate_results"])
        if len(baseline) != len(candidate):
            raise SystemExit(f"row mismatch in {path}: {pair['label']}")
        by_seed[seed] = {
            index: {"baseline": base_row, "candidate": cand_row}
            for index, (base_row, cand_row) in enumerate(zip(baseline, candidate, strict=True))
        }
    return by_seed


def distance(reference: str, prediction: str, profile: str) -> int:
    return int(character_counts(reference, prediction, profile).distance)


def prediction_bundle(
    *,
    seed: int,
    index: int,
    current_predictions: dict[str, dict[int, dict[str, dict[str, Any]]]],
    shortlist_predictions: dict[str, dict[int, dict[str, dict[str, Any]]]],
    profile: str,
) -> dict[str, Any]:
    seed_key = str(seed)
    current_pair = current_predictions[seed_key][index]
    shortlist_pair = shortlist_predictions[seed_key][index]
    nearest = current_pair["baseline"]
    current = current_pair["candidate"]
    shortlist = shortlist_pair["candidate"]
    if nearest["reference"] != current["reference"] or nearest["reference"] != shortlist["reference"]:
        raise SystemExit(f"reference mismatch seed={seed} index={index}")
    reference = str(nearest["reference"])
    nearest_pred = str(nearest["prediction"])
    current_pred = str(current["prediction"])
    shortlist_pred = str(shortlist["prediction"])
    nearest_dist = distance(reference, nearest_pred, profile)
    current_dist = distance(reference, current_pred, profile)
    shortlist_dist = distance(reference, shortlist_pred, profile)
    return {
        "reference": reference,
        "source": nearest.get("source"),
        "split": nearest.get("split"),
        "nearest_image": nearest.get("image"),
        "current_image": current.get("image"),
        "shortlist_image": shortlist.get("image"),
        "nearest_prediction": nearest_pred,
        "current_prediction": current_pred,
        "shortlist_prediction": shortlist_pred,
        "nearest_distance": nearest_dist,
        "current_distance": current_dist,
        "shortlist_distance": shortlist_dist,
        "current_delta_vs_nearest": int(current_dist - nearest_dist),
        "shortlist_delta_vs_nearest": int(shortlist_dist - nearest_dist),
        "shortlist_delta_vs_current": int(shortlist_dist - current_dist),
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cases": len(cases),
        "by_current_category": Counter(),
        "by_shortlist_category": Counter(),
        "by_source": Counter(),
        "by_next_model_target": Counter(),
        "current_tesseract_delta_sum": 0,
        "shortlist_tesseract_delta_sum": 0,
        "shortlist_vs_current_tesseract_delta_sum": 0,
        "current_changed_groups": 0,
        "shortlist_changed_groups": 0,
        "shortlist_extra_improvement_groups": 0,
        "current_false_or_wrong_groups": 0,
        "shortlist_missed_groups": 0,
    }
    rank_by_target: dict[str, Counter] = defaultdict(Counter)
    for row in cases:
        current_category = row["current_policy_category"]
        shortlist_category = row["shortlist_policy_category"]
        target = row["next_model_target"]
        summary["by_current_category"][current_category] += 1
        summary["by_shortlist_category"][shortlist_category] += 1
        summary["by_source"][str(row.get("source"))] += 1
        summary["by_next_model_target"][target] += 1
        summary["current_tesseract_delta_sum"] += int(row["ocr"]["current_delta_vs_nearest"])
        summary["shortlist_tesseract_delta_sum"] += int(row["ocr"]["shortlist_delta_vs_nearest"])
        summary["shortlist_vs_current_tesseract_delta_sum"] += int(row["ocr"]["shortlist_delta_vs_current"])
        summary["current_changed_groups"] += int(row["current_code"] != row["nearest_code"])
        summary["shortlist_changed_groups"] += int(row["shortlist_code"] != row["nearest_code"])
        summary["shortlist_extra_improvement_groups"] += int(row["ocr"]["shortlist_delta_vs_current"] < 0)
        summary["current_false_or_wrong_groups"] += int(current_category in {"false_change", "wrong_change"})
        summary["shortlist_missed_groups"] += int(shortlist_category == "missed_oracle")
        rank_by_target[target][str(row["oracle_topk_rank"])] += 1
    for key in ["by_current_category", "by_shortlist_category", "by_source", "by_next_model_target"]:
        summary[key] = dict(sorted(summary[key].items()))
    summary["oracle_rank_by_next_model_target"] = {
        key: dict(sorted(counter.items(), key=lambda item: int(item[0])))
        for key, counter in sorted(rank_by_target.items())
    }
    return summary


def next_model_target(current_category: str, shortlist_category: str, shortlist_delta_vs_current: int) -> str:
    if current_category in {"false_change", "wrong_change"}:
        return "reject_current_bad_change"
    if shortlist_category == "missed_oracle":
        return "improve_first_stage_shortlist_recall"
    if shortlist_delta_vs_current < 0:
        return "recover_shortlist_oracle_change"
    if current_category == "exact_oracle_change":
        return "keep_current_good_change"
    return "keep_noop"


def make_cases(
    current_rows: list[dict[str, Any]],
    shortlist_rows: list[dict[str, Any]],
    current_predictions: dict[str, dict[int, dict[str, dict[str, Any]]]],
    shortlist_predictions: dict[str, dict[int, dict[str, dict[str, Any]]]],
    profile: str,
) -> list[dict[str, Any]]:
    current_by_key = {group_key(row): row for row in current_rows}
    shortlist_by_key = {group_key(row): row for row in shortlist_rows}
    if set(current_by_key) != set(shortlist_by_key):
        missing_current = sorted(set(shortlist_by_key) - set(current_by_key))[:5]
        missing_shortlist = sorted(set(current_by_key) - set(shortlist_by_key))[:5]
        raise SystemExit(f"policy key mismatch: missing_current={missing_current}, missing_shortlist={missing_shortlist}")

    cases = []
    for key in sorted(current_by_key):
        current = current_by_key[key]
        shortlist = shortlist_by_key[key]
        seed, source_index, candidate_index = key
        nearest_code = int(shortlist["nearest_code"])
        oracle_code = int(shortlist["assignment_oracle_code_index"])
        current_code = int(current["code_index"])
        shortlist_code = int(shortlist["code_index"])
        current_category = category(current_code, nearest_code, oracle_code)
        shortlist_category = category(shortlist_code, nearest_code, oracle_code)
        ocr = prediction_bundle(
            seed=seed,
            index=source_index,
            current_predictions=current_predictions,
            shortlist_predictions=shortlist_predictions,
            profile=profile,
        )
        case = {
            "key": {"real_seed": seed, "source_index": source_index, "candidate_index": candidate_index},
            "source": ocr["source"],
            "split": ocr["split"],
            "reference": ocr["reference"],
            "source_image": shortlist.get("source_image") or current.get("source_image"),
            "current_policy_category": current_category,
            "shortlist_policy_category": shortlist_category,
            "next_model_target": next_model_target(
                current_category, shortlist_category, int(ocr["shortlist_delta_vs_current"])
            ),
            "nearest_code": nearest_code,
            "oracle_code": oracle_code,
            "current_code": current_code,
            "shortlist_code": shortlist_code,
            "current_topk_rank": int(current["topk_rank"]),
            "shortlist_topk_rank": int(shortlist["topk_rank"]),
            "oracle_topk_rank": int(shortlist.get("oracle_topk_rank", shortlist["topk_rank"])),
            "shortlist_size": int(shortlist.get("shortlist_size", 0)),
            "oracle_in_shortlist": bool(shortlist.get("oracle_in_shortlist")),
            "current_selector_score": float(current.get("selector_score", 0.0)),
            "shortlist_selector_score": float(shortlist.get("selector_score", 0.0)),
            "ocr": ocr,
            "images": {
                "nearest": ocr["nearest_image"],
                "current": ocr["current_image"],
                "shortlist": ocr["shortlist_image"],
                "current_candidate_source": current.get("image_path"),
                "shortlist_candidate_source": shortlist.get("image_path"),
            },
        }
        cases.append(case)
    return cases


def report_table(cases: list[dict[str, Any]], target: str, limit: int = 16) -> list[str]:
    selected = [row for row in cases if row["next_model_target"] == target]
    selected.sort(key=lambda row: (row["ocr"]["shortlist_delta_vs_current"], row["key"]["real_seed"], row["key"]["source_index"]))
    lines = [
        f"### {target}",
        "",
        "| key | source | ref | nearest -> current -> shortlist | deltas current/shortlist/vs-current | codes nearest/current/shortlist/oracle | rank oracle |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in selected[:limit]:
        key = row["key"]
        ocr = row["ocr"]
        lines.append(
            f"| {key['real_seed']}/{key['source_index']}/{key['candidate_index']} | {row['source']} | "
            f"`{row['reference']}` | `{ocr['nearest_prediction']}` -> `{ocr['current_prediction']}` -> "
            f"`{ocr['shortlist_prediction']}` | {ocr['current_delta_vs_nearest']}/"
            f"{ocr['shortlist_delta_vs_nearest']}/{ocr['shortlist_delta_vs_current']} | "
            f"{row['nearest_code']}/{row['current_code']}/{row['shortlist_code']}/{row['oracle_code']} | "
            f"{row['oracle_topk_rank']} |"
        )
    if not selected:
        lines.append("| n/a | n/a | n/a | n/a | 0 | n/a | 0 |")
    lines.append("")
    return lines


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        f"# {result['experiment_id']}",
        "",
        "This is a diagnostic case bank for already-evaluated actual `.oscr` assignment policies.",
        "Held-out OCR outcomes are used only to categorize failures and should not become deployable features.",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['cases']}`",
        f"- Current Tesseract delta vs nearest: `{summary['current_tesseract_delta_sum']}`",
        f"- Shortlist Tesseract delta vs nearest: `{summary['shortlist_tesseract_delta_sum']}`",
        f"- Shortlist vs current delta: `{summary['shortlist_vs_current_tesseract_delta_sum']}`",
        f"- Current categories: `{summary['by_current_category']}`",
        f"- Shortlist categories: `{summary['by_shortlist_category']}`",
        f"- Next-model targets: `{summary['by_next_model_target']}`",
        f"- Oracle rank by target: `{summary['oracle_rank_by_next_model_target']}`",
        "",
        "## Priority Cases",
        "",
    ]
    for target in [
        "recover_shortlist_oracle_change",
        "reject_current_bad_change",
        "improve_first_stage_shortlist_recall",
        "keep_current_good_change",
    ]:
        lines.extend(report_table(result["cases"], target))
    lines.extend(
        [
            "## Interpretation",
            "",
            "The immediate selector problem is not extra bitrate. It is a high-precision choice among existing code-index options.",
            "The highest-value targets are the shortlist oracle recoveries, while the current false/wrong changes define the abstention floor.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def scalar_metrics(prefix: str, value: object, out: dict[str, Any]) -> None:
    if isinstance(value, (int, float)):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            scalar_metrics(f"{prefix}_{key}", child, out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-policy", type=Path, required=True)
    parser.add_argument("--shortlist-policy", type=Path, required=True)
    parser.add_argument("--current-compare", type=Path, required=True)
    parser.add_argument("--shortlist-compare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_actual_assignment_failure_bank")
    parser.add_argument("--profile", default=PROFILE)
    args = parser.parse_args()

    current_rows = read_jsonl(args.current_policy)
    shortlist_rows = read_jsonl(args.shortlist_policy)
    current_predictions = load_compare_predictions(args.current_compare)
    shortlist_predictions = load_compare_predictions(args.shortlist_compare)
    cases = make_cases(current_rows, shortlist_rows, current_predictions, shortlist_predictions, args.profile)
    summary = summarize_cases(cases)
    result = {
        "experiment_id": args.experiment_id,
        "profile": args.profile,
        "validity": "valid_diagnostic",
        "inputs": {
            "current_policy": str(args.current_policy),
            "shortlist_policy": str(args.shortlist_policy),
            "current_compare": str(args.current_compare),
            "shortlist_compare": str(args.shortlist_compare),
        },
        "summary": summary,
        "cases": cases,
    }
    metrics: dict[str, Any] = {}
    scalar_metrics("summary", summary, metrics)
    result["aggregate"] = {"scalar_metrics": {key: {"value": value} for key, value in metrics.items()}}
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
