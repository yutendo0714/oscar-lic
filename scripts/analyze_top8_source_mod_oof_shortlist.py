#!/usr/bin/env python3
"""Evaluate source-modulo OOF top-8 score files as a first-stage shortlist.

This is a diagnostic for candidate-set coverage only. It filters each score
file to the source-modulo slice that was held out during that score model's
training, then asks whether the resulting OOF score can improve the clean
validation candidate shortlist beyond deterministic top-k rank caps.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from analyze_top8_two_stage_bottleneck import (  # type: ignore
    ROOT,
    average_precision,
    candidate_key,
    evaluate_selection,
    git_commit,
    group_key,
    load_score_source,
    load_table,
    nearest_row,
    oracle_row,
    roc_auc,
    sha256_file,
    split_groups,
)


def infer_remainder(path: Path) -> int:
    match = re.search(r"(?:^|_)r([0-9]+)(?:_|\\.)", path.name)
    if not match:
        raise ValueError(f"cannot infer source-mod remainder from {path}")
    return int(match.group(1))


def build_oof_scores(
    score_files: list[Path],
    table_candidates: dict[tuple[int, int, int, int, int], dict[str, Any]],
    *,
    modulo: int,
) -> tuple[dict[tuple[int, int, int, int, int], float], dict[str, Any]]:
    by_remainder: dict[int, dict[tuple[int, int, int, int, int], float]] = {}
    inputs: list[dict[str, Any]] = []
    duplicate_remainders: list[int] = []
    for path in score_files:
        remainder = infer_remainder(path)
        if remainder in by_remainder:
            duplicate_remainders.append(remainder)
        _, scores, meta = load_score_source(path, table_candidates)
        by_remainder[remainder] = scores
        inputs.append({"remainder": remainder, **meta})
    oof: dict[tuple[int, int, int, int, int], float] = {}
    missing = 0
    for key in table_candidates:
        source_index = int(key[2])
        remainder = source_index % modulo
        scores = by_remainder.get(remainder)
        if scores is None or key not in scores:
            missing += 1
            continue
        oof[key] = float(scores[key])
    meta = {
        "modulo": modulo,
        "score_files": inputs,
        "remainders_loaded": sorted(by_remainder),
        "duplicate_remainders": duplicate_remainders,
        "matched_oof_candidates": len(oof),
        "missing_oof_candidates": missing,
    }
    return oof, meta


def nonnearest_ranked(
    rows: list[dict[str, Any]],
    scores: dict[tuple[int, int, int, int, int], float],
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, int, int, dict[str, Any]]] = []
    for row in rows:
        if int(row.get("is_nearest", 0)):
            continue
        score = scores.get(candidate_key(row))
        if score is None or not np.isfinite(score):
            continue
        ranked.append((float(score), -int(row.get("topk_rank", 999)), -int(row["code_index"]), row))
    ranked.sort(reverse=True)
    return [row for _, _, _, row in ranked]


def topk_set(rows: list[dict[str, Any]], cap: int) -> set[int]:
    return {
        int(row["code_index"])
        for row in rows
        if not int(row.get("is_nearest", 0)) and int(row.get("topk_rank", 999)) <= cap
    }


def score_set(
    rows: list[dict[str, Any]],
    scores: dict[tuple[int, int, int, int, int], float],
    count: int,
) -> set[int]:
    return {int(row["code_index"]) for row in nonnearest_ranked(rows, scores)[:count]}


def oracle_if_in_shortlist_policy(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    shortlists: dict[tuple[int, int, int, int], set[int]],
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        oracle_code = int(oracle["code_index"])
        selected[key] = int(nearest["code_index"])
        if oracle_code != int(nearest["code_index"]) and oracle_code in shortlists.get(key, set()):
            selected[key] = oracle_code
    return selected


def top_scored_policy(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        ranked = nonnearest_ranked(rows, scores)
        selected[key] = int(ranked[0]["code_index"]) if ranked else int(nearest_row(rows)["code_index"])
    return selected


def shortlist_summary(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int | None,
    score_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shortlists: dict[tuple[int, int, int, int], set[int]] = {}
    sizes: list[int] = []
    rank_counter = Counter()
    deep_hits: list[dict[str, Any]] = []
    for key, rows in groups.items():
        codes: set[int] = set()
        if topk_cap is not None:
            codes |= topk_set(rows, topk_cap)
        if score_count > 0:
            codes |= score_set(rows, scores, score_count)
        shortlists[key] = codes
        sizes.append(len(codes))
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        oracle_changes = int(oracle["code_index"]) != int(nearest["code_index"])
        if oracle_changes:
            rank_counter["oracle_change_groups"] += 1
            oracle_code = int(oracle["code_index"])
            oracle_topk = int(oracle.get("topk_rank", 999))
            if oracle_code in codes:
                rank_counter["covered_oracle_changes"] += 1
                rank_counter[f"covered_topk_rank_le{min(oracle_topk, 8)}"] += 1
                if oracle_topk > 4 and len(deep_hits) < 40:
                    deep_hits.append(
                        {
                            "group_key": list(key),
                            "source": rows[0].get("source"),
                            "reference": rows[0].get("reference"),
                            "oracle_code": oracle_code,
                            "oracle_topk_rank": oracle_topk,
                            "oracle_tesseract_delta_vs_nearest": int(oracle.get("tesseract_delta_vs_nearest", 0)),
                        }
                    )
    metrics = evaluate_selection(groups, oracle_if_in_shortlist_policy(groups, shortlists))
    details = {
        "shortlist_size": {
            "mean": float(np.mean(sizes)) if sizes else 0.0,
            "max": int(max(sizes)) if sizes else 0,
            "min": int(min(sizes)) if sizes else 0,
        },
        "coverage_counts": {key: int(value) for key, value in rank_counter.items()},
        "deep_rank_hits": deep_hits,
    }
    return metrics, details


def group_score_separation(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
) -> dict[str, Any]:
    group_scores: list[float] = []
    labels: list[int] = []
    for rows in groups.values():
        ranked = nonnearest_ranked(rows, scores)
        if not ranked:
            continue
        group_scores.append(float(scores[candidate_key(ranked[0])]))
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        labels.append(int(int(oracle["code_index"]) != int(nearest["code_index"])))
    s = np.asarray(group_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    return {
        "groups_with_scores": int(len(group_scores)),
        "oracle_change_groups": int(y.sum()) if len(y) else 0,
        "auc": roc_auc(s, y) if len(y) else None,
        "ap": average_precision(s, y) if len(y) else None,
    }


def compact(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics.get('exact_changed_groups', 0)}/{metrics.get('oracle_change_groups', 0)} exact, "
        f"false {metrics.get('false_change', 0)}, wrong {metrics.get('wrong_change', 0)}, "
        f"miss {metrics.get('missed_oracle', 0)}, "
        f"T {metrics.get('tesseract_delta_vs_nearest', 0)}, P {metrics.get('parseq_delta_vs_nearest', 0)}"
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    val = result["partitions"].get("val", {})
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Diagnostic-only source-modulo OOF shortlist audit. Each candidate score is taken only from the model whose source-modulo validation slice contains that candidate's source index.",
        "",
        "## Validation Summary",
        "",
        "| shortlist | size mean/max | summary |",
        "|---|---:|---|",
    ]
    for name, entry in val.get("shortlists", {}).items():
        size = entry["details"]["shortlist_size"]
        lines.append(f"| {name} | {size['mean']:.2f}/{size['max']} | {compact(entry['metrics'])} |")
    top = val.get("top_scored_policy")
    if top:
        lines.extend(["", "## Unsafe Top-Score Policy", "", compact(top), ""])
    sep = val.get("group_score_separation")
    if sep:
        lines.extend(
            [
                "## Group Score Separation",
                "",
                f"AUC/AP for max non-nearest OOF score predicting an oracle-change group: {sep.get('auc')}/{sep.get('ap')}.",
                "",
            ]
        )
    lines.extend(["## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-source-mod-oof-shortlist",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
        "  table:",
        f"    path: {result['inputs']['table']['path']}",
        f"    sha256: {result['inputs']['table']['sha256']}",
    ]
    for item in result["inputs"]["score_files"]:
        safe = f"score_r{item['remainder']}"
        lines.extend([f"  {safe}:", f"    path: {item['path']}", f"    sha256: {item['sha256']}"])
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
            "  analyzer:",
            "    path: scripts/analyze_top8_source_mod_oof_shortlist.py",
            f"    sha256: {sha256_file(Path(__file__))}",
            "conclusion: diagnostic OOF shortlist audit; no selector promoted",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, action="append", required=True)
    parser.add_argument("--source-modulo", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups, by_candidate = load_table(args.table)
    partitions = split_groups(groups)
    oof_scores, oof_meta = build_oof_scores(args.score_file, by_candidate, modulo=args.source_modulo)
    partition_results: dict[str, Any] = {}
    for part_name, part_groups in partitions.items():
        shortlists: dict[str, Any] = {}
        for k in [1, 2, 4, 8]:
            metrics, details = shortlist_summary(part_groups, oof_scores, topk_cap=None, score_count=k)
            shortlists[f"oof_score_top{k}"] = {"metrics": metrics, "details": details}
        for k in [1, 2, 4]:
            metrics, details = shortlist_summary(part_groups, oof_scores, topk_cap=4, score_count=k)
            shortlists[f"topk4_union_oof_top{k}"] = {"metrics": metrics, "details": details}
        metrics, details = shortlist_summary(part_groups, oof_scores, topk_cap=4, score_count=0)
        shortlists["deterministic_topk_rank_le4"] = {"metrics": metrics, "details": details}
        metrics, details = shortlist_summary(part_groups, oof_scores, topk_cap=8, score_count=0)
        shortlists["full_top8_candidate_set"] = {"metrics": metrics, "details": details}
        partition_results[part_name] = {
            "groups": len(part_groups),
            "shortlists": shortlists,
            "top_scored_policy": evaluate_selection(part_groups, top_scored_policy(part_groups, oof_scores)),
            "group_score_separation": group_score_separation(part_groups, oof_scores),
        }
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_oof_shortlist_not_promoted",
        "code_commit": git_commit(),
        "inputs": {
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "score_files": oof_meta["score_files"],
        },
        "config": {"source_modulo": args.source_modulo},
        "data_summary": {
            "groups": len(groups),
            "candidates": len(by_candidate),
            "oof": {k: v for k, v in oof_meta.items() if k != "score_files"},
        },
        "partitions": partition_results,
        "hashes": {"script": sha256_file(Path(__file__))},
        "interpretation": (
            "Source-modulo OOF scores are valid as held-out score evidence for each source-index slice, "
            "but this diagnostic still uses an oracle chooser inside each candidate set. Therefore a "
            "shortlist improvement is first-stage evidence only, not a deployable policy. Promotion remains "
            "blocked until a no-op-aware selector beats the current counted actual-bitstream floor."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report)
    val_shortlists = result["partitions"].get("val", {}).get("shortlists", {})
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "val": {
                    name: {
                        "summary": compact(entry["metrics"]),
                        "size_mean": entry["details"]["shortlist_size"]["mean"],
                    }
                    for name, entry in val_shortlists.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
