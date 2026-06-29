#!/usr/bin/env python3
"""Analyze top-4 assignment failure strata without training another selector.

This diagnostic uses OCR deltas only as held-out analysis labels. It does not
produce a deployable policy and should not be treated as promotion evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_assignment_topk_policy_selector as base  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_idx(rows: list[dict[str, Any]], indices: list[int]) -> int:
    return next(idx for idx in indices if int(rows[idx]["is_nearest"]))


def full_oracle_idx(rows: list[dict[str, Any]], indices: list[int]) -> int:
    oracle = [idx for idx in indices if int(rows[idx]["label_assignment_oracle_choice"])]
    if oracle:
        return oracle[0]
    return nearest_idx(rows, indices)


def select_by_score(
    rows: list[dict[str, Any]],
    candidates_by_group: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, candidates in candidates_by_group.items():
        selected[key] = max(candidates, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))
    return selected


def select_oracle(
    rows: list[dict[str, Any]],
    groups_by_key: dict[tuple[int, int, int, int], list[int]],
) -> dict[tuple[int, int, int, int], int]:
    return {key: full_oracle_idx(rows, indices) for key, indices in groups_by_key.items()}


def select_nearest(
    rows: list[dict[str, Any]],
    groups_by_key: dict[tuple[int, int, int, int], list[int]],
) -> dict[tuple[int, int, int, int], int]:
    return {key: nearest_idx(rows, indices) for key, indices in groups_by_key.items()}


def selection_type(
    rows: list[dict[str, Any]],
    full_group: list[int],
    selected: int,
) -> str:
    nearest = nearest_idx(rows, full_group)
    oracle = full_oracle_idx(rows, full_group)
    if selected == nearest:
        return "nearest_missed" if oracle != nearest else "nearest_correct"
    if selected == oracle:
        return "exact_oracle"
    if oracle != nearest:
        return "wrong_change"
    if int(rows[selected]["parseq_delta_vs_nearest"]) > 0 or int(rows[selected]["tesseract_delta_vs_nearest"]) > 0:
        return "false_harmful_change"
    if int(rows[selected]["parseq_delta_vs_nearest"]) < 0 or int(rows[selected]["tesseract_delta_vs_nearest"]) < 0:
        return "false_safe_improve_label_gap"
    return "false_neutral_change"


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def bin_ref_length(length: int) -> str:
    if length <= 4:
        return "01_len_1_4"
    if length <= 8:
        return "02_len_5_8"
    if length <= 12:
        return "03_len_9_12"
    return "04_len_13_plus"


def bin_tesseract_distance(distance: int) -> str:
    if distance == 0:
        return "00_exact"
    if distance <= 2:
        return "01_dist_1_2"
    if distance <= 5:
        return "02_dist_3_5"
    if distance <= 9:
        return "03_dist_6_9"
    return "04_dist_10_plus"


def bin_parseq_distance(distance: int) -> str:
    if distance == 0:
        return "00_exact"
    if distance == 1:
        return "01_dist_1"
    if distance <= 3:
        return "02_dist_2_3"
    return "03_dist_4_plus"


def bin_area(area: float) -> str:
    if area < 4096:
        return "01_area_lt_4096"
    if area < 8192:
        return "02_area_4096_8191"
    if area < 16384:
        return "03_area_8192_16383"
    return "04_area_16384_plus"


def bin_score_rank(rank: Any, oracle_changes: bool) -> str:
    if not oracle_changes:
        return "00_no_oracle_change"
    if rank is None:
        return "99_missing"
    rank = int(rank)
    if rank == 1:
        return "01_rank1"
    if rank == 2:
        return "02_rank2"
    if rank <= 4:
        return "03_rank3_4"
    return "04_rank5_plus"


def empty_metrics() -> dict[str, Any]:
    return {
        "groups": 0,
        "oracle_change_groups": 0,
        "shortlist_oracle_change_groups": 0,
        "shortlist_missing_oracle_groups": 0,
        "full_oracle_tesseract_delta_sum": 0,
        "shortlist_oracle_tesseract_delta_sum": 0,
        "nearest_tesseract_distance_sum": 0,
        "nearest_parseq_distance_sum": 0,
        "reference_length_sum": 0,
        "candidate_count_sum": 0,
        "policies": defaultdict(lambda: defaultdict(int)),
    }


def add_group_metrics(
    bucket: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    group: list[int],
    candidates: list[int],
    selections: dict[str, int],
) -> None:
    nearest = nearest_idx(rows, group)
    oracle = full_oracle_idx(rows, group)
    shortlist_oracle = full_oracle_idx(rows, candidates)
    oracle_changes = oracle != nearest
    shortlist_changes = shortlist_oracle != nearest
    bucket["groups"] += 1
    bucket["oracle_change_groups"] += int(oracle_changes)
    bucket["shortlist_oracle_change_groups"] += int(shortlist_changes)
    bucket["shortlist_missing_oracle_groups"] += int(oracle_changes and oracle not in candidates)
    bucket["full_oracle_tesseract_delta_sum"] += int(rows[oracle]["tesseract_delta_vs_nearest"])
    bucket["shortlist_oracle_tesseract_delta_sum"] += int(rows[shortlist_oracle]["tesseract_delta_vs_nearest"])
    bucket["nearest_tesseract_distance_sum"] += int(rows[nearest]["tesseract_nearest_distance"])
    bucket["nearest_parseq_distance_sum"] += int(rows[nearest]["parseq_nearest_distance"])
    bucket["reference_length_sum"] += int(rows[nearest].get("reference_length", len(str(rows[nearest].get("reference", "")))))
    bucket["candidate_count_sum"] += len(candidates)
    for policy_name, selected in selections.items():
        policy = bucket["policies"][policy_name]
        kind = selection_type(rows, group, selected)
        policy[f"selection_type__{kind}"] += 1
        policy["changed_groups"] += int(selected != nearest)
        policy["exact_oracle_changes"] += int(selected == oracle and oracle_changes)
        policy["missed_oracle_changes"] += int(oracle_changes and selected != oracle)
        policy["false_changes"] += int(selected != nearest and not oracle_changes)
        policy["wrong_changes"] += int(selected != nearest and oracle_changes and selected != oracle)
        policy["parseq_delta_sum"] += int(rows[selected]["parseq_delta_vs_nearest"])
        policy["tesseract_delta_sum"] += int(rows[selected]["tesseract_delta_vs_nearest"])
        policy["parseq_worsen_groups"] += int(int(rows[selected]["parseq_delta_vs_nearest"]) > 0)
        policy["tesseract_worsen_groups"] += int(int(rows[selected]["tesseract_delta_vs_nearest"]) > 0)
        policy["tesseract_improve_groups"] += int(int(rows[selected]["tesseract_delta_vs_nearest"]) < 0)


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    groups = int(bucket["groups"])
    result = {key: value for key, value in bucket.items() if key != "policies"}
    if groups:
        result["oracle_change_rate"] = bucket["oracle_change_groups"] / groups
        result["shortlist_oracle_change_rate"] = bucket["shortlist_oracle_change_groups"] / groups
        result["shortlist_missing_oracle_rate"] = bucket["shortlist_missing_oracle_groups"] / groups
        result["nearest_tesseract_distance_mean"] = bucket["nearest_tesseract_distance_sum"] / groups
        result["nearest_parseq_distance_mean"] = bucket["nearest_parseq_distance_sum"] / groups
        result["reference_length_mean"] = bucket["reference_length_sum"] / groups
        result["candidate_count_mean"] = bucket["candidate_count_sum"] / groups
    result["policies"] = {name: dict(metrics) for name, metrics in bucket["policies"].items()}
    return result


def add_to_strata(
    strata: dict[str, dict[str, dict[str, Any]]],
    labels: dict[str, str],
    *,
    rows: list[dict[str, Any]],
    group: list[int],
    candidates: list[int],
    selections: dict[str, int],
) -> None:
    for stratum_name, label in labels.items():
        add_group_metrics(
            strata[stratum_name][label],
            rows=rows,
            group=group,
            candidates=candidates,
            selections=selections,
        )


def top_rows_for_policy(
    details: list[dict[str, Any]],
    policy_name: str,
    *,
    selection_prefix: str,
    max_rows: int = 20,
) -> list[dict[str, Any]]:
    ranked = []
    for row in details:
        selection = row["policies"][policy_name]
        if not selection["selection_type"].startswith(selection_prefix):
            continue
        ranked.append(
            {
                "key": row["key"],
                "source": row["source"],
                "reference": row["reference"],
                "reference_length": row["reference_length"],
                "nearest_tesseract_distance": row["nearest_tesseract_distance"],
                "nearest_parseq_distance": row["nearest_parseq_distance"],
                "oracle_tesseract_delta": row["oracle_tesseract_delta"],
                "oracle_topk_rank": row["oracle_topk_rank"],
                "oracle_score_rank_shortlist": row["oracle_score_rank_shortlist"],
                "safe_score_rank_shortlist": row["safe_score_rank_shortlist"],
                "selected_code": selection["selected_code"],
                "selected_topk_rank": selection["selected_topk_rank"],
                "selection_type": selection["selection_type"],
                "selected_tesseract_delta": selection["tesseract_delta"],
                "selected_parseq_delta": selection["parseq_delta"],
            }
        )
    ranked.sort(key=lambda item: (-abs(int(item["selected_tesseract_delta"])), item["source"], item["reference"]))
    return ranked[:max_rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--score-model-seed", type=int, default=2)
    parser.add_argument("--shortlist-topk", type=int, default=4)
    args = parser.parse_args()

    rows = base.read_jsonl(args.table)
    groups = base.grouped_indices(rows)
    oracle_by_model = base.load_score_features(args.oracle_score)
    safe_by_model = base.load_score_features(args.safe_score)
    oracle_scores = np.asarray(
        [oracle_by_model[args.score_model_seed][idx] for idx in range(len(rows))],
        dtype=np.float32,
    )
    safe_scores = np.asarray(
        [safe_by_model[args.score_model_seed][idx] for idx in range(len(rows))],
        dtype=np.float32,
    )
    fused_scores = oracle_scores + safe_scores
    conservative_scores = np.minimum(oracle_scores, safe_scores)

    candidates = {
        key: base.candidate_indices_for_group(rows, indices, oracle_scores, safe_scores, topk=args.shortlist_topk)
        for key, indices in groups.items()
    }
    val_groups = {key: indices for key, indices in groups.items() if int(rows[indices[0]]["seed"]) == args.val_seed}
    val_candidates = {key: candidates[key] for key in val_groups}

    policy_selected = {
        "nearest": select_nearest(rows, val_groups),
        "full_oracle": select_oracle(rows, val_groups),
        "shortlist_oracle": select_oracle(rows, val_candidates),
        "oracle_score_argmax": select_by_score(rows, val_candidates, oracle_scores),
        "safe_score_argmax": select_by_score(rows, val_candidates, safe_scores),
        "fused_score_argmax": select_by_score(rows, val_candidates, fused_scores),
        "conservative_score_argmax": select_by_score(rows, val_candidates, conservative_scores),
    }

    overall = empty_metrics()
    strata: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(empty_metrics))
    detail_rows: list[dict[str, Any]] = []

    for key, group in val_groups.items():
        nearest = nearest_idx(rows, group)
        oracle = full_oracle_idx(rows, group)
        cand = val_candidates[key]
        shortlist_oracle = full_oracle_idx(rows, cand)
        nearest_row = rows[nearest]
        oracle_changes = oracle != nearest
        ref_len = int(nearest_row.get("reference_length", len(str(nearest_row.get("reference", "")))))
        tdist = int(nearest_row["tesseract_nearest_distance"])
        pdist = int(nearest_row["parseq_nearest_distance"])
        area = numeric(nearest_row.get("img_area"))
        labels = {
            "by_source": str(nearest_row.get("source", "unknown")),
            "by_source_split": str(nearest_row.get("split", "unknown")),
            "by_reference_length": bin_ref_length(ref_len),
            "by_tesseract_base_distance": bin_tesseract_distance(tdist),
            "by_parseq_base_distance": bin_parseq_distance(pdist),
            "by_image_area": bin_area(area),
            "by_oracle_score_rank_shortlist": bin_score_rank(
                None if oracle not in cand else sorted(cand, key=lambda idx: (-float(oracle_scores[idx]), int(idx))).index(oracle) + 1,
                oracle_changes,
            ),
            "by_safe_score_rank_shortlist": bin_score_rank(
                None if oracle not in cand else sorted(cand, key=lambda idx: (-float(safe_scores[idx]), int(idx))).index(oracle) + 1,
                oracle_changes,
            ),
        }
        selections = {name: selected_by_key[key] for name, selected_by_key in policy_selected.items()}
        add_group_metrics(overall, rows=rows, group=group, candidates=cand, selections=selections)
        add_to_strata(strata, labels, rows=rows, group=group, candidates=cand, selections=selections)

        policy_details = {}
        for policy_name, selected in selections.items():
            policy_details[policy_name] = {
                "selected_row_index": int(selected),
                "selected_code": int(rows[selected]["code_index"]),
                "selected_topk_rank": int(rows[selected]["topk_rank"]),
                "selection_type": selection_type(rows, group, selected),
                "parseq_delta": int(rows[selected]["parseq_delta_vs_nearest"]),
                "tesseract_delta": int(rows[selected]["tesseract_delta_vs_nearest"]),
                "oracle_score": float(oracle_scores[selected]),
                "safe_score": float(safe_scores[selected]),
            }
        oracle_rank_short = None if oracle not in cand else sorted(
            cand, key=lambda idx: (-float(oracle_scores[idx]), int(idx))
        ).index(oracle) + 1
        safe_rank_short = None if oracle not in cand else sorted(
            cand, key=lambda idx: (-float(safe_scores[idx]), int(idx))
        ).index(oracle) + 1
        detail_rows.append(
            {
                "key": list(key),
                "source": nearest_row.get("source"),
                "source_split": nearest_row.get("split"),
                "reference": nearest_row.get("reference"),
                "reference_length": ref_len,
                "nearest_row_index": int(nearest),
                "oracle_row_index": int(oracle),
                "shortlist_oracle_row_index": int(shortlist_oracle),
                "nearest_code": int(rows[nearest]["code_index"]),
                "oracle_code": int(rows[oracle]["code_index"]),
                "shortlist_oracle_code": int(rows[shortlist_oracle]["code_index"]),
                "nearest_tesseract_distance": tdist,
                "nearest_parseq_distance": pdist,
                "oracle_tesseract_delta": int(rows[oracle]["tesseract_delta_vs_nearest"]),
                "oracle_parseq_delta": int(rows[oracle]["parseq_delta_vs_nearest"]),
                "shortlist_oracle_tesseract_delta": int(rows[shortlist_oracle]["tesseract_delta_vs_nearest"]),
                "oracle_topk_rank": int(rows[oracle]["topk_rank"]),
                "oracle_in_shortlist": int(oracle in cand),
                "oracle_score_rank_shortlist": oracle_rank_short,
                "safe_score_rank_shortlist": safe_rank_short,
                "candidate_count": len(cand),
                "strata": labels,
                "policies": policy_details,
            }
        )

    tables: list[dict[str, Any]] = []
    table = {
        "label": "eval300_seed1_top4_failure_strata",
        "overall": finalize_bucket(overall),
    }
    for stratum_name, buckets in strata.items():
        table[stratum_name] = {label: finalize_bucket(bucket) for label, bucket in sorted(buckets.items())}
    tables.append(table)

    source_pressure = {}
    for source, bucket in table["by_source"].items():
        source_pressure[source] = {
            "groups": bucket["groups"],
            "oracle_change_groups": bucket["oracle_change_groups"],
            "shortlist_missing_oracle_groups": bucket["shortlist_missing_oracle_groups"],
            "safe_score_false_or_wrong": int(
                bucket["policies"]["safe_score_argmax"].get("false_changes", 0)
                + bucket["policies"]["safe_score_argmax"].get("wrong_changes", 0)
            ),
            "safe_score_exact_oracle": int(bucket["policies"]["safe_score_argmax"].get("exact_oracle_changes", 0)),
            "oracle_score_false_or_wrong": int(
                bucket["policies"]["oracle_score_argmax"].get("false_changes", 0)
                + bucket["policies"]["oracle_score_argmax"].get("wrong_changes", 0)
            ),
            "oracle_score_exact_oracle": int(bucket["policies"]["oracle_score_argmax"].get("exact_oracle_changes", 0)),
        }

    diagnosis = {
        "dominant_source_pressure": source_pressure,
        "safe_score_false_or_wrong_examples": top_rows_for_policy(
            detail_rows, "safe_score_argmax", selection_prefix="false", max_rows=15
        )
        + top_rows_for_policy(detail_rows, "safe_score_argmax", selection_prefix="wrong", max_rows=15),
        "oracle_score_false_or_wrong_examples": top_rows_for_policy(
            detail_rows, "oracle_score_argmax", selection_prefix="false", max_rows=15
        )
        + top_rows_for_policy(detail_rows, "oracle_score_argmax", selection_prefix="wrong", max_rows=15),
        "shortlist_missing_examples": [
            {
                "key": row["key"],
                "source": row["source"],
                "reference": row["reference"],
                "nearest_tesseract_distance": row["nearest_tesseract_distance"],
                "oracle_tesseract_delta": row["oracle_tesseract_delta"],
                "oracle_topk_rank": row["oracle_topk_rank"],
            }
            for row in detail_rows
            if row["oracle_in_shortlist"] == 0
        ],
    }

    result = {
        "description": "Failure-strata audit for Eval600 source-OOF to Eval300 top-4 assignment selection.",
        "validity": "diagnostic_no_promotion",
        "track": "A_pure_visual_bitstream_policy_diagnostic",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "val_seed": args.val_seed,
            "score_model_seed": args.score_model_seed,
            "shortlist_topk": args.shortlist_topk,
        },
        "artifacts": {
            "table_sha256": sha256_file(args.table),
            "oracle_score_sha256": sha256_file(args.oracle_score),
            "safe_score_sha256": sha256_file(args.safe_score),
        },
        "tables": tables,
        "diagnosis": diagnosis,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    with args.details_output.open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
