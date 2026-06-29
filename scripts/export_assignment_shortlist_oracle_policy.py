#!/usr/bin/env python3
"""Export a non-deployable shortlist-oracle assignment policy.

This diagnostic keeps the candidate selection fixed and changes only the
decoder-known codebook index.  It selects the true assignment oracle if that
row is present in a deployable score-defined shortlist; otherwise it falls back
to nearest/no-op.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GroupKey = tuple[int, int, int]
RowKey = tuple[int, int, int, int, int]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_key(row: dict[str, Any]) -> GroupKey:
    return (int(row.get("real_seed", row["seed"])), int(row["source_index"]), int(row["candidate_index"]))


def row_key(row: dict[str, Any]) -> RowKey:
    return (
        int(row.get("real_seed", row["seed"])),
        int(row["source_index"]),
        int(row["candidate_index"]),
        int(row["code_index"]),
        int(row["topk_rank"]),
    )


def group_rows(rows: list[dict[str, Any]]) -> dict[GroupKey, list[dict[str, Any]]]:
    groups: dict[GroupKey, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return groups


def load_score_rows(path: Path, *, val_seed: int, model_seed: int) -> dict[GroupKey, list[dict[str, Any]]]:
    rows = [
        row
        for row in read_jsonl(path)
        if int(row["val_seed"]) == int(val_seed)
        and int(row["model_seed"]) == int(model_seed)
        and int(row["seed"]) == int(val_seed)
    ]
    if not rows:
        raise SystemExit(f"no score rows matched val_seed={val_seed} model_seed={model_seed}: {path}")
    return group_rows(rows)


def nearest_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in rows if int(row["is_nearest"])]
    if len(matches) != 1:
        raise SystemExit(f"group has {len(matches)} nearest rows")
    return matches[0]


def oracle_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [row for row in rows if int(row.get("label_assignment_oracle_choice", 0))]
    if len(matches) > 1:
        raise SystemExit("group has multiple assignment-oracle rows")
    return matches[0] if matches else None


def top_score_keys(rows: list[dict[str, Any]], topk: int) -> set[RowKey]:
    ranked = sorted(rows, key=lambda row: (float(row["score"]), -int(row["topk_rank"])), reverse=True)
    return {row_key(row) for row in ranked[:topk]}


def build_shortlist(
    table_rows: list[dict[str, Any]],
    oracle_scores: dict[GroupKey, list[dict[str, Any]]],
    safe_scores: dict[GroupKey, list[dict[str, Any]]],
    *,
    topk: int,
) -> tuple[set[RowKey], dict[str, Any]]:
    key = group_key(table_rows[0])
    shortlist = {row_key(nearest_row(table_rows))}
    if key not in oracle_scores:
        raise SystemExit(f"missing oracle-score rows for group {key}")
    if key not in safe_scores:
        raise SystemExit(f"missing safe-score rows for group {key}")
    oracle_top = top_score_keys(oracle_scores[key], topk)
    safe_top = top_score_keys(safe_scores[key], topk)
    shortlist |= oracle_top
    shortlist |= safe_top
    table_keys = {row_key(row) for row in table_rows}
    unknown = shortlist - table_keys
    if unknown:
        raise SystemExit(f"shortlist keys missing from table for group {key}: {sorted(unknown)[:3]}")
    return shortlist, {
        "shortlist_size": len(shortlist),
        "oracle_score_topk_size": len(oracle_top),
        "safe_score_topk_size": len(safe_top),
    }


def summarize(chosen: list[dict[str, Any]], all_groups: dict[GroupKey, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = {
        "groups": len(chosen),
        "changed_groups": 0,
        "exact_oracle_changes": 0,
        "missed_oracle_changes": 0,
        "false_changes": 0,
        "wrong_changes": 0,
        "parseq_delta_sum_vs_nearest": 0,
        "tesseract_delta_sum_vs_nearest": 0,
        "parseq_worsen_groups": 0,
        "tesseract_improve_groups": 0,
        "tesseract_worsen_groups": 0,
        "shortlist_size_mean": 0.0,
        "shortlist_oracle_contained_groups": 0,
    }
    by_real_seed: dict[str, dict[str, Any]] = {}
    for row in chosen:
        key = group_key(row)
        rows = all_groups[key]
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        if oracle is None:
            raise SystemExit(f"group has no assignment oracle row: {key}")
        selected_code = int(row["code_index"])
        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        changed = selected_code != nearest_code
        oracle_changes = oracle_code != nearest_code
        exact = changed and selected_code == oracle_code
        missed = oracle_changes and selected_code != oracle_code
        false = changed and not oracle_changes
        wrong = changed and oracle_changes and selected_code != oracle_code
        parseq_delta = int(row.get("parseq_delta_vs_nearest", 0))
        tesseract_delta = int(row.get("tesseract_delta_vs_nearest", 0))
        seed_key = str(int(row.get("real_seed", row["seed"])))
        seed_stats = by_real_seed.setdefault(seed_key, {name: 0 for name in totals if name not in {"shortlist_size_mean"}})
        for stats in (totals, seed_stats):
            stats["changed_groups"] += int(changed)
            stats["exact_oracle_changes"] += int(exact)
            stats["missed_oracle_changes"] += int(missed)
            stats["false_changes"] += int(false)
            stats["wrong_changes"] += int(wrong)
            stats["parseq_delta_sum_vs_nearest"] += parseq_delta
            stats["tesseract_delta_sum_vs_nearest"] += tesseract_delta
            stats["parseq_worsen_groups"] += int(parseq_delta > 0)
            stats["tesseract_improve_groups"] += int(tesseract_delta < 0)
            stats["tesseract_worsen_groups"] += int(tesseract_delta > 0)
        seed_stats["groups"] += 1
        if bool(row.get("oracle_in_shortlist")):
            totals["shortlist_oracle_contained_groups"] += 1
            seed_stats["shortlist_oracle_contained_groups"] += 1
        totals["shortlist_size_mean"] += float(row["shortlist_size"])
        seed_stats["shortlist_size_mean"] = seed_stats.get("shortlist_size_sum", 0.0) + float(row["shortlist_size"])
        seed_stats["shortlist_size_sum"] = seed_stats["shortlist_size_mean"]
    if chosen:
        totals["shortlist_size_mean"] /= len(chosen)
    for stats in by_real_seed.values():
        groups = int(stats["groups"])
        stats["shortlist_size_mean"] = float(stats.pop("shortlist_size_sum", 0.0)) / groups if groups else 0.0
    return {"aggregate": totals, "by_real_seed": by_real_seed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--score-model-seed", type=int, default=2)
    parser.add_argument("--shortlist-topk", type=int, default=4)
    args = parser.parse_args()

    table_rows = [
        row for row in read_jsonl(args.table)
        if int(row["seed"]) == int(args.val_seed)
    ]
    if not table_rows:
        raise SystemExit(f"no table rows matched seed={args.val_seed}: {args.table}")
    groups = group_rows(table_rows)
    oracle_scores = load_score_rows(args.oracle_score, val_seed=args.val_seed, model_seed=args.score_model_seed)
    safe_scores = load_score_rows(args.safe_score, val_seed=args.val_seed, model_seed=args.score_model_seed)
    chosen: list[dict[str, Any]] = []
    policy_name = f"shortlist_oracle_top{args.shortlist_topk}_scoreseed{args.score_model_seed}"
    for key, rows in sorted(groups.items()):
        shortlist, shortlist_info = build_shortlist(rows, oracle_scores, safe_scores, topk=args.shortlist_topk)
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        if oracle is None:
            raise SystemExit(f"group has no assignment oracle row: {key}")
        oracle_in_shortlist = row_key(oracle) in shortlist
        selected = oracle if oracle_in_shortlist else nearest
        out = {
            "source_index": int(selected["source_index"]),
            "candidate_index": int(selected["candidate_index"]),
            "assignment_code_index": int(selected["code_index"]),
            "code_index": int(selected["code_index"]),
            "nearest_code": int(selected["nearest_code"]),
            "topk_rank": int(selected["topk_rank"]),
            "selected_by_policy": True,
            "selector_score": 1.0 if oracle_in_shortlist else 0.0,
            "policy": policy_name,
            "seed": int(selected["seed"]),
            "real_seed": int(selected.get("real_seed", selected["seed"])),
            "score_val_seed": int(args.val_seed),
            "score_model_seed": int(args.score_model_seed),
            "score_model_seeds": [int(args.score_model_seed)],
            "shortlist_topk": int(args.shortlist_topk),
            "shortlist_size": int(shortlist_info["shortlist_size"]),
            "oracle_in_shortlist": bool(oracle_in_shortlist),
            "parseq_delta_vs_nearest": int(selected.get("parseq_delta_vs_nearest", 0)),
            "tesseract_delta_vs_nearest": int(selected.get("tesseract_delta_vs_nearest", 0)),
            "label_assignment_oracle_choice": int(selected.get("label_assignment_oracle_choice", 0)),
            "group_oracle_changes_code": int(selected.get("group_oracle_changes_code", 0)),
            "assignment_oracle_code_index": int(oracle["code_index"]),
            "source_image": selected.get("source_image"),
            "image_path": selected.get("image_path"),
        }
        chosen.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "description": "Non-deployable oracle over a deployable score-defined top-k shortlist; if the oracle code is outside the shortlist, fall back to nearest/no-op.",
        "table": str(args.table),
        "oracle_score": str(args.oracle_score),
        "safe_score": str(args.safe_score),
        "output": str(args.output),
        "val_seed": int(args.val_seed),
        "score_model_seed": int(args.score_model_seed),
        "shortlist_topk": int(args.shortlist_topk),
        "policy": policy_name,
        **summarize(chosen, groups),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
