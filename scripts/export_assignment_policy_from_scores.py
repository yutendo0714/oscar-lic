#!/usr/bin/env python3
"""Export learned assignment scores to an explicit code-index selection table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_rows(rows: Iterable[dict]) -> dict[tuple[int, int], list[dict]]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        key = (int(row["source_index"]), int(row["candidate_index"]))
        groups.setdefault(key, []).append(row)
    return groups


def average_model_seed_scores(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int, int, int], list[dict]] = {}
    for row in rows:
        key = (
            int(row["source_index"]),
            int(row["candidate_index"]),
            int(row["code_index"]),
            int(row["topk_rank"]),
        )
        grouped.setdefault(key, []).append(row)
    averaged = []
    for values in grouped.values():
        out = dict(values[0])
        scores = [float(row["score"]) for row in values]
        out["score"] = sum(scores) / len(scores)
        out["score_count"] = len(scores)
        out["score_min"] = min(scores)
        out["score_max"] = max(scores)
        out["score_model_seed"] = "ensemble"
        out["score_model_seeds"] = sorted(int(row["model_seed"]) for row in values)
        averaged.append(out)
    return averaged


def best_score(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: (float(row["score"]), -int(row["topk_rank"])))


def nearest(rows: list[dict]) -> dict:
    matches = [row for row in rows if int(row["is_nearest"])]
    if len(matches) != 1:
        raise SystemExit(f"group has {len(matches)} nearest rows")
    return matches[0]


def choose_rows(groups: dict[tuple[int, int], list[dict]], args: argparse.Namespace) -> list[dict]:
    if args.policy == "always_best_score":
        return [best_score(rows) for rows in groups.values()]
    if args.policy == "threshold":
        if args.score_threshold is None:
            raise SystemExit("--score-threshold is required for --policy threshold")
        chosen = []
        for rows in groups.values():
            best = best_score(rows)
            chosen.append(best if float(best["score"]) >= args.score_threshold else nearest(rows))
        return chosen
    if args.policy == "top_changed":
        if args.changed_budget is None or args.changed_budget < 0:
            raise SystemExit("--changed-budget must be non-negative for --policy top_changed")
        pairs = []
        for rows in groups.values():
            current_best = best_score(rows)
            current_nearest = nearest(rows)
            if int(current_best["code_index"]) == int(current_nearest["code_index"]):
                continue
            margin = float(current_best["score"]) - float(current_nearest["score"])
            pairs.append((margin, float(current_best["score"]), current_best, current_nearest))
        changed = {
            (int(best["source_index"]), int(best["candidate_index"])): best
            for _, _, best, _ in sorted(pairs, key=lambda item: (item[0], item[1]), reverse=True)[: args.changed_budget]
        }
        chosen = []
        for key, rows in groups.items():
            chosen.append(changed.get(key, nearest(rows)))
        return chosen
    raise SystemExit(f"unsupported policy: {args.policy}")


def write_outputs(chosen: list[dict], output: Path, args: argparse.Namespace) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    tesseract_delta_sum = 0
    parseq_delta_sum = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in sorted(chosen, key=lambda item: (int(item["source_index"]), int(item["candidate_index"]))):
            code_index = int(row["code_index"])
            nearest_code = int(row["nearest_code"])
            changed += int(code_index != nearest_code)
            tesseract_delta_sum += int(row.get("tesseract_delta_vs_nearest", 0))
            parseq_delta_sum += int(row.get("parseq_delta_vs_nearest", 0))
            out = {
                "source_index": int(row["source_index"]),
                "candidate_index": int(row["candidate_index"]),
                "assignment_code_index": code_index,
                "code_index": code_index,
                "nearest_code": nearest_code,
                "topk_rank": int(row["topk_rank"]),
                "selected_by_policy": True,
                "selector_score": float(row["score"]),
                "score_val_seed": int(row["val_seed"]),
                "score_model_seed": row.get("score_model_seed", int(row["model_seed"])),
                "score_model_seeds": row.get("score_model_seeds", [int(row["model_seed"])]),
                "score_count": int(row.get("score_count", 1)),
                "score_min": float(row.get("score_min", row["score"])),
                "score_max": float(row.get("score_max", row["score"])),
                "seed": int(row["seed"]),
                "real_seed": int(row.get("real_seed", row["seed"])),
                "policy": args.policy,
                "parseq_delta_vs_nearest": int(row.get("parseq_delta_vs_nearest", 0)),
                "tesseract_delta_vs_nearest": int(row.get("tesseract_delta_vs_nearest", 0)),
                "label_assignment_oracle_choice": int(row.get("label_assignment_oracle_choice", 0)),
                "source_image": row.get("source_image"),
                "image_path": row.get("image_path"),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    summary = {
        "scores": str(args.scores),
        "output": str(output),
        "policy": args.policy,
        "val_seed": args.val_seed,
        "model_seeds": args.model_seed,
        "real_seed": args.real_seed,
        "score_threshold": args.score_threshold,
        "changed_budget": args.changed_budget,
        "groups": len(chosen),
        "changed_groups": changed,
        "tesseract_delta_sum_vs_nearest": tesseract_delta_sum,
        "parseq_delta_sum_vs_nearest": parseq_delta_sum,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--real-seed", type=int)
    parser.add_argument("--policy", choices=["always_best_score", "threshold", "top_changed"], default="always_best_score")
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--changed-budget", type=int)
    args = parser.parse_args()
    if args.model_seed is None:
        args.model_seed = [0]

    rows = [
        row
        for row in read_jsonl(args.scores)
        if int(row["val_seed"]) == args.val_seed
        and int(row["model_seed"]) in set(args.model_seed)
        and int(row["seed"]) == args.val_seed
        and (args.real_seed is None or int(row.get("real_seed", row["seed"])) == args.real_seed)
    ]
    if not rows:
        raise SystemExit("no score rows matched the requested val/model seed")
    rows = average_model_seed_scores(rows)
    groups = group_rows(rows)
    chosen = choose_rows(groups, args)
    write_outputs(chosen, args.output, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
