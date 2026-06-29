#!/usr/bin/env python3
"""Audit supervision density for the next top-8 assignment verifier.

This is not a selector and does not export bitstreams. It quantifies the
candidate-level and group-level label inventory that a future verifier would
see on the N117 OOF shortlist substrate, so that the next implementation does
not merely resweep the already failed proposal/risk knobs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def ranked_oof_codes(
    rows: list[dict[str, Any]],
    scores: dict[tuple[int, int, int, int, int], float],
    count: int,
) -> list[int]:
    ranked: list[tuple[float, int, int, int]] = []
    for row in rows:
        if int(row.get("is_nearest", 0)):
            continue
        score = scores.get(bottleneck.candidate_key(row))
        if score is None or not np.isfinite(score):
            continue
        ranked.append((float(score), -int(row.get("topk_rank", 999)), -int(row["code_index"]), int(row["code_index"])))
    ranked.sort(reverse=True)
    return [code for _, _, _, code in ranked[:count]]


def shortlist_codes(
    rows: list[dict[str, Any]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
) -> set[int]:
    codes = {
        int(row["code_index"])
        for row in rows
        if not int(row.get("is_nearest", 0)) and int(row.get("topk_rank", 999)) <= topk_cap
    }
    codes.update(ranked_oof_codes(rows, scores, score_count))
    return codes


def summarize_frontier(
    covered_oracle_rows: list[dict[str, Any]],
    *,
    floor_tesseract_delta: int,
) -> dict[str, Any]:
    sorted_rows = sorted(
        covered_oracle_rows,
        key=lambda row: (int(row.get("tesseract_delta_vs_nearest", 0)), int(row.get("topk_rank", 999))),
    )
    cumulative = 0
    trace: list[dict[str, Any]] = []
    min_to_match = None
    min_to_beat = None
    for idx, row in enumerate(sorted_rows, start=1):
        cumulative += int(row.get("tesseract_delta_vs_nearest", 0))
        item = {
            "k": idx,
            "cumulative_tesseract_delta": int(cumulative),
            "source": row.get("source"),
            "source_index": int(row.get("source_index", -1)),
            "reference": row.get("reference"),
            "oracle_code": int(row["code_index"]),
            "oracle_topk_rank": int(row.get("topk_rank", -1)),
            "oracle_tesseract_delta_vs_nearest": int(row.get("tesseract_delta_vs_nearest", 0)),
            "oracle_parseq_delta_vs_nearest": int(row.get("parseq_delta_vs_nearest", 0)),
        }
        if idx <= 12:
            trace.append(item)
        if min_to_match is None and cumulative <= floor_tesseract_delta:
            min_to_match = idx
        if min_to_beat is None and cumulative < floor_tesseract_delta:
            min_to_beat = idx
    return {
        "covered_oracle_rows": int(len(sorted_rows)),
        "floor_tesseract_delta": int(floor_tesseract_delta),
        "best_possible_cumulative_tesseract_delta": int(cumulative),
        "min_exact_changes_to_match_floor": min_to_match,
        "min_exact_changes_to_beat_floor": min_to_beat,
        "best_first_12_trace": trace,
    }


def summarize_partition(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
    source_modulo: int,
    floor_tesseract_delta: int,
) -> dict[str, Any]:
    counts = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_mod: dict[str, Counter] = defaultdict(Counter)
    oracle_topk_ranks: list[int] = []
    covered_oracle_rows: list[dict[str, Any]] = []
    missed_oracle_examples: list[dict[str, Any]] = []
    noheadroom_examples: list[dict[str, Any]] = []

    for key, rows in sorted(groups.items()):
        first = rows[0]
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        oracle_changes = oracle_code != nearest_code
        source = str(first.get("source", "unknown"))
        mod_key = f"r{int(first['source_index']) % source_modulo}"
        codes = shortlist_codes(rows, scores, topk_cap=topk_cap, score_count=score_count)
        nonnearest_rows = [row for row in rows if not int(row.get("is_nearest", 0))]
        shortlist_rows = [row for row in nonnearest_rows if int(row["code_index"]) in codes]
        positive_in_shortlist = oracle_changes and oracle_code in codes

        counts["groups"] += 1
        counts["oracle_change_groups"] += int(oracle_changes)
        counts["noheadroom_groups"] += int(not oracle_changes)
        counts["shortlist_groups"] += int(bool(shortlist_rows))
        counts["candidate_rows"] += len(nonnearest_rows)
        counts["shortlist_candidate_rows"] += len(shortlist_rows)
        counts["positive_candidate_rows"] += int(positive_in_shortlist)
        counts["shortlist_wrong_candidate_rows"] += max(0, len(shortlist_rows) - int(positive_in_shortlist))
        counts["noheadroom_shortlist_candidate_rows"] += len(shortlist_rows) if not oracle_changes else 0
        counts["oracle_covered_by_shortlist"] += int(positive_in_shortlist)
        counts["oracle_missed_by_shortlist"] += int(oracle_changes and not positive_in_shortlist)
        if oracle_changes:
            oracle_topk_ranks.append(int(oracle.get("topk_rank", 999)))
            for cap in [1, 2, 4, 8]:
                counts[f"oracle_topk_rank_le{cap}"] += int(int(oracle.get("topk_rank", 999)) <= cap)
        if positive_in_shortlist:
            covered_oracle_rows.append(oracle)
        if oracle_changes and not positive_in_shortlist and len(missed_oracle_examples) < 24:
            missed_oracle_examples.append(
                {
                    "group_key": list(key),
                    "source": source,
                    "reference": first.get("reference"),
                    "oracle_code": oracle_code,
                    "oracle_topk_rank": int(oracle.get("topk_rank", -1)),
                    "oracle_tesseract_delta_vs_nearest": int(oracle.get("tesseract_delta_vs_nearest", 0)),
                    "shortlist_size": len(shortlist_rows),
                }
            )
        if (not oracle_changes) and shortlist_rows and len(noheadroom_examples) < 24:
            best = sorted(shortlist_rows, key=lambda row: (int(row.get("topk_rank", 999)), int(row["code_index"])))[0]
            noheadroom_examples.append(
                {
                    "group_key": list(key),
                    "source": source,
                    "reference": first.get("reference"),
                    "best_shortlist_code": int(best["code_index"]),
                    "best_shortlist_topk_rank": int(best.get("topk_rank", -1)),
                    "best_shortlist_tesseract_delta_vs_nearest": int(best.get("tesseract_delta_vs_nearest", 0)),
                    "shortlist_size": len(shortlist_rows),
                }
            )

        for bucket in [by_source[source], by_mod[mod_key]]:
            bucket["groups"] += 1
            bucket["oracle_change_groups"] += int(oracle_changes)
            bucket["noheadroom_groups"] += int(not oracle_changes)
            bucket["oracle_covered_by_shortlist"] += int(positive_in_shortlist)
            bucket["oracle_missed_by_shortlist"] += int(oracle_changes and not positive_in_shortlist)
            bucket["shortlist_candidate_rows"] += len(shortlist_rows)
            bucket["positive_candidate_rows"] += int(positive_in_shortlist)
            bucket["noheadroom_shortlist_candidate_rows"] += len(shortlist_rows) if not oracle_changes else 0
            bucket["wrong_candidate_rows"] += max(0, len(shortlist_rows) - int(positive_in_shortlist))

    candidate_positive = counts["positive_candidate_rows"]
    candidate_negative = counts["shortlist_candidate_rows"] - candidate_positive
    group_positive = counts["oracle_covered_by_shortlist"]
    group_nochange = counts["noheadroom_groups"] + counts["oracle_missed_by_shortlist"]
    class_balance = {
        "candidate_positive_rows": int(candidate_positive),
        "candidate_negative_rows": int(candidate_negative),
        "candidate_negative_per_positive": float(candidate_negative / candidate_positive) if candidate_positive else None,
        "group_change_positive_labels": int(group_positive),
        "group_nochange_or_uncovered_labels": int(group_nochange),
        "group_nochange_per_change_positive": float(group_nochange / group_positive) if group_positive else None,
    }
    return {
        "counts": {key: int(value) for key, value in counts.items()},
        "class_balance": class_balance,
        "oracle_topk_rank_stats": {
            "count": int(len(oracle_topk_ranks)),
            "mean": float(np.mean(oracle_topk_ranks)) if oracle_topk_ranks else None,
            "max": int(max(oracle_topk_ranks)) if oracle_topk_ranks else None,
        },
        "best_exact_frontier": summarize_frontier(covered_oracle_rows, floor_tesseract_delta=floor_tesseract_delta),
        "by_source": {key: {k: int(v) for k, v in value.items()} for key, value in sorted(by_source.items())},
        "by_source_mod": {key: {k: int(v) for k, v in value.items()} for key, value in sorted(by_mod.items())},
        "missed_oracle_examples": missed_oracle_examples,
        "noheadroom_examples": noheadroom_examples,
    }


def compact_counts(entry: dict[str, Any]) -> str:
    counts = entry["counts"]
    balance = entry["class_balance"]
    return (
        f"{counts.get('oracle_covered_by_shortlist', 0)}/"
        f"{counts.get('oracle_change_groups', 0)} covered, "
        f"noheadroom {counts.get('noheadroom_groups', 0)}, "
        f"cand +/- {balance.get('candidate_positive_rows', 0)}/"
        f"{balance.get('candidate_negative_rows', 0)}"
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "Diagnostic-only supervision-density audit for the N117 OOF top-8 shortlist. It uses OCR deltas only as labels/evaluation and does not train or promote a selector.",
        "",
        "## Summary",
        "",
        "| partition | label inventory | min exact to beat current floor | best covered-oracle Tesseract |",
        "|---|---|---:|---:|",
    ]
    for partition in ["train", "val", "all"]:
        entry = result["partitions"].get(partition)
        if not entry:
            continue
        frontier = entry["best_exact_frontier"]
        lines.append(
            f"| {partition} | {compact_counts(entry)} | "
            f"{frontier['min_exact_changes_to_beat_floor']} | "
            f"{frontier['best_possible_cumulative_tesseract_delta']} |"
        )
    lines.extend(
        [
            "",
            "## Class Balance",
            "",
            "| partition | candidate negative / positive | group no-change / change-positive |",
            "|---|---:|---:|",
        ]
    )
    for partition in ["train", "val", "all"]:
        entry = result["partitions"].get(partition)
        if not entry:
            continue
        bal = entry["class_balance"]
        cand = bal["candidate_negative_per_positive"]
        group = bal["group_nochange_per_change_positive"]
        lines.append(
            f"| {partition} | "
            f"{cand:.2f}" if cand is not None else f"| {partition} | n/a"
        )
        lines[-1] += f" | {group:.2f} |" if group is not None else " | n/a |"
    val = result["partitions"].get("val", {})
    if val:
        lines.extend(["", "## Validation Source Breakdown", "", "| source | groups | oracle | covered | no-headroom | candidate +/- |"])
        lines.append("|---|---:|---:|---:|---:|---:|")
        for source, row in val["by_source"].items():
            lines.append(
                f"| {source} | {row.get('groups', 0)} | {row.get('oracle_change_groups', 0)} | "
                f"{row.get('oracle_covered_by_shortlist', 0)} | {row.get('noheadroom_groups', 0)} | "
                f"{row.get('positive_candidate_rows', 0)}/{row.get('wrong_candidate_rows', 0)} |"
            )
        lines.extend(["", "## Validation Modulo Breakdown", "", "| source_mod | groups | oracle | covered | no-headroom | candidate +/- |"])
        lines.append("|---|---:|---:|---:|---:|---:|")
        for mod, row in val["by_source_mod"].items():
            lines.append(
                f"| {mod} | {row.get('groups', 0)} | {row.get('oracle_change_groups', 0)} | "
                f"{row.get('oracle_covered_by_shortlist', 0)} | {row.get('noheadroom_groups', 0)} | "
                f"{row.get('positive_candidate_rows', 0)}/{row.get('wrong_candidate_rows', 0)} |"
            )
        lines.extend(["", "## Best Exact Frontier"])
        lines.append("")
        lines.append("This is a non-deployable ordering of covered oracle changes. It estimates how many exact useful changes a future verifier must accept to beat the current actual-bitstream floor without bad changes.")
        lines.extend(["", "| k | cumulative Tesseract | source | reference | topk | delta |", "|---:|---:|---|---|---:|---:|"])
        for item in val["best_exact_frontier"]["best_first_12_trace"]:
            lines.append(
                f"| {item['k']} | {item['cumulative_tesseract_delta']} | {item['source']} | "
                f"{item['reference']} | {item['oracle_topk_rank']} | {item['oracle_tesseract_delta_vs_nearest']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["interpretation"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_start_wandb(args: argparse.Namespace, result: dict[str, Any]) -> Any | None:
    if args.no_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.experiment_id,
        mode=args.wandb_mode,
        config=result["config"],
        tags=["top8", "supervision-gap", "diagnostic"],
    )


def finish_wandb(run: Any | None, result: dict[str, Any], output: Path, report: Path) -> None:
    if run is None:
        return
    import wandb

    val = result["partitions"].get("val", {})
    metrics: dict[str, Any] = {}
    if val:
        counts = val["counts"]
        bal = val["class_balance"]
        frontier = val["best_exact_frontier"]
        metrics = {
            "val/oracle_change_groups": counts.get("oracle_change_groups", 0),
            "val/oracle_covered_by_shortlist": counts.get("oracle_covered_by_shortlist", 0),
            "val/oracle_missed_by_shortlist": counts.get("oracle_missed_by_shortlist", 0),
            "val/noheadroom_groups": counts.get("noheadroom_groups", 0),
            "val/candidate_negative_per_positive": bal.get("candidate_negative_per_positive"),
            "val/group_nochange_per_change_positive": bal.get("group_nochange_per_change_positive"),
            "val/min_exact_to_beat_floor": frontier.get("min_exact_changes_to_beat_floor"),
            "val/best_covered_oracle_tesseract_delta": frontier.get("best_possible_cumulative_tesseract_delta"),
        }
    wandb.log(metrics)
    artifact = wandb.Artifact(result["experiment_id"].replace("/", "_"), type="diagnostic")
    artifact.add_file(str(output))
    artifact.add_file(str(report))
    run.log_artifact(artifact)
    run.finish()


def build_manifest(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    score_inputs = {}
    for idx, path in enumerate(args.score_file):
        score_inputs[f"score_r{idx}"] = {"path": rel(path), "sha256": sha256_file(path)}
    return {
        "experiment_id": result["experiment_id"],
        "hypothesis_id": "H4-top8-verifier-supervision-gap",
        "status": "completed",
        "code_commit": result["code_commit"],
        "command": result["command"],
        "wandb_run_id": result.get("wandb_run_id"),
        "inputs": {
            "table": {"path": rel(args.table), "sha256": sha256_file(args.table)},
            **score_inputs,
        },
        "outputs": {
            "result_json": {"path": rel(args.output), "sha256": sha256_file(args.output)},
            "report": {"path": rel(args.report), "sha256": sha256_file(args.report)},
        },
        "scripts": {
            "analyzer": {"path": rel(ROOT / "scripts/analyze_top8_verifier_supervision_gap.py"), "sha256": sha256_file(ROOT / "scripts/analyze_top8_verifier_supervision_gap.py")},
            "oof_shortlist": {"path": rel(ROOT / "scripts/analyze_top8_source_mod_oof_shortlist.py"), "sha256": sha256_file(ROOT / "scripts/analyze_top8_source_mod_oof_shortlist.py")},
            "bottleneck": {"path": rel(ROOT / "scripts/analyze_top8_two_stage_bottleneck.py"), "sha256": sha256_file(ROOT / "scripts/analyze_top8_two_stage_bottleneck.py")},
        },
        "conclusion": result["interpretation"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, action="append", required=True)
    parser.add_argument("--source-modulo", type=int, default=5)
    parser.add_argument("--topk-cap", type=int, default=4)
    parser.add_argument("--score-count", type=int, default=4)
    parser.add_argument("--current-floor-tesseract-delta", type=int, default=-8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_verifier_supervision_gap_2026_06_26")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oscar-lic")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.table = args.table.resolve()
    args.score_file = [path.resolve() for path in args.score_file]
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    args.manifest = args.manifest.resolve()

    groups_all, by_candidate = bottleneck.load_table(args.table)
    splits = bottleneck.split_groups(groups_all)
    oof_scores, oof_meta = oof_shortlist.build_oof_scores(args.score_file, by_candidate, modulo=args.source_modulo)

    partitions = {}
    for name in ["train", "val", "all"]:
        if name in splits:
            partitions[name] = summarize_partition(
                splits[name],
                oof_scores,
                topk_cap=args.topk_cap,
                score_count=args.score_count,
                source_modulo=args.source_modulo,
                floor_tesseract_delta=args.current_floor_tesseract_delta,
            )

    val = partitions.get("val", {})
    if val:
        counts = val["counts"]
        frontier = val["best_exact_frontier"]
        interpretation = (
            "N117 shortlist supervision remains label-sparse and no-headroom dominated: "
            f"validation covers {counts.get('oracle_covered_by_shortlist', 0)}/"
            f"{counts.get('oracle_change_groups', 0)} oracle-change groups, but has "
            f"{counts.get('noheadroom_groups', 0)} no-headroom groups and "
            f"{val['class_balance'].get('candidate_negative_per_positive')} shortlist candidate negatives per positive. "
            f"A verifier must accept at least {frontier.get('min_exact_changes_to_beat_floor')} exact useful changes "
            f"with near-zero bad moves to beat the current actual-bitstream Tesseract floor "
            f"({args.current_floor_tesseract_delta}). This supports adding clean no-headroom/hard-positive supervision "
            "or changing the verification objective before any further N117/N120 score-risk tuning."
        )
    else:
        interpretation = "No validation partition found; diagnostic cannot estimate promotion target."

    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_supervision_gap_not_a_selector",
        "code_commit": git_commit(),
        "command": " ".join(sys.argv),
        "config": {
            "topk_cap": args.topk_cap,
            "score_count": args.score_count,
            "source_modulo": args.source_modulo,
            "current_floor_tesseract_delta": args.current_floor_tesseract_delta,
        },
        "inputs": {
            "table": {"path": rel(args.table), "sha256": sha256_file(args.table)},
            "oof_scores": oof_meta,
        },
        "partitions": partitions,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(ROOT / "scripts/analyze_top8_verifier_supervision_gap.py"),
            "oof_shortlist_script": sha256_file(ROOT / "scripts/analyze_top8_source_mod_oof_shortlist.py"),
            "bottleneck_script": sha256_file(ROOT / "scripts/analyze_top8_two_stage_bottleneck.py"),
        },
    }

    run = maybe_start_wandb(args, result)
    if run is not None:
        result["wandb_run_id"] = run.id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.report, result)
    finish_wandb(run, result, args.output, args.report)

    manifest = build_manifest(args, result)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    print(json.dumps({"experiment_id": args.experiment_id, "wandb_run_id": result.get("wandb_run_id"), "interpretation": interpretation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
