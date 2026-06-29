#!/usr/bin/env python3
"""Select train-only hard cohorts for the next top-8 verifier objective.

The output is a training-supervision manifest, not a selector policy. It keeps
validation rows out of the cohort and pairs each covered train hard-positive
oracle row with source/rank-similar no-headroom candidates from the same N117
OOF shortlist substrate.
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

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_top8_source_mod_oof_shortlist as oof_shortlist  # noqa: E402
import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402
from analyze_top8_verifier_supervision_gap import shortlist_codes  # noqa: E402


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


def row_score(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row.get("topk_rank", 999)),
        float(row.get("assignment_relative_error", 0.0)),
        int(row["code_index"]),
    )


def cohort_base(
    row: dict[str, Any],
    *,
    label: int,
    cohort_type: str,
    group_key: tuple[int, int, int, int],
    source_modulo: int,
    positive_id: str | None = None,
) -> dict[str, Any]:
    return {
        "cohort_id": None,
        "matched_positive_id": positive_id,
        "target_label": int(label),
        "cohort_type": cohort_type,
        "assignment_partition": row.get("assignment_partition", row.get("split")),
        "group_key": list(group_key),
        "real_seed": int(row.get("real_seed", row.get("seed", 0))),
        "seed": int(row.get("seed", row.get("original_seed", 0))),
        "source": row.get("source"),
        "source_index": int(row["source_index"]),
        "source_mod": int(row["source_index"]) % source_modulo,
        "candidate_index": int(row["candidate_index"]),
        "candidate_slot": int(row.get("candidate_slot", -1)),
        "code_index": int(row["code_index"]),
        "nearest_code": int(row["nearest_code"]),
        "topk_rank": int(row.get("topk_rank", -1)),
        "assignment_relative_error": float(row.get("assignment_relative_error", 0.0)),
        "reference": row.get("reference"),
        "reference_length": int(row.get("reference_length", len(str(row.get("reference", ""))))),
        "tesseract_delta_vs_nearest": int(row.get("tesseract_delta_vs_nearest", 0)),
        "parseq_delta_vs_nearest": int(row.get("parseq_delta_vs_nearest", 0)),
        "training_feature_note": "labels may use OCR deltas; future model inputs must exclude OCR strings/deltas/reference fields",
    }


def collect_candidates(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    topk_cap: int,
    score_count: int,
    source_modulo: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    noheadroom: list[dict[str, Any]] = []
    counters = Counter()
    for group_key, rows in sorted(groups.items()):
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        codes = shortlist_codes(rows, scores, topk_cap=topk_cap, score_count=score_count)
        shortlist_rows = [
            row for row in rows if not int(row.get("is_nearest", 0)) and int(row["code_index"]) in codes
        ]
        oracle_changes = oracle_code != nearest_code
        counters["groups"] += 1
        counters["oracle_change_groups"] += int(oracle_changes)
        counters["noheadroom_groups"] += int(not oracle_changes)
        if oracle_changes and oracle_code in codes:
            item = cohort_base(
                oracle,
                label=1,
                cohort_type="covered_oracle_positive",
                group_key=group_key,
                source_modulo=source_modulo,
            )
            # Higher priority for large OCR gains and deep-rank hard positives.
            item["positive_priority"] = (
                -float(oracle.get("tesseract_delta_vs_nearest", 0))
                + 0.1 * float(oracle.get("topk_rank", 0))
            )
            positives.append(item)
            counters["covered_oracle_positive"] += 1
        elif oracle_changes:
            counters["missed_oracle_positive"] += 1
        if not oracle_changes and shortlist_rows:
            best = sorted(shortlist_rows, key=row_score)[0]
            item = cohort_base(
                best,
                label=0,
                cohort_type="matched_noheadroom_candidate",
                group_key=group_key,
                source_modulo=source_modulo,
            )
            item["negative_priority"] = (
                1.0 / max(1.0, float(best.get("topk_rank", 999)))
                + 0.01 * float(best.get("img_source_variant_changed_fraction", 0.0))
            )
            noheadroom.append(item)
            counters["available_noheadroom_candidate"] += 1
    return positives, noheadroom, {key: int(value) for key, value in counters.items()}


def match_negatives(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    negatives_per_positive: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    unused = {idx for idx in range(len(negatives))}
    positives_sorted = sorted(
        positives,
        key=lambda row: (-float(row.get("positive_priority", 0.0)), int(row.get("topk_rank", 999))),
    )
    for pos_idx, positive in enumerate(positives_sorted):
        positive_id = f"pos_{pos_idx:04d}"
        positive["cohort_id"] = positive_id
        positive["matched_positive_id"] = positive_id
        selected.append(positive)
        chosen: list[int] = []
        while unused and len(chosen) < negatives_per_positive:
            def distance(index: int) -> tuple[float, int]:
                neg = negatives[index]
                same_source_penalty = 0.0 if neg["source"] == positive["source"] else 6.0
                same_mod_penalty = 0.0 if neg["source_mod"] == positive["source_mod"] else 2.0
                length_penalty = abs(int(neg["reference_length"]) - int(positive["reference_length"])) / 8.0
                rank_penalty = abs(int(neg["topk_rank"]) - int(positive["topk_rank"])) / 4.0
                priority_bonus = -0.1 * float(neg.get("negative_priority", 0.0))
                return (same_source_penalty + same_mod_penalty + length_penalty + rank_penalty + priority_bonus, index)

            best = min(unused, key=distance)
            unused.remove(best)
            chosen.append(best)
        for local_idx, neg_idx in enumerate(chosen):
            negative = dict(negatives[neg_idx])
            negative["cohort_id"] = f"{positive_id}_neg_{local_idx:02d}"
            negative["matched_positive_id"] = positive_id
            selected.append(negative)
    return selected


def summarize(rows: list[dict[str, Any]], inventory: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(row["cohort_type"] for row in rows)
    sources = Counter(str(row.get("source")) for row in rows)
    positives = [row for row in rows if int(row["target_label"]) == 1]
    negatives = [row for row in rows if int(row["target_label"]) == 0]
    return {
        "rows": int(len(rows)),
        "positive_rows": int(len(positives)),
        "negative_rows": int(len(negatives)),
        "negative_per_positive": float(len(negatives) / len(positives)) if positives else None,
        "cohort_type_counts": {key: int(value) for key, value in sorted(counts.items())},
        "source_counts": {key: int(value) for key, value in sorted(sources.items())},
        "inventory": inventory,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        f"# {result['experiment_id']}",
        "",
        f"W&B: `{result.get('wandb_run_id', 'not_logged')}`",
        "",
        "Train-only hard cohort selection for the next N117 verifier objective. No validation rows are written to the cohort file.",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Positives: `{summary['positive_rows']}`",
        f"- Negatives: `{summary['negative_rows']}`",
        f"- Negative/positive: `{summary['negative_per_positive']}`",
        "",
        "## Source Counts",
        "",
        "| source | rows |",
        "|---|---:|",
    ]
    for source, count in summary["source_counts"].items():
        lines.append(f"| {source} | {count} |")
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
        tags=["top8", "hard-cohort", "diagnostic"],
    )


def finish_wandb(run: Any | None, result: dict[str, Any], output: Path, report: Path, summary_path: Path) -> None:
    if run is None:
        return
    import wandb

    summary = result["summary"]
    wandb.log(
        {
            "cohort/rows": summary["rows"],
            "cohort/positive_rows": summary["positive_rows"],
            "cohort/negative_rows": summary["negative_rows"],
            "cohort/negative_per_positive": summary["negative_per_positive"],
        }
    )
    artifact = wandb.Artifact(result["experiment_id"].replace("/", "_"), type="diagnostic")
    artifact.add_file(str(output))
    artifact.add_file(str(summary_path))
    artifact.add_file(str(report))
    run.log_artifact(artifact)
    run.finish()


def build_manifest(args: argparse.Namespace, result: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    score_inputs = {}
    for idx, path in enumerate(args.score_file):
        score_inputs[f"score_r{idx}"] = {"path": rel(path), "sha256": sha256_file(path)}
    return {
        "experiment_id": result["experiment_id"],
        "hypothesis_id": "H4-top8-verifier-hard-cohorts",
        "status": "completed",
        "code_commit": result["code_commit"],
        "command": result["command"],
        "wandb_run_id": result.get("wandb_run_id"),
        "inputs": {
            "table": {"path": rel(args.table), "sha256": sha256_file(args.table)},
            **score_inputs,
        },
        "outputs": {
            "cohort_jsonl": {"path": rel(args.output), "sha256": sha256_file(args.output)},
            "summary_json": {"path": rel(summary_path), "sha256": sha256_file(summary_path)},
            "report": {"path": rel(args.report), "sha256": sha256_file(args.report)},
        },
        "scripts": {
            "selector": {"path": rel(ROOT / "scripts/select_top8_verifier_hard_cohorts.py"), "sha256": sha256_file(ROOT / "scripts/select_top8_verifier_hard_cohorts.py")},
            "supervision_gap": {"path": rel(ROOT / "scripts/analyze_top8_verifier_supervision_gap.py"), "sha256": sha256_file(ROOT / "scripts/analyze_top8_verifier_supervision_gap.py")},
            "oof_shortlist": {"path": rel(ROOT / "scripts/analyze_top8_source_mod_oof_shortlist.py"), "sha256": sha256_file(ROOT / "scripts/analyze_top8_source_mod_oof_shortlist.py")},
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
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_verifier_hard_cohorts_2026_06_26")
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
    summary_path = (args.summary_output.resolve() if args.summary_output else args.output.with_suffix(args.output.suffix + ".summary.json"))

    groups_all, by_candidate = bottleneck.load_table(args.table)
    splits = bottleneck.split_groups(groups_all)
    if "train" not in splits:
        raise SystemExit("candidate table has no train partition")
    oof_scores, oof_meta = oof_shortlist.build_oof_scores(args.score_file, by_candidate, modulo=args.source_modulo)
    positives, negatives, inventory = collect_candidates(
        splits["train"],
        oof_scores,
        topk_cap=args.topk_cap,
        score_count=args.score_count,
        source_modulo=args.source_modulo,
    )
    rows = match_negatives(positives, negatives, negatives_per_positive=args.negatives_per_positive)
    summary = summarize(rows, inventory)
    interpretation = (
        f"Prepared a train-only hard cohort with {summary['positive_rows']} covered-oracle positives and "
        f"{summary['negative_rows']} matched no-headroom negatives ({summary['negative_per_positive']} negatives per positive). "
        "This is a supervision manifest for a future contrastive/selective verifier, not a selector policy and not validation-tuned."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "train_only_cohort_manifest_not_selector",
        "code_commit": git_commit(),
        "command": " ".join(sys.argv),
        "config": {
            "topk_cap": args.topk_cap,
            "score_count": args.score_count,
            "source_modulo": args.source_modulo,
            "negatives_per_positive": args.negatives_per_positive,
            "partition_written": "train",
        },
        "inputs": {
            "table": {"path": rel(args.table), "sha256": sha256_file(args.table)},
            "oof_scores": oof_meta,
        },
        "summary": summary,
        "interpretation": interpretation,
    }
    run = maybe_start_wandb(args, result)
    if run is not None:
        result["wandb_run_id"] = run.id

    write_jsonl(args.output, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.report, result)
    finish_wandb(run, result, args.output, args.report, summary_path)
    manifest = build_manifest(args, result, summary_path)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    print(json.dumps({"experiment_id": args.experiment_id, "wandb_run_id": result.get("wandb_run_id"), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
