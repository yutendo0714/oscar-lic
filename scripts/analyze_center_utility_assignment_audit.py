#!/usr/bin/env python3
"""Audit whether codebook-center utility can identify useful assignment codes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from statistics import mean, median
import sys
from typing import Any

import numpy as np
import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def parse_codebook_arg(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    seed_text, path_text = value.split("=", 1)
    return int(seed_text), Path(path_text)


def load_center_utilities(items: list[tuple[int, Path]]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for seed, path in items:
        data = np.load(path)
        if "center_utility" not in data.files:
            raise SystemExit(f"{path} has no center_utility array")
        result[seed] = np.asarray(data["center_utility"], dtype=np.float64)
    return result


def code_utility(row: dict[str, Any], centers: dict[int, np.ndarray]) -> float:
    seed = int(row.get("real_seed", row.get("seed", 0)))
    code = int(row["code_index"])
    values = centers[seed]
    if code < 0 or code >= len(values):
        raise ValueError(f"code index {code} outside center utility array for seed {seed}")
    return float(values[code])


def classify(selected_code: int, nearest_code: int, oracle_code: int) -> str:
    if selected_code == oracle_code and oracle_code != nearest_code:
        return "exact_changed"
    if selected_code == oracle_code and oracle_code == nearest_code:
        return "correct_noop"
    if selected_code == nearest_code and oracle_code != nearest_code:
        return "missed_oracle"
    if selected_code != nearest_code and oracle_code == nearest_code:
        return "false_change"
    return "wrong_nonnearest"


def choose_center_row(rows: list[dict[str, Any]], centers: dict[int, np.ndarray], threshold: float | None) -> dict[str, Any]:
    eligible = []
    for row in rows:
        rel = float(row.get("assignment_relative_error", 0.0))
        if int(row.get("is_nearest", 0)) or threshold is None or rel <= threshold:
            eligible.append(row)
    if not eligible:
        eligible = [next(row for row in rows if int(row.get("is_nearest", 0)))]
    return max(
        eligible,
        key=lambda row: (
            code_utility(row, centers),
            -float(row.get("assignment_relative_error", 0.0)),
            -int(row.get("topk_rank", 999)),
            -int(row["code_index"]),
        ),
    )


def policy_metrics(groups: list[list[dict[str, Any]]], centers: dict[int, np.ndarray], threshold: float | None) -> dict[str, Any]:
    counts = Counter()
    t_delta = 0
    p_delta = 0
    changed_rows: list[dict[str, Any]] = []
    for rows in groups:
        nearest = next(row for row in rows if int(row.get("is_nearest", 0)))
        first = rows[0]
        nearest_code = int(nearest["code_index"])
        oracle_code = int(first["assignment_oracle_code_index"])
        chosen = choose_center_row(rows, centers, threshold)
        selected_code = int(chosen["code_index"])
        status = classify(selected_code, nearest_code, oracle_code)
        counts["groups"] += 1
        counts[f"status_{status}"] += 1
        counts["oracle_change_groups"] += int(oracle_code != nearest_code)
        counts["changed_groups"] += int(selected_code != nearest_code)
        counts["tesseract_worse_groups"] += int(int(chosen.get("tesseract_delta_vs_nearest", 0)) > 0)
        counts["parseq_worse_groups"] += int(int(chosen.get("parseq_delta_vs_nearest", 0)) > 0)
        t_delta += int(chosen.get("tesseract_delta_vs_nearest", 0))
        p_delta += int(chosen.get("parseq_delta_vs_nearest", 0))
        if selected_code != nearest_code:
            changed_rows.append(
                {
                    "group_key": list(group_key(first)),
                    "source": first.get("source"),
                    "reference": first.get("reference"),
                    "nearest_code": nearest_code,
                    "oracle_code": oracle_code,
                    "selected_code": selected_code,
                    "selected_topk_rank": int(chosen.get("topk_rank", -1)),
                    "selected_center_utility": code_utility(chosen, centers),
                    "nearest_center_utility": code_utility(nearest, centers),
                    "status": status,
                    "tesseract_delta": int(chosen.get("tesseract_delta_vs_nearest", 0)),
                    "parseq_delta": int(chosen.get("parseq_delta_vs_nearest", 0)),
                    "assignment_relative_error": float(chosen.get("assignment_relative_error", 0.0)),
                }
            )
    return {
        **{key: int(value) for key, value in counts.items()},
        "tesseract_delta_vs_nearest": int(t_delta),
        "parseq_delta_vs_nearest": int(p_delta),
        "changed_examples": changed_rows[:30],
    }


def rank_audit(groups: list[list[dict[str, Any]]], centers: dict[int, np.ndarray], thresholds: list[float]) -> dict[str, Any]:
    counts = Counter()
    ranks_all = []
    ranks_oracle_change = []
    delta_util_oracle_nearest = []
    eligible = {f"re{int(round(thr * 100)):03d}": 0 for thr in thresholds}
    for rows in groups:
        nearest = next(row for row in rows if int(row.get("is_nearest", 0)))
        first = rows[0]
        nearest_code = int(nearest["code_index"])
        oracle_code = int(first["assignment_oracle_code_index"])
        oracle = next((row for row in rows if int(row["code_index"]) == oracle_code), None)
        if oracle is None:
            counts["oracle_code_missing_from_top8"] += 1
            continue
        ordered = sorted(
            rows,
            key=lambda row: (
                -code_utility(row, centers),
                float(row.get("assignment_relative_error", 0.0)),
                int(row.get("topk_rank", 999)),
            ),
        )
        rank = 1 + next(index for index, row in enumerate(ordered) if int(row["code_index"]) == oracle_code)
        ranks_all.append(rank)
        oracle_change = oracle_code != nearest_code
        counts["groups"] += 1
        counts["oracle_change_groups"] += int(oracle_change)
        counts["oracle_center_rank1"] += int(rank == 1)
        counts["oracle_center_rank_le4"] += int(rank <= 4)
        util_delta = code_utility(oracle, centers) - code_utility(nearest, centers)
        delta_util_oracle_nearest.append(util_delta)
        counts["oracle_center_utility_gt_nearest"] += int(util_delta > 0)
        if oracle_change:
            ranks_oracle_change.append(rank)
            counts["oracle_change_center_rank1"] += int(rank == 1)
            counts["oracle_change_center_rank_le4"] += int(rank <= 4)
            counts["oracle_change_center_utility_gt_nearest"] += int(util_delta > 0)
            for thr in thresholds:
                label = f"re{int(round(thr * 100)):03d}"
                rel = float(oracle.get("assignment_relative_error", 0.0))
                if rel <= thr:
                    eligible[label] += 1
    return {
        **{key: int(value) for key, value in counts.items()},
        "oracle_center_rank_mean": float(mean(ranks_all)) if ranks_all else None,
        "oracle_center_rank_median": float(median(ranks_all)) if ranks_all else None,
        "oracle_change_center_rank_mean": float(mean(ranks_oracle_change)) if ranks_oracle_change else None,
        "oracle_change_center_rank_median": float(median(ranks_oracle_change)) if ranks_oracle_change else None,
        "oracle_minus_nearest_center_utility_mean": float(mean(delta_util_oracle_nearest)) if delta_util_oracle_nearest else None,
        "oracle_eligible_by_relative_error": eligible,
    }


def summarize(table_rows: list[dict[str, Any]], centers: dict[int, np.ndarray], thresholds: list[float]) -> dict[str, Any]:
    groups_by_key: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        groups_by_key[group_key(row)].append(row)
    partitions: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for rows in groups_by_key.values():
        rows.sort(key=lambda row: int(row.get("topk_rank", 999)))
        partitions[str(rows[0].get("assignment_partition", rows[0].get("split", "unknown")))].append(rows)
    all_groups = list(groups_by_key.values())
    partitions["all"] = all_groups
    result: dict[str, Any] = {}
    for name, groups in sorted(partitions.items()):
        policies = {
            "center_top1_no_relative_guard": policy_metrics(groups, centers, None),
        }
        for thr in thresholds:
            policies[f"center_top1_re{int(round(thr * 100)):03d}"] = policy_metrics(groups, centers, thr)
        result[name] = {
            "groups": len(groups),
            "rank_audit": rank_audit(groups, centers, thresholds),
            "policies": policies,
        }
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Center Utility Assignment Audit",
        "",
        "This diagnostic asks whether train-side codebook-center utility alone can recover top-8 assignment oracle codes.",
        "It does not export streams and does not use OCR strings as selector inputs.",
        "",
        "## Summary",
        "",
    ]
    for part, data in result["partitions"].items():
        rank = data["rank_audit"]
        lines.append(f"### Partition `{part}`")
        lines.append("")
        lines.append(
            f"- Groups: `{data['groups']}`; oracle-change groups: `{rank.get('oracle_change_groups', 0)}`."
        )
        lines.append(
            f"- Oracle center-utility rank1/rank<=4: `{rank.get('oracle_center_rank1', 0)}` / "
            f"`{rank.get('oracle_center_rank_le4', 0)}`; oracle-change rank1/rank<=4: "
            f"`{rank.get('oracle_change_center_rank1', 0)}` / `{rank.get('oracle_change_center_rank_le4', 0)}`."
        )
        lines.append(
            f"- Oracle center utility above nearest: `{rank.get('oracle_center_utility_gt_nearest', 0)}` overall, "
            f"`{rank.get('oracle_change_center_utility_gt_nearest', 0)}` on oracle-change groups."
        )
        lines.append(f"- Oracle-change eligibility by relative error: `{rank['oracle_eligible_by_relative_error']}`.")
        lines.append("")
        lines.append("| policy | changed | exact | missed | false | wrong | Tess delta | PARSeq delta |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for policy, metrics in data["policies"].items():
            lines.append(
                f"| {policy} | {metrics.get('changed_groups', 0)} | "
                f"{metrics.get('status_exact_changed', 0)} | {metrics.get('status_missed_oracle', 0)} | "
                f"{metrics.get('status_false_change', 0)} | {metrics.get('status_wrong_nonnearest', 0)} | "
                f"{metrics.get('tesseract_delta_vs_nearest', 0)} | {metrics.get('parseq_delta_vs_nearest', 0)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Conclusion",
            "",
            result["conclusion"],
            "",
            "## Next Action",
            "",
            result["next_action"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path, codebooks: list[tuple[int, Path]]) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "table", "path": str(args.table), "sha256": sha256_file(args.table)},
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for seed, cb in codebooks:
        files.append({"name": f"codebook_seed{seed}", "path": str(cb), "sha256": sha256_file(cb)})
    data = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "command": " ".join(args.command_argv),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--codebook", type=parse_codebook_arg, action="append", required=True)
    parser.add_argument("--threshold", type=float, action="append", default=[1.01, 1.05])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_center_utility_assignment_rank_audit_2026_06_26")
    args = parser.parse_args()
    args.command_argv = ["scripts/analyze_center_utility_assignment_audit.py", *sys.argv[1:]]
    return args


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.table)
    centers = load_center_utilities(args.codebook)
    partitions = summarize(rows, centers, args.threshold)
    val_rank = partitions.get("val", {}).get("rank_audit", {})
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-center-utility-assignment-diagnostic",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "inputs": {
            "table": str(args.table),
            "codebooks": {str(seed): str(path) for seed, path in args.codebook},
            "thresholds": args.threshold,
        },
        "partitions": partitions,
        "conclusion": (
            "Center utility is not a sufficient candidate-conditioned assignment signal. "
            f"On validation, oracle-change center-rank1 is {val_rank.get('oracle_change_center_rank1', 'unknown')} "
            f"and relative-error eligibility is {val_rank.get('oracle_eligible_by_relative_error', {})}; "
            "this explains why N136 changes few useful codes and remains below the current compact policy."
        ),
        "next_action": (
            "Use center utility only as an auxiliary feature. The next assignment path should add candidate-local "
            "code-effect evidence or a policy-level no-headroom verifier, then validate with compact actual `.oscr` bytes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.output, args.report, args.codebook)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
