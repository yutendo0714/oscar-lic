#!/usr/bin/env python3
"""Search simple diagnostic veto rules over exception probe evidence rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import yaml


DEPLOYABLE_NUMERIC_FEATURES = [
    "width",
    "height",
    "aspect_ratio",
    "current_luma_mean",
    "current_luma_std",
    "current_dark_fraction",
    "current_bright_fraction",
    "exception_luma_mean",
    "exception_luma_std",
    "exception_dark_fraction",
    "exception_bright_fraction",
    "exception_minus_current_dark_fraction",
    "exception_minus_current_bright_fraction",
    "current_exception_mad",
    "current_exception_mse",
    "current_exception_max_abs",
    "rate_delta_bytes",
    "rate_delta_bpp",
    "current_actual_bpp",
    "exception_actual_bpp",
    "current_psnr_db",
    "exception_psnr_db",
    "psnr_delta_db",
]

AUDIT_NUMERIC_FEATURES = ["reference_length"]


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


def latest_unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int, int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["source_index"]),
            int(row["candidate_index"]),
            str(row["reference"]),
            str(row["action_taken"]),
        )
        by_key[key] = row
    return list(by_key.values())


def baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(row["strict_label"] for row in rows)),
        "actions": dict(Counter(row["action_taken"] for row in rows)),
        "sources": dict(Counter(row["source"] for row in rows)),
        "rate_delta_bytes": int(sum(int(row["rate_delta_bytes"]) for row in rows)),
        "tesseract_unicode_delta": int(sum(int(row["tesseract_unicode_strict_v1_delta"]) for row in rows)),
        "tesseract_latin_delta": int(sum(int(row["tesseract_latin_alnum_ci_v1_delta"]) for row in rows)),
        "parseq_unicode_delta": int(sum(int(row["parseq_unicode_strict_v1_delta"]) for row in rows)),
        "parseq_latin_delta": int(sum(int(row["parseq_latin_alnum_ci_v1_delta"]) for row in rows)),
    }


def threshold_values(rows: list[dict[str, Any]], feature: str) -> list[float]:
    values = sorted({float(row[feature]) for row in rows if row.get(feature) is not None})
    if not values:
        return []
    picks = set()
    for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
        picks.add(values[round(frac * (len(values) - 1))])
    for row in rows:
        if row["strict_label"] == "harmful_any_profile":
            picks.add(float(row[feature]))
    return sorted(picks)


def make_predicates(rows: list[dict[str, Any]], include_audit_only: bool) -> list[dict[str, Any]]:
    predicates: list[dict[str, Any]] = []

    def add(name: str, deployable: bool, func: Callable[[dict[str, Any]], bool]) -> None:
        mask = [func(row) for row in rows]
        if any(mask) and not all(mask):
            predicates.append({"name": name, "deployable": deployable, "mask": mask})

    for action in sorted({row["action_taken"] for row in rows}):
        add(f"action == {action}", True, lambda row, action=action: row["action_taken"] == action)
    for feature in DEPLOYABLE_NUMERIC_FEATURES:
        for threshold in threshold_values(rows, feature):
            add(f"{feature} <= {threshold:.6g}", True, lambda row, feature=feature, threshold=threshold: float(row[feature]) <= threshold)
            add(f"{feature} >= {threshold:.6g}", True, lambda row, feature=feature, threshold=threshold: float(row[feature]) >= threshold)
    if include_audit_only:
        for source in sorted({row["source"] for row in rows}):
            add(f"source == {source}", False, lambda row, source=source: row["source"] == source)
        for tag in sorted({tag for row in rows for tag in row.get("probe_tags", [])}):
            add(f"tag contains {tag}", False, lambda row, tag=tag: tag in row.get("probe_tags", []))
        for feature in AUDIT_NUMERIC_FEATURES:
            for threshold in threshold_values(rows, feature):
                add(f"{feature} <= {threshold:.6g}", False, lambda row, feature=feature, threshold=threshold: float(row[feature]) <= threshold)
                add(f"{feature} >= {threshold:.6g}", False, lambda row, feature=feature, threshold=threshold: float(row[feature]) >= threshold)
    return predicates


def apply_rule(rows: list[dict[str, Any]], mask: list[bool], name: str, deployable: bool) -> dict[str, Any]:
    vetoed = [row for row, flag in zip(rows, mask) if flag]
    kept = [row for row, flag in zip(rows, mask) if not flag]
    harmful_total = sum(1 for row in rows if row["strict_label"] == "harmful_any_profile")
    harmful_vetoed = sum(1 for row in vetoed if row["strict_label"] == "harmful_any_profile")
    beneficial_vetoed = sum(1 for row in vetoed if row["strict_label"] == "beneficial_no_profile_harm")
    data = baseline(kept)
    data.update(
        {
            "rule": name,
            "deployable_features_only": bool(deployable),
            "vetoed_rows": len(vetoed),
            "vetoed_labels": dict(Counter(row["strict_label"] for row in vetoed)),
            "vetoed_actions": dict(Counter(row["action_taken"] for row in vetoed)),
            "vetoed_sources": dict(Counter(row["source"] for row in vetoed)),
            "harmful_total": int(harmful_total),
            "harmful_vetoed": int(harmful_vetoed),
            "harmful_retained": int(harmful_total - harmful_vetoed),
            "beneficial_vetoed": int(beneficial_vetoed),
            "vetoed_examples": [
                {
                    "seed": row["seed"],
                    "source_index": row["source_index"],
                    "source": row["source"],
                    "reference": row["reference"],
                    "action": row["action_taken"],
                    "label": row["strict_label"],
                    "tess_unicode_delta": row["tesseract_unicode_strict_v1_delta"],
                    "tess_latin_delta": row["tesseract_latin_alnum_ci_v1_delta"],
                }
                for row in vetoed[:12]
            ],
        }
    )
    return data


def rank_key(rule: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    return (
        int(rule["harmful_retained"]),
        int(rule["tesseract_unicode_delta"] + rule["tesseract_latin_delta"]),
        int(rule["beneficial_vetoed"]),
        int(rule["vetoed_rows"]),
        -int(rule["rate_delta_bytes"]),
        len(rule["rule"]),
    )


def search_rules(rows: list[dict[str, Any]], include_audit_only: bool) -> list[dict[str, Any]]:
    predicates = make_predicates(rows, include_audit_only=include_audit_only)
    candidates: list[dict[str, Any]] = []
    for pred in predicates:
        candidates.append(apply_rule(rows, pred["mask"], pred["name"], bool(pred["deployable"])))
    for left, right in itertools.combinations(predicates, 2):
        if not include_audit_only and (not left["deployable"] or not right["deployable"]):
            continue
        mask = [bool(a or b) for a, b in zip(left["mask"], right["mask"])]
        deployable = bool(left["deployable"] and right["deployable"])
        candidates.append(apply_rule(rows, mask, f"({left['name']}) OR ({right['name']})", deployable))
    return sorted(candidates, key=rank_key)


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception Probe Veto Rule Search",
        "",
        "Rules are diagnostic vetoes over actual exception/drop labels: a veto means keep the current stream for that row.",
        "Lower OCR deltas are better. This is not a deployable selector claim.",
        "",
        "## Baseline",
        "",
        f"Rows: `{result['baseline']['rows']}`; labels: `{json.dumps(result['baseline']['labels'], sort_keys=True)}`; "
        f"Tesseract delta `{result['baseline']['tesseract_unicode_delta']}` unicode / "
        f"`{result['baseline']['tesseract_latin_delta']}` latin; rate delta bytes `{result['baseline']['rate_delta_bytes']}`.",
        "",
        "## Best Deployable-Feature Rules",
        "",
        "| rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["top_deployable_rules"]:
        lines.append(
            f"| {row['rule']} | {row['harmful_retained']} | {row['harmful_vetoed']} | "
            f"{row['beneficial_vetoed']} | {row['tesseract_unicode_delta']} | "
            f"{row['tesseract_latin_delta']} | {row['rate_delta_bytes']} | {row['vetoed_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Deployable Harm-Budget Frontier",
            "",
            "| max retained harmful | rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for budget, row in result["best_deployable_by_harm_budget"].items():
        lines.append(
            f"| {budget} | {row['rule']} | {row['harmful_retained']} | {row['harmful_vetoed']} | "
            f"{row['beneficial_vetoed']} | {row['tesseract_unicode_delta']} | "
            f"{row['tesseract_latin_delta']} | {row['rate_delta_bytes']} | {row['vetoed_rows']} |"
        )
    lines.extend(["", "## Best Audit-Only Rules", "", "| rule | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in result["top_audit_rules"]:
        lines.append(
            f"| {row['rule']} | {row['harmful_retained']} | {row['harmful_vetoed']} | "
            f"{row['beneficial_vetoed']} | {row['tesseract_unicode_delta']} | "
            f"{row['tesseract_latin_delta']} | {row['rate_delta_bytes']} | {row['vetoed_rows']} |"
        )
    lines.extend(["", "## Conclusion", "", result["conclusion"], "", "## Next Action", "", result["next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def log_wandb(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any] | None:
    if not args.wandb_project:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_id,
            tags=["oscar-lic", "exception-centers", "veto-rule-search", "diagnostic"],
            config={"experiment_id": args.experiment_id},
        )
        best = result["top_deployable_rules"][0]
        wandb.log(
            {
                "baseline/tesseract_unicode_delta": result["baseline"]["tesseract_unicode_delta"],
                "baseline/tesseract_latin_delta": result["baseline"]["tesseract_latin_delta"],
                "best_deployable/harmful_retained": best["harmful_retained"],
                "best_deployable/beneficial_vetoed": best["beneficial_vetoed"],
                "best_deployable/tesseract_unicode_delta": best["tesseract_unicode_delta"],
                "best_deployable/tesseract_latin_delta": best["tesseract_latin_delta"],
            }
        )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "feature_table", "path": str(args.feature_table), "sha256": sha256_file(args.feature_table)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic",
        "command": " ".join(["scripts/search_exception_probe_veto_rules.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_probe_veto_rule_search_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = latest_unique(read_jsonl(args.feature_table))
    deployable_rules = search_rules(rows, include_audit_only=False)
    audit_rules = search_rules(rows, include_audit_only=True)
    best_by_budget: dict[str, dict[str, Any]] = {}
    for budget in range(0, 5):
        feasible = [rule for rule in deployable_rules if int(rule["harmful_retained"]) <= budget]
        if feasible:
            best_by_budget[str(budget)] = sorted(
                feasible,
                key=lambda rule: (
                    int(rule["tesseract_unicode_delta"]) + int(rule["tesseract_latin_delta"]),
                    int(rule["beneficial_vetoed"]),
                    -int(rule["rate_delta_bytes"]),
                    int(rule["vetoed_rows"]),
                ),
            )[0]
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-hard-negative-veto-diagnostic",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "baseline": baseline(rows),
        "top_deployable_rules": deployable_rules[: args.top_k],
        "best_deployable_by_harm_budget": best_by_budget,
        "top_audit_rules": audit_rules[: args.top_k],
        "conclusion": (
            "Simple deployable feature vetoes expose whether the four harmful exception/drop rows are separable "
            "without sacrificing too much of the replacement gain. Audit-only source/text rules are upper controls, "
            "not deployable policies."
        ),
        "next_action": (
            "Use the best deployable-feature rule families as features or weak veto targets in a diagnostic verifier, "
            "then evaluate with strict held-out thresholding and actual compact OCR."
        ),
    }
    result["wandb"] = log_wandb(args, result)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.report)
    print(json.dumps({"result": str(args.result), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
