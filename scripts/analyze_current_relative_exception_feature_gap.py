#!/usr/bin/env python3
"""Audit N145 current-relative labels against N143 exception action features."""

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


FEATURES = [
    "extra_relative_error",
    "extra_distance_margin",
    "extra_second_margin",
    "base_best_distance",
    "extra_best_distance",
    "residual_norm",
    "base_center_utility",
    "extra_center_utility",
]


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


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    data = sorted(values)

    def pick(frac: float) -> float:
        index = round(frac * (len(data) - 1))
        return float(data[index])

    return {
        "min": float(data[0]),
        "q25": pick(0.25),
        "median": pick(0.5),
        "q75": pick(0.75),
        "max": float(data[-1]),
    }


def summarize_deltas(rows: list[dict[str, Any]], model: str, profile: str) -> dict[str, Any]:
    deltas = [int(row["label"]["ocr"][model][profile]["delta_errors"]) for row in rows]
    return {
        "rows": len(rows),
        "delta_errors": int(sum(deltas)),
        "improved_rows": int(sum(1 for value in deltas if value < 0)),
        "worsened_rows": int(sum(1 for value in deltas if value > 0)),
        "equal_rows": int(sum(1 for value in deltas if value == 0)),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": len(rows),
        "strict_labels": dict(Counter(row["label"]["strict_label"] for row in rows)),
        "actions": dict(Counter(row["label"]["action_taken"] for row in rows)),
        "sources": dict(Counter(row["action"].get("source", "unknown") for row in rows)),
        "tesseract_unicode": summarize_deltas(rows, "tesseract", "unicode_strict_v1"),
        "tesseract_latin": summarize_deltas(rows, "tesseract", "latin_alnum_ci_v1"),
        "parseq_unicode": summarize_deltas(rows, "parseq", "unicode_strict_v1"),
        "parseq_latin": summarize_deltas(rows, "parseq", "latin_alnum_ci_v1"),
        "features": {},
    }
    for feature in FEATURES:
        result["features"][feature] = quantiles([float(row["action"][feature]) for row in rows if feature in row["action"]])
    return result


def join_rows(action_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_by_key = {(int(row["seed"]), int(row["source_index"])): row for row in label_rows if bool(row["current_selected"])}
    joined = []
    for action in action_rows:
        if action["partition"] != "val" or int(action["seed"]) not in {1, 2}:
            continue
        key = (int(action["seed"]), int(action["source_index"]))
        label = label_by_key.get(key)
        if label is None:
            continue
        joined.append({"key": key, "action": action, "label": label})
    return joined


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    action_rows = read_jsonl(args.action_table)
    label_rows = read_jsonl(args.current_relative_labels)
    joined = join_rows(action_rows, label_rows)
    replacements = [row for row in joined if row["label"]["action_taken"] == "replace_with_exception"]
    drops = [row for row in joined if row["label"]["action_taken"] == "drop_current"]
    proxy_true = [row for row in replacements if bool(row["action"].get("oracle_change_proxy"))]
    proxy_false = [row for row in replacements if not bool(row["action"].get("oracle_change_proxy"))]
    beneficial = [row for row in joined if row["label"]["strict_label"] == "beneficial_no_profile_harm"]
    harmful_or_mixed = [
        row
        for row in joined
        if row["label"]["strict_label"] in {"harmful_any_profile", "mixed_harm_and_benefit"}
    ]
    keep_current_only_counterfactual = {
        "description": "If all current-only/drop rows were kept as current instead of dropped, these exception-minus-current OCR deltas would be avoided.",
        "rows": len(drops),
        "dropped_tesseract_unicode_delta": summarize_deltas(drops, "tesseract", "unicode_strict_v1")["delta_errors"],
        "dropped_tesseract_latin_delta": summarize_deltas(drops, "tesseract", "latin_alnum_ci_v1")["delta_errors"],
        "dropped_parseq_unicode_delta": summarize_deltas(drops, "parseq", "unicode_strict_v1")["delta_errors"],
        "dropped_parseq_latin_delta": summarize_deltas(drops, "parseq", "latin_alnum_ci_v1")["delta_errors"],
        "avoided_tesseract_unicode_delta": -summarize_deltas(drops, "tesseract", "unicode_strict_v1")["delta_errors"],
        "avoided_tesseract_latin_delta": -summarize_deltas(drops, "tesseract", "latin_alnum_ci_v1")["delta_errors"],
        "avoided_parseq_unicode_delta": -summarize_deltas(drops, "parseq", "unicode_strict_v1")["delta_errors"],
        "avoided_parseq_latin_delta": -summarize_deltas(drops, "parseq", "latin_alnum_ci_v1")["delta_errors"],
        "extra_bytes_to_keep_current": int(
            sum(
                int(row["label"]["current_stream"]["actual_total_bytes"])
                - int(row["label"]["exception_stream"]["actual_total_bytes"])
                for row in drops
            )
        ),
    }
    by_case = defaultdict(list)
    by_source = defaultdict(list)
    for row in joined:
        by_case[row["action"]["policy_case"]].append(row)
        by_source[row["action"].get("source", "unknown")].append(row)
    return {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-current-relative-feature-gap",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "joined_rows": len(joined),
        "missing_current_selected_labels": int(
            sum(
                1
                for row in action_rows
                if row["partition"] == "val"
                and int(row["seed"]) in {1, 2}
                and (int(row["seed"]), int(row["source_index"]))
                not in {(int(label["seed"]), int(label["source_index"])) for label in label_rows if bool(label["current_selected"])}
            )
        ),
        "summary": {
            "all_current_selected": summarize_group(joined),
            "replace_with_exception": summarize_group(replacements),
            "drop_current": summarize_group(drops),
            "replacement_proxy_true": summarize_group(proxy_true),
            "replacement_proxy_false": summarize_group(proxy_false),
            "strict_beneficial": summarize_group(beneficial),
            "strict_harmful_or_mixed": summarize_group(harmful_or_mixed),
            "by_policy_case": {name: summarize_group(rows) for name, rows in sorted(by_case.items())},
            "by_source": {name: summarize_group(rows) for name, rows in sorted(by_source.items())},
            "keep_current_only_counterfactual": keep_current_only_counterfactual,
        },
        "conclusion": (
            "N143 proxy/action features explain exception eligibility, but the executed N145 labels show current-relative "
            "safety is dominated by whether current-only rows should be kept. Proxy-true replacement is not enough, and "
            "drop-current safety needs explicit supervision."
        ),
        "next_action": (
            "Construct counterfactual current-preserving labels for current-selected units, especially hard negative "
            "drop/replacement examples, before training a replace/drop verifier."
        ),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Current-Relative Exception Feature Gap",
        "",
        f"Joined current-selected validation rows: `{result['joined_rows']}`.",
        "",
        "## Main Splits",
        "",
        "| subset | rows | labels | tess unicode delta | tess latin delta | parseq unicode delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name in [
        "all_current_selected",
        "replace_with_exception",
        "drop_current",
        "replacement_proxy_true",
        "replacement_proxy_false",
        "strict_beneficial",
        "strict_harmful_or_mixed",
    ]:
        data = result["summary"][name]
        lines.append(
            f"| {name} | {data['rows']} | `{json.dumps(data['strict_labels'], sort_keys=True)}` | "
            f"{data['tesseract_unicode']['delta_errors']} | {data['tesseract_latin']['delta_errors']} | "
            f"{data['parseq_unicode']['delta_errors']} |"
        )
    counter = result["summary"]["keep_current_only_counterfactual"]
    lines.extend(
        [
            "",
            "## Keep-Current-Only Counterfactual",
            "",
            f"Dropped current rows: `{counter['rows']}`.",
            f"Dropping them causes exception-minus-current Tesseract deltas `+{counter['dropped_tesseract_unicode_delta']}` "
            f"unicode and `+{counter['dropped_tesseract_latin_delta']}` latin chars; keeping current would remove those "
            f"deltas at `"
            f"{counter['extra_bytes_to_keep_current']}` transmitted bytes over this seed1/2 subset.",
            "",
            "## Source Breakdown",
            "",
            "| source | rows | labels | tess unicode delta | tess latin delta |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for source, data in result["summary"]["by_source"].items():
        lines.append(
            f"| {source} | {data['rows']} | `{json.dumps(data['strict_labels'], sort_keys=True)}` | "
            f"{data['tesseract_unicode']['delta_errors']} | {data['tesseract_latin']['delta_errors']} |"
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
            tags=["oscar-lic", "exception-centers", "current-relative-labels", "feature-gap"],
            config={"experiment_id": args.experiment_id},
        )
        summary = result["summary"]
        wandb.log(
            {
                "rows/joined": result["joined_rows"],
                "drop_current/tesseract_unicode_delta": summary["drop_current"]["tesseract_unicode"]["delta_errors"],
                "drop_current/tesseract_latin_delta": summary["drop_current"]["tesseract_latin"]["delta_errors"],
                "replace/tesseract_unicode_delta": summary["replace_with_exception"]["tesseract_unicode"]["delta_errors"],
                "replace/tesseract_latin_delta": summary["replace_with_exception"]["tesseract_latin"]["delta_errors"],
                "counterfactual/keep_current_extra_bytes": summary["keep_current_only_counterfactual"][
                    "extra_bytes_to_keep_current"
                ],
            }
        )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover - wandb availability is environment-dependent.
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "action_table", "path": str(args.action_table), "sha256": sha256_file(args.action_table)},
        {
            "name": "current_relative_labels",
            "path": str(args.current_relative_labels),
            "sha256": sha256_file(args.current_relative_labels),
        },
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic",
        "command": " ".join(["scripts/analyze_current_relative_exception_feature_gap.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-table", type=Path, required=True)
    parser.add_argument("--current-relative-labels", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_current_relative_feature_gap_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_result(args)
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
