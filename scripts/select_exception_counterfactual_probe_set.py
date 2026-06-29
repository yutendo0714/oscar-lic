#!/usr/bin/env python3
"""Select probe rows for current-relative exception counterfactual labeling."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml


SELECTION_RE = re.compile(r"seed(?P<seed>\d+)_(?P<partition>train225|val75)_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


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


def infer_selection_key(path: Path) -> tuple[int, str]:
    match = SELECTION_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer selection seed/partition from {path}")
    partition = "train" if match.group("partition") == "train225" else "val"
    return int(match.group("seed")), partition


def load_current_selection(paths: list[Path]) -> dict[tuple[int, str, int, int], dict[str, Any]]:
    rows: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    for path in paths:
        seed, partition = infer_selection_key(path)
        for row in read_jsonl(path):
            if "selected_by_policy" in row and not bool(row["selected_by_policy"]):
                continue
            key = (seed, partition, int(row["source_index"]), int(row["candidate_index"]))
            rows[key] = row
    return rows


def add_probe(
    probes: dict[tuple[str, int, str, int, int], dict[str, Any]],
    action: dict[str, Any],
    *,
    tag: str,
    training_use: bool,
    selection: dict[str, Any] | None,
    label: dict[str, Any] | None = None,
) -> None:
    key = (str(action["partition"]), int(action["seed"]), str(action["policy_case"]), int(action["source_index"]), int(action["candidate_index"]))
    if key not in probes:
        probes[key] = {
            "probe_tags": [],
            "training_use": training_use,
            "partition": action["partition"],
            "seed": int(action["seed"]),
            "source_index": int(action["source_index"]),
            "candidate_index": int(action["candidate_index"]),
            "source": action.get("source"),
            "reference": action.get("reference"),
            "policy_case": action["policy_case"],
            "oracle_change_proxy": bool(action["oracle_change_proxy"]),
            "exception_eligible": bool(action["exception_eligible"]),
            "extra_relative_error": float(action["extra_relative_error"]),
            "extra_distance_margin": float(action["extra_distance_margin"]),
            "extra_second_margin": float(action["extra_second_margin"]),
            "residual_norm": float(action["residual_norm"]),
            "nearest_code": int(action["nearest_code"]),
            "base_best_code": int(action["base_best_code"]),
            "extra_best_code": int(action["extra_best_code"]),
            "oracle_code": int(action["oracle_code"]),
            "source_image": selection.get("source_image") if selection else None,
            "selection_row": selection,
            "label_row": label,
            "notes": "Selected for counterfactual stream/OCR labeling. Validation reference rows are audit-only.",
        }
    probes[key]["probe_tags"].append(tag)
    probes[key]["training_use"] = bool(probes[key]["training_use"] and training_use)


def build_probes(args: argparse.Namespace) -> list[dict[str, Any]]:
    actions = read_jsonl(args.action_table)
    selection = load_current_selection(args.current_selection)
    probes: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}

    for action in actions:
        if action["partition"] != "train":
            continue
        sel = selection.get((int(action["seed"]), "train", int(action["source_index"]), int(action["candidate_index"])))
        rel = float(action["extra_relative_error"])
        source = str(action.get("source"))
        if action["policy_case"] == "current_only" and rel <= args.close_threshold:
            add_probe(probes, action, tag="train_close_current_only", training_use=True, selection=sel)
        if action["policy_case"] == "current_only" and source == "iam_words" and rel <= args.iam_threshold:
            add_probe(probes, action, tag="train_iam_current_only_near", training_use=True, selection=sel)
        if action["policy_case"] == "current_and_exception" and not bool(action["oracle_change_proxy"]):
            add_probe(probes, action, tag="train_replacement_proxy_false", training_use=True, selection=sel)

    iam_proxy_true = [
        action
        for action in actions
        if action["partition"] == "train"
        and action["policy_case"] == "current_and_exception"
        and str(action.get("source")) == "iam_words"
        and bool(action["oracle_change_proxy"])
    ]
    iam_proxy_true = sorted(iam_proxy_true, key=lambda row: float(row["extra_relative_error"]))[: args.iam_positive_controls]
    for action in iam_proxy_true:
        sel = selection.get((int(action["seed"]), "train", int(action["source_index"]), int(action["candidate_index"])))
        add_probe(probes, action, tag="train_iam_proxy_true_control", training_use=True, selection=sel)

    if args.current_relative_labels:
        labels = read_jsonl(args.current_relative_labels)
        val_actions = {
            (int(action["seed"]), int(action["source_index"])): action
            for action in actions
            if action["partition"] == "val" and int(action["seed"]) in {1, 2}
        }
        for label in labels:
            if label["strict_label"] not in {"harmful_any_profile", "mixed_harm_and_benefit"}:
                continue
            if not bool(label["current_selected"]):
                continue
            action = val_actions.get((int(label["seed"]), int(label["source_index"])))
            if action is None:
                continue
            sel = selection.get((int(action["seed"]), "val", int(action["source_index"]), int(action["candidate_index"])))
            add_probe(probes, action, tag="val_harmful_mixed_reference", training_use=False, selection=sel, label=label)

    rows = list(probes.values())
    for index, row in enumerate(sorted(rows, key=lambda item: (not item["training_use"], item["partition"], item["seed"], item["source_index"], item["candidate_index"]))):
        row["probe_index"] = index
        row["probe_tags"] = sorted(set(row["probe_tags"]))
    return sorted(rows, key=lambda item: item["probe_index"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "training_rows": sum(1 for row in rows if row["training_use"]),
        "audit_only_rows": sum(1 for row in rows if not row["training_use"]),
        "tags": dict(Counter(tag for row in rows for tag in row["probe_tags"])),
        "sources": dict(Counter(str(row["source"]) for row in rows)),
        "policy_cases": dict(Counter(str(row["policy_case"]) for row in rows)),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception Counterfactual Probe Set",
        "",
        "This is a selection manifest for future actual compact stream/OCR counterfactual labeling.",
        "Rows with `training_use=false` are validation references and must not be used as training labels.",
        "",
        "## Summary",
        "",
        f"Rows: `{result['summary']['rows']}`; training rows: `{result['summary']['training_rows']}`; "
        f"audit-only rows: `{result['summary']['audit_only_rows']}`.",
        "",
        f"Tags: `{json.dumps(result['summary']['tags'], sort_keys=True)}`.",
        "",
        f"Sources: `{json.dumps(result['summary']['sources'], sort_keys=True)}`.",
        "",
        "## Next Action",
        "",
        result["next_action"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def log_wandb(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any] | None:
    if not args.wandb_project:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_id,
            tags=["oscar-lic", "exception-centers", "counterfactual-probes"],
            config={"experiment_id": args.experiment_id},
        )
        wandb.log(
            {
                "rows": result["summary"]["rows"],
                "training_rows": result["summary"]["training_rows"],
                "audit_only_rows": result["summary"]["audit_only_rows"],
                "tags/train_close_current_only": result["summary"]["tags"].get("train_close_current_only", 0),
                "tags/val_harmful_mixed_reference": result["summary"]["tags"].get("val_harmful_mixed_reference", 0),
            }
        )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover - wandb availability is environment-dependent.
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, table_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "action_table", "path": str(args.action_table), "sha256": sha256_file(args.action_table)},
        {"name": "probe_table", "path": str(table_path), "sha256": sha256_file(table_path)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for selection in args.current_selection:
        files.append({"name": f"selection_{selection.name}", "path": str(selection), "sha256": sha256_file(selection)})
    if args.current_relative_labels:
        files.append(
            {"name": "current_relative_labels", "path": str(args.current_relative_labels), "sha256": sha256_file(args.current_relative_labels)}
        )
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_probe_selection",
        "command": " ".join(["scripts/select_exception_counterfactual_probe_set.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-table", type=Path, required=True)
    parser.add_argument("--current-selection", type=Path, action="append", required=True)
    parser.add_argument("--current-relative-labels", type=Path, default=None)
    parser.add_argument("--close-threshold", type=float, default=1.5)
    parser.add_argument("--iam-threshold", type=float, default=2.0)
    parser.add_argument("--iam-positive-controls", type=int, default=10)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_counterfactual_probe_set_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_probes(args)
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-counterfactual-probe-selection",
        "status": "completed_probe_selection",
        "code_commit": git_commit(),
        "summary": summarize(rows),
        "next_action": (
            "Generate compact current/drop/replacement streams for the training-use probe rows and run OCR to obtain "
            "current-relative hard negative labels. Keep validation reference rows audit-only."
        ),
    }
    result["wandb"] = log_wandb(args, result)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_table, rows)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.output_table, args.report)
    print(json.dumps({"table": str(args.output_table), "result": str(args.result), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
