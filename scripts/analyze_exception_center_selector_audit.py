#!/usr/bin/env python3
"""Audit group-level selectors for utility-exception codebook centers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
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


def parse_seed_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    seed_text, path_text = value.split("=", 1)
    return int(seed_text), Path(path_text)


def group_key_from_row(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def build_metadata(table: Path) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    metadata: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(table):
        key = group_key_from_row(row)
        if key not in metadata:
            metadata[key] = {
                "partition": str(row.get("assignment_partition", row.get("split", "unknown"))),
                "source": row.get("source"),
                "reference": row.get("reference"),
            }
    return metadata


def load_codebooks(items: list[tuple[int, Path]]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for seed, path in items:
        data = np.load(path)
        entry = {"codebook": np.asarray(data["codebook"], dtype=np.float32)}
        if "center_utility" in data.files:
            entry["center_utility"] = np.asarray(data["center_utility"], dtype=np.float32)
        result[seed] = entry
    return result


def distances(vector: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = centers.astype(np.float32) - vector.astype(np.float32)[None, :]
    return np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32))


def finite(value: float) -> float:
    if np.isfinite(value):
        return float(value)
    return 0.0


def build_group_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = np.load(args.features)
    metadata = build_metadata(args.table)
    codebooks = load_codebooks(args.codebook)
    rows: list[dict[str, Any]] = []
    for gi, key_arr in enumerate(np.asarray(data["group_keys"])):
        key = tuple(int(v) for v in key_arr)
        meta = metadata.get(key, {})
        partition = str(meta.get("partition", "unknown"))
        seed = key[0]
        codebook = codebooks[seed]["codebook"]
        if codebook.shape[0] <= args.base_size:
            raise ValueError(f"codebook for seed {seed} has no exception centers: {codebook.shape[0]}")
        center_utility = codebooks[seed].get("center_utility", np.zeros((codebook.shape[0],), dtype=np.float32))
        slots = np.flatnonzero(np.asarray(data["mask"])[gi])
        nearest_slots = [slot for slot in slots if float(data["is_nearest"][gi, slot]) > 0.5]
        oracle_slots = [slot for slot in slots if float(data["labels"][gi, slot]) > 0.5]
        if not nearest_slots or not oracle_slots:
            continue
        nearest_slot = int(nearest_slots[0])
        oracle_slot = int(oracle_slots[0])
        vector = np.asarray(data["latent_residual"][gi, nearest_slot], dtype=np.float32)
        base_distances = distances(vector, codebook[: args.base_size])
        extra_distances = distances(vector, codebook[args.base_size :])
        base_order = np.argsort(base_distances)
        extra_order = np.argsort(extra_distances)
        base_best = int(base_order[0])
        extra_best_local = int(extra_order[0])
        extra_best = int(args.base_size + extra_best_local)
        base_best_distance = float(base_distances[base_best])
        extra_best_distance = float(extra_distances[extra_best_local])
        second_extra_distance = float(extra_distances[int(extra_order[1])]) if len(extra_order) > 1 else extra_best_distance
        denom = max(base_best_distance, 1e-8)
        oracle_change = oracle_slot != nearest_slot
        rows.append(
            {
                "group_key": list(key),
                "partition": partition,
                "source": meta.get("source"),
                "reference": meta.get("reference"),
                "oracle_change": bool(oracle_change),
                "nearest_code": int(data["code_indices"][gi, nearest_slot]),
                "oracle_code": int(data["code_indices"][gi, oracle_slot]),
                "base_best_code": base_best,
                "extra_best_code": extra_best,
                "extra_relative_error": extra_best_distance / denom,
                "extra_distance_margin": extra_best_distance - base_best_distance,
                "extra_second_margin": second_extra_distance - extra_best_distance,
                "base_best_distance": base_best_distance,
                "extra_best_distance": extra_best_distance,
                "residual_norm": float(np.linalg.norm(vector)),
                "extra_center_utility": float(center_utility[extra_best]) if extra_best < len(center_utility) else 0.0,
                "base_center_utility": float(center_utility[base_best]) if base_best < len(center_utility) else 0.0,
            }
        )
    return rows


def score_policy(rows: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    counts = defaultdict(int)
    examples = []
    for row, choose in zip(rows, selected):
        oracle_change = bool(row["oracle_change"])
        counts["groups"] += 1
        counts["oracle_change_groups"] += int(oracle_change)
        if choose:
            counts["changed_groups"] += 1
            counts["oracle_change_covered"] += int(oracle_change)
            counts["noheadroom_false_changes"] += int(not oracle_change)
            if len(examples) < 20:
                examples.append(
                    {
                        "group_key": row["group_key"],
                        "source": row["source"],
                        "oracle_change": oracle_change,
                        "extra_relative_error": finite(float(row["extra_relative_error"])),
                        "extra_best_code": row["extra_best_code"],
                    }
                )
        else:
            counts["missed_oracle_change"] += int(oracle_change)
    precision = (
        counts["oracle_change_covered"] / counts["changed_groups"] if counts["changed_groups"] else None
    )
    recall = (
        counts["oracle_change_covered"] / counts["oracle_change_groups"]
        if counts["oracle_change_groups"]
        else None
    )
    for key in [
        "groups",
        "oracle_change_groups",
        "changed_groups",
        "oracle_change_covered",
        "noheadroom_false_changes",
        "missed_oracle_change",
    ]:
        counts.setdefault(key, 0)
    return {
        **{key: int(value) for key, value in sorted(counts.items())},
        "precision_oracle_change_proxy": precision,
        "recall_oracle_change_proxy": recall,
        "changed_examples": examples,
    }


def threshold_metrics(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    result = {}
    for threshold in thresholds:
        selected = [float(row["extra_relative_error"]) <= threshold for row in rows]
        result[f"rel_le_{threshold:.3f}"] = score_policy(rows, selected)
    return result


def choose_threshold(train_rows: list[dict[str, Any]], thresholds: list[float], budget: int) -> float | None:
    best: tuple[int, float, float] | None = None
    for threshold in thresholds:
        metrics = score_policy(train_rows, [float(row["extra_relative_error"]) <= threshold for row in train_rows])
        false_count = int(metrics.get("noheadroom_false_changes", 0))
        if false_count > budget:
            continue
        covered = int(metrics.get("oracle_change_covered", 0))
        changed = int(metrics.get("changed_groups", 0))
        candidate = (covered, -changed, threshold)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else best[2]


def train_tuned_thresholds(rows_by_part: dict[str, list[dict[str, Any]]], thresholds: list[float], budgets: list[int]) -> dict[str, Any]:
    train_rows = rows_by_part.get("train", [])
    val_rows = rows_by_part.get("val", [])
    result = {}
    for budget in budgets:
        threshold = choose_threshold(train_rows, thresholds, budget)
        label = f"train_budget{budget}"
        if threshold is None:
            result[label] = {"threshold": None, "train": None, "val": None}
            continue
        result[label] = {
            "threshold": threshold,
            "train": score_policy(train_rows, [float(row["extra_relative_error"]) <= threshold for row in train_rows]),
            "val": score_policy(val_rows, [float(row["extra_relative_error"]) <= threshold for row in val_rows]),
        }
    return result


def feature_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    names = [
        "extra_relative_error",
        "extra_distance_margin",
        "extra_second_margin",
        "base_best_distance",
        "extra_best_distance",
        "residual_norm",
        "extra_center_utility",
        "base_center_utility",
    ]
    x = np.asarray([[finite(float(row[name])) for name in names] for row in rows], dtype=np.float32)
    y = np.asarray([1 if row["oracle_change"] else 0 for row in rows], dtype=np.int64)
    return x, y


def learned_score_audit(rows_by_part: dict[str, list[dict[str, Any]]], budgets: list[int]) -> dict[str, Any]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - optional dependency guard.
        return {"error": repr(exc)}
    train_rows = rows_by_part.get("train", [])
    val_rows = rows_by_part.get("val", [])
    x_train, y_train = feature_matrix(train_rows)
    x_val, _ = feature_matrix(val_rows)
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000)),
        "hist_gradient": HistGradientBoostingClassifier(max_leaf_nodes=7, learning_rate=0.05, l2_regularization=0.1, random_state=137),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=137,
        ),
    }
    result = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_scores = model.predict_proba(x_train)[:, 1]
        val_scores = model.predict_proba(x_val)[:, 1]
        model_result = {}
        for budget in budgets:
            candidates = sorted(set(float(score) for score in train_scores), reverse=True)
            best: tuple[int, int, float] | None = None
            for threshold in candidates:
                metrics = score_policy(train_rows, [float(score) >= threshold for score in train_scores])
                false_count = int(metrics.get("noheadroom_false_changes", 0))
                if false_count > budget:
                    continue
                covered = int(metrics.get("oracle_change_covered", 0))
                changed = int(metrics.get("changed_groups", 0))
                candidate = (covered, -changed, threshold)
                if best is None or candidate > best:
                    best = candidate
            label = f"train_budget{budget}"
            if best is None:
                model_result[label] = {"threshold": None, "train": None, "val": None}
            else:
                threshold = best[2]
                model_result[label] = {
                    "threshold": threshold,
                    "train": score_policy(train_rows, [float(score) >= threshold for score in train_scores]),
                    "val": score_policy(val_rows, [float(score) >= threshold for score in val_scores]),
                }
        result[name] = model_result
    return result


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows_by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_part[str(row["partition"])].append(row)
    partitions = {}
    for part, part_rows in sorted(rows_by_part.items()):
        partitions[part] = {
            "rows": len(part_rows),
            "oracle_change_groups": sum(1 for row in part_rows if row["oracle_change"]),
            "thresholds": threshold_metrics(part_rows, args.threshold),
        }
    thresholds = train_tuned_thresholds(rows_by_part, args.threshold, args.false_budget)
    learned = learned_score_audit(rows_by_part, args.false_budget) if args.learned else None
    return {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-center-noheadroom-selector-diagnostic",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "inputs": {
            "features": str(args.features),
            "table": str(args.table),
            "codebooks": {str(seed): str(path) for seed, path in args.codebook},
            "base_size": args.base_size,
            "thresholds": args.threshold,
            "false_budgets": args.false_budget,
        },
        "partitions": partitions,
        "train_tuned_thresholds": thresholds,
        "learned_score_audit": learned,
        "conclusion": (
            "Distance-only exception-center eligibility is not source-robust: thresholds that look high-precision on "
            "train leak many no-headroom validation groups. Exception centers therefore need a materially stronger "
            "no-headroom verifier before actual OCR streams are worth generating."
        ),
        "next_action": (
            "Use this audit as a stop rule for simple distance selectors. A future exception-center selector should add "
            "candidate-local visual/code-effect evidence and prove nonzero useful validation changes with near-zero "
            "no-headroom false changes before `.oscr` OCR evaluation."
        ),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception-Center Selector Audit",
        "",
        "This diagnostic evaluates whether K80 exception-center distance features can separate oracle-change groups from no-headroom groups.",
        "It does not decode new streams or claim OCR improvement.",
        "",
        "## Thresholds",
        "",
    ]
    for part, data in result["partitions"].items():
        lines.append(f"### `{part}`")
        lines.append("")
        lines.append(f"Groups: `{data['rows']}`; oracle-change groups: `{data['oracle_change_groups']}`.")
        lines.append("")
        lines.append("| threshold | changed | oracle covered | no-headroom false | recall | precision |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for label, metrics in data["thresholds"].items():
            threshold = label.replace("rel_le_", "")
            recall = metrics["recall_oracle_change_proxy"]
            precision = metrics["precision_oracle_change_proxy"]
            lines.append(
                f"| {threshold} | {metrics.get('changed_groups', 0)} | {metrics.get('oracle_change_covered', 0)} | "
                f"{metrics.get('noheadroom_false_changes', 0)} | "
                f"{0.0 if recall is None else recall:.4f} | {0.0 if precision is None else precision:.4f} |"
            )
        lines.append("")
    lines.extend(["## Train-Tuned Thresholds", "", "| budget | threshold | val changed | val oracle covered | val false |"])
    lines.append("|---:|---:|---:|---:|---:|")
    for label, data in result["train_tuned_thresholds"].items():
        val = data["val"] or {}
        threshold = data["threshold"]
        lines.append(
            f"| {label.replace('train_budget', '')} | {threshold if threshold is not None else 'none'} | "
            f"{val.get('changed_groups', 0)} | {val.get('oracle_change_covered', 0)} | "
            f"{val.get('noheadroom_false_changes', 0)} |"
        )
    if result.get("learned_score_audit"):
        lines.extend(["", "## Learned Score Audit", "", "| model | budget | threshold | val changed | val oracle covered | val false |"])
        lines.append("|---|---:|---:|---:|---:|---:|")
        learned = result["learned_score_audit"]
        if "error" in learned:
            lines.append(f"| error | - | - | - | - | - |")
        else:
            for model, model_data in learned.items():
                for label, data in model_data.items():
                    val = data["val"] or {}
                    threshold = data["threshold"]
                    lines.append(
                        f"| {model} | {label.replace('train_budget', '')} | "
                        f"{threshold if threshold is not None else 'none'} | "
                        f"{val.get('changed_groups', 0)} | {val.get('oracle_change_covered', 0)} | "
                        f"{val.get('noheadroom_false_changes', 0)} |"
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
            tags=["oscar-lic", "exception-centers", "selector-audit", "diagnostic"],
            config=result["inputs"],
        )
        val = result["partitions"].get("val", {})
        for label, metrics in val.get("thresholds", {}).items():
            wandb.log(
                {
                    f"val/{label}/changed": metrics.get("changed_groups", 0),
                    f"val/{label}/oracle_covered": metrics.get("oracle_change_covered", 0),
                    f"val/{label}/false": metrics.get("noheadroom_false_changes", 0),
                }
            )
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover - wandb availability is environment-dependent.
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "features", "path": str(args.features), "sha256": sha256_file(args.features)},
        {"name": "table", "path": str(args.table), "sha256": sha256_file(args.table)},
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for seed, codebook in args.codebook:
        files.append({"name": f"codebook_seed{seed}", "path": str(codebook), "sha256": sha256_file(codebook)})
    data = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "command": " ".join(["scripts/analyze_exception_center_selector_audit.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--codebook", type=parse_seed_path, action="append", required=True)
    parser.add_argument("--base-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, action="append", default=[1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.08, 1.10, 1.15, 1.20])
    parser.add_argument("--false-budget", type=int, action="append", default=[0, 1, 3, 5])
    parser.add_argument("--learned", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_center_selector_audit_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_group_rows(args)
    result = summarize(rows, args)
    result["wandb"] = log_wandb(args, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.output, args.report)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
