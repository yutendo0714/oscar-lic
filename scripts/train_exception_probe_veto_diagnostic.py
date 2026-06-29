#!/usr/bin/env python3
"""Train a tiny leave-one-out diagnostic harmful-row veto scorer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import yaml


NUMERIC_FEATURES = [
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
CATEGORICAL_FEATURES = ["action_taken"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def row_matrix(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    records = []
    y = []
    for row in rows:
        record = {feature: row.get(feature) for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
        records.append(record)
        y.append(1 if row["strict_label"] == "harmful_any_profile" else 0)
    return records, np.asarray(y, dtype=np.int64)


def as_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.asarray([[float(record.get(feature, np.nan)) for feature in NUMERIC_FEATURES] for record in records], dtype=np.float32)
    categorical = np.asarray([[str(record.get(feature, "")) for feature in CATEGORICAL_FEATURES] for record in records], dtype=object)
    return numeric, categorical


def make_model(kind: str, seed: int):
    numeric_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical_pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    pre = ColumnTransformer(
        [
            ("numeric", numeric_pipe, list(range(len(NUMERIC_FEATURES)))),
            ("categorical", categorical_pipe, list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))),
        ]
    )
    if kind == "logistic":
        clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000, random_state=seed)
    elif kind == "forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
        )
    else:
        raise ValueError(kind)
    return make_pipeline(pre, clf)


def combined_feature_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    numeric, categorical = as_arrays(records)
    return np.concatenate([numeric.astype(object), categorical], axis=1)


def loo_scores(rows: list[dict[str, Any]], kind: str, seed: int) -> list[float]:
    records, y = row_matrix(rows)
    x = combined_feature_matrix(records)
    scores = []
    for index in range(len(rows)):
        train_idx = [i for i in range(len(rows)) if i != index]
        if len(set(int(y[i]) for i in train_idx)) < 2:
            scores.append(float(y[train_idx].mean()))
            continue
        model = make_model(kind, seed)
        model.fit(x[train_idx], y[train_idx])
        score = float(model.predict_proba(x[[index]])[0, 1])
        scores.append(score)
    return scores


def baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(row["strict_label"] for row in rows)),
        "rate_delta_bytes": int(sum(int(row["rate_delta_bytes"]) for row in rows)),
        "tesseract_unicode_delta": int(sum(int(row["tesseract_unicode_strict_v1_delta"]) for row in rows)),
        "tesseract_latin_delta": int(sum(int(row["tesseract_latin_alnum_ci_v1_delta"]) for row in rows)),
        "parseq_unicode_delta": int(sum(int(row["parseq_unicode_strict_v1_delta"]) for row in rows)),
        "parseq_latin_delta": int(sum(int(row["parseq_latin_alnum_ci_v1_delta"]) for row in rows)),
    }


def apply_veto(rows: list[dict[str, Any]], scores: list[float], threshold: float, model_name: str) -> dict[str, Any]:
    kept = [row for row, score in zip(rows, scores) if score < threshold]
    vetoed = [row for row, score in zip(rows, scores) if score >= threshold]
    data = baseline(kept)
    harmful_total = sum(1 for row in rows if row["strict_label"] == "harmful_any_profile")
    data.update(
        {
            "model": model_name,
            "threshold": float(threshold),
            "vetoed_rows": len(vetoed),
            "vetoed_labels": dict(Counter(row["strict_label"] for row in vetoed)),
            "harmful_total": int(harmful_total),
            "harmful_vetoed": int(sum(1 for row in vetoed if row["strict_label"] == "harmful_any_profile")),
            "harmful_retained": int(sum(1 for row in kept if row["strict_label"] == "harmful_any_profile")),
            "beneficial_vetoed": int(sum(1 for row in vetoed if row["strict_label"] == "beneficial_no_profile_harm")),
        }
    )
    return data


def frontiers(rows: list[dict[str, Any]], scores: list[float], model_name: str) -> dict[str, Any]:
    thresholds = sorted(set(scores + [min(scores) - 1e-9, max(scores) + 1e-9]))
    policies = [apply_veto(rows, scores, threshold, model_name) for threshold in thresholds]
    best_by_budget = {}
    for budget in range(0, 5):
        feasible = [policy for policy in policies if int(policy["harmful_retained"]) <= budget]
        if feasible:
            best_by_budget[str(budget)] = sorted(
                feasible,
                key=lambda policy: (
                    int(policy["tesseract_unicode_delta"]) + int(policy["tesseract_latin_delta"]),
                    int(policy["beneficial_vetoed"]),
                    -int(policy["rate_delta_bytes"]),
                    int(policy["vetoed_rows"]),
                ),
            )[0]
    return {
        "best_by_harm_budget": best_by_budget,
        "top_policies": sorted(
            policies,
            key=lambda policy: (
                int(policy["harmful_retained"]),
                int(policy["tesseract_unicode_delta"]) + int(policy["tesseract_latin_delta"]),
                int(policy["beneficial_vetoed"]),
                int(policy["vetoed_rows"]),
            ),
        )[:12],
    }


def attach_scores(rows: list[dict[str, Any]], all_scores: dict[str, list[float]]) -> list[dict[str, Any]]:
    out = []
    for index, row in enumerate(rows):
        item = {
            "seed": row["seed"],
            "source_index": row["source_index"],
            "candidate_index": row["candidate_index"],
            "source": row["source"],
            "reference": row["reference"],
            "action_taken": row["action_taken"],
            "strict_label": row["strict_label"],
            "tesseract_unicode_delta": row["tesseract_unicode_strict_v1_delta"],
            "tesseract_latin_delta": row["tesseract_latin_alnum_ci_v1_delta"],
            "rate_delta_bytes": row["rate_delta_bytes"],
        }
        for name, scores in all_scores.items():
            item[f"{name}_loo_harm_score"] = float(scores[index])
        out.append(item)
    return out


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Exception Probe Veto Diagnostic Verifier",
        "",
        "This is a leave-one-out diagnostic over 53 actual-labeled rows, not a deployable policy.",
        "",
        "## Baseline",
        "",
        f"Rows `{result['baseline']['rows']}`, labels `{json.dumps(result['baseline']['labels'], sort_keys=True)}`, "
        f"Tesseract `{result['baseline']['tesseract_unicode_delta']}` unicode / `{result['baseline']['tesseract_latin_delta']}` latin, "
        f"rate delta bytes `{result['baseline']['rate_delta_bytes']}`.",
    ]
    for model_name, model_result in result["models"].items():
        lines.extend(
            [
                "",
                f"## {model_name}",
                "",
                "| max retained harmful | threshold | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for budget, policy in model_result["best_by_harm_budget"].items():
            lines.append(
                f"| {budget} | {policy['threshold']:.6g} | {policy['harmful_retained']} | "
                f"{policy['harmful_vetoed']} | {policy['beneficial_vetoed']} | "
                f"{policy['tesseract_unicode_delta']} | {policy['tesseract_latin_delta']} | "
                f"{policy['rate_delta_bytes']} | {policy['vetoed_rows']} |"
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
            tags=["oscar-lic", "exception-centers", "diagnostic-verifier", "loo"],
            config={"experiment_id": args.experiment_id},
        )
        for model_name, model_result in result["models"].items():
            zero = model_result["best_by_harm_budget"].get("0")
            one = model_result["best_by_harm_budget"].get("1")
            if zero:
                wandb.log(
                    {
                        f"{model_name}/zero_harm_tess_unicode": zero["tesseract_unicode_delta"],
                        f"{model_name}/zero_harm_tess_latin": zero["tesseract_latin_delta"],
                        f"{model_name}/zero_harm_beneficial_vetoed": zero["beneficial_vetoed"],
                    }
                )
            if one:
                wandb.log({f"{model_name}/one_harm_tess_unicode": one["tesseract_unicode_delta"], f"{model_name}/one_harm_tess_latin": one["tesseract_latin_delta"]})
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, scores_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "feature_table", "path": str(args.feature_table), "sha256": sha256_file(args.feature_table)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "scores", "path": str(scores_path), "sha256": sha256_file(scores_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": args.experiment_id,
                "status": "completed_diagnostic",
                "command": " ".join(["scripts/train_exception_probe_veto_diagnostic.py", *sys.argv[1:]]),
                "code_commit": git_commit(),
                "inputs_and_outputs": files,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_probe_veto_diagnostic_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = latest_unique(read_jsonl(args.feature_table))
    all_scores = {
        "logistic": loo_scores(rows, "logistic", args.seed),
        "forest": loo_scores(rows, "forest", args.seed),
    }
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-exception-hard-negative-veto-diagnostic",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "baseline": baseline(rows),
        "models": {name: frontiers(rows, scores, name) for name, scores in all_scores.items()},
        "conclusion": (
            "Leave-one-out harmful-row scoring is a tiny diagnostic only. It tests whether deployable image/rate/action "
            "features carry enough signal to approximate the N154 veto frontier without source/reference metadata."
        ),
        "next_action": (
            "If the LOO frontier is competitive with N154, fold these scores into a larger hard-negative collection; "
            "otherwise collect more labels before training any verifier."
        ),
    }
    result["wandb"] = log_wandb(args, result)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.scores, attach_scores(rows, all_scores))
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.scores, args.report)
    print(json.dumps({"scores": str(args.scores), "result": str(args.result), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
