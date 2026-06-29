#!/usr/bin/env python3
"""Train a leakage-controlled top-k-rankcap candidate chooser.

This diagnostic keeps the first-stage candidate set simple and clean: non-nearest
candidates with topk_rank <= K from the frozen Eval300 top8 table. It then asks
whether a second-stage chooser can pick the oracle code inside that set.

No held-out OCR strings, OCR predictions, OCR deltas or labels are used as
features. OCR deltas enter only for evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_top8_two_stage_bottleneck as bottleneck  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def allowed_feature_name(name: str) -> bool:
    if name in {"topk_rank", "assignment_relative_error", "candidate_slot"}:
        return True
    if name.startswith("codebook_") or name.startswith("img_"):
        blocked = [
            "oracle",
            "label",
            "tesseract",
            "parseq",
            "prediction",
            "reference",
            "source_image",
            "image_path",
            "nearest_distance",
            "distance",
            "exact",
        ]
        return not any(token in name for token in blocked)
    return False


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows[: min(len(rows), 64)]:
        for key, value in row.items():
            if allowed_feature_name(key) and isinstance(value, (int, float)):
                names.add(key)
    return sorted(names)


def feature_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            value = finite_float(row.get(name))
            matrix[i, j] = 0.0 if value is None else float(value)
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix


def oracle_in_cap(rows: list[dict[str, Any]], cap: int) -> bool:
    nearest = bottleneck.nearest_row(rows)
    oracle = bottleneck.oracle_row(rows)
    return int(oracle["code_index"]) != int(nearest["code_index"]) and int(oracle.get("topk_rank", 999)) <= cap


def candidate_rows_for_group(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not int(row.get("is_nearest", 0)) and int(row.get("topk_rank", 999)) <= cap
    ]


def build_rows(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    cap: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int, int, int], int]]:
    rows: list[dict[str, Any]] = []
    row_index: dict[tuple[int, int, int, int, int], int] = {}
    for group_rows in groups.values():
        for row in candidate_rows_for_group(group_rows, cap):
            row_index[bottleneck.candidate_key(row)] = len(rows)
            rows.append(row)
    return rows, row_index


def trainable_indices(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    row_index: dict[tuple[int, int, int, int, int], int],
    cap: int,
) -> np.ndarray:
    indices: list[int] = []
    for rows in groups.values():
        if not oracle_in_cap(rows, cap):
            continue
        for row in candidate_rows_for_group(rows, cap):
            idx = row_index.get(bottleneck.candidate_key(row))
            if idx is not None:
                indices.append(idx)
    return np.asarray(indices, dtype=np.int64)


def labels_for(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row.get("label_assignment_oracle_choice", 0)) for row in rows], dtype=np.int32)


def classifier_grid(random_state: int) -> dict[str, Any]:
    return {
        "logistic_l2_c02": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000, random_state=random_state),
        ),
        "hist_gradient_l2": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=0.2,
            random_state=random_state,
        ),
        "random_forest_d4": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=4,
        ),
    }


def score_classifier(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1].astype(np.float64)
        return prob.reshape(-1).astype(np.float64)
    if hasattr(model, "decision_function"):
        return model.decision_function(x).astype(np.float64)
    return model.predict(x).astype(np.float64)


def selected_by_scores(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    row_index: dict[tuple[int, int, int, int, int], int],
    scores: np.ndarray,
    cap: int,
    *,
    covered_only: bool,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        nearest_code = int(nearest["code_index"])
        oracle_changes = int(oracle["code_index"]) != nearest_code
        selected[key] = nearest_code
        if not oracle_changes:
            continue
        if covered_only and int(oracle.get("topk_rank", 999)) > cap:
            continue
        candidates = []
        for row in candidate_rows_for_group(rows, cap):
            idx = row_index.get(bottleneck.candidate_key(row))
            if idx is None:
                continue
            candidates.append((float(scores[idx]), -int(row.get("topk_rank", 999)), -int(row["code_index"]), row))
        if candidates:
            selected[key] = int(max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]["code_index"])
    return selected


def rank_counts_for_oracle(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    row_index: dict[tuple[int, int, int, int, int], int],
    scores: np.ndarray,
    cap: int,
) -> dict[str, Any]:
    counts = Counter()
    audits = []
    for key, rows in sorted(groups.items()):
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        counts["oracle_change_groups"] += 1
        if int(oracle.get("topk_rank", 999)) > cap:
            counts["oracle_not_in_cap"] += 1
            if len(audits) < 25:
                audits.append(
                    {
                        "group_key": list(key),
                        "source": rows[0].get("source"),
                        "reference": rows[0].get("reference"),
                        "oracle_topk_rank": int(oracle.get("topk_rank", -1)),
                        "status": "oracle_not_in_cap",
                    }
                )
            continue
        ranked = []
        for row in candidate_rows_for_group(rows, cap):
            idx = row_index.get(bottleneck.candidate_key(row))
            if idx is None:
                continue
            ranked.append((float(scores[idx]), -int(row.get("topk_rank", 999)), -int(row["code_index"]), row))
        ranked.sort(reverse=True)
        rank = None
        for pos, (_, _, _, row) in enumerate(ranked, start=1):
            if int(row["code_index"]) == int(oracle["code_index"]):
                rank = pos
                break
        if rank is None:
            counts["oracle_missing_score"] += 1
            continue
        for limit in [1, 2, 4]:
            counts[f"rank_le{limit}"] += int(rank <= limit)
        if rank > 1 and len(audits) < 25:
            top = ranked[0][3]
            audits.append(
                {
                    "group_key": list(key),
                    "source": rows[0].get("source"),
                    "reference": rows[0].get("reference"),
                    "oracle_code": int(oracle["code_index"]),
                    "oracle_topk_rank": int(oracle.get("topk_rank", -1)),
                    "score_rank": int(rank),
                    "top_code": int(top["code_index"]),
                    "top_topk_rank": int(top.get("topk_rank", -1)),
                    "status": "oracle_not_top1",
                }
            )
    return {**{key: int(value) for key, value in counts.items()}, "audits": audits}


def compact_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics.get('exact_changed_groups', 0)}/{metrics.get('oracle_change_groups', 0)} exact, "
        f"false {metrics.get('false_change', 0)}, wrong {metrics.get('wrong_change', 0)}, "
        f"miss {metrics.get('missed_oracle', 0)}, T {metrics.get('tesseract_delta_vs_nearest', 0)}, "
        f"P {metrics.get('parseq_delta_vs_nearest', 0)}"
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Leakage-controlled second-stage chooser over non-nearest `topk_rank <= K` candidates.",
        "This is a candidate-chooser diagnostic only; no `.oscr` policy is promoted.",
        "",
        "## Validation",
        "",
        "| model | rank1 | rank<=2 | rank<=4 | covered-only policy | all-change policy |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, row in result["models"].items():
        ranks = row["val_rank_counts"]
        lines.append(
            f"| {name} | {ranks.get('rank_le1', 0)} | {ranks.get('rank_le2', 0)} | "
            f"{ranks.get('rank_le4', 0)} | {compact_metrics(row['val_policy_covered_only'])} | "
            f"{compact_metrics(row['val_policy_all_oracle_changes'])} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-rankcap-second-stage",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
        "  table:",
        f"    path: {result['inputs']['table']['path']}",
        f"    sha256: {result['inputs']['table']['sha256']}",
        "outputs:",
        "  result_json:",
        f"    path: {output}",
        f"    sha256: {sha256_file(output)}",
        "  report:",
        f"    path: {report}",
        f"    sha256: {sha256_file(report)}",
        "scripts:",
        "  chooser:",
        "    path: scripts/train_top8_rankcap_candidate_chooser.py",
        f"    sha256: {sha256_file(Path(__file__))}",
        "  bottleneck_audit:",
        "    path: scripts/analyze_top8_two_stage_bottleneck.py",
        f"    sha256: {sha256_file(SCRIPT_DIR / 'analyze_top8_two_stage_bottleneck.py')}",
        "conclusion: diagnostic rankcap candidate chooser; no selector promoted",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups, _ = bottleneck.load_table(args.table)
    groups_by_partition = bottleneck.split_groups(groups)
    all_rows, row_index = build_rows(groups, args.cap)
    names = feature_names(all_rows)
    x = feature_matrix(all_rows, names)
    y = labels_for(all_rows)
    train_groups = groups_by_partition["train"]
    val_groups = groups_by_partition["val"]
    train_idx = trainable_indices(train_groups, row_index, args.cap)
    train_summary = {
        "candidate_rows": int(len(all_rows)),
        "feature_count": int(len(names)),
        "train_rows_used": int(len(train_idx)),
        "train_positive_rows_used": int(y[train_idx].sum()),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "cap": int(args.cap),
    }
    models: dict[str, Any] = {}
    for name, model in classifier_grid(args.seed).items():
        model.fit(x[train_idx], y[train_idx])
        scores = score_classifier(model, x)
        models[name] = {
            "train_rank_counts": rank_counts_for_oracle(train_groups, row_index, scores, args.cap),
            "val_rank_counts": rank_counts_for_oracle(val_groups, row_index, scores, args.cap),
            "train_policy_covered_only": bottleneck.evaluate_selection(
                train_groups,
                selected_by_scores(train_groups, row_index, scores, args.cap, covered_only=True),
            ),
            "val_policy_covered_only": bottleneck.evaluate_selection(
                val_groups,
                selected_by_scores(val_groups, row_index, scores, args.cap, covered_only=True),
            ),
            "val_policy_all_oracle_changes": bottleneck.evaluate_selection(
                val_groups,
                selected_by_scores(val_groups, row_index, scores, args.cap, covered_only=False),
            ),
        }
    best_name, best_row = max(
        models.items(),
        key=lambda item: (
            item[1]["val_rank_counts"].get("rank_le1", 0),
            -item[1]["val_policy_covered_only"].get("wrong_change", 0),
            -item[1]["val_policy_covered_only"].get("false_change", 0),
            -item[1]["val_policy_covered_only"].get("missed_oracle", 0),
        ),
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_rankcap_candidate_chooser_not_promoted",
        "code_commit": bottleneck.git_commit(),
        "inputs": {"table": {"path": str(args.table), "sha256": sha256_file(args.table)}},
        "config": {
            "cap": args.cap,
            "seed": args.seed,
            "feature_rule": "topk_rank, assignment_relative_error, candidate_slot, codebook_* and img_* numeric columns; OCR/label/oracle/reference/source ids excluded",
            "feature_names": names,
        },
        "data_summary": train_summary,
        "models": models,
        "best_model": {"name": best_name, "val_rank_counts": best_row["val_rank_counts"]},
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "bottleneck_script": sha256_file(SCRIPT_DIR / "analyze_top8_two_stage_bottleneck.py"),
        },
        "interpretation": (
            "This uses the clean, deterministic topk-rankcap shortlist rather than the provenance-sensitive "
            "source_mod5_r4 scores. If it cannot improve rank1 inside the rank<=4 candidate set, the next step "
            "should change candidate-local representation or add teacher-auxiliary representation learning, not "
            "threshold this chooser into an actual `.oscr` policy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "data_summary": train_summary,
                "best_model": result["best_model"],
                "val": {
                    name: {
                        "rank": row["val_rank_counts"],
                        "covered": {
                            key: value
                            for key, value in row["val_policy_covered_only"].items()
                            if key
                            in [
                                "exact_changed_groups",
                                "false_change",
                                "wrong_change",
                                "missed_oracle",
                                "tesseract_delta_vs_nearest",
                                "parseq_delta_vs_nearest",
                            ]
                        },
                    }
                    for name, row in models.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
