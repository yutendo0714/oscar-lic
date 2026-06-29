#!/usr/bin/env python3
"""Train top-8 selectors from CRNN/ABINet logit-summary features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def row_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
        int(row["code_index"]),
    )


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    key = row_key(row)
    return key[:4]


def load_feature_cache(path: Path) -> tuple[dict[tuple[int, int, int, int, int], np.ndarray], list[str], dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    keys = data["row_keys"].astype(np.int64)
    features = data["features"].astype(np.float32)
    names = [str(name) for name in data["feature_names"].tolist()]
    lookup = {tuple(int(value) for value in key): features[index] for index, key in enumerate(keys)}
    meta = {"path": str(path), "sha256": sha256_file(path), "rows": int(features.shape[0]), "feature_dim": int(features.shape[1])}
    return lookup, names, meta


def classify(selected_code: int, nearest_code: int, oracle_code: int) -> str:
    if selected_code == oracle_code:
        return "exact"
    if selected_code == nearest_code and oracle_code != nearest_code:
        return "missed_oracle"
    if selected_code != nearest_code and oracle_code == nearest_code:
        return "false_change"
    return "wrong_change"


def build_dataset(
    table_rows: list[dict[str, Any]],
    crnn_lookup: dict[tuple[int, int, int, int, int], np.ndarray],
    abinet_lookup: dict[tuple[int, int, int, int, int], np.ndarray],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int, int], list[int]], np.ndarray, list[str]]:
    rows = []
    grouped: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    feature_rows = []
    feature_names = [
        "topk_rank",
        "log1p_assignment_relative_error",
        "is_nearest",
        "code_equals_nearest",
        "code_index_norm",
    ]
    for prefix in ("crnn", "abinet"):
        feature_names.extend([f"{prefix}_raw_{idx:02d}" for idx in range(62)])
        feature_names.extend([f"{prefix}_delta_{idx:02d}" for idx in range(62)])
        feature_names.extend([f"{prefix}_abs_delta_{idx:02d}" for idx in range(62)])

    nearest_feature_by_group: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for row in table_rows:
        if int(row.get("code_equals_nearest", int(row["code_index"]) == int(row["nearest_code"]))):
            key = row_key(row)
            gkey = key[:4]
            nearest_feature_by_group[gkey] = (crnn_lookup[key], abinet_lookup[key])

    for row in table_rows:
        key = row_key(row)
        gkey = key[:4]
        if key not in crnn_lookup or key not in abinet_lookup or gkey not in nearest_feature_by_group:
            raise RuntimeError(f"missing logit feature for row {key}")
        crnn = crnn_lookup[key]
        abinet = abinet_lookup[key]
        crnn_nearest, abinet_nearest = nearest_feature_by_group[gkey]
        assignment_error = min(float(row.get("assignment_relative_error", 0.0)), 1.0e6)
        base = np.asarray(
            [
                float(row.get("topk_rank", 0)),
                float(np.log1p(max(assignment_error, 0.0))),
                float(row.get("is_nearest", int(row["code_index"]) == int(row["nearest_code"]))),
                float(row.get("code_equals_nearest", int(row["code_index"]) == int(row["nearest_code"]))),
                float(row["code_index"]) / max(float(row.get("codebook_size", 64)) - 1.0, 1.0),
            ],
            dtype=np.float32,
        )
        features = np.concatenate(
            [
                base,
                crnn,
                crnn - crnn_nearest,
                np.abs(crnn - crnn_nearest),
                abinet,
                abinet - abinet_nearest,
                np.abs(abinet - abinet_nearest),
            ]
        ).astype(np.float32)
        meta = {
            "group_key": gkey,
            "row_key": key,
            "partition": str(row.get("assignment_partition", row.get("split", ""))),
            "source": str(row.get("source", "unknown")),
            "code_index": int(row["code_index"]),
            "nearest_code": int(row["nearest_code"]),
            "oracle_code": int(row["assignment_oracle_code_index"]),
            "is_nearest": bool(int(row.get("is_nearest", int(row["code_index"]) == int(row["nearest_code"])))),
            "topk_rank": int(row.get("topk_rank", 0)),
            "tesseract_delta": int(row.get("tesseract_delta_vs_nearest", 0)),
            "parseq_delta": int(row.get("parseq_delta_vs_nearest", 0)),
            "label_exact": int(row.get("label_assignment_oracle_choice", 0)),
            "label_safe": int(row.get("label_tesseract_parseq_safe_improves", 0)),
        }
        grouped[gkey].append(len(rows))
        rows.append(meta)
        feature_rows.append(features)
    return rows, grouped, np.vstack(feature_rows).astype(np.float32), feature_names


def train_models(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    if len(set(int(v) for v in y.tolist())) < 2:
        return {}
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ).fit(x, y),
        "hist_gradient": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.035,
            l2_regularization=0.03,
            random_state=seed,
        ).fit(x, y),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ).fit(x, y),
    }


def predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.zeros(len(x), dtype=np.float32)
    return proba[:, 1].astype(np.float32)


def best_nonnearest(group_rows: list[int], rows: list[dict[str, Any]], scores: np.ndarray) -> int | None:
    candidates = [idx for idx in group_rows if not rows[idx]["is_nearest"]]
    if not candidates:
        return None
    return max(candidates, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])))


def evaluate(
    rows: list[dict[str, Any]],
    grouped: dict[tuple[int, int, int, int], list[int]],
    scores: np.ndarray,
    partition: str,
    threshold: float | None,
    oracle_change_only: bool,
) -> dict[str, Any]:
    counts = Counter()
    t_sum = 0
    p_sum = 0
    t_worse = 0
    audits = []
    rank_values = []
    for gkey, indices in grouped.items():
        first = rows[indices[0]]
        if first["partition"] != partition:
            continue
        nearest = next(idx for idx in indices if rows[idx]["is_nearest"])
        oracle_code = int(first["oracle_code"])
        nearest_code = int(first["nearest_code"])
        best = best_nonnearest(indices, rows, scores)
        selected = nearest
        if best is not None:
            if oracle_change_only:
                selected = best if oracle_code != nearest_code else nearest
            elif threshold is not None and float(scores[best]) >= threshold:
                selected = best
        selected_code = int(rows[selected]["code_index"])
        status = classify(selected_code, nearest_code, oracle_code)
        counts[status] += 1
        changed = selected_code != nearest_code
        counts["changed_groups"] += int(changed)
        counts["oracle_change_groups"] += int(oracle_code != nearest_code)
        counts["exact_changed_groups"] += int(status == "exact" and changed)
        if changed:
            t_delta = int(rows[selected]["tesseract_delta"])
            p_delta = int(rows[selected]["parseq_delta"])
        else:
            t_delta = 0
            p_delta = 0
        t_sum += t_delta
        p_sum += p_delta
        t_worse += int(t_delta > 0)
        if oracle_code != nearest_code and best is not None:
            ranked = sorted(
                [idx for idx in indices if not rows[idx]["is_nearest"]],
                key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])),
                reverse=True,
            )
            oracle_idx = next((idx for idx in indices if int(rows[idx]["code_index"]) == oracle_code), None)
            if oracle_idx in ranked:
                rank_values.append(1 + ranked.index(oracle_idx))
        audits.append(
            {
                "group_key": list(gkey),
                "selected_code": selected_code,
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "top_code": None if best is None else int(rows[best]["code_index"]),
                "top_score": None if best is None else float(scores[best]),
                "status": status,
                "selected_tesseract_delta_vs_nearest": int(t_delta),
                "selected_parseq_delta_vs_nearest": int(p_delta),
            }
        )
    return {
        "groups": int(sum(1 for indices in grouped.values() if rows[indices[0]]["partition"] == partition)),
        "oracle_change_groups": int(counts["oracle_change_groups"]),
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "exact_changed_groups": int(counts["exact_changed_groups"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "tesseract_delta_vs_nearest": int(t_sum),
        "parseq_delta_vs_nearest": int(p_sum),
        "tesseract_worse_groups": int(t_worse),
        "oracle_rank_le1": int(sum(rank <= 1 for rank in rank_values)),
        "oracle_rank_le2": int(sum(rank <= 2 for rank in rank_values)),
        "oracle_rank_le4": int(sum(rank <= 4 for rank in rank_values)),
        "audits": audits[:20],
    }


def tune_threshold(rows, grouped, scores, max_false_wrong: int) -> tuple[float, dict[str, Any]]:
    train_best_scores = []
    for indices in grouped.values():
        if rows[indices[0]]["partition"] != "train":
            continue
        best = best_nonnearest(indices, rows, scores)
        if best is not None:
            train_best_scores.append(float(scores[best]))
    candidates = sorted(set(train_best_scores), reverse=True)
    if not candidates:
        return 1.0, evaluate(rows, grouped, scores, "train", 1.0, False)
    candidates = [max(candidates) + 1.0] + candidates + [min(candidates) - 1.0]
    best_threshold = candidates[0]
    best_metrics = evaluate(rows, grouped, scores, "train", best_threshold, False)
    best_key = (10**9, 10**9, 10**9, 0, 0)
    for threshold in candidates:
        metrics = evaluate(rows, grouped, scores, "train", threshold, False)
        false_wrong = metrics["false_change"] + metrics["wrong_change"]
        if false_wrong > max_false_wrong or metrics["parseq_delta_vs_nearest"] > 0:
            continue
        key = (
            metrics["tesseract_worse_groups"],
            metrics["tesseract_delta_vs_nearest"],
            false_wrong,
            -metrics["exact_changed_groups"],
            metrics["changed_groups"],
        )
        if key < best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, {k: v for k, v in best_metrics.items() if k != "audits"}


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "CRNN/ABINet logit-summary feature selector over Eval300 top-8 assignment candidates.",
        "This is an OCR-aware encoder-side diagnostic and does not export counted `.oscr` streams.",
        "",
        "## Validation Policies",
        "",
        "| target | model | budget | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | rank1 | rank<=4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        for policy in run["policies"]:
            m = policy["val_metrics"]
            lines.append(
                f"| {run['target']} | {run['model_name']} | {policy['max_false_wrong']} | "
                f"{m['changed_groups']} | {m['exact_changed_groups']} | {m['false_change']} | "
                f"{m['wrong_change']} | {m['missed_oracle']} | {m['tesseract_delta_vs_nearest']} | "
                f"{m['parseq_delta_vs_nearest']} | {m['oracle_rank_le1']} | {m['oracle_rank_le4']} |"
            )
    lines.extend(["", "## Best Policy", "", "```json", json.dumps(result["best_policy"], indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--crnn-features", type=Path, required=True)
    parser.add_argument("--abinet-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--max-false-wrong", type=int, action="append")
    args = parser.parse_args()
    if args.max_false_wrong is None:
        args.max_false_wrong = [0, 1, 2]

    table_rows = read_jsonl(args.table)
    crnn_lookup, crnn_names, crnn_meta = load_feature_cache(args.crnn_features)
    abinet_lookup, abinet_names, abinet_meta = load_feature_cache(args.abinet_features)
    rows, grouped, x, feature_names = build_dataset(table_rows, crnn_lookup, abinet_lookup)
    partitions = np.asarray([row["partition"] for row in rows])
    nonnearest = np.asarray([not row["is_nearest"] for row in rows])
    train_mask = (partitions == "train") & nonnearest
    val_mask = (partitions == "val") & nonnearest

    runs = []
    for target_name, target_field in (("exact_oracle", "label_exact"), ("safe_improve", "label_safe")):
        y = np.asarray([row[target_field] for row in rows], dtype=np.int64)
        models = train_models(x[train_mask], y[train_mask], args.model_seed)
        for model_name, model in models.items():
            scores = predict_scores(model, x)
            oracle_val = evaluate(rows, grouped, scores, "val", None, True)
            policies = []
            for budget in args.max_false_wrong:
                threshold, train_metrics = tune_threshold(rows, grouped, scores, int(budget))
                val_metrics = evaluate(rows, grouped, scores, "val", threshold, False)
                policies.append(
                    {
                        "max_false_wrong": int(budget),
                        "threshold": float(threshold),
                        "train_metrics": train_metrics,
                        "val_metrics": {k: v for k, v in val_metrics.items() if k != "audits"},
                        "val_audits": val_metrics["audits"],
                    }
                )
            runs.append(
                {
                    "target": target_name,
                    "model_name": model_name,
                    "train_positive_rows": int(y[train_mask].sum()),
                    "val_positive_rows": int(y[val_mask].sum()),
                    "oracle_change_only_val_metrics": {k: v for k, v in oracle_val.items() if k != "audits"},
                    "policies": policies,
                }
            )

    candidates = [(run, policy) for run in runs for policy in run["policies"]]
    best_run, best_policy = sorted(
        candidates,
        key=lambda item: (
            item[1]["val_metrics"]["tesseract_delta_vs_nearest"],
            item[1]["val_metrics"]["false_change"] + item[1]["val_metrics"]["wrong_change"],
            item[1]["val_metrics"]["tesseract_worse_groups"],
            -item[1]["val_metrics"]["exact_changed_groups"],
        ),
    )[0]
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_strhub_logit_feature_selector_not_promoted",
        "inputs": {
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "crnn_features": crnn_meta,
            "abinet_features": abinet_meta,
        },
        "config": {
            "model_seed": args.model_seed,
            "max_false_wrong": args.max_false_wrong,
            "feature_dim": int(x.shape[1]),
            "feature_blocks": ["base_rank_error", "crnn_raw_delta_absdelta", "abinet_raw_delta_absdelta"],
        },
        "data_summary": {
            "rows": int(len(rows)),
            "groups": int(len(grouped)),
            "train_rows": int(train_mask.sum()),
            "val_rows": int(val_mask.sum()),
        },
        "runs": runs,
        "best_policy": {
            "target": best_run["target"],
            "model_name": best_run["model_name"],
            **best_policy,
        },
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "crnn_feature_names": crnn_names,
            "abinet_feature_names": abinet_names,
        },
        "interpretation": (
            "This diagnostic tests whether richer CRNN/ABINet logit-distribution summaries "
            "carry deployable text-aware evidence beyond teacher strings/losses. It is not "
            "promoted unless a train-tuned table policy exceeds the current actual-bitstream "
            "-8 Tesseract floor with low false/wrong changes."
        ),
    }
    write_json(args.output, result)
    write_report(args.report, result)


if __name__ == "__main__":
    main()
