#!/usr/bin/env python3
"""Cross-fit no-op guard for the teacher-auxiliary top-8 ranker.

This diagnostic keeps teacher outputs out of inference inputs. CRNN/ABINet
loss utility is used only through the imported teacher-auxiliary ranker
objective. The new part is a group-level acceptor trained on source-OOF
candidate scores so that no-op calibration is not tuned on in-sample ranker
scores.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_top8_latent_teacher_aux_ranker as aux_ranker  # noqa: E402
import train_top8_trainval_tabular_listwise_ranker as table_ranker  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def build_group_sources(table_rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], str]:
    sources: dict[tuple[int, int, int, int], str] = {}
    for row in table_rows:
        sources[group_key(row)] = str(row.get("source", "unknown"))
    return sources


def clone_with_partition(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    group_sources: dict[tuple[int, int, int, int], str],
    heldout_source: str,
) -> dict[str, np.ndarray]:
    out = dict(arrays)
    partition = np.asarray(arrays["partition"], dtype=object).copy()
    for record in records:
        index = int(record["group_index"])
        if partition[index] != "train":
            continue
        key = tuple(int(record["key"][name]) for name in ("real_seed", "seed", "source_index", "candidate_index"))
        if group_sources.get(key, "unknown") == heldout_source:
            partition[index] = "oof_holdout"
    out["partition"] = partition.astype(str)
    return out


def nonnearest_positions(arrays: dict[str, np.ndarray], group_index: int) -> list[int]:
    return [
        pos
        for pos in range(arrays["mask"].shape[1])
        if bool(arrays["mask"][group_index, pos]) and not bool(arrays["is_nearest"][group_index, pos])
    ]


def sorted_nonnearest(arrays: dict[str, np.ndarray], scores: np.ndarray, group_index: int) -> list[int]:
    candidate_count = arrays["mask"].shape[1]
    return sorted(
        nonnearest_positions(arrays, group_index),
        key=lambda pos: (
            float(scores[aux_ranker.flat_index(group_index, pos, candidate_count)]),
            -int(arrays["topk_ranks"][group_index, pos]),
        ),
        reverse=True,
    )


def group_features(
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    teacher_pred: np.ndarray,
    group_index: int,
) -> tuple[np.ndarray, int]:
    candidate_count = arrays["mask"].shape[1]
    ranked = sorted_nonnearest(arrays, scores, group_index)
    if not ranked:
        return np.zeros(18, dtype=np.float32), int(np.flatnonzero(arrays["is_nearest"][group_index])[0])
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else ranked[0]
    score_values = np.asarray(
        [float(scores[aux_ranker.flat_index(group_index, pos, candidate_count)]) for pos in ranked],
        dtype=np.float32,
    )
    teacher_values = np.asarray(
        [float(teacher_pred[aux_ranker.flat_index(group_index, pos, candidate_count)]) for pos in ranked],
        dtype=np.float32,
    )
    top_flat = aux_ranker.flat_index(group_index, top, candidate_count)
    second_flat = aux_ranker.flat_index(group_index, second, candidate_count)
    core = arrays["core"][group_index, top]
    code_vec = arrays["code"][group_index, top]
    window_tensor = arrays["window"][group_index, top]
    features = np.asarray(
        [
            float(scores[top_flat]),
            float(scores[top_flat] - scores[second_flat]),
            float(score_values.mean()),
            float(score_values.std()),
            float(score_values.max() - score_values.min()),
            float(teacher_pred[top_flat]),
            float(teacher_values.mean()),
            float(teacher_values.std()),
            float(arrays["topk_ranks"][group_index, top]),
            float(core[1]),
            float(arrays["code_indices"][group_index, top] == arrays["code_indices"][group_index, second]),
            float(np.abs(code_vec).mean()),
            float(np.square(code_vec).mean()),
            float(np.abs(window_tensor).mean()),
            float(np.square(window_tensor).mean()),
            float(core[0]),
            float(core[2]),
            1.0,
        ],
        dtype=np.float32,
    )
    return features, top


def selected_status(arrays: dict[str, np.ndarray], record: dict[str, Any], selected_pos: int) -> str:
    selected_code = int(arrays["code_indices"][int(record["group_index"]), selected_pos])
    return table_ranker.classify(selected_code, int(record["nearest_code"]), int(record["oracle_code"]))


def build_group_matrix(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    scores: np.ndarray,
    teacher_pred: np.ndarray,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows = []
    labels = []
    metas = []
    for record in records:
        group_index = int(record["group_index"])
        if arrays["partition"][group_index] != partition:
            continue
        feats, top_pos = group_features(arrays, scores, teacher_pred, group_index)
        status = selected_status(arrays, record, top_pos)
        rows.append(feats)
        labels.append(1 if status == "exact" and int(record["oracle_code"]) != int(record["nearest_code"]) else 0)
        metas.append(
            {
                "group_index": group_index,
                "top_pos": int(top_pos),
                "status_if_changed": status,
                "oracle_changes": bool(int(record["oracle_code"]) != int(record["nearest_code"])),
            }
        )
    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=np.int64), metas


def fit_acceptors(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> dict[str, Any]:
    if len(set(int(value) for value in y_train.tolist())) < 2:
        return {"constant": ConstantAcceptor(float(y_train.mean()))}
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ).fit(x_train, y_train),
        "hist_gradient": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            l2_regularization=0.02,
            random_state=seed,
        ).fit(x_train, y_train),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ).fit(x_train, y_train),
    }


class ConstantAcceptor:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.probability, dtype=np.float32)
        return np.stack([1.0 - p, p], axis=1)


def predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.zeros(len(x), dtype=np.float32)
    return proba[:, 1].astype(np.float32)


def evaluate_policy(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    scores: np.ndarray,
    teacher_pred: np.ndarray,
    partition: str,
    accept_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    counts = Counter()
    t_sum = 0
    p_sum = 0
    t_worse = 0
    p_worse = 0
    audits = []
    row = 0
    for record in records:
        group_index = int(record["group_index"])
        if arrays["partition"][group_index] != partition:
            continue
        _, top_pos = group_features(arrays, scores, teacher_pred, group_index)
        accept = bool(float(accept_scores[row]) >= threshold)
        nearest_pos = int(record["nearest_pos"])
        selected_pos = top_pos if accept else nearest_pos
        status = selected_status(arrays, record, selected_pos)
        selected_code = int(arrays["code_indices"][group_index, selected_pos])
        changed = selected_code != int(record["nearest_code"])
        counts[status] += 1
        counts["changed_groups"] += int(changed)
        counts["oracle_change_groups"] += int(int(record["oracle_code"]) != int(record["nearest_code"]))
        counts["exact_changed_groups"] += int(status == "exact" and changed)
        if changed:
            t_delta = int(arrays["tesseract_delta"][group_index, selected_pos])
            p_delta = int(arrays["parseq_delta"][group_index, selected_pos])
        else:
            t_delta = 0
            p_delta = 0
        t_sum += t_delta
        p_sum += p_delta
        t_worse += int(t_delta > 0)
        p_worse += int(p_delta > 0)
        audits.append(
            {
                "group_index": group_index,
                "key": record["key"],
                "accept_score": float(accept_scores[row]),
                "threshold": float(threshold),
                "accepted": accept,
                "selected_code": selected_code,
                "top_code": int(arrays["code_indices"][group_index, top_pos]),
                "nearest_code": int(record["nearest_code"]),
                "oracle_code": int(record["oracle_code"]),
                "status": status,
                "selected_tesseract_delta_vs_nearest": int(t_delta),
                "selected_parseq_delta_vs_nearest": int(p_delta),
            }
        )
        row += 1
    groups = row
    return {
        "groups": int(groups),
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
        "parseq_worse_groups": int(p_worse),
        "audits": audits[:20],
    }


def tune_threshold(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    scores: np.ndarray,
    teacher_pred: np.ndarray,
    accept_scores: np.ndarray,
    max_false_wrong: int,
) -> tuple[float, dict[str, Any]]:
    candidates = sorted(set(float(x) for x in accept_scores), reverse=True)
    candidates = [max(candidates) + 1.0] + candidates + [min(candidates) - 1.0]
    best_threshold = candidates[0]
    best_metrics = evaluate_policy(arrays, records, scores, teacher_pred, "train", accept_scores, best_threshold)
    best_key = (
        10**9,
        10**9,
        10**9,
        0,
        0,
    )
    for threshold in candidates:
        metrics = evaluate_policy(arrays, records, scores, teacher_pred, "train", accept_scores, threshold)
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
    best_metrics = {k: v for k, v in best_metrics.items() if k != "audits"}
    return best_threshold, best_metrics


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Source-OOF no-op guard over the CRNN/ABINet teacher-auxiliary latent/code ranker.",
        "Teacher losses shape the ranker during training only; acceptors see only ranker-derived/deployable group features.",
        "",
        "## Validation Policies",
        "",
        "| teacher weight | seed | acceptor | budget | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | T worse |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        for policy in run["policies"]:
            metrics = policy["val_metrics"]
            lines.append(
                "| "
                f"{run['teacher_weight']:.3f} | {run['model_seed']} | {policy['acceptor']} | {policy['max_false_wrong']} | "
                f"{metrics['changed_groups']} | {metrics['exact_changed_groups']} | {metrics['false_change']} | "
                f"{metrics['wrong_change']} | {metrics['missed_oracle']} | {metrics['tesseract_delta_vs_nearest']} | "
                f"{metrics['parseq_delta_vs_nearest']} | {metrics['tesseract_worse_groups']} |"
            )
    best = result["best_policy"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
            "## Best Policy",
            "",
            "```json",
            json.dumps(best, indent=2),
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--crnn", type=Path, required=True)
    parser.add_argument("--abinet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, action="append")
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--max-false-wrong", type=int, action="append")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.teacher_weight is None:
        args.teacher_weight = [0.0, 0.20]
    if args.model_seed is None:
        args.model_seed = [0]
    if args.max_false_wrong is None:
        args.max_false_wrong = [0, 1, 2]

    table_rows = aux_ranker.read_jsonl(args.table)
    data = aux_ranker.load_npz(args.features)
    arrays, records, _ = aux_ranker.build_arrays(
        data,
        table_rows,
        aux_ranker.build_teacher_lookup(args.crnn),
        aux_ranker.build_teacher_lookup(args.abinet),
        args.profile,
    )
    group_sources = build_group_sources(table_rows)
    train_sources = sorted(
        {
            group_sources[
                tuple(int(record["key"][name]) for name in ("real_seed", "seed", "source_index", "candidate_index"))
            ]
            for record in records
            if arrays["partition"][int(record["group_index"])] == "train"
        }
    )

    runs = []
    for teacher_weight in args.teacher_weight:
        for model_seed in args.model_seed:
            candidate_count = arrays["mask"].shape[1]
            oof_scores = np.zeros(arrays["mask"].size, dtype=np.float32)
            oof_teacher = np.zeros(arrays["mask"].size, dtype=np.float32)
            fold_summaries = []
            for source in train_sources:
                fold_arrays = clone_with_partition(arrays, records, group_sources, source)
                scores, teacher_pred, meta = aux_ranker.train_one(fold_arrays, args, model_seed, teacher_weight)
                heldout_groups = 0
                for record in records:
                    group_index = int(record["group_index"])
                    if arrays["partition"][group_index] != "train":
                        continue
                    key = tuple(int(record["key"][name]) for name in ("real_seed", "seed", "source_index", "candidate_index"))
                    if group_sources.get(key, "unknown") != source:
                        continue
                    start = group_index * candidate_count
                    stop = start + candidate_count
                    oof_scores[start:stop] = scores[start:stop]
                    oof_teacher[start:stop] = teacher_pred[start:stop]
                    heldout_groups += 1
                fold_summaries.append({"source": source, "heldout_train_groups": heldout_groups, "train_meta": meta})

            full_scores, full_teacher, full_meta = aux_ranker.train_one(arrays, args, model_seed, teacher_weight)
            x_train, y_train, train_metas = build_group_matrix(arrays, records, oof_scores, oof_teacher, "train")
            x_val, y_val, val_metas = build_group_matrix(arrays, records, full_scores, full_teacher, "val")
            acceptors = fit_acceptors(x_train, y_train, model_seed)
            policies = []
            for name, model in acceptors.items():
                train_prob = predict_proba(model, x_train)
                val_prob = predict_proba(model, x_val)
                for budget in args.max_false_wrong:
                    threshold, train_metrics = tune_threshold(
                        arrays, records, oof_scores, oof_teacher, train_prob, int(budget)
                    )
                    val_metrics = evaluate_policy(
                        arrays, records, full_scores, full_teacher, "val", val_prob, threshold
                    )
                    policies.append(
                        {
                            "acceptor": name,
                            "max_false_wrong": int(budget),
                            "threshold": float(threshold),
                            "train_metrics": train_metrics,
                            "val_metrics": {k: v for k, v in val_metrics.items() if k != "audits"},
                            "val_audits": val_metrics["audits"],
                        }
                    )
            # Direct score-threshold control over top candidate score.
            direct_train_scores = x_train[:, 0]
            direct_val_scores = x_val[:, 0]
            for budget in args.max_false_wrong:
                threshold, train_metrics = tune_threshold(
                    arrays, records, oof_scores, oof_teacher, direct_train_scores, int(budget)
                )
                val_metrics = evaluate_policy(
                    arrays, records, full_scores, full_teacher, "val", direct_val_scores, threshold
                )
                policies.append(
                    {
                        "acceptor": "direct_top_score",
                        "max_false_wrong": int(budget),
                        "threshold": float(threshold),
                        "train_metrics": train_metrics,
                        "val_metrics": {k: v for k, v in val_metrics.items() if k != "audits"},
                        "val_audits": val_metrics["audits"],
                    }
                )
            runs.append(
                {
                    "teacher_weight": float(teacher_weight),
                    "model_seed": int(model_seed),
                    "train_exact_selected_oof": int(y_train.sum()),
                    "val_exact_selected_full_before_guard": int(y_val.sum()),
                    "folds": fold_summaries,
                    "full_train_meta": full_meta,
                    "policies": policies,
                }
            )

    def policy_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int, int, int, int]:
        _, policy = item
        metrics = policy["val_metrics"]
        return (
            metrics["tesseract_delta_vs_nearest"],
            metrics["false_change"] + metrics["wrong_change"],
            metrics["tesseract_worse_groups"],
            -metrics["exact_changed_groups"],
            metrics["changed_groups"],
        )

    candidates = [(run, policy) for run in runs for policy in run["policies"]]
    best_run, best_policy = sorted(candidates, key=policy_key)[0]
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_crossfit_guard_not_promoted",
        "inputs": {
            "features": {"path": str(args.features), "sha256": sha256_file(args.features)},
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "crnn": {"path": str(args.crnn), "sha256": sha256_file(args.crnn)},
            "abinet": {"path": str(args.abinet), "sha256": sha256_file(args.abinet)},
        },
        "config": {
            "profile": args.profile,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pairwise_weight": args.pairwise_weight,
            "margin": args.margin,
            "teacher_weights": args.teacher_weight,
            "model_seeds": args.model_seed,
            "max_false_wrong": args.max_false_wrong,
            "train_sources": train_sources,
        },
        "data_summary": {
            "groups": int(arrays["mask"].shape[0]),
            "candidates_per_group": int(arrays["mask"].shape[1]),
            "train_groups": int(np.sum(arrays["partition"] == "train")),
            "val_groups": int(np.sum(arrays["partition"] == "val")),
        },
        "runs": runs,
        "best_policy": {
            "teacher_weight": best_run["teacher_weight"],
            "model_seed": best_run["model_seed"],
            **best_policy,
        },
        "interpretation": (
            "This tests whether source-OOF ranker scores can train a separate no-op guard for "
            "the teacher-auxiliary candidate ranker. It is a pre-promotion diagnostic; counted "
            "OSCR export remains blocked unless the validation policy exceeds the current "
            "actual-bitstream -8 Tesseract floor with low false/wrong changes."
        ),
        "hashes": {"script": sha256_file(Path(__file__))},
    }
    write_json(args.output, result)
    write_report(args.report, result)


if __name__ == "__main__":
    main()
