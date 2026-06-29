#!/usr/bin/env python3
"""Train a top-k selector with score, visual and latent/code-effect evidence.

This is a non-promotion diagnostic. It reuses the source-OOF top-k shortlist,
then augments the shallow selector features with OSCAR-LIC-native latent and
codebook summaries from the exported top8 window artifact, plus optional cached
fixed visual embeddings. OCR deltas are labels/evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_assignment_topk_policy_selector as base  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_stats(vec: np.ndarray) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return [
        float(arr.mean()),
        float(arr.std()),
        float(np.mean(np.abs(arr))),
        float(np.linalg.norm(arr) / np.sqrt(max(arr.size, 1))),
        float(np.max(np.abs(arr))) if arr.size else 0.0,
    ]


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def normalized_l2(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32).reshape(-1)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(av - bv) / np.sqrt(max(av.size, 1)))


def find_slot(z: dict[str, np.ndarray], group_index: int, row: dict[str, Any]) -> int:
    code = int(row["code_index"])
    rank = int(row["topk_rank"])
    mask = np.asarray(z["mask"][group_index], dtype=bool)
    code_indices = np.asarray(z["code_indices"][group_index], dtype=np.int64)
    topk_ranks = np.asarray(z["topk_ranks"][group_index], dtype=np.int64)
    matches = np.where(mask & (code_indices == code) & (topk_ranks == rank))[0]
    if len(matches):
        return int(matches[0])
    matches = np.where(mask & (code_indices == code))[0]
    if len(matches):
        return int(matches[0])
    raise KeyError(f"no latent slot for key={base.group_key(row)} code={code} rank={rank}")


def build_latent_code_features(rows: list[dict], artifact: Path) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.load(artifact)
    needed = [
        "baseline_score",
        "code_center",
        "code_indices",
        "core_features",
        "delta_center",
        "group_keys",
        "is_nearest",
        "latent_residual",
        "latent_target",
        "latent_window_residual",
        "latent_window_target",
        "latent_window_yhat",
        "latent_yhat",
        "mask",
        "nearest_center",
        "nearest_indices",
        "topk_ranks",
    ]
    data = {key: np.asarray(raw[key]) for key in needed}
    group_to_index = {tuple(int(v) for v in group): i for i, group in enumerate(data["group_keys"])}
    features: list[list[float]] = []
    missing = 0
    for row in rows:
        key = base.group_key(row)
        if key not in group_to_index:
            missing += 1
            features.append([0.0] * 67)
            continue
        gi = group_to_index[key]
        slot = find_slot(data, gi, row)
        code = np.asarray(data["code_center"][gi, slot], dtype=np.float32)
        nearest = np.asarray(data["nearest_center"][gi, slot], dtype=np.float32)
        delta = np.asarray(data["delta_center"][gi, slot], dtype=np.float32)
        yhat = np.asarray(data["latent_yhat"][gi, slot], dtype=np.float32)
        target = np.asarray(data["latent_target"][gi, slot], dtype=np.float32)
        residual = np.asarray(data["latent_residual"][gi, slot], dtype=np.float32)
        win_yhat = np.asarray(data["latent_window_yhat"][gi, slot], dtype=np.float32)
        win_target = np.asarray(data["latent_window_target"][gi, slot], dtype=np.float32)
        win_residual = np.asarray(data["latent_window_residual"][gi, slot], dtype=np.float32)
        row_features: list[float] = []
        for vec in [code, nearest, delta, yhat, target, residual]:
            row_features.extend(vector_stats(vec))
        row_features.extend(
            [
                safe_cosine(delta, residual),
                safe_cosine(code, target),
                safe_cosine(nearest, yhat),
                safe_cosine(code - nearest, target - yhat),
                normalized_l2(delta, residual),
                normalized_l2(code, target),
                normalized_l2(nearest, yhat),
            ]
        )
        for tensor in [win_yhat, win_target, win_residual, win_target - win_yhat]:
            row_features.extend(vector_stats(tensor))
        row_features.extend(
            [
                float(data["baseline_score"][gi, slot]),
                float(data["is_nearest"][gi, slot]),
                float(data["topk_ranks"][gi, slot]) / 7.0,
                float(data["code_indices"][gi, slot]) / 63.0,
                float(data["nearest_indices"][gi, slot]) / 63.0,
                float(abs(int(data["code_indices"][gi, slot]) - int(data["nearest_indices"][gi, slot]))) / 63.0,
                *[float(v) for v in np.asarray(data["core_features"][gi, slot], dtype=np.float32).reshape(-1)],
            ]
        )
        features.append(row_features)
    array = np.asarray(features, dtype=np.float32)
    metadata = {
        "path": str(artifact),
        "sha256": sha256_file(artifact),
        "feature_dim": int(array.shape[1]),
        "missing_rows": int(missing),
    }
    return array, metadata


def load_alexnet_features(cache: Path, expected_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(cache, allow_pickle=True)
    feats = np.asarray(data["alexnet_features"], dtype=np.float32)
    if feats.shape[0] != expected_rows:
        raise SystemExit(f"alexnet feature row mismatch: {feats.shape[0]} != {expected_rows}")
    metadata = {
        "path": str(cache),
        "sha256": sha256_file(cache),
        "feature_dim": int(feats.shape[1]),
        "encoder_checkpoint": str(data["encoder_checkpoint"]) if "encoder_checkpoint" in data else None,
        "encoder_sha256": str(data["encoder_sha256"]) if "encoder_sha256" in data else None,
    }
    return feats, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--latent-artifact", type=Path, required=True)
    parser.add_argument("--alexnet-feature-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-output", type=Path)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--score-model-seed", type=int, action="append")
    parser.add_argument("--selector-seed", type=int, action="append")
    parser.add_argument("--shortlist-topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--harmful-margin-weight", type=float, default=1.5)
    parser.add_argument("--harmful-margin", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.score_model_seed:
        args.score_model_seed = [2]
    if not args.selector_seed:
        args.selector_seed = [0, 1, 2]

    rows = base.read_jsonl(args.table)
    groups = base.grouped_indices(rows)
    base_features = base.build_base_features(rows)
    latent_features, latent_meta = build_latent_code_features(rows, args.latent_artifact)
    feature_blocks = [("base_score_image_scalar", base_features), ("latent_code_summary", latent_features)]
    alexnet_meta = None
    if args.alexnet_feature_cache is not None:
        alexnet_features, alexnet_meta = load_alexnet_features(args.alexnet_feature_cache, len(rows))
        feature_blocks.append(("alexnet_patch_embedding", alexnet_features))

    oracle_by_model = base.load_score_features(args.oracle_score)
    safe_by_model = base.load_score_features(args.safe_score)
    runs: list[dict[str, Any]] = []
    feature_meta_by_seed: dict[str, Any] = {}
    for score_model_seed in args.score_model_seed:
        oracle_scores = np.asarray(
            [oracle_by_model[score_model_seed][idx] for idx in range(len(rows))],
            dtype=np.float32,
        )
        safe_scores = np.asarray(
            [safe_by_model[score_model_seed][idx] for idx in range(len(rows))],
            dtype=np.float32,
        )
        selector_features = base.build_selector_features(rows, groups, base_features, oracle_scores, safe_scores)
        combined = np.concatenate(
            [selector_features.astype(np.float32)]
            + [block.astype(np.float32) for _, block in feature_blocks[1:]],
            axis=1,
        )
        feature_meta_by_seed[str(score_model_seed)] = {"feature_dim": int(combined.shape[1])}
        if args.feature_output is not None:
            args.feature_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.feature_output,
                features=combined,
                score_model_seed=np.asarray([score_model_seed], dtype=np.int64),
            )
        group_candidates = {
            key: base.candidate_indices_for_group(rows, indices, oracle_scores, safe_scores, topk=args.shortlist_topk)
            for key, indices in groups.items()
        }
        train_args = argparse.Namespace(
            val_seed=args.val_seed,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            harmful_margin_weight=args.harmful_margin_weight,
            harmful_margin=args.harmful_margin,
            cpu=args.cpu,
            score_output=None,
        )
        for selector_seed in args.selector_seed:
            runs.append(base.train_one(rows, groups, combined, group_candidates, score_model_seed, selector_seed, train_args))

    result = {
        "description": "Top-k assignment selector with source-OOF scores plus latent/code summaries and optional fixed visual embeddings. Non-promotion diagnostic only.",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "latent_artifact": str(args.latent_artifact),
            "alexnet_feature_cache": None if args.alexnet_feature_cache is None else str(args.alexnet_feature_cache),
            "val_seed": int(args.val_seed),
            "score_model_seeds": args.score_model_seed,
            "selector_seeds": args.selector_seed,
            "shortlist_topk": int(args.shortlist_topk),
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "harmful_margin_weight": float(args.harmful_margin_weight),
            "harmful_margin": float(args.harmful_margin),
            "cpu": bool(args.cpu),
        },
        "feature_blocks": [
            {"name": name, "feature_dim": int(block.shape[1])} for name, block in feature_blocks
        ],
        "latent_artifact": latent_meta,
        "alexnet_feature_cache": alexnet_meta,
        "features_by_score_seed": feature_meta_by_seed,
        "per_run": runs,
        "aggregate": base.aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
