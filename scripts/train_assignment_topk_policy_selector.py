#!/usr/bin/env python3
"""Train a top-k assignment policy selector with explicit no-op option."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", row.get("seed", 0)))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def grouped_indices(rows: list[dict]) -> dict[tuple[int, int, int, int], list[int]]:
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_key(row)].append(index)
    return dict(groups)


def load_score_features(path: Path) -> dict[int, dict[int, float]]:
    by_model: dict[int, dict[int, float]] = defaultdict(dict)
    for row in read_jsonl(path):
        by_model[int(row["model_seed"])][int(row["row_index"])] = float(row["score"])
    return dict(by_model)


def rank_map(indices: list[int], scores: np.ndarray) -> dict[int, int]:
    ordered = sorted(indices, key=lambda idx: (-float(scores[idx]), idx))
    return {idx: rank + 1 for rank, idx in enumerate(ordered)}


def build_base_features(rows: list[dict]) -> np.ndarray:
    optional_prefixes = ["codebook_", "img_"]
    optional = sorted(
        key
        for key, value in rows[0].items()
        if any(key.startswith(prefix) for prefix in optional_prefixes) and isinstance(value, (int, float))
    )
    values = []
    for row in rows:
        rel = float(row.get("assignment_relative_error", 1.0))
        rel_clipped = min(max(rel, 1.0), 10.0)
        base = [
            float(row["topk_rank"]) / 7.0,
            rel_clipped,
            np.log1p(rel_clipped - 1.0),
            float(row["is_nearest"]),
            float(row.get("code_equals_nearest", int(row["is_nearest"]))),
        ]
        base.extend(float(row[key]) for key in optional)
        values.append(base)
    return np.asarray(values, dtype=np.float32)


def build_selector_features(
    rows: list[dict],
    groups: dict[tuple[int, int, int, int], list[int]],
    base_features: np.ndarray,
    oracle_scores: np.ndarray,
    safe_scores: np.ndarray,
) -> np.ndarray:
    features = np.zeros((len(rows), base_features.shape[1] + 8), dtype=np.float32)
    features[:, : base_features.shape[1]] = base_features
    for indices in groups.values():
        nearest = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
        oracle_ranks = rank_map(indices, oracle_scores)
        safe_ranks = rank_map(indices, safe_scores)
        oracle_nearest = float(oracle_scores[nearest])
        safe_nearest = float(safe_scores[nearest])
        oracle_group_max = max(float(oracle_scores[idx]) for idx in indices)
        safe_group_max = max(float(safe_scores[idx]) for idx in indices)
        for idx in indices:
            offset = base_features.shape[1]
            features[idx, offset + 0] = float(oracle_scores[idx])
            features[idx, offset + 1] = float(safe_scores[idx])
            features[idx, offset + 2] = float(oracle_scores[idx]) - oracle_nearest
            features[idx, offset + 3] = float(safe_scores[idx]) - safe_nearest
            features[idx, offset + 4] = 1.0 / float(oracle_ranks[idx])
            features[idx, offset + 5] = 1.0 / float(safe_ranks[idx])
            features[idx, offset + 6] = oracle_group_max - oracle_nearest
            features[idx, offset + 7] = safe_group_max - safe_nearest
    return features


def candidate_indices_for_group(
    rows: list[dict],
    group: list[int],
    oracle_scores: np.ndarray,
    safe_scores: np.ndarray,
    topk: int,
) -> list[int]:
    nearest = next(idx for idx in group if int(rows[idx]["is_nearest"]))
    chosen = {nearest}
    for scores in [oracle_scores, safe_scores]:
        ordered = sorted(group, key=lambda idx: (-float(scores[idx]), int(rows[idx]["topk_rank"])))
        chosen.update(ordered[:topk])
    return sorted(chosen, key=lambda idx: (0 if int(rows[idx]["is_nearest"]) else 1, int(rows[idx]["topk_rank"])))


class Selector(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).reshape(-1)


def policy_metrics(rows: list[dict], selected: list[int]) -> dict:
    parseq_delta = int(sum(int(rows[idx]["parseq_delta_vs_nearest"]) for idx in selected))
    tesseract_delta = int(sum(int(rows[idx]["tesseract_delta_vs_nearest"]) for idx in selected))
    changed = int(sum(1 for idx in selected if not int(rows[idx]["is_nearest"])))
    false_changes = int(
        sum(1 for idx in selected if not int(rows[idx]["is_nearest"]) and not int(rows[idx]["label_assignment_oracle_choice"]))
    )
    wrong_changes = int(
        sum(
            1
            for idx in selected
            if not int(rows[idx]["is_nearest"])
            and any(int(row["label_assignment_oracle_choice"]) for row in rows if False)
        )
    )
    # Compute wrong changes group-wise below where group context is available; keep this key for compatibility.
    return {
        "groups": len(selected),
        "changed_groups": changed,
        "false_changes": false_changes,
        "wrong_changes": wrong_changes,
        "parseq_delta_sum": parseq_delta,
        "tesseract_delta_sum": tesseract_delta,
        "multi_teacher_delta_sum": parseq_delta + tesseract_delta,
        "parseq_worsen_groups": int(sum(1 for idx in selected if int(rows[idx]["parseq_delta_vs_nearest"]) > 0)),
        "tesseract_improve_groups": int(sum(1 for idx in selected if int(rows[idx]["tesseract_delta_vs_nearest"]) < 0)),
        "tesseract_worsen_groups": int(sum(1 for idx in selected if int(rows[idx]["tesseract_delta_vs_nearest"]) > 0)),
    }


def policy_metrics_by_group(rows: list[dict], groups: dict[tuple[int, int, int, int], list[int]], selected: list[int]) -> dict:
    metrics = policy_metrics(rows, selected)
    selected_by_group = {group_key(rows[idx]): idx for idx in selected}
    wrong = 0
    missed = 0
    exact = 0
    for key, indices in groups.items():
        selected_idx = selected_by_group[key]
        oracle = [idx for idx in indices if int(rows[idx]["label_assignment_oracle_choice"])]
        nearest = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
        oracle_idx = oracle[0] if oracle else nearest
        if selected_idx == oracle_idx:
            if selected_idx != nearest:
                exact += 1
        elif oracle_idx != nearest and selected_idx == nearest:
            missed += 1
        elif selected_idx != nearest:
            wrong += 1
    metrics["wrong_changes"] = wrong
    metrics["missed_oracle_changes"] = missed
    metrics["exact_oracle_changes"] = exact
    return metrics


def select_argmax(
    rows: list[dict],
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    logits: np.ndarray,
    margin: float | None = None,
) -> list[int]:
    selected = []
    for candidates in group_candidates.values():
        nearest = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        best = max(candidates, key=lambda idx: (float(logits[idx]), -int(rows[idx]["topk_rank"])))
        if margin is not None and best != nearest and float(logits[best]) - float(logits[nearest]) < margin:
            best = nearest
        selected.append(best)
    return selected


def tune_margin(
    rows: list[dict],
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    metric_groups: dict[tuple[int, int, int, int], list[int]],
    logits: np.ndarray,
) -> float:
    margins = []
    for candidates in group_candidates.values():
        nearest = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        best = max(candidates, key=lambda idx: (float(logits[idx]), -int(rows[idx]["topk_rank"])))
        margins.append(float(logits[best]) - float(logits[nearest]))
    candidates = sorted(set(margins))
    if not candidates:
        return 1e6
    candidates = [max(candidates) + 1e-6] + candidates + [min(candidates) - 1e-6]
    best_margin = candidates[0]
    best_key = None
    for margin in candidates:
        selected = select_argmax(rows, group_candidates, logits, margin=margin)
        metrics = policy_metrics_by_group(rows, metric_groups, selected)
        key = (
            metrics["parseq_worsen_groups"],
            metrics["tesseract_worsen_groups"],
            metrics["wrong_changes"],
            metrics["tesseract_delta_sum"],
            metrics["false_changes"],
            metrics["changed_groups"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_margin = float(margin)
    return best_margin


def oracle_select(rows: list[dict], group_candidates: dict[tuple[int, int, int, int], list[int]]) -> list[int]:
    selected = []
    for candidates in group_candidates.values():
        oracle = [idx for idx in candidates if int(rows[idx]["label_assignment_oracle_choice"])]
        if oracle:
            selected.append(oracle[0])
        else:
            selected.append(next(idx for idx in candidates if int(rows[idx]["is_nearest"])))
    return selected


def train_one(
    rows: list[dict],
    groups: dict[tuple[int, int, int, int], list[int]],
    features: np.ndarray,
    group_candidates: dict[tuple[int, int, int, int], list[int]],
    score_model_seed: int,
    selector_seed: int,
    args: argparse.Namespace,
) -> dict:
    random.seed(selector_seed)
    np.random.seed(selector_seed)
    torch.manual_seed(selector_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_groups = {key: cand for key, cand in group_candidates.items() if int(rows[cand[0]]["seed"]) != args.val_seed}
    val_groups = {key: cand for key, cand in group_candidates.items() if int(rows[cand[0]]["seed"]) == args.val_seed}
    full_train_groups = {key: groups[key] for key in train_groups}
    full_val_groups = {key: groups[key] for key in val_groups}
    train_row_indices = sorted({idx for candidates in train_groups.values() for idx in candidates})
    train_features_np = features[train_row_indices]
    feat_mean = train_features_np.mean(axis=0, keepdims=True)
    feat_std = train_features_np.std(axis=0, keepdims=True)
    feat_std[feat_std < 1e-6] = 1.0
    norm = ((features - feat_mean) / feat_std).astype(np.float32)

    model = Selector(features.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    x = torch.from_numpy(norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_positions = []
    train_targets = []
    harmful_positions = []
    nearest_positions = []
    for candidates in train_groups.values():
        target_matches = [idx for idx in candidates if int(rows[idx]["label_assignment_oracle_choice"])]
        if target_matches:
            target = target_matches[0]
        else:
            target = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        train_positions.append(torch.tensor(candidates, dtype=torch.long, device=device))
        train_targets.append(torch.tensor([candidates.index(target)], dtype=torch.long, device=device))
        nearest = next(idx for idx in candidates if int(rows[idx]["is_nearest"]))
        for idx in candidates:
            if idx != nearest and (
                int(rows[idx]["parseq_delta_vs_nearest"]) > 0 or int(rows[idx]["tesseract_delta_vs_nearest"]) > 0
            ):
                harmful_positions.append(idx)
                nearest_positions.append(nearest)
    harmful_positions_t = torch.tensor(harmful_positions, dtype=torch.long, device=device)
    nearest_positions_t = torch.tensor(nearest_positions, dtype=torch.long, device=device)

    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x)
        losses = [
            torch.nn.functional.cross_entropy(logits[pos].unsqueeze(0), target)
            for pos, target in zip(train_positions, train_targets)
        ]
        loss = torch.stack(losses).mean()
        if len(harmful_positions_t) and args.harmful_margin_weight > 0:
            harm_loss = torch.nn.functional.softplus(
                logits[harmful_positions_t] - logits[nearest_positions_t] + args.harmful_margin
            ).mean()
            loss = loss + args.harmful_margin_weight * harm_loss
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()

    train_margin = tune_margin(rows, train_groups, full_train_groups, logits)
    policies = {
        "nearest": policy_metrics_by_group(
            rows,
            full_val_groups,
            [next(idx for idx in cand if int(rows[idx]["is_nearest"])) for cand in val_groups.values()],
        ),
        "shortlist_oracle": policy_metrics_by_group(rows, full_val_groups, oracle_select(rows, val_groups)),
        "argmax": policy_metrics_by_group(rows, full_val_groups, select_argmax(rows, val_groups, logits)),
        "train_tuned_margin": policy_metrics_by_group(
            rows,
            full_val_groups,
            select_argmax(rows, val_groups, logits, margin=train_margin),
        ),
    }

    if args.score_output is not None:
        with args.score_output.open("a", encoding="utf-8") as handle:
            for row_index, row in enumerate(rows):
                handle.write(
                    json.dumps(
                        {
                            "row_index": row_index,
                            "score_model_seed": score_model_seed,
                            "selector_seed": selector_seed,
                            "logit": float(logits[row_index]),
                            "seed": int(row["seed"]),
                            "real_seed": int(row.get("real_seed", row["seed"])),
                            "source_index": int(row["source_index"]),
                            "candidate_index": int(row["candidate_index"]),
                            "code_index": int(row["code_index"]),
                            "topk_rank": int(row["topk_rank"]),
                            "is_nearest": int(row["is_nearest"]),
                            "in_shortlist": int(any(row_index in cand for cand in group_candidates.values())),
                            "label_assignment_oracle_choice": int(row["label_assignment_oracle_choice"]),
                            "label_tesseract_parseq_safe_improves": int(row["label_tesseract_parseq_safe_improves"]),
                            "parseq_delta_vs_nearest": int(row["parseq_delta_vs_nearest"]),
                            "tesseract_delta_vs_nearest": int(row["tesseract_delta_vs_nearest"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return {
        "score_model_seed": score_model_seed,
        "selector_seed": selector_seed,
        "device": str(device),
        "feature_dim": int(features.shape[1]),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "train_margin": float(train_margin),
        "candidate_count_mean": float(np.mean([len(candidates) for candidates in val_groups.values()])),
        "policies": policies,
    }


def aggregate(runs: list[dict]) -> dict:
    policy_names = sorted({name for run in runs for name in run["policies"]})
    output = {"policy_metrics": {}}
    for policy in policy_names:
        metric_names = sorted({metric for run in runs for metric in run["policies"][policy]})
        output["policy_metrics"][policy] = {}
        for metric in metric_names:
            values = np.asarray([float(run["policies"][policy][metric]) for run in runs], dtype=np.float64)
            output["policy_metrics"][policy][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    output["candidate_count_mean"] = float(np.mean([run["candidate_count_mean"] for run in runs])) if runs else 0.0
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--score-model-seed", type=int, action="append", default=[0, 1, 2])
    parser.add_argument("--selector-seed", type=int, action="append", default=[0, 1, 2])
    parser.add_argument("--shortlist-topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--harmful-margin-weight", type=float, default=1.0)
    parser.add_argument("--harmful-margin", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.table)
    groups = grouped_indices(rows)
    base_features = build_base_features(rows)
    oracle_by_model = load_score_features(args.oracle_score)
    safe_by_model = load_score_features(args.safe_score)
    if args.score_output is not None and args.score_output.exists():
        args.score_output.unlink()

    runs = []
    for score_model_seed in args.score_model_seed:
        oracle_scores = np.asarray(
            [oracle_by_model[score_model_seed][idx] for idx in range(len(rows))],
            dtype=np.float32,
        )
        safe_scores = np.asarray(
            [safe_by_model[score_model_seed][idx] for idx in range(len(rows))],
            dtype=np.float32,
        )
        features = build_selector_features(rows, groups, base_features, oracle_scores, safe_scores)
        group_candidates = {
            key: candidate_indices_for_group(rows, indices, oracle_scores, safe_scores, topk=args.shortlist_topk)
            for key, indices in groups.items()
        }
        for selector_seed in args.selector_seed:
            runs.append(train_one(rows, groups, features, group_candidates, score_model_seed, selector_seed, args))

    result = {
        "description": "Top-k assignment policy selector with explicit nearest/no-op option.",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "val_seed": args.val_seed,
            "score_model_seeds": args.score_model_seed,
            "selector_seeds": args.selector_seed,
            "shortlist_topk": args.shortlist_topk,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "harmful_margin_weight": args.harmful_margin_weight,
            "harmful_margin": args.harmful_margin,
        },
        "per_run": runs,
        "aggregate": aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
