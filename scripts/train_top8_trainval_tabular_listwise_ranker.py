#!/usr/bin/env python3
"""Train a train-split tabular listwise ranker for top-8 assignment.

This diagnostic uses the existing Eval300 train/val assignment table. It trains
only on train-partition groups and evaluates candidate ranking on the held-out
val partition. OCR columns are used only for labels and evaluation, never as
input features.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


CORE_FEATURES = {"topk_rank", "assignment_relative_error"}
PREFIX_FEATURES = ("codebook_", "img_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def row_seed(row: dict[str, Any]) -> int:
    return int(row.get("real_seed", row.get("seed", 0)))


def group_key(row: dict[str, Any]) -> tuple[int, str, int, int]:
    return (
        row_seed(row),
        str(row.get("assignment_partition", row.get("split", ""))),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def numeric_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names = set()
    for row in rows:
        for key, value in row.items():
            if not is_number(value):
                continue
            if key in CORE_FEATURES or any(key.startswith(prefix) for prefix in PREFIX_FEATURES):
                if "oracle" in key or "label" in key:
                    continue
                if key.startswith(("parseq_", "tesseract_", "multi_teacher_")):
                    continue
                names.add(key)
    return sorted(names)


def feature_value(row: dict[str, Any], name: str) -> float:
    value = float(row.get(name, 0.0))
    if name == "assignment_relative_error":
        return math.log1p(min(max(value, 0.0), 1.0e8))
    if math.isfinite(value):
        return value
    return 0.0


def build_features(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray([[feature_value(row, name) for name in names] for row in rows], dtype=np.float32)


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int, int], list[int]]:
    groups: dict[tuple[int, str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_key(row)].append(index)
    for indices in groups.values():
        indices.sort(key=lambda idx: (int(rows[idx]["topk_rank"]), int(rows[idx]["code_index"]), idx))
    return groups


def nonnearest_indices(rows: list[dict[str, Any]], indices: list[int]) -> list[int]:
    return [idx for idx in indices if int(rows[idx]["code_index"]) != int(rows[idx]["nearest_code"])]


def target_index_for_group(rows: list[dict[str, Any]], indices: list[int]) -> int | None:
    nearest_code = int(rows[indices[0]]["nearest_code"])
    oracle_matches = [idx for idx in indices if int(rows[idx].get("label_assignment_oracle_choice", 0))]
    oracle_code = int(rows[oracle_matches[0]]["code_index"]) if oracle_matches else nearest_code
    if oracle_code == nearest_code:
        return None
    choices = [idx for idx in nonnearest_indices(rows, indices) if int(rows[idx]["code_index"]) == oracle_code]
    if not choices:
        return None
    return min(choices, key=lambda idx: (int(rows[idx]["topk_rank"]), idx))


def build_listwise_groups(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, str, int, int], list[int]],
    partition: str,
) -> list[dict[str, Any]]:
    out = []
    for key, indices in sorted(groups.items()):
        if key[1] != partition:
            continue
        target_idx = target_index_for_group(rows, indices)
        if target_idx is None:
            continue
        choices = nonnearest_indices(rows, indices)
        if target_idx not in choices or len(choices) < 2:
            continue
        out.append(
            {
                "key": key,
                "indices": choices,
                "target_local": choices.index(target_idx),
                "target_index": target_idx,
            }
        )
    return out


def classify(selected_code: int, nearest_code: int, oracle_code: int) -> str:
    if selected_code == oracle_code:
        return "exact"
    if oracle_code == nearest_code and selected_code != nearest_code:
        return "false_change"
    if oracle_code != nearest_code and selected_code == nearest_code:
        return "missed_oracle"
    return "wrong_change"


def rank_counts(values: list[int]) -> dict[str, int]:
    return {
        "le1": int(sum(value <= 1 for value in values)),
        "le2": int(sum(value <= 2 for value in values)),
        "le4": int(sum(value <= 4 for value in values)),
        "le8": int(sum(value <= 8 for value in values)),
    }


class TabularRanker(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def normalize(train: np.ndarray, all_features: np.ndarray) -> np.ndarray:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return ((all_features - mean) / std).astype(np.float32)


def train_model(
    x_all: torch.Tensor,
    train_groups: list[dict[str, Any]],
    model_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
    model = TabularRanker(x_all.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_all)
        losses = []
        for group in train_groups:
            idx = torch.tensor(group["indices"], dtype=torch.long, device=device)
            target = torch.tensor([int(group["target_local"])], dtype=torch.long, device=device)
            group_logits = logits[idx].reshape(1, -1)
            losses.append(F.cross_entropy(group_logits, target))
            if args.pairwise_weight > 0:
                target_logit = group_logits[0, int(group["target_local"])]
                neg = torch.cat([group_logits[0, : int(group["target_local"])], group_logits[0, int(group["target_local"]) + 1 :]])
                if neg.numel():
                    losses.append(args.pairwise_weight * F.softplus(-(target_logit - neg)).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        scores = model(x_all).detach().cpu().numpy()
    train_ranks = []
    for group in train_groups:
        group_scores = scores[group["indices"]]
        target_score = group_scores[int(group["target_local"])]
        train_ranks.append(int(1 + np.sum(group_scores > target_score)))
    return scores, {
        "model_seed": int(model_seed),
        "final_loss": float(last_loss),
        "train_target_rank": rank_counts(train_ranks),
    }


def evaluate_val(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, str, int, int], list[int]],
    scores: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audits = []
    counts = Counter()
    rank_values = []
    topk_values = []
    tess_delta = 0
    parseq_delta = 0
    for key, indices in sorted(groups.items()):
        if key[1] != "val":
            continue
        nearest = next(idx for idx in indices if int(rows[idx]["code_index"]) == int(rows[idx]["nearest_code"]))
        oracle = next((idx for idx in indices if int(rows[idx].get("label_assignment_oracle_choice", 0))), nearest)
        nearest_code = int(rows[nearest]["code_index"])
        oracle_code = int(rows[oracle]["code_index"])
        nonnearest = nonnearest_indices(rows, indices)
        by_score = sorted(nonnearest, key=lambda idx: (float(scores[idx]), -int(rows[idx]["topk_rank"])), reverse=True)
        best_idx = by_score[0] if by_score else nearest
        selected_idx = best_idx if oracle_code != nearest_code else nearest
        selected_code = int(rows[selected_idx]["code_index"])
        status = classify(selected_code, nearest_code, oracle_code)
        counts[status] += 1
        counts["changed_groups"] += int(selected_code != nearest_code)
        tess_delta += int(rows[selected_idx]["tesseract_delta_vs_nearest"])
        parseq_delta += int(rows[selected_idx]["parseq_delta_vs_nearest"])
        oracle_rank = None
        if oracle_code != nearest_code:
            oracle_rank = 1 + next(rank for rank, idx in enumerate(by_score) if int(rows[idx]["code_index"]) == oracle_code)
            rank_values.append(int(oracle_rank))
            topk_values.append(int(rows[oracle]["topk_rank"]))
        audits.append(
            {
                "key": {"real_seed": key[0], "partition": key[1], "source_index": key[2], "candidate_index": key[3]},
                "source": rows[nearest].get("source"),
                "reference": rows[nearest].get("reference"),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "selected_code_oracle_change_only": selected_code,
                "selected_status_oracle_change_only": status,
                "oracle_nonnearest_score_rank": oracle_rank,
                "oracle_topk_rank": int(rows[oracle]["topk_rank"]),
                "best_nonnearest_code": int(rows[best_idx]["code_index"]) if nonnearest else nearest_code,
                "best_nonnearest_score": float(scores[best_idx]) if nonnearest else 0.0,
                "selected_tesseract_delta_vs_nearest": int(rows[selected_idx]["tesseract_delta_vs_nearest"]),
                "selected_parseq_delta_vs_nearest": int(rows[selected_idx]["parseq_delta_vs_nearest"]),
            }
        )
    metrics = {
        "groups": int(sum(1 for key in groups if key[1] == "val")),
        "oracle_change_groups": int(len(rank_values)),
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "tesseract_delta_vs_nearest": int(tess_delta),
        "parseq_delta_vs_nearest": int(parseq_delta),
        "oracle_nonnearest_score_rank_counts": rank_counts(rank_values),
        "oracle_topk_rank_counts": rank_counts(topk_values),
    }
    return audits, metrics


def write_report(path: Path, result: dict[str, Any]) -> None:
    m = result["metrics"]["oracle_change_only_policy"]
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Train-partition tabular listwise ranker for Eval300 top-8 assignment.",
        "This is a candidate-ranking diagnostic, not an actual `.oscr` promotion.",
        "",
        "## Split",
        "",
        f"- train listwise groups: `{result['metrics']['train_listwise_groups']}`",
        f"- validation groups: `{m['groups']}`",
        f"- validation oracle-change groups: `{m['oracle_change_groups']}`",
        f"- feature count: `{result['metrics']['feature_count']}`",
        "",
        "## Oracle-Change-Only Policy",
        "",
        "| changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {m['changed_groups']} | {m['exact']} | {m['false_change']} | {m['wrong_change']} | {m['missed_oracle']} | {m['tesseract_delta_vs_nearest']} | {m['parseq_delta_vs_nearest']} |",
        "",
        "## Rank Audit",
        "",
        "| metric | <=1 | <=2 | <=4 | <=8 |",
        "|---|---:|---:|---:|---:|",
    ]
    score_counts = m["oracle_nonnearest_score_rank_counts"]
    topk_counts = m["oracle_topk_rank_counts"]
    lines.append(f"| learned score rank | {score_counts['le1']} | {score_counts['le2']} | {score_counts['le4']} | {score_counts['le8']} |")
    lines.append(f"| first-stage topk rank | {topk_counts['le1']} | {topk_counts['le2']} | {topk_counts['le4']} | {topk_counts['le8']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If score rank1 is still low, broader train-split scalar supervision is not enough.",
            "- Promotion requires a separate no-op acceptor and counted `.oscr` evaluation; neither is attempted here.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_trainval_tabular_listwise_ranker")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--model-seeds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.table)
    groups = group_rows(rows)
    feature_names = numeric_feature_names(rows)
    features = build_features(rows, feature_names)
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if str(row.get("assignment_partition", row.get("split"))) == "train"],
        dtype=np.int64,
    )
    train_nonnearest = np.asarray(
        [idx for idx in train_indices if int(rows[idx]["code_index"]) != int(rows[idx]["nearest_code"])],
        dtype=np.int64,
    )
    train_groups = build_listwise_groups(rows, groups, "train")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    features_norm = normalize(features[train_nonnearest], features)
    x_all = torch.from_numpy(features_norm).to(device)
    all_scores = []
    model_metas = []
    for offset in range(args.model_seeds):
        scores, meta = train_model(x_all, train_groups, args.seed + offset, args, device)
        all_scores.append(scores)
        model_metas.append(meta)
    scores = np.mean(np.stack(all_scores, axis=0), axis=0)
    audits, metrics = evaluate_val(rows, groups, scores)
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_trainval_tabular_ranker_not_promoted_selector",
        "inputs": {
            "table": str(args.table),
            "table_sha256": sha256_file(args.table),
        },
        "config": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pairwise_weight": args.pairwise_weight,
            "model_seeds": args.model_seeds,
            "seed": args.seed,
            "feature_names": feature_names,
            "feature_rule": "topk_rank, assignment_relative_error, codebook_* and img_* numeric columns; OCR/label/oracle columns excluded",
        },
        "models": model_metas,
        "metrics": {
            "feature_count": int(len(feature_names)),
            "train_listwise_groups": int(len(train_groups)),
            "oracle_change_only_policy": metrics,
        },
        "group_audits": audits,
        "aggregate": {
            "scalar_metrics": {
                "val_oracle_change_groups": {"value": float(metrics["oracle_change_groups"])},
                "val_exact": {"value": float(metrics["exact"])},
                "val_wrong_change": {"value": float(metrics["wrong_change"])},
                "val_tesseract_delta_vs_nearest": {"value": float(metrics["tesseract_delta_vs_nearest"])},
                "val_oracle_nonnearest_score_rank_le1": {"value": float(metrics["oracle_nonnearest_score_rank_counts"]["le1"])},
                "val_oracle_nonnearest_score_rank_le4": {"value": float(metrics["oracle_nonnearest_score_rank_counts"]["le4"])},
            }
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
