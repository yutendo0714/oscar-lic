#!/usr/bin/env python3
"""Train a tiny non-nearest ranker on the top-8 evidence cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key3_from_record(row: dict[str, Any]) -> tuple[int, int, int]:
    key = row["key"]
    return (int(key["real_seed"]), int(key["source_index"]), int(key["candidate_index"]))


class EvidenceRanker(torch.nn.Module):
    def __init__(self, in_channels: int, tab_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.patch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        self.tab = torch.nn.Sequential(
            torch.nn.Linear(tab_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 32),
            torch.nn.GELU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(48 + 32, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, patches: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.patch(patches), self.tab(tabular)], dim=1)).reshape(-1)


def classify(selected_code: int, nearest_code: int, oracle_code: int) -> str:
    if selected_code == oracle_code:
        return "exact"
    if oracle_code == nearest_code and selected_code != nearest_code:
        return "false_change"
    if oracle_code != nearest_code and selected_code == nearest_code:
        return "missed_oracle"
    return "wrong_change"


def rank_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {
        "le1": sum(1 for row in rows if int(row[field]) <= 1),
        "le2": sum(1 for row in rows if int(row[field]) <= 2),
        "le4": sum(1 for row in rows if int(row[field]) <= 4),
        "le8": sum(1 for row in rows if int(row[field]) <= 8),
    }


def normalize_tabular(train: np.ndarray, all_features: np.ndarray) -> np.ndarray:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((all_features - mean) / std).astype(np.float32)


def group_indices(records: list[dict[str, Any]], indices: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in indices.tolist():
        groups[key3_from_record(records[idx])].append(idx)
    for group in groups.values():
        group.sort(key=lambda idx: int(records[idx]["topk_rank"]))
    return groups


def audit_groups(
    records: list[dict[str, Any]],
    all_groups: dict[tuple[int, int, int], list[int]],
    nonnearest_groups: dict[tuple[int, int, int], list[int]],
    scores: np.ndarray,
    deltas: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audits = []
    counts = Counter()
    tesseract_delta = 0
    parseq_delta = 0
    for key, all_indices in sorted(all_groups.items()):
        first = records[all_indices[0]]
        nearest_code = int(first["nearest_code"])
        oracle_code = int(first["oracle_code"])
        nonnearest = nonnearest_groups.get(key, [])
        best_idx = max(nonnearest, key=lambda idx: (float(scores[idx]), -int(records[idx]["topk_rank"]))) if nonnearest else None
        selected_idx = None
        selected_code = nearest_code
        if oracle_code != nearest_code and best_idx is not None:
            selected_idx = best_idx
            selected_code = int(records[best_idx]["code_index"])
        status = classify(selected_code, nearest_code, oracle_code)
        counts[status] += 1
        counts["changed_groups"] += int(selected_code != nearest_code)
        if selected_idx is not None:
            t_delta = int(deltas[selected_idx, 0])
            p_delta = int(deltas[selected_idx, 1])
        else:
            t_delta = 0
            p_delta = 0
        tesseract_delta += t_delta
        parseq_delta += p_delta
        by_score = sorted(nonnearest, key=lambda idx: (float(scores[idx]), -int(records[idx]["topk_rank"])), reverse=True)
        oracle_rank = None
        if oracle_code != nearest_code:
            oracle_rank = 1 + next(rank for rank, idx in enumerate(by_score) if int(records[idx]["code_index"]) == oracle_code)
        audits.append(
            {
                "key": first["key"],
                "source": first["source"],
                "reference": first["reference"],
                "next_model_target": first["next_model_target"],
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "selected_code_oracle_change_only": selected_code,
                "selected_status_oracle_change_only": status,
                "oracle_nonnearest_score_rank": oracle_rank,
                "oracle_topk_rank": int(next(records[idx]["topk_rank"] for idx in all_indices if int(records[idx]["code_index"]) == oracle_code)),
                "best_nonnearest_code": int(records[best_idx]["code_index"]) if best_idx is not None else nearest_code,
                "best_nonnearest_score": float(scores[best_idx]) if best_idx is not None else 0.0,
                "best_nonnearest_tesseract_delta_vs_nearest": int(deltas[best_idx, 0]) if best_idx is not None else 0,
                "best_nonnearest_parseq_delta_vs_nearest": int(deltas[best_idx, 1]) if best_idx is not None else 0,
            }
        )
    metrics = {
        "groups": len(all_groups),
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "tesseract_delta_vs_nearest": int(tesseract_delta),
        "parseq_delta_vs_nearest": int(parseq_delta),
    }
    return audits, metrics


def aggregate_audit(group_audits: list[dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_audits:
        by_target[str(row["next_model_target"])].append(row)
    out: dict[str, Any] = {}
    for target, rows in sorted(by_target.items()):
        rank_rows = [row for row in rows if row["oracle_nonnearest_score_rank"] is not None]
        out[target] = {
            "groups": len(rows),
            "selected_status_counts": dict(Counter(str(row["selected_status_oracle_change_only"]) for row in rows)),
            "oracle_nonnearest_score_rank_counts": rank_counts(rank_rows, "oracle_nonnearest_score_rank") if rank_rows else {},
            "oracle_topk_rank_counts": rank_counts(rows, "oracle_topk_rank"),
            "best_nonnearest_tesseract_delta_sum": int(sum(int(row["best_nonnearest_tesseract_delta_vs_nearest"]) for row in rows)),
        }
    return out


def train_fold(
    cache: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    val_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(args.seed + val_seed)
    np.random.seed(args.seed + val_seed)
    torch.manual_seed(args.seed + val_seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.manual_seed_all(args.seed + val_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    keys = cache["keys"]
    patches_np = cache["patches"].astype(np.float32)
    tabular_np = np.concatenate([cache["code_features"].astype(np.float32), cache["glyph_features"].astype(np.float32)], axis=1)
    deltas = cache["deltas_tesseract_parseq"]
    is_nonnearest = np.asarray(
        [int(records[index]["code_index"]) != int(records[index]["nearest_code"]) for index in range(len(records))],
        dtype=bool,
    )
    labels = np.asarray([int(records[index]["is_oracle_code"]) for index in range(len(records))], dtype=np.float32)
    train_mask = (keys[:, 0] != val_seed) & is_nonnearest
    val_mask = (keys[:, 0] == val_seed) & is_nonnearest
    train_indices = np.nonzero(train_mask)[0]
    val_seed_indices = np.nonzero(keys[:, 0] == val_seed)[0]
    val_indices = np.nonzero(val_mask)[0]
    tabular_norm = normalize_tabular(tabular_np[train_indices], tabular_np)
    model = EvidenceRanker(
        in_channels=patches_np.shape[1],
        tab_dim=tabular_norm.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    patches = torch.from_numpy(patches_np).to(device)
    tabular = torch.from_numpy(tabular_norm).to(device)
    y_train = torch.from_numpy(labels[train_indices]).to(device)
    train_tensor = torch.from_numpy(train_indices).long().to(device)
    positives = float(y_train.sum().item())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(patches[train_tensor], tabular[train_tensor])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(patches, tabular)).detach().cpu().numpy()
    all_groups = group_indices(records, val_seed_indices)
    nonnearest_groups = group_indices(records, val_indices)
    audits, metrics = audit_groups(records, all_groups, nonnearest_groups, scores, deltas)
    return {
        "val_seed": int(val_seed),
        "device": str(device),
        "train_nonnearest_rows": int(len(train_indices)),
        "val_nonnearest_rows": int(len(val_indices)),
        "train_positive_rows": int(labels[train_indices].sum()),
        "val_positive_rows": int(labels[val_indices].sum()),
        "oracle_change_only_policy": metrics,
        "group_audits": audits,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Tiny patch/code-effect non-nearest ranker over the N088 evidence cache.",
        "This isolates candidate ranking from no-op acceptance and is not deployable by itself.",
        "",
        "## Oracle-Change-Only Policy",
        "",
        "| val seed | changed | exact | false | wrong | missed | Tesseract delta | PARSeq delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        m = row["oracle_change_only_policy"]
        lines.append(
            f"| {row['val_seed']} | {m['changed_groups']} | {m['exact']} | {m['false_change']} | "
            f"{m['wrong_change']} | {m['missed_oracle']} | {m['tesseract_delta_vs_nearest']} | "
            f"{m['parseq_delta_vs_nearest']} |"
        )
    lines.extend(["", "## Target Rank Audit", ""])
    lines.append("| target | groups | selected status | oracle nonnearest score <=1/2/4/8 | oracle topk <=1/2/4/8 | best-nonnearest Tesseract delta sum |")
    lines.append("|---|---:|---|---|---|---:|")
    for target, row in result["aggregate"]["target_rank_audit"].items():
        status = ", ".join(f"{k}:{v}" for k, v in sorted(row["selected_status_counts"].items()))
        score_counts = row["oracle_nonnearest_score_rank_counts"] or {"le1": 0, "le2": 0, "le4": 0, "le8": 0}
        topk_counts = row["oracle_topk_rank_counts"]
        lines.append(
            f"| {target} | {row['groups']} | {status} | "
            f"{score_counts['le1']}/{score_counts['le2']}/{score_counts['le4']}/{score_counts['le8']} | "
            f"{topk_counts['le1']}/{topk_counts['le2']}/{topk_counts['le4']}/{topk_counts['le8']} | "
            f"{row['best_nonnearest_tesseract_delta_sum']} |"
        )
    lines.extend(["", "## Interpretation", "", "- Compare recover rank1 against the N087 glyphcode-only floor of `1/13`.", "- If rank1 remains low, this cache/model is not enough for the next no-op acceptor.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--glyph-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_evidence_cache_nonnearest_ranker")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    glyph = read_json(args.glyph_features)
    records = glyph["candidate_records"]
    with np.load(args.cache, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    per_seed = [train_fold(cache, records, val_seed, args) for val_seed in sorted(set(cache["keys"][:, 0].tolist()))]
    all_audits = [audit for row in per_seed for audit in row["group_audits"]]
    target_audit = aggregate_audit(all_audits)
    recover = target_audit.get("recover_shortlist_oracle_change") or {}
    score_counts = recover.get("oracle_nonnearest_score_rank_counts", {})
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_patch_ranker_not_promoted_selector",
        "inputs": {
            "cache": str(args.cache),
            "cache_sha256": sha256_file(args.cache),
            "glyph_features": str(args.glyph_features),
            "glyph_features_sha256": sha256_file(args.glyph_features),
        },
        "config": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "model": "tiny Conv2d patch branch + tabular MLP; BCE over non-nearest rows only",
        },
        "per_seed": per_seed,
        "aggregate": {
            "target_rank_audit": target_audit,
            "scalar_metrics": {
                "recover_oracle_nonnearest_score_rank_le1": {"value": float(score_counts.get("le1", 0))},
                "recover_oracle_nonnearest_score_rank_le2": {"value": float(score_counts.get("le2", 0))},
                "recover_oracle_nonnearest_score_rank_le4": {"value": float(score_counts.get("le4", 0))},
            },
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
