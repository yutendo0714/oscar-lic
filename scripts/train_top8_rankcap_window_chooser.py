#!/usr/bin/env python3
"""Train a rankcap candidate chooser over local latent-window evidence.

This is the window/CNN counterpart to the tabular rankcap chooser. It keeps the
first-stage candidate set clean (`topk_rank <= K`) and asks whether local
MLIC++ latent code-effect windows improve candidate choice inside that set.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


class WindowChooser(torch.nn.Module):
    def __init__(self, in_channels: int, tab_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.window = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
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
            torch.nn.Linear(64 + 32, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, window: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.window(window), self.tab(tab)], dim=1)).reshape(-1)


def build_feature_arrays(data: dict[str, np.ndarray], train_group_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Candidate-local code-effect context. Teacher/OCR outputs are intentionally absent.
    window = np.concatenate(
        [
            data["latent_window_yhat"].astype(np.float32),
            data["latent_window_target"].astype(np.float32),
            data["latent_window_residual"].astype(np.float32),
            (data["latent_window_target"].astype(np.float32) - data["latent_window_yhat"].astype(np.float32)),
        ],
        axis=2,
    )
    tab = np.concatenate(
        [
            data["core_features"].astype(np.float32),
            stats(data["code_center"].astype(np.float32)),
            stats(data["nearest_center"].astype(np.float32)),
            stats(data["delta_center"].astype(np.float32)),
            stats(data["latent_yhat"].astype(np.float32)),
            stats(data["latent_target"].astype(np.float32)),
            stats(data["latent_residual"].astype(np.float32)),
        ],
        axis=2,
    )
    train_rows = train_group_mask[:, None] & data["mask"].astype(bool)
    w_train = window[train_rows]
    mean = w_train.mean(axis=(0, 2, 3), keepdims=True)
    std = w_train.std(axis=(0, 2, 3), keepdims=True)
    std[std < 1.0e-6] = 1.0
    window = ((window - mean.reshape(1, 1, -1, 1, 1)) / std.reshape(1, 1, -1, 1, 1)).astype(np.float32)
    t_train = tab[train_rows]
    t_mean = t_train.mean(axis=0, keepdims=True)
    t_std = t_train.std(axis=0, keepdims=True)
    t_std[t_std < 1.0e-6] = 1.0
    tab = ((tab - t_mean.reshape(1, 1, -1)) / t_std.reshape(1, 1, -1)).astype(np.float32)
    return window, tab


def stats(values: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            values.mean(axis=2),
            values.std(axis=2),
            np.mean(np.abs(values), axis=2),
            np.sqrt(np.mean(np.square(values), axis=2)),
            np.max(np.abs(values), axis=2),
        ],
        axis=2,
    ).astype(np.float32)


def group_lookup(groups: dict[tuple[int, int, int, int], list[dict[str, Any]]]) -> dict[tuple[int, int, int, int], list[dict[str, Any]]]:
    return groups


def group_key_from_data(data: dict[str, np.ndarray], group_index: int) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in data["group_keys"][group_index])


def table_group_for(data: dict[str, np.ndarray], groups: dict[tuple[int, int, int, int], list[dict[str, Any]]], group_index: int) -> list[dict[str, Any]]:
    key = group_key_from_data(data, group_index)
    return groups[key]


def candidate_positions(data: dict[str, np.ndarray], group_index: int, cap: int) -> list[int]:
    mask = data["mask"][group_index].astype(bool)
    ranks = data["topk_ranks"][group_index]
    nearest = data["is_nearest"][group_index].astype(bool)
    return [int(pos) for pos in np.flatnonzero(mask & (~nearest) & (ranks <= cap))]


def oracle_position(data: dict[str, np.ndarray], group_index: int) -> int:
    labels = data["labels"][group_index]
    matches = np.flatnonzero(labels > 0.5)
    if len(matches) == 0:
        return int(np.argmax(labels))
    return int(matches[0])


def trainable_groups(
    data: dict[str, np.ndarray],
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_indices: np.ndarray,
    cap: int,
) -> list[dict[str, Any]]:
    out = []
    for group_index in group_indices.tolist():
        table_rows = table_group_for(data, groups, group_index)
        if not bottleneck.oracle_row(table_rows)["code_index"] != bottleneck.nearest_row(table_rows)["code_index"]:
            continue
        oracle_pos = oracle_position(data, group_index)
        positions = candidate_positions(data, group_index, cap)
        if oracle_pos not in positions:
            continue
        out.append({"group_index": int(group_index), "positions": positions, "target_local": positions.index(oracle_pos)})
    return out


def selected_codes(
    data: dict[str, np.ndarray],
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_indices: np.ndarray,
    scores: np.ndarray,
    cap: int,
    *,
    covered_only: bool,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for group_index in group_indices.tolist():
        key = group_key_from_data(data, group_index)
        rows = groups[key]
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        nearest_code = int(nearest["code_index"])
        selected[key] = nearest_code
        if int(oracle["code_index"]) == nearest_code:
            continue
        if covered_only and int(oracle.get("topk_rank", 999)) > cap:
            continue
        positions = candidate_positions(data, group_index, cap)
        if not positions:
            continue
        best = max(positions, key=lambda pos: (float(scores[group_index, pos]), -int(data["topk_ranks"][group_index, pos])))
        selected[key] = int(data["code_indices"][group_index, best])
    return selected


def rank_counts(
    data: dict[str, np.ndarray],
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    group_indices: np.ndarray,
    scores: np.ndarray,
    cap: int,
) -> dict[str, Any]:
    counts = Counter()
    audits = []
    for group_index in group_indices.tolist():
        key = group_key_from_data(data, group_index)
        rows = groups[key]
        nearest = bottleneck.nearest_row(rows)
        oracle = bottleneck.oracle_row(rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        counts["oracle_change_groups"] += 1
        oracle_pos = oracle_position(data, group_index)
        oracle_rank = int(data["topk_ranks"][group_index, oracle_pos])
        if oracle_rank > cap:
            counts["oracle_not_in_cap"] += 1
            continue
        positions = candidate_positions(data, group_index, cap)
        ranked = sorted(positions, key=lambda pos: (float(scores[group_index, pos]), -int(data["topk_ranks"][group_index, pos])), reverse=True)
        rank = 1 + ranked.index(oracle_pos)
        for limit in [1, 2, 4]:
            counts[f"rank_le{limit}"] += int(rank <= limit)
        if rank > 1 and len(audits) < 25:
            top = ranked[0]
            audits.append(
                {
                    "group_key": list(key),
                    "source": rows[0].get("source"),
                    "reference": rows[0].get("reference"),
                    "oracle_code": int(data["code_indices"][group_index, oracle_pos]),
                    "oracle_topk_rank": oracle_rank,
                    "score_rank": int(rank),
                    "top_code": int(data["code_indices"][group_index, top]),
                    "top_topk_rank": int(data["topk_ranks"][group_index, top]),
                }
            )
    return {**{key: int(value) for key, value in counts.items()}, "audits": audits}


def train_seed(
    data: dict[str, np.ndarray],
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    partition: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_mask = partition == "train"
    val_mask = partition == "val"
    window_np, tab_np = build_feature_arrays(data, train_mask)
    train_group_indices = np.flatnonzero(train_mask)
    val_group_indices = np.flatnonzero(val_mask)
    train_groups = trainable_groups(data, groups, train_group_indices, args.cap)
    window = torch.from_numpy(window_np.reshape(-1, *window_np.shape[2:])).to(device)
    tab = torch.from_numpy(tab_np.reshape(-1, tab_np.shape[2])).to(device)
    model = WindowChooser(window.shape[1], tab.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    candidate_count = data["mask"].shape[1]
    last_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(window, tab).reshape(data["mask"].shape)
        losses = []
        for group in train_groups:
            idx = torch.tensor(group["positions"], dtype=torch.long, device=device)
            target = torch.tensor([group["target_local"]], dtype=torch.long, device=device)
            group_logits = logits[group["group_index"], idx].reshape(1, -1)
            losses.append(F.cross_entropy(group_logits, target))
            if args.pairwise_weight > 0:
                target_logit = group_logits[0, group["target_local"]]
                negatives = torch.cat([group_logits[0, : group["target_local"]], group_logits[0, group["target_local"] + 1 :]])
                if negatives.numel():
                    losses.append(args.pairwise_weight * F.softplus(-(target_logit - negatives)).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        scores = model(window, tab).reshape(data["mask"].shape).detach().cpu().numpy()
    train_selected = selected_codes(data, groups, train_group_indices, scores, args.cap, covered_only=True)
    val_selected = selected_codes(data, groups, val_group_indices, scores, args.cap, covered_only=True)
    val_all_selected = selected_codes(data, groups, val_group_indices, scores, args.cap, covered_only=False)
    train_subset = {group_key_from_data(data, idx): groups[group_key_from_data(data, idx)] for idx in train_group_indices.tolist()}
    val_subset = {group_key_from_data(data, idx): groups[group_key_from_data(data, idx)] for idx in val_group_indices.tolist()}
    return {
        "seed": int(seed),
        "device": str(device),
        "final_loss": float(last_loss),
        "train_groups": int(len(train_groups)),
        "train_rank_counts": rank_counts(data, groups, train_group_indices, scores, args.cap),
        "val_rank_counts": rank_counts(data, groups, val_group_indices, scores, args.cap),
        "train_policy_covered_only": bottleneck.evaluate_selection(train_subset, train_selected),
        "val_policy_covered_only": bottleneck.evaluate_selection(val_subset, val_selected),
        "val_policy_all_oracle_changes": bottleneck.evaluate_selection(val_subset, val_all_selected),
    }


def partition_array(data: dict[str, np.ndarray], groups: dict[tuple[int, int, int, int], list[dict[str, Any]]]) -> np.ndarray:
    parts = []
    for group_index in range(data["group_keys"].shape[0]):
        rows = groups[group_key_from_data(data, group_index)]
        parts.append(str(rows[0].get("assignment_partition", rows[0].get("split", ""))))
    return np.asarray(parts, dtype=object)


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
        "Rankcap local latent-window candidate chooser. Diagnostic only; no `.oscr` selector is promoted.",
        "",
        "| seed | rank1 | rank<=2 | rank<=4 | covered-only policy | all-change policy |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in result["runs"]:
        ranks = row["val_rank_counts"]
        lines.append(
            f"| {row['seed']} | {ranks.get('rank_le1', 0)} | {ranks.get('rank_le2', 0)} | "
            f"{ranks.get('rank_le4', 0)} | {compact_metrics(row['val_policy_covered_only'])} | "
            f"{compact_metrics(row['val_policy_all_oracle_changes'])} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    lines = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-rankcap-window-chooser",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
        "  features:",
        f"    path: {result['inputs']['features']['path']}",
        f"    sha256: {result['inputs']['features']['sha256']}",
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
        "    path: scripts/train_top8_rankcap_window_chooser.py",
        f"    sha256: {sha256_file(Path(__file__))}",
        "  bottleneck_audit:",
        "    path: scripts/analyze_top8_two_stage_bottleneck.py",
        f"    sha256: {sha256_file(SCRIPT_DIR / 'analyze_top8_two_stage_bottleneck.py')}",
        "conclusion: diagnostic rankcap window chooser; no selector promoted",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    parser.add_argument("--model-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_npz(args.features)
    groups, _ = bottleneck.load_table(args.table)
    partition = partition_array(data, groups)
    runs = [train_seed(data, groups, partition, args, args.seed + i) for i in range(args.model_seeds)]
    best = max(
        runs,
        key=lambda row: (
            row["val_rank_counts"].get("rank_le1", 0),
            -row["val_policy_covered_only"].get("wrong_change", 0),
            -row["val_policy_covered_only"].get("missed_oracle", 0),
        ),
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_rankcap_window_chooser_not_promoted",
        "code_commit": bottleneck.git_commit(),
        "inputs": {
            "features": {"path": str(args.features), "sha256": sha256_file(args.features)},
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
        },
        "config": {
            "cap": args.cap,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pairwise_weight": args.pairwise_weight,
            "model_seeds": args.model_seeds,
            "seed": args.seed,
        },
        "data_summary": {
            "groups": int(data["group_keys"].shape[0]),
            "train_groups": int(np.sum(partition == "train")),
            "val_groups": int(np.sum(partition == "val")),
            "candidate_slots": int(data["mask"].shape[1]),
            "window_channels": 128,
        },
        "runs": runs,
        "best_run": best,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "bottleneck_script": sha256_file(SCRIPT_DIR / "analyze_top8_two_stage_bottleneck.py"),
        },
        "interpretation": (
            "This is the first clean rankcap4 chooser that uses local latent-window code-effect evidence. "
            "It should be compared with N115's shallow tabular chooser and N114's rankcap upper bound. "
            "No actual `.oscr` promotion is justified unless wrong/missed counts fall near the current "
            "precision floor and table-level Tesseract beats `-8` with PARSeq unchanged."
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
                "best_seed": best["seed"],
                "best_val_rank_counts": best["val_rank_counts"],
                "best_val_policy": {
                    key: value
                    for key, value in best["val_policy_covered_only"].items()
                    if key in ["exact_changed_groups", "wrong_change", "missed_oracle", "tesseract_delta_vs_nearest", "parseq_delta_vs_nearest"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
