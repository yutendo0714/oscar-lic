#!/usr/bin/env python3
"""Train a top-k-prior candidate verifier over local latent/code tensors.

This is a two-stage diagnostic:
1. Candidate is fixed to the source-robust top non-nearest prior.
2. A small verifier decides accept/no-op from candidate-local latent/code evidence.

No counted `.oscr` streams are exported here.
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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_top8_trainval_tabular_listwise_ranker as table_ranker  # noqa: E402


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def table_partitions(table_rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], str]:
    out = {}
    for row in table_rows:
        key = (int(row["real_seed"]), int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))
        out[key] = str(row.get("assignment_partition", row.get("split", "")))
    return out


def top_prior_positions(data: dict[str, np.ndarray]) -> np.ndarray:
    mask = data["mask"].astype(bool)
    nearest = data["is_nearest"].astype(bool)
    ranks = data["topk_ranks"].astype(np.int64)
    selected = []
    for group_index in range(mask.shape[0]):
        valid = mask[group_index] & (~nearest[group_index])
        if not valid.any():
            selected.append(int(np.flatnonzero(mask[group_index])[0]))
            continue
        values = ranks[group_index].copy()
        values[~valid] = np.iinfo(np.int64).max
        selected.append(int(np.argmin(values)))
    return np.asarray(selected, dtype=np.int64)


def build_group_arrays(
    data: dict[str, np.ndarray],
    table_rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    partitions = table_partitions(table_rows)
    selected = top_prior_positions(data)
    group_count = data["mask"].shape[0]
    batch = np.arange(group_count)
    nearest_pos = data["is_nearest"].astype(bool).argmax(axis=1)
    oracle_pos = data["labels"].argmax(axis=1)
    oracle_changed = oracle_pos != nearest_pos
    selected_is_oracle = selected == oracle_pos
    labels_exact = (selected_is_oracle & oracle_changed).astype(np.float32)
    labels_safe = (
        (data["tesseract_delta"][batch, selected] < 0) & (data["parseq_delta"][batch, selected] <= 0)
    ).astype(np.float32)
    partition = []
    records = []
    for group_index, key_values in enumerate(data["group_keys"]):
        real_seed, _seed, source_index, candidate_index = [int(value) for value in key_values]
        key = (real_seed, int(_seed), source_index, candidate_index)
        part = partitions.get(key, "unknown")
        partition.append(part)
        records.append(
            {
                "group_index": group_index,
                "key": {
                    "real_seed": real_seed,
                    "source_index": source_index,
                    "candidate_index": candidate_index,
                    "partition": part,
                },
                "nearest_code": int(data["code_indices"][group_index, nearest_pos[group_index]]),
                "oracle_code": int(data["code_indices"][group_index, oracle_pos[group_index]]),
                "top_prior_code": int(data["code_indices"][group_index, selected[group_index]]),
                "top_prior_pos": int(selected[group_index]),
                "top_prior_topk_rank": int(data["topk_ranks"][group_index, selected[group_index]]),
                "label_exact_changed": int(labels_exact[group_index]),
                "label_safe_improve": int(labels_safe[group_index]),
                "top_prior_tesseract_delta_vs_nearest": int(data["tesseract_delta"][group_index, selected[group_index]]),
                "top_prior_parseq_delta_vs_nearest": int(data["parseq_delta"][group_index, selected[group_index]]),
            }
        )
    core = data["core_features"][batch, selected].astype(np.float32)
    code = np.concatenate(
        [
            data["code_center"][batch, selected].astype(np.float32),
            data["nearest_center"][batch, selected].astype(np.float32),
            data["delta_center"][batch, selected].astype(np.float32),
            data["latent_yhat"][batch, selected].astype(np.float32),
            data["latent_target"][batch, selected].astype(np.float32),
            data["latent_residual"][batch, selected].astype(np.float32),
        ],
        axis=1,
    )
    window = np.concatenate(
        [
            data["latent_window_yhat"][batch, selected].astype(np.float32),
            data["latent_window_target"][batch, selected].astype(np.float32),
            data["latent_window_residual"][batch, selected].astype(np.float32),
        ],
        axis=1,
    )
    arrays = {
        "partition": np.asarray(partition),
        "selected_pos": selected,
        "nearest_pos": nearest_pos,
        "oracle_pos": oracle_pos,
        "labels_exact": labels_exact,
        "labels_safe": labels_safe,
        "core": core,
        "code": code,
        "window": window,
        "tesseract_delta": data["tesseract_delta"][batch, selected].astype(np.int16),
        "parseq_delta": data["parseq_delta"][batch, selected].astype(np.int16),
        "selected_code": data["code_indices"][batch, selected].astype(np.int64),
        "nearest_code": data["code_indices"][batch, nearest_pos].astype(np.int64),
        "oracle_code": data["code_indices"][batch, oracle_pos].astype(np.int64),
    }
    return arrays, records


def normalize(arrays: dict[str, np.ndarray], train_mask: np.ndarray) -> dict[str, np.ndarray]:
    out = dict(arrays)
    for name in ["core", "code"]:
        values = arrays[name][train_mask]
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        out[name] = ((arrays[name] - mean) / std).astype(np.float32)
    values = arrays["window"][train_mask]
    mean = values.mean(axis=(0, 2, 3), keepdims=True)
    std = values.std(axis=(0, 2, 3), keepdims=True)
    std[std < 1e-6] = 1.0
    out["window"] = ((arrays["window"] - mean) / std).astype(np.float32)
    return out


class TopPriorVerifier(torch.nn.Module):
    def __init__(self, core_dim: int, code_dim: int, window_channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.core = torch.nn.Sequential(
            torch.nn.Linear(core_dim, 24),
            torch.nn.GELU(),
            torch.nn.LayerNorm(24),
        )
        self.code = torch.nn.Sequential(
            torch.nn.Linear(code_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim // 2),
        )
        self.window = torch.nn.Sequential(
            torch.nn.Conv2d(window_channels, 48, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(48, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, hidden_dim // 2, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.LayerNorm(hidden_dim // 2),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(24 + hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, core: torch.Tensor, code: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.core(core), self.code(code), self.window(window)], dim=1)).reshape(-1)


def policy_metrics(records: list[dict[str, Any]], accept: np.ndarray, mask: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    counts = Counter()
    t_sum = 0
    p_sum = 0
    for index, row in enumerate(records):
        if not bool(mask[index]):
            continue
        nearest_code = int(arrays["nearest_code"][index])
        oracle_code = int(arrays["oracle_code"][index])
        selected_code = int(arrays["selected_code"][index]) if bool(accept[index]) else nearest_code
        status = table_ranker.classify(selected_code, nearest_code, oracle_code)
        if bool(accept[index]):
            t_delta = int(arrays["tesseract_delta"][index])
            p_delta = int(arrays["parseq_delta"][index])
        else:
            t_delta = 0
            p_delta = 0
        counts[status] += 1
        counts["oracle_change_groups"] += int(oracle_code != nearest_code)
        counts["changed_groups"] += int(selected_code != nearest_code)
        counts["exact_changed_groups"] += int(status == "exact" and oracle_code != nearest_code)
        counts["parseq_worsen_groups"] += int(p_delta > 0)
        counts["tesseract_worsen_groups"] += int(t_delta > 0)
        counts["tesseract_improve_groups"] += int(t_delta < 0)
        t_sum += t_delta
        p_sum += p_delta
    return {
        "groups": int(mask.sum()),
        "oracle_change_groups": int(counts["oracle_change_groups"]),
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "exact_changed_groups": int(counts["exact_changed_groups"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "parseq_worsen_groups": int(counts["parseq_worsen_groups"]),
        "tesseract_worsen_groups": int(counts["tesseract_worsen_groups"]),
        "tesseract_improve_groups": int(counts["tesseract_improve_groups"]),
        "tesseract_delta_vs_nearest": int(t_sum),
        "parseq_delta_vs_nearest": int(p_sum),
    }


def choose_threshold(scores: np.ndarray, records: list[dict[str, Any]], train_mask: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    values = sorted(set(float(score) for score in scores[train_mask]))
    thresholds = [max(values) + 1e-6] + values + [min(values) - 1e-6] if values else [1.0]
    rows = []
    for threshold in thresholds:
        accept = scores >= threshold
        metrics = policy_metrics(records, accept, train_mask, arrays)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    candidates = [row for row in rows if row["false_change"] + row["wrong_change"] <= 1 and row["parseq_worsen_groups"] == 0]
    if not candidates:
        candidates = [row for row in rows if row["parseq_worsen_groups"] == 0]
    if not candidates:
        candidates = rows
    return min(
        candidates,
        key=lambda row: (
            row["tesseract_delta_vs_nearest"],
            row["tesseract_worsen_groups"],
            row["false_change"] + row["wrong_change"],
            -row["exact_changed_groups"],
            row["changed_groups"],
        ),
    )


def train_seed(
    arrays_raw: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    model_seed: int,
    target_name: str,
) -> dict[str, Any]:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_mask = arrays_raw["partition"] == "train"
    val_mask = arrays_raw["partition"] == "val"
    arrays = normalize(arrays_raw, train_mask)
    labels = arrays_raw["labels_exact"] if target_name == "exact_changed" else arrays_raw["labels_safe"]
    x_core = torch.from_numpy(arrays["core"]).to(device)
    x_code = torch.from_numpy(arrays["code"]).to(device)
    x_window = torch.from_numpy(arrays["window"]).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    train_idx = torch.from_numpy(np.flatnonzero(train_mask)).long().to(device)
    model = TopPriorVerifier(
        core_dim=arrays["core"].shape[1],
        code_dim=arrays["code"].shape[1],
        window_channels=arrays["window"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    positives = float(labels[train_mask].sum())
    negatives = float(train_mask.sum() - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_core[train_idx], x_code[train_idx], x_window[train_idx])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[train_idx], pos_weight=pos_weight)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x_core, x_code, x_window)).detach().cpu().numpy()
    threshold_row = choose_threshold(scores, records, train_mask, arrays_raw)
    accept = scores >= float(threshold_row["threshold"])
    train_metrics = policy_metrics(records, accept, train_mask, arrays_raw)
    val_metrics = policy_metrics(records, accept, val_mask, arrays_raw)
    return {
        "target": target_name,
        "model_seed": int(model_seed),
        "device": str(device),
        "final_loss": final_loss,
        "train_positive_groups": int(labels[train_mask].sum()),
        "val_positive_groups": int(labels[val_mask].sum()),
        "threshold": float(threshold_row["threshold"]),
        "train": train_metrics,
        "val": val_metrics,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Top-k-prior candidate verifier over local latent/code tensors.",
        "This is a table-level diagnostic and does not export counted `.oscr` streams.",
        "",
        "## Results",
        "",
        "| target | seed | train positives | val positives | train changed | train exact | train false+wrong | train T | val changed | val exact | val false+wrong | val T | val P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        train = row["train"]
        val = row["val"]
        lines.append(
            f"| `{row['target']}` | {row['model_seed']} | {row['train_positive_groups']} | {row['val_positive_groups']} | "
            f"{train['changed_groups']} | {train['exact_changed_groups']} | {train['false_change'] + train['wrong_change']} | {train['tesseract_delta_vs_nearest']} | "
            f"{val['changed_groups']} | {val['exact_changed_groups']} | {val['false_change'] + val['wrong_change']} | {val['tesseract_delta_vs_nearest']} | {val['parseq_delta_vs_nearest']} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_topk_prior_latent_verifier")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--model-seed", type=int, action="append", default=[])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    seeds = args.model_seed or [0, 1, 2]
    data = load_npz(args.features)
    table_rows = read_jsonl(args.table)
    arrays, records = build_group_arrays(data, table_rows)
    runs = []
    for target in ["exact_changed", "safe_improve"]:
        for seed in seeds:
            runs.append(train_seed(arrays, records, args, int(seed), target))
    best = min(
        runs,
        key=lambda row: (
            row["val"]["tesseract_delta_vs_nearest"],
            row["val"]["false_change"] + row["val"]["wrong_change"],
            -row["val"]["exact_changed_groups"],
        ),
    )
    interpretation = (
        "This tests whether materially richer local latent/code evidence can turn the N097 top-k prior into a safer verifier. "
        "Promotion still requires beating the current actual-bitstream -8 Tesseract floor with low false/wrong changes and PARSeq no worse."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_topk_prior_latent_verifier_not_promoted_selector",
        "inputs": {
            "features": {"path": str(args.features), "sha256": sha256_file(args.features)},
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
        },
        "config": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "model_seeds": seeds,
            "model": "top prior fixed; core MLP + code MLP + latent-window CNN verifier",
        },
        "runs": runs,
        "best_run": best,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "base_ranker_script": sha256_file(SCRIPT_DIR / "train_top8_trainval_tabular_listwise_ranker.py"),
        },
        "aggregate": {
            "scalar_metrics": {
                "best_val_tesseract_delta": {"value": float(best["val"]["tesseract_delta_vs_nearest"])},
                "best_val_parseq_delta": {"value": float(best["val"]["parseq_delta_vs_nearest"])},
                "best_val_exact_changed": {"value": float(best["val"]["exact_changed_groups"])},
                "best_val_false_wrong": {"value": float(best["val"]["false_change"] + best["val"]["wrong_change"])},
            }
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "best_run": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
