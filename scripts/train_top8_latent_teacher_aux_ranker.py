#!/usr/bin/env python3
"""Train a top-8 latent/code listwise ranker with teacher-loss auxiliary head.

This is a diagnostic candidate-ranking experiment. CRNN/ABINet teacher outputs
are not used as selector inputs; they regularize an auxiliary head from
deployable latent/code evidence during train only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from oscarlic.text_metrics import character_counts  # noqa: E402
import train_top8_trainval_tabular_listwise_ranker as table_ranker  # noqa: E402


CORE_KEEP = (0, 1, 2, 3)
CORE_KEEP_NAMES = (
    "topk_rank",
    "log1p_clipped_assignment_relative_error",
    "is_nearest",
    "code_equals_nearest",
)


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


def group_key_values(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def row_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (*group_key_values(row), int(row["code_index"]))


def table_group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return group_key_values(row)


def teacher_distance(row: dict[str, Any], profile: str) -> int:
    return character_counts(str(row.get("reference", "")), str(row.get("prediction", "")), profile).distance


def rank_counts(values: list[int]) -> dict[str, int]:
    return {
        "le1": int(sum(value <= 1 for value in values)),
        "le2": int(sum(value <= 2 for value in values)),
        "le4": int(sum(value <= 4 for value in values)),
        "le8": int(sum(value <= 8 for value in values)),
    }


def build_partitions(table_rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], str]:
    partitions = {}
    for row in table_rows:
        partitions[table_group_key(row)] = str(row.get("assignment_partition", row.get("split", "")))
    return partitions


def build_row_lookup(table_rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int, int], dict[str, Any]]:
    return {row_key(row): row for row in table_rows}


def build_teacher_lookup(path: Path) -> dict[tuple[int, int, int, int, int], dict[str, Any]]:
    return {row_key(row): row for row in read_jsonl(path)}


def flat_index(group_index: int, position: int, candidates: int) -> int:
    return group_index * candidates + position


def normalize_feature_array(values: np.ndarray, train_mask: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    train_values = values[train_mask]
    mean = train_values.mean(axis=axes, keepdims=True)
    std = train_values.std(axis=axes, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return ((values - mean) / std).astype(np.float32)


def build_arrays(
    data: dict[str, np.ndarray],
    table_rows: list[dict[str, Any]],
    crnn_rows: dict[tuple[int, int, int, int, int], dict[str, Any]],
    abinet_rows: dict[tuple[int, int, int, int, int], dict[str, Any]],
    profile: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    partitions = build_partitions(table_rows)
    table_lookup = build_row_lookup(table_rows)
    group_count, candidate_count = data["mask"].shape
    partition = []
    group_records = []
    row_records = []
    teacher_loss_utility = np.zeros((group_count, candidate_count), dtype=np.float32)
    teacher_string_utility = np.zeros((group_count, candidate_count), dtype=np.float32)
    missing = Counter()

    for group_index, key_values in enumerate(data["group_keys"]):
        group_key = tuple(int(value) for value in key_values)
        partition.append(partitions.get(group_key, "unknown"))
        nearest_pos = int(np.flatnonzero(data["is_nearest"][group_index].astype(bool))[0])
        nearest_code = int(data["code_indices"][group_index, nearest_pos])
        oracle_pos = int(np.argmax(data["labels"][group_index]))
        oracle_code = int(data["code_indices"][group_index, oracle_pos])
        group_records.append(
            {
                "group_index": group_index,
                "key": {
                    "real_seed": int(group_key[0]),
                    "seed": int(group_key[1]),
                    "source_index": int(group_key[2]),
                    "candidate_index": int(group_key[3]),
                    "partition": partitions.get(group_key, "unknown"),
                },
                "nearest_pos": nearest_pos,
                "oracle_pos": oracle_pos,
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
            }
        )
        nearest_key = (*group_key, nearest_code)
        crnn_nearest = crnn_rows.get(nearest_key)
        abinet_nearest = abinet_rows.get(nearest_key)
        if crnn_nearest is None or abinet_nearest is None:
            missing["nearest_teacher"] += 1
            crnn_nearest = crnn_nearest or {}
            abinet_nearest = abinet_nearest or {}
        crnn_nearest_loss = float(crnn_nearest.get("teacher_loss_mean", 0.0))
        abinet_nearest_loss = float(abinet_nearest.get("teacher_loss_mean", 0.0))
        crnn_nearest_dist = teacher_distance(crnn_nearest, profile) if crnn_nearest else 0
        abinet_nearest_dist = teacher_distance(abinet_nearest, profile) if abinet_nearest else 0
        for pos in range(candidate_count):
            code = int(data["code_indices"][group_index, pos])
            key = (*group_key, code)
            row = table_lookup.get(key)
            crnn = crnn_rows.get(key)
            abinet = abinet_rows.get(key)
            if row is None:
                missing["table_row"] += 1
            if crnn is None:
                missing["crnn_row"] += 1
                crnn = {}
            if abinet is None:
                missing["abinet_row"] += 1
                abinet = {}
            crnn_loss_delta = float(crnn.get("teacher_loss_mean", crnn_nearest_loss)) - crnn_nearest_loss
            abinet_loss_delta = float(abinet.get("teacher_loss_mean", abinet_nearest_loss)) - abinet_nearest_loss
            crnn_dist_delta = (teacher_distance(crnn, profile) if crnn else crnn_nearest_dist) - crnn_nearest_dist
            abinet_dist_delta = (teacher_distance(abinet, profile) if abinet else abinet_nearest_dist) - abinet_nearest_dist
            teacher_loss_utility[group_index, pos] = -float(crnn_loss_delta + abinet_loss_delta)
            teacher_string_utility[group_index, pos] = -float(crnn_dist_delta + abinet_dist_delta)
            row_records.append(
                {
                    "group_index": group_index,
                    "position": pos,
                    "flat_index": flat_index(group_index, pos, candidate_count),
                    "code_index": code,
                    "is_valid": bool(data["mask"][group_index, pos]),
                    "is_nearest": bool(data["is_nearest"][group_index, pos]),
                    "is_oracle": bool(data["labels"][group_index, pos] > 0.5),
                    "teacher_loss_utility": float(teacher_loss_utility[group_index, pos]),
                    "teacher_string_utility": float(teacher_string_utility[group_index, pos]),
                    "tesseract_delta_vs_nearest": int(data["tesseract_delta"][group_index, pos]),
                    "parseq_delta_vs_nearest": int(data["parseq_delta"][group_index, pos]),
                }
            )
    if missing:
        raise RuntimeError(f"missing rows while joining teacher/table data: {dict(missing)}")

    batch = np.arange(group_count)[:, None]
    core = data["core_features"][:, :, list(CORE_KEEP)].astype(np.float32)
    code = np.concatenate(
        [
            data["code_center"].astype(np.float32),
            data["nearest_center"].astype(np.float32),
            data["delta_center"].astype(np.float32),
            data["latent_yhat"].astype(np.float32),
            data["latent_target"].astype(np.float32),
            data["latent_residual"].astype(np.float32),
        ],
        axis=2,
    )
    window = np.concatenate(
        [
            data["latent_window_yhat"].astype(np.float32),
            data["latent_window_target"].astype(np.float32),
            data["latent_window_residual"].astype(np.float32),
        ],
        axis=2,
    )
    arrays = {
        "partition": np.asarray(partition),
        "mask": data["mask"].astype(bool),
        "is_nearest": data["is_nearest"].astype(bool),
        "labels": data["labels"].astype(np.float32),
        "core": core,
        "code": code,
        "window": window,
        "code_indices": data["code_indices"].astype(np.int64),
        "topk_ranks": data["topk_ranks"].astype(np.int64),
        "tesseract_delta": data["tesseract_delta"].astype(np.int16),
        "parseq_delta": data["parseq_delta"].astype(np.int16),
        "teacher_loss_utility": teacher_loss_utility,
        "teacher_string_utility": teacher_string_utility,
    }
    return arrays, group_records, row_records


def train_eval_masks(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = arrays["mask"].reshape(-1)
    nonnearest = (~arrays["is_nearest"]).reshape(-1)
    partition = np.repeat(arrays["partition"], arrays["mask"].shape[1])
    train_mask = valid & nonnearest & (partition == "train")
    val_mask = valid & nonnearest & (partition == "val")
    return valid, train_mask, val_mask


def normalize_arrays(arrays: dict[str, np.ndarray], train_mask_flat: np.ndarray) -> dict[str, np.ndarray]:
    group_count, candidate_count = arrays["mask"].shape
    train_mask = train_mask_flat.reshape(group_count, candidate_count)
    out = dict(arrays)
    out["core"] = normalize_feature_array(arrays["core"], train_mask, axes=(0,))
    out["code"] = normalize_feature_array(arrays["code"], train_mask, axes=(0,))
    out["window"] = normalize_feature_array(arrays["window"], train_mask, axes=(0, 2, 3))
    flat_teacher = arrays["teacher_loss_utility"].reshape(-1)
    mean = float(flat_teacher[train_mask_flat].mean())
    std = float(flat_teacher[train_mask_flat].std())
    if std < 1.0e-6:
        std = 1.0
    out["teacher_loss_target"] = ((arrays["teacher_loss_utility"] - mean) / std).astype(np.float32)
    out["teacher_loss_mean"] = np.asarray([mean], dtype=np.float32)
    out["teacher_loss_std"] = np.asarray([std], dtype=np.float32)
    return out


def listwise_groups(arrays: dict[str, np.ndarray], partition: str) -> list[dict[str, Any]]:
    groups = []
    group_count, candidate_count = arrays["mask"].shape
    for group_index in range(group_count):
        if arrays["partition"][group_index] != partition:
            continue
        nearest_pos = int(np.flatnonzero(arrays["is_nearest"][group_index])[0])
        oracle_pos = int(np.argmax(arrays["labels"][group_index]))
        if oracle_pos == nearest_pos:
            continue
        choices = [
            flat_index(group_index, pos, candidate_count)
            for pos in range(candidate_count)
            if bool(arrays["mask"][group_index, pos]) and not bool(arrays["is_nearest"][group_index, pos])
        ]
        target = flat_index(group_index, oracle_pos, candidate_count)
        if target not in choices or len(choices) < 2:
            continue
        groups.append(
            {
                "group_index": group_index,
                "indices": choices,
                "target_local": choices.index(target),
                "target_index": target,
            }
        )
    return groups


class LatentTeacherAuxRanker(torch.nn.Module):
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
        trunk_dim = 24 + hidden_dim
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(trunk_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.score_head = torch.nn.Linear(hidden_dim, 1)
        self.teacher_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, core: torch.Tensor, code: torch.Tensor, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([self.core(core), self.code(code), self.window(window)], dim=1)
        hidden = self.trunk(features)
        return self.score_head(hidden).reshape(-1), self.teacher_head(hidden).reshape(-1)


def flatten_candidate_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    group_count, candidate_count = arrays["mask"].shape
    return {
        "core": arrays["core"].reshape(group_count * candidate_count, arrays["core"].shape[-1]),
        "code": arrays["code"].reshape(group_count * candidate_count, arrays["code"].shape[-1]),
        "window": arrays["window"].reshape(
            group_count * candidate_count,
            arrays["window"].shape[2],
            arrays["window"].shape[3],
            arrays["window"].shape[4],
        ),
        "teacher_loss_target": arrays["teacher_loss_target"].reshape(-1),
    }


def train_one(
    arrays_raw: dict[str, np.ndarray],
    args: argparse.Namespace,
    model_seed: int,
    teacher_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    _, train_mask, _ = train_eval_masks(arrays_raw)
    arrays = normalize_arrays(arrays_raw, train_mask)
    flat = flatten_candidate_arrays(arrays)
    x_core = torch.from_numpy(flat["core"]).to(device)
    x_code = torch.from_numpy(flat["code"]).to(device)
    x_window = torch.from_numpy(flat["window"]).to(device)
    teacher_target = torch.from_numpy(flat["teacher_loss_target"]).to(device)
    train_indices = torch.from_numpy(np.flatnonzero(train_mask)).long().to(device)
    train_groups = listwise_groups(arrays, "train")
    model = LatentTeacherAuxRanker(
        core_dim=flat["core"].shape[1],
        code_dim=flat["code"].shape[1],
        window_channels=flat["window"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final = {"loss": 0.0, "ce": 0.0, "pairwise": 0.0, "teacher": 0.0}
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        scores, teacher_pred = model(x_core, x_code, x_window)
        ce_losses = []
        pair_losses = []
        for group in train_groups:
            idx = torch.tensor(group["indices"], dtype=torch.long, device=device)
            target = torch.tensor([int(group["target_local"])], dtype=torch.long, device=device)
            group_scores = scores[idx].reshape(1, -1)
            ce_losses.append(F.cross_entropy(group_scores, target))
            if args.pairwise_weight > 0:
                target_score = group_scores[0, int(group["target_local"])]
                neg = torch.cat(
                    [
                        group_scores[0, : int(group["target_local"])],
                        group_scores[0, int(group["target_local"]) + 1 :],
                    ]
                )
                if neg.numel():
                    pair_losses.append(F.softplus(args.margin - target_score + neg).mean())
        ce = torch.stack(ce_losses).mean()
        pair = torch.stack(pair_losses).mean() if pair_losses else torch.zeros((), device=device)
        teacher = F.smooth_l1_loss(teacher_pred[train_indices], teacher_target[train_indices])
        loss = ce + args.pairwise_weight * pair + teacher_weight * teacher
        loss.backward()
        opt.step()
        final = {
            "loss": float(loss.detach().cpu().item()),
            "ce": float(ce.detach().cpu().item()),
            "pairwise": float(pair.detach().cpu().item()),
            "teacher": float(teacher.detach().cpu().item()),
        }
    model.eval()
    with torch.no_grad():
        scores, teacher_pred = model(x_core, x_code, x_window)
    return scores.detach().cpu().numpy(), teacher_pred.detach().cpu().numpy(), {
        "device": str(device),
        "final_losses": final,
        "train_listwise_groups": int(len(train_groups)),
        "teacher_loss_mean": float(arrays["teacher_loss_mean"][0]),
        "teacher_loss_std": float(arrays["teacher_loss_std"][0]),
    }


def evaluate_scores(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    scores: np.ndarray,
    teacher_pred: np.ndarray,
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_count, candidate_count = arrays["mask"].shape
    counts = Counter()
    rank_values = []
    topk_values = []
    t_sum = 0
    p_sum = 0
    audits = []
    for record in records:
        group_index = int(record["group_index"])
        if arrays["partition"][group_index] != partition:
            continue
        nearest_pos = int(record["nearest_pos"])
        oracle_pos = int(record["oracle_pos"])
        nearest_code = int(record["nearest_code"])
        oracle_code = int(record["oracle_code"])
        nonnearest = [
            pos
            for pos in range(candidate_count)
            if bool(arrays["mask"][group_index, pos]) and pos != nearest_pos
        ]
        by_score = sorted(
            nonnearest,
            key=lambda pos: (float(scores[flat_index(group_index, pos, candidate_count)]), -int(arrays["topk_ranks"][group_index, pos])),
            reverse=True,
        )
        best_pos = by_score[0] if by_score else nearest_pos
        selected_pos = best_pos if oracle_code != nearest_code else nearest_pos
        selected_code = int(arrays["code_indices"][group_index, selected_pos])
        status = table_ranker.classify(selected_code, nearest_code, oracle_code)
        counts[status] += 1
        counts["oracle_change_groups"] += int(oracle_code != nearest_code)
        counts["changed_groups"] += int(selected_code != nearest_code)
        counts["exact_changed_groups"] += int(status == "exact" and selected_code != nearest_code)
        if selected_code != nearest_code:
            t_delta = int(arrays["tesseract_delta"][group_index, selected_pos])
            p_delta = int(arrays["parseq_delta"][group_index, selected_pos])
        else:
            t_delta = 0
            p_delta = 0
        t_sum += t_delta
        p_sum += p_delta
        oracle_rank = None
        if oracle_code != nearest_code:
            oracle_rank = 1 + next(rank for rank, pos in enumerate(by_score) if pos == oracle_pos)
            rank_values.append(int(oracle_rank))
            topk_values.append(int(arrays["topk_ranks"][group_index, oracle_pos]))
        audits.append(
            {
                "group_index": group_index,
                "key": record["key"],
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "selected_code_oracle_change_only": selected_code,
                "selected_status_oracle_change_only": status,
                "oracle_nonnearest_score_rank": oracle_rank,
                "oracle_topk_rank": int(arrays["topk_ranks"][group_index, oracle_pos]),
                "best_nonnearest_code": int(arrays["code_indices"][group_index, best_pos]) if by_score else nearest_code,
                "best_nonnearest_score": float(scores[flat_index(group_index, best_pos, candidate_count)]) if by_score else 0.0,
                "best_nonnearest_teacher_pred": float(teacher_pred[flat_index(group_index, best_pos, candidate_count)]) if by_score else 0.0,
                "selected_tesseract_delta_vs_nearest": int(t_delta),
                "selected_parseq_delta_vs_nearest": int(p_delta),
            }
        )
    metrics = {
        "groups": int(sum(1 for record in records if arrays["partition"][int(record["group_index"])] == partition)),
        "oracle_change_groups": int(counts["oracle_change_groups"]),
        "changed_groups": int(counts["changed_groups"]),
        "exact": int(counts["exact"]),
        "exact_changed_groups": int(counts["exact_changed_groups"]),
        "false_change": int(counts["false_change"]),
        "wrong_change": int(counts["wrong_change"]),
        "missed_oracle": int(counts["missed_oracle"]),
        "tesseract_delta_vs_nearest": int(t_sum),
        "parseq_delta_vs_nearest": int(p_sum),
        "oracle_nonnearest_score_rank_counts": rank_counts(rank_values),
        "oracle_topk_rank_counts": rank_counts(topk_values),
    }
    return audits, metrics


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Top-8 latent/code listwise ranker with CRNN/ABINet teacher-loss auxiliary regularization.",
        "Teacher outputs are train-only auxiliary targets, not selector inputs. This is not an actual `.oscr` promotion.",
        "",
        "## Validation Summary",
        "",
        "| teacher weight | seed | exact changed | wrong | Tesseract | PARSeq | rank<=1 | rank<=4 | teacher loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        m = row["val_metrics"]
        ranks = m["oracle_nonnearest_score_rank_counts"]
        lines.append(
            f"| {row['teacher_weight']:.3f} | {row['model_seed']} | {m['exact_changed_groups']} | {m['wrong_change']} | "
            f"{m['tesseract_delta_vs_nearest']} | {m['parseq_delta_vs_nearest']} | "
            f"{ranks['le1']} | {ranks['le4']} | {row['train_meta']['final_losses']['teacher']:.4f} |"
        )
    best = result["best_run"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
            "## Best Run",
            "",
            f"- teacher weight: `{best['teacher_weight']}`",
            f"- seed: `{best['model_seed']}`",
            f"- validation Tesseract delta: `{best['val_metrics']['tesseract_delta_vs_nearest']}`",
            f"- validation exact changed groups: `{best['val_metrics']['exact_changed_groups']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--crnn", type=Path, required=True)
    parser.add_argument("--abinet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_latent_teacher_aux_ranker")
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, action="append", default=[])
    parser.add_argument("--model-seed", type=int, action="append", default=[])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    teacher_weights = args.teacher_weight or [0.0, 0.05, 0.2]
    model_seeds = args.model_seed or [0, 1, 2]

    data = load_npz(args.features)
    table_rows = read_jsonl(args.table)
    crnn_rows = build_teacher_lookup(args.crnn)
    abinet_rows = build_teacher_lookup(args.abinet)
    arrays, group_records, row_records = build_arrays(data, table_rows, crnn_rows, abinet_rows, args.profile)
    runs = []
    for teacher_weight in teacher_weights:
        for seed in model_seeds:
            scores, teacher_pred, meta = train_one(arrays, args, int(seed), float(teacher_weight))
            val_audits, val_metrics = evaluate_scores(arrays, group_records, scores, teacher_pred, "val")
            train_audits, train_metrics = evaluate_scores(arrays, group_records, scores, teacher_pred, "train")
            runs.append(
                {
                    "teacher_weight": float(teacher_weight),
                    "model_seed": int(seed),
                    "train_meta": meta,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "val_audits": val_audits,
                    "train_audits": train_audits,
                }
            )
    best = min(
        runs,
        key=lambda row: (
            row["val_metrics"]["tesseract_delta_vs_nearest"],
            row["val_metrics"]["wrong_change"],
            -row["val_metrics"]["exact_changed_groups"],
            -row["val_metrics"]["oracle_nonnearest_score_rank_counts"]["le1"],
        ),
    )
    best_baseline = min(
        (row for row in runs if row["teacher_weight"] == 0.0),
        key=lambda row: (
            row["val_metrics"]["tesseract_delta_vs_nearest"],
            row["val_metrics"]["wrong_change"],
            -row["val_metrics"]["exact_changed_groups"],
        ),
    )
    interpretation = (
        "This tests teacher-output regularization as distillation into deployable latent/code evidence. "
        "It deliberately excludes reference length and OCR-distance core fields from the input core tensor. "
        "Any positive result would still require a separate no-op acceptor and counted `.oscr` evaluation."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_latent_code_ranker_teacher_aux_not_promoted_selector",
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
            "teacher_weights": teacher_weights,
            "model_seeds": model_seeds,
            "core_keep_names": list(CORE_KEEP_NAMES),
            "input_exclusions": [
                "reference_length",
                "parseq_nearest_distance",
                "tesseract_nearest_distance",
                "teacher predictions/loss/confidence",
            ],
        },
        "data_summary": {
            "groups": int(arrays["mask"].shape[0]),
            "train_groups": int(np.sum(arrays["partition"] == "train")),
            "val_groups": int(np.sum(arrays["partition"] == "val")),
            "row_records": int(len(row_records)),
            "train_listwise_groups": int(len(listwise_groups(arrays, "train"))),
            "val_oracle_change_groups": int(len(listwise_groups(arrays, "val"))),
        },
        "runs": runs,
        "best_run": best,
        "best_baseline_teacher_weight0_run": best_baseline,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "base_ranker_script": sha256_file(ROOT / "scripts/train_top8_trainval_tabular_listwise_ranker.py"),
        },
        "aggregate": {
            "scalar_metrics": {
                "best_val_tesseract_delta": {"value": float(best["val_metrics"]["tesseract_delta_vs_nearest"])},
                "best_val_parseq_delta": {"value": float(best["val_metrics"]["parseq_delta_vs_nearest"])},
                "best_val_exact_changed": {"value": float(best["val_metrics"]["exact_changed_groups"])},
                "best_val_wrong": {"value": float(best["val_metrics"]["wrong_change"])},
                "best_val_rank_le1": {"value": float(best["val_metrics"]["oracle_nonnearest_score_rank_counts"]["le1"])},
                "best_val_rank_le4": {"value": float(best["val_metrics"]["oracle_nonnearest_score_rank_counts"]["le4"])},
                "baseline_weight0_val_tesseract_delta": {
                    "value": float(best_baseline["val_metrics"]["tesseract_delta_vs_nearest"])
                },
                "baseline_weight0_val_exact_changed": {"value": float(best_baseline["val_metrics"]["exact_changed_groups"])},
            }
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "best_run": best["val_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
