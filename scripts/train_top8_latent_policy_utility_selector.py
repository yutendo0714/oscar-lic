#!/usr/bin/env python3
"""Train a no-op-inclusive latent/code selector on train OCR utility labels.

This is a non-promotable policy-objective diagnostic: Tesseract/PARSeq deltas
from the train partition are used as teacher labels, so held-out OCR validity is
not claimed. The goal is to test whether direct policy-level utility targets
behave differently from exact oracle-code ranking.
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
sys.path.insert(0, str(ROOT / "scripts"))

import train_top8_latent_teacher_aux_ranker as base  # noqa: E402
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


def utility_values(arrays: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    tess = arrays["tesseract_delta"].astype(np.float32)
    parseq = arrays["parseq_delta"].astype(np.float32)
    utility = -tess
    utility -= args.parseq_worse_penalty * np.maximum(parseq, 0.0)
    utility -= args.tesseract_worse_penalty * np.maximum(tess, 0.0)
    utility[~arrays["mask"]] = -1.0e6
    nearest = arrays["is_nearest"].astype(bool)
    utility[nearest] = 0.0
    return utility.astype(np.float32)


def target_for_group(arrays: dict[str, np.ndarray], utilities: np.ndarray, group_index: int, min_gain: float) -> int:
    nearest_pos = int(np.flatnonzero(arrays["is_nearest"][group_index])[0])
    values = utilities[group_index].copy()
    values[~arrays["mask"][group_index]] = -1.0e6
    best_pos = int(np.argmax(values))
    if float(values[best_pos]) < min_gain:
        return nearest_pos
    return best_pos


def policy_groups(arrays: dict[str, np.ndarray], partition: str, utilities: np.ndarray, min_gain: float) -> list[dict[str, Any]]:
    group_count, candidate_count = arrays["mask"].shape
    groups = []
    for group_index in range(group_count):
        if arrays["partition"][group_index] != partition:
            continue
        choices = [
            base.flat_index(group_index, pos, candidate_count)
            for pos in range(candidate_count)
            if bool(arrays["mask"][group_index, pos])
        ]
        target_pos = target_for_group(arrays, utilities, group_index, min_gain)
        target = base.flat_index(group_index, target_pos, candidate_count)
        if target not in choices:
            continue
        groups.append({"group_index": group_index, "indices": choices, "target_local": choices.index(target), "target_pos": target_pos})
    return groups


def train_one(
    arrays_raw: dict[str, np.ndarray],
    utilities: np.ndarray,
    args: argparse.Namespace,
    model_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    valid_flat = arrays_raw["mask"].reshape(-1)
    partition = np.repeat(arrays_raw["partition"], arrays_raw["mask"].shape[1])
    train_mask = valid_flat & (partition == "train")
    arrays = base.normalize_arrays(arrays_raw, train_mask)
    flat = base.flatten_candidate_arrays(arrays)
    x_core = torch.from_numpy(flat["core"]).to(device)
    x_code = torch.from_numpy(flat["code"]).to(device)
    x_window = torch.from_numpy(flat["window"]).to(device)
    model = base.LatentTeacherAuxRanker(
        core_dim=flat["core"].shape[1],
        code_dim=flat["code"].shape[1],
        window_channels=flat["window"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    groups = policy_groups(arrays, "train", utilities, args.min_gain)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final = {"loss": 0.0, "ce": 0.0, "pairwise": 0.0}
    for _ in range(args.epochs):
        model.train()
        opt.zero_grad()
        scores, _teacher = model(x_core, x_code, x_window)
        ce_losses = []
        pair_losses = []
        for group in groups:
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
        loss = ce + args.pairwise_weight * pair
        loss.backward()
        opt.step()
        final = {
            "loss": float(loss.detach().cpu().item()),
            "ce": float(ce.detach().cpu().item()),
            "pairwise": float(pair.detach().cpu().item()),
        }
    model.eval()
    with torch.no_grad():
        scores, _teacher = model(x_core, x_code, x_window)
    return scores.detach().cpu().numpy(), {"device": str(device), "final_losses": final, "train_policy_groups": int(len(groups))}


def evaluate_policy(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    scores: np.ndarray,
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_count, candidate_count = arrays["mask"].shape
    counts = Counter()
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
        valid = [pos for pos in range(candidate_count) if bool(arrays["mask"][group_index, pos])]
        selected_pos = max(
            valid,
            key=lambda pos: (float(scores[base.flat_index(group_index, pos, candidate_count)]), -int(arrays["topk_ranks"][group_index, pos])),
        )
        selected_code = int(arrays["code_indices"][group_index, selected_pos])
        status = table_ranker.classify(selected_code, nearest_code, oracle_code)
        changed = selected_pos != nearest_pos
        t_delta = int(arrays["tesseract_delta"][group_index, selected_pos]) if changed else 0
        p_delta = int(arrays["parseq_delta"][group_index, selected_pos]) if changed else 0
        counts[status] += 1
        counts["oracle_change_groups"] += int(oracle_code != nearest_code)
        counts["changed_groups"] += int(changed)
        counts["exact_changed_groups"] += int(status == "exact" and changed)
        counts["tesseract_worse_groups"] += int(t_delta > 0)
        counts["parseq_worse_groups"] += int(p_delta > 0)
        t_sum += t_delta
        p_sum += p_delta
        audits.append(
            {
                "group_index": group_index,
                "key": record["key"],
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "selected_code": selected_code,
                "selected_status": status,
                "selected_tesseract_delta_vs_nearest": t_delta,
                "selected_parseq_delta_vs_nearest": p_delta,
                "oracle_topk_rank": int(arrays["topk_ranks"][group_index, oracle_pos]),
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
        "tesseract_worse_groups": int(counts["tesseract_worse_groups"]),
        "parseq_worse_groups": int(counts["parseq_worse_groups"]),
    }
    return audits, metrics


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "No-op-inclusive latent/code selector trained on train OCR utility labels.",
        "This is non-promotable because train Tesseract/PARSeq deltas are used as teacher labels.",
        "",
        "## Validation Policy",
        "",
        "| seed | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | T worse | P worse |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        m = row["val_metrics"]
        lines.append(
            f"| {row['model_seed']} | {m['changed_groups']} | {m['exact_changed_groups']} | "
            f"{m['false_change']} | {m['wrong_change']} | {m['missed_oracle']} | "
            f"{m['tesseract_delta_vs_nearest']} | {m['parseq_delta_vs_nearest']} | "
            f"{m['tesseract_worse_groups']} | {m['parseq_worse_groups']} |"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
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
    parser.add_argument("--experiment-id", default="eval300_top8_latent_policy_utility_selector")
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--parseq-worse-penalty", type=float, default=8.0)
    parser.add_argument("--tesseract-worse-penalty", type=float, default=2.0)
    parser.add_argument("--min-gain", type=float, default=1.0)
    parser.add_argument("--model-seed", type=int, action="append", default=[])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    seeds = args.model_seed or [0, 1, 2]

    data = base.load_npz(args.features)
    table_rows = read_jsonl(args.table)
    # Teacher files are included only to reuse the common array builder; their
    # values are not used in this objective.
    crnn_rows = base.build_teacher_lookup(args.crnn)
    abinet_rows = base.build_teacher_lookup(args.abinet)
    arrays, group_records, _row_records = base.build_arrays(data, table_rows, crnn_rows, abinet_rows, args.profile)
    utilities = utility_values(arrays, args)
    train_targets = policy_groups(arrays, "train", utilities, args.min_gain)
    runs = []
    for seed in seeds:
        scores, meta = train_one(arrays, utilities, args, int(seed))
        val_audits, val_metrics = evaluate_policy(arrays, group_records, scores, "val")
        train_audits, train_metrics = evaluate_policy(arrays, group_records, scores, "train")
        runs.append(
            {
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
            row["val_metrics"]["false_change"] + row["val_metrics"]["wrong_change"],
            row["val_metrics"]["parseq_worse_groups"],
            -row["val_metrics"]["exact_changed_groups"],
        ),
    )
    interpretation = (
        "This tests whether no-op-inclusive policy utility labels change the failure mode. "
        "Because train Tesseract/PARSeq deltas define the target, this is diagnostic only; "
        "promotion would require held-out OCR separation and counted `.oscr` streams."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_train_ocr_utility_policy_selector_not_promoted",
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
            "parseq_worse_penalty": args.parseq_worse_penalty,
            "tesseract_worse_penalty": args.tesseract_worse_penalty,
            "min_gain": args.min_gain,
            "model_seeds": seeds,
            "target_note": "Train Tesseract/PARSeq deltas are used as teacher labels; non-promotable diagnostic.",
        },
        "data_summary": {
            "train_policy_groups": int(len(train_targets)),
            "train_targets_changed": int(sum(group["target_pos"] != int(np.flatnonzero(arrays["is_nearest"][group["group_index"]])[0]) for group in train_targets)),
            "val_groups": int(np.sum(arrays["partition"] == "val")),
        },
        "runs": runs,
        "best_run": best,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
            "base_teacher_aux_script": sha256_file(ROOT / "scripts/train_top8_latent_teacher_aux_ranker.py"),
        },
        "aggregate": {
            "scalar_metrics": {
                "best_val_tesseract_delta": {"value": float(best["val_metrics"]["tesseract_delta_vs_nearest"])},
                "best_val_parseq_delta": {"value": float(best["val_metrics"]["parseq_delta_vs_nearest"])},
                "best_val_changed": {"value": float(best["val_metrics"]["changed_groups"])},
                "best_val_exact_changed": {"value": float(best["val_metrics"]["exact_changed_groups"])},
                "best_val_false_wrong": {
                    "value": float(best["val_metrics"]["false_change"] + best["val_metrics"]["wrong_change"])
                },
            }
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "best_run": best["val_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
