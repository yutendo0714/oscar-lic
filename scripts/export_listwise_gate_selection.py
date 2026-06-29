#!/usr/bin/env python3
"""Export held-out candidate selections from the listwise patch gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_candidate_selector import (
    choose_threshold,
    choose_top_images,
    read_jsonl,
    rows_by_image,
    split_by_group,
    tune_threshold,
)
from train_candidate_patch_listwise_gate import (
    LATENT_TO_PIXEL,
    build_patch_tensor,
    train_once,
    tune_evaluator_constrained_threshold,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--base-recon-split", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--aux-bce-weight", type=float, default=0.1)
    parser.add_argument("--harm-logit-penalty", type=float, default=0.0)
    parser.add_argument(
        "--target-policy",
        choices=["multi_teacher", "pareto", "tesseract_parseq_safe", "parseq_tesseract_safe"],
        default="tesseract_parseq_safe",
    )
    parser.add_argument("--policy", choices=["threshold_0", "train_tuned", "constrained_tuned", "topk"], default="topk")
    parser.add_argument("--export-subset", choices=["train", "val"], default="val")
    parser.add_argument("--image-budget", type=int, default=25)
    parser.add_argument("--primary-threshold-evaluator", default="tesseract")
    parser.add_argument("--protected-threshold-evaluator", default="parseq")
    parser.add_argument("--protected-max-delta", type=int, default=0)
    parser.add_argument("--latent-to-pixel", type=int, default=LATENT_TO_PIXEL)
    args = parser.parse_args()

    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and args.device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    rows = read_jsonl(args.table)
    split_rows = read_jsonl(args.source_split)
    patches = build_patch_tensor(rows, args.base_recon_split, args.patch_size, args.latent_to_pixel)
    scores, train_mask, val_mask, model_meta = train_once(args, rows, patches, args.seed)
    grouped_train = rows_by_image(rows, np.flatnonzero(train_mask))
    grouped_val = rows_by_image(rows, np.flatnonzero(val_mask))
    grouped_export = grouped_train if args.export_subset == "train" else grouped_val

    if args.policy == "threshold_0":
        threshold_meta = {"threshold": 0.5}
        selected = choose_threshold(grouped_export, scores, 0.5)
    elif args.policy == "train_tuned":
        threshold_meta = tune_threshold(grouped_train, scores)
        selected = choose_threshold(grouped_export, scores, threshold_meta["threshold"])
    elif args.policy == "constrained_tuned":
        threshold_meta = tune_evaluator_constrained_threshold(
            grouped_train,
            scores,
            args.primary_threshold_evaluator,
            args.protected_threshold_evaluator,
            args.protected_max_delta,
        )
        selected = choose_threshold(grouped_export, scores, threshold_meta["threshold"])
    else:
        threshold_meta = {"image_budget": args.image_budget}
        selected = choose_top_images(grouped_export, scores, min(args.image_budget, len(grouped_export)))

    export_source_indices = sorted(grouped_export)
    source_to_export = {source_index: new_index for new_index, source_index in enumerate(export_source_indices)}
    split_export = []
    for source_index in export_source_indices:
        record = dict(split_rows[source_index])
        record["original_source_index"] = source_index
        split_export.append(record)

    selection_rows = []
    for source_index in export_source_indices:
        chosen = selected[source_index]
        if chosen is None:
            continue
        row_index, row = chosen
        exported = dict(row)
        exported["source_index"] = source_to_export[source_index]
        exported["original_source_index"] = source_index
        exported["original_row_index"] = int(row_index)
        exported["selector_score"] = float(scores[row_index])
        exported["selected_by_policy"] = True
        exported["selection_policy"] = args.policy
        selection_rows.append(exported)

    summary = {
        "table": str(args.table),
        "source_split": str(args.source_split),
        "base_recon_split": str(args.base_recon_split),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "target_policy": args.target_policy,
        "policy": args.policy,
        "export_subset": args.export_subset,
        "threshold": threshold_meta,
        "model": model_meta,
        "export_images": len(export_source_indices),
        "selected_images": len(selection_rows),
        "selected_fraction": len(selection_rows) / len(export_source_indices) if export_source_indices else None,
        "export_original_source_indices": export_source_indices,
        "selection_output": str(args.selection_output),
        "split_output": str(args.split_output),
    }

    write_jsonl(args.split_output, split_export)
    write_jsonl(args.selection_output, selection_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
