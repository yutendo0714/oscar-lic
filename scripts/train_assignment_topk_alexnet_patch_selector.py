#!/usr/bin/env python3
"""Train a top-k selector with fixed AlexNet patch embeddings.

This diagnostic uses a cached torchvision AlexNet ImageNet1K checkpoint as a
frozen visual feature extractor over nearest/candidate/difference patches. It
does not promote bitstreams and should not be used for publication claims until
the external weight/data license status is reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import alexnet

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_assignment_patch_ranker as patch_ranker  # noqa: E402
import train_assignment_topk_policy_selector as base  # noqa: E402
import train_assignment_topk_policy_selector_split_diagnostic as split_diag  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AlexNetConvPool(torch.nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        model = alexnet(weights=None)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        self.features = model.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(image)).flatten(1)


def extract_alexnet_features(
    images_np: np.ndarray,
    checkpoint: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = AlexNetConvPool(checkpoint).to(device)
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images_np), batch_size):
            batch = torch.from_numpy(images_np[start : start + batch_size]).to(device)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std
            outputs.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-output", type=Path)
    parser.add_argument("--val-seed", type=int, default=1)
    parser.add_argument("--score-model-seed", type=int, action="append")
    parser.add_argument("--selector-seed", type=int, action="append")
    parser.add_argument("--shortlist-topk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--harmful-margin-weight", type=float, default=1.0)
    parser.add_argument("--harmful-margin", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--image-mode", choices=["full", "diff_crop"], default="diff_crop")
    parser.add_argument("--diff-threshold", type=float, default=0.002)
    parser.add_argument("--crop-pad", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.encoder_checkpoint.is_file():
        raise SystemExit(f"missing encoder checkpoint: {args.encoder_checkpoint}")
    if not args.score_model_seed:
        args.score_model_seed = [2]
    if not args.selector_seed:
        args.selector_seed = [0, 1, 2]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows = base.read_jsonl(args.table)
    full_groups = base.grouped_indices(rows)
    base_features = base.build_base_features(rows)
    oracle_by_model = base.load_score_features(args.oracle_score)
    safe_by_model = base.load_score_features(args.safe_score)

    images_np = patch_ranker.build_image_tensor(
        rows,
        width=args.width,
        height=args.height,
        image_mode=args.image_mode,
        diff_threshold=args.diff_threshold,
        crop_pad=args.crop_pad,
    )
    visual_features = extract_alexnet_features(
        images_np,
        args.encoder_checkpoint,
        batch_size=args.batch_size,
        device=device,
    )
    if args.feature_output is not None:
        args.feature_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.feature_output,
            alexnet_features=visual_features,
            encoder_checkpoint=str(args.encoder_checkpoint),
            encoder_sha256=sha256_file(args.encoder_checkpoint),
            image_mode=args.image_mode,
            width=int(args.width),
            height=int(args.height),
        )

    runs: list[dict[str, Any]] = []
    for score_model_seed in args.score_model_seed:
        oracle_scores = np.asarray(
            [oracle_by_model[score_model_seed][idx] for idx in range(len(rows))],
            dtype=np.float32,
        )
        safe_scores = np.asarray(
            [safe_by_model[score_model_seed][idx] for idx in range(len(rows))],
            dtype=np.float32,
        )
        selector_features = base.build_selector_features(rows, full_groups, base_features, oracle_scores, safe_scores)
        features = np.concatenate([selector_features.astype(np.float32), visual_features.astype(np.float32)], axis=1)
        group_candidates = {
            key: base.candidate_indices_for_group(rows, indices, oracle_scores, safe_scores, topk=args.shortlist_topk)
            for key, indices in full_groups.items()
        }
        for selector_seed in args.selector_seed:
            runs.append(
                split_diag.train_one_split(
                    rows,
                    full_groups,
                    features,
                    group_candidates,
                    score_model_seed=score_model_seed,
                    selector_seed=selector_seed,
                    val_real_seed=None,
                        val_source_remainder=None,
                        args=argparse.Namespace(
                            split_mode="seed",
                            val_seed=args.val_seed,
                            epochs=args.epochs,
                            hidden_dim=args.hidden_dim,
                            dropout=args.dropout,
                            lr=args.lr,
                        weight_decay=args.weight_decay,
                        harmful_margin_weight=args.harmful_margin_weight,
                        harmful_margin=args.harmful_margin,
                        cpu=args.cpu,
                    ),
                )
            )

    result = {
        "description": "Top-k assignment selector with fixed AlexNet patch embeddings. Non-promotion diagnostic only.",
        "config": {
            "table": str(args.table),
            "oracle_score": str(args.oracle_score),
            "safe_score": str(args.safe_score),
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "encoder_checkpoint_sha256": sha256_file(args.encoder_checkpoint),
            "feature_output": None if args.feature_output is None else str(args.feature_output),
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
            "width": int(args.width),
            "height": int(args.height),
            "image_mode": args.image_mode,
            "diff_threshold": float(args.diff_threshold),
            "crop_pad": int(args.crop_pad),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "cpu": bool(args.cpu),
        },
        "per_run": runs,
        "aggregate": split_diag.aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
