#!/usr/bin/env python3
"""Extract PARSeq teacher encoder features for source-OOF top-k assignment rows.

The output is a feature cache for diagnostic selectors. PARSeq is the registered
training teacher, not a held-out evaluator. These features are not deployable
without accounting for OCR encoder cost, so any downstream selector is a
teacher-evidence diagnostic rather than a promotion candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "external" / "parseq"))

import train_assignment_topk_policy_selector as base  # noqa: E402
from strhub.data.module import SceneTextDataModule  # noqa: E402
from strhub.models.utils import create_model  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_path(row: dict[str, Any]) -> Path:
    path = Path(row["image_path"])
    if path.is_absolute():
        return path
    return ROOT / path


def pooled_memory_features(memory: torch.Tensor) -> np.ndarray:
    # memory: [B, tokens, channels]
    mean = memory.mean(dim=1)
    std = memory.std(dim=1, unbiased=False)
    maxv = memory.amax(dim=1)
    minv = memory.amin(dim=1)
    return torch.cat([mean, std, maxv, minv], dim=1).detach().cpu().numpy().astype(np.float32)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_relative_features(
    rows: list[dict[str, Any]],
    groups: dict[tuple[int, int, int, int], list[int]],
    embeddings: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(rows), embeddings.shape[1] * 3 + 4), dtype=np.float32)
    for indices in groups.values():
        nearest = next(idx for idx in indices if int(rows[idx]["is_nearest"]))
        nearest_emb = embeddings[nearest]
        for idx in indices:
            emb = embeddings[idx]
            diff = emb - nearest_emb
            output[idx, : embeddings.shape[1]] = emb
            output[idx, embeddings.shape[1] : 2 * embeddings.shape[1]] = diff
            output[idx, 2 * embeddings.shape[1] : 3 * embeddings.shape[1]] = np.abs(diff)
            tail = 3 * embeddings.shape[1]
            output[idx, tail + 0] = float(np.linalg.norm(diff) / np.sqrt(max(diff.size, 1)))
            output[idx, tail + 1] = safe_cosine(emb, nearest_emb)
            output[idx, tail + 2] = float(np.mean(np.abs(diff)))
            output[idx, tail + 3] = float(np.max(np.abs(diff))) if diff.size else 0.0
    return output


@torch.inference_mode()
def extract_embeddings(rows: list[dict[str, Any]], batch_size: int, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    model = create_model("parseq", pretrained=True, decode_ar=True, refine_iters=1).eval().to(device)
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    features: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = []
        for row in batch:
            image_path = resolve_image_path(row)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            images.append(transform(Image.open(image_path).convert("RGB")))
        tensor = torch.stack(images).to(device)
        memory = model.model.encode(tensor)
        features.append(pooled_memory_features(memory))
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "parseq-bb5792a6.pt"
    metadata = {
        "model": "parseq",
        "role": "train_teacher_feature_diagnostic",
        "checkpoint_path": str(checkpoint) if checkpoint.is_file() else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "embedding_feature": "encoder_memory_mean_std_max_min",
        "img_size": list(model.hparams.img_size),
        "device": device,
        "torch": torch.__version__,
    }
    return np.concatenate(features, axis=0).astype(np.float32), metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--oracle-score", type=Path, required=True)
    parser.add_argument("--safe-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-model-seed", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but unavailable")

    rows = base.read_jsonl(args.table)
    groups = base.grouped_indices(rows)
    base_features = base.build_base_features(rows)
    oracle_by_model = base.load_score_features(args.oracle_score)
    safe_by_model = base.load_score_features(args.safe_score)
    oracle_scores = np.asarray([oracle_by_model[args.score_model_seed][idx] for idx in range(len(rows))], dtype=np.float32)
    safe_scores = np.asarray([safe_by_model[args.score_model_seed][idx] for idx in range(len(rows))], dtype=np.float32)
    selector_features = base.build_selector_features(rows, groups, base_features, oracle_scores, safe_scores)
    embeddings, metadata = extract_embeddings(rows, args.batch_size, device)
    parseq_relative = build_relative_features(rows, groups, embeddings)
    features = np.concatenate([selector_features.astype(np.float32), parseq_relative], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise SystemExit("non-finite feature cache")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        parseq_embeddings=embeddings,
        score_model_seed=np.asarray([args.score_model_seed], dtype=np.int64),
        metadata=np.asarray([json.dumps(metadata, ensure_ascii=False)]),
    )
    result = {
        "output": str(args.output),
        "rows": len(rows),
        "feature_dim": int(features.shape[1]),
        "parseq_embedding_dim": int(embeddings.shape[1]),
        "sha256": sha256_file(args.output),
        "metadata": metadata,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
