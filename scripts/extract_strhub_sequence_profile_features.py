#!/usr/bin/env python3
"""Extract binned StrHub sequence-profile features for candidate images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strhub.data.module import SceneTextDataModule  # noqa: E402
from strhub.models.base import CTCSystem  # noqa: E402
from strhub.models.utils import create_model  # noqa: E402


BIN_STAT_NAMES = ("mean", "std", "min", "max")
SIGNAL_NAMES = ("max_prob", "margin", "entropy", "pred_id_norm", "transition", "blank_or_pad_prob")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def resolve_image(row: dict, base: Path) -> Path:
    value = row.get("image_path", row.get("image"))
    if value is None:
        raise KeyError("row has neither image_path nor image")
    path = Path(value)
    if not path.is_absolute():
        split_relative = (base / path).resolve()
        project_relative = (ROOT / path).resolve()
        path = split_relative if split_relative.is_file() else project_relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def create_strhub_model(model_name: str, iter_size: int):
    if model_name == "abinet":
        return create_model("abinet", pretrained=True, iter_size=iter_size)
    return create_model(model_name, pretrained=True)


def bin_stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return [0.0] * len(BIN_STAT_NAMES)
    return [float(values.mean()), float(values.std()), float(values.min()), float(values.max())]


def sequence_signals(logits: torch.Tensor, model) -> dict[str, np.ndarray]:
    probs = logits.softmax(-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    max_prob = top2[:, 0].detach().cpu().numpy()
    margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()
    probs_safe = probs.clamp_min(1.0e-9)
    entropy = (-(probs_safe * probs_safe.log()).sum(dim=-1)).detach().cpu().numpy()
    pred_ids = probs.argmax(dim=-1).detach().cpu().numpy().astype(np.float32)
    vocab_norm = max(float(probs.shape[-1] - 1), 1.0)
    pred_norm = pred_ids / vocab_norm
    transition = np.zeros_like(pred_norm)
    if pred_norm.size > 1:
        transition[1:] = (pred_ids[1:] != pred_ids[:-1]).astype(np.float32)
    blank_or_pad_id = None
    if isinstance(model, CTCSystem):
        blank_or_pad_id = int(model.blank_id)
    elif hasattr(model, "pad_id"):
        blank_or_pad_id = int(model.pad_id)
    blank_prob = np.zeros_like(max_prob)
    if blank_or_pad_id is not None and 0 <= blank_or_pad_id < probs.shape[-1]:
        blank_prob = probs[:, blank_or_pad_id].detach().cpu().numpy()
    return {
        "max_prob": max_prob,
        "margin": margin,
        "entropy": entropy,
        "pred_id_norm": pred_norm,
        "transition": transition,
        "blank_or_pad_prob": blank_prob,
    }


def summarize_sequence(logits: torch.Tensor, model, bins: int) -> np.ndarray:
    signals = sequence_signals(logits, model)
    time_steps = logits.shape[0]
    edges = np.linspace(0, time_steps, bins + 1, dtype=np.int64)
    values: list[float] = []
    for bin_index in range(bins):
        start = int(edges[bin_index])
        end = int(edges[bin_index + 1])
        if end <= start and start < time_steps:
            end = start + 1
        for signal_name in SIGNAL_NAMES:
            values.extend(bin_stats(signals[signal_name][start:end]))
    values.extend([float(time_steps), float(logits.shape[-1])])
    return np.asarray(values, dtype=np.float32)


def feature_names(prefix: str, bins: int) -> list[str]:
    names = []
    for bin_index in range(bins):
        for signal_name in SIGNAL_NAMES:
            names.extend([f"{prefix}_bin{bin_index:02d}_{signal_name}_{stat}" for stat in BIN_STAT_NAMES])
    names.extend([f"{prefix}_time_steps", f"{prefix}_vocab_size"])
    return names


@torch.inference_mode()
def extract_features(
    model,
    transform,
    rows: list[dict],
    base: Path,
    batch_size: int,
    device: str,
    max_len: int,
    bins: int,
) -> tuple[np.ndarray, dict]:
    features = []
    seconds = []
    is_ctc = isinstance(model, CTCSystem)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = [transform(Image.open(resolve_image(row, base)).convert("RGB")) for row in batch]
        tensor = torch.stack(images).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = perf_counter()
        logits = model(tensor) if is_ctc else model.forward(tensor, max_len)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (perf_counter() - start_time) / max(len(batch), 1)
        for idx in range(len(batch)):
            features.append(summarize_sequence(logits[idx], model, bins))
            seconds.append(elapsed)
    return np.vstack(features).astype(np.float32), {
        "mean_infer_seconds": float(np.mean(seconds)),
        "samples": int(len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-name", choices=["crnn", "abinet", "trba", "vitstr"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--iter-size", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=25)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but unavailable")
    model = create_strhub_model(args.model_name, args.iter_size).eval().to(device)
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    features, timing = extract_features(
        model,
        transform,
        rows,
        args.input_jsonl.parent,
        args.batch_size,
        device,
        args.max_len,
        args.bins,
    )
    keys = np.asarray(
        [
            [
                int(row.get("real_seed", row.get("seed", 0))),
                int(row["seed"]),
                int(row["source_index"]),
                int(row["candidate_index"]),
                int(row["code_index"]),
            ]
            for row in rows
        ],
        dtype=np.int64,
    )
    image_paths = np.asarray([project_path(resolve_image(row, args.input_jsonl.parent)) for row in rows])
    names = np.asarray(feature_names(args.model_name, args.bins))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        row_keys=keys,
        features=features,
        feature_names=names,
        image_paths=image_paths,
    )
    checkpoint = sorted((Path(torch.hub.get_dir()) / "checkpoints").glob(f"{args.model_name}-*.pt"))
    checkpoint_path = checkpoint[0] if checkpoint else None
    summary = {
        "input_jsonl": project_path(args.input_jsonl),
        "input_sha256": sha256_file(args.input_jsonl),
        "output": project_path(args.output),
        "model_name": args.model_name,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bins": int(args.bins),
        "feature_dim": int(features.shape[1]),
        **timing,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
