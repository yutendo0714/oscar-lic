#!/usr/bin/env python3
"""Augment candidate utility rows with MLIC++ latent/residual statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from PIL import Image
import torch
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
MLIC_ROOT = ROOT / "external/mlic/MLIC++"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MLIC_ROOT))

from inspect_mlicpp_decoded_yhat import decompress_with_y_hat, load_model, pad_to_multiple_64  # noqa: E402
from oscarlic.candidates import LatentCandidateLayout  # noqa: E402
from run_mlicpp_latent_residual_stub import candidate_global_channels  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(value: torch.Tensor) -> dict:
    value = value.float()
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "mean_abs": float(value.abs().mean().item()),
        "rms": float(torch.sqrt(torch.mean(value ** 2)).item()),
        "max_abs": float(value.abs().max().item()),
    }


def patch_for_candidate(value: torch.Tensor, layout: LatentCandidateLayout, candidate_index: int) -> torch.Tensor:
    candidates = list(layout.iter_candidates())
    candidate = candidates[candidate_index]
    c0, c1 = candidate_global_channels(layout, candidate)
    return value[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1]


def augment_group(model, rows: list[dict], device: str) -> list[dict]:
    image_path = Path(rows[0]["source_image"])
    image = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, _, _ = pad_to_multiple_64(x)
    with torch.no_grad():
        compressed = model.compress(x_pad)
        decoded = decompress_with_y_hat(model, compressed["strings"], compressed["shape"])
        y_target = model.g_a(x_pad)
        residual = y_target - decoded["y_hat"]
    _, channels, latent_height, latent_width = residual.shape
    first = rows[0]
    layout = LatentCandidateLayout(
        latent_height=latent_height,
        latent_width=latent_width,
        slice_num=10,
        slice_channels=channels // 10,
        tile_height=int(first["tile"]),
        tile_width=int(first["tile"]),
        channel_group_size=None,
    )
    augmented = []
    for row in rows:
        candidate_index = int(row["candidate_index"])
        y_hat_patch = patch_for_candidate(decoded["y_hat"], layout, candidate_index)
        y_target_patch = patch_for_candidate(y_target, layout, candidate_index)
        residual_patch = patch_for_candidate(residual, layout, candidate_index)
        out = dict(row)
        for prefix, values in [
            ("y_hat", stats(y_hat_patch)),
            ("y_target", stats(y_target_patch)),
            ("latent_residual", stats(residual_patch)),
        ]:
            for key, value in values.items():
                out[f"{prefix}_{key}"] = value
        augmented.append(out)
    return augmented


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device != "cuda":
        raise SystemExit("MLIC++ compress/decompress uses torch.cuda.synchronize(); run with --cuda on a visible GPU")

    rows = read_jsonl(args.input)
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["source_index"])].append(row)
    model, checkpoint = load_model(args.checkpoint, device)
    output_rows = []
    for source_index, group in sorted(groups.items()):
        output_rows.extend(augment_group(model, group, device))
        print(json.dumps({"source_index": source_index, "rows": len(group)}, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "images": len(groups),
        "features_added": [
            "y_hat_mean_abs",
            "y_hat_rms",
            "y_target_mean_abs",
            "y_target_rms",
            "latent_residual_mean_abs",
            "latent_residual_rms",
            "latent_residual_max_abs",
        ],
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
