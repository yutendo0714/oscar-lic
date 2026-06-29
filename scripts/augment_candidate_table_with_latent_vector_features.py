#!/usr/bin/env python3
"""Add low-dimensional MLIC++ latent vector features for candidate ranking."""

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


def patch_for_candidate(value: torch.Tensor, layout: LatentCandidateLayout, candidate_index: int) -> torch.Tensor:
    candidates = list(layout.iter_candidates())
    candidate = candidates[candidate_index]
    c0, c1 = candidate_global_channels(layout, candidate)
    return value[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1].float()


def channel_features(prefix: str, value: torch.Tensor, max_channels: int) -> dict:
    # value shape: 1 x C x H x W
    channels = min(value.shape[1], max_channels)
    selected = value[:, :channels]
    mean = selected.mean(dim=(0, 2, 3))
    mean_abs = selected.abs().mean(dim=(0, 2, 3))
    rms = torch.sqrt(torch.mean(selected * selected, dim=(0, 2, 3)))
    max_abs = selected.abs().amax(dim=(0, 2, 3))
    output = {}
    for index in range(channels):
        output[f"{prefix}_ch{index:02d}_mean"] = float(mean[index].item())
        output[f"{prefix}_ch{index:02d}_mean_abs"] = float(mean_abs[index].item())
        output[f"{prefix}_ch{index:02d}_rms"] = float(rms[index].item())
        output[f"{prefix}_ch{index:02d}_max_abs"] = float(max_abs[index].item())
    return output


def spatial_features(prefix: str, value: torch.Tensor, max_positions: int) -> dict:
    # Average across channels and expose the local 4x4-ish residual layout.
    spatial = value.abs().mean(dim=1).reshape(-1)
    count = min(int(spatial.numel()), max_positions)
    return {f"{prefix}_spatial_abs{index:02d}": float(spatial[index].item()) for index in range(count)}


def augment_group(model, rows: list[dict], device: str, args: argparse.Namespace) -> list[dict]:
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
    output = []
    for row in rows:
        candidate_index = int(row["candidate_index"])
        y_hat_patch = patch_for_candidate(decoded["y_hat"], layout, candidate_index)
        y_target_patch = patch_for_candidate(y_target, layout, candidate_index)
        residual_patch = patch_for_candidate(residual, layout, candidate_index)
        out = dict(row)
        out.update(channel_features("latent_vec_yhat", y_hat_patch, args.max_channels))
        out.update(channel_features("latent_vec_ytarget", y_target_patch, args.max_channels))
        out.update(channel_features("latent_vec_residual", residual_patch, args.max_channels))
        out.update(spatial_features("latent_vec_residual", residual_patch, args.max_spatial_positions))
        output.append(out)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--max-channels", type=int, default=32)
    parser.add_argument("--max-spatial-positions", type=int, default=16)
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
        output_rows.extend(augment_group(model, group, device, args))
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
        "max_channels": args.max_channels,
        "max_spatial_positions": args.max_spatial_positions,
        "features_added": [
            "latent_vec_yhat_chXX_*",
            "latent_vec_ytarget_chXX_*",
            "latent_vec_residual_chXX_*",
            "latent_vec_residual_spatial_absXX",
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
