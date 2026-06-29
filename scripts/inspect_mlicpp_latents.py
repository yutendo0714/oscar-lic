#!/usr/bin/env python3
"""Inspect MLIC++ latent likelihood structure for OSCAR candidate design.

This is a diagnostic script, not an evaluation codec. It records complete
compress/decompress stream bytes for rate accounting and separately records
likelihood-derived slice/anchor estimates to guide candidate-unit design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
MLIC_ROOT = ROOT / "external/mlic/MLIC++"
sys.path.insert(0, str(MLIC_ROOT))

from config.config import model_config  # noqa: E402
from models.mlicpp import MLICPlusPlus  # noqa: E402
from utils.utils import read_body, read_uints, write_body, write_uints  # noqa: E402


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pad_to_multiple_64(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = x.shape
    pad_h = 0 if height % 64 == 0 else 64 * (height // 64 + 1) - height
    pad_w = 0 if width % 64 == 0 else 64 * (width // 64 + 1) - width
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0), pad_h, pad_w


def read_image_list(split: Path | None, images: Iterable[Path] | None, max_images: int | None) -> list[Path]:
    paths: list[Path] = []
    if split is not None:
        for line in split.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                record = json.loads(line)
                value = record.get("image_path") or record.get("image") or record.get("path")
                if value is None:
                    raise ValueError(f"split row does not contain image path: {line[:120]}")
                paths.append(Path(value))
            else:
                paths.append(Path(line))
    if images:
        paths.extend(images)
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise SystemExit("no images provided")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing images: {missing[:5]}")
    return paths


def load_model(checkpoint_path: Path, device: str) -> tuple[MLICPlusPlus, dict]:
    model = MLICPlusPlus(config=model_config()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.update(force=True)
    model.eval()
    return model, checkpoint


def write_stream(path: Path, height: int, width: int, shape, strings) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        write_uints(handle, (height, width))
        write_body(handle, shape, strings)
    return path.stat().st_size


def read_stream(path: Path):
    with path.open("rb") as handle:
        original_size = read_uints(handle, 2)
        strings, shape = read_body(handle)
    return original_size, strings, shape


def tensor_stats(value: torch.Tensor) -> dict:
    flat = value.detach().float().reshape(-1)
    quantiles = torch.quantile(flat, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=flat.device))
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(quantiles[0].item()),
        "p10": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p90": float(quantiles[3].item()),
        "max": float(quantiles[4].item()),
    }


def anchor_mask_like(likelihoods: torch.Tensor) -> torch.Tensor:
    _, _, height, width = likelihoods.shape
    mask = torch.zeros((1, 1, height, width), dtype=torch.bool, device=likelihoods.device)
    mask[:, :, 0::2, 1::2] = True
    mask[:, :, 1::2, 0::2] = True
    return mask.expand_as(likelihoods)


def likelihood_breakdown(likelihoods: torch.Tensor, *, slice_num: int, pixels: int) -> dict:
    eps = torch.finfo(likelihoods.dtype).eps
    bits = -torch.log2(likelihoods.clamp_min(eps))
    slices = bits.chunk(slice_num, dim=1)
    likelihood_slices = likelihoods.chunk(slice_num, dim=1)
    rows = []
    for idx, (slice_bits, slice_likelihoods) in enumerate(zip(slices, likelihood_slices)):
        mask = anchor_mask_like(slice_bits)
        anchor_bits = slice_bits[mask].sum()
        nonanchor_bits = slice_bits[~mask].sum()
        total_bits = slice_bits.sum()
        rows.append(
            {
                "slice_index": idx,
                "channels": int(slice_bits.shape[1]),
                "estimated_total_bits": float(total_bits.item()),
                "estimated_total_bpp": float(total_bits.item() / pixels),
                "estimated_anchor_bits": float(anchor_bits.item()),
                "estimated_anchor_bpp": float(anchor_bits.item() / pixels),
                "estimated_nonanchor_bits": float(nonanchor_bits.item()),
                "estimated_nonanchor_bpp": float(nonanchor_bits.item() / pixels),
                "estimated_bits_per_symbol": float(slice_bits.mean().item()),
                "likelihood_stats": tensor_stats(slice_likelihoods),
                "bit_stats": tensor_stats(slice_bits),
            }
        )
    return {
        "estimated_total_bits": float(bits.sum().item()),
        "estimated_total_bpp": float(bits.sum().item() / pixels),
        "estimated_bits_per_symbol": float(bits.mean().item()),
        "slice_rows": rows,
    }


def run_one(model: MLICPlusPlus, image_path: Path, output_dir: Path, device: str, index: int) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    pixels = width * height
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = pad_to_multiple_64(x)

    with torch.no_grad():
        forward_start = perf_counter()
        out_forward = model(x_pad)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_seconds = perf_counter() - forward_start

    y_likelihoods = out_forward["likelihoods"]["y_likelihoods"]
    z_likelihoods = out_forward["likelihoods"]["z_likelihoods"]
    y_breakdown = likelihood_breakdown(y_likelihoods, slice_num=model.slice_num, pixels=pixels)
    z_bits = -torch.log2(z_likelihoods.clamp_min(torch.finfo(z_likelihoods.dtype).eps))

    stream_path = output_dir / "streams" / f"{index:03d}_{image_path.stem}.mlicpp"
    with torch.no_grad():
        compressed = model.compress(x_pad)
    stream_bytes = write_stream(stream_path, height, width, compressed["shape"], compressed["strings"])

    model.update_resolutions(16, 16)
    _, strings, shape = read_stream(stream_path)
    with torch.no_grad():
        decoded = model.decompress(strings, shape)
    x_hat = decoded["x_hat"][:, :, :height, :width]

    mse = F.mse_loss(x_hat.clamp(0, 1), x).item()
    psnr = float(-10.0 * torch.log10(torch.tensor(mse)).item())
    estimated_total_bits = y_breakdown["estimated_total_bits"] + float(z_bits.sum().item())
    return {
        "image": project_path(image_path),
        "width": width,
        "height": height,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "padded_width": int(x_pad.shape[-1]),
        "padded_height": int(x_pad.shape[-2]),
        "latent_y_shape": list(y_likelihoods.shape),
        "latent_z_shape": list(z_likelihoods.shape),
        "actual_total_bytes": stream_bytes,
        "actual_total_bpp": 8.0 * stream_bytes / pixels,
        "stream_path": project_path(stream_path),
        "psnr_db_tensor": psnr,
        "forward_seconds": forward_seconds,
        "encode_seconds_model_reported": float(compressed["cost_time"]),
        "decode_seconds_model_reported": float(decoded["cost_time"]),
        "estimated_rate_note": "likelihood-derived diagnostic only; publication bpp must use actual_total_bytes",
        "estimated_total_bits": estimated_total_bits,
        "estimated_total_bpp": estimated_total_bits / pixels,
        "estimated_y": y_breakdown,
        "estimated_z": {
            "estimated_total_bits": float(z_bits.sum().item()),
            "estimated_total_bpp": float(z_bits.sum().item() / pixels),
            "estimated_bits_per_symbol": float(z_bits.mean().item()),
            "likelihood_stats": tensor_stats(z_likelihoods),
            "bit_stats": tensor_stats(z_bits),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--images", type=Path, nargs="*")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    image_paths = read_image_list(args.split, args.images, args.max_images)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)

    rows = []
    for index, image_path in enumerate(image_paths):
        rows.append(run_one(model, image_path, output_dir, device, index))

    actual_bpps = [row["actual_total_bpp"] for row in rows]
    estimated_bpps = [row["estimated_total_bpp"] for row in rows]
    y_share_by_slice = []
    for slice_idx in range(model.slice_num):
        slice_bits = sum(row["estimated_y"]["slice_rows"][slice_idx]["estimated_total_bits"] for row in rows)
        total_y_bits = sum(row["estimated_y"]["estimated_total_bits"] for row in rows)
        y_share_by_slice.append(
            {
                "slice_index": slice_idx,
                "estimated_y_bit_share": float(slice_bits / total_y_bits) if total_y_bits else 0.0,
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "images": len(rows),
        "avg_actual_total_bpp": float(sum(actual_bpps) / len(actual_bpps)),
        "avg_estimated_total_bpp": float(sum(estimated_bpps) / len(estimated_bpps)),
        "avg_actual_minus_estimated_bpp": float(
            sum(a - e for a, e in zip(actual_bpps, estimated_bpps)) / len(rows)
        ),
        "slice_num": model.slice_num,
        "slice_ch": model.slice_ch,
        "candidate_layout_implication": "upstream MLIC++ emits one main y ANS stream; per-slice likelihoods guide design but do not provide actual per-slice bytes",
        "estimated_y_share_by_slice": y_share_by_slice,
    }

    results_path = output_dir / "latent_inspection.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"summary": summary, "results_path": str(results_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
