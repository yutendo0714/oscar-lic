#!/usr/bin/env python3
"""Run an actual-bitstream MLIC++ compress/decompress smoke test.

This script intentionally uses the upstream MLIC++ model and stream helpers,
but keeps the experiment orchestration in the OSCAR-LIC tree. Reported bpp is
computed from the complete stream file size produced by the helper format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
MLIC_ROOT = ROOT / "external/mlic/MLIC++"
sys.path.insert(0, str(MLIC_ROOT))

from config.config import model_config  # noqa: E402
from models.mlicpp import MLICPlusPlus  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402
from utils.utils import read_body, read_uints, torch2img, write_body, write_uints  # noqa: E402


DEFAULT_IMAGES = [
    ROOT / "data/raw/clic/professional/valid/schicka-307.png",
    ROOT / "data/raw/clic/professional/valid/todd-quackenbush-222.png",
]


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


def compute_psnr_only(rec: Image.Image, gt: Image.Image) -> float:
    rec_np = np.asarray(rec).copy()
    gt_np = np.asarray(gt).copy()
    rec_t = torch.from_numpy(rec_np).float()
    gt_t = torch.from_numpy(gt_np).float()
    mse = torch.mean((rec_t - gt_t) ** 2).item()
    return 20.0 * torch.log10(torch.tensor(255.0)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()


def load_model(checkpoint_path: Path, device: str) -> tuple[MLICPlusPlus, dict]:
    model = MLICPlusPlus(config=model_config()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.update(force=True)
    model.eval()
    return model, checkpoint


def run_one(model: MLICPlusPlus, image_path: Path, output_dir: Path, device: str, index: int) -> dict:
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x = transform(image).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = pad_to_multiple_64(x)

    stream_path = output_dir / "streams" / f"{index:03d}_{image_path.stem}.mlicpp"
    rec_path = output_dir / "reconstructions" / f"{index:03d}_{image_path.stem}_rec.png"
    gt_path = output_dir / "reconstructions" / f"{index:03d}_{image_path.stem}_gt.png"

    with torch.no_grad():
        out = model.compress(x_pad)
    stream_bytes = write_stream(stream_path, height, width, out["shape"], out["strings"])

    # Match upstream test.py behavior: reset cached attention resolution between
    # independent compression/decompression calls to avoid resolution leakage.
    model.update_resolutions(16, 16)

    original_size, strings, shape = read_stream(stream_path)
    dec_start = perf_counter()
    with torch.no_grad():
        decoded = model.decompress(strings, shape)
    dec_wall = perf_counter() - dec_start
    x_hat = decoded["x_hat"][:, :, : original_size[0], : original_size[1]]

    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec = torch2img(x_hat.detach().cpu())
    image.save(gt_path)
    rec.save(rec_path)
    try:
        psnr, msssim = compute_metrics(rec, image)
    except AssertionError:
        psnr, msssim = compute_psnr_only(rec, image), None

    return {
        "image": project_path(image_path),
        "width": width,
        "height": height,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "actual_total_bytes": stream_bytes,
        "actual_total_bpp": 8.0 * stream_bytes / (height * width),
        "stream_path": project_path(stream_path),
        "reconstruction_path": project_path(rec_path),
        "psnr_db": psnr,
        "ms_ssim": msssim,
        "encode_seconds_model_reported": float(out["cost_time"]),
        "decode_seconds_model_reported": float(decoded["cost_time"]),
        "decode_seconds_wall": dec_wall,
        "bitstream_format": "upstream_mlicpp_write_uints_write_body",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results/mlicpp_smoke")
    parser.add_argument("--images", type=Path, nargs="*", default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    image_paths = args.images if args.images else [p for p in DEFAULT_IMAGES if p.is_file()]
    if not image_paths:
        image_paths = sorted((ROOT / "data/toy").glob("*.png"))[:2]
    if not image_paths:
        raise SystemExit("no smoke images found")
    for image_path in image_paths:
        if not image_path.is_file():
            raise SystemExit(f"missing image: {image_path}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    torch.backends.cudnn.deterministic = True

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)

    if not args.no_warmup:
        image = Image.open(image_paths[0]).convert("RGB")
        x = transforms.ToTensor()(image).unsqueeze(0).to(device)
        x_pad, _, _ = pad_to_multiple_64(x)
        with torch.no_grad():
            _ = model.compress(x_pad)
        model.update_resolutions(16, 16)

    rows = []
    for index, image_path in enumerate(image_paths):
        rows.append(run_one(model, image_path, args.output_dir, device, index))

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "images": len(rows),
        "avg_actual_total_bpp": sum(row["actual_total_bpp"] for row in rows) / len(rows),
        "avg_psnr_db": sum(row["psnr_db"] for row in rows) / len(rows),
        "avg_ms_ssim": (
            sum(row["ms_ssim"] for row in rows if row["ms_ssim"] is not None)
            / sum(1 for row in rows if row["ms_ssim"] is not None)
            if any(row["ms_ssim"] is not None for row in rows)
            else None
        ),
    }

    results_path = args.output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "results_path": str(results_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
