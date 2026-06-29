#!/usr/bin/env python3
"""Run a DCAE actual-bitstream compress/decompress smoke test.

The upstream DCAE script imports optional analysis packages at module import
time. This wrapper keeps the smoke path minimal while preserving the upstream
binary stream layout: original H/W, y stream length+bytes, z stream length+bytes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from time import perf_counter

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage


ROOT = Path(__file__).resolve().parents[1]
DCAE_ROOT = ROOT / "external/dcae"
sys.path.insert(0, str(DCAE_ROOT))

import compressai.models as compressai_models  # noqa: E402
from compressai.entropy_models import EntropyBottleneck  # noqa: E402
from compressai.models.utils import update_registered_buffers  # noqa: E402


class DCAECompressionModelCompat(nn.Module):
    """Compatibility shim for DCAE's pre-CompressAI-1.2 base-model API."""

    def __init__(self, init_weights: bool = True):
        super().__init__()
        if init_weights:
            self._initialize_weights()

    def aux_loss(self):
        return sum(module.loss() for module in self.modules() if isinstance(module, EntropyBottleneck))

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def update(self, force: bool = False):
        updated = False
        for module in self.children():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force)
        return updated

    def load_state_dict(self, state_dict, strict: bool = True):
        if hasattr(self, "entropy_bottleneck"):
            update_registered_buffers(
                self.entropy_bottleneck,
                "entropy_bottleneck",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )
        return super().load_state_dict(state_dict, strict=strict)


compressai_models.CompressionModel = DCAECompressionModelCompat

from models import DCAE  # noqa: E402


DEFAULT_IMAGES = [
    ROOT / "data/toy/toy_00.png",
    ROOT / "data/toy/toy_01.png",
]


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pad_to_multiple(x: torch.Tensor, multiple: int = 128) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, height, width = x.shape
    new_h = (height + multiple - 1) // multiple * multiple
    new_w = (width + multiple - 1) // multiple * multiple
    padding_left = (new_w - width) // 2
    padding_right = new_w - width - padding_left
    padding_top = (new_h - height) // 2
    padding_bottom = new_h - height - padding_top
    padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)
    return padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))


def calculate_padding(height: int, width: int, multiple: int = 128) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    new_h = (height + multiple - 1) // multiple * multiple
    new_w = (width + multiple - 1) // multiple * multiple
    padding_left = (new_w - width) // 2
    padding_right = new_w - width - padding_left
    padding_top = (new_h - height) // 2
    padding_bottom = new_h - height - padding_top
    return (new_h, new_w), (padding_left, padding_right, padding_top, padding_bottom)


def write_stream(path: Path, height: int, width: int, strings) -> int:
    if height > 65535 or width > 65535:
        raise ValueError("upstream DCAE stream format stores H/W as uint16")
    y_string = strings[0][0]
    z_string = strings[1][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack(">H", height))
        handle.write(struct.pack(">H", width))
        handle.write(struct.pack(">I", len(y_string)))
        handle.write(y_string)
        handle.write(struct.pack(">I", len(z_string)))
        handle.write(z_string)
    return path.stat().st_size


def read_stream(path: Path):
    with path.open("rb") as handle:
        height = struct.unpack(">H", handle.read(2))[0]
        width = struct.unpack(">H", handle.read(2))[0]
        length_y = struct.unpack(">I", handle.read(4))[0]
        string_y = handle.read(length_y)
        length_z = struct.unpack(">I", handle.read(4))[0]
        string_z = handle.read(length_z)
    padded_size, padding = calculate_padding(height, width)
    z_shape = [padded_size[0] // 64, padded_size[1] // 64]
    return {
        "height": height,
        "width": width,
        "strings": [[string_y], [string_z]],
        "shape": z_shape,
        "padding": padding,
    }


def compute_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def load_model(checkpoint_path: Path, device: str, allow_unsafe_torch_load: bool) -> tuple[DCAE, dict]:
    if not allow_unsafe_torch_load:
        raise SystemExit(
            "DCAE checkpoints are PyTorch pickle archives. Re-run with "
            "--allow-unsafe-torch-load only after verifying source and SHA256."
        )
    model = DCAE().to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.update(force=True)
    return model, checkpoint


def read_images(images: list[Path] | None, max_images: int | None) -> list[Path]:
    paths = images if images else [path for path in DEFAULT_IMAGES if path.is_file()]
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise SystemExit("no smoke images found")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing images: {missing[:5]}")
    return paths


def run_one(model: DCAE, image_path: Path, output_dir: Path, device: str, index: int) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_padded, padding = pad_to_multiple(x)

    stream_path = output_dir / "streams" / f"{index:03d}_{image_path.stem}.dcae.bin"
    rec_path = output_dir / "reconstructions" / f"{index:03d}_{image_path.stem}_rec.png"
    gt_path = output_dir / "reconstructions" / f"{index:03d}_{image_path.stem}_gt.png"

    if device == "cuda":
        torch.cuda.synchronize()
    encode_start = perf_counter()
    with torch.no_grad():
        out_enc = model.compress(x_padded)
    if device == "cuda":
        torch.cuda.synchronize()
    encode_seconds = perf_counter() - encode_start

    stream_bytes = write_stream(stream_path, height, width, out_enc["strings"])
    stream = read_stream(stream_path)

    if device == "cuda":
        torch.cuda.synchronize()
    decode_start = perf_counter()
    with torch.no_grad():
        out_dec = model.decompress(stream["strings"], stream["shape"])
    if device == "cuda":
        torch.cuda.synchronize()
    decode_seconds = perf_counter() - decode_start

    x_hat = crop(out_dec["x_hat"], stream["padding"]).clamp(0, 1)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(gt_path)
    ToPILImage()(x_hat.squeeze(0).detach().cpu()).save(rec_path)

    return {
        "image": project_path(image_path),
        "width": width,
        "height": height,
        "padding_left_right_top_bottom": list(padding),
        "actual_total_bytes": stream_bytes,
        "actual_total_bpp": 8.0 * stream_bytes / (height * width),
        "stream_path": project_path(stream_path),
        "reconstruction_path": project_path(rec_path),
        "psnr_db": compute_psnr(x, x_hat),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "bitstream_format": "upstream_dcae_bin_h16w16_len32_y_len32_z",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="*")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--allow-unsafe-torch-load", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    torch.backends.cudnn.enabled = False

    images = read_images(args.images, args.max_images)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device, args.allow_unsafe_torch_load)

    rows = [run_one(model, image_path, output_dir, device, index) for index, image_path in enumerate(images)]
    summary = {
        "checkpoint": project_path(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "images": len(rows),
        "avg_actual_total_bpp": sum(row["actual_total_bpp"] for row in rows) / len(rows),
        "avg_psnr_db": sum(row["psnr_db"] for row in rows) / len(rows),
    }

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "results_path": project_path(results_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
