#!/usr/bin/env python3
"""Run CompressAI zoo models with actual serialized stream accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from time import perf_counter

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage

import compressai
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean
from pytorch_msssim import ms_ssim


ROOT = Path(__file__).resolve().parents[1]
MODEL_FACTORIES = {
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
}
MAGIC = b"CAI0"


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_image_list(split: Path | None, images: list[Path] | None, max_images: int | None) -> list[Path]:
    paths: list[Path] = []
    if split is not None:
        base = split.parent
        for line_no, line in enumerate(split.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                record = json.loads(line)
                value = record.get("image_path") or record.get("image") or record.get("path")
                if value is None:
                    raise ValueError(f"{split}:{line_no}: row does not contain an image path")
                path = Path(value)
            else:
                path = Path(line)
            if not path.is_absolute():
                split_relative = (base / path).resolve()
                project_relative = (ROOT / path).resolve()
                path = split_relative if split_relative.is_file() else project_relative
            paths.append(path)
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


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = x.shape
    pad_h = 0 if height % multiple == 0 else multiple * (height // multiple + 1) - height
    pad_w = 0 if width % multiple == 0 else multiple * (width // multiple + 1) - width
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0), pad_h, pad_w


def write_uints(fd, values) -> None:
    fd.write(struct.pack(f">{len(values)}I", *values))


def read_uints(fd, n: int):
    return struct.unpack(f">{n}I", fd.read(4 * n))


def write_bytes(fd, value: bytes) -> None:
    fd.write(struct.pack(f">{len(value)}s", value))


def read_bytes(fd, n: int) -> bytes:
    return struct.unpack(f">{n}s", fd.read(n))[0]


def flatten_strings(strings) -> list[bytes]:
    flat: list[bytes] = []
    for entry in strings:
        if isinstance(entry, (bytes, bytearray)):
            flat.append(bytes(entry))
        elif isinstance(entry, list):
            if len(entry) != 1 or not isinstance(entry[0], (bytes, bytearray)):
                raise ValueError(f"unsupported string entry shape: {type(entry)} {len(entry)}")
            flat.append(bytes(entry[0]))
        else:
            raise ValueError(f"unsupported string entry type: {type(entry)}")
    return flat


def unflatten_strings(flat: list[bytes]) -> list[list[bytes]]:
    return [[value] for value in flat]


def write_stream(path: Path, *, height: int, width: int, pad_h: int, pad_w: int, shape, strings, model_id: str, metric: str, quality: int) -> int:
    model_bytes = model_id.encode("utf-8")
    metric_bytes = metric.encode("utf-8")
    flat = flatten_strings(strings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fd:
        fd.write(MAGIC)
        write_uints(fd, (height, width, pad_h, pad_w, int(shape[0]), int(shape[1]), quality, len(model_bytes), len(metric_bytes), len(flat)))
        write_bytes(fd, model_bytes)
        write_bytes(fd, metric_bytes)
        for value in flat:
            write_uints(fd, (len(value),))
            write_bytes(fd, value)
    return path.stat().st_size


def read_stream(path: Path):
    with path.open("rb") as fd:
        magic = fd.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic in {path}")
        height, width, pad_h, pad_w, shape_h, shape_w, quality, model_len, metric_len, n_strings = read_uints(fd, 10)
        model_id = read_bytes(fd, model_len).decode("utf-8")
        metric = read_bytes(fd, metric_len).decode("utf-8")
        flat = []
        for _ in range(n_strings):
            length = read_uints(fd, 1)[0]
            flat.append(read_bytes(fd, length))
    return {
        "height": height,
        "width": width,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "shape": (shape_h, shape_w),
        "quality": quality,
        "model_id": model_id,
        "metric": metric,
        "strings": unflatten_strings(flat),
    }


def compute_metrics(rec: Image.Image, gt: Image.Image) -> tuple[float, float | None]:
    import numpy as np

    rec_np = np.asarray(rec).copy()
    gt_np = np.asarray(gt).copy()
    rec_t = torch.from_numpy(rec_np).float().unsqueeze(0).permute(0, 3, 1, 2)
    gt_t = torch.from_numpy(gt_np).float().unsqueeze(0).permute(0, 3, 1, 2)
    mse = torch.mean((rec_t - gt_t) ** 2).item()
    psnr = 20.0 * torch.log10(torch.tensor(255.0)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()
    if min(rec.size) <= 160:
        return psnr, None
    return psnr, float(ms_ssim(rec_t, gt_t, data_range=255.0).item())


def run_one(model, args, image_path: Path, output_dir: Path, device: str, index: int) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = pad_to_multiple(x)

    stream_path = output_dir / "streams" / f"{index:03d}_{image_path.stem}.{args.model}.q{args.quality}.cai"
    rec_path = output_dir / "reconstructions" / f"{index:03d}_{image_path.stem}_rec.png"
    gt_path = output_dir / "reconstructions" / f"{index:03d}_{image_path.stem}_gt.png"

    torch.cuda.synchronize() if device == "cuda" else None
    enc_start = perf_counter()
    with torch.no_grad():
        out = model.compress(x_pad)
    torch.cuda.synchronize() if device == "cuda" else None
    enc_seconds = perf_counter() - enc_start

    stream_bytes = write_stream(
        stream_path,
        height=height,
        width=width,
        pad_h=pad_h,
        pad_w=pad_w,
        shape=out["shape"],
        strings=out["strings"],
        model_id=args.model,
        metric=args.metric,
        quality=args.quality,
    )

    stream = read_stream(stream_path)
    torch.cuda.synchronize() if device == "cuda" else None
    dec_start = perf_counter()
    with torch.no_grad():
        decoded = model.decompress(stream["strings"], stream["shape"])
    torch.cuda.synchronize() if device == "cuda" else None
    dec_seconds = perf_counter() - dec_start

    x_hat = decoded["x_hat"][:, :, :height, :width].clamp(0, 1).cpu()
    rec = ToPILImage()(x_hat.squeeze(0))
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(gt_path)
    rec.save(rec_path)
    psnr, msssim = compute_metrics(rec, image)

    return {
        "image": project_path(image_path),
        "width": width,
        "height": height,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "model": args.model,
        "metric": args.metric,
        "quality": args.quality,
        "actual_total_bytes": stream_bytes,
        "actual_total_bpp": 8.0 * stream_bytes / (height * width),
        "stream_path": project_path(stream_path),
        "reconstruction_path": project_path(rec_path),
        "psnr_db": psnr,
        "ms_ssim": msssim,
        "encode_seconds": enc_seconds,
        "decode_seconds": dec_seconds,
        "bitstream_format": "oscar_compressai_smoke_v0_complete_file",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_FACTORIES), default="bmshj2018-hyperprior")
    parser.add_argument("--metric", choices=["mse", "ms-ssim"], default="mse")
    parser.add_argument("--quality", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--images", type=Path, nargs="*")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--entropy-coder", default="ans")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but unavailable")
    compressai.set_entropy_coder(args.entropy_coder)
    model = MODEL_FACTORIES[args.model](quality=args.quality, metric=args.metric, pretrained=True).to(device).eval()
    model.update(force=True)

    images = read_image_list(args.split, args.images, args.max_images)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_one(model, args, image, output_dir, device, index) for index, image in enumerate(images)]
    summary = {
        "model": args.model,
        "metric": args.metric,
        "quality": args.quality,
        "entropy_coder": args.entropy_coder,
        "device": device,
        "torch": torch.__version__,
        "compressai": compressai.__version__,
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

    (output_dir / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "results_path": str(output_dir / "results.jsonl")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
