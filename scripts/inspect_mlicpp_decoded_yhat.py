#!/usr/bin/env python3
"""Decode MLIC++ streams while exposing reconstructed y_hat candidate stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
MLIC_ROOT = ROOT / "external/mlic/MLIC++"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MLIC_ROOT))

from compressai.ans import RansDecoder  # noqa: E402
from oscarlic.candidates import LatentCandidateLayout  # noqa: E402
from config.config import model_config  # noqa: E402
from models.mlicpp import MLICPlusPlus  # noqa: E402
from utils.ckbd import (  # noqa: E402
    ckbd_anchor,
    ckbd_nonanchor,
    decompress_anchor,
    decompress_nonanchor,
)


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


def decompress_with_y_hat(model: MLICPlusPlus, strings, shape) -> dict:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    y_strings = strings[0][0]
    z_strings = strings[1]
    z_hat = model.entropy_bottleneck.decompress(z_strings, shape)
    model.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
    hyper_params = model.h_s(z_hat)
    hyper_scales, hyper_means = hyper_params.chunk(2, 1)
    y_hat_slices = []

    cdf = model.gaussian_conditional.quantized_cdf.tolist()
    cdf_lengths = model.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
    offsets = model.gaussian_conditional.offset.reshape(-1).int().tolist()
    decoder = RansDecoder()
    decoder.set_stream(y_strings)

    for idx in range(model.slice_num):
        if idx == 0:
            params_anchor = model.entropy_parameters_anchor[idx](hyper_params)
            scales_anchor, means_anchor = params_anchor.chunk(2, 1)
            scales_anchor = ckbd_anchor(scales_anchor)
            means_anchor = ckbd_anchor(means_anchor)
            slice_anchor = decompress_anchor(model.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
            lrp_anchor = model.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
            slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

            local_ctx = model.local_context[idx](slice_anchor)
            params_nonanchor = model.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
            scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
            scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
            means_nonanchor = ckbd_nonanchor(means_nonanchor)
            slice_nonanchor = decompress_nonanchor(model.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
            lrp_nonanchor = model.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
            slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
            y_hat_slices.append(slice_nonanchor + slice_anchor)
        else:
            global_inter_ctx = model.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
            channel_ctx = model.channel_context[idx](torch.cat(y_hat_slices, dim=1))
            params_anchor = model.entropy_parameters_anchor[idx](torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1))
            scales_anchor, means_anchor = params_anchor.chunk(2, 1)
            scales_anchor = ckbd_anchor(scales_anchor)
            means_anchor = ckbd_anchor(means_anchor)
            slice_anchor = decompress_anchor(model.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
            lrp_anchor = model.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
            slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

            global_intra_ctx = model.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
            local_ctx = model.local_context[idx](slice_anchor)
            params_nonanchor = model.entropy_parameters_nonanchor[idx](
                torch.cat([local_ctx, global_intra_ctx, global_inter_ctx, channel_ctx, hyper_params], dim=1)
            )
            scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
            scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
            means_nonanchor = ckbd_nonanchor(means_nonanchor)
            slice_nonanchor = decompress_nonanchor(model.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
            lrp_nonanchor = model.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
            slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
            y_hat_slices.append(slice_nonanchor + slice_anchor)

    y_hat = torch.cat(y_hat_slices, dim=1)
    x_hat = model.g_s(y_hat)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"x_hat": x_hat, "y_hat": y_hat, "cost_time": time.time() - start_time}


def candidate_stats(y_hat: torch.Tensor, *, tile: int, channel_group_size: int | None, top_k: int) -> dict:
    _, channels, latent_height, latent_width = y_hat.shape
    layout = LatentCandidateLayout(
        latent_height=latent_height,
        latent_width=latent_width,
        slice_num=10,
        slice_channels=channels // 10,
        tile_height=tile,
        tile_width=tile,
        channel_group_size=channel_group_size,
    )
    rows = []
    for candidate in layout.iter_candidates():
        value = y_hat[
            :,
            candidate.slice_index * layout.slice_channels + candidate.channel0:
            candidate.slice_index * layout.slice_channels + candidate.channel1,
            candidate.y0:candidate.y1,
            candidate.x0:candidate.x1,
        ]
        rows.append(
            {
                "candidate_index": candidate.index,
                "slice_index": candidate.slice_index,
                "y0": candidate.y0,
                "y1": candidate.y1,
                "x0": candidate.x0,
                "x1": candidate.x1,
                "channel0": candidate.channel0,
                "channel1": candidate.channel1,
                "mean_abs_y_hat": float(value.abs().mean().item()),
                "rms_y_hat": float(torch.sqrt(torch.mean(value.float() ** 2)).item()),
            }
        )
    top = sorted(rows, key=lambda row: row["rms_y_hat"], reverse=True)[:top_k]
    return {
        "latent_shape": list(y_hat.shape),
        "tile": tile,
        "channel_group_size": channel_group_size,
        "candidate_count": layout.candidate_count,
        "top_candidates_by_rms": top,
        "mean_candidate_rms": float(sum(row["rms_y_hat"] for row in rows) / len(rows)),
        "max_candidate_rms": float(top[0]["rms_y_hat"]) if top else None,
    }


def run_one(model: MLICPlusPlus, image_path: Path, device: str, index: int, args: argparse.Namespace) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = pad_to_multiple_64(x)

    with torch.no_grad():
        compressed = model.compress(x_pad)
        decoded_reference = model.decompress(compressed["strings"], compressed["shape"])
        decoded_exposed = decompress_with_y_hat(model, compressed["strings"], compressed["shape"])

    x_ref = decoded_reference["x_hat"]
    x_exposed = decoded_exposed["x_hat"]
    max_abs_diff = float((x_ref - x_exposed).abs().max().item())
    stats = candidate_stats(
        decoded_exposed["y_hat"],
        tile=args.tile,
        channel_group_size=args.channel_group_size,
        top_k=args.top_k,
    )
    return {
        "index": index,
        "image": project_path(image_path),
        "width": width,
        "height": height,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "latent_y_shape": stats["latent_shape"],
        "candidate_count": stats["candidate_count"],
        "tile": stats["tile"],
        "channel_group_size": stats["channel_group_size"],
        "top_candidates_by_rms": stats["top_candidates_by_rms"],
        "mean_candidate_rms": stats["mean_candidate_rms"],
        "max_candidate_rms": stats["max_candidate_rms"],
        "decode_x_hat_max_abs_diff_vs_upstream": max_abs_diff,
        "decode_seconds_upstream": float(decoded_reference["cost_time"]),
        "decode_seconds_exposed_y_hat": float(decoded_exposed["cost_time"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--images", type=Path, nargs="*")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--tile", type=int, default=4)
    parser.add_argument("--channel-group-size", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device != "cuda":
        raise SystemExit("MLIC++ compress/decompress uses torch.cuda.synchronize(); run with --cuda on a visible GPU")
    if args.tile <= 0 or args.top_k <= 0:
        raise SystemExit("--tile and --top-k must be positive")

    image_paths = read_image_list(args.split, args.images, args.max_images)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)

    rows = [run_one(model, image_path, device, index, args) for index, image_path in enumerate(image_paths)]
    max_diff = max(row["decode_x_hat_max_abs_diff_vs_upstream"] for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "images": len(rows),
        "tile": args.tile,
        "channel_group_size": args.channel_group_size,
        "max_decode_x_hat_abs_diff_vs_upstream": max_diff,
        "avg_candidate_count": sum(row["candidate_count"] for row in rows) / len(rows),
        "avg_mean_candidate_rms": sum(row["mean_candidate_rms"] for row in rows) / len(rows),
        "avg_max_candidate_rms": sum(row["max_candidate_rms"] for row in rows) / len(rows),
    }
    (output_dir / "decoded_yhat_stats.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
