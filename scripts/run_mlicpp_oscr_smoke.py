#!/usr/bin/env python3
"""Package frozen-base MLIC++ streams in the OSCAR `.oscr` container."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from math import floor
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MLIC_ROOT))

from oscarlic.bitstream import Section, read_container, write_container  # noqa: E402
from oscarlic.candidates import LatentCandidateLayout, decode_gate_payload, encode_gate_payload  # noqa: E402
from config.config import model_config  # noqa: E402
from models.mlicpp import MLICPlusPlus  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402
from utils.utils import read_body, torch2img, write_body  # noqa: E402


SECTION_BASE_MAIN = 2
SECTION_TEXT_GATE = 10
SECTION_TEXT_MAIN = 12
CODEC_MLICPP_UPSTREAM_BODY = 1001
CODEC_OSCAR_GATE_V0 = 1
CODEC_OSCAR_DUMMY_ENHANCEMENT = 0
MODEL_ID_MLICPP_CORRECTED = 1001
MODEL_VERSION_CORRECTED_0025 = 1


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


def read_image_list(split: Path | None, images: list[Path] | None, max_images: int | None) -> list[Path]:
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


def serialize_mlicpp_body(shape, strings) -> bytes:
    handle = BytesIO()
    write_body(handle, shape, strings)
    return handle.getvalue()


def parse_mlicpp_body(payload: bytes):
    return read_body(BytesIO(payload))


def deterministic_bytes(length: int) -> bytes:
    return bytes(((index * 31 + 17) % 251 for index in range(length)))


def evenly_spaced_indices(total: int, selected: int) -> list[int]:
    if selected <= 0:
        return []
    if selected >= total:
        return list(range(total))
    return sorted({min(total - 1, floor(index * total / selected)) for index in range(selected)})


def compute_psnr_only(rec: Image.Image, gt: Image.Image) -> float:
    rec_t = torch.from_numpy(np.asarray(rec).copy()).float()
    gt_t = torch.from_numpy(np.asarray(gt).copy()).float()
    mse = torch.mean((rec_t - gt_t) ** 2).item()
    return 20.0 * torch.log10(torch.tensor(255.0)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()


def run_one(
    model: MLICPlusPlus,
    image_path: Path,
    output_dir: Path,
    device: str,
    index: int,
    *,
    with_dummy_text_sections: bool,
    candidate_tile: tuple[int, int],
    candidate_channel_group_size: int | None,
    selected_fraction: float,
    enhancement_global_bytes: int,
    enhancement_bytes_per_selected: int,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = pad_to_multiple_64(x)

    stream_path = output_dir / "streams" / f"{index:03d}_{image_path.stem}.oscr"
    rec_path = output_dir / "reconstructions" / f"{index:03d}_{image_path.stem}_rec.png"
    gt_path = output_dir / "reconstructions" / f"{index:03d}_{image_path.stem}_gt.png"

    with torch.no_grad():
        compressed = model.compress(x_pad)
    base_payload = serialize_mlicpp_body(compressed["shape"], compressed["strings"])
    sections = [
        Section(
            section_type=SECTION_BASE_MAIN,
            codec=CODEC_MLICPP_UPSTREAM_BODY,
            flags=0,
            payload=base_payload,
        )
    ]
    text_section_info = {
        "candidate_count": 0,
        "selected_candidate_count": 0,
        "gate_payload_bytes": 0,
        "enhancement_payload_bytes": 0,
        "text_payload_bytes": 0,
        "gate_mode": None,
        "candidate_tile_height": None,
        "candidate_tile_width": None,
        "candidate_channel_group_size": None,
    }
    if with_dummy_text_sections:
        latent_height = x_pad.shape[-2] // 16
        latent_width = x_pad.shape[-1] // 16
        layout = LatentCandidateLayout(
            latent_height=latent_height,
            latent_width=latent_width,
            slice_num=10,
            slice_channels=32,
            tile_height=candidate_tile[0],
            tile_width=candidate_tile[1],
            channel_group_size=candidate_channel_group_size,
        )
        selected_count = round(layout.candidate_count * selected_fraction)
        if selected_fraction > 0 and selected_count == 0:
            selected_count = 1
        selected_count = max(0, min(layout.candidate_count, selected_count))
        selected_indices = evenly_spaced_indices(layout.candidate_count, selected_count)
        gate_payload = encode_gate_payload(
            num_candidates=layout.candidate_count,
            selected_indices=selected_indices,
            layout_version=1,
            probability_model_id=0,
        )
        gate_decoded = decode_gate_payload(gate_payload)
        enhancement_payload_len = enhancement_global_bytes + enhancement_bytes_per_selected * len(selected_indices)
        enhancement_payload = deterministic_bytes(enhancement_payload_len)
        sections.extend(
            [
                Section(
                    section_type=SECTION_TEXT_GATE,
                    codec=CODEC_OSCAR_GATE_V0,
                    flags=0,
                    payload=gate_payload,
                    dependency_mask=1,
                ),
                Section(
                    section_type=SECTION_TEXT_MAIN,
                    codec=CODEC_OSCAR_DUMMY_ENHANCEMENT,
                    flags=0,
                    payload=enhancement_payload,
                    dependency_mask=1,
                ),
            ]
        )
        text_section_info = {
            "candidate_count": layout.candidate_count,
            "selected_candidate_count": len(selected_indices),
            "gate_payload_bytes": len(gate_payload),
            "enhancement_payload_bytes": len(enhancement_payload),
            "text_payload_bytes": len(gate_payload) + len(enhancement_payload),
            "gate_mode": gate_decoded["mode"],
            "candidate_tile_height": candidate_tile[0],
            "candidate_tile_width": candidate_tile[1],
            "candidate_channel_group_size": candidate_channel_group_size,
        }

    write_container(
        stream_path,
        width=width,
        height=height,
        model_id=MODEL_ID_MLICPP_CORRECTED,
        model_version=MODEL_VERSION_CORRECTED_0025,
        sections=sections,
    )

    parsed = read_container(stream_path)
    base_sections = [section for section in parsed.sections if section.section_type == SECTION_BASE_MAIN]
    if len(base_sections) != 1:
        raise ValueError(f"expected one BASE_MAIN section, found {len(base_sections)}")
    if base_sections[0].codec != CODEC_MLICPP_UPSTREAM_BODY:
        raise ValueError(f"unexpected BASE_MAIN codec: {base_sections[0].codec}")

    model.update_resolutions(16, 16)
    strings, shape = parse_mlicpp_body(base_sections[0].payload)
    if device == "cuda":
        torch.cuda.synchronize()
    decode_start = perf_counter()
    with torch.no_grad():
        decoded = model.decompress(strings, shape)
    if device == "cuda":
        torch.cuda.synchronize()
    decode_seconds_wall = perf_counter() - decode_start
    x_hat = decoded["x_hat"][:, :, :height, :width]

    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec = torch2img(x_hat.detach().cpu())
    image.save(gt_path)
    rec.save(rec_path)
    try:
        psnr, msssim = compute_metrics(rec, image)
    except AssertionError:
        psnr, msssim = compute_psnr_only(rec, image), None

    total_bytes = stream_path.stat().st_size
    payload_bytes_total = sum(section.length for section in parsed.sections)
    structural_overhead_bytes = total_bytes - payload_bytes_total
    return {
        "image": project_path(image_path),
        "width": width,
        "height": height,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "actual_total_bytes": total_bytes,
        "actual_total_bpp": 8.0 * total_bytes / (width * height),
        "base_payload_bytes": len(base_payload),
        "base_payload_bpp": 8.0 * len(base_payload) / (width * height),
        "base_payload_sha256": sha256(base_payload).hexdigest(),
        "optional_payload_bytes": text_section_info["text_payload_bytes"],
        "optional_payload_bpp": 8.0 * text_section_info["text_payload_bytes"] / (width * height),
        "structural_overhead_bytes": structural_overhead_bytes,
        "structural_overhead_bpp": 8.0 * structural_overhead_bytes / (width * height),
        "non_base_total_bytes": total_bytes - len(base_payload),
        "non_base_total_bpp": 8.0 * (total_bytes - len(base_payload)) / (width * height),
        "oscr_overhead_bytes": total_bytes - len(base_payload),
        "oscr_overhead_bpp": 8.0 * (total_bytes - len(base_payload)) / (width * height),
        "stream_path": project_path(stream_path),
        "reconstruction_path": project_path(rec_path),
        "psnr_db": psnr,
        "ms_ssim": msssim,
        "encode_seconds_model_reported": float(compressed["cost_time"]),
        "decode_seconds_model_reported": float(decoded["cost_time"]),
        "decode_seconds_wall": decode_seconds_wall,
        "bitstream_format": "oscr_v0_1_single_base_main_mlicpp_upstream_body",
        "section_count": len(parsed.sections),
        "model_id": parsed.model_id,
        "model_version": parsed.model_version,
        **text_section_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--images", type=Path, nargs="*")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--with-dummy-text-sections", action="store_true")
    parser.add_argument("--candidate-tile", type=int, nargs=2, default=(4, 4), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--candidate-channel-group-size", type=int)
    parser.add_argument("--selected-fraction", type=float, default=0.10)
    parser.add_argument("--enhancement-global-bytes", type=int, default=0)
    parser.add_argument("--enhancement-bytes-per-selected", type=int, default=0)
    args = parser.parse_args()

    if args.selected_fraction < 0 or args.selected_fraction > 1:
        raise SystemExit("--selected-fraction must be in [0, 1]")
    if args.enhancement_global_bytes < 0 or args.enhancement_bytes_per_selected < 0:
        raise SystemExit("enhancement byte counts must be nonnegative")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    image_paths = read_image_list(args.split, args.images, args.max_images)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)

    rows = []
    for index, image_path in enumerate(image_paths):
        rows.append(
            run_one(
                model,
                image_path,
                output_dir,
                device,
                index,
                with_dummy_text_sections=args.with_dummy_text_sections,
                candidate_tile=tuple(args.candidate_tile),
                candidate_channel_group_size=args.candidate_channel_group_size,
                selected_fraction=args.selected_fraction,
                enhancement_global_bytes=args.enhancement_global_bytes,
                enhancement_bytes_per_selected=args.enhancement_bytes_per_selected,
            )
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
        "avg_actual_total_bpp": sum(row["actual_total_bpp"] for row in rows) / len(rows),
        "avg_base_payload_bpp": sum(row["base_payload_bpp"] for row in rows) / len(rows),
        "avg_optional_payload_bpp": sum(row["optional_payload_bpp"] for row in rows) / len(rows),
        "avg_non_base_total_bytes": sum(row["non_base_total_bytes"] for row in rows) / len(rows),
        "avg_non_base_total_bpp": sum(row["non_base_total_bpp"] for row in rows) / len(rows),
        "avg_oscr_overhead_bytes": sum(row["oscr_overhead_bytes"] for row in rows) / len(rows),
        "avg_structural_overhead_bytes": sum(row["structural_overhead_bytes"] for row in rows) / len(rows),
        "avg_gate_payload_bytes": sum(row["gate_payload_bytes"] for row in rows) / len(rows),
        "avg_enhancement_payload_bytes": sum(row["enhancement_payload_bytes"] for row in rows) / len(rows),
        "avg_psnr_db": sum(row["psnr_db"] for row in rows) / len(rows),
        "avg_ms_ssim": (
            sum(row["ms_ssim"] for row in rows if row["ms_ssim"] is not None)
            / sum(1 for row in rows if row["ms_ssim"] is not None)
            if any(row["ms_ssim"] is not None for row in rows)
            else None
        ),
        "bitstream_format": "oscr_v0_1_single_base_main_mlicpp_upstream_body",
        "with_dummy_text_sections": args.with_dummy_text_sections,
        "candidate_tile": list(args.candidate_tile),
        "candidate_channel_group_size": args.candidate_channel_group_size,
        "selected_fraction": args.selected_fraction,
        "enhancement_global_bytes": args.enhancement_global_bytes,
        "enhancement_bytes_per_selected": args.enhancement_bytes_per_selected,
    }

    results_path = output_dir / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "results_path": str(results_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
