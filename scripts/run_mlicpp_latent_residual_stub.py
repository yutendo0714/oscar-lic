#!/usr/bin/env python3
"""Counted frozen-base MLIC++ latent residual enhancement stub."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
MLIC_ROOT = ROOT / "external/mlic/MLIC++"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MLIC_ROOT))

from inspect_mlicpp_decoded_yhat import (  # noqa: E402
    decompress_with_y_hat,
    load_model,
    pad_to_multiple_64,
    project_path,
    read_image_list,
)
from oscarlic.bitstream import Section, read_container, write_container  # noqa: E402
from oscarlic.candidates import LatentCandidateLayout, decode_gate_payload, encode_gate_payload  # noqa: E402
from utils.utils import read_body, torch2img, write_body  # noqa: E402


SECTION_BASE_MAIN = 2
SECTION_TEXT_GATE = 10
SECTION_TEXT_MAIN = 12
CODEC_MLICPP_UPSTREAM_BODY = 1001
CODEC_OSCAR_GATE_V0 = 1
CODEC_OSCAR_LATENT_RESIDUAL_V0 = 2
MODEL_ID_MLICPP_CORRECTED = 1001
MODEL_VERSION_CORRECTED_0025 = 1
RESIDUAL_HEADER = struct.Struct("<Hf")


def serialize_mlicpp_body(shape, strings) -> bytes:
    handle = BytesIO()
    write_body(handle, shape, strings)
    return handle.getvalue()


def parse_mlicpp_body(payload: bytes):
    return read_body(BytesIO(payload))


def psnr_from_tensors(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    mse = F.mse_loss(x_hat.clamp(0, 1), x).item()
    if mse == 0:
        return float("inf")
    return float(-10.0 * torch.log10(torch.tensor(mse)).item())


def candidate_global_channels(layout: LatentCandidateLayout, candidate) -> tuple[int, int]:
    base = candidate.slice_index * layout.slice_channels
    return base + candidate.channel0, base + candidate.channel1


def select_candidates_by_residual(
    residual: torch.Tensor,
    layout: LatentCandidateLayout,
    selected_fraction: float,
) -> list[int]:
    rows = []
    for candidate in layout.iter_candidates():
        c0, c1 = candidate_global_channels(layout, candidate)
        value = residual[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1]
        rows.append((float(value.abs().mean().item()), candidate.index))
    count = round(layout.candidate_count * selected_fraction)
    if selected_fraction > 0 and count == 0:
        count = 1
    count = max(0, min(layout.candidate_count, count))
    return sorted(index for _, index in sorted(rows, reverse=True)[:count])


def read_candidate_selection(path: Path | None, mode: str) -> dict[int, list[int]] | None:
    if path is None:
        return None
    if mode != "explicit":
        raise ValueError(f"unsupported candidate selection mode: {mode}")
    selected: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "selected_by_policy" in row and not bool(row["selected_by_policy"]):
                continue
            source_index = int(row["source_index"])
            selected.setdefault(source_index, []).append(int(row["candidate_index"]))
    return {source_index: sorted(set(indices)) for source_index, indices in selected.items()}


def encode_residual_payload(
    residual: torch.Tensor,
    layout: LatentCandidateLayout,
    selected_indices: list[int],
    quant_step: float,
) -> bytes:
    candidates = list(layout.iter_candidates())
    payload = bytearray(RESIDUAL_HEADER.pack(1, quant_step))
    for index in selected_indices:
        candidate = candidates[index]
        c0, c1 = candidate_global_channels(layout, candidate)
        value = residual[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1]
        quantized = torch.round(value / quant_step).clamp(-127, 127).to(torch.int8).detach().cpu().numpy()
        payload.extend(quantized.tobytes(order="C"))
    return bytes(payload)


def apply_residual_payload(
    y_hat: torch.Tensor,
    layout: LatentCandidateLayout,
    selected_indices: list[int],
    payload: bytes,
) -> torch.Tensor:
    if len(payload) < RESIDUAL_HEADER.size:
        raise ValueError("residual payload is shorter than header")
    version, quant_step = RESIDUAL_HEADER.unpack_from(payload, 0)
    if version != 1:
        raise ValueError(f"unsupported residual payload version {version}")
    offset = RESIDUAL_HEADER.size
    candidates = list(layout.iter_candidates())
    y_enhanced = y_hat.clone()
    for index in selected_indices:
        candidate = candidates[index]
        c0, c1 = candidate_global_channels(layout, candidate)
        shape = (1, c1 - c0, candidate.y1 - candidate.y0, candidate.x1 - candidate.x0)
        value_count = int(np.prod(shape))
        end = offset + value_count
        if end > len(payload):
            raise ValueError("residual payload ended inside a candidate tensor")
        raw = np.frombuffer(payload[offset:end], dtype=np.int8).copy()
        delta = torch.from_numpy(raw).to(device=y_hat.device, dtype=y_hat.dtype).reshape(shape) * float(quant_step)
        y_enhanced[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1] += delta
        offset = end
    if offset != len(payload):
        raise ValueError("residual payload has trailing bytes")
    return y_enhanced


def run_one(model, image_path: Path, output_dir: Path, device: str, index: int, args: argparse.Namespace) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = pad_to_multiple_64(x)

    with torch.no_grad():
        compressed = model.compress(x_pad)
        base_decoded = decompress_with_y_hat(model, compressed["strings"], compressed["shape"])
        y_target = model.g_a(x_pad)
        residual = y_target - base_decoded["y_hat"]
        residual_payload_source = residual * args.residual_scale

    _, channels, latent_height, latent_width = residual.shape
    layout = LatentCandidateLayout(
        latent_height=latent_height,
        latent_width=latent_width,
        slice_num=10,
        slice_channels=channels // 10,
        tile_height=args.tile,
        tile_width=args.tile,
        channel_group_size=args.channel_group_size,
    )
    selected_indices = (
        args.selection_map.get(index, [])
        if args.selection_map is not None
        else select_candidates_by_residual(residual, layout, args.selected_fraction)
    )
    gate_payload = encode_gate_payload(num_candidates=layout.candidate_count, selected_indices=selected_indices)
    residual_payload = encode_residual_payload(residual_payload_source, layout, selected_indices, args.quant_step)
    base_payload = serialize_mlicpp_body(compressed["shape"], compressed["strings"])

    stream_path = output_dir / "streams" / f"{index:03d}_{image_path.stem}.oscr"
    write_container(
        stream_path,
        width=width,
        height=height,
        model_id=MODEL_ID_MLICPP_CORRECTED,
        model_version=MODEL_VERSION_CORRECTED_0025,
        flags=1,
        sections=[
            Section(SECTION_BASE_MAIN, CODEC_MLICPP_UPSTREAM_BODY, 0, base_payload),
            Section(SECTION_TEXT_GATE, CODEC_OSCAR_GATE_V0, 0, gate_payload, dependency_mask=1),
            Section(SECTION_TEXT_MAIN, CODEC_OSCAR_LATENT_RESIDUAL_V0, 0, residual_payload, dependency_mask=1),
        ],
    )

    parsed = read_container(stream_path)
    base_sections = [section for section in parsed.sections if section.section_type == SECTION_BASE_MAIN]
    gate_sections = [section for section in parsed.sections if section.section_type == SECTION_TEXT_GATE]
    main_sections = [section for section in parsed.sections if section.section_type == SECTION_TEXT_MAIN]
    if len(base_sections) != 1 or len(gate_sections) != 1 or len(main_sections) != 1:
        raise ValueError("expected one base, gate and main section")

    strings, shape = parse_mlicpp_body(base_sections[0].payload)
    with torch.no_grad():
        decoded = decompress_with_y_hat(model, strings, shape)
        decoded_gate = decode_gate_payload(gate_sections[0].payload)
        y_enhanced = apply_residual_payload(decoded["y_hat"], layout, decoded_gate["selected_indices"], main_sections[0].payload)
        x_enhanced = model.g_s(y_enhanced)

    x_base = decoded["x_hat"][:, :, :height, :width]
    x_enhanced = x_enhanced[:, :, :height, :width]
    rec_dir = output_dir / "reconstructions"
    rec_dir.mkdir(parents=True, exist_ok=True)
    base_rec_path = rec_dir / f"{index:03d}_{image_path.stem}_base.png"
    enhanced_rec_path = rec_dir / f"{index:03d}_{image_path.stem}_enhanced.png"
    torch2img(x_base.detach().cpu()).save(base_rec_path)
    torch2img(x_enhanced.detach().cpu()).save(enhanced_rec_path)

    total_bytes = stream_path.stat().st_size
    payload_bytes_total = sum(section.length for section in parsed.sections)
    return {
        "index": index,
        "image": project_path(image_path),
        "reference": getattr(args, "reference", None),
        "width": width,
        "height": height,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "tile": args.tile,
        "channel_group_size": args.channel_group_size,
        "candidate_count": layout.candidate_count,
        "selected_candidate_count": len(selected_indices),
        "selected_candidate_indices": selected_indices,
        "selected_fraction": args.selected_fraction,
        "selection_mode": args.candidate_selection_mode if args.candidate_selection_table else "residual_fraction",
        "candidate_selection_table": (
            project_path(args.candidate_selection_table) if args.candidate_selection_table else None
        ),
        "quant_step": args.quant_step,
        "residual_scale": args.residual_scale,
        "base_payload_bytes": len(base_payload),
        "gate_payload_bytes": len(gate_payload),
        "residual_payload_bytes": len(residual_payload),
        "structural_overhead_bytes": total_bytes - payload_bytes_total,
        "actual_total_bytes": total_bytes,
        "actual_total_bpp": 8.0 * total_bytes / (width * height),
        "base_payload_bpp": 8.0 * len(base_payload) / (width * height),
        "enhancement_payload_bpp": 8.0 * (len(gate_payload) + len(residual_payload)) / (width * height),
        "psnr_base_db": psnr_from_tensors(x_base, x),
        "psnr_enhanced_db": psnr_from_tensors(x_enhanced, x),
        "psnr_delta_db": psnr_from_tensors(x_enhanced, x) - psnr_from_tensors(x_base, x),
        "stream_path": project_path(stream_path),
        "base_reconstruction_path": project_path(base_rec_path),
        "enhanced_reconstruction_path": project_path(enhanced_rec_path),
        "reconstruction_path": project_path(enhanced_rec_path),
        "psnr_db": psnr_from_tensors(x_enhanced, x),
        "gate_mode": decoded_gate["mode"],
        "bitstream_format": "oscr_v0_1_mlicpp_base_plus_latent_residual_stub",
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
    parser.add_argument("--selected-fraction", type=float, default=0.10)
    parser.add_argument("--candidate-selection-table", type=Path)
    parser.add_argument("--candidate-selection-mode", choices=["explicit"], default="explicit")
    parser.add_argument("--quant-step", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device != "cuda":
        raise SystemExit("MLIC++ compress/decompress uses torch.cuda.synchronize(); run with --cuda on a visible GPU")
    if args.tile <= 0:
        raise SystemExit("--tile must be positive")
    if not 0 <= args.selected_fraction <= 1:
        raise SystemExit("--selected-fraction must be in [0, 1]")
    if args.quant_step <= 0:
        raise SystemExit("--quant-step must be positive")
    if args.residual_scale <= 0:
        raise SystemExit("--residual-scale must be positive")

    image_paths = read_image_list(args.split, args.images, args.max_images)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)
    args.selection_map = read_candidate_selection(args.candidate_selection_table, args.candidate_selection_mode)

    rows = [run_one(model, image_path, output_dir, device, index, args) for index, image_path in enumerate(image_paths)]
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
        "selected_fraction": args.selected_fraction,
        "candidate_selection_table": str(args.candidate_selection_table) if args.candidate_selection_table else None,
        "selection_mode": args.candidate_selection_mode if args.candidate_selection_table else "residual_fraction",
        "quant_step": args.quant_step,
        "residual_scale": args.residual_scale,
        "avg_actual_total_bpp": sum(row["actual_total_bpp"] for row in rows) / len(rows),
        "avg_base_payload_bpp": sum(row["base_payload_bpp"] for row in rows) / len(rows),
        "avg_enhancement_payload_bpp": sum(row["enhancement_payload_bpp"] for row in rows) / len(rows),
        "avg_selected_candidate_count": sum(row["selected_candidate_count"] for row in rows) / len(rows),
        "avg_gate_payload_bytes": sum(row["gate_payload_bytes"] for row in rows) / len(rows),
        "avg_residual_payload_bytes": sum(row["residual_payload_bytes"] for row in rows) / len(rows),
        "avg_psnr_base_db": sum(row["psnr_base_db"] for row in rows) / len(rows),
        "avg_psnr_enhanced_db": sum(row["psnr_enhanced_db"] for row in rows) / len(rows),
        "avg_psnr_delta_db": sum(row["psnr_delta_db"] for row in rows) / len(rows),
        "bitstream_format": "oscr_v0_1_mlicpp_base_plus_latent_residual_stub",
    }
    results_path = output_dir / "results.jsonl"
    results_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "results_path": str(results_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
