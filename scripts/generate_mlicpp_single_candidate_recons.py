#!/usr/bin/env python3
"""Generate one-candidate MLIC++ latent residual reconstructions for utility labels."""

from __future__ import annotations

import argparse
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

from inspect_mlicpp_decoded_yhat import decompress_with_y_hat, load_model, pad_to_multiple_64, project_path  # noqa: E402
from oscarlic.candidates import LatentCandidateLayout, encode_gate_payload  # noqa: E402
from run_mlicpp_latent_residual_stub import (  # noqa: E402
    apply_residual_payload,
    candidate_global_channels,
    encode_residual_payload,
    psnr_from_tensors,
    serialize_mlicpp_body,
)
from utils.utils import torch2img  # noqa: E402


THREE_SECTION_STRUCTURAL_BYTES = 44 + 3 * 40 + 4


def read_records(path: Path, max_images: int | None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        rows.append(row)
        if max_images is not None and len(rows) >= max_images:
            break
    if not rows:
        raise ValueError(f"empty split: {path}")
    return rows


def candidate_scores(residual: torch.Tensor, layout: LatentCandidateLayout) -> list[tuple[float, int]]:
    rows = []
    for candidate in layout.iter_candidates():
        c0, c1 = candidate_global_channels(layout, candidate)
        value = residual[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1]
        rows.append((float(value.abs().mean().item()), candidate.index))
    return sorted(rows, reverse=True)


def selected_candidate_indices(residual: torch.Tensor, layout: LatentCandidateLayout, max_candidates: int | None) -> list[int]:
    if max_candidates is None:
        return [candidate.index for candidate in layout.iter_candidates()]
    return sorted(index for _, index in candidate_scores(residual, layout)[:max_candidates])


def run_one(model, record: dict, output_dir: Path, device: str, image_index: int, args: argparse.Namespace) -> list[dict]:
    image_path = Path(record["image_path"])
    image = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = pad_to_multiple_64(x)
    with torch.no_grad():
        compressed = model.compress(x_pad)
        decoded = decompress_with_y_hat(model, compressed["strings"], compressed["shape"])
        residual = (model.g_a(x_pad) - decoded["y_hat"]) * args.residual_scale
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
    candidates = list(layout.iter_candidates())
    selected = selected_candidate_indices(residual, layout, args.max_candidates_per_image)
    base_payload_bytes = len(serialize_mlicpp_body(compressed["shape"], compressed["strings"]))
    rows = []
    rec_dir = output_dir / "images"
    rec_dir.mkdir(parents=True, exist_ok=True)
    for candidate_index in selected:
        candidate = candidates[candidate_index]
        gate_payload = encode_gate_payload(num_candidates=layout.candidate_count, selected_indices=[candidate_index])
        residual_payload = encode_residual_payload(residual, layout, [candidate_index], args.quant_step)
        with torch.no_grad():
            y_enhanced = apply_residual_payload(decoded["y_hat"], layout, [candidate_index], residual_payload)
            x_enhanced = model.g_s(y_enhanced)[:, :, : image.height, : image.width]
        rec_path = rec_dir / f"{image_index:04d}_cand{candidate_index:04d}_{image_path.stem}.png"
        torch2img(x_enhanced.detach().cpu()).save(rec_path)
        actual_total_bytes = base_payload_bytes + len(gate_payload) + len(residual_payload) + THREE_SECTION_STRUCTURAL_BYTES
        rows.append(
            {
                "image_path": project_path(rec_path),
                "text": record["text"],
                "source": record.get("source"),
                "split": record.get("split"),
                "source_image": str(image_path),
                "source_image_project": project_path(image_path),
                "source_index": image_index,
                "candidate_index": candidate_index,
                "slice_index": candidate.slice_index,
                "y0": candidate.y0,
                "y1": candidate.y1,
                "x0": candidate.x0,
                "x1": candidate.x1,
                "channel0": candidate.channel0,
                "channel1": candidate.channel1,
                "candidate_count": layout.candidate_count,
                "base_payload_bytes": base_payload_bytes,
                "gate_payload_bytes": len(gate_payload),
                "residual_payload_bytes": len(residual_payload),
                "structural_overhead_bytes": THREE_SECTION_STRUCTURAL_BYTES,
                "actual_total_bytes": actual_total_bytes,
                "actual_total_bpp": 8.0 * actual_total_bytes / (image.width * image.height),
                "psnr_enhanced_db": psnr_from_tensors(x_enhanced, x),
                "tile": args.tile,
                "quant_step": args.quant_step,
                "residual_scale": args.residual_scale,
                "pad_h": pad_h,
                "pad_w": pad_w,
            }
        )
    print(json.dumps({"image_index": image_index, "candidates": len(rows)}, ensure_ascii=False), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-candidates-per-image", type=int)
    parser.add_argument("--tile", type=int, default=4)
    parser.add_argument("--channel-group-size", type=int)
    parser.add_argument("--quant-step", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device != "cuda":
        raise SystemExit("MLIC++ compress/decompress uses torch.cuda.synchronize(); run with --cuda on a visible GPU")
    if args.max_candidates_per_image is not None and args.max_candidates_per_image <= 0:
        raise SystemExit("--max-candidates-per-image must be positive when supplied")

    records = read_records(args.split, args.max_images)
    output_dir = args.output.parent.resolve() / (args.output.stem + "_assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)
    rows = []
    for image_index, record in enumerate(records):
        rows.extend(run_one(model, record, output_dir, device, image_index, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
        "split": str(args.split),
        "images": len(records),
        "rows": len(rows),
        "tile": args.tile,
        "channel_group_size": args.channel_group_size,
        "quant_step": args.quant_step,
        "residual_scale": args.residual_scale,
        "max_candidates_per_image": args.max_candidates_per_image,
        "output": str(args.output),
        "assets_dir": str(output_dir),
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
