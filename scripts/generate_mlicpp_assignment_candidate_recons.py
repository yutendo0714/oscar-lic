#!/usr/bin/env python3
"""Generate single-code assignment alternatives for selected MLIC++ candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
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
from run_mlicpp_latent_codebook_stub import (  # noqa: E402
    apply_codebook,
    assign_codes,
    candidate_vector,
    quantize_residual_for_codebook,
    read_candidate_selection,
)
from utils.utils import torch2img  # noqa: E402


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_image_path(split_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    split_relative = (split_path.parent / path).resolve()
    if split_relative.is_file():
        return split_relative
    return (ROOT / path).resolve()


def read_records(path: Path, max_images: int | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = resolve_image_path(path, row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_no}: missing image {image_path}")
            row["image_path"] = image_path
            row["text"] = str(row["text"])
            records.append(row)
            if max_images is not None and len(records) >= max_images:
                break
    if not records:
        raise ValueError(f"empty split: {path}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--candidate-selection-table", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-split", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--tile", type=int, default=4)
    parser.add_argument("--channel-group-size", type=int)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--codebook-quant-step", type=float)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--include-nearest", action="store_true")
    args = parser.parse_args()

    if args.topk <= 0:
        raise SystemExit("--topk must be positive")
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but unavailable")
    if device != "cuda":
        raise SystemExit("MLIC++ path expects CUDA")

    records = read_records(args.split, args.max_images)
    selection_map = read_candidate_selection(args.candidate_selection_table, "explicit") or {}
    model, checkpoint = load_model(args.checkpoint, device)
    codebook_np = np.load(args.codebook)["codebook"]
    codebook = torch.from_numpy(codebook_np).to(device=device)
    output_dir = args.output_dir.resolve()
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_index, record in enumerate(records):
        selected = selection_map.get(source_index, [])
        if not selected:
            continue
        image_path = record["image_path"]
        image = Image.open(image_path).convert("RGB")
        x = transforms.ToTensor()(image).unsqueeze(0).to(device)
        x_pad, _, _ = pad_to_multiple_64(x)
        with torch.no_grad():
            compressed = model.compress(x_pad)
            decoded = decompress_with_y_hat(model, compressed["strings"], compressed["shape"])
            residual = (model.g_a(x_pad) - decoded["y_hat"]) * args.residual_scale
            residual_for_codebook = quantize_residual_for_codebook(residual, args.codebook_quant_step)
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
        nearest_codes, _, _ = assign_codes(
            residual_for_codebook,
            layout,
            selected,
            codebook,
            None,
            args.tile,
        )
        candidates = list(layout.iter_candidates())
        vectors = torch.stack(
            [candidate_vector(residual_for_codebook, layout, candidates[index], tile=args.tile) for index in selected],
            dim=0,
        )
        distances = torch.cdist(vectors.float(), codebook.float())
        nearest_distances, _ = distances.min(dim=1)
        topk = min(args.topk, codebook.shape[0])
        topk_distances, topk_indices = distances.topk(k=topk, dim=1, largest=False)
        for slot, candidate_index in enumerate(selected):
            for rank, code_index_tensor in enumerate(topk_indices[slot]):
                code_index = int(code_index_tensor.item())
                is_nearest = code_index == int(nearest_codes[slot])
                if is_nearest and not args.include_nearest:
                    continue
                codes = list(nearest_codes)
                codes[slot] = code_index
                with torch.no_grad():
                    y_enhanced = apply_codebook(decoded["y_hat"], layout, selected, codes, codebook, args.tile)
                    x_enhanced = model.g_s(y_enhanced)[:, :, : image.height, : image.width]
                rec_path = recon_dir / (
                    f"{source_index:03d}_slot{slot:02d}_cand{candidate_index:05d}_code{code_index:03d}_"
                    f"{image_path.stem}.png"
                )
                torch2img(x_enhanced.detach().cpu()).save(rec_path)
                relative_error = float(
                    (topk_distances[slot, rank] / nearest_distances[slot].clamp_min(1e-8)).detach().cpu().item()
                )
                rows.append(
                    {
                        "image_path": project_path(rec_path),
                        "text": record["text"],
                        "source": record.get("source"),
                        "split": record.get("split"),
                        "method_id": args.method_id,
                        "source_image": project_path(image_path),
                        "source_index": source_index,
                        "candidate_index": int(candidate_index),
                        "candidate_slot": int(slot),
                        "nearest_code": int(nearest_codes[slot]),
                        "code_index": code_index,
                        "topk_rank": int(rank),
                        "is_nearest": bool(is_nearest),
                        "assignment_relative_error": relative_error,
                    }
                )
        print(json.dumps({"source_index": source_index, "selected": len(selected), "rows": len(rows)}), flush=True)

    args.output_split.parent.mkdir(parents=True, exist_ok=True)
    args.output_split.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
        "split": str(args.split),
        "candidate_selection_table": str(args.candidate_selection_table),
        "codebook": str(args.codebook),
        "output_split": str(args.output_split),
        "rows": len(rows),
        "topk": args.topk,
        "include_nearest": args.include_nearest,
        "method_id": args.method_id,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
