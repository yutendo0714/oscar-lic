#!/usr/bin/env python3
"""Augment top-k assignment rows with candidate residual-vector features."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from run_mlicpp_latent_codebook_stub import candidate_vector, quantize_residual_for_codebook  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_codebook(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["codebook"], dtype=np.float32)


def row_seed(row: dict) -> int:
    return int(row.get("real_seed", row["seed"]))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((torch.dot(a, b) / denom).detach().cpu().item())


def vector_stats(prefix: str, value: torch.Tensor) -> dict:
    value = value.float().reshape(-1)
    return {
        f"{prefix}_mean": float(value.mean().detach().cpu().item()),
        f"{prefix}_std": float(value.std(unbiased=False).detach().cpu().item()),
        f"{prefix}_abs_mean": float(value.abs().mean().detach().cpu().item()),
        f"{prefix}_rms": float(torch.sqrt(torch.mean(value * value)).detach().cpu().item()),
        f"{prefix}_l2": float(torch.linalg.vector_norm(value).detach().cpu().item()),
        f"{prefix}_max_abs": float(value.abs().max().detach().cpu().item()),
        f"{prefix}_positive_frac": float((value > 0).float().mean().detach().cpu().item()),
    }


def channel_stats(prefix: str, patch: torch.Tensor, max_channels: int) -> dict:
    # patch shape: 1 x C x H x W
    channels = min(int(patch.shape[1]), max_channels)
    selected = patch[:, :channels].float()
    mean = selected.mean(dim=(0, 2, 3))
    abs_mean = selected.abs().mean(dim=(0, 2, 3))
    rms = torch.sqrt(torch.mean(selected * selected, dim=(0, 2, 3)))
    max_abs = selected.abs().amax(dim=(0, 2, 3))
    output = {}
    for index in range(channels):
        output[f"{prefix}_ch{index:02d}_mean"] = float(mean[index].detach().cpu().item())
        output[f"{prefix}_ch{index:02d}_abs_mean"] = float(abs_mean[index].detach().cpu().item())
        output[f"{prefix}_ch{index:02d}_rms"] = float(rms[index].detach().cpu().item())
        output[f"{prefix}_ch{index:02d}_max_abs"] = float(max_abs[index].detach().cpu().item())
    return output


def spatial_stats(prefix: str, patch: torch.Tensor, max_positions: int) -> dict:
    spatial = patch.float().abs().mean(dim=1).reshape(-1)
    count = min(int(spatial.numel()), max_positions)
    return {f"{prefix}_spatial_abs{index:02d}": float(spatial[index].detach().cpu().item()) for index in range(count)}


def coordinate_features(prefix: str, value: torch.Tensor, max_dims: int) -> dict:
    if max_dims <= 0:
        return {}
    vector = value.float().reshape(-1)
    count = min(int(vector.numel()), max_dims)
    return {f"{prefix}_dim{index:03d}": float(vector[index].detach().cpu().item()) for index in range(count)}


def patch_for_candidate(
    value: torch.Tensor,
    layout: LatentCandidateLayout,
    candidate,
    *,
    tile: int,
) -> torch.Tensor:
    c0 = candidate.slice_index * layout.slice_channels
    c1 = c0 + layout.slice_channels
    patch = value[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1]
    padded = torch.zeros((1, c1 - c0, tile, tile), dtype=value.dtype, device=value.device)
    padded[:, :, : patch.shape[-2], : patch.shape[-1]] = patch
    return padded


def augment_row(row: dict, vector: torch.Tensor, patch: torch.Tensor, codebook: torch.Tensor, args: argparse.Namespace) -> dict:
    code_index = int(row["code_index"])
    nearest_index = int(row["nearest_code"])
    code = codebook[code_index].float()
    nearest = codebook[nearest_index].float()
    delta = code - nearest
    residual_minus_code = vector - code
    residual_minus_nearest = vector - nearest
    dist_code = torch.linalg.vector_norm(residual_minus_code)
    dist_nearest = torch.linalg.vector_norm(residual_minus_nearest)
    safe_dist_nearest = dist_nearest.clamp_min(args.relative_floor)
    out = dict(row)
    out.update(vector_stats("latent_vec_residual", vector))
    out.update(vector_stats("latent_vec_residual_minus_code", residual_minus_code))
    out.update(vector_stats("latent_vec_residual_minus_nearest", residual_minus_nearest))
    out.update(vector_stats("latent_vec_code_minus_nearest", delta))
    out.update(channel_stats("latent_vec_residual", patch, args.max_channels))
    out.update(spatial_stats("latent_vec_residual", patch, args.max_spatial_positions))
    out.update(coordinate_features("latent_vec_residual", vector, args.max_vector_dims))
    out.update(
        {
            "latent_vec_dist_code": float(torch.linalg.vector_norm(residual_minus_code).detach().cpu().item()),
            "latent_vec_dist_nearest": float(dist_nearest.detach().cpu().item()),
            "latent_vec_dist_delta_code_minus_nearest": float(
                (dist_code - dist_nearest)
                .detach()
                .cpu()
                .item()
            ),
            "latent_vec_relative_error_recomputed": float(
                (dist_code / safe_dist_nearest)
                .detach()
                .cpu()
                .item()
            ),
            "latent_vec_relative_error_log1p": float(
                torch.log1p(dist_code / safe_dist_nearest)
                .detach()
                .cpu()
                .item()
            ),
            "latent_vec_residual_code_cosine": cosine(vector, code),
            "latent_vec_residual_nearest_cosine": cosine(vector, nearest),
            "latent_vec_residual_delta_cosine": cosine(vector, delta),
            "latent_vec_code_nearest_cosine": cosine(code, nearest),
            "latent_vec_residual_code_dot": float(torch.dot(vector, code).detach().cpu().item()),
            "latent_vec_residual_nearest_dot": float(torch.dot(vector, nearest).detach().cpu().item()),
            "latent_vec_residual_delta_dot": float(torch.dot(vector, delta).detach().cpu().item()),
            "latent_vec_code_nearest_dot": float(torch.dot(code, nearest).detach().cpu().item()),
        }
    )
    return out


def augment_group(
    model,
    rows: list[dict],
    codebooks: dict[int, torch.Tensor],
    device: str,
    args: argparse.Namespace,
) -> list[dict]:
    seed = row_seed(rows[0])
    if seed not in codebooks:
        raise SystemExit(f"missing codebook for real seed {seed}")
    codebook = codebooks[seed]
    image_path = resolve_path(str(rows[0]["source_image"]))
    image = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, _, _ = pad_to_multiple_64(x)
    with torch.no_grad():
        compressed = model.compress(x_pad)
        decoded = decompress_with_y_hat(model, compressed["strings"], compressed["shape"])
        residual = (model.g_a(x_pad) - decoded["y_hat"]) * args.residual_scale
        residual_for_codebook = quantize_residual_for_codebook(residual, args.codebook_quant_step)
    _, channels, latent_height, latent_width = residual_for_codebook.shape
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
    vector_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    output = []
    for row in rows:
        candidate_index = int(row["candidate_index"])
        if candidate_index not in vector_cache:
            candidate = candidates[candidate_index]
            patch = patch_for_candidate(residual_for_codebook, layout, candidate, tile=args.tile)
            vector = candidate_vector(residual_for_codebook, layout, candidate, tile=args.tile).float()
            vector_cache[candidate_index] = (vector, patch)
        vector, patch = vector_cache[candidate_index]
        output.append(augment_row(row, vector, patch, codebook, args))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebook", action="append", nargs=2, metavar=("SEED", "NPZ"), required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--tile", type=int, default=4)
    parser.add_argument("--channel-group-size", type=int)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--codebook-quant-step", type=float, default=0.05)
    parser.add_argument("--max-channels", type=int, default=16)
    parser.add_argument("--max-spatial-positions", type=int, default=16)
    parser.add_argument("--max-vector-dims", type=int, default=0)
    parser.add_argument("--relative-floor", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device != "cuda":
        raise SystemExit("MLIC++ compress/decompress uses CUDA-only synchronization paths; rerun with --cuda")

    codebooks = {int(seed): torch.from_numpy(load_codebook(Path(path))).to(device=device) for seed, path in args.codebook}
    rows = read_jsonl(args.input)
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row_seed(row), str(row["source_image"]))].append(row)

    model, checkpoint = load_model(args.checkpoint, device)
    output_rows = []
    for (seed, source_image), group in sorted(groups.items()):
        output_rows.extend(augment_group(model, group, codebooks, device, args))
        print(
            json.dumps(
                {
                    "real_seed": seed,
                    "source_image": project_path(resolve_path(source_image)),
                    "rows": len(group),
                    "candidate_count": len({int(row["candidate_index"]) for row in group}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "description": "Assignment utility table augmented with candidate residual-vector/codebook relation features.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": float(checkpoint.get("loss")) if checkpoint.get("loss") is not None else None,
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "images": len(groups),
        "codebook_seeds": sorted(codebooks),
        "tile": args.tile,
        "channel_group_size": args.channel_group_size,
        "residual_scale": args.residual_scale,
        "codebook_quant_step": args.codebook_quant_step,
        "max_channels": args.max_channels,
        "max_spatial_positions": args.max_spatial_positions,
        "max_vector_dims": args.max_vector_dims,
        "relative_floor": args.relative_floor,
        "features_added": [
            "latent_vec_residual_*",
            "latent_vec_residual_minus_code_*",
            "latent_vec_residual_minus_nearest_*",
            "latent_vec_code_minus_nearest_*",
            "latent_vec_dist_*",
            "latent_vec_*_cosine",
            "latent_vec_*_dot",
        ],
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
