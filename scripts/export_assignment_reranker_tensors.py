#!/usr/bin/env python3
"""Export tensor artifacts for local code-effect assignment rerankers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MLIC_ROOT = ROOT / "external/mlic/MLIC++"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def row_real_seed(row: dict) -> int:
    return int(row.get("real_seed", row.get("original_seed", row.get("seed", 0))))


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (row_real_seed(row), int(row["seed"]), int(row["source_index"]), int(row["candidate_index"]))


def load_codebook(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["codebook"], dtype=np.float32)


def load_scores(paths: list[Path], val_seed: int | None, model_seed: int | None) -> dict[tuple[int, int, int, int, int], float]:
    scores = {}
    for path in paths:
        for row in read_jsonl(path):
            if val_seed is not None and int(row.get("seed", val_seed)) != val_seed:
                continue
            if model_seed is not None and int(row.get("model_seed", model_seed)) != model_seed:
                continue
            key = (
                int(row.get("real_seed", row.get("original_seed", row.get("seed", 0)))),
                int(row["seed"]),
                int(row["source_index"]),
                int(row["candidate_index"]),
                int(row["code_index"]),
            )
            scores[key] = float(row["score"])
    return scores


def select_rows(rows: list[dict], partition: str, seed: int | None) -> list[dict]:
    output = []
    for row in rows:
        if seed is not None and int(row["seed"]) != seed:
            continue
        row_partition = str(row.get("assignment_partition", row.get("split", "")))
        if partition != "all" and row_partition != partition:
            continue
        output.append(row)
    return output


def image_triplet(nearest_path: str, variant_path: str, width: int, height: int) -> np.ndarray:
    with Image.open(resolve_path(nearest_path)) as image:
        nearest = image.convert("L").resize((width, height), Image.BICUBIC)
        nearest_arr = np.asarray(nearest, dtype=np.float32) / 255.0
    with Image.open(resolve_path(variant_path)) as image:
        variant = image.convert("L").resize((width, height), Image.BICUBIC)
        variant_arr = np.asarray(variant, dtype=np.float32) / 255.0
    diff = np.abs(variant_arr - nearest_arr)
    return np.stack([nearest_arr, variant_arr, diff], axis=0).astype(np.float16)


def rel_error(row: dict) -> float:
    value = float(row.get("assignment_relative_error", 0.0))
    return math.log1p(min(max(value, 0.0), 1.0e8))


def load_latent_helpers():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(MLIC_ROOT))
    import torch
    from torchvision import transforms

    from inspect_mlicpp_decoded_yhat import decompress_with_y_hat, load_model, pad_to_multiple_64
    from oscarlic.candidates import LatentCandidateLayout
    from run_mlicpp_latent_codebook_stub import candidate_vector, quantize_residual_for_codebook

    return {
        "torch": torch,
        "transforms": transforms,
        "decompress_with_y_hat": decompress_with_y_hat,
        "load_model": load_model,
        "pad_to_multiple_64": pad_to_multiple_64,
        "LatentCandidateLayout": LatentCandidateLayout,
        "candidate_vector": candidate_vector,
        "quantize_residual_for_codebook": quantize_residual_for_codebook,
    }


def latent_context_for_image(
    source_image: str,
    model,
    helpers: dict,
    device: str,
    args: argparse.Namespace,
) -> tuple[object, list, object, object, object]:
    torch = helpers["torch"]
    transforms = helpers["transforms"]
    LatentCandidateLayout = helpers["LatentCandidateLayout"]
    image = Image.open(resolve_path(source_image)).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, _, _ = helpers["pad_to_multiple_64"](x)
    with torch.no_grad():
        compressed = model.compress(x_pad)
        decoded = helpers["decompress_with_y_hat"](model, compressed["strings"], compressed["shape"])
        y_target = model.g_a(x_pad)
        y_hat = decoded["y_hat"]
        residual = (y_target - y_hat) * args.residual_scale
        residual_for_codebook = helpers["quantize_residual_for_codebook"](residual, args.codebook_quant_step)
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
    return layout, list(layout.iter_candidates()), y_hat, y_target, residual_for_codebook


def latent_vector(value, layout, candidates: list, candidate_index: int, helpers: dict, tile: int) -> np.ndarray:
    vector = helpers["candidate_vector"](value, layout, candidates[candidate_index], tile=tile).float().reshape(-1)
    return vector.detach().cpu().numpy().astype(np.float32)


def candidate_global_channels(layout, candidate) -> tuple[int, int]:
    channel0 = candidate.slice_index * layout.slice_channels + candidate.channel0
    channel1 = candidate.slice_index * layout.slice_channels + candidate.channel1
    return channel0, channel1


def latent_window(
    value,
    layout,
    candidate,
    *,
    radius: int,
) -> np.ndarray:
    c0, c1 = candidate_global_channels(layout, candidate)
    channels = c1 - c0
    window_h = (2 * radius + 1) * layout.tile_height
    window_w = (2 * radius + 1) * layout.tile_width
    requested_y0 = candidate.y0 - radius * layout.tile_height
    requested_x0 = candidate.x0 - radius * layout.tile_width
    y0 = max(0, requested_y0)
    x0 = max(0, requested_x0)
    y1 = min(layout.latent_height, candidate.y0 + (radius + 1) * layout.tile_height)
    x1 = min(layout.latent_width, candidate.x0 + (radius + 1) * layout.tile_width)
    patch = value[:, c0:c1, y0:y1, x0:x1]
    padded = value.new_zeros((1, channels, window_h, window_w))
    dy = y0 - requested_y0
    dx = x0 - requested_x0
    padded[:, :, dy : dy + patch.shape[-2], dx : dx + patch.shape[-1]] = patch
    return padded.squeeze(0).detach().cpu().numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--codebook", action="append", nargs=2, metavar=("SEED", "NPZ"), required=True)
    parser.add_argument("--score", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--partition", choices=["all", "train", "val"], default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--score-model-seed", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--include-latents", action="store_true")
    parser.add_argument("--include-latent-windows", action="store_true")
    parser.add_argument("--latent-window-radius", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--tile", type=int, default=4)
    parser.add_argument("--channel-group-size", type=int)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--codebook-quant-step", type=float, default=0.05)
    args = parser.parse_args()

    if args.include_latents and args.checkpoint is None:
        raise SystemExit("--checkpoint is required with --include-latents")
    if args.include_latent_windows and not args.include_latents:
        raise SystemExit("--include-latent-windows requires --include-latents")
    if args.latent_window_radius < 0:
        raise SystemExit("--latent-window-radius must be nonnegative")

    codebooks = {int(seed): load_codebook(Path(path)) for seed, path in args.codebook}
    rows = select_rows(read_jsonl(args.table), partition=args.partition, seed=args.seed)
    if not rows:
        raise SystemExit("no rows selected")
    groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    ordered_groups = [(key, sorted(value, key=lambda row: int(row["topk_rank"]))) for key, value in sorted(groups.items())]
    max_candidates = args.max_candidates or max(len(value) for _, value in ordered_groups)
    if max_candidates <= 0:
        raise SystemExit("--max-candidates must be positive")

    codebook_dim = next(iter(codebooks.values())).shape[1]
    group_count = len(ordered_groups)
    mask = np.zeros((group_count, max_candidates), dtype=np.bool_)
    group_keys = np.zeros((group_count, 4), dtype=np.int64)
    code_indices = np.full((group_count, max_candidates), -1, dtype=np.int64)
    nearest_indices = np.full((group_count, max_candidates), -1, dtype=np.int64)
    topk_ranks = np.full((group_count, max_candidates), -1, dtype=np.int64)
    is_nearest = np.zeros((group_count, max_candidates), dtype=np.float32)
    labels = np.zeros((group_count, max_candidates), dtype=np.float32)
    parseq_delta = np.zeros((group_count, max_candidates), dtype=np.float32)
    tesseract_delta = np.zeros((group_count, max_candidates), dtype=np.float32)
    score = np.full((group_count, max_candidates), np.nan, dtype=np.float32)
    core = np.zeros((group_count, max_candidates, 7), dtype=np.float32)
    code_center = np.zeros((group_count, max_candidates, codebook_dim), dtype=np.float16)
    nearest_center = np.zeros((group_count, max_candidates, codebook_dim), dtype=np.float16)
    delta_center = np.zeros((group_count, max_candidates, codebook_dim), dtype=np.float16)
    latent_yhat = (
        np.zeros((group_count, max_candidates, codebook_dim), dtype=np.float16) if args.include_latents else None
    )
    latent_target = (
        np.zeros((group_count, max_candidates, codebook_dim), dtype=np.float16) if args.include_latents else None
    )
    latent_residual = (
        np.zeros((group_count, max_candidates, codebook_dim), dtype=np.float16) if args.include_latents else None
    )
    latent_window_shape = None
    latent_window_yhat = None
    latent_window_target = None
    latent_window_residual = None
    if args.include_latent_windows:
        if codebook_dim % (args.tile * args.tile) != 0:
            raise SystemExit(f"codebook dim {codebook_dim} is not divisible by tile area {args.tile * args.tile}")
        latent_window_channels = codebook_dim // (args.tile * args.tile)
        latent_window_hw = (2 * args.latent_window_radius + 1) * args.tile
        latent_window_shape = (group_count, max_candidates, latent_window_channels, latent_window_hw, latent_window_hw)
        latent_window_yhat = np.zeros(latent_window_shape, dtype=np.float16)
        latent_window_target = np.zeros(latent_window_shape, dtype=np.float16)
        latent_window_residual = np.zeros(latent_window_shape, dtype=np.float16)
    images = (
        np.zeros((group_count, max_candidates, 3, args.image_height, args.image_width), dtype=np.float16)
        if args.include_images
        else None
    )
    group_meta = []
    score_by_code = load_scores(args.score, val_seed=args.seed, model_seed=args.score_model_seed) if args.score else {}
    latent_helpers = None
    latent_model = None
    latent_device = None
    latent_cache = {}
    if args.include_latents:
        latent_helpers = load_latent_helpers()
        torch = latent_helpers["torch"]
        latent_device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        if args.cuda and latent_device != "cuda":
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
        if latent_device != "cuda":
            raise SystemExit("MLIC++ compress/decompress uses CUDA-only synchronization paths; rerun with --cuda")
        latent_model, _ = latent_helpers["load_model"](args.checkpoint, latent_device)

    truncated_groups = 0
    oracle_missing_groups = 0
    for group_index, (key, group_rows) in enumerate(ordered_groups):
        group_keys[group_index] = np.asarray(key, dtype=np.int64)
        nearest_matches = [row for row in group_rows if int(row["is_nearest"])]
        if len(nearest_matches) != 1:
            raise SystemExit(f"group {key} has {len(nearest_matches)} nearest rows")
        nearest_row = nearest_matches[0]
        oracle_matches = [row for row in group_rows if int(row["label_assignment_oracle_choice"])]
        oracle_missing = not oracle_matches
        if oracle_missing:
            oracle_missing_groups += 1
        codebook = codebooks[key[0]]
        nearest_code = int(nearest_row["code_index"])
        latent_context = None
        if args.include_latents:
            if latent_helpers is None or latent_model is None or latent_device is None:
                raise RuntimeError("latent helpers were not initialized")
            latent_key = (key[0], str(nearest_row["source_image"]))
            if latent_key not in latent_cache:
                latent_cache[latent_key] = latent_context_for_image(
                    str(nearest_row["source_image"]),
                    latent_model,
                    latent_helpers,
                    latent_device,
                    args,
                )
            latent_context = latent_cache[latent_key]
        limited = group_rows[:max_candidates]
        truncated_groups += int(len(group_rows) > max_candidates)
        group_meta.append(
            {
                "group_index": group_index,
                "real_seed": key[0],
                "seed": key[1],
                "source_index": key[2],
                "candidate_index": key[3],
                "source": str(nearest_row.get("source", "")),
                "reference": str(nearest_row.get("reference", "")),
                "source_image": str(nearest_row.get("source_image", "")),
                "nearest_code": nearest_code,
                "oracle_code": int(oracle_matches[0]["code_index"]) if oracle_matches else nearest_code,
                "rows": len(group_rows),
                "truncated_rows": max(0, len(group_rows) - max_candidates),
            }
        )
        for row_position, row in enumerate(limited):
            code_index = int(row["code_index"])
            mask[group_index, row_position] = True
            code_indices[group_index, row_position] = code_index
            nearest_indices[group_index, row_position] = nearest_code
            topk_ranks[group_index, row_position] = int(row["topk_rank"])
            is_nearest[group_index, row_position] = float(int(row["is_nearest"]))
            labels[group_index, row_position] = float(
                int(row["label_assignment_oracle_choice"]) or (oracle_missing and int(row["is_nearest"]))
            )
            parseq_delta[group_index, row_position] = float(row["parseq_delta_vs_nearest"])
            tesseract_delta[group_index, row_position] = float(row["tesseract_delta_vs_nearest"])
            score_key = (*key, code_index)
            if score_key in score_by_code:
                score[group_index, row_position] = float(score_by_code[score_key])
            core[group_index, row_position] = np.asarray(
                [
                    float(int(row["topk_rank"])),
                    rel_error(row),
                    float(int(row["is_nearest"])),
                    float(int(row["code_equals_nearest"])),
                    float(row.get("reference_length", 0.0)),
                    float(row.get("parseq_nearest_distance", 0.0)),
                    float(row.get("tesseract_nearest_distance", 0.0)),
                ],
                dtype=np.float32,
            )
            code_vec = codebook[code_index].astype(np.float32)
            nearest_vec = codebook[nearest_code].astype(np.float32)
            code_center[group_index, row_position] = code_vec.astype(np.float16)
            nearest_center[group_index, row_position] = nearest_vec.astype(np.float16)
            delta_center[group_index, row_position] = (code_vec - nearest_vec).astype(np.float16)
            if images is not None:
                images[group_index, row_position] = image_triplet(
                    str(nearest_row["image_path"]),
                    str(row["image_path"]),
                    width=args.image_width,
                    height=args.image_height,
                )
            if latent_yhat is not None and latent_target is not None and latent_residual is not None:
                if latent_context is None or latent_helpers is None:
                    raise RuntimeError("latent context was not initialized")
                layout, candidates, y_hat, y_target, residual_for_codebook = latent_context
                candidate_index = int(row["candidate_index"])
                candidate = candidates[candidate_index]
                yhat_vec = latent_vector(y_hat, layout, candidates, candidate_index, latent_helpers, args.tile)
                target_vec = latent_vector(y_target, layout, candidates, candidate_index, latent_helpers, args.tile)
                residual_vec = latent_vector(
                    residual_for_codebook,
                    layout,
                    candidates,
                    candidate_index,
                    latent_helpers,
                    args.tile,
                )
                if yhat_vec.size != codebook_dim:
                    raise SystemExit(
                        f"latent vector dim {yhat_vec.size} does not match codebook dim {codebook_dim}"
                    )
                latent_yhat[group_index, row_position] = yhat_vec.astype(np.float16)
                latent_target[group_index, row_position] = target_vec.astype(np.float16)
                latent_residual[group_index, row_position] = residual_vec.astype(np.float16)
                if (
                    latent_window_yhat is not None
                    and latent_window_target is not None
                    and latent_window_residual is not None
                ):
                    yhat_window = latent_window(
                        y_hat,
                        layout,
                        candidate,
                        radius=args.latent_window_radius,
                    )
                    target_window = latent_window(
                        y_target,
                        layout,
                        candidate,
                        radius=args.latent_window_radius,
                    )
                    residual_window = latent_window(
                        residual_for_codebook,
                        layout,
                        candidate,
                        radius=args.latent_window_radius,
                    )
                    if latent_window_shape is not None and yhat_window.shape != latent_window_shape[2:]:
                        raise SystemExit(
                            f"latent window shape {yhat_window.shape} does not match expected {latent_window_shape[2:]}"
                        )
                    latent_window_yhat[group_index, row_position] = yhat_window.astype(np.float16)
                    latent_window_target[group_index, row_position] = target_window.astype(np.float16)
                    latent_window_residual[group_index, row_position] = residual_window.astype(np.float16)

    arrays = {
        "group_keys": group_keys,
        "mask": mask,
        "code_indices": code_indices,
        "nearest_indices": nearest_indices,
        "topk_ranks": topk_ranks,
        "is_nearest": is_nearest,
        "labels": labels,
        "parseq_delta": parseq_delta,
        "tesseract_delta": tesseract_delta,
        "baseline_score": score,
        "core_features": core,
        "code_center": code_center,
        "nearest_center": nearest_center,
        "delta_center": delta_center,
    }
    if images is not None:
        arrays["image_triplet"] = images
    if latent_yhat is not None and latent_target is not None and latent_residual is not None:
        arrays["latent_yhat"] = latent_yhat
        arrays["latent_target"] = latent_target
        arrays["latent_residual"] = latent_residual
    if latent_window_yhat is not None and latent_window_target is not None and latent_window_residual is not None:
        arrays["latent_window_yhat"] = latent_window_yhat
        arrays["latent_window_target"] = latent_window_target
        arrays["latent_window_residual"] = latent_window_residual
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    metadata_path = args.metadata_output or args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "description": "Tensor artifact for local code-effect assignment reranker training.",
        "table": str(args.table),
        "output": str(args.output),
        "scores": [str(path) for path in args.score],
        "codebooks": {str(seed): str(path) for seed, path in args.codebook},
        "config": {
            "partition": args.partition,
            "seed": args.seed,
            "score_model_seed": args.score_model_seed,
            "max_candidates": max_candidates,
            "include_images": args.include_images,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "include_latents": args.include_latents,
            "include_latent_windows": args.include_latent_windows,
            "latent_window_radius": args.latent_window_radius,
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "cuda": args.cuda,
            "tile": args.tile,
            "channel_group_size": args.channel_group_size,
            "residual_scale": args.residual_scale,
            "codebook_quant_step": args.codebook_quant_step,
        },
        "groups": group_count,
        "rows": len(rows),
        "positive_labels": int(labels.sum()),
        "max_candidates_observed": max(len(value) for _, value in ordered_groups),
        "truncated_groups": truncated_groups,
        "oracle_missing_groups": oracle_missing_groups,
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "core_feature_names": [
            "topk_rank",
            "log1p_clipped_assignment_relative_error",
            "is_nearest",
            "code_equals_nearest",
            "reference_length",
            "parseq_nearest_distance",
            "tesseract_nearest_distance",
        ],
        "group_metadata": group_meta,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(metadata_path),
                "groups": group_count,
                "rows": len(rows),
                "arrays": metadata["arrays"],
                "truncated_groups": truncated_groups,
                "oracle_missing_groups": oracle_missing_groups,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
