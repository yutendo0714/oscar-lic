#!/usr/bin/env python3
"""Counted MLIC++ latent codebook enhancement diagnostic."""

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

from inspect_mlicpp_decoded_yhat import decompress_with_y_hat, load_model, pad_to_multiple_64, project_path, read_image_list  # noqa: E402
from oscarlic.bitstream import Section, read_container, write_container  # noqa: E402
from oscarlic.candidates import (  # noqa: E402
    LatentCandidateLayout,
    decode_compact_codebook_payload,
    decode_gate_payload,
    encode_compact_codebook_payload,
    encode_gate_payload,
)
from run_mlicpp_latent_residual_stub import (  # noqa: E402
    CODEC_MLICPP_UPSTREAM_BODY,
    MODEL_ID_MLICPP_CORRECTED,
    MODEL_VERSION_CORRECTED_0025,
    SECTION_BASE_MAIN,
    SECTION_TEXT_GATE,
    SECTION_TEXT_MAIN,
    candidate_global_channels,
    parse_mlicpp_body,
    psnr_from_tensors,
    select_candidates_by_residual,
    serialize_mlicpp_body,
)
from utils.utils import torch2img  # noqa: E402


CODEC_OSCAR_GATE_V0 = 1
CODEC_OSCAR_LATENT_CODEBOOK_V0 = 3
CODEC_OSCAR_COMPACT_CODEBOOK_V0 = 4
SECTION_TEXT_SYMBOLS = 14
CODEBOOK_HEADER = struct.Struct("<HI")


def read_candidate_selection(path: Path | None, mode: str) -> dict[int, list[int]] | None:
    if path is None:
        return None
    if mode == "explicit":
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
    if mode != "oracle_multi_teacher_best_negative":
        raise ValueError(f"unsupported candidate selection mode: {mode}")
    by_source: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_index = int(row["source_index"])
            delta = int(row["multi_teacher_delta_distance"])
            if delta >= 0:
                continue
            current = by_source.get(source_index)
            if current is None or (delta, -int(row["improving_evaluator_count"])) < (
                int(current["multi_teacher_delta_distance"]),
                -int(current["improving_evaluator_count"]),
            ):
                by_source[source_index] = row
    return {source_index: [int(row["candidate_index"])] for source_index, row in by_source.items()}


def read_candidate_rows(path: Path | None, mode: str) -> dict[tuple[int, int], dict]:
    if path is None:
        return {}
    rows: dict[tuple[int, int], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if mode == "explicit" and "selected_by_policy" in row and not bool(row["selected_by_policy"]):
                continue
            if mode == "oracle_multi_teacher_best_negative" and int(row.get("multi_teacher_delta_distance", 0)) >= 0:
                continue
            rows[(int(row["source_index"]), int(row["candidate_index"]))] = row
    return rows


def candidate_vector(
    value: torch.Tensor,
    layout: LatentCandidateLayout,
    candidate,
    *,
    tile: int,
) -> torch.Tensor:
    c0, c1 = candidate_global_channels(layout, candidate)
    patch = value[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1]
    padded = torch.zeros((1, c1 - c0, tile, tile), dtype=value.dtype, device=value.device)
    padded[:, :, : patch.shape[-2], : patch.shape[-1]] = patch
    return padded.reshape(-1)


def quantize_residual_for_codebook(residual: torch.Tensor, quant_step: float | None) -> torch.Tensor:
    if quant_step is None or quant_step <= 0:
        return residual
    return torch.round(residual / quant_step).clamp(-127, 127) * quant_step


def collect_examples(
    model,
    image_paths: list[Path],
    device: str,
    args: argparse.Namespace,
    selection_map: dict[int, list[int]] | None,
    selection_rows: dict[tuple[int, int], dict],
) -> tuple[torch.Tensor, list[dict], list[dict]]:
    vectors = []
    vector_rows = []
    cache = []
    for index, image_path in enumerate(image_paths):
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
        selected = (
            selection_map.get(index, [])
            if selection_map is not None
            else select_candidates_by_residual(residual, layout, args.selected_fraction)
        )
        explicit_code_indices = []
        candidates = list(layout.iter_candidates())
        for candidate_index in selected:
            vectors.append(candidate_vector(residual_for_codebook, layout, candidates[candidate_index], tile=args.tile))
            row = selection_rows.get(
                (index, int(candidate_index)),
                {"source_index": index, "candidate_index": int(candidate_index)},
            )
            vector_rows.append(row)
            explicit_code = row.get("assignment_code_index", row.get("code_index"))
            explicit_code_indices.append(None if explicit_code is None else int(explicit_code))
        cache.append(
            {
                "image_path": image_path,
                "compressed": compressed,
                "layout": layout,
                "selected": selected,
                "explicit_code_indices": explicit_code_indices,
                "width": image.width,
                "height": image.height,
            }
        )
        print(json.dumps({"phase": "collect", "index": index, "selected": len(selected)}, ensure_ascii=False), flush=True)
    if not vectors:
        raise ValueError("no selected candidates to train codebook")
    return torch.stack(vectors, dim=0), cache, vector_rows


def train_kmeans(vectors: torch.Tensor, k: int, iters: int) -> torch.Tensor:
    if vectors.shape[0] < k:
        k = vectors.shape[0]
    order = torch.linspace(0, vectors.shape[0] - 1, steps=k, device=vectors.device).round().long()
    centers = vectors[order].clone()
    for _ in range(iters):
        distances = torch.cdist(vectors.float(), centers.float())
        labels = distances.argmin(dim=1)
        new_centers = []
        for center_index in range(k):
            members = vectors[labels == center_index]
            if members.numel() == 0:
                new_centers.append(centers[center_index])
            else:
                new_centers.append(members.mean(dim=0))
        updated = torch.stack(new_centers, dim=0)
        if torch.allclose(updated, centers):
            centers = updated
            break
        centers = updated
    return centers


def utility_score(row: dict, protected_penalty: float) -> float:
    tesseract_gain = max(0.0, -float(row.get("tesseract_delta_distance", 0.0)))
    parseq_harm = max(0.0, float(row.get("parseq_delta_distance", 0.0)))
    selector_score = float(row.get("selector_score", 0.0))
    return tesseract_gain - protected_penalty * parseq_harm + 0.01 * selector_score


def train_utility_weighted_kmeans(
    vectors: torch.Tensor,
    rows: list[dict],
    *,
    k: int,
    iters: int,
    alpha: float,
    protected_penalty: float,
) -> torch.Tensor:
    if vectors.shape[0] < k:
        k = vectors.shape[0]
    utilities = torch.tensor(
        [max(0.0, utility_score(row, protected_penalty)) for row in rows],
        dtype=torch.float32,
        device=vectors.device,
    )
    weights = 1.0 + alpha * utilities
    order = torch.argsort(utilities, descending=True)
    if int((utilities > 0).sum().item()) < k:
        distance_order = torch.linspace(0, vectors.shape[0] - 1, steps=k, device=vectors.device).round().long()
        initial = torch.cat([order, distance_order], dim=0)
        # torch.unique sorts for 1D tensors, so preserve enough deterministic fallback by de-duplicating in Python.
        seen = []
        seen_set = set()
        for value in initial.detach().cpu().tolist():
            if int(value) not in seen_set:
                seen.append(int(value))
                seen_set.add(int(value))
            if len(seen) == k:
                break
        order = torch.tensor(seen, dtype=torch.long, device=vectors.device)
    else:
        order = order[:k]
    centers = vectors[order[:k]].clone()
    for _ in range(iters):
        distances = torch.cdist(vectors.float(), centers.float())
        labels = distances.argmin(dim=1)
        new_centers = []
        for center_index in range(k):
            mask = labels == center_index
            members = vectors[mask]
            if members.numel() == 0:
                new_centers.append(centers[center_index])
            else:
                member_weights = weights[mask].reshape(-1, 1).to(dtype=members.dtype)
                new_centers.append((members * member_weights).sum(dim=0) / member_weights.sum())
        updated = torch.stack(new_centers, dim=0)
        if torch.allclose(updated, centers):
            centers = updated
            break
        centers = updated
    return centers


def compute_center_utilities(
    vectors: torch.Tensor,
    rows: list[dict],
    codebook: torch.Tensor,
    *,
    protected_penalty: float,
) -> torch.Tensor:
    utilities = torch.tensor(
        [utility_score(row, protected_penalty) for row in rows],
        dtype=torch.float32,
        device=vectors.device,
    )
    distances = torch.cdist(vectors.float(), codebook.float())
    labels = distances.argmin(dim=1)
    center_utilities = torch.zeros((codebook.shape[0],), dtype=torch.float32, device=vectors.device)
    for center_index in range(codebook.shape[0]):
        members = utilities[labels == center_index]
        if members.numel():
            center_utilities[center_index] = members.mean()
    return center_utilities


def encode_code_payload(code_indices: list[int], *, codebook_id: int = 0) -> bytes:
    if any(index < 0 or index > 255 for index in code_indices):
        raise ValueError("code indices must fit uint8")
    return CODEBOOK_HEADER.pack(1, codebook_id) + bytes(code_indices)


def decode_code_payload(payload: bytes) -> dict:
    if len(payload) < CODEBOOK_HEADER.size:
        raise ValueError("codebook payload is shorter than header")
    version, codebook_id = CODEBOOK_HEADER.unpack_from(payload, 0)
    if version != 1:
        raise ValueError(f"unsupported codebook payload version {version}")
    return {"codebook_id": codebook_id, "code_indices": list(payload[CODEBOOK_HEADER.size:])}


def has_zero_code(codebook: torch.Tensor) -> bool:
    return bool(codebook.shape[0] > 0 and torch.allclose(codebook[0], torch.zeros_like(codebook[0])))


def assign_codes(
    residual: torch.Tensor,
    layout: LatentCandidateLayout,
    selected: list[int],
    codebook: torch.Tensor,
    center_utilities: torch.Tensor | None,
    tile: int,
    *,
    assignment_mode: str = "nearest",
    assignment_topk: int = 8,
    assignment_max_relative_error: float = 1.05,
    zero_code_max_relative_error: float | None = None,
    explicit_code_indices: list[int | None] | None = None,
) -> tuple[list[int], int, float]:
    candidates = list(layout.iter_candidates())
    vectors = torch.stack([candidate_vector(residual, layout, candidates[index], tile=tile) for index in selected], dim=0)
    distances = torch.cdist(vectors.float(), codebook.float())
    nearest_distances, nearest_indices = distances.min(dim=1)
    if assignment_mode == "explicit":
        if explicit_code_indices is None:
            raise ValueError("explicit assignment mode requires explicit_code_indices")
        if len(explicit_code_indices) != len(selected):
            raise ValueError("explicit code indices and selected candidates length mismatch")
        explicit_codes = []
        for value, fallback in zip(explicit_code_indices, nearest_indices.detach().cpu().tolist()):
            code_index = int(fallback) if value is None else int(value)
            if code_index < 0 or code_index >= int(codebook.shape[0]):
                raise ValueError(f"explicit code index {code_index} outside codebook size {int(codebook.shape[0])}")
            explicit_codes.append(code_index)
        codes = torch.tensor(explicit_codes, dtype=torch.long, device=distances.device)
    elif zero_code_max_relative_error is not None and has_zero_code(codebook) and codebook.shape[0] > 1:
        nonzero_distances = distances[:, 1:]
        best_nonzero_distances, best_nonzero_indices = nonzero_distances.min(dim=1)
        relative = best_nonzero_distances / vectors.float().norm(dim=1).clamp_min(1e-8)
        codes = torch.where(
            relative > zero_code_max_relative_error,
            torch.zeros_like(best_nonzero_indices),
            best_nonzero_indices + 1,
        )
    elif assignment_mode == "utility_biased" and center_utilities is not None and codebook.shape[0] > 1:
        relative = distances / nearest_distances.reshape(-1, 1).clamp_min(1e-8)
        mask = relative <= assignment_max_relative_error
        if assignment_topk > 0 and assignment_topk < codebook.shape[0]:
            topk_indices = distances.topk(k=assignment_topk, dim=1, largest=False).indices
            topk_mask = torch.zeros_like(mask)
            topk_mask.scatter_(1, topk_indices, True)
            mask &= topk_mask
        utilities = center_utilities.to(device=distances.device, dtype=distances.dtype).reshape(1, -1)
        scores = utilities.expand_as(distances) - 1e-6 * relative
        scores = torch.where(mask, scores, torch.full_like(scores, -float("inf")))
        empty = ~torch.isfinite(scores).any(dim=1)
        codes = scores.argmax(dim=1)
        codes = torch.where(empty, nearest_indices, codes)
    else:
        codes = nearest_indices
    chosen_distances = distances.gather(1, codes.reshape(-1, 1)).squeeze(1)
    chosen_relative = chosen_distances / nearest_distances.clamp_min(1e-8)
    changed_count = int((codes != nearest_indices).sum().item())
    return [int(index) for index in codes.detach().cpu().tolist()], changed_count, float(chosen_relative.mean().item())


def apply_codebook(
    y_hat: torch.Tensor,
    layout: LatentCandidateLayout,
    selected: list[int],
    code_indices: list[int],
    codebook: torch.Tensor,
    tile: int,
) -> torch.Tensor:
    if len(selected) != len(code_indices):
        raise ValueError("selected candidates and code indices length mismatch")
    candidates = list(layout.iter_candidates())
    y_enhanced = y_hat.clone()
    for candidate_index, code_index in zip(selected, code_indices):
        candidate = candidates[candidate_index]
        c0, c1 = candidate_global_channels(layout, candidate)
        code = codebook[code_index].to(device=y_hat.device, dtype=y_hat.dtype).reshape(1, c1 - c0, tile, tile)
        y_enhanced[:, c0:c1, candidate.y0:candidate.y1, candidate.x0:candidate.x1] += code[
            :, :, : candidate.y1 - candidate.y0, : candidate.x1 - candidate.x0
        ]
    return y_enhanced


def run_one(
    model,
    item: dict,
    output_dir: Path,
    device: str,
    index: int,
    codebook: torch.Tensor,
    center_utilities: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict:
    image_path = item["image_path"]
    image = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    x_pad, pad_h, pad_w = pad_to_multiple_64(x)
    compressed = item["compressed"]
    layout = item["layout"]
    selected = item["selected"]

    with torch.no_grad():
        decoded = decompress_with_y_hat(model, compressed["strings"], compressed["shape"])
        residual = (model.g_a(x_pad) - decoded["y_hat"]) * args.residual_scale
        residual_for_codebook = quantize_residual_for_codebook(residual, args.codebook_quant_step)
        codes, assignment_changed_count, assignment_relative_error = (
            assign_codes(
                residual_for_codebook,
                layout,
                selected,
                codebook,
                center_utilities,
                args.tile,
                assignment_mode=args.assignment_mode,
                assignment_topk=args.assignment_topk,
                assignment_max_relative_error=args.assignment_max_relative_error,
                zero_code_max_relative_error=args.zero_code_max_relative_error,
                explicit_code_indices=item.get("explicit_code_indices"),
            )
            if selected
            else ([], 0, 0.0)
        )
        gate_payload = encode_gate_payload(num_candidates=layout.candidate_count, selected_indices=selected) if selected else b""
        code_payload = encode_code_payload(codes) if selected else b""
        compact_payload = encode_compact_codebook_payload(selected, codes) if selected else b""
        base_payload = serialize_mlicpp_body(compressed["shape"], compressed["strings"])

    stream_path = output_dir / "streams" / f"{index:03d}_{image_path.stem}.oscr"
    sections = [Section(SECTION_BASE_MAIN, CODEC_MLICPP_UPSTREAM_BODY, 0, base_payload)]
    if selected:
        if args.compact_text_section:
            sections.append(
                Section(
                    SECTION_TEXT_SYMBOLS,
                    CODEC_OSCAR_COMPACT_CODEBOOK_V0,
                    0,
                    compact_payload,
                    dependency_mask=1,
                )
            )
        else:
            sections.extend(
                [
                    Section(SECTION_TEXT_GATE, CODEC_OSCAR_GATE_V0, 0, gate_payload, dependency_mask=1),
                    Section(SECTION_TEXT_MAIN, CODEC_OSCAR_LATENT_CODEBOOK_V0, 0, code_payload, dependency_mask=1),
                ]
            )
    write_container(
        stream_path,
        width=image.width,
        height=image.height,
        model_id=MODEL_ID_MLICPP_CORRECTED,
        model_version=MODEL_VERSION_CORRECTED_0025,
        flags=1,
        sections=sections,
    )

    parsed = read_container(stream_path)
    base_section = [section for section in parsed.sections if section.section_type == SECTION_BASE_MAIN][0]
    strings, shape = parse_mlicpp_body(base_section.payload)
    with torch.no_grad():
        decoded_from_stream = decompress_with_y_hat(model, strings, shape)
        if selected:
            if args.compact_text_section:
                compact_section = [section for section in parsed.sections if section.section_type == SECTION_TEXT_SYMBOLS][0]
                decoded_compact = decode_compact_codebook_payload(
                    compact_section.payload,
                    num_candidates=layout.candidate_count,
                )
                selected_indices = decoded_compact["selected_indices"]
                code_indices = decoded_compact["code_indices"]
            else:
                gate_section = [section for section in parsed.sections if section.section_type == SECTION_TEXT_GATE][0]
                code_section = [section for section in parsed.sections if section.section_type == SECTION_TEXT_MAIN][0]
                decoded_gate = decode_gate_payload(gate_section.payload)
                decoded_codes = decode_code_payload(code_section.payload)
                selected_indices = decoded_gate["selected_indices"]
                code_indices = decoded_codes["code_indices"]
            y_enhanced = apply_codebook(
                decoded_from_stream["y_hat"],
                layout,
                selected_indices,
                code_indices,
                codebook,
                args.tile,
            )
            x_enhanced = model.g_s(y_enhanced)[:, :, : image.height, : image.width]
        else:
            x_enhanced = decoded_from_stream["x_hat"][:, :, : image.height, : image.width]
    x_base = decoded_from_stream["x_hat"][:, :, : image.height, : image.width]

    rec_dir = output_dir / "reconstructions"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec_path = rec_dir / f"{index:03d}_{image_path.stem}_enhanced.png"
    torch2img(x_enhanced.detach().cpu()).save(rec_path)

    total_bytes = stream_path.stat().st_size
    payload_bytes_total = sum(section.length for section in parsed.sections)
    return {
        "index": index,
        "image": project_path(image_path),
        "width": image.width,
        "height": image.height,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "tile": args.tile,
        "channel_group_size": args.channel_group_size,
        "candidate_count": layout.candidate_count,
        "selected_candidate_count": len(selected),
        "selected_fraction": args.selected_fraction,
        "residual_scale": args.residual_scale,
        "codebook_size": int(codebook.shape[0]),
        "base_payload_bytes": len(base_payload),
        "gate_payload_bytes": 0 if args.compact_text_section else len(gate_payload),
        "code_payload_bytes": 0 if args.compact_text_section else len(code_payload),
        "compact_payload_bytes": len(compact_payload) if args.compact_text_section else 0,
        "split_gate_payload_bytes": len(gate_payload),
        "split_code_payload_bytes": len(code_payload),
        "compact_text_section": bool(args.compact_text_section),
        "compact_section_type": SECTION_TEXT_SYMBOLS if args.compact_text_section else None,
        "compact_section_codec": CODEC_OSCAR_COMPACT_CODEBOOK_V0 if args.compact_text_section else None,
        "zero_code_count": sum(1 for code in codes if code == 0) if has_zero_code(codebook) else 0,
        "zero_code_max_relative_error": args.zero_code_max_relative_error,
        "assignment_mode": args.assignment_mode,
        "assignment_changed_count": assignment_changed_count,
        "assignment_mean_relative_error": assignment_relative_error,
        "structural_overhead_bytes": total_bytes - payload_bytes_total,
        "actual_total_bytes": total_bytes,
        "actual_total_bpp": 8.0 * total_bytes / (image.width * image.height),
        "base_payload_bpp": 8.0 * len(base_payload) / (image.width * image.height),
        "enhancement_payload_bpp": 8.0
        * ((len(compact_payload) if args.compact_text_section else len(gate_payload) + len(code_payload)))
        / (image.width * image.height),
        "psnr_base_db": psnr_from_tensors(x_base, x),
        "psnr_enhanced_db": psnr_from_tensors(x_enhanced, x),
        "psnr_delta_db": psnr_from_tensors(x_enhanced, x) - psnr_from_tensors(x_base, x),
        "stream_path": project_path(stream_path),
        "reconstruction_path": project_path(rec_path),
        "bitstream_format": (
            "oscr_v0_1_mlicpp_base_plus_compact_latent_codebook_stub"
            if args.compact_text_section
            else "oscr_v0_1_mlicpp_base_plus_latent_codebook_stub"
        ),
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
    parser.add_argument("--selected-fraction", type=float, default=0.05)
    parser.add_argument("--candidate-selection-table", type=Path)
    parser.add_argument(
        "--candidate-selection-mode",
        choices=["oracle_multi_teacher_best_negative", "explicit"],
        default="oracle_multi_teacher_best_negative",
    )
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--codebook-quant-step", type=float)
    parser.add_argument("--codebook-size", type=int, default=16)
    parser.add_argument("--codebook-train-mode", choices=["kmeans", "utility_weighted_kmeans"], default="kmeans")
    parser.add_argument("--utility-weight-alpha", type=float, default=4.0)
    parser.add_argument("--utility-protected-penalty", type=float, default=4.0)
    parser.add_argument("--assignment-mode", choices=["nearest", "utility_biased", "explicit"], default="nearest")
    parser.add_argument("--assignment-topk", type=int, default=8)
    parser.add_argument("--assignment-max-relative-error", type=float, default=1.05)
    parser.add_argument("--append-zero-code", action="store_true")
    parser.add_argument("--zero-code-max-relative-error", type=float)
    parser.add_argument(
        "--compact-text-section",
        action="store_true",
        help="Encode selected candidate/code pairs in one compact optional section instead of TEXT_GATE + TEXT_MAIN.",
    )
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--load-codebook", type=Path)
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if device != "cuda":
        raise SystemExit("MLIC++ compress/decompress uses torch.cuda.synchronize(); run with --cuda on a visible GPU")
    if not 0 <= args.selected_fraction <= 1:
        raise SystemExit("--selected-fraction must be in [0, 1]")
    if args.codebook_size <= 0 or args.codebook_size > 256:
        raise SystemExit("--codebook-size must be in [1, 256]")
    if args.append_zero_code and args.codebook_size >= 256:
        raise SystemExit("--append-zero-code requires --codebook-size <= 255")
    if args.zero_code_max_relative_error is not None and args.zero_code_max_relative_error <= 0:
        raise SystemExit("--zero-code-max-relative-error must be positive")
    if args.assignment_topk <= 0:
        raise SystemExit("--assignment-topk must be positive")
    if args.assignment_max_relative_error < 1:
        raise SystemExit("--assignment-max-relative-error must be >= 1")

    image_paths = read_image_list(args.split, args.images, args.max_images)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)

    selection_map = read_candidate_selection(args.candidate_selection_table, args.candidate_selection_mode)
    selection_rows = read_candidate_rows(args.candidate_selection_table, args.candidate_selection_mode)
    vectors, cache, vector_rows = collect_examples(model, image_paths, device, args, selection_map, selection_rows)
    if args.load_codebook:
        data = np.load(args.load_codebook)
        codebook = torch.from_numpy(data["codebook"]).to(device=device, dtype=vectors.dtype)
        if "center_utility" in data.files:
            center_utilities = torch.from_numpy(data["center_utility"]).to(device=device, dtype=torch.float32)
        else:
            center_utilities = torch.zeros((codebook.shape[0],), dtype=torch.float32, device=device)
        codebook_source = str(args.load_codebook)
    else:
        if args.codebook_train_mode == "kmeans":
            codebook = train_kmeans(vectors, args.codebook_size, args.kmeans_iters)
        elif args.codebook_train_mode == "utility_weighted_kmeans":
            codebook = train_utility_weighted_kmeans(
                vectors,
                vector_rows,
                k=args.codebook_size,
                iters=args.kmeans_iters,
                alpha=args.utility_weight_alpha,
                protected_penalty=args.utility_protected_penalty,
            )
        else:
            raise SystemExit(f"unsupported codebook train mode: {args.codebook_train_mode}")
        if args.append_zero_code:
            zero = torch.zeros((1, codebook.shape[1]), dtype=codebook.dtype, device=codebook.device)
            codebook = torch.cat([zero, codebook], dim=0)
        center_utilities = compute_center_utilities(
            vectors,
            vector_rows,
            codebook,
            protected_penalty=args.utility_protected_penalty,
        )
        codebook_source = "trained_on_current_split"
    np.savez_compressed(
        output_dir / "latent_residual_codebook.npz",
        codebook=codebook.detach().cpu().numpy(),
        center_utility=center_utilities.detach().cpu().numpy(),
        tile=args.tile,
        channel_group_size=-1 if args.channel_group_size is None else args.channel_group_size,
        residual_scale=args.residual_scale,
        codebook_quant_step=-1.0 if args.codebook_quant_step is None else args.codebook_quant_step,
        codebook_train_mode=args.codebook_train_mode,
        utility_weight_alpha=args.utility_weight_alpha,
        utility_protected_penalty=args.utility_protected_penalty,
        append_zero_code=bool(args.append_zero_code),
    )
    rows = [
        run_one(model, item, output_dir, device, index, codebook, center_utilities, args)
        for index, item in enumerate(cache)
    ]
    selected_candidate_total = sum(row["selected_candidate_count"] for row in rows)
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
        "residual_scale": args.residual_scale,
        "codebook_quant_step": args.codebook_quant_step,
        "codebook_size": int(codebook.shape[0]),
        "codebook_train_mode": args.codebook_train_mode,
        "utility_weight_alpha": args.utility_weight_alpha,
        "utility_protected_penalty": args.utility_protected_penalty,
        "assignment_mode": args.assignment_mode,
        "assignment_topk": args.assignment_topk,
        "assignment_max_relative_error": args.assignment_max_relative_error,
        "append_zero_code": bool(args.append_zero_code),
        "zero_code_max_relative_error": args.zero_code_max_relative_error,
        "codebook_source": codebook_source,
        "kmeans_examples": int(vectors.shape[0]),
        "avg_actual_total_bpp": sum(row["actual_total_bpp"] for row in rows) / len(rows),
        "avg_base_payload_bpp": sum(row["base_payload_bpp"] for row in rows) / len(rows),
        "avg_enhancement_payload_bpp": sum(row["enhancement_payload_bpp"] for row in rows) / len(rows),
        "avg_selected_candidate_count": sum(row["selected_candidate_count"] for row in rows) / len(rows),
        "avg_gate_payload_bytes": sum(row["gate_payload_bytes"] for row in rows) / len(rows),
        "avg_code_payload_bytes": sum(row["code_payload_bytes"] for row in rows) / len(rows),
        "avg_compact_payload_bytes": sum(row["compact_payload_bytes"] for row in rows) / len(rows),
        "avg_split_gate_payload_bytes": sum(row["split_gate_payload_bytes"] for row in rows) / len(rows),
        "avg_split_code_payload_bytes": sum(row["split_code_payload_bytes"] for row in rows) / len(rows),
        "avg_zero_code_count": sum(row["zero_code_count"] for row in rows) / len(rows),
        "avg_assignment_changed_count": sum(row["assignment_changed_count"] for row in rows) / len(rows),
        "avg_assignment_changed_fraction": (
            sum(row["assignment_changed_count"] for row in rows) / selected_candidate_total
            if selected_candidate_total
            else 0.0
        ),
        "avg_assignment_mean_relative_error": (
            sum(row["assignment_mean_relative_error"] * row["selected_candidate_count"] for row in rows)
            / selected_candidate_total
            if selected_candidate_total
            else 0.0
        ),
        "center_utility_min": float(center_utilities.min().item()) if center_utilities.numel() else 0.0,
        "center_utility_max": float(center_utilities.max().item()) if center_utilities.numel() else 0.0,
        "center_utility_mean": float(center_utilities.mean().item()) if center_utilities.numel() else 0.0,
        "avg_psnr_base_db": sum(row["psnr_base_db"] for row in rows) / len(rows),
        "avg_psnr_enhanced_db": sum(row["psnr_enhanced_db"] for row in rows) / len(rows),
        "avg_psnr_delta_db": sum(row["psnr_delta_db"] for row in rows) / len(rows),
        "compact_text_section": bool(args.compact_text_section),
        "bitstream_format": (
            "oscr_v0_1_mlicpp_base_plus_compact_latent_codebook_stub"
            if args.compact_text_section
            else "oscr_v0_1_mlicpp_base_plus_latent_codebook_stub"
        ),
        "note": "codebook is trained on this diagnostic split and treated as decoder-known model state, not transmitted side information",
    }
    (output_dir / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
