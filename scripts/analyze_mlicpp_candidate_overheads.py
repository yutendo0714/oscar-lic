#!/usr/bin/env python3
"""Estimate concrete OSCAR gate/enhancement bytes from MLIC++ latent shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.candidates import LatentCandidateLayout, gate_payload_bytes  # noqa: E402


MLICPP_SLICE_NUM = 10
MLICPP_SLICE_CHANNELS = 32
TEXT_GATE_SECTION_TABLE_BYTES = 40
TEXT_MAIN_SECTION_TABLE_BYTES = 40


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_image_paths(split: Path) -> list[Path]:
    paths: list[Path] = []
    for line_no, line in enumerate(split.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            record = json.loads(line)
            value = record.get("image_path") or record.get("image") or record.get("path")
            if value is None:
                raise ValueError(f"{split}:{line_no}: record has no image path")
            paths.append(Path(value))
        else:
            paths.append(Path(line))
    if not paths:
        raise ValueError(f"{split} contains no image paths")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing images: {missing[:5]}")
    return paths


def latent_shape_for_image(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    pad_h = 0 if height % 64 == 0 else 64 * (height // 64 + 1) - height
    pad_w = 0 if width % 64 == 0 else 64 * (width // 64 + 1) - width
    return {
        "image": project_path(path),
        "width": width,
        "height": height,
        "pixels": width * height,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "latent_height": (height + pad_h) // 16,
        "latent_width": (width + pad_w) // 16,
    }


def selected_count(candidate_count: int, fraction: float) -> int:
    count = round(candidate_count * fraction)
    if fraction > 0 and count == 0:
        count = 1
    return max(0, min(candidate_count, count))


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def parse_channel_group(value: str) -> int | None:
    if value.lower() in {"none", "slice", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("channel group size must be positive or none")
    return parsed


def run(args: argparse.Namespace) -> dict:
    image_paths = read_image_paths(args.split)
    images = [latent_shape_for_image(path) for path in image_paths]
    configs = []
    for tile in args.tiles:
        for channel_group_size in args.channel_group_sizes:
            for fraction in args.selected_fractions:
                rows = []
                for image in images:
                    layout = LatentCandidateLayout(
                        latent_height=image["latent_height"],
                        latent_width=image["latent_width"],
                        slice_num=MLICPP_SLICE_NUM,
                        slice_channels=MLICPP_SLICE_CHANNELS,
                        tile_height=tile,
                        tile_width=tile,
                        channel_group_size=channel_group_size,
                    )
                    chosen = selected_count(layout.candidate_count, fraction)
                    gate = gate_payload_bytes(num_candidates=layout.candidate_count, selected_indices=range(chosen))
                    enhancement_payload_bytes = args.enhancement_global_bytes + args.enhancement_bytes_per_selected * chosen
                    optional_section_table_bytes = TEXT_GATE_SECTION_TABLE_BYTES
                    if args.include_text_main:
                        optional_section_table_bytes += TEXT_MAIN_SECTION_TABLE_BYTES
                    optional_payload_bytes = gate["total_gate_payload_bytes"]
                    if args.include_text_main:
                        optional_payload_bytes += enhancement_payload_bytes
                    optional_total_bytes = optional_section_table_bytes + optional_payload_bytes
                    rows.append(
                        {
                            **image,
                            "tile": tile,
                            "channel_group_size": channel_group_size,
                            "selected_fraction": fraction,
                            "candidate_count": layout.candidate_count,
                            "selected_candidate_count": chosen,
                            "gate_payload_bytes": gate["total_gate_payload_bytes"],
                            "gate_mode": gate["mode"],
                            "enhancement_payload_bytes": enhancement_payload_bytes,
                            "optional_section_table_bytes": optional_section_table_bytes,
                            "optional_total_bytes": optional_total_bytes,
                            "optional_total_bpp": 8.0 * optional_total_bytes / image["pixels"],
                        }
                    )
                configs.append(
                    {
                        "tile": tile,
                        "channel_group_size": channel_group_size,
                        "selected_fraction": fraction,
                        "include_text_main": args.include_text_main,
                        "enhancement_global_bytes": args.enhancement_global_bytes,
                        "enhancement_bytes_per_selected": args.enhancement_bytes_per_selected,
                        "candidate_count": summarize([row["candidate_count"] for row in rows]),
                        "selected_candidate_count": summarize([row["selected_candidate_count"] for row in rows]),
                        "gate_payload_bytes": summarize([row["gate_payload_bytes"] for row in rows]),
                        "enhancement_payload_bytes": summarize([row["enhancement_payload_bytes"] for row in rows]),
                        "optional_total_bytes": summarize([row["optional_total_bytes"] for row in rows]),
                        "optional_total_bpp": summarize([row["optional_total_bpp"] for row in rows]),
                        "gate_modes": sorted(set(row["gate_mode"] for row in rows)),
                    }
                )

    return {
        "split": project_path(args.split),
        "images": len(images),
        "section_table_bytes": {
            "text_gate": TEXT_GATE_SECTION_TABLE_BYTES,
            "text_main": TEXT_MAIN_SECTION_TABLE_BYTES if args.include_text_main else 0,
        },
        "configs": configs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--channel-group-sizes", type=parse_channel_group, nargs="+", default=[None])
    parser.add_argument("--selected-fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--include-text-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enhancement-global-bytes", type=int, default=40)
    parser.add_argument("--enhancement-bytes-per-selected", type=int, default=40)
    args = parser.parse_args()

    if any(tile <= 0 for tile in args.tiles):
        raise SystemExit("--tiles must be positive")
    if any(fraction < 0 or fraction > 1 for fraction in args.selected_fractions):
        raise SystemExit("--selected-fractions must be in [0, 1]")
    if args.enhancement_global_bytes < 0 or args.enhancement_bytes_per_selected < 0:
        raise SystemExit("enhancement byte counts must be nonnegative")

    output = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": output["images"], "configs": len(output["configs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
