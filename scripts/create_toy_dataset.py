#!/usr/bin/env python3
"""Create a deterministic, project-owned smoke dataset.

The images are not a scientific benchmark and must never appear in final claims.
They exist to test file I/O, bit accounting, OCR adapters and manifests before
licensed datasets are available.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STRINGS = [
    "TOTAL 123.45",
    "ROOM B-208",
    "OPEN 24H",
    "ID A7C9",
    "DATE 2026-06-24",
]


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/toy"))
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    font = ImageFont.load_default(size=24)
    manifest = []

    for index, text in enumerate(STRINGS):
        width, height = 384, 160
        base = np.full((height, width, 3), 242, dtype=np.int16)
        gradient = np.linspace(-12, 12, width, dtype=np.float32)[None, :, None]
        noise = rng.normal(0, 2.0 + index * 0.5, size=base.shape)
        array = np.clip(base + gradient + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        draw = ImageDraw.Draw(image)
        # Non-text visual content ensures smoke tests do not exercise a blank canvas only.
        draw.rectangle((12, 12, width - 13, height - 13), outline=(70, 70, 70), width=2)
        draw.line((20, 120, width - 20, 120), fill=(130, 130, 130), width=1)
        x = 28 + py_rng.randint(0, 12)
        y = 45 + py_rng.randint(-5, 5)
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rounded_rectangle((bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6), radius=4, fill=(255, 255, 255))
        draw.text((x, y), text, font=font, fill=(15, 15, 15))
        path = args.output / f"toy_{index:02d}.png"
        image.save(path, format="PNG", optimize=False)
        manifest.append({
            "sample_id": f"toy_{index:02d}",
            "dataset_id": "oscar_toy_v1",
            "split": "smoke",
            "relative_path": path.name,
            "width": width,
            "height": height,
            "sha256": file_hash(path),
            "annotations": [{"region_id": "r0", "bbox_xyxy": list(bbox), "transcription": text, "script": "Latn", "language": "en"}],
            "scientific_use": "forbidden",
        })

    manifest_path = args.output / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8")
    metadata = {
        "dataset_id": "oscar_toy_v1", "generator": "scripts/create_toy_dataset.py",
        "seed": args.seed, "samples": len(manifest), "scientific_use": "smoke_test_only",
    }
    (args.output / "README.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} images and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
