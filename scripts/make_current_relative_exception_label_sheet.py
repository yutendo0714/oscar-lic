#!/usr/bin/env python3
"""Create contact sheets from current-relative exception label rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    labels = set(args.strict_label or [])
    actions = set(args.action or [])
    out = []
    for row in rows:
        if labels and row["strict_label"] not in labels:
            continue
        if actions and row["action_taken"] not in actions:
            continue
        out.append(row)
    return out[: args.max_rows]


def crop_text(text: str, limit: int = 84) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def load_pair_images(row: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    current_path = resolve_path(row["current_stream"]["reconstruction_path"])
    exception_path = resolve_path(row["exception_stream"]["reconstruction_path"])
    return Image.open(current_path).convert("RGB"), Image.open(exception_path).convert("RGB")


def scaled(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / max(image.width, 1), max_h / max(image.height, 1), 2.0)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


def panel(row: dict[str, Any], args: argparse.Namespace, font: ImageFont.ImageFont, small: ImageFont.ImageFont) -> Image.Image:
    cur_img, exc_img = load_pair_images(row)
    cur_img = scaled(cur_img, 240, 80)
    exc_img = scaled(exc_img, 240, 80)
    width = 620
    height = 190
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    color = (170, 25, 25)
    if row["strict_label"] == "beneficial_no_profile_harm":
        color = (25, 125, 55)
    elif row["strict_label"] == "neutral_all_profiles":
        color = (80, 80, 80)
    elif row["strict_label"] == "mixed_harm_and_benefit":
        color = (150, 90, 20)
    draw.rectangle((0, 0, width - 1, height - 1), outline=color, width=2)
    row_id = row.get("global_index", row.get("probe_index", "?"))
    draw.text((10, 8), f"idx {row_id} seed {row['seed']} src {row['source_index']} | {row['action_taken']} | {row['strict_label']}", fill=color, font=font)
    draw.text((10, 30), "current", fill=(0, 0, 0), font=small)
    draw.text((320, 30), "exception/base", fill=(0, 0, 0), font=small)
    canvas.paste(cur_img, (10, 50))
    canvas.paste(exc_img, (320, 50))
    model = args.model
    profile = args.profile
    cur_pred = row["ocr"][model]["current_prediction"]
    exc_pred = row["ocr"][model]["exception_prediction"]
    delta = row["ocr"][model][profile]["delta_errors"]
    latin_delta = row["ocr"][model]["latin_alnum_ci_v1"]["delta_errors"]
    text_y = 134
    lines = [
        f"ref: {crop_text(row['reference'])}",
        f"cur: {crop_text(cur_pred)}",
        f"exc: {crop_text(exc_pred)}",
        f"{model} {profile} delta {delta:+d}; latin {latin_delta:+d}; bytes {row['rate_delta']['actual_total_bytes']:+d}",
    ]
    for line in lines:
        draw.text((10, text_y), line, fill=(0, 0, 0), font=small)
        text_y += 13
    return canvas


def build_sheet(args: argparse.Namespace) -> dict[str, Any]:
    rows = filter_rows(read_jsonl(args.labels), args)
    font = load_font(14)
    small = load_font(12)
    panels = [panel(row, args, font, small) for row in rows]
    cols = args.columns
    panel_w = 620
    panel_h = 190
    sheet_rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * panel_w, max(1, sheet_rows) * panel_h), "white")
    for i, item in enumerate(panels):
        sheet.paste(item, ((i % cols) * panel_w, (i // cols) * panel_h))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    return {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "labels": str(args.labels),
        "output": str(args.output),
        "rows": len(rows),
        "filters": {
            "strict_label": args.strict_label or [],
            "action": args.action or [],
            "model": args.model,
            "profile": args.profile,
        },
        "sha256": sha256_file(args.output),
    }


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, result: dict[str, Any]) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "labels", "path": str(args.labels), "sha256": sha256_file(args.labels)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "sheet", "path": str(args.output), "sha256": sha256_file(args.output)},
    ]
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic",
        "command": " ".join(["scripts/make_current_relative_exception_label_sheet.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--strict-label", action="append", default=None)
    parser.add_argument("--action", action="append", default=None)
    parser.add_argument("--model", choices=["tesseract", "parseq"], default="tesseract")
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_exception_current_relative_label_sheet_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_sheet(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(args.manifest, args, args.result, result)
    print(json.dumps({"result": str(args.result), "sheet": str(args.output), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
