#!/usr/bin/env python3
"""Create contact sheets for the explicit exception-center OCR diagnostic."""

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
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts  # noqa: E402


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


def resolve_image(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def selected_indices(policy_dir: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for path in sorted(policy_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            seed = int(row["real_seed"])
            offset = {1: 0, 2: 75}.get(seed)
            if offset is None:
                continue
            result[offset + int(row["source_index"])] = row
    return result


def load_font() -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), 14)
    return ImageFont.load_default()


def make_panel(
    *,
    current: dict[str, Any],
    exception: dict[str, Any],
    policy: dict[str, Any],
    profile: str,
    font: ImageFont.ImageFont,
) -> Image.Image:
    cur_img = Image.open(resolve_image(current["image"])).convert("RGB")
    exc_img = Image.open(resolve_image(exception["image"])).convert("RGB")
    scale = min(220 / max(cur_img.width, exc_img.width, 1), 70 / max(cur_img.height, exc_img.height, 1), 2.0)
    size_cur = (max(1, int(cur_img.width * scale)), max(1, int(cur_img.height * scale)))
    size_exc = (max(1, int(exc_img.width * scale)), max(1, int(exc_img.height * scale)))
    cur_img = cur_img.resize(size_cur, Image.Resampling.BICUBIC)
    exc_img = exc_img.resize(size_exc, Image.Resampling.BICUBIC)
    cur_dist = character_counts(current["reference"], current["prediction"], profile).distance
    exc_dist = character_counts(exception["reference"], exception["prediction"], profile).distance
    delta = exc_dist - cur_dist
    panel_w = 520
    panel_h = 155
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=(180, 180, 180))
    panel.paste(cur_img, (10, 28))
    panel.paste(exc_img, (270, 28))
    draw.text((10, 8), "current", fill=(0, 0, 0), font=font)
    draw.text((270, 8), "exception", fill=(0, 0, 0), font=font)
    color = (20, 120, 40) if delta < 0 else (180, 30, 30) if delta > 0 else (70, 70, 70)
    text_lines = [
        f"ref: {current['reference']}",
        f"cur: {current['prediction']}",
        f"exc: {exception['prediction']}",
        f"delta {delta:+d}; proxy_oracle={policy.get('oracle_change_proxy')}; rel={float(policy.get('extra_relative_error', 0.0)):.3f}",
    ]
    y = 98
    for line in text_lines:
        draw.text((10, y), line[:86], fill=color if line.startswith("delta") else (0, 0, 0), font=font)
        y += 14
    return panel


def create_sheet(args: argparse.Namespace) -> dict[str, Any]:
    current_rows = read_jsonl(args.current_tesseract_results)
    exception_rows = read_jsonl(args.exception_tesseract_results)
    selected = selected_indices(args.policy_dir)
    font = load_font()
    panels = [
        make_panel(
            current=current_rows[index],
            exception=exception_rows[index],
            policy=selected[index],
            profile=args.profile,
            font=font,
        )
        for index in sorted(selected)
    ]
    cols = args.columns
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 520, rows * 155), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * 520, (i // cols) * 155))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    return {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "output": str(args.output),
        "rows": len(panels),
        "profile": args.profile,
        "sha256": sha256_file(args.output),
    }


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, result: dict[str, Any]) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "sheet", "path": str(args.output), "sha256": sha256_file(args.output)},
        {"name": "current_tesseract_results", "path": str(args.current_tesseract_results), "sha256": sha256_file(args.current_tesseract_results)},
        {"name": "exception_tesseract_results", "path": str(args.exception_tesseract_results), "sha256": sha256_file(args.exception_tesseract_results)},
    ]
    for path_item in sorted(args.policy_dir.glob("*.jsonl")):
        files.append({"name": path_item.name, "path": str(path_item), "sha256": sha256_file(path_item)})
    data = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "command": " ".join(["scripts/make_exception_center_contact_sheet.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--current-tesseract-results", type=Path, required=True)
    parser.add_argument("--exception-tesseract-results", type=Path, required=True)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_exception_center_rel100_contact_sheet_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = create_sheet(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(args.manifest, args, args.result, result)
    print(json.dumps({"result": str(args.result), "sheet": str(args.output), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
