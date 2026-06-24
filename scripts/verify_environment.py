#!/usr/bin/env python3
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oscarlic.manifests import basic_environment, run_command, utc_now_iso  # noqa: E402

PACKAGES = ["PyYAML", "jsonschema", "numpy", "Pillow", "pytest", "torch", "torchvision", "compressai"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture OSCAR-LIC execution environment")
    parser.add_argument("--output", type=Path, default=ROOT / "environment/resolved_environment.json")
    args = parser.parse_args()

    info = basic_environment()
    info["packages"] = {}
    for package in PACKAGES:
        try:
            info["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info["packages"][package] = None
    info["commands"] = {
        "git": run_command(["git", "--version"]),
        "nvidia_smi": run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": run_command(["nvcc", "--version"]),
        "conda": run_command(["conda", "--version"]),
    }
    info["executables"] = {name: shutil.which(name) for name in ["git", "nvidia-smi", "nvcc", "conda", "ffmpeg"]}
    try:
        import torch
        info["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        }
    except Exception as exc:
        info["torch_runtime"] = {"available": False, "error": str(exc)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
