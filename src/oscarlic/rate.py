"""Actual-byte rate accounting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


def validate_dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("width and height must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")


def bpp_from_bytes(total_bytes: int, width: int, height: int) -> float:
    validate_dimensions(width, height)
    if not isinstance(total_bytes, int):
        raise TypeError("total_bytes must be an integer")
    if total_bytes < 0:
        raise ValueError("total_bytes cannot be negative")
    return 8.0 * total_bytes / (width * height)


def file_bpp(path: str | Path, width: int, height: int) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    total_bytes = path.stat().st_size
    return {
        "path": str(path),
        "actual_total_bytes": total_bytes,
        "actual_total_bpp": bpp_from_bytes(total_bytes, width, height),
        "width": width,
        "height": height,
    }


def section_rate_breakdown(
    sections: Mapping[str, int], width: int, height: int, total_bytes: int | None = None
) -> dict:
    validate_dimensions(width, height)
    clean: dict[str, int] = {}
    for name, value in sections.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid byte count for {name!r}: {value!r}")
        clean[str(name)] = value
    section_sum = sum(clean.values())
    if total_bytes is not None and section_sum != total_bytes:
        raise ValueError(f"section sum {section_sum} != total_bytes {total_bytes}")
    return {
        "actual_total_bytes": section_sum,
        "actual_total_bpp": bpp_from_bytes(section_sum, width, height),
        "sections": {
            name: {"bytes": value, "bpp": bpp_from_bytes(value, width, height)}
            for name, value in clean.items()
        },
    }
