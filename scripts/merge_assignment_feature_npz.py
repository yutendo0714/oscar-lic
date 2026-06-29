#!/usr/bin/env python3
"""Merge assignment feature NPZ artifacts along the group axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_filter(spec: str | None) -> tuple[int, set[int]] | None:
    if not spec:
        return None
    if "=" not in spec:
        raise SystemExit(f"filter must be KEY_INDEX=VALUE[,VALUE...], got: {spec}")
    key_text, values_text = spec.split("=", 1)
    key_index = int(key_text)
    values = {int(value) for value in values_text.split(",") if value}
    if not values:
        raise SystemExit(f"empty filter values in: {spec}")
    return key_index, values


def select_groups(data: dict[str, np.ndarray], selector: tuple[int, set[int]] | None) -> dict[str, np.ndarray]:
    if selector is None:
        return data
    key_index, values = selector
    group_keys = data["group_keys"]
    keep = np.asarray([int(row[key_index]) in values for row in group_keys], dtype=bool)
    return {key: value[keep] for key, value in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        nargs="+",
        metavar=("NPZ", "FILTER"),
        required=True,
        help="Input NPZ plus optional key filter, e.g. artifact.npz 1=1,2.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    chunks = []
    inputs = []
    common_keys: set[str] | None = None
    for item in args.input:
        path = Path(item[0])
        selector = parse_filter(item[1] if len(item) > 1 else None)
        with np.load(path) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        selected = select_groups(data, selector)
        if common_keys is None:
            common_keys = set(selected)
        else:
            common_keys &= set(selected)
        chunks.append(selected)
        inputs.append(
            {
                "path": str(path),
                "filter": item[1] if len(item) > 1 else None,
                "groups": int(selected["group_keys"].shape[0]),
            }
        )

    if common_keys is None:
        raise SystemExit("no inputs")
    keys = sorted(common_keys)
    if not keys:
        raise SystemExit("inputs have no common arrays")
    merged = {}
    for key in keys:
        shapes = [chunk[key].shape[1:] for chunk in chunks]
        dtypes = [chunk[key].dtype for chunk in chunks]
        if len(set(shapes)) != 1:
            raise SystemExit(f"array {key} has incompatible non-group shapes: {shapes}")
        if len(set(str(dtype) for dtype in dtypes)) != 1:
            raise SystemExit(f"array {key} has incompatible dtypes: {dtypes}")
        merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)
    metadata_path = args.metadata_output or args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata = {
        "description": "Merged assignment feature NPZ artifact.",
        "inputs": inputs,
        "output": str(args.output),
        "arrays": {key: list(value.shape) for key, value in merged.items()},
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
