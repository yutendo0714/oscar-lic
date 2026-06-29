#!/usr/bin/env python3
"""Sanitize a NumPy feature cache by replacing non-finite values.

This is intended for diagnostic feature artifacts where an upstream optional
score column may be undefined. The script records counts before writing the
cleaned cache so invalid runs remain auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--nan-value", type=float, default=0.0)
    parser.add_argument("--posinf-value", type=float, default=0.0)
    parser.add_argument("--neginf-value", type=float, default=0.0)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    arrays = {key: np.asarray(data[key]) for key in data.files}
    summary = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "nan_value": float(args.nan_value),
        "posinf_value": float(args.posinf_value),
        "neginf_value": float(args.neginf_value),
        "arrays": {},
    }
    cleaned = {}
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating):
            nonfinite = ~np.isfinite(value)
            clean = np.nan_to_num(
                value,
                nan=args.nan_value,
                posinf=args.posinf_value,
                neginf=args.neginf_value,
            ).astype(value.dtype, copy=False)
            cleaned[key] = clean
            bad_cols = []
            if value.ndim == 2:
                bad_cols = np.where(nonfinite.any(axis=0))[0].astype(int).tolist()
            summary["arrays"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nan_count": int(np.isnan(value).sum()),
                "posinf_count": int(np.isposinf(value).sum()),
                "neginf_count": int(np.isneginf(value).sum()),
                "bad_columns": bad_cols,
            }
        else:
            cleaned[key] = value
            summary["arrays"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nan_count": 0,
                "posinf_count": 0,
                "neginf_count": 0,
                "bad_columns": [],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **cleaned)
    summary["output_sha256"] = sha256_file(args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
