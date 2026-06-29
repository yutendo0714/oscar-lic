#!/usr/bin/env python3
"""Augment top-k assignment utility rows with codebook-center geometry features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_codebook(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    output = {"codebook": np.asarray(data["codebook"], dtype=np.float32)}
    if "center_utility" in data.files:
        output["center_utility"] = np.asarray(data["center_utility"], dtype=np.float32)
    else:
        output["center_utility"] = np.zeros((output["codebook"].shape[0],), dtype=np.float32)
    return output


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def add_onehot(row: dict, prefix: str, index: int, size: int) -> None:
    for value in range(size):
        row[f"{prefix}_{value:03d}"] = int(value == index)


def augment_row(row: dict, codebooks: dict[int, dict[str, np.ndarray]], include_onehot: bool) -> dict:
    seed = int(row.get("real_seed", row["seed"]))
    if seed not in codebooks:
        raise SystemExit(f"missing codebook for seed {seed}")
    codebook = codebooks[seed]["codebook"]
    center_utility = codebooks[seed]["center_utility"]
    code_index = int(row["code_index"])
    nearest_index = int(row["nearest_code"])
    code = codebook[code_index]
    nearest = codebook[nearest_index]
    delta = code - nearest
    out = dict(row)
    out.update(
        {
            "codebook_size": int(codebook.shape[0]),
            "codebook_dim": int(codebook.shape[1]),
            "codebook_code_index_norm": code_index / max(codebook.shape[0] - 1, 1),
            "codebook_nearest_index_norm": nearest_index / max(codebook.shape[0] - 1, 1),
            "codebook_code_center_utility": float(center_utility[code_index]),
            "codebook_nearest_center_utility": float(center_utility[nearest_index]),
            "codebook_delta_center_utility": float(center_utility[code_index] - center_utility[nearest_index]),
            "codebook_code_l2": float(np.linalg.norm(code)),
            "codebook_nearest_l2": float(np.linalg.norm(nearest)),
            "codebook_delta_l2": float(np.linalg.norm(delta)),
            "codebook_code_mean": float(code.mean()),
            "codebook_nearest_mean": float(nearest.mean()),
            "codebook_delta_mean": float(delta.mean()),
            "codebook_code_rms": float(np.sqrt(np.mean(code * code))),
            "codebook_nearest_rms": float(np.sqrt(np.mean(nearest * nearest))),
            "codebook_delta_rms": float(np.sqrt(np.mean(delta * delta))),
            "codebook_code_abs_mean": float(np.abs(code).mean()),
            "codebook_nearest_abs_mean": float(np.abs(nearest).mean()),
            "codebook_delta_abs_mean": float(np.abs(delta).mean()),
            "codebook_code_max_abs": float(np.abs(code).max()),
            "codebook_nearest_max_abs": float(np.abs(nearest).max()),
            "codebook_delta_max_abs": float(np.abs(delta).max()),
            "codebook_code_nearest_cosine": cosine(code, nearest),
            "codebook_code_nearest_dot": float(np.dot(code, nearest)),
        }
    )
    if include_onehot:
        add_onehot(out, "codebook_code_onehot", code_index, codebook.shape[0])
        add_onehot(out, "codebook_nearest_onehot", nearest_index, codebook.shape[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebook", action="append", nargs=2, metavar=("SEED", "NPZ"), required=True)
    parser.add_argument("--include-onehot", action="store_true")
    args = parser.parse_args()

    codebooks = {int(seed): load_codebook(Path(path)) for seed, path in args.codebook}
    rows = [augment_row(row, codebooks, args.include_onehot) for row in read_jsonl(args.input)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "description": "Assignment utility table augmented with codebook-center geometry features.",
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(rows),
        "codebook_seeds": sorted(codebooks),
        "include_onehot": bool(args.include_onehot),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
