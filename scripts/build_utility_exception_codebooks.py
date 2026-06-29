#!/usr/bin/env python3
"""Build augmented K64+K exception codebooks from train oracle-change residuals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
import yaml


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


def parse_seed_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    seed_text, path_text = value.split("=", 1)
    return int(seed_text), Path(path_text)


def group_key_from_row(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def build_partition_map(table: Path) -> dict[tuple[int, int, int, int], str]:
    result = {}
    for row in read_jsonl(table):
        key = group_key_from_row(row)
        result.setdefault(key, str(row.get("assignment_partition", row.get("split", "unknown"))))
    return result


def fit_centers(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    actual_k = min(k, len(vectors))
    if actual_k <= 0:
        return np.empty((0, vectors.shape[1] if vectors.ndim == 2 else 512), dtype=np.float32)
    if actual_k == len(vectors):
        return vectors.astype(np.float32).copy()
    model = KMeans(n_clusters=actual_k, random_state=seed, n_init=20, max_iter=500)
    model.fit(vectors.astype(np.float32))
    return model.cluster_centers_.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--base-codebook", type=parse_seed_path, action="append", required=True)
    parser.add_argument("--extra-centers", type=int, default=16)
    parser.add_argument("--kmeans-seed", type=int, default=138)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_utility_exception_k16_codebooks_2026_06_26")
    args = parser.parse_args()

    data = np.load(args.features)
    partition = build_partition_map(args.table)
    group_keys = np.asarray(data["group_keys"])
    mask = np.asarray(data["mask"]).astype(bool)
    labels = np.asarray(data["labels"])
    is_nearest = np.asarray(data["is_nearest"])
    residuals = np.asarray(data["latent_residual"], dtype=np.float32)
    train_vectors_by_seed: dict[int, list[np.ndarray]] = defaultdict(list)
    for gi, key_arr in enumerate(group_keys):
        key = tuple(int(v) for v in key_arr)
        if partition.get(key) != "train":
            continue
        valid = np.flatnonzero(mask[gi])
        oracle_slots = [slot for slot in valid if labels[gi, slot] > 0.5]
        nearest_slots = [slot for slot in valid if is_nearest[gi, slot] > 0.5]
        if not oracle_slots or not nearest_slots or int(oracle_slots[0]) == int(nearest_slots[0]):
            continue
        train_vectors_by_seed[key[0]].append(residuals[gi, int(oracle_slots[0])].astype(np.float32))

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for seed, base_path in args.base_codebook:
        base_npz = np.load(base_path)
        base = np.asarray(base_npz["codebook"], dtype=np.float32)
        base_util = np.asarray(base_npz["center_utility"], dtype=np.float32) if "center_utility" in base_npz.files else np.zeros(len(base), dtype=np.float32)
        vectors = np.stack(train_vectors_by_seed.get(seed, []), axis=0)
        extra = fit_centers(vectors, args.extra_centers, args.kmeans_seed + seed)
        extra_util = np.full((len(extra),), float(base_util.max() if len(base_util) else 0.0), dtype=np.float32)
        codebook = np.concatenate([base, extra], axis=0).astype(np.float32)
        center_utility = np.concatenate([base_util, extra_util], axis=0).astype(np.float32)
        out_dir = args.output_root / f"seed{seed}_k{len(codebook)}_extra{len(extra)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "latent_residual_codebook.npz"
        np.savez_compressed(
            out_path,
            codebook=codebook,
            center_utility=center_utility,
            tile=int(base_npz["tile"]) if "tile" in base_npz.files else 4,
            channel_group_size=int(base_npz["channel_group_size"]) if "channel_group_size" in base_npz.files else -1,
            residual_scale=float(base_npz["residual_scale"]) if "residual_scale" in base_npz.files else 0.25,
            codebook_quant_step=float(base_npz["codebook_quant_step"]) if "codebook_quant_step" in base_npz.files else 0.05,
            codebook_train_mode=f"utility_exception_k{args.extra_centers}",
            utility_weight_alpha=float(base_npz["utility_weight_alpha"]) if "utility_weight_alpha" in base_npz.files else -1.0,
            utility_protected_penalty=float(base_npz["utility_protected_penalty"]) if "utility_protected_penalty" in base_npz.files else -1.0,
            append_zero_code=False,
        )
        outputs.append(
            {
                "seed": seed,
                "path": str(out_path),
                "sha256": sha256_file(out_path),
                "base_codebook": str(base_path),
                "base_size": int(len(base)),
                "extra_size": int(len(extra)),
                "total_size": int(len(codebook)),
                "train_oracle_vectors": int(len(vectors)),
            }
        )

    result = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "code_commit": git_commit(),
        "features": str(args.features),
        "table": str(args.table),
        "extra_centers": args.extra_centers,
        "outputs": outputs,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    manifest_files = [
        {"name": "features", "path": str(args.features), "sha256": sha256_file(args.features)},
        {"name": "table", "path": str(args.table), "sha256": sha256_file(args.table)},
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "summary", "path": str(summary_path), "sha256": sha256_file(summary_path)},
    ]
    for _, base_path in args.base_codebook:
        manifest_files.append({"name": f"base_{base_path.parent.name}", "path": str(base_path), "sha256": sha256_file(base_path)})
    manifest_files.extend({"name": f"augmented_seed{row['seed']}", "path": row["path"], "sha256": row["sha256"]} for row in outputs)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        yaml.safe_dump(
            {
                "experiment_id": args.experiment_id,
                "status": "completed",
                "command": " ".join(["scripts/build_utility_exception_codebooks.py", *sys.argv[1:]]),
                "code_commit": git_commit(),
                "inputs_and_outputs": manifest_files,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
