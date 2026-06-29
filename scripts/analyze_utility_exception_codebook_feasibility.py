#!/usr/bin/env python3
"""Audit utility-positive residual exception centers for assignment feasibility."""

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


def build_group_metadata(table: Path) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    metadata: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(table):
        key = group_key_from_row(row)
        if key in metadata:
            continue
        metadata[key] = {
            "partition": str(row.get("assignment_partition", row.get("split", "unknown"))),
            "source": row.get("source"),
            "reference": row.get("reference"),
        }
    return metadata


def load_codebooks(items: list[tuple[int, Path]]) -> dict[int, np.ndarray]:
    result = {}
    for seed, path in items:
        data = np.load(path)
        result[seed] = np.asarray(data["codebook"], dtype=np.float32)
    return result


def row_distance(vector: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = centers.astype(np.float32) - vector.astype(np.float32)[None, :]
    return np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32))


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def fit_exception_centers(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    if len(vectors) == 0:
        return np.empty((0, 512), dtype=np.float32)
    actual_k = min(k, len(vectors))
    if actual_k == len(vectors):
        return vectors.astype(np.float32).copy()
    model = KMeans(n_clusters=actual_k, random_state=seed, n_init=20, max_iter=500)
    model.fit(vectors.astype(np.float32))
    return model.cluster_centers_.astype(np.float32)


def evaluate_split(
    *,
    group_keys: np.ndarray,
    mask: np.ndarray,
    is_nearest: np.ndarray,
    labels: np.ndarray,
    code_indices: np.ndarray,
    residuals: np.ndarray,
    tesseract_delta: np.ndarray,
    parseq_delta: np.ndarray,
    metadata: dict[tuple[int, int, int, int], dict[str, Any]],
    codebooks: dict[int, np.ndarray],
    exception_centers_by_seed: dict[int, np.ndarray],
    partition: str,
    thresholds: list[float],
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    oracle_rel_base: list[float] = []
    oracle_rel_aug: list[float] = []
    noheadroom_rel_aug: list[float] = []
    examples = []
    for gi, key_arr in enumerate(group_keys):
        key = tuple(int(v) for v in key_arr)
        meta = metadata.get(key, {})
        if meta.get("partition") != partition:
            continue
        seed = key[0]
        base = codebooks[seed]
        extra = exception_centers_by_seed.get(seed, np.empty((0, base.shape[1]), dtype=np.float32))
        augmented = np.concatenate([base, extra], axis=0) if len(extra) else base
        valid_slots = np.flatnonzero(mask[gi])
        nearest_slots = [slot for slot in valid_slots if is_nearest[gi, slot] > 0.5]
        if not nearest_slots:
            continue
        nearest_slot = int(nearest_slots[0])
        oracle_slots = [slot for slot in valid_slots if labels[gi, slot] > 0.5]
        if not oracle_slots:
            continue
        oracle_slot = int(oracle_slots[0])
        oracle_code = int(code_indices[gi, oracle_slot])
        nearest_is_oracle = oracle_slot == nearest_slot
        counts["groups"] += 1
        counts["oracle_change_groups"] += int(not nearest_is_oracle)
        vector = residuals[gi, oracle_slot].astype(np.float32)
        base_dist = row_distance(vector, base)
        base_nearest = float(base_dist.min())
        aug_dist = row_distance(vector, augmented)
        aug_argmin = int(np.argmin(aug_dist))
        aug_nearest = float(aug_dist[aug_argmin])
        aug_uses_exception = aug_argmin >= len(base)
        rel_aug = aug_nearest / max(base_nearest, 1e-8)
        if not nearest_is_oracle:
            oracle_code_dist = float(base_dist[oracle_code])
            # The base nearest distance is the denominator; the oracle row may not be the base nearest.
            oracle_rel_base.append(oracle_code_dist / max(base_nearest, 1e-8))
            oracle_rel_aug.append(rel_aug)
            counts["augmented_selects_exception_on_oracle_change"] += int(aug_uses_exception)
            for thr in thresholds:
                label = f"re{int(round(thr * 100)):03d}"
                counts[f"oracle_change_aug_eligible_{label}"] += int(rel_aug <= thr)
                counts[f"oracle_change_exception_close_{label}"] += int(aug_uses_exception and rel_aug <= thr)
        else:
            noheadroom_rel_aug.append(rel_aug)
            counts["augmented_selects_exception_on_noheadroom"] += int(aug_uses_exception)
            for thr in thresholds:
                label = f"re{int(round(thr * 100)):03d}"
                counts[f"noheadroom_aug_close_{label}"] += int(rel_aug <= thr and aug_uses_exception)
        if len(examples) < 25 and (not nearest_is_oracle) and aug_uses_exception:
            examples.append(
                {
                    "group_key": list(key),
                    "source": meta.get("source"),
                    "reference": meta.get("reference"),
                    "base_nearest_distance": base_nearest,
                    "augmented_nearest_distance": aug_nearest,
                    "augmented_relative": rel_aug,
                    "tesseract_delta_oracle_row": int(tesseract_delta[gi, oracle_slot]),
                    "parseq_delta_oracle_row": int(parseq_delta[gi, oracle_slot]),
                }
            )
    return {
        **{key: int(value) for key, value in sorted(counts.items())},
        "oracle_change_base_relative": summarize_values(oracle_rel_base),
        "oracle_change_augmented_relative": summarize_values(oracle_rel_aug),
        "noheadroom_augmented_relative": summarize_values(noheadroom_rel_aug),
        "oracle_exception_examples": examples,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    data = np.load(args.features)
    metadata = build_group_metadata(args.table)
    codebooks = load_codebooks(args.codebook)
    group_keys = np.asarray(data["group_keys"])
    mask = np.asarray(data["mask"]).astype(bool)
    is_nearest = np.asarray(data["is_nearest"])
    labels = np.asarray(data["labels"])
    code_indices = np.asarray(data["code_indices"])
    residuals = np.asarray(data["latent_residual"], dtype=np.float32)
    tesseract_delta = np.asarray(data["tesseract_delta"])
    parseq_delta = np.asarray(data["parseq_delta"])
    train_vectors_by_seed: dict[int, list[np.ndarray]] = defaultdict(list)
    for gi, key_arr in enumerate(group_keys):
        key = tuple(int(v) for v in key_arr)
        if metadata.get(key, {}).get("partition") != "train":
            continue
        oracle_slots = [slot for slot in np.flatnonzero(mask[gi]) if labels[gi, slot] > 0.5]
        nearest_slots = [slot for slot in np.flatnonzero(mask[gi]) if is_nearest[gi, slot] > 0.5]
        if not oracle_slots or not nearest_slots or int(oracle_slots[0]) == int(nearest_slots[0]):
            continue
        train_vectors_by_seed[key[0]].append(residuals[gi, int(oracle_slots[0])].astype(np.float32))
    train_counts = {str(seed): len(vectors) for seed, vectors in train_vectors_by_seed.items()}
    results = {}
    for k in args.extra_centers:
        exception_by_seed = {
            seed: fit_exception_centers(np.stack(vectors, axis=0), k, args.kmeans_seed + seed)
            for seed, vectors in train_vectors_by_seed.items()
            if vectors
        }
        label = f"extra{k}"
        results[label] = {
            "extra_centers_per_seed": {str(seed): int(len(centers)) for seed, centers in exception_by_seed.items()},
            "train": evaluate_split(
                group_keys=group_keys,
                mask=mask,
                is_nearest=is_nearest,
                labels=labels,
                code_indices=code_indices,
                residuals=residuals,
                tesseract_delta=tesseract_delta,
                parseq_delta=parseq_delta,
                metadata=metadata,
                codebooks=codebooks,
                exception_centers_by_seed=exception_by_seed,
                partition="train",
                thresholds=args.threshold,
            ),
            "val": evaluate_split(
                group_keys=group_keys,
                mask=mask,
                is_nearest=is_nearest,
                labels=labels,
                code_indices=code_indices,
                residuals=residuals,
                tesseract_delta=tesseract_delta,
                parseq_delta=parseq_delta,
                metadata=metadata,
                codebooks=codebooks,
                exception_centers_by_seed=exception_by_seed,
                partition="val",
                thresholds=args.threshold,
            ),
        }
    return {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-utility-exception-symbol-feasibility",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "inputs": {
            "features": str(args.features),
            "table": str(args.table),
            "codebooks": {str(seed): str(path) for seed, path in args.codebook},
            "extra_centers": args.extra_centers,
            "thresholds": args.threshold,
        },
        "train_oracle_change_vectors_per_seed": train_counts,
        "results": results,
        "conclusion": (
            "Utility-positive exception centers test whether changing the symbol proposal distribution can reduce "
            "the relative-error feasibility bottleneck observed in N137. This is a feasibility diagnostic only; "
            "new centers have no OCR evaluation until actual candidate reconstructions are generated."
        ),
        "next_action": (
            "If exception centers improve validation oracle feasibility without creating many close no-headroom "
            "opportunities, regenerate a small actual compact stream/OCR smoke for the best low-K setting."
        ),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Utility Exception Codebook Feasibility",
        "",
        "This audit appends train oracle-change residual exception centers to the K64 utility-weighted codebook and checks distance feasibility only.",
        "It does not claim OCR improvement because the new centers have not been decoded and evaluated.",
        "",
        f"Train oracle-change vectors per seed: `{result['train_oracle_change_vectors_per_seed']}`.",
        "",
        "| extra centers | val oracle exception close re101 | val oracle exception close re105 | val exception chosen on oracle | val exception close no-headroom re101 | val exception close no-headroom re105 | val aug rel median |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in result["results"].items():
        val = data["val"]
        med = val["oracle_change_augmented_relative"]["median"]
        lines.append(
            f"| {label.replace('extra', '')} | {val.get('oracle_change_exception_close_re101', 0)} | "
            f"{val.get('oracle_change_exception_close_re105', 0)} | "
            f"{val.get('augmented_selects_exception_on_oracle_change', 0)} | "
            f"{val.get('noheadroom_aug_close_re101', 0)} | {val.get('noheadroom_aug_close_re105', 0)} | "
            f"{med:.4f} |"
        )
    lines.extend(["", "## Conclusion", "", result["conclusion"], "", "## Next Action", "", result["next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "features", "path": str(args.features), "sha256": sha256_file(args.features)},
        {"name": "table", "path": str(args.table), "sha256": sha256_file(args.table)},
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for seed, codebook in args.codebook:
        files.append({"name": f"codebook_seed{seed}", "path": str(codebook), "sha256": sha256_file(codebook)})
    data = {
        "experiment_id": args.experiment_id,
        "status": "completed",
        "command": " ".join(["scripts/analyze_utility_exception_codebook_feasibility.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--codebook", type=parse_seed_path, action="append", required=True)
    parser.add_argument("--extra-centers", type=int, action="append", default=[4, 8, 16, 32, 64])
    parser.add_argument("--threshold", type=float, action="append", default=[1.01, 1.05])
    parser.add_argument("--kmeans-seed", type=int, default=137)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_utility_exception_codebook_feasibility_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.output, args.report)
    print(json.dumps({"output": str(args.output), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
