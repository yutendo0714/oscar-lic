#!/usr/bin/env python3
"""Paired bootstrap comparison between two quality-selection policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, normalize_text  # noqa: E402


def parse_q_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected q=path")
    q, path = value.split("=", 1)
    return int(q), Path(path)


def parse_eval_q_path(value: str) -> tuple[str, int, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected label:q=path")
    left, path = value.split("=", 1)
    label, q = left.split(":", 1)
    return label, int(q), Path(path)


def parse_selection(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected label=path_or_quality")
    label, spec = value.split("=", 1)
    return label, spec


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_by_quality(items: list[tuple[int, Path]]) -> dict[int, list[dict]]:
    return {q: read_jsonl(path) for q, path in sorted(items)}


def load_eval(items: list[tuple[str, int, Path]]) -> dict[str, dict[int, list[dict]]]:
    result: dict[str, dict[int, list[dict]]] = {}
    for label, q, path in items:
        result.setdefault(label, {})[q] = read_jsonl(path)
    return result


def resolve_selection(spec: str, n: int) -> list[int]:
    path = Path(spec)
    if path.is_file():
        rows = read_jsonl(path)
        if len(rows) != n:
            raise ValueError(f"selection rows {len(rows)} != {n}")
        return [int(row["selected_quality"]) for row in rows]
    return [int(spec) for _ in range(n)]


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def selection_stats(
    *,
    indices: list[int],
    selected: list[int],
    codec: dict[int, list[dict]],
    eval_rows: dict[int, list[dict]],
    profile: str,
    base_q: int,
    per_selected_overhead_bytes: int,
    global_overhead_bytes: int,
) -> dict:
    errors = 0
    ref_chars = 0
    exact = 0
    bytes_total = 0
    pixels_total = 0
    selected_count = 0
    for idx in indices:
        q = selected[idx]
        codec_row = codec[q][idx]
        eval_row = eval_rows[q][idx]
        counts = character_counts(eval_row["reference"], eval_row["prediction"], profile)
        errors += counts.distance
        ref_chars += counts.reference_length
        exact += int(normalize_text(eval_row["reference"], profile) == normalize_text(eval_row["prediction"], profile))
        bytes_total += codec_row["actual_total_bytes"]
        pixels_total += codec_row["width"] * codec_row["height"]
        selected_count += int(q > base_q)
    overhead = selected_count * per_selected_overhead_bytes + (global_overhead_bytes if selected_count else 0)
    bytes_total += overhead
    return {
        "cer": errors / ref_chars if ref_chars else 0.0,
        "exact_rate": exact / len(indices) if indices else 0.0,
        "bpp": 8.0 * bytes_total / pixels_total,
        "bytes": bytes_total,
        "overhead_bytes": overhead,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-result", type=parse_q_path, action="append", required=True)
    parser.add_argument("--eval-result", type=parse_eval_q_path, action="append", required=True)
    parser.add_argument("--selection-a", type=parse_selection, required=True)
    parser.add_argument("--selection-b", type=parse_selection, required=True)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--per-selected-overhead-bytes", type=int, default=40)
    parser.add_argument("--global-overhead-bytes", type=int, default=40)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    codec = load_by_quality(args.codec_result)
    qualities = sorted(codec)
    n = len(codec[qualities[0]])
    evals = load_eval(args.eval_result)
    selection_a_label, selection_a_spec = args.selection_a
    selection_b_label, selection_b_spec = args.selection_b
    selected_a = resolve_selection(selection_a_spec, n)
    selected_b = resolve_selection(selection_b_spec, n)

    rng = random.Random(args.seed)
    all_indices = list(range(n))
    output = {
        "selection_a": selection_a_label,
        "selection_b": selection_b_label,
        "profile": args.profile,
        "samples": n,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "per_selected_overhead_bytes": args.per_selected_overhead_bytes,
        "global_overhead_bytes": args.global_overhead_bytes,
        "evaluators": {},
    }

    for eval_label, eval_rows in evals.items():
        observed_a = selection_stats(
            indices=all_indices,
            selected=selected_a,
            codec=codec,
            eval_rows=eval_rows,
            profile=args.profile,
            base_q=qualities[0],
            per_selected_overhead_bytes=args.per_selected_overhead_bytes,
            global_overhead_bytes=args.global_overhead_bytes,
        )
        observed_b = selection_stats(
            indices=all_indices,
            selected=selected_b,
            codec=codec,
            eval_rows=eval_rows,
            profile=args.profile,
            base_q=qualities[0],
            per_selected_overhead_bytes=0,
            global_overhead_bytes=0,
        )
        deltas = {"cer": [], "exact_rate": [], "bpp": []}
        for _ in range(args.bootstrap_samples):
            indices = [rng.randrange(n) for _ in range(n)]
            stat_a = selection_stats(
                indices=indices,
                selected=selected_a,
                codec=codec,
                eval_rows=eval_rows,
                profile=args.profile,
                base_q=qualities[0],
                per_selected_overhead_bytes=args.per_selected_overhead_bytes,
                global_overhead_bytes=args.global_overhead_bytes,
            )
            stat_b = selection_stats(
                indices=indices,
                selected=selected_b,
                codec=codec,
                eval_rows=eval_rows,
                profile=args.profile,
                base_q=qualities[0],
                per_selected_overhead_bytes=0,
                global_overhead_bytes=0,
            )
            for key in deltas:
                deltas[key].append(stat_a[key] - stat_b[key])
        output["evaluators"][eval_label] = {
            "observed_a": observed_a,
            "observed_b": observed_b,
            "delta_a_minus_b": {
                key: {
                    "observed": observed_a[key] - observed_b[key],
                    "ci95": [quantile(values, 0.025), quantile(values, 0.975)],
                }
                for key, values in deltas.items()
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
