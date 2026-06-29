#!/usr/bin/env python3
"""Non-oracle quality-allocation controls for actual-byte OCR smoke studies."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import evaluate_pairs  # noqa: E402


def parse_q_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected q=path")
    key, path = value.split("=", 1)
    return int(key), Path(path)


def parse_eval_q_path(value: str) -> tuple[str, int, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected label:q=path")
    left, path = value.split("=", 1)
    label, q = left.split(":", 1)
    return label, int(q), Path(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_by_quality(items: list[tuple[int, Path]]) -> dict[int, list[dict]]:
    return {q: read_jsonl(path) for q, path in sorted(items)}


def assert_aligned(codec: dict[int, list[dict]], evals: dict[str, dict[int, list[dict]]]) -> None:
    qualities = sorted(codec)
    n = len(codec[qualities[0]])
    for q in qualities:
        if len(codec[q]) != n:
            raise ValueError(f"codec q{q}: row count mismatch")
        for label, by_q in evals.items():
            if len(by_q[q]) != n:
                raise ValueError(f"{label} q{q}: row count mismatch")
    for idx in range(n):
        reference = None
        for label, by_q in evals.items():
            for q in qualities:
                row_ref = by_q[q][idx]["reference"]
                reference = row_ref if reference is None else reference
                if row_ref != reference:
                    raise ValueError(f"reference mismatch at row {idx}, {label}, q{q}")


def read_confidences(path: Path | None, expected_rows: int) -> list[float]:
    if path is None:
        return [0.5 for _ in range(expected_rows)]
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(f"confidence rows {len(rows)} != expected {expected_rows}")
    if any("confidence" not in row for row in rows):
        raise ValueError("confidence source must include confidence in every row")
    return [float(row["confidence"]) for row in rows]


def policy_score(policy: str, confidence: float, rng: random.Random, idx: int, q: int) -> float:
    if policy == "random":
        return rng.random()
    if policy == "low_confidence":
        return 1.0 - confidence
    if policy == "high_confidence":
        return confidence
    if policy == "uniform_cost":
        return 1.0
    raise ValueError(f"unsupported policy: {policy}")


def select_by_policy(
    *,
    policy: str,
    target_bytes: int,
    qualities: list[int],
    codec: dict[int, list[dict]],
    confidences: list[float],
    seed: int,
    per_selected_overhead_bytes: int,
    global_overhead_bytes: int,
) -> list[int]:
    rng = random.Random(seed)
    base_q = qualities[0]
    selected = [base_q for _ in codec[base_q]]
    current_bytes = sum(row["actual_total_bytes"] for row in codec[base_q])
    enhanced_count = 0

    while True:
        remaining = target_bytes - current_bytes
        best = None
        for idx, current_q in enumerate(selected):
            for q in qualities:
                if q <= current_q:
                    continue
                delta_bytes = codec[q][idx]["actual_total_bytes"] - codec[current_q][idx]["actual_total_bytes"]
                if current_q == base_q:
                    delta_bytes += per_selected_overhead_bytes
                    if enhanced_count == 0:
                        delta_bytes += global_overhead_bytes
                if delta_bytes <= 0 or delta_bytes > remaining:
                    continue
                score = policy_score(policy, confidences[idx], rng, idx, q)
                utility = score / delta_bytes
                candidate = (utility, score, -delta_bytes, idx, q, delta_bytes)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            break
        _, _, _, idx, q, delta_bytes = best
        if selected[idx] == base_q:
            enhanced_count += 1
        selected[idx] = q
        current_bytes += delta_bytes
    return selected


def summarize(
    *,
    label: str,
    selected: list[int],
    qualities: list[int],
    codec: dict[int, list[dict]],
    evals: dict[str, dict[int, list[dict]]],
    profile: str,
    per_selected_overhead_bytes: int,
    global_overhead_bytes: int,
) -> dict:
    base_q = qualities[0]
    selected_count = sum(1 for q in selected if q > base_q)
    overhead = selected_count * per_selected_overhead_bytes + (global_overhead_bytes if selected_count else 0)
    total_pixels = sum(codec[base_q][idx]["width"] * codec[base_q][idx]["height"] for idx in range(len(selected)))
    total_bytes = sum(codec[q][idx]["actual_total_bytes"] for idx, q in enumerate(selected)) + overhead
    result = {
        "label": label,
        "profile": profile,
        "samples": len(selected),
        "quality_counts": dict(sorted(Counter(selected).items())),
        "selected_enhancement_count": selected_count,
        "selection_overhead_bytes": overhead,
        "actual_total_bytes": total_bytes,
        "actual_total_bpp": 8.0 * total_bytes / total_pixels,
    }
    for eval_label, by_q in evals.items():
        pairs = [(by_q[q][idx]["reference"], by_q[q][idx]["prediction"]) for idx, q in enumerate(selected)]
        result[eval_label] = evaluate_pairs(pairs, profile).to_dict()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-result", type=parse_q_path, action="append", required=True)
    parser.add_argument("--eval-result", type=parse_eval_q_path, action="append", required=True)
    parser.add_argument("--confidence-results", type=Path)
    parser.add_argument("--policy", action="append", default=["random", "low_confidence", "high_confidence", "uniform_cost"])
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--per-selected-overhead-bytes", type=int, default=40)
    parser.add_argument("--global-overhead-bytes", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.per_selected_overhead_bytes < 0 or args.global_overhead_bytes < 0:
        raise ValueError("overhead byte counts must be nonnegative")

    codec = load_by_quality(args.codec_result)
    qualities = sorted(codec)
    evals: dict[str, dict[int, list[dict]]] = {}
    for label, q, path in args.eval_result:
        evals.setdefault(label, {})[q] = read_jsonl(path)
    for label, by_q in evals.items():
        if sorted(by_q) != qualities:
            raise ValueError(f"{label} qualities do not match codec qualities")
    assert_aligned(codec, evals)
    confidences = read_confidences(args.confidence_results, len(codec[qualities[0]]))

    rows = []
    for q in qualities:
        uniform = [q for _ in codec[q]]
        rows.append(
            summarize(
                label=f"uniform_q{q}",
                selected=uniform,
                qualities=qualities,
                codec=codec,
                evals=evals,
                profile=args.profile,
                per_selected_overhead_bytes=0,
                global_overhead_bytes=0,
            )
        )
    for target_q in qualities[1:]:
        target_bytes = sum(row["actual_total_bytes"] for row in codec[target_q])
        for policy in args.policy:
            selected = select_by_policy(
                policy=policy,
                target_bytes=target_bytes,
                qualities=qualities,
                codec=codec,
                confidences=confidences,
                seed=args.seed + target_q,
                per_selected_overhead_bytes=args.per_selected_overhead_bytes,
                global_overhead_bytes=args.global_overhead_bytes,
            )
            rows.append(
                summarize(
                    label=f"{policy}_to_uniform_q{target_q}_bytes",
                    selected=selected,
                    qualities=qualities,
                    codec=codec,
                    evals=evals,
                    profile=args.profile,
                    per_selected_overhead_bytes=args.per_selected_overhead_bytes,
                    global_overhead_bytes=args.global_overhead_bytes,
                )
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"
    summary_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(summary_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
