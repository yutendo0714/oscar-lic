#!/usr/bin/env python3
"""Greedy quality oracle with a weighted multi-teacher OCR objective."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, evaluate_pairs  # noqa: E402


def parse_q_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected q=path")
    key, path = value.split("=", 1)
    return int(key), Path(path)


def parse_objective_q_path(value: str) -> tuple[str, float, int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected label:weight:q=path")
    left, path = value.split("=", 1)
    parts = left.rsplit(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected label:weight:q=path")
    label, weight, q = parts
    if not label:
        raise argparse.ArgumentTypeError("objective label is empty")
    return label, float(weight), int(q), Path(path)


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


def load_objectives(items: list[tuple[str, float, int, Path]]) -> dict[str, dict]:
    objectives: dict[str, dict] = {}
    for label, weight, q, path in items:
        if label not in objectives:
            objectives[label] = {"weight": weight, "by_q": {}}
        if objectives[label]["weight"] != weight:
            raise ValueError(f"objective {label!r} has inconsistent weights")
        objectives[label]["by_q"][q] = read_jsonl(path)
    return objectives


def assert_aligned(
    codec: dict[int, list[dict]],
    objectives: dict[str, dict],
    evals: dict[str, dict[int, list[dict]]],
) -> None:
    qualities = sorted(codec)
    n = len(codec[qualities[0]])
    for q in qualities:
        if len(codec[q]) != n:
            raise ValueError(f"codec quality {q}: row count mismatch")
        for label, objective in objectives.items():
            if len(objective["by_q"][q]) != n:
                raise ValueError(f"objective {label} quality {q}: row count mismatch")
        for label, by_q in evals.items():
            if len(by_q[q]) != n:
                raise ValueError(f"eval {label} quality {q}: row count mismatch")
    for idx in range(n):
        reference = None
        for q in qualities:
            for objective in objectives.values():
                row_ref = objective["by_q"][q][idx]["reference"]
                reference = row_ref if reference is None else reference
                if row_ref != reference:
                    raise ValueError(f"objective reference mismatch at row {idx}, q{q}")
            for by_q in evals.values():
                if by_q[q][idx]["reference"] != reference:
                    raise ValueError(f"eval reference mismatch at row {idx}, q{q}")


def sample_loss(reference: str, prediction: str, profile: str) -> int:
    return character_counts(reference, prediction, profile).distance


def objective_loss_for_sample(objectives: dict[str, dict], q: int, idx: int, profile: str) -> float:
    total = 0.0
    for objective in objectives.values():
        row = objective["by_q"][q][idx]
        total += objective["weight"] * sample_loss(row["reference"], row["prediction"], profile)
    return total


def summarize_selection(
    label: str,
    qualities: list[int],
    selected: list[int],
    codec: dict[int, list[dict]],
    objectives: dict[str, dict],
    evals: dict[str, dict[int, list[dict]]],
    profile: str,
    *,
    apply_selection_overhead: bool,
    per_selected_overhead_bytes: int,
    global_overhead_bytes: int,
) -> dict:
    total_pixels = sum(codec[qualities[0]][idx]["width"] * codec[qualities[0]][idx]["height"] for idx in range(len(selected)))
    base_q = qualities[0]
    raw_selected_count = sum(1 for q in selected if q > base_q)
    selected_count = raw_selected_count if apply_selection_overhead else 0
    overhead_bytes = 0
    if apply_selection_overhead and selected_count:
        overhead_bytes = selected_count * per_selected_overhead_bytes + global_overhead_bytes
    total_bytes = sum(codec[q][idx]["actual_total_bytes"] for idx, q in enumerate(selected)) + overhead_bytes
    result = {
        "label": label,
        "profile": profile,
        "samples": len(selected),
        "quality_counts": dict(sorted(Counter(selected).items())),
        "selected_enhancement_count": selected_count,
        "selection_overhead_bytes": overhead_bytes,
        "per_selected_overhead_bytes": per_selected_overhead_bytes if apply_selection_overhead else 0,
        "global_overhead_bytes": global_overhead_bytes if apply_selection_overhead else 0,
        "actual_total_bytes": total_bytes,
        "actual_total_bpp": 8.0 * total_bytes / total_pixels,
        "weighted_objective_loss": sum(
            objective_loss_for_sample(objectives, q, idx, profile)
            for idx, q in enumerate(selected)
        ),
    }
    for objective_label, objective in objectives.items():
        pairs = [
            (objective["by_q"][q][idx]["reference"], objective["by_q"][q][idx]["prediction"])
            for idx, q in enumerate(selected)
        ]
        result[objective_label] = evaluate_pairs(pairs, profile).to_dict()
    for eval_label, by_q in evals.items():
        pairs = [(by_q[q][idx]["reference"], by_q[q][idx]["prediction"]) for idx, q in enumerate(selected)]
        result[eval_label] = evaluate_pairs(pairs, profile).to_dict()
    return result


def greedy_select(
    target_bytes: int,
    qualities: list[int],
    codec: dict[int, list[dict]],
    objectives: dict[str, dict],
    profile: str,
    *,
    per_selected_overhead_bytes: int,
    global_overhead_bytes: int,
) -> list[int]:
    base_q = qualities[0]
    selected = [base_q for _ in codec[base_q]]
    current_bytes = sum(row["actual_total_bytes"] for row in codec[base_q])
    enhanced_count = 0
    current_loss = [objective_loss_for_sample(objectives, base_q, idx, profile) for idx in range(len(selected))]

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
                new_loss = objective_loss_for_sample(objectives, q, idx, profile)
                delta_loss = current_loss[idx] - new_loss
                if delta_loss <= 0:
                    continue
                ratio = delta_loss / delta_bytes
                candidate = (ratio, delta_loss, -delta_bytes, idx, q, new_loss, delta_bytes)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            break
        _, _, _, idx, q, new_loss, delta_bytes = best
        if selected[idx] == base_q:
            enhanced_count += 1
        selected[idx] = q
        current_loss[idx] = new_loss
        current_bytes += delta_bytes
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-result", type=parse_q_path, action="append", required=True)
    parser.add_argument("--objective-result", type=parse_objective_q_path, action="append", required=True)
    parser.add_argument("--eval-result", type=parse_eval_q_path, action="append", default=[])
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--per-selected-overhead-bytes", type=int, default=0)
    parser.add_argument("--global-overhead-bytes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.per_selected_overhead_bytes < 0 or args.global_overhead_bytes < 0:
        raise ValueError("overhead byte counts must be nonnegative")

    codec = load_by_quality(args.codec_result)
    qualities = sorted(codec)
    objectives = load_objectives(args.objective_result)
    for label, objective in objectives.items():
        if sorted(objective["by_q"]) != qualities:
            raise ValueError(f"objective {label} qualities do not match codec qualities")
    evals: dict[str, dict[int, list[dict]]] = {}
    for label, q, path in args.eval_result:
        evals.setdefault(label, {})[q] = read_jsonl(path)
    for label, by_q in evals.items():
        if sorted(by_q) != qualities:
            raise ValueError(f"eval {label} qualities do not match codec qualities")
    assert_aligned(codec, objectives, evals)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in qualities:
        uniform = [q for _ in codec[q]]
        rows.append(
            summarize_selection(
                f"uniform_q{q}",
                qualities,
                uniform,
                codec,
                objectives,
                evals,
                args.profile,
                apply_selection_overhead=False,
                per_selected_overhead_bytes=args.per_selected_overhead_bytes,
                global_overhead_bytes=args.global_overhead_bytes,
            )
        )

    for q in qualities:
        target_bytes = sum(row["actual_total_bytes"] for row in codec[q])
        selected = greedy_select(
            target_bytes,
            qualities,
            codec,
            objectives,
            args.profile,
            per_selected_overhead_bytes=args.per_selected_overhead_bytes,
            global_overhead_bytes=args.global_overhead_bytes,
        )
        label = f"greedy_to_uniform_q{q}_bytes"
        rows.append(
            summarize_selection(
                label,
                qualities,
                selected,
                codec,
                objectives,
                evals,
                args.profile,
                apply_selection_overhead=True,
                per_selected_overhead_bytes=args.per_selected_overhead_bytes,
                global_overhead_bytes=args.global_overhead_bytes,
            )
        )
        selection_path = output_dir / f"{label}.jsonl"
        selection_path.write_text(
            "".join(
                json.dumps(
                    {
                        "index": idx,
                        "selected_quality": selected_q,
                        "reference": next(iter(objectives.values()))["by_q"][selected_q][idx]["reference"],
                        "actual_total_bytes": codec[selected_q][idx]["actual_total_bytes"],
                        "source_image": codec[selected_q][idx]["image"],
                        "weighted_objective_loss": objective_loss_for_sample(objectives, selected_q, idx, args.profile),
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for idx, selected_q in enumerate(selected)
            ),
            encoding="utf-8",
        )

    summary_path = output_dir / "summary.jsonl"
    summary_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(summary_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
