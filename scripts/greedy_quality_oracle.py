#!/usr/bin/env python3
"""Greedy per-sample quality oracle using actual bytes and OCR loss."""

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


def assert_aligned(codec: dict[int, list[dict]], teacher: dict[int, list[dict]], evals: dict[str, dict[int, list[dict]]]) -> None:
    qualities = sorted(codec)
    n = len(codec[qualities[0]])
    for q in qualities:
        if len(codec[q]) != n or len(teacher[q]) != n:
            raise ValueError(f"quality {q}: row count mismatch")
        for label, by_q in evals.items():
            if len(by_q[q]) != n:
                raise ValueError(f"{label} quality {q}: row count mismatch")
    for idx in range(n):
        ref = teacher[qualities[0]][idx]["reference"]
        for q in qualities:
            if teacher[q][idx]["reference"] != ref:
                raise ValueError(f"teacher reference mismatch at row {idx}, q{q}")
            for label, by_q in evals.items():
                if by_q[q][idx]["reference"] != ref:
                    raise ValueError(f"{label} reference mismatch at row {idx}, q{q}")


def sample_loss(reference: str, prediction: str, profile: str) -> int:
    return character_counts(reference, prediction, profile).distance


def summarize_selection(
    label: str,
    qualities: list[int],
    selected: list[int],
    codec: dict[int, list[dict]],
    teacher: dict[int, list[dict]],
    evals: dict[str, dict[int, list[dict]]],
    profile: str,
) -> dict:
    total_pixels = sum(codec[qualities[0]][idx]["width"] * codec[qualities[0]][idx]["height"] for idx in range(len(selected)))
    total_bytes = sum(codec[q][idx]["actual_total_bytes"] for idx, q in enumerate(selected))
    teacher_pairs = [(teacher[q][idx]["reference"], teacher[q][idx]["prediction"]) for idx, q in enumerate(selected)]
    result = {
        "label": label,
        "profile": profile,
        "samples": len(selected),
        "quality_counts": dict(sorted(Counter(selected).items())),
        "actual_total_bytes": total_bytes,
        "actual_total_bpp": 8.0 * total_bytes / total_pixels,
        "teacher": evaluate_pairs(teacher_pairs, profile).to_dict(),
    }
    for eval_label, by_q in evals.items():
        pairs = [(by_q[q][idx]["reference"], by_q[q][idx]["prediction"]) for idx, q in enumerate(selected)]
        result[eval_label] = evaluate_pairs(pairs, profile).to_dict()
    return result


def greedy_select(
    target_bytes: int,
    qualities: list[int],
    codec: dict[int, list[dict]],
    teacher: dict[int, list[dict]],
    profile: str,
) -> list[int]:
    base_q = qualities[0]
    selected = [base_q for _ in codec[base_q]]
    current_bytes = sum(row["actual_total_bytes"] for row in codec[base_q])
    current_loss = [
        sample_loss(teacher[base_q][idx]["reference"], teacher[base_q][idx]["prediction"], profile)
        for idx in range(len(selected))
    ]

    while True:
        remaining = target_bytes - current_bytes
        best = None
        for idx, current_q in enumerate(selected):
            for q in qualities:
                if q <= current_q:
                    continue
                delta_bytes = codec[q][idx]["actual_total_bytes"] - codec[current_q][idx]["actual_total_bytes"]
                if delta_bytes <= 0 or delta_bytes > remaining:
                    continue
                new_loss = sample_loss(teacher[q][idx]["reference"], teacher[q][idx]["prediction"], profile)
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
        selected[idx] = q
        current_loss[idx] = new_loss
        current_bytes += delta_bytes
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-result", type=parse_q_path, action="append", required=True)
    parser.add_argument("--teacher-result", type=parse_q_path, action="append", required=True)
    parser.add_argument("--eval-result", type=parse_eval_q_path, action="append", default=[])
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    codec = load_by_quality(args.codec_result)
    teacher = load_by_quality(args.teacher_result)
    evals: dict[str, dict[int, list[dict]]] = {}
    for label, q, path in args.eval_result:
        evals.setdefault(label, {})[q] = read_jsonl(path)
    qualities = sorted(codec)
    if sorted(teacher) != qualities:
        raise ValueError("teacher qualities do not match codec qualities")
    for label, by_q in evals.items():
        if sorted(by_q) != qualities:
            raise ValueError(f"{label} qualities do not match codec qualities")
    assert_aligned(codec, teacher, evals)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in qualities:
        uniform = [q for _ in codec[q]]
        rows.append(summarize_selection(f"uniform_q{q}", qualities, uniform, codec, teacher, evals, args.profile))

    for q in qualities:
        target_bytes = sum(row["actual_total_bytes"] for row in codec[q])
        selected = greedy_select(target_bytes, qualities, codec, teacher, args.profile)
        label = f"greedy_to_uniform_q{q}_bytes"
        rows.append(summarize_selection(label, qualities, selected, codec, teacher, evals, args.profile))
        selection_path = output_dir / f"{label}.jsonl"
        selection_path.write_text(
            "".join(
                json.dumps(
                    {
                        "index": idx,
                        "selected_quality": selected_q,
                        "reference": teacher[selected_q][idx]["reference"],
                        "teacher_prediction": teacher[selected_q][idx]["prediction"],
                        "actual_total_bytes": codec[selected_q][idx]["actual_total_bytes"],
                        "source_image": codec[selected_q][idx]["image"],
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
