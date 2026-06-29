#!/usr/bin/env python3
"""Build OCR-supervised code-assignment utility rows for top-k code variants."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, normalize_text  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def distance(reference: str, prediction: str, profile: str) -> int:
    return character_counts(reference, prediction, profile).distance


def exact(reference: str, prediction: str, profile: str) -> int:
    return int(normalize_text(reference, profile) == normalize_text(prediction, profile))


def require_same_image(split_row: dict, parseq_row: dict, tesseract_row: dict) -> None:
    split_image = split_row["image_path"]
    if parseq_row["image"] != split_image:
        raise SystemExit(f"PARSeq row mismatch: {parseq_row['image']} != {split_image}")
    if tesseract_row["image"] != split_image:
        raise SystemExit(f"Tesseract row mismatch: {tesseract_row['image']} != {split_image}")


def oracle_for_group(rows: list[dict]) -> dict:
    nearest = next(row for row in rows if row["is_nearest"])
    safe_rows = [row for row in rows if row["parseq_distance"] <= nearest["parseq_distance"]]
    if not safe_rows:
        safe_rows = [nearest]
    return min(
        safe_rows,
        key=lambda row: (
            row["tesseract_distance"],
            row["parseq_distance"],
            int(row["topk_rank"]),
            float(row["assignment_relative_error"]),
            int(row["code_index"]),
        ),
    )


def annotate_set(seed: int, candidate_split: Path, parseq_ocr: Path, tesseract_ocr: Path, profile: str) -> list[dict]:
    split_rows = read_jsonl(candidate_split)
    parseq_rows = read_jsonl(parseq_ocr)
    tesseract_rows = read_jsonl(tesseract_ocr)
    if not (len(split_rows) == len(parseq_rows) == len(tesseract_rows)):
        raise SystemExit(
            f"row-count mismatch for seed {seed}: split={len(split_rows)}, "
            f"parseq={len(parseq_rows)}, tesseract={len(tesseract_rows)}"
        )

    rows = []
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for split_row, parseq_row, tesseract_row in zip(split_rows, parseq_rows, tesseract_rows):
        require_same_image(split_row, parseq_row, tesseract_row)
        reference = str(split_row["text"])
        out = {
            "seed": seed,
            "source_index": int(split_row["source_index"]),
            "candidate_index": int(split_row["candidate_index"]),
            "candidate_slot": int(split_row["candidate_slot"]),
            "source": split_row.get("source"),
            "split": split_row.get("split"),
            "source_image": split_row.get("source_image"),
            "image_path": split_row["image_path"],
            "reference": reference,
            "reference_length": len(normalize_text(reference, profile)),
            "nearest_code": int(split_row["nearest_code"]),
            "code_index": int(split_row["code_index"]),
            "topk_rank": int(split_row["topk_rank"]),
            "is_nearest": int(bool(split_row["is_nearest"])),
            "assignment_relative_error": float(split_row["assignment_relative_error"]),
            "parseq_prediction": parseq_row.get("prediction", ""),
            "parseq_confidence": parseq_row.get("confidence"),
            "parseq_distance": distance(reference, parseq_row.get("prediction", ""), profile),
            "parseq_exact": exact(reference, parseq_row.get("prediction", ""), profile),
            "tesseract_prediction": tesseract_row.get("prediction", ""),
            "tesseract_distance": distance(reference, tesseract_row.get("prediction", ""), profile),
            "tesseract_exact": exact(reference, tesseract_row.get("prediction", ""), profile),
        }
        out["code_equals_nearest"] = int(out["code_index"] == out["nearest_code"])
        rows.append(out)
        grouped[(out["source_index"], out["candidate_index"])].append(out)

    for key, group_rows in grouped.items():
        nearest_rows = [row for row in group_rows if row["is_nearest"]]
        if len(nearest_rows) != 1:
            raise SystemExit(f"group {key} has {len(nearest_rows)} nearest rows")
        nearest = nearest_rows[0]
        oracle = oracle_for_group(group_rows)
        for row in group_rows:
            row["parseq_nearest_distance"] = nearest["parseq_distance"]
            row["tesseract_nearest_distance"] = nearest["tesseract_distance"]
            row["parseq_delta_vs_nearest"] = row["parseq_distance"] - nearest["parseq_distance"]
            row["tesseract_delta_vs_nearest"] = row["tesseract_distance"] - nearest["tesseract_distance"]
            row["multi_teacher_delta_vs_nearest"] = row["parseq_delta_vs_nearest"] + row["tesseract_delta_vs_nearest"]
            row["label_tesseract_parseq_safe_improves"] = int(
                row["tesseract_delta_vs_nearest"] < 0 and row["parseq_delta_vs_nearest"] <= 0
            )
            row["label_parseq_tesseract_safe_improves"] = int(
                row["parseq_delta_vs_nearest"] < 0 and row["tesseract_delta_vs_nearest"] <= 0
            )
            row["label_multi_teacher_improves"] = int(row["multi_teacher_delta_vs_nearest"] < 0)
            row["label_no_evaluator_worsens"] = int(
                row["parseq_delta_vs_nearest"] <= 0 and row["tesseract_delta_vs_nearest"] <= 0
            )
            row["label_worsens_any"] = int(
                row["parseq_delta_vs_nearest"] > 0 or row["tesseract_delta_vs_nearest"] > 0
            )
            row["assignment_oracle_code_index"] = oracle["code_index"]
            row["assignment_oracle_topk_rank"] = oracle["topk_rank"]
            row["assignment_oracle_relative_error"] = oracle["assignment_relative_error"]
            row["assignment_oracle_tesseract_delta"] = oracle["tesseract_distance"] - nearest["tesseract_distance"]
            row["assignment_oracle_parseq_delta"] = oracle["parseq_distance"] - nearest["parseq_distance"]
            row["label_assignment_oracle_choice"] = int(row is oracle)
            row["group_has_safe_tesseract_gain"] = int(oracle["tesseract_distance"] < nearest["tesseract_distance"])
            row["group_oracle_changes_code"] = int(oracle["code_index"] != nearest["code_index"])
    return rows


def summarize(rows: list[dict], output: Path) -> dict:
    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["source_index"], row["candidate_index"])].append(row)

    rank_counts = Counter()
    changed_rank_counts = Counter()
    source_counts = Counter()
    safe_gain_groups = 0
    changed_groups = 0
    for group_rows in groups.values():
        nearest = next(row for row in group_rows if row["is_nearest"])
        oracle = next(row for row in group_rows if row["label_assignment_oracle_choice"])
        rank_counts[str(oracle["topk_rank"])] += 1
        source_counts[str(oracle.get("source"))] += 1
        safe_gain_groups += int(oracle["tesseract_distance"] < nearest["tesseract_distance"])
        if oracle["code_index"] != nearest["code_index"]:
            changed_groups += 1
            changed_rank_counts[str(oracle["topk_rank"])] += 1

    label_counts = {
        "tesseract_parseq_safe_improves": int(sum(row["label_tesseract_parseq_safe_improves"] for row in rows)),
        "parseq_tesseract_safe_improves": int(sum(row["label_parseq_tesseract_safe_improves"] for row in rows)),
        "multi_teacher_improves": int(sum(row["label_multi_teacher_improves"] for row in rows)),
        "no_evaluator_worsens": int(sum(row["label_no_evaluator_worsens"] for row in rows)),
        "worsens_any": int(sum(row["label_worsens_any"] for row in rows)),
        "assignment_oracle_choice": int(sum(row["label_assignment_oracle_choice"] for row in rows)),
    }
    per_seed = {}
    for seed in sorted({row["seed"] for row in rows}):
        seed_groups = {
            key: group_rows
            for key, group_rows in groups.items()
            if key[0] == seed
        }
        seed_changed = 0
        seed_gain = 0
        for group_rows in seed_groups.values():
            nearest = next(row for row in group_rows if row["is_nearest"])
            oracle = next(row for row in group_rows if row["label_assignment_oracle_choice"])
            seed_changed += int(oracle["code_index"] != nearest["code_index"])
            seed_gain += int(oracle["tesseract_distance"] < nearest["tesseract_distance"])
        per_seed[str(seed)] = {
            "groups": len(seed_groups),
            "safe_tesseract_gain_groups": seed_gain,
            "oracle_changed_groups": seed_changed,
        }
    return {
        "description": "Top-k code-assignment OCR utility table.",
        "output": str(output),
        "rows": len(rows),
        "groups": len(groups),
        "label_counts": label_counts,
        "safe_tesseract_gain_groups": safe_gain_groups,
        "oracle_changed_groups": changed_groups,
        "oracle_rank_counts": dict(sorted(rank_counts.items(), key=lambda item: int(item[0]))),
        "changed_oracle_rank_counts": dict(sorted(changed_rank_counts.items(), key=lambda item: int(item[0]))),
        "oracle_source_counts": dict(sorted(source_counts.items())),
        "per_seed": per_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-input",
        action="append",
        nargs=4,
        metavar=("SEED", "CANDIDATE_SPLIT", "PARSEQ_OCR", "TESSERACT_OCR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="unicode_strict_v1")
    args = parser.parse_args()

    rows = []
    for seed_text, split_path, parseq_path, tesseract_path in args.seed_input:
        rows.extend(
            annotate_set(
                seed=int(seed_text),
                candidate_split=Path(split_path),
                parseq_ocr=Path(parseq_path),
                tesseract_ocr=Path(tesseract_path),
                profile=args.profile,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = summarize(rows, args.output)
    summary["profile"] = args.profile
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
