#!/usr/bin/env python3
"""Add PARSeq teacher target arrays to an assignment reranker NPZ artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("original_seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--conf-weight", type=float, default=1.0)
    args = parser.parse_args()

    data = {key: value for key, value in np.load(args.features).items()}
    table = {}
    for row in read_jsonl(args.table):
        table[(group_key(row), int(row["code_index"]))] = row

    group_keys = data["group_keys"]
    code_indices = data["code_indices"]
    mask = data["mask"].astype(bool)
    is_nearest = data["is_nearest"].astype(bool)
    shape = mask.shape
    edit_delta = np.zeros(shape, dtype=np.float32)
    norm_edit_delta = np.zeros(shape, dtype=np.float32)
    conf_delta = np.zeros(shape, dtype=np.float32)
    edit_gain = np.zeros(shape, dtype=np.float32)
    norm_edit_gain = np.zeros(shape, dtype=np.float32)
    conf_gain = np.zeros(shape, dtype=np.float32)
    combo_gain = np.zeros(shape, dtype=np.float32)
    missing = 0
    valid_rows = 0
    for group_index in range(shape[0]):
        key = tuple(int(value) for value in group_keys[group_index])
        for row_position in range(shape[1]):
            if not mask[group_index, row_position]:
                continue
            valid_rows += 1
            row = table.get((key, int(code_indices[group_index, row_position])))
            if row is None:
                missing += 1
                continue
            edit_delta[group_index, row_position] = float(row.get("img_teacher_parseq_edit_delta_vs_nearest", 0.0))
            norm_edit_delta[group_index, row_position] = float(
                row.get("img_teacher_parseq_norm_edit_delta_vs_nearest", 0.0)
            )
            conf_delta[group_index, row_position] = float(
                row.get("img_teacher_parseq_confidence_delta_vs_nearest", 0.0)
            )
            if is_nearest[group_index, row_position]:
                continue
            edit_gain[group_index, row_position] = max(0.0, -edit_delta[group_index, row_position])
            norm_edit_gain[group_index, row_position] = max(0.0, -norm_edit_delta[group_index, row_position])
            conf_gain[group_index, row_position] = max(0.0, conf_delta[group_index, row_position])
            combo_gain[group_index, row_position] = (
                max(edit_gain[group_index, row_position], norm_edit_gain[group_index, row_position])
                + args.conf_weight * conf_gain[group_index, row_position]
            )

    data.update(
        {
            "teacher_parseq_edit_delta": edit_delta,
            "teacher_parseq_norm_edit_delta": norm_edit_delta,
            "teacher_parseq_confidence_delta": conf_delta,
            "teacher_parseq_edit_gain": edit_gain,
            "teacher_parseq_norm_edit_gain": norm_edit_gain,
            "teacher_parseq_conf_gain": conf_gain,
            "teacher_parseq_combo_gain": combo_gain,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)

    summary = {
        "features": str(args.features),
        "table": str(args.table),
        "output": str(args.output),
        "conf_weight": args.conf_weight,
        "groups": int(shape[0]),
        "max_candidates": int(shape[1]),
        "valid_rows": int(valid_rows),
        "missing_rows": int(missing),
        "positive_rows": {
            "edit_gain": int(((edit_gain > 0) & mask).sum()),
            "norm_edit_gain": int(((norm_edit_gain > 0) & mask).sum()),
            "conf_gain": int(((conf_gain > 0) & mask).sum()),
            "combo_gain": int(((combo_gain > 0) & mask).sum()),
        },
        "positive_groups": {
            "edit_gain": int(((edit_gain > 0) & mask).any(axis=1).sum()),
            "norm_edit_gain": int(((norm_edit_gain > 0) & mask).any(axis=1).sum()),
            "conf_gain": int(((conf_gain > 0) & mask).any(axis=1).sum()),
            "combo_gain": int(((combo_gain > 0) & mask).any(axis=1).sum()),
        },
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
