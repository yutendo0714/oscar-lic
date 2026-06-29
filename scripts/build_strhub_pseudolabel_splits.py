#!/usr/bin/env python3
"""Build source-image OCR pseudo-label splits for STRHub teacher diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def group_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def make_source_split(table_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int, int, int], dict] = {}
    for row in table_rows:
        key = group_key(row)
        if key in groups:
            continue
        groups[key] = {
            "image_path": row["source_image"],
            "text": row.get("reference", ""),
            "reference": row.get("reference", ""),
            "source": row.get("source"),
            "split": row.get("split"),
            "seed": row.get("seed"),
            "real_seed": row.get("real_seed", row.get("seed")),
            "original_seed": row.get("original_seed", row.get("seed")),
            "source_index": row.get("source_index"),
            "candidate_index": row.get("candidate_index"),
        }
    return [groups[key] for key in sorted(groups)]


def read_source_predictions(path: Path) -> dict[tuple[int, int, int, int], dict]:
    predictions = {}
    for row in read_jsonl(path):
        predictions[group_key(row)] = row
    return predictions


def make_pseudolabel_table(table_rows: list[dict], source_predictions: dict[tuple[int, int, int, int], dict], teacher_name: str) -> list[dict]:
    output = []
    missing = []
    for row in table_rows:
        key = group_key(row)
        source_row = source_predictions.get(key)
        if source_row is None:
            missing.append(key)
            continue
        pseudo = str(source_row.get("prediction", ""))
        out = dict(row)
        out["original_reference"] = row.get("reference", "")
        out["reference"] = pseudo
        out["text"] = pseudo
        out[f"{teacher_name}_source_prediction"] = pseudo
        out[f"{teacher_name}_source_confidence"] = source_row.get("confidence")
        out[f"{teacher_name}_source_image"] = source_row.get("image")
        output.append(out)
    if missing:
        raise SystemExit(f"missing {teacher_name} source predictions: {missing[:5]} ({len(missing)} rows)")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-source-split", type=Path, required=True)
    parser.add_argument("--crnn-source-results", type=Path)
    parser.add_argument("--abinet-source-results", type=Path)
    parser.add_argument("--output-crnn-pseudolabel-table", type=Path)
    parser.add_argument("--output-abinet-pseudolabel-table", type=Path)
    args = parser.parse_args()

    table_rows = read_jsonl(args.table)
    source_rows = make_source_split(table_rows)
    write_jsonl(args.output_source_split, source_rows)
    summary = {
        "table": str(args.table),
        "source_split": str(args.output_source_split),
        "source_groups": len(source_rows),
        "table_rows": len(table_rows),
    }
    if args.crnn_source_results:
        if not args.output_crnn_pseudolabel_table:
            raise SystemExit("--output-crnn-pseudolabel-table is required with --crnn-source-results")
        crnn_rows = make_pseudolabel_table(table_rows, read_source_predictions(args.crnn_source_results), "crnn")
        write_jsonl(args.output_crnn_pseudolabel_table, crnn_rows)
        summary["crnn_pseudolabel_table"] = str(args.output_crnn_pseudolabel_table)
        summary["crnn_rows"] = len(crnn_rows)
    if args.abinet_source_results:
        if not args.output_abinet_pseudolabel_table:
            raise SystemExit("--output-abinet-pseudolabel-table is required with --abinet-source-results")
        abinet_rows = make_pseudolabel_table(table_rows, read_source_predictions(args.abinet_source_results), "abinet")
        write_jsonl(args.output_abinet_pseudolabel_table, abinet_rows)
        summary["abinet_pseudolabel_table"] = str(args.output_abinet_pseudolabel_table)
        summary["abinet_rows"] = len(abinet_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
