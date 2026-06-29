"""Select hard no-headroom candidates for top-k assignment diagnostics.

The output intentionally matches the compact candidate-selection JSONL consumed by
``generate_mlicpp_assignment_candidate_recons.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_excluded_sources(paths: list[Path]) -> set[int]:
    excluded: set[int] = set()
    for path in paths:
        for row in _read_jsonl(path):
            if "source_index" in row:
                excluded.add(int(row["source_index"]))
    return excluded


def _source_has_safe_gain(rows: list[dict[str, Any]]) -> bool:
    return any(
        float(row.get("tesseract_delta_distance", 0.0)) < 0
        and float(row.get("parseq_delta_distance", 0.0)) <= 0
        for row in rows
    )


def _candidate_hard_score(row: dict[str, Any]) -> float:
    """Rank candidates likely to tempt a false assignment change.

    This is deliberately deployable-feature shaped: image/base patch errors,
    edge density and raw residual harm are used only to choose negative labels
    for offline OCR enumeration. The score itself is not a metric.
    """

    patch_error = float(row.get("patch_base_abs_error_mean", 0.0))
    patch_mse = float(row.get("patch_base_mse", 0.0))
    edge_density = float(row.get("orig_patch_edge_density", 0.0))
    area_fraction = float(row.get("candidate_pixel_area_fraction", 0.0))
    tess_harm = max(0.0, float(row.get("tesseract_delta_distance", 0.0)))
    parseq_harm = max(0.0, float(row.get("parseq_delta_distance", 0.0)))
    no_worse = float(row.get("label_no_evaluator_worsens", 0.0))
    return (
        20.0 * patch_error
        + 200.0 * patch_mse
        + 0.25 * edge_density
        + 0.10 * area_fraction
        + 0.20 * tess_harm
        + 0.50 * parseq_harm
        + 0.05 * no_worse
    )


def _source_priority(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> tuple[float, int]:
    base_tess_distance = float(rows[0].get("tesseract_base_distance", 0.0))
    base_tess_exact = float(rows[0].get("tesseract_base_exact", 0.0))
    any_harm = any(float(row.get("worsening_evaluator_count", 0.0)) > 0 for row in rows)
    hard_score = _candidate_hard_score(candidate)
    # Prefer exact/near-exact no-headroom groups, but keep visually active candidates first.
    return (
        4.0 * base_tess_exact
        - 0.25 * base_tess_distance
        + 0.5 * float(any_harm)
        + hard_score,
        int(rows[0].get("source_index", 0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-selection", action="append", default=[], type=Path)
    parser.add_argument("--max-sources", type=int, default=120)
    parser.add_argument("--max-base-tesseract-distance", type=float, default=2.0)
    parser.add_argument("--require-parseq-base-exact", action="store_true")
    parser.add_argument("--require-no-safe-source", action="store_true")
    parser.add_argument("--include-source-name", action="append", default=[])
    parser.add_argument("--max-per-source-name", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.utility_table)
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[int(row["source_index"])].append(row)

    excluded_sources = _read_excluded_sources(args.exclude_selection)
    included_sources = set(args.include_source_name)
    candidates: list[tuple[tuple[float, int], dict[str, Any]]] = []
    for source_index, source_rows in by_source.items():
        if source_index in excluded_sources:
            continue
        first = source_rows[0]
        if included_sources and str(first.get("source", "unknown")) not in included_sources:
            continue
        if args.require_no_safe_source and _source_has_safe_gain(source_rows):
            continue
        if args.require_parseq_base_exact and int(first.get("parseq_base_exact", 0)) != 1:
            continue
        if float(first.get("tesseract_base_distance", 0.0)) > args.max_base_tesseract_distance:
            continue
        best = max(source_rows, key=_candidate_hard_score)
        candidates.append((_source_priority(source_rows, best), best))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    per_source_name: dict[str, int] = defaultdict(int)
    for _, row in candidates:
        source_name = str(row.get("source", "unknown"))
        if args.max_per_source_name and per_source_name[source_name] >= args.max_per_source_name:
            continue
        per_source_name[source_name] += 1
        selected.append(
            {
                "source_index": int(row["source_index"]),
                "candidate_index": int(row["candidate_index"]),
                "selected_by_policy": True,
                "assignment_code_index": None,
                "selector_score": _candidate_hard_score(row),
                "tesseract_delta_distance": int(row.get("tesseract_delta_distance", 0)),
                "parseq_delta_distance": int(row.get("parseq_delta_distance", 0)),
                "tesseract_base_distance": int(row.get("tesseract_base_distance", 0)),
                "parseq_base_distance": int(row.get("parseq_base_distance", 0)),
                "source": source_name,
                "reference": row.get("reference"),
                "selection_reason": "hard_noheadroom_no_safe_source",
            }
        )
        if len(selected) >= args.max_sources:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "utility_table": str(args.utility_table),
        "output": str(args.output),
        "rows": len(selected),
        "eligible_sources": len(candidates),
        "excluded_sources": len(excluded_sources),
        "max_base_tesseract_distance": args.max_base_tesseract_distance,
        "require_parseq_base_exact": args.require_parseq_base_exact,
        "require_no_safe_source": args.require_no_safe_source,
        "include_source_name": sorted(included_sources),
        "source_counts": dict(per_source_name),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
