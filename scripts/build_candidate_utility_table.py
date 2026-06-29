#!/usr/bin/env python3
"""Build a supervised candidate utility table from single-candidate analyses."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_analysis(path: Path) -> dict[tuple[int, int], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in data["details"]:
        key = (int(row["source_index"]), int(row["candidate_index"]))
        rows[key] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", action="append", nargs=2, metavar=("LABEL", "ANALYSIS_JSON"))
    args = parser.parse_args()

    if not args.evaluator:
        raise SystemExit("provide at least one --evaluator LABEL ANALYSIS_JSON")

    candidate_rows = read_jsonl(args.candidate_split)
    analyses = [(label, read_analysis(Path(path))) for label, path in args.evaluator]
    output_rows = []
    label_counts = Counter()
    for row in candidate_rows:
        key = (int(row["source_index"]), int(row["candidate_index"]))
        out = {
            "source_index": key[0],
            "candidate_index": key[1],
            "source": row.get("source"),
            "source_image": row.get("source_image"),
            "reference": row.get("text"),
            "slice_index": row["slice_index"],
            "y0": row["y0"],
            "y1": row["y1"],
            "x0": row["x0"],
            "x1": row["x1"],
            "channel0": row["channel0"],
            "channel1": row["channel1"],
            "candidate_count": row["candidate_count"],
            "actual_total_bytes": row["actual_total_bytes"],
            "actual_total_bpp": row["actual_total_bpp"],
            "gate_payload_bytes": row["gate_payload_bytes"],
            "residual_payload_bytes": row["residual_payload_bytes"],
            "psnr_enhanced_db": row["psnr_enhanced_db"],
            "tile": row["tile"],
            "quant_step": row["quant_step"],
            "residual_scale": row["residual_scale"],
        }
        multi_teacher_delta = 0
        improving_evaluators = 0
        worsening_evaluators = 0
        for label, analysis in analyses:
            detail = analysis[key]
            delta = int(detail["delta_distance_candidate_minus_base"])
            out[f"{label}_base_distance"] = detail["base_distance"]
            out[f"{label}_candidate_distance"] = detail["candidate_distance"]
            out[f"{label}_delta_distance"] = delta
            out[f"{label}_base_exact"] = detail["base_exact"]
            out[f"{label}_candidate_exact"] = detail["candidate_exact"]
            out[f"{label}_delta_exact"] = detail["delta_exact_candidate_minus_base"]
            multi_teacher_delta += delta
            improving_evaluators += int(delta < 0)
            worsening_evaluators += int(delta > 0)
        out["multi_teacher_delta_distance"] = multi_teacher_delta
        out["improving_evaluator_count"] = improving_evaluators
        out["worsening_evaluator_count"] = worsening_evaluators
        out["label_any_improves"] = int(improving_evaluators > 0)
        out["label_no_evaluator_worsens"] = int(worsening_evaluators == 0)
        out["label_pareto_improves"] = int(improving_evaluators > 0 and worsening_evaluators == 0)
        out["label_multi_teacher_improves"] = int(multi_teacher_delta < 0)
        label_counts.update(
            {
                "any_improves": out["label_any_improves"],
                "pareto_improves": out["label_pareto_improves"],
                "multi_teacher_improves": out["label_multi_teacher_improves"],
                "worsens_any": int(worsening_evaluators > 0),
            }
        )
        output_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    summary = {
        "candidate_rows": len(output_rows),
        "evaluators": [label for label, _ in analyses],
        "label_counts": dict(label_counts),
        "output": str(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
