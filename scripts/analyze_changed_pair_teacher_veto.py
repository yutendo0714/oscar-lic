#!/usr/bin/env python3
"""Analyze whether train-teacher deltas can veto changed OCR pairs safely."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pairs(path_by_model: dict[str, Path]) -> dict[tuple[str, int], dict]:
    pairs: dict[tuple[str, int], dict] = {}
    for model, path in path_by_model.items():
        rows = read_jsonl(path)
        for row in rows:
            key = (str(row["pair_label"]), int(row["row_index"]))
            variant = str(row["variant"])
            pair = pairs.setdefault(
                key,
                {
                    "pair_label": key[0],
                    "row_index": key[1],
                    "reference": row["reference"],
                    "source": row.get("source"),
                    "split": row.get("split"),
                    "tesseract_delta": int(row["delta_distance"]),
                    "delta_exact": int(row["delta_exact"]),
                    "models": {},
                },
            )
            pair["models"].setdefault(model, {})[variant] = row
    for key, pair in pairs.items():
        for model, variants in pair["models"].items():
            if set(variants) != {"baseline", "candidate"}:
                raise ValueError(f"{key} {model}: missing paired variants")
            baseline = variants["baseline"]
            candidate = variants["candidate"]
            pair["models"][model] = {
                "baseline_prediction": baseline["prediction"],
                "candidate_prediction": candidate["prediction"],
                "loss_delta": candidate["teacher_loss_mean"] - baseline["teacher_loss_mean"],
                "confidence_delta": candidate["confidence"] - baseline["confidence"],
            }
    return pairs


def summarize_rule(pairs: list[dict], name: str, veto_fn: Callable[[dict], bool]) -> dict:
    kept = []
    vetoed = []
    for pair in pairs:
        (vetoed if veto_fn(pair) else kept).append(pair)
    return {
        "rule": name,
        "kept": len(kept),
        "vetoed": len(vetoed),
        "kept_tesseract_delta": sum(pair["tesseract_delta"] for pair in kept),
        "vetoed_tesseract_delta": sum(pair["tesseract_delta"] for pair in vetoed),
        "kept_improvements": sum(1 for pair in kept if pair["tesseract_delta"] < 0),
        "kept_worsens": sum(1 for pair in kept if pair["tesseract_delta"] > 0),
        "vetoed_improvements": sum(1 for pair in vetoed if pair["tesseract_delta"] < 0),
        "vetoed_worsens": sum(1 for pair in vetoed if pair["tesseract_delta"] > 0),
        "vetoed_keys": [[pair["pair_label"], pair["row_index"]] for pair in vetoed],
    }


def write_report(output: dict, path: Path) -> None:
    rows = [
        "| rule | kept | vetoed | kept Tesseract delta | vetoed improvements | vetoed worsens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rule in output["rules"]:
        rows.append(
            f"| {rule['rule']} | {rule['kept']} | {rule['vetoed']} | "
            f"{rule['kept_tesseract_delta']} | {rule['vetoed_improvements']} | {rule['vetoed_worsens']} |"
        )
    lines = [
        "# Current-Best Changed-Pair Teacher Veto Diagnostic",
        "",
        "This diagnostic tests fixed train-teacher veto rules only on rows where the current actual-bitstream policy changes Tesseract output.",
        "It is not a promotion experiment, because the rule outcomes are interpreted against held-out Tesseract deltas.",
        "",
        *rows,
        "",
        "## Per-Pair Deltas",
        "",
    ]
    for pair in output["pairs"]:
        lines.append(
            f"- `{pair['pair_label']}` row `{pair['row_index']}` ref `{pair['reference']}` "
            f"Tesseract delta `{pair['tesseract_delta']}`"
        )
        for model, values in pair["models"].items():
            lines.append(
                f"  - {model}: loss_delta `{values['loss_delta']:.6f}`, "
                f"confidence_delta `{values['confidence_delta']:.6f}`, "
                f"pred `{values['baseline_prediction']}` -> `{values['candidate_prediction']}`"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crnn-results", type=Path, required=True)
    parser.add_argument("--abinet-results", type=Path, required=True)
    parser.add_argument("--parseq-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    pairs = sorted(
        load_pairs(
            {
                "crnn": args.crnn_results,
                "abinet": args.abinet_results,
                "parseq": args.parseq_results,
            }
        ).values(),
        key=lambda pair: (pair["pair_label"], pair["row_index"]),
    )
    model_names = ["crnn", "abinet", "parseq"]
    rules = [
        (
            "veto_if_mean_loss_increases",
            lambda pair: sum(pair["models"][model]["loss_delta"] for model in model_names) / len(model_names) > 0.0,
        ),
        (
            "veto_if_crnn_abinet_loss_both_increase",
            lambda pair: pair["models"]["crnn"]["loss_delta"] > 0.0 and pair["models"]["abinet"]["loss_delta"] > 0.0,
        ),
        (
            "veto_if_all_confidence_decrease",
            lambda pair: all(pair["models"][model]["confidence_delta"] < 0.0 for model in model_names),
        ),
        (
            "veto_if_any_teacher_prediction_changes",
            lambda pair: any(
                pair["models"][model]["baseline_prediction"] != pair["models"][model]["candidate_prediction"]
                for model in model_names
            ),
        ),
        (
            "veto_if_no_teacher_loss_improves",
            lambda pair: all(pair["models"][model]["loss_delta"] >= 0.0 for model in model_names),
        ),
    ]
    output = {
        "description": "Fixed train-teacher veto diagnostics for current-best changed OCR pairs.",
        "validity": "diagnostic_only_not_a_promotion",
        "pairs": pairs,
        "rules": [summarize_rule(pairs, name, veto_fn) for name, veto_fn in rules],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(output, args.report)
    print(json.dumps({"output": str(args.output), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
