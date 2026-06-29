#!/usr/bin/env python3
"""Enrich the actual assignment failure bank with deployable feature evidence.

This is a diagnostic-only analysis. Held-out OCR strings and deltas enter only
through the existing failure-bank target labels; they are not used as deployable
features or as a promoted selector.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


NUMERIC_FEATURES = [
    "topk_rank",
    "assignment_relative_error",
    "codebook_delta_l2",
    "codebook_delta_rms",
    "codebook_delta_abs_mean",
    "codebook_code_nearest_cosine",
    "img_width",
    "img_height",
    "img_area",
    "img_aspect",
    "img_source_mean",
    "img_source_std",
    "img_source_grad_mean",
    "img_source_edge_density",
    "img_source_nearest_mse",
    "img_source_nearest_changed_fraction",
    "img_variant_nearest_mse",
    "img_variant_nearest_changed_fraction",
    "img_variant_nearest_grad_abs_delta",
    "img_variant_nearest_edge_density_delta",
    "img_variant_nearest_bbox_area_fraction",
    "img_variant_nearest_bbox_height_fraction",
    "img_variant_nearest_bbox_width_fraction",
    "img_source_variant_mse",
    "img_source_variant_changed_fraction",
    "img_source_variant_grad_abs_delta",
    "img_source_variant_edge_density_delta",
]

TARGET_ORDER = [
    "recover_shortlist_oracle_change",
    "reject_current_bad_change",
    "improve_first_stage_shortlist_recall",
    "keep_current_good_change",
    "keep_noop",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key3_from_case(case: dict[str, Any]) -> tuple[int, int, int]:
    key = case["key"]
    return (int(key["real_seed"]), int(key["source_index"]), int(key["candidate_index"]))


def key3_from_row(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"]))


def key4_from_row(row: dict[str, Any]) -> tuple[int, int, int, int]:
    a, b, c = key3_from_row(row)
    return (a, b, c, int(row["code_index"]))


def parse_score_file(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--score-file expects name=path, got {value!r}")
    name, path = value.split("=", 1)
    if not name:
        raise SystemExit(f"empty score-file name in {value!r}")
    return name, Path(path)


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def scalar_metrics(prefix: str, value: object, out: dict[str, Any]) -> None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            scalar_metrics(f"{prefix}_{key}", child, out)


def load_candidate_rows(path: Path) -> tuple[dict[tuple[int, int, int, int], dict[str, Any]], dict[tuple[int, int, int], list[dict[str, Any]]]]:
    rows = read_jsonl(path)
    by_key4: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    by_group: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key4[key4_from_row(row)] = row
        by_group[key3_from_row(row)].append(row)
    for group_rows in by_group.values():
        group_rows.sort(key=lambda row: (int(row["topk_rank"]), int(row["code_index"])))
    return by_key4, dict(by_group)


def load_scores(path: Path) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    raw_rows = read_jsonl(path)
    values: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    for row in raw_rows:
        score = finite_float(row.get("score"))
        if score is None:
            continue
        values[key4_from_row(row)].append(score)
    grouped: dict[tuple[int, int, int], list[tuple[tuple[int, int, int, int], float]]] = defaultdict(list)
    result: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for key, score_values in values.items():
        stats = summarize_values(score_values)
        result[key] = stats
        grouped[key[:3]].append((key, float(stats["mean"])))
    for group_items in grouped.values():
        group_items.sort(key=lambda item: (-item[1], item[0][3]))
        for rank, (key, _) in enumerate(group_items, start=1):
            result[key]["rank_desc_mean"] = rank
            result[key]["group_count"] = len(group_items)
    return result


def numeric_features(row: dict[str, Any] | None) -> dict[str, float | None]:
    if row is None:
        return {name: None for name in NUMERIC_FEATURES}
    return {name: finite_float(row.get(name)) for name in NUMERIC_FEATURES}


def score_bundle(
    key3: tuple[int, int, int],
    code: int,
    score_sources: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    key4 = (*key3, int(code))
    out: dict[str, dict[str, Any]] = {}
    for name, scores in score_sources.items():
        out[name] = scores.get(key4, {"count": 0, "mean": None, "std": None, "min": None, "max": None})
    return out


def choose_target_role(case: dict[str, Any]) -> tuple[str, int]:
    target = case["next_model_target"]
    if target == "recover_shortlist_oracle_change":
        return "shortlist", int(case["shortlist_code"])
    if target == "keep_current_good_change":
        return "current", int(case["current_code"])
    if target == "reject_current_bad_change":
        return "current_bad", int(case["current_code"])
    if target == "improve_first_stage_shortlist_recall":
        return "oracle_missing", int(case["oracle_code"])
    return "nearest", int(case["nearest_code"])


def enrich_cases(
    cases: list[dict[str, Any]],
    candidate_by_key4: dict[tuple[int, int, int, int], dict[str, Any]],
    candidate_by_group: dict[tuple[int, int, int], list[dict[str, Any]]],
    score_sources: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for case in cases:
        key3 = key3_from_case(case)
        codes = {
            "nearest": int(case["nearest_code"]),
            "current": int(case["current_code"]),
            "shortlist": int(case["shortlist_code"]),
            "oracle": int(case["oracle_code"]),
        }
        role, target_code = choose_target_role(case)
        codes["target_role_candidate"] = target_code
        row_features: dict[str, Any] = {}
        row_scores: dict[str, Any] = {}
        row_presence: dict[str, bool] = {}
        for role_name, code in codes.items():
            row = candidate_by_key4.get((*key3, code))
            row_presence[role_name] = row is not None
            row_features[role_name] = numeric_features(row)
            row_scores[role_name] = score_bundle(key3, code, score_sources)
        group_rows = candidate_by_group.get(key3, [])
        enriched.append(
            {
                "key": case["key"],
                "source": case.get("source"),
                "split": case.get("split"),
                "reference": case.get("reference"),
                "next_model_target": case["next_model_target"],
                "current_policy_category": case["current_policy_category"],
                "shortlist_policy_category": case["shortlist_policy_category"],
                "codes": codes,
                "target_candidate_role": role,
                "oracle_in_shortlist": bool(case.get("oracle_in_shortlist")),
                "oracle_topk_rank": int(case["oracle_topk_rank"]),
                "current_topk_rank": int(case["current_topk_rank"]),
                "shortlist_topk_rank": int(case["shortlist_topk_rank"]),
                "current_selector_score": finite_float(case.get("current_selector_score")),
                "shortlist_selector_score": finite_float(case.get("shortlist_selector_score")),
                "ocr_deltas": {
                    "current_vs_nearest": int(case["ocr"]["current_delta_vs_nearest"]),
                    "shortlist_vs_nearest": int(case["ocr"]["shortlist_delta_vs_nearest"]),
                    "shortlist_vs_current": int(case["ocr"]["shortlist_delta_vs_current"]),
                },
                "row_presence": row_presence,
                "features": row_features,
                "scores": row_scores,
                "group_codes_available": [int(row["code_index"]) for row in group_rows],
                "images": case.get("images", {}),
                "source_image": case.get("source_image"),
                "ocr": {
                    "nearest_prediction": case["ocr"].get("nearest_prediction"),
                    "current_prediction": case["ocr"].get("current_prediction"),
                    "shortlist_prediction": case["ocr"].get("shortlist_prediction"),
                },
            }
        )
    return enriched


def summarize_enriched(enriched: list[dict[str, Any]], score_names: list[str]) -> dict[str, Any]:
    by_target = Counter(row["next_model_target"] for row in enriched)
    target_stats: dict[str, Any] = {}
    for target in TARGET_ORDER:
        rows = [row for row in enriched if row["next_model_target"] == target]
        if not rows:
            continue
        feature_stats: dict[str, Any] = {}
        score_stats: dict[str, Any] = {}
        for feature in NUMERIC_FEATURES:
            vals = [
                row["features"]["target_role_candidate"][feature]
                for row in rows
                if row["features"]["target_role_candidate"].get(feature) is not None
            ]
            feature_stats[feature] = summarize_values([float(v) for v in vals])
        for score_name in score_names:
            means = []
            ranks = []
            for row in rows:
                stats = row["scores"]["target_role_candidate"].get(score_name, {})
                mean = finite_float(stats.get("mean"))
                rank = finite_float(stats.get("rank_desc_mean"))
                if mean is not None:
                    means.append(mean)
                if rank is not None:
                    ranks.append(rank)
            score_stats[score_name] = {
                "mean_score": summarize_values(means),
                "rank_desc_mean": summarize_values(ranks),
                "rank_le_1": int(sum(1 for rank in ranks if rank <= 1)),
                "rank_le_2": int(sum(1 for rank in ranks if rank <= 2)),
                "rank_le_4": int(sum(1 for rank in ranks if rank <= 4)),
            }
        target_stats[target] = {
            "count": len(rows),
            "oracle_topk_rank_counts": dict(sorted(Counter(str(row["oracle_topk_rank"]) for row in rows).items())),
            "target_candidate_role_counts": dict(sorted(Counter(row["target_candidate_role"] for row in rows).items())),
            "missing_target_feature_rows": int(sum(not row["row_presence"]["target_role_candidate"] for row in rows)),
            "feature_stats": feature_stats,
            "score_stats": score_stats,
        }
    missing_by_role: dict[str, int] = Counter()
    for row in enriched:
        for role, present in row["row_presence"].items():
            if not present:
                missing_by_role[role] += 1
    summary = {
        "cases": len(enriched),
        "by_next_model_target": dict(sorted(by_target.items())),
        "missing_feature_rows_by_role": dict(sorted(missing_by_role.items())),
        "targets": target_stats,
    }
    if "recover_shortlist_oracle_change" in target_stats and "reject_current_bad_change" in target_stats:
        summary["separation_note"] = (
            "Diagnostic only: recover and reject strata are held-out OCR-defined targets; "
            "single-case reject contrasts are not a robust threshold."
        )
    return summary


def ascii_text(value: object, limit: int = 80) -> str:
    text = str(value)
    text = text.replace("\n", " ")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text.encode("ascii", "replace").decode("ascii")


def open_image(path_value: object, height: int = 96) -> Image.Image:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return Image.new("RGB", (height * 2, height), (240, 240, 240))
    img = Image.open(path).convert("RGB")
    scale = height / max(1, img.height)
    width = max(1, int(round(img.width * scale)))
    return img.resize((width, height), Image.Resampling.BICUBIC)


def make_contact_sheet(path: Path, rows: list[dict[str, Any]], limit: int = 16) -> None:
    if not rows:
        return
    font = ImageFont.load_default()
    labels = ["source", "nearest", "current", "shortlist"]
    thumb_h = 96
    label_h = 48
    gap = 8
    margin = 10
    rendered_rows: list[tuple[Image.Image, list[str]]] = []
    max_w = 0
    for row in rows[:limit]:
        images = [
            open_image(row.get("source_image"), thumb_h),
            open_image(row["images"].get("nearest"), thumb_h),
            open_image(row["images"].get("current"), thumb_h),
            open_image(row["images"].get("shortlist"), thumb_h),
        ]
        col_w = max(max(img.width for img in images), 120)
        canvas_w = len(images) * col_w + (len(images) - 1) * gap
        canvas_h = thumb_h + label_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        x = 0
        for label, img in zip(labels, images, strict=True):
            canvas.paste(img, (x + (col_w - img.width) // 2, label_h))
            draw.text((x + 2, 2), label, fill=(0, 0, 0), font=font)
            x += col_w + gap
        key = row["key"]
        title = (
            f"{key['real_seed']}/{key['source_index']}/{key['candidate_index']} "
            f"{row['next_model_target']} ref={ascii_text(row['reference'], 24)} "
            f"codes n/c/s/o={row['codes']['nearest']}/{row['codes']['current']}/"
            f"{row['codes']['shortlist']}/{row['codes']['oracle']} "
            f"d={row['ocr_deltas']['current_vs_nearest']}/"
            f"{row['ocr_deltas']['shortlist_vs_nearest']}/"
            f"{row['ocr_deltas']['shortlist_vs_current']}"
        )
        rendered_rows.append((canvas, [title]))
        max_w = max(max_w, canvas_w, 900)
    row_h = thumb_h + label_h + 32
    sheet = Image.new("RGB", (max_w + margin * 2, row_h * len(rendered_rows) + margin), "white")
    draw = ImageDraw.Draw(sheet)
    y = margin
    for canvas, title_lines in rendered_rows:
        draw.text((margin, y), ascii_text(title_lines[0], 160), fill=(0, 0, 0), font=font)
        sheet.paste(canvas, (margin, y + 22))
        y += row_h
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_contact_sheets(out_dir: Path, enriched: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for target in TARGET_ORDER[:-1]:
        rows = [row for row in enriched if row["next_model_target"] == target]
        rows.sort(key=lambda row: (row["ocr_deltas"]["shortlist_vs_current"], row["key"]["real_seed"], row["key"]["source_index"]))
        if not rows:
            continue
        path = out_dir / f"{target}.png"
        make_contact_sheet(path, rows)
        out[target] = str(path)
    return out


def report_feature_rows(summary: dict[str, Any], feature_names: list[str]) -> list[str]:
    lines = [
        "| target | count | feature | mean | min | max |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for target in TARGET_ORDER:
        target_data = summary["targets"].get(target)
        if not target_data:
            continue
        for feature in feature_names:
            stats = target_data["feature_stats"][feature]
            mean = stats["mean"]
            lines.append(
                f"| {target} | {target_data['count']} | `{feature}` | "
                f"{mean if mean is not None else 'null'} | "
                f"{stats['min'] if stats['min'] is not None else 'null'} | "
                f"{stats['max'] if stats['max'] is not None else 'null'} |"
            )
    return lines


def report_score_rows(summary: dict[str, Any], score_names: list[str]) -> list[str]:
    lines = [
        "| target | score | mean score | mean rank | rank<=1 | rank<=2 | rank<=4 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for target in TARGET_ORDER:
        target_data = summary["targets"].get(target)
        if not target_data:
            continue
        for score_name in score_names:
            stats = target_data["score_stats"][score_name]
            lines.append(
                f"| {target} | `{score_name}` | "
                f"{stats['mean_score']['mean'] if stats['mean_score']['mean'] is not None else 'null'} | "
                f"{stats['rank_desc_mean']['mean'] if stats['rank_desc_mean']['mean'] is not None else 'null'} | "
                f"{stats['rank_le_1']} | {stats['rank_le_2']} | {stats['rank_le_4']} |"
            )
    return lines


def report_priority_rows(enriched: list[dict[str, Any]], score_names: list[str], limit: int = 24) -> list[str]:
    rows = [row for row in enriched if row["next_model_target"] != "keep_noop"]
    rows.sort(key=lambda row: (TARGET_ORDER.index(row["next_model_target"]), row["key"]["real_seed"], row["key"]["source_index"]))
    header_scores = " | ".join(f"{name} score/rank" for name in score_names)
    lines = [
        f"| target | key | source | ref | role | codes n/c/s/o | OCR deltas c/s/s-c | current gate | {header_scores} |",
        "|---|---|---|---|---|---|---:|---:|" + "---|" * len(score_names),
    ]
    for row in rows[:limit]:
        key = row["key"]
        score_cells = []
        for name in score_names:
            stats = row["scores"]["target_role_candidate"].get(name, {})
            score_cells.append(f"{stats.get('mean', 'null')}/{stats.get('rank_desc_mean', 'null')}")
        lines.append(
            f"| {row['next_model_target']} | {key['real_seed']}/{key['source_index']}/{key['candidate_index']} | "
            f"{row['source']} | `{ascii_text(row['reference'], 28)}` | {row['target_candidate_role']} | "
            f"{row['codes']['nearest']}/{row['codes']['current']}/{row['codes']['shortlist']}/{row['codes']['oracle']} | "
            f"{row['ocr_deltas']['current_vs_nearest']}/{row['ocr_deltas']['shortlist_vs_nearest']}/"
            f"{row['ocr_deltas']['shortlist_vs_current']} | {row['current_selector_score']} | "
            f"{' | '.join(score_cells)} |"
        )
    return lines


def write_report(path: Path, result: dict[str, Any], score_names: list[str]) -> None:
    summary = result["summary"]
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Diagnostic-only enrichment of the actual assignment failure bank with deployable candidate features.",
        "Held-out OCR outcomes define the strata but are not used as deployable inputs.",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary['cases']}`",
        f"- Target counts: `{summary['by_next_model_target']}`",
        f"- Missing feature rows by role: `{summary['missing_feature_rows_by_role']}`",
        "",
        "## Score Rank Diagnostic",
        "",
    ]
    lines.extend(report_score_rows(summary, score_names))
    lines.extend(
        [
            "",
            "## Feature Means",
            "",
        ]
    )
    lines.extend(
        report_feature_rows(
            summary,
            [
                "topk_rank",
                "assignment_relative_error",
                "codebook_delta_l2",
                "codebook_code_nearest_cosine",
                "img_variant_nearest_mse",
                "img_variant_nearest_changed_fraction",
                "img_variant_nearest_bbox_area_fraction",
                "img_source_variant_mse",
                "img_source_edge_density",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Priority Rows",
            "",
        ]
    )
    lines.extend(report_priority_rows(result["enriched_cases"], score_names))
    if result.get("contact_sheets"):
        lines.extend(["", "## Contact Sheets", ""])
        for target, sheet in result["contact_sheets"].items():
            lines.append(f"- {target}: `{sheet}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This audit should guide the next verifier design, not directly tune thresholds on the 75 held-out groups.",
            "- The recover target is a high-precision selection problem inside the verified shortlist; the reject target is the abstention floor.",
            "- If current deployable scores rank recover targets highly but also rank the reject case highly, add evidence rather than sweep thresholds.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--score-file", action="append", default=[], help="Named score source as name=path.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path)
    parser.add_argument("--experiment-id", default="eval300_actual_assignment_failure_feature_audit")
    args = parser.parse_args()

    bank = read_json(args.bank)
    candidate_by_key4, candidate_by_group = load_candidate_rows(args.candidate_table)
    score_sources: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]] = {}
    score_inputs: dict[str, str] = {}
    for item in args.score_file:
        name, path = parse_score_file(item)
        score_sources[name] = load_scores(path)
        score_inputs[name] = str(path)
    enriched = enrich_cases(bank["cases"], candidate_by_key4, candidate_by_group, score_sources)
    summary = summarize_enriched(enriched, list(score_sources))
    contact_sheets: dict[str, str] = {}
    if args.contact_sheet_dir:
        contact_sheets = write_contact_sheets(args.contact_sheet_dir, enriched)

    result = {
        "experiment_id": args.experiment_id,
        "validity": "valid_diagnostic",
        "inputs": {
            "bank": str(args.bank),
            "candidate_table": str(args.candidate_table),
            "score_files": score_inputs,
        },
        "summary": summary,
        "contact_sheets": contact_sheets,
        "enriched_cases": enriched,
    }
    metrics: dict[str, Any] = {}
    scalar_metrics("summary", summary, metrics)
    result["aggregate"] = {"scalar_metrics": {key: {"value": value} for key, value in metrics.items()}}
    write_json(args.output, result)
    write_report(args.report, result, list(score_sources))
    print(json.dumps({"output": str(args.output), "report": str(args.report), "contact_sheets": contact_sheets}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
