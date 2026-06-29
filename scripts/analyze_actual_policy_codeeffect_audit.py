#!/usr/bin/env python3
"""Audit actual-policy code-effect cases against structured latent/code features.

This is a diagnostic script. It uses already-measured OCR labels only to describe
policy errors and oracle headroom; it must not be used as a deployable selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key_from_row(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["real_seed"]), int(row["source_index"]), int(row["candidate_index"]))


def stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def vector_stats(
    z: np.lib.npyio.NpzFile,
    group_index: int,
    slot: int | None,
    core_names: list[str],
) -> dict[str, Any] | None:
    if slot is None:
        return None
    if not bool(z["mask"][group_index, slot]):
        return None

    code_center = z["code_center"][group_index, slot].astype(np.float32)
    nearest_center = z["nearest_center"][group_index, slot].astype(np.float32)
    delta_center = z["delta_center"][group_index, slot].astype(np.float32)
    latent_residual = z["latent_residual"][group_index, slot].astype(np.float32)
    window_residual = z["latent_window_residual"][group_index, slot].astype(np.float32)
    core = z["core_features"][group_index, slot].astype(np.float32)

    def norm(x: np.ndarray) -> float:
        return float(np.linalg.norm(x.reshape(-1).astype(np.float32)))

    def abs_mean(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x.reshape(-1).astype(np.float32))))

    def max_abs(x: np.ndarray) -> float:
        return float(np.max(np.abs(x.reshape(-1).astype(np.float32))))

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        aa = a.reshape(-1).astype(np.float32)
        bb = b.reshape(-1).astype(np.float32)
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(aa, bb) / denom)

    candidate_quant_error = norm(latent_residual - code_center)
    nearest_quant_error = norm(latent_residual - nearest_center)
    relative_quant_error = candidate_quant_error / max(nearest_quant_error, 1e-12)
    window_l2 = norm(window_residual)
    center_l2 = norm(latent_residual)

    out: dict[str, Any] = {
        "topk_rank": int(z["topk_ranks"][group_index, slot]),
        "code_index": int(z["code_indices"][group_index, slot]),
        "nearest_index": int(z["nearest_indices"][group_index, slot]),
        "is_nearest": bool(float(z["is_nearest"][group_index, slot]) > 0.5),
        "oracle_choice_label": bool(float(z["labels"][group_index, slot]) > 0.5),
        "parseq_delta": float(z["parseq_delta"][group_index, slot]),
        "tesseract_delta": float(z["tesseract_delta"][group_index, slot]),
        "code_delta_l2": norm(delta_center),
        "code_delta_abs_mean": abs_mean(delta_center),
        "code_delta_max_abs": max_abs(delta_center),
        "candidate_quant_error_l2": candidate_quant_error,
        "nearest_quant_error_l2": nearest_quant_error,
        "relative_quant_error": relative_quant_error,
        "residual_delta_cosine": cosine(latent_residual, delta_center),
        "code_nearest_cosine": cosine(code_center, nearest_center),
        "latent_residual_l2": center_l2,
        "latent_residual_abs_mean": abs_mean(latent_residual),
        "latent_residual_max_abs": max_abs(latent_residual),
        "window_residual_l2": window_l2,
        "window_residual_abs_mean": abs_mean(window_residual),
        "window_residual_max_abs": max_abs(window_residual),
        "center_window_l2_fraction": float((center_l2 * center_l2) / max(window_l2 * window_l2, 1e-12)),
    }
    for idx, name in enumerate(core_names):
        value = float(core[idx])
        if name == "log1p_clipped_assignment_relative_error":
            out["assignment_relative_error_from_core"] = float(math.exp(value) - 1.0)
        else:
            out[f"core_{name}"] = value
    return out


def find_slot_by_code(z: np.lib.npyio.NpzFile, group_index: int, code_index: int) -> int | None:
    codes = z["code_indices"][group_index]
    mask = z["mask"][group_index]
    for slot, (code, valid) in enumerate(zip(codes, mask)):
        if bool(valid) and int(code) == int(code_index):
            return int(slot)
    return None


def category_from_policy(row: dict[str, Any], oracle_code: int, nearest_code: int) -> str:
    selected_code = int(row["code_index"])
    changed = selected_code != nearest_code
    oracle_changed = oracle_code != nearest_code
    if changed and oracle_changed and selected_code == oracle_code:
        return "exact_oracle_change"
    if changed and oracle_changed and selected_code != oracle_code:
        return "wrong_change"
    if changed and not oracle_changed:
        return "false_change"
    if (not changed) and oracle_changed:
        return "missed_oracle"
    return "correct_nearest"


def summarize_feature_groups(
    rows: list[dict[str, Any]],
    feature_key: str,
    feature_names: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        category = str(row["current_policy_category"])
        feats = row.get(feature_key)
        if not feats:
            continue
        for name in feature_names:
            value = feats.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped[category][name].append(float(value))
    return {
        category: {name: stats(values) for name, values in sorted(features.items())}
        for category, features in sorted(grouped.items())
    }


def rank_separators(
    rows: list[dict[str, Any]],
    category_a: str,
    category_b: str,
    feature_key: str,
    feature_names: list[str],
    topn: int = 16,
) -> list[dict[str, Any]]:
    a_rows = [r for r in rows if r["current_policy_category"] == category_a and r.get(feature_key)]
    b_rows = [r for r in rows if r["current_policy_category"] == category_b and r.get(feature_key)]
    ranked: list[dict[str, Any]] = []
    for name in feature_names:
        av = [float(r[feature_key][name]) for r in a_rows if isinstance(r[feature_key].get(name), (int, float))]
        bv = [float(r[feature_key][name]) for r in b_rows if isinstance(r[feature_key].get(name), (int, float))]
        if not av or not bv:
            continue
        aa = np.asarray(av, dtype=np.float64)
        bb = np.asarray(bv, dtype=np.float64)
        pooled = float(np.sqrt(np.var(aa) + np.var(bb) + 1e-12))
        effect = float((aa.mean() - bb.mean()) / pooled)
        ranked.append(
            {
                "feature": name,
                f"{category_a}_mean": float(aa.mean()),
                f"{category_b}_mean": float(bb.mean()),
                "standardized_mean_difference": effect,
                "abs_standardized_mean_difference": abs(effect),
                f"{category_a}_n": int(aa.size),
                f"{category_b}_n": int(bb.size),
            }
        )
    ranked.sort(key=lambda x: x["abs_standardized_mean_difference"], reverse=True)
    return ranked[:topn]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-policy", type=Path, required=True)
    parser.add_argument("--shortlist-policy", type=Path, required=True)
    parser.add_argument("--top8-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_path = args.current_policy
    shortlist_path = args.shortlist_policy
    top8_path = args.top8_features
    meta_path = Path(f"{top8_path}.meta.json")

    current_rows = load_jsonl(current_path)
    shortlist_rows = load_jsonl(shortlist_path)
    shortlist_by_key = {key_from_row(row): row for row in shortlist_rows}

    z = np.load(top8_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    core_names = list(meta["core_feature_names"])
    group_by_key = {
        (int(item["real_seed"]), int(item["source_index"]), int(item["candidate_index"])): (idx, item)
        for idx, item in enumerate(meta["group_metadata"])
    }

    feature_names = [
        "topk_rank",
        "assignment_relative_error_from_core",
        "code_delta_l2",
        "code_delta_abs_mean",
        "code_delta_max_abs",
        "candidate_quant_error_l2",
        "nearest_quant_error_l2",
        "relative_quant_error",
        "residual_delta_cosine",
        "code_nearest_cosine",
        "latent_residual_l2",
        "latent_residual_abs_mean",
        "latent_residual_max_abs",
        "window_residual_l2",
        "window_residual_abs_mean",
        "window_residual_max_abs",
        "center_window_l2_fraction",
        "core_reference_length",
        "core_parseq_nearest_distance",
        "core_tesseract_nearest_distance",
    ]

    audited: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for current in current_rows:
        key = key_from_row(current)
        if key not in group_by_key:
            missing.append({"key": list(key), "reason": "missing_top8_group"})
            continue
        if key not in shortlist_by_key:
            missing.append({"key": list(key), "reason": "missing_shortlist_policy"})
            continue
        group_index, group_meta = group_by_key[key]
        shortlist = shortlist_by_key[key]
        nearest_code = int(group_meta["nearest_code"])
        oracle_code = int(group_meta["oracle_code"])
        current_code = int(current["code_index"])
        shortlist_code = int(shortlist["code_index"])

        current_slot = find_slot_by_code(z, group_index, current_code)
        shortlist_slot = find_slot_by_code(z, group_index, shortlist_code)
        nearest_slot = find_slot_by_code(z, group_index, nearest_code)
        oracle_slot = find_slot_by_code(z, group_index, oracle_code)

        current_category = category_from_policy(current, oracle_code, nearest_code)
        shortlist_category = category_from_policy(shortlist, oracle_code, nearest_code)

        audited.append(
            {
                "key": {
                    "real_seed": key[0],
                    "source_index": key[1],
                    "candidate_index": key[2],
                },
                "source": group_meta.get("source"),
                "reference": group_meta.get("reference"),
                "reference_length": len(str(group_meta.get("reference", ""))),
                "source_image": group_meta.get("source_image"),
                "nearest_code": nearest_code,
                "oracle_code": oracle_code,
                "current_code": current_code,
                "shortlist_code": shortlist_code,
                "oracle_changed_code": oracle_code != nearest_code,
                "current_policy_category": current_category,
                "shortlist_policy_category": shortlist_category,
                "shortlist_oracle_in_policy": bool(shortlist.get("oracle_in_shortlist", False)),
                "current_tesseract_delta": int(current.get("tesseract_delta_vs_nearest", 0)),
                "current_parseq_delta": int(current.get("parseq_delta_vs_nearest", 0)),
                "shortlist_tesseract_delta": int(shortlist.get("tesseract_delta_vs_nearest", 0)),
                "shortlist_parseq_delta": int(shortlist.get("parseq_delta_vs_nearest", 0)),
                "current_selector_score": float(current.get("selector_score", 0.0)),
                "shortlist_selector_score": float(shortlist.get("selector_score", 0.0)),
                "slots": {
                    "nearest": nearest_slot,
                    "oracle": oracle_slot,
                    "current": current_slot,
                    "shortlist": shortlist_slot,
                },
                "nearest_features": vector_stats(z, group_index, nearest_slot, core_names),
                "oracle_features": vector_stats(z, group_index, oracle_slot, core_names),
                "current_features": vector_stats(z, group_index, current_slot, core_names),
                "shortlist_features": vector_stats(z, group_index, shortlist_slot, core_names),
            }
        )

    counts = Counter(row["current_policy_category"] for row in audited)
    shortlist_counts = Counter(row["shortlist_policy_category"] for row in audited)
    source_counts = Counter(row["source"] for row in audited)

    current_changed_rows = [
        row
        for row in audited
        if row["current_code"] != row["nearest_code"]
    ]
    oracle_changed_rows = [row for row in audited if row["oracle_changed_code"]]
    shortlist_extra_rows = [
        row
        for row in audited
        if row["shortlist_policy_category"] == "exact_oracle_change"
        and row["current_policy_category"] != "exact_oracle_change"
    ]
    shortlist_missed_rows = [
        row
        for row in audited
        if row["shortlist_policy_category"] == "missed_oracle"
    ]

    result = {
        "description": (
            "Diagnostic audit of actual-counted current-best and top-4 shortlist-oracle "
            "assignment policies against Eval300 top-8 structured latent/code-effect features. "
            "Held-out OCR labels are used only to categorize already-evaluated policy errors."
        ),
        "validity": "diagnostic_only_not_a_selector_or_promotion",
        "inputs": {
            "current_policy": str(current_path),
            "shortlist_policy": str(shortlist_path),
            "top8_features": str(top8_path),
            "top8_meta": str(meta_path),
        },
        "input_hashes": {
            str(current_path): sha256_file(current_path),
            str(shortlist_path): sha256_file(shortlist_path),
            str(top8_path): sha256_file(top8_path),
            str(meta_path): sha256_file(meta_path),
        },
        "counts": {
            "audited_groups": len(audited),
            "missing_groups": len(missing),
            "current_policy_categories": dict(sorted(counts.items())),
            "shortlist_policy_categories": dict(sorted(shortlist_counts.items())),
            "sources": dict(sorted(source_counts.items())),
            "current_tesseract_delta_sum": int(sum(row["current_tesseract_delta"] for row in audited)),
            "shortlist_tesseract_delta_sum": int(sum(row["shortlist_tesseract_delta"] for row in audited)),
            "current_parseq_delta_sum": int(sum(row["current_parseq_delta"] for row in audited)),
            "shortlist_parseq_delta_sum": int(sum(row["shortlist_parseq_delta"] for row in audited)),
        },
        "feature_group_summaries": {
            "current_selected_by_current_category": summarize_feature_groups(
                audited, "current_features", feature_names
            ),
            "oracle_candidate_by_current_category": summarize_feature_groups(
                [row for row in audited if row.get("oracle_features")],
                "oracle_features",
                feature_names,
            ),
        },
        "feature_separators": {
            "current_false_vs_exact_current_selected": rank_separators(
                audited,
                "false_change",
                "exact_oracle_change",
                "current_features",
                feature_names,
            ),
            "missed_vs_exact_oracle_candidates": rank_separators(
                [row for row in audited if row.get("oracle_features")],
                "missed_oracle",
                "exact_oracle_change",
                "oracle_features",
                feature_names,
            ),
        },
        "interesting_cases": {
            "current_changed": current_changed_rows,
            "current_oracle_changed": oracle_changed_rows,
            "shortlist_extra_over_current": shortlist_extra_rows,
            "shortlist_missed_oracle": shortlist_missed_rows,
        },
        "missing": missing,
        "conclusion": {
            "summary": (
                "The current bad change is code/latent-feature-wise close to useful changes, "
                "while many missed useful oracle candidates have higher top-k ranks and worse "
                "residual-MSE-style quantization error than nearest. This supports a richer "
                "candidate-conditioned verifier/objective rather than scalar code geometry, "
                "teacher vetoes, or residual-fidelity thresholds."
            ),
            "next_action": (
                "Use this audit to define a new representation or policy-level verifier. "
                "Do not tune a deployable threshold on these held-out Tesseract categories."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    exact = counts.get("exact_oracle_change", 0)
    false = counts.get("false_change", 0)
    missed = counts.get("missed_oracle", 0)
    shortlist_extra = len(shortlist_extra_rows)
    shortlist_missed = len(shortlist_missed_rows)

    def fmt_stat(section: dict[str, Any], category: str, feature: str) -> str:
        item = section.get(category, {}).get(feature, {})
        if not item or item.get("n", 0) == 0:
            return "n/a"
        return f"{item['mean']:.4g} med {item['median']:.4g} (n={item['n']})"

    current_summary = result["feature_group_summaries"]["current_selected_by_current_category"]
    oracle_summary = result["feature_group_summaries"]["oracle_candidate_by_current_category"]

    report = [
        "# Eval300 Actual-Policy Code-Effect Audit",
        "",
        "This is a diagnostic-only audit. It categorizes already-evaluated actual `.oscr` policies with held-out Tesseract labels, but does not define a deployable selector.",
        "",
        "## Inputs",
        "",
        f"- current policy: `{current_path}`",
        f"- shortlist-oracle policy: `{shortlist_path}`",
        f"- top-8 structured feature artifact: `{top8_path}`",
        "",
        "## Counts",
        "",
        f"- audited groups: {len(audited)}",
        f"- current policy: {exact} exact oracle changes, {false} false changes, {missed} missed oracle changes",
        f"- current Tesseract/PARSeq deltas: {result['counts']['current_tesseract_delta_sum']} / {result['counts']['current_parseq_delta_sum']}",
        f"- shortlist Tesseract/PARSeq deltas: {result['counts']['shortlist_tesseract_delta_sum']} / {result['counts']['shortlist_parseq_delta_sum']}",
        f"- shortlist extra exact changes over current: {shortlist_extra}",
        f"- shortlist missed oracle groups: {shortlist_missed}",
        "",
        "## Feature Signals",
        "",
        "| feature | current exact | current false | missed oracle candidate |",
        "|---|---:|---:|---:|",
    ]
    for feature in [
        "topk_rank",
        "assignment_relative_error_from_core",
        "candidate_quant_error_l2",
        "relative_quant_error",
        "code_delta_l2",
        "residual_delta_cosine",
        "window_residual_l2",
        "center_window_l2_fraction",
        "core_tesseract_nearest_distance",
    ]:
        report.append(
            "| "
            + feature
            + " | "
            + fmt_stat(current_summary, "exact_oracle_change", feature)
            + " | "
            + fmt_stat(current_summary, "false_change", feature)
            + " | "
            + fmt_stat(oracle_summary, "missed_oracle", feature)
            + " |"
        )

    report.extend(
        [
            "",
            "## Notable Cases",
            "",
            "| category | seed/source/cand | source | ref | nearest -> current / oracle / shortlist | Tesseract deltas current/shortlist | ranks current/oracle | rel-error current/oracle |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    notable_rows = current_changed_rows + [
        row for row in shortlist_missed_rows if row not in current_changed_rows
    ]
    for row in notable_rows:
        cur_f = row.get("current_features") or {}
        ora_f = row.get("oracle_features") or {}
        key = row["key"]
        report.append(
            f"| {row['current_policy_category']} | "
            f"{key['real_seed']}/{key['source_index']}/{key['candidate_index']} | "
            f"{row['source']} | {row['reference']} | "
            f"{row['nearest_code']} -> {row['current_code']} / {row['oracle_code']} / {row['shortlist_code']} | "
            f"{row['current_tesseract_delta']}/{row['shortlist_tesseract_delta']} | "
            f"{cur_f.get('topk_rank', 'n/a')}/{ora_f.get('topk_rank', 'n/a')} | "
            f"{cur_f.get('assignment_relative_error_from_core', float('nan')):.4g}/"
            f"{ora_f.get('assignment_relative_error_from_core', float('nan')):.4g} |"
        )

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The one current false change is not separable by a simple low residual-error or rank rule; its rank and relative quantization error sit inside the useful-change range.",
            "- Many missed useful oracle candidates have non-nearest ranks up to 6 or 7 and often worse residual-fidelity error than nearest, so residual-MSE or top-1 confidence is the wrong objective.",
            "- The actual-counted top-4 shortlist oracle mostly converts current missed groups at unchanged payload size, while the two remaining misses are first-stage shortlist coverage failures.",
            "- Next work should change the candidate-local code-effect evidence or policy-level verifier. This audit should not be used to tune held-out-OCR thresholds.",
            "",
        ]
    )
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
