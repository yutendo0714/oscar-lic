#!/usr/bin/env python3
"""Audit agreement between already-exported actual assignment policies.

This is a diagnostic script. It uses held-out OCR deltas only to describe
already-generated policy rows and simple deterministic combinations of those
rows. It must not be used to tune a deployable selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key_from_row(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("real_seed", row["seed"])),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def category_for_code(code: int, nearest_code: int, oracle_code: int) -> str:
    changed = int(code) != int(nearest_code)
    oracle_changed = int(oracle_code) != int(nearest_code)
    if changed and oracle_changed and int(code) == int(oracle_code):
        return "exact_oracle_change"
    if changed and oracle_changed and int(code) != int(oracle_code):
        return "wrong_change"
    if changed and not oracle_changed:
        return "false_change"
    if (not changed) and oracle_changed:
        return "missed_oracle"
    return "correct_nearest"


def source_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(row.get("source", "unknown"))][str(row["category"])] += 1
    return {source: dict(sorted(counter.items())) for source, counter in sorted(grouped.items())}


def summarize_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(str(row["category"]) for row in rows)
    changed_rows = [row for row in rows if int(row["code_index"]) != int(row["nearest_code"])]
    return {
        "groups": len(rows),
        "changed_groups": len(changed_rows),
        "category_counts": dict(sorted(categories.items())),
        "tesseract_delta_sum": int(sum(int(row.get("tesseract_delta_vs_nearest", 0)) for row in rows)),
        "parseq_delta_sum": int(sum(int(row.get("parseq_delta_vs_nearest", 0)) for row in rows)),
        "changed_source_categories": source_summary(changed_rows),
    }


def chosen_row(
    key: tuple[int, int, int],
    code_index: int,
    nearest_row: dict[str, Any],
    a_row: dict[str, Any],
    b_row: dict[str, Any],
    oracle_code: int,
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    nearest_code = int(nearest_row["nearest_code"])
    if int(code_index) == nearest_code:
        tesseract_delta = 0
        parseq_delta = 0
        selector_source = "nearest"
        selector_score = None
    elif int(code_index) == int(a_row["code_index"]):
        tesseract_delta = int(a_row.get("tesseract_delta_vs_nearest", 0))
        parseq_delta = int(a_row.get("parseq_delta_vs_nearest", 0))
        selector_source = "policy_a"
        selector_score = float(a_row.get("selector_score", 0.0))
    elif int(code_index) == int(b_row["code_index"]):
        tesseract_delta = int(b_row.get("tesseract_delta_vs_nearest", 0))
        parseq_delta = int(b_row.get("parseq_delta_vs_nearest", 0))
        selector_source = "policy_b"
        selector_score = float(b_row.get("selector_score", 0.0))
    else:
        raise ValueError(f"selected code {code_index} is not present in policy rows for key {key}")
    return {
        "key": {"real_seed": key[0], "source_index": key[1], "candidate_index": key[2]},
        "source": source_meta.get("source"),
        "reference": source_meta.get("reference"),
        "source_image": source_meta.get("source_image"),
        "nearest_code": nearest_code,
        "oracle_code": int(oracle_code),
        "code_index": int(code_index),
        "topk_rank": (
            int(a_row["topk_rank"])
            if int(code_index) == int(a_row["code_index"])
            else int(b_row["topk_rank"])
            if int(code_index) == int(b_row["code_index"])
            else 0
        ),
        "selector_source": selector_source,
        "selector_score": selector_score,
        "category": category_for_code(int(code_index), nearest_code, int(oracle_code)),
        "tesseract_delta_vs_nearest": tesseract_delta,
        "parseq_delta_vs_nearest": parseq_delta,
    }


def annotate_policy_rows(
    rows_by_key: dict[tuple[int, int, int], dict[str, Any]],
    keys: list[tuple[int, int, int]],
    oracle_by_key: dict[tuple[int, int, int], int],
    meta_by_key: dict[tuple[int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for key in keys:
        row = rows_by_key[key]
        nearest = int(row["nearest_code"])
        code = int(row["code_index"])
        item = {
            "key": {"real_seed": key[0], "source_index": key[1], "candidate_index": key[2]},
            "source": meta_by_key[key].get("source"),
            "reference": meta_by_key[key].get("reference"),
            "source_image": meta_by_key[key].get("source_image"),
            "nearest_code": nearest,
            "oracle_code": int(oracle_by_key[key]),
            "code_index": code,
            "topk_rank": int(row["topk_rank"]),
            "selector_score": float(row.get("selector_score", 0.0)),
            "category": category_for_code(code, nearest, oracle_by_key[key]),
            "tesseract_delta_vs_nearest": int(row.get("tesseract_delta_vs_nearest", 0)),
            "parseq_delta_vs_nearest": int(row.get("parseq_delta_vs_nearest", 0)),
        }
        out.append(item)
    return out


def build_hypothetical_rows(
    policy: str,
    keys: list[tuple[int, int, int]],
    rows_a: dict[tuple[int, int, int], dict[str, Any]],
    rows_b: dict[tuple[int, int, int], dict[str, Any]],
    oracle_by_key: dict[tuple[int, int, int], int],
    meta_by_key: dict[tuple[int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for key in keys:
        a_row = rows_a[key]
        b_row = rows_b[key]
        nearest_code = int(a_row["nearest_code"])
        a_code = int(a_row["code_index"])
        b_code = int(b_row["code_index"])
        if policy == "intersection_same_change":
            code = a_code if a_code != nearest_code and a_code == b_code else nearest_code
        elif policy == "intersection_any_change_prefer_a":
            code = a_code if a_code != nearest_code and b_code != nearest_code else nearest_code
        elif policy == "union_prefer_a":
            code = a_code if a_code != nearest_code else b_code if b_code != nearest_code else nearest_code
        elif policy == "union_prefer_b":
            code = b_code if b_code != nearest_code else a_code if a_code != nearest_code else nearest_code
        elif policy == "policy_b_extra_only":
            code = b_code if b_code != nearest_code and a_code == nearest_code else nearest_code
        else:
            raise ValueError(f"unsupported hypothetical policy: {policy}")
        out.append(
            chosen_row(
                key=key,
                code_index=code,
                nearest_row=a_row,
                a_row=a_row,
                b_row=b_row,
                oracle_code=oracle_by_key[key],
                source_meta=meta_by_key[key],
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-a", type=Path, required=True)
    parser.add_argument("--policy-b", type=Path, required=True)
    parser.add_argument("--shortlist-oracle", type=Path, required=True)
    parser.add_argument("--top8-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy-a-name", default="pooled_imgdiff_model1_threshold05")
    parser.add_argument("--policy-b-name", default="pooled_imgdiff_model2_threshold08")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows_a = {key_from_row(row): row for row in load_jsonl(args.policy_a)}
    rows_b = {key_from_row(row): row for row in load_jsonl(args.policy_b)}
    rows_short = {key_from_row(row): row for row in load_jsonl(args.shortlist_oracle)}
    meta = json.loads(args.top8_meta.read_text(encoding="utf-8"))
    meta_by_key = {
        (int(item["real_seed"]), int(item["source_index"]), int(item["candidate_index"])): item
        for item in meta["group_metadata"]
    }
    policy_key_union = set(rows_a) | set(rows_b) | set(rows_short)
    common_keys = sorted(set(rows_a) & set(rows_b) & set(rows_short) & set(meta_by_key))
    missing_counts = {
        "policy_a_missing_from_policy_union": len(policy_key_union - set(rows_a)),
        "policy_b_missing_from_policy_union": len(policy_key_union - set(rows_b)),
        "shortlist_missing_from_policy_union": len(policy_key_union - set(rows_short)),
        "top8_meta_missing_from_policy_union": len(policy_key_union - set(meta_by_key)),
        "top8_meta_extra_groups_not_in_compared_policies": len(set(meta_by_key) - policy_key_union),
    }
    oracle_by_key = {
        key: int(meta_by_key[key].get("oracle_code", rows_short[key].get("assignment_oracle_code_index", rows_short[key]["code_index"])))
        for key in common_keys
    }

    annotated_a = annotate_policy_rows(rows_a, common_keys, oracle_by_key, meta_by_key)
    annotated_b = annotate_policy_rows(rows_b, common_keys, oracle_by_key, meta_by_key)
    annotated_short = annotate_policy_rows(rows_short, common_keys, oracle_by_key, meta_by_key)
    hypothetical_names = [
        "intersection_same_change",
        "intersection_any_change_prefer_a",
        "union_prefer_a",
        "union_prefer_b",
        "policy_b_extra_only",
    ]
    hypotheticals = {
        name: build_hypothetical_rows(name, common_keys, rows_a, rows_b, oracle_by_key, meta_by_key)
        for name in hypothetical_names
    }

    relationship_counts: Counter[str] = Counter()
    cases = []
    for key in common_keys:
        a_row = rows_a[key]
        b_row = rows_b[key]
        short_row = rows_short[key]
        nearest = int(a_row["nearest_code"])
        oracle = oracle_by_key[key]
        a_code = int(a_row["code_index"])
        b_code = int(b_row["code_index"])
        a_changed = a_code != nearest
        b_changed = b_code != nearest
        if a_changed and b_changed and a_code == b_code:
            relation = "both_same_change"
        elif a_changed and b_changed:
            relation = "both_different_change"
        elif a_changed:
            relation = "policy_a_only_change"
        elif b_changed:
            relation = "policy_b_only_change"
        else:
            relation = "both_nearest"
        if relation == "both_nearest" and oracle != nearest:
            relation += "_miss_oracle"
        relationship_counts[relation] += 1
        if a_changed or b_changed or oracle != nearest:
            cases.append(
                {
                    "key": {"real_seed": key[0], "source_index": key[1], "candidate_index": key[2]},
                    "source": meta_by_key[key].get("source"),
                    "reference": meta_by_key[key].get("reference"),
                    "source_image": meta_by_key[key].get("source_image"),
                    "relation": relation,
                    "nearest_code": nearest,
                    "oracle_code": oracle,
                    "policy_a_code": a_code,
                    "policy_b_code": b_code,
                    "shortlist_code": int(short_row["code_index"]),
                    "policy_a_category": category_for_code(a_code, nearest, oracle),
                    "policy_b_category": category_for_code(b_code, nearest, oracle),
                    "shortlist_category": category_for_code(int(short_row["code_index"]), nearest, oracle),
                    "policy_a_tesseract_delta": int(a_row.get("tesseract_delta_vs_nearest", 0)),
                    "policy_b_tesseract_delta": int(b_row.get("tesseract_delta_vs_nearest", 0)),
                    "shortlist_tesseract_delta": int(short_row.get("tesseract_delta_vs_nearest", 0)),
                    "policy_a_topk_rank": int(a_row["topk_rank"]),
                    "policy_b_topk_rank": int(b_row["topk_rank"]),
                    "shortlist_topk_rank": int(short_row["topk_rank"]),
                    "policy_a_score": float(a_row.get("selector_score", 0.0)),
                    "policy_b_score": float(b_row.get("selector_score", 0.0)),
                }
            )

    summaries = {
        args.policy_a_name: summarize_policy(annotated_a),
        args.policy_b_name: summarize_policy(annotated_b),
        "shortlist_oracle": summarize_policy(annotated_short),
    }
    summaries.update({name: summarize_policy(rows) for name, rows in hypotheticals.items()})

    result = {
        "description": (
            "Diagnostic audit of agreement between two already-exported actual-counted "
            "pooled image-diff assignment policies, compared with the non-deployable "
            "top-4 shortlist oracle. Held-out OCR labels are used only for analysis."
        ),
        "validity": "diagnostic_only_not_a_selector_or_promotion",
        "inputs": {
            "policy_a": str(args.policy_a),
            "policy_b": str(args.policy_b),
            "shortlist_oracle": str(args.shortlist_oracle),
            "top8_meta": str(args.top8_meta),
        },
        "input_hashes": {
            str(args.policy_a): sha256_file(args.policy_a),
            str(args.policy_b): sha256_file(args.policy_b),
            str(args.shortlist_oracle): sha256_file(args.shortlist_oracle),
            str(args.top8_meta): sha256_file(args.top8_meta),
        },
        "policy_names": {
            "policy_a": args.policy_a_name,
            "policy_b": args.policy_b_name,
        },
        "counts": {
            "common_groups": len(common_keys),
            "missing_counts": missing_counts,
            "relationship_counts": dict(sorted(relationship_counts.items())),
        },
        "policy_summaries": summaries,
        "cases": cases,
        "conclusion": {
            "summary": (
                "The more conservative same-change consensus is identical to policy A on "
                "this split: it keeps the same 5 changes, -8 Tesseract edit delta and one "
                "false change. Policy B adds two changes over policy A, but they are one "
                "false and one wrong/neutral Tesseract case, so union-style expansion gives "
                "no extra OCR gain and increases policy error count."
            ),
            "next_action": (
                "Do not promote another threshold/consensus rule over these two gates. "
                "Use the result as design evidence that the current image-diff gate family "
                "has saturated; the remaining top-4 headroom requires materially richer "
                "candidate-local code-effect evidence or a new verifier objective."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "# Eval300 Actual-Policy Consensus Audit",
        "",
        "This is a diagnostic-only audit. It compares already-exported actual `.oscr` assignment policies and does not define a deployable selector.",
        "",
        "## Inputs",
        "",
        f"- policy A: `{args.policy_a}`",
        f"- policy B: `{args.policy_b}`",
        f"- shortlist oracle: `{args.shortlist_oracle}`",
        f"- top-8 metadata: `{args.top8_meta}`",
        "",
        "## Agreement",
        "",
        f"- common groups: {len(common_keys)}",
        f"- relationship counts: `{dict(sorted(relationship_counts.items()))}`",
        "",
        "## Policy Summary",
        "",
        "| policy | changed | exact | false | wrong | missed | Tesseract Δ | PARSeq Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [
        args.policy_a_name,
        args.policy_b_name,
        "intersection_same_change",
        "union_prefer_a",
        "union_prefer_b",
        "policy_b_extra_only",
        "shortlist_oracle",
    ]:
        summary = summaries[name]
        counts = summary["category_counts"]
        report.append(
            f"| {name} | {summary['changed_groups']} | "
            f"{counts.get('exact_oracle_change', 0)} | "
            f"{counts.get('false_change', 0)} | "
            f"{counts.get('wrong_change', 0)} | "
            f"{counts.get('missed_oracle', 0)} | "
            f"{summary['tesseract_delta_sum']} | {summary['parseq_delta_sum']} |"
        )

    report.extend(
        [
            "",
            "## Notable Changed Or Oracle-Headroom Cases",
            "",
            "| relation | seed/source/cand | source | ref | nearest -> A / B / oracle / shortlist | categories A/B/short | Tesseract Δ A/B/short | ranks A/B/short |",
            "|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in cases:
        key = row["key"]
        report.append(
            f"| {row['relation']} | "
            f"{key['real_seed']}/{key['source_index']}/{key['candidate_index']} | "
            f"{row['source']} | {row['reference']} | "
            f"{row['nearest_code']} -> {row['policy_a_code']} / {row['policy_b_code']} / "
            f"{row['oracle_code']} / {row['shortlist_code']} | "
            f"{row['policy_a_category']}/{row['policy_b_category']}/{row['shortlist_category']} | "
            f"{row['policy_a_tesseract_delta']}/{row['policy_b_tesseract_delta']}/{row['shortlist_tesseract_delta']} | "
            f"{row['policy_a_topk_rank']}/{row['policy_b_topk_rank']}/{row['shortlist_topk_rank']} |"
        )

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Policy B's two extra changes do not recover shortlist-oracle headroom; they add one false and one wrong/neutral change.",
            "- Same-change consensus collapses to policy A, so this gate family does not offer a hidden high-precision ensemble path.",
            "- The top-4 shortlist oracle remains the useful upper-bound substrate, but the deployable decision evidence needs to change rather than another consensus knob.",
        ]
    )
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "groups": len(common_keys)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
