#!/usr/bin/env python3
"""Two-stage bottleneck audit for top-8 assignment policies.

This diagnostic separates candidate ranking from no-op/change acceptance. It
uses held-out OCR deltas only as labels/evaluation, never as deployable features
or as a promoted selector.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


SCORE_KEYS = [
    "score",
    "selector_score",
    "candidate_probability",
    "group_probability",
    "group_change_prob",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def group_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    seed = int(row.get("seed", row.get("original_seed", 0)))
    return (
        int(row.get("real_seed", row.get("original_seed", seed))),
        seed,
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def candidate_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (*group_key(row), int(row["code_index"]))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return None
    order = np.argsort(-scores)
    hits = 0
    total = 0.0
    for rank, label in enumerate(labels[order], start=1):
        if label:
            hits += 1
            total += hits / rank
    return float(total / pos)


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


def load_table(path: Path) -> tuple[dict[tuple[int, int, int, int], list[dict[str, Any]]], dict[tuple[int, int, int, int, int], dict[str, Any]]]:
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_candidate: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        grouped[group_key(row)].append(row)
        by_candidate[candidate_key(row)] = row
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row.get("topk_rank", 999)), int(row["code_index"])))
    return dict(grouped), by_candidate


def row_for_code(rows: list[dict[str, Any]], code: int) -> dict[str, Any]:
    for row in rows:
        if int(row["code_index"]) == int(code):
            return row
    raise KeyError(code)


def nearest_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if int(row.get("is_nearest", 0)):
            return row
    return min(rows, key=lambda row: int(row.get("topk_rank", 999)))


def oracle_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if int(row.get("label_assignment_oracle_choice", 0)):
            return row
    code = int(rows[0]["assignment_oracle_code_index"])
    return row_for_code(rows, code)


def evaluate_selection(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    selected_codes: dict[tuple[int, int, int, int], int],
) -> dict[str, Any]:
    counts = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    t_sum = 0
    p_sum = 0
    changed_deltas: list[float] = []
    audits: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        first = rows[0]
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        nearest_code = int(nearest["code_index"])
        oracle_code = int(oracle["code_index"])
        selected_code = int(selected_codes.get(key, nearest_code))
        selected = row_for_code(rows, selected_code)
        changed = selected_code != nearest_code
        oracle_changes = oracle_code != nearest_code
        t_delta = int(selected.get("tesseract_delta_vs_nearest", 0)) if changed else 0
        p_delta = int(selected.get("parseq_delta_vs_nearest", 0)) if changed else 0
        status = "exact"
        if selected_code == oracle_code:
            counts["exact"] += 1
            if changed:
                counts["exact_changed_groups"] += 1
        elif selected_code == nearest_code and oracle_changes:
            status = "missed_oracle"
            counts["missed_oracle"] += 1
        elif selected_code != nearest_code and not oracle_changes:
            status = "false_change"
            counts["false_change"] += 1
        else:
            status = "wrong_change"
            counts["wrong_change"] += 1
        counts["groups"] += 1
        counts["oracle_change_groups"] += int(oracle_changes)
        counts["changed_groups"] += int(changed)
        counts["tesseract_worse_groups"] += int(t_delta > 0)
        counts["parseq_worse_groups"] += int(p_delta > 0)
        t_sum += t_delta
        p_sum += p_delta
        if changed:
            changed_deltas.append(float(t_delta))
        source = str(first.get("source", "unknown"))
        by_source[source][status] += 1
        by_source[source]["groups"] += 1
        by_source[source]["changed_groups"] += int(changed)
        by_source[source]["tesseract_delta_vs_nearest"] += t_delta
        if status != "exact" and len(audits) < 40:
            audits.append(
                {
                    "group_key": list(key),
                    "source": source,
                    "reference": first.get("reference"),
                    "nearest_code": nearest_code,
                    "oracle_code": oracle_code,
                    "selected_code": selected_code,
                    "status": status,
                    "selected_topk_rank": int(selected.get("topk_rank", -1)),
                    "oracle_topk_rank": int(oracle.get("topk_rank", -1)),
                    "selected_tesseract_delta_vs_nearest": t_delta,
                    "selected_parseq_delta_vs_nearest": p_delta,
                }
            )
    out = {key: int(value) for key, value in counts.items()}
    out["tesseract_delta_vs_nearest"] = int(t_sum)
    out["parseq_delta_vs_nearest"] = int(p_sum)
    out["changed_tesseract_delta_stats"] = summarize_values(changed_deltas)
    out["by_source"] = {name: {k: int(v) for k, v in counter.items()} for name, counter in sorted(by_source.items())}
    out["audits"] = audits
    return out


def split_groups(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]]
) -> dict[str, dict[tuple[int, int, int, int], list[dict[str, Any]]]]:
    by_partition: dict[str, dict[tuple[int, int, int, int], list[dict[str, Any]]]] = defaultdict(dict)
    for key, rows in groups.items():
        partition = str(rows[0].get("assignment_partition", rows[0].get("split", "unknown")))
        by_partition[partition][key] = rows
    by_partition["all"] = groups
    return dict(sorted(by_partition.items()))


def rankcap_policy(groups: dict[tuple[int, int, int, int], list[dict[str, Any]]], cap: int) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        if int(oracle["code_index"]) != int(nearest["code_index"]) and int(oracle.get("topk_rank", 999)) <= cap:
            selected[key] = int(oracle["code_index"])
        else:
            selected[key] = int(nearest["code_index"])
    return selected


def best_budget_oracle_policy(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]], budget: int
) -> dict[tuple[int, int, int, int], int]:
    changed: list[tuple[int, int, tuple[int, int, int, int], int]] = []
    selected = {key: int(nearest_row(rows)["code_index"]) for key, rows in groups.items()}
    for key, rows in groups.items():
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        changed.append(
            (
                int(oracle.get("tesseract_delta_vs_nearest", 0)),
                int(oracle.get("topk_rank", 999)),
                key,
                int(oracle["code_index"]),
            )
        )
    for _, _, key, code in sorted(changed)[:budget]:
        selected[key] = code
    return selected


def discover_score_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            matches = glob.glob(str(ROOT / pattern), recursive=True)
        for match in matches:
            path = Path(match)
            if path.is_file():
                paths.add(path.resolve())
    return sorted(paths)


def score_name(path: Path, used: set[str]) -> str:
    name = path.stem
    for suffix in ["_2026_06_25", "_2026_06_26", "_scores", "_score"]:
        name = name.replace(suffix, "")
    name = name.replace("eval300_", "").replace("assignment_", "").replace("top8_", "")
    name = name[:96]
    base = name
    index = 2
    while name in used:
        name = f"{base}_{index}"
        index += 1
    used.add(name)
    return name


def load_score_source(
    path: Path,
    table_candidates: dict[tuple[int, int, int, int, int], dict[str, Any]],
) -> tuple[str | None, dict[tuple[int, int, int, int, int], float], dict[str, Any]]:
    accum: dict[tuple[int, int, int, int, int], list[float]] = defaultdict(list)
    rows_read = 0
    rows_with_score = 0
    malformed = 0
    score_key_counts = Counter()
    for row in read_jsonl(path):
        rows_read += 1
        try:
            key = candidate_key(row)
        except (KeyError, ValueError, TypeError):
            malformed += 1
            continue
        if key not in table_candidates:
            continue
        value = None
        used_key = None
        for key_name in SCORE_KEYS:
            value = safe_float(row.get(key_name))
            if value is not None:
                used_key = key_name
                break
        if value is None:
            malformed += 1
            continue
        accum[key].append(value)
        rows_with_score += 1
        score_key_counts[str(used_key)] += 1
    scores = {key: float(np.mean(values)) for key, values in accum.items()}
    matched_groups = {key[:4] for key in scores}
    meta = {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha256_file(path),
        "rows_read": int(rows_read),
        "rows_with_matching_score": int(rows_with_score),
        "matched_candidates": int(len(scores)),
        "matched_groups": int(len(matched_groups)),
        "malformed_or_unscored_rows": int(malformed),
        "score_key_counts": {key: int(value) for key, value in score_key_counts.items()},
    }
    if len(scores) == 0:
        return None, {}, meta
    return path.name, scores, meta


def builtin_sources(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]]
) -> dict[str, tuple[dict[tuple[int, int, int, int, int], float], dict[str, Any]]]:
    out: dict[str, tuple[dict[tuple[int, int, int, int, int], float], dict[str, Any]]] = {}
    for name in ["table_topk_rate_prior", "table_topk_only", "table_relative_error_only"]:
        out[name] = ({}, {"path": "derived_from_candidate_table", "sha256": None})
    for rows in groups.values():
        for row in rows:
            key = candidate_key(row)
            topk = float(row.get("topk_rank", 999.0))
            rel = float(row.get("assignment_relative_error", 0.0))
            out["table_topk_rate_prior"][0][key] = -topk - 1.0e-6 * rel
            out["table_topk_only"][0][key] = -topk
            out["table_relative_error_only"][0][key] = -rel - 1.0e-3 * topk
    return out


def best_scored_nonnearest(
    rows: list[dict[str, Any]], scores: dict[tuple[int, int, int, int, int], float]
) -> tuple[dict[str, Any] | None, float | None]:
    ranked: list[tuple[float, int, int, dict[str, Any]]] = []
    for row in rows:
        if int(row.get("is_nearest", 0)):
            continue
        score = scores.get(candidate_key(row))
        if score is None:
            continue
        ranked.append((score, -int(row.get("topk_rank", 999)), -int(row["code_index"]), row))
    if not ranked:
        return None, None
    score, _, _, row = max(ranked, key=lambda item: (item[0], item[1], item[2]))
    return row, float(score)


def oracle_score_rank(
    rows: list[dict[str, Any]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    nonnearest_only: bool,
) -> int | None:
    oracle = oracle_row(rows)
    if nonnearest_only and int(oracle.get("is_nearest", 0)):
        return None
    if candidate_key(oracle) not in scores:
        return None
    ranked: list[tuple[float, int, int, int]] = []
    for row in rows:
        if nonnearest_only and int(row.get("is_nearest", 0)):
            continue
        score = scores.get(candidate_key(row))
        if score is None:
            continue
        ranked.append((float(score), -int(row.get("topk_rank", 999)), -int(row["code_index"]), int(row["code_index"])))
    ranked.sort(reverse=True)
    for index, (_, _, _, code) in enumerate(ranked, start=1):
        if code == int(oracle["code_index"]):
            return index
    return None


def score_rankcap_policy(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
    cap: int,
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        selected[key] = int(nearest["code_index"])
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        rank = oracle_score_rank(rows, scores, nonnearest_only=True)
        if rank is not None and rank <= cap:
            selected[key] = int(oracle["code_index"])
    return selected


def oracle_acceptor_policy(
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
) -> dict[tuple[int, int, int, int], int]:
    selected: dict[tuple[int, int, int, int], int] = {}
    for key, rows in groups.items():
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        selected[key] = int(nearest["code_index"])
        if int(oracle["code_index"]) == int(nearest["code_index"]):
            continue
        best, _ = best_scored_nonnearest(rows, scores)
        if best is not None:
            selected[key] = int(best["code_index"])
    return selected


def summarize_score_source(
    groups_by_partition: dict[str, dict[tuple[int, int, int, int], list[dict[str, Any]]]],
    scores: dict[tuple[int, int, int, int, int], float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for partition, groups in groups_by_partition.items():
        rank_counts = Counter()
        group_scores: list[float] = []
        group_labels: list[int] = []
        top_rows: list[dict[str, Any]] = []
        for rows in groups.values():
            nearest = nearest_row(rows)
            oracle = oracle_row(rows)
            oracle_changes = int(oracle["code_index"]) != int(nearest["code_index"])
            rank = oracle_score_rank(rows, scores, nonnearest_only=True)
            if oracle_changes:
                rank_counts["oracle_change_groups"] += 1
                if rank is None:
                    rank_counts["oracle_missing_score"] += 1
                else:
                    for cap in [1, 2, 4, 8]:
                        rank_counts[f"oracle_rank_le{cap}"] += int(rank <= cap)
            best, best_score = best_scored_nonnearest(rows, scores)
            if best_score is not None:
                group_scores.append(best_score)
                group_labels.append(int(oracle_changes))
                if len(top_rows) < 120:
                    top_rows.append(
                        {
                            "source": rows[0].get("source"),
                            "oracle_changes": bool(oracle_changes),
                            "top_code": int(best["code_index"]) if best is not None else None,
                            "oracle_code": int(oracle["code_index"]),
                            "nearest_code": int(nearest["code_index"]),
                            "score": best_score,
                            "top_tesseract_delta_vs_nearest": int(best.get("tesseract_delta_vs_nearest", 0)) if best else 0,
                            "top_parseq_delta_vs_nearest": int(best.get("parseq_delta_vs_nearest", 0)) if best else 0,
                        }
                    )
        oracle_acceptor = evaluate_selection(groups, oracle_acceptor_policy(groups, scores))
        rankcap = {
            f"score_rankcap_le{cap}": evaluate_selection(groups, score_rankcap_policy(groups, scores, cap))
            for cap in [1, 2, 4, 8]
        }
        y = np.asarray(group_labels, dtype=np.int32)
        s = np.asarray(group_scores, dtype=np.float64)
        separation = {
            "groups_with_nonnearest_scores": int(len(group_scores)),
            "positive_oracle_change_groups": int(y.sum()) if len(y) else 0,
            "max_nonnearest_score_auc_for_change": roc_auc(s, y) if len(y) else None,
            "max_nonnearest_score_ap_for_change": average_precision(s, y) if len(y) else None,
            "max_nonnearest_score_positive_stats": summarize_values([float(v) for v, label in zip(s, y) if label]),
            "max_nonnearest_score_noheadroom_stats": summarize_values([float(v) for v, label in zip(s, y) if not label]),
        }
        out[partition] = {
            "rank_counts": {key: int(value) for key, value in rank_counts.items()},
            "oracle_acceptor_top1_candidate": oracle_acceptor,
            "score_rankcap_oracle_choice": rankcap,
            "group_change_separation": separation,
        }
    return out


def source_modulo_slices(
    val_groups: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    scores: dict[tuple[int, int, int, int, int], float],
    *,
    modulo: int,
    remainder: int,
) -> dict[str, Any]:
    heldout = {key: rows for key, rows in val_groups.items() if int(rows[0]["source_index"]) % modulo == remainder}
    in_train = {key: rows for key, rows in val_groups.items() if int(rows[0]["source_index"]) % modulo != remainder}
    return summarize_score_source(
        {
            f"clean_val_source_mod{modulo}_heldout_r{remainder}": heldout,
            f"clean_val_source_mod{modulo}_in_train_not_r{remainder}": in_train,
        },
        scores,
    )


def baseline_upper_bounds(groups_by_partition: dict[str, dict[tuple[int, int, int, int], list[dict[str, Any]]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for partition, groups in groups_by_partition.items():
        out[partition] = {
            f"topk_rankcap_le{cap}": evaluate_selection(groups, rankcap_policy(groups, cap))
            for cap in [1, 2, 4, 8]
        }
        out[partition]["perfect_candidate_budget5"] = evaluate_selection(groups, best_budget_oracle_policy(groups, 5))
        out[partition]["perfect_candidate_budget17"] = evaluate_selection(groups, best_budget_oracle_policy(groups, 17))
        out[partition]["nearest_no_enhancement"] = evaluate_selection(
            groups, {key: int(nearest_row(rows)["code_index"]) for key, rows in groups.items()}
        )
        out[partition]["full_top8_oracle"] = evaluate_selection(
            groups, {key: int(oracle_row(rows)["code_index"]) for key, rows in groups.items()}
        )
    return out


def compact_policy_metric(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics.get('exact_changed_groups', 0)}/"
        f"{metrics.get('oracle_change_groups', 0)} exact, "
        f"false {metrics.get('false_change', 0)}, wrong {metrics.get('wrong_change', 0)}, "
        f"miss {metrics.get('missed_oracle', 0)}, T {metrics.get('tesseract_delta_vs_nearest', 0)}, "
        f"P {metrics.get('parseq_delta_vs_nearest', 0)}"
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    val_bounds = result["upper_bounds"].get("val", {})
    lines = [
        f"# {result['experiment_id']}",
        "",
        "Diagnostic-only two-stage bottleneck audit for the top-8 assignment substrate.",
        "",
        "## Upper Bounds",
        "",
        "| policy | val summary |",
        "|---|---|",
    ]
    for name in [
        "nearest_no_enhancement",
        "perfect_candidate_budget5",
        "topk_rankcap_le1",
        "topk_rankcap_le2",
        "topk_rankcap_le4",
        "topk_rankcap_le8",
        "full_top8_oracle",
    ]:
        if name in val_bounds:
            lines.append(f"| {name} | {compact_policy_metric(val_bounds[name])} |")
    lines.extend(["", "## Score-Source Candidate Bottleneck", ""])
    lines.append(
        "| score source | val rank1 | val rank<=4 | oracle-acceptor top1 | max-nonnearest AUC/AP |"
    )
    lines.append("|---|---:|---:|---|---:|")
    rows = []
    for name, summary in result["score_sources"].items():
        val = summary.get("partitions", {}).get("val")
        if not val:
            continue
        rank_counts = val["rank_counts"]
        top1 = rank_counts.get("oracle_rank_le1", 0)
        top4 = rank_counts.get("oracle_rank_le4", 0)
        sep = val["group_change_separation"]
        auc = sep.get("max_nonnearest_score_auc_for_change")
        ap = sep.get("max_nonnearest_score_ap_for_change")
        metric = val["oracle_acceptor_top1_candidate"]
        rows.append((top1, top4, metric.get("tesseract_delta_vs_nearest", 0), name, metric, auc, ap))
    for top1, top4, _, name, metric, auc, ap in sorted(rows, key=lambda item: (-item[0], -item[1], item[3]))[:30]:
        auc_ap = "n/a" if auc is None else f"{auc:.3f}/{ap:.3f}"
        lines.append(f"| {name} | {top1} | {top4} | {compact_policy_metric(metric)} | {auc_ap} |")
    slice_rows = []
    for name, summary in result["score_sources"].items():
        slices = summary.get("provenance_slices")
        if not isinstance(slices, dict):
            continue
        for slice_name, slice_summary in slices.items():
            rank_counts = slice_summary["rank_counts"]
            metric = slice_summary["oracle_acceptor_top1_candidate"]
            sep = slice_summary["group_change_separation"]
            slice_rows.append(
                (
                    name,
                    slice_name,
                    metric.get("groups", 0),
                    rank_counts.get("oracle_change_groups", 0),
                    rank_counts.get("oracle_rank_le1", 0),
                    rank_counts.get("oracle_rank_le4", 0),
                    metric,
                    sep.get("max_nonnearest_score_auc_for_change"),
                    sep.get("max_nonnearest_score_ap_for_change"),
                )
            )
    if slice_rows:
        lines.extend(["", "## Provenance Slices", ""])
        lines.append(
            "| score source | slice | groups | oracle changes | rank1 | rank<=4 | oracle-acceptor top1 | AUC/AP |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|---:|")
        for name, slice_name, groups, changes, rank1, rank4, metric, auc, ap in slice_rows:
            auc_ap = "n/a" if auc is None else f"{auc:.3f}/{ap:.3f}"
            lines.append(
                f"| {name} | {slice_name} | {groups} | {changes} | {rank1} | {rank4} | "
                f"{compact_policy_metric(metric)} | {auc_ap} |"
            )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, result: dict[str, Any], output: Path, report: Path) -> None:
    input_lines = []
    for name, meta in result["inputs"].items():
        if isinstance(meta, dict) and meta.get("path") and meta.get("sha256"):
            input_lines.extend([f"  {name}:", f"    path: {meta['path']}", f"    sha256: {meta['sha256']}"])
    for name, summary in result["score_sources"].items():
        meta = summary.get("meta", {})
        if meta.get("path") and meta.get("sha256"):
            safe = name.replace(":", "_").replace("/", "_")
            input_lines.extend([f"  score_{safe}:", f"    path: {meta['path']}", f"    sha256: {meta['sha256']}"])
    manifest = [
        f"experiment_id: {result['experiment_id']}",
        "hypothesis_id: H4-top8-two-stage-bottleneck",
        "status: completed",
        f"code_commit: {result['code_commit']}",
        f"command: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}",
        "inputs:",
        *input_lines,
        "outputs:",
        "  result_json:",
        f"    path: {output}",
        f"    sha256: {sha256_file(output)}",
        "  report:",
        f"    path: {report}",
        f"    sha256: {sha256_file(report)}",
        "scripts:",
        "  analyzer:",
        f"    path: scripts/analyze_top8_two_stage_bottleneck.py",
        f"    sha256: {sha256_file(Path(__file__))}",
        "conclusion: diagnostic bottleneck audit; no selector promoted",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(manifest) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--score-glob", action="append", default=[])
    parser.add_argument("--score-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups, by_candidate = load_table(args.table)
    groups_by_partition = split_groups(groups)

    score_sources: dict[str, tuple[dict[tuple[int, int, int, int, int], float], dict[str, Any]]] = {}
    score_sources.update(builtin_sources(groups))
    paths = discover_score_paths(args.score_glob) + [path.resolve() for path in args.score_file]
    used_names = set(score_sources)
    skipped: list[dict[str, Any]] = []
    for path in paths:
        if "shortlist_oracle" in path.name:
            skipped.append(
                {
                    "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                    "sha256": sha256_file(path),
                    "reason": "oracle_policy_file_not_a_deployable_score_source",
                }
            )
            continue
        _, scores, meta = load_score_source(path, by_candidate)
        if not scores:
            skipped.append(meta)
            continue
        name = score_name(path, used_names)
        score_sources[name] = (scores, meta)

    source_results: dict[str, Any] = {}
    for name, (scores, meta) in sorted(score_sources.items()):
        partitions = summarize_score_source(groups_by_partition, scores)
        source_results[name] = {
            "meta": meta,
            "partitions": partitions,
        }
        if "source_mod5_r4" in name and "val" in groups_by_partition:
            source_results[name]["provenance_slices"] = source_modulo_slices(
                groups_by_partition["val"], scores, modulo=5, remainder=4
            )

    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_two_stage_bottleneck_not_promoted",
        "code_commit": git_commit(),
        "inputs": {
            "table": {
                "path": str(args.table),
                "sha256": sha256_file(args.table),
            }
        },
        "config": {
            "score_glob": args.score_glob,
            "explicit_score_files": [str(path) for path in args.score_file],
            "score_keys": SCORE_KEYS,
        },
        "data_summary": {
            "groups": int(len(groups)),
            "candidates": int(len(by_candidate)),
            "partitions": {name: int(len(part_groups)) for name, part_groups in groups_by_partition.items()},
            "loaded_score_sources": int(len(source_results)),
            "skipped_score_files": skipped,
        },
        "upper_bounds": baseline_upper_bounds(groups_by_partition),
        "score_sources": source_results,
        "hashes": {
            "script": sha256_file(Path(__file__)),
        },
        "interpretation": (
            "The audit is diagnostic-only. It asks whether existing score families are primarily limited by "
            "candidate top-1 ranking or by no-op/change calibration. Any positive score-rankcap number is an "
            "upper control with an oracle chooser inside the score shortlist; it is not a deployable policy. "
            "The source-modulo provenance slice is especially important: source_mod5_r4 scores are useful as a "
            "high-recall candidate-set hint, but clean-val rows outside the held-out source-modulo remainder are "
            "not valid held-out evidence for that score source. "
            "Promotion still requires an actual counted `.oscr` export and OCR comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, result, args.output, args.report)
    val_bounds = result["upper_bounds"].get("val", {})
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "loaded_score_sources": len(source_results),
                "val_upper_bounds": {
                    name: {
                        key: value
                        for key, value in metrics.items()
                        if key
                        in [
                            "exact_changed_groups",
                            "false_change",
                            "wrong_change",
                            "missed_oracle",
                            "tesseract_delta_vs_nearest",
                            "parseq_delta_vs_nearest",
                        ]
                    }
                    for name, metrics in val_bounds.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
