#!/usr/bin/env python3
"""Audit OCR/text teacher-consensus signals on the Eval300 top-8 table."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts  # noqa: E402


Key = tuple[int, int, int, int, int]
GroupKey = tuple[int, str, int, int]


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


def row_key(row: dict[str, Any]) -> Key:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        int(row["seed"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
        int(row["code_index"]),
    )


def group_key(row: dict[str, Any]) -> GroupKey:
    return (
        int(row.get("real_seed", row.get("seed", 0))),
        str(row["assignment_partition"]),
        int(row["source_index"]),
        int(row["candidate_index"]),
    )


def distance(row: dict[str, Any], profile: str) -> int:
    return character_counts(str(row["reference"]), str(row["prediction"]), profile).distance


def nearest_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in rows if int(row["code_index"]) == int(row["nearest_code"]))


def oracle_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if int(row.get("label_assignment_oracle_choice", 0))), nearest_row(rows))


def classify(selected: dict[str, Any], nearest: dict[str, Any], oracle: dict[str, Any]) -> str:
    selected_code = int(selected["code_index"])
    nearest_code = int(nearest["code_index"])
    oracle_code = int(oracle["code_index"])
    if selected_code == oracle_code:
        return "exact" if oracle_code != nearest_code else "correct_nearest"
    if selected_code == nearest_code and oracle_code != nearest_code:
        return "missed_oracle"
    if selected_code != nearest_code and oracle_code == nearest_code:
        return "false_change"
    return "wrong_change"


def enrich_rows(
    table_rows: list[dict[str, Any]],
    crnn_rows: dict[Key, dict[str, Any]],
    abinet_rows: dict[Key, dict[str, Any]],
    profile: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    missing = {"crnn": 0, "abinet": 0}
    for row in table_rows:
        key = row_key(row)
        crnn = crnn_rows.get(key)
        abinet = abinet_rows.get(key)
        if crnn is None:
            missing["crnn"] += 1
        if abinet is None:
            missing["abinet"] += 1
        if crnn is None or abinet is None:
            continue
        item = dict(row)
        item["parseq_distance_teacher"] = int(row["parseq_distance"])
        item["crnn_distance_teacher"] = distance(crnn, profile)
        item["abinet_distance_teacher"] = distance(abinet, profile)
        item["crnn_prediction_teacher"] = crnn.get("prediction", "")
        item["abinet_prediction_teacher"] = abinet.get("prediction", "")
        item["crnn_confidence_teacher"] = float(crnn.get("confidence", 0.0))
        item["abinet_confidence_teacher"] = float(abinet.get("confidence", 0.0))
        item["crnn_loss_mean_teacher"] = float(crnn.get("teacher_loss_mean", 0.0))
        item["abinet_loss_mean_teacher"] = float(abinet.get("teacher_loss_mean", 0.0))
        item["crnn_loss_sum_teacher"] = float(crnn.get("teacher_loss_sum", 0.0))
        item["abinet_loss_sum_teacher"] = float(abinet.get("teacher_loss_sum", 0.0))
        enriched.append(item)
    if any(missing.values()):
        raise SystemExit(f"missing teacher rows: {missing}")
    return enriched


def add_group_deltas(groups: dict[GroupKey, list[dict[str, Any]]]) -> None:
    for rows in groups.values():
        nearest = nearest_row(rows)
        nearest_values = {
            "parseq": int(nearest["parseq_distance_teacher"]),
            "crnn": int(nearest["crnn_distance_teacher"]),
            "abinet": int(nearest["abinet_distance_teacher"]),
            "crnn_loss_mean": float(nearest["crnn_loss_mean_teacher"]),
            "abinet_loss_mean": float(nearest["abinet_loss_mean_teacher"]),
            "crnn_loss_sum": float(nearest["crnn_loss_sum_teacher"]),
            "abinet_loss_sum": float(nearest["abinet_loss_sum_teacher"]),
        }
        for row in rows:
            row["teacher_parseq_delta"] = int(row["parseq_distance_teacher"]) - nearest_values["parseq"]
            row["teacher_crnn_delta"] = int(row["crnn_distance_teacher"]) - nearest_values["crnn"]
            row["teacher_abinet_delta"] = int(row["abinet_distance_teacher"]) - nearest_values["abinet"]
            row["teacher_string_delta_sum"] = (
                int(row["teacher_parseq_delta"]) + int(row["teacher_crnn_delta"]) + int(row["teacher_abinet_delta"])
            )
            row["teacher_string_improve_count"] = int(row["teacher_parseq_delta"] < 0) + int(
                row["teacher_crnn_delta"] < 0
            ) + int(row["teacher_abinet_delta"] < 0)
            row["teacher_string_worsen_count"] = int(row["teacher_parseq_delta"] > 0) + int(
                row["teacher_crnn_delta"] > 0
            ) + int(row["teacher_abinet_delta"] > 0)
            row["teacher_crnn_loss_mean_delta"] = float(row["crnn_loss_mean_teacher"]) - nearest_values["crnn_loss_mean"]
            row["teacher_abinet_loss_mean_delta"] = (
                float(row["abinet_loss_mean_teacher"]) - nearest_values["abinet_loss_mean"]
            )
            row["teacher_loss_mean_delta_sum"] = (
                float(row["teacher_crnn_loss_mean_delta"]) + float(row["teacher_abinet_loss_mean_delta"])
            )
            row["teacher_crnn_loss_sum_delta"] = float(row["crnn_loss_sum_teacher"]) - nearest_values["crnn_loss_sum"]
            row["teacher_abinet_loss_sum_delta"] = (
                float(row["abinet_loss_sum_teacher"]) - nearest_values["abinet_loss_sum"]
            )
            row["teacher_loss_sum_delta_sum"] = (
                float(row["teacher_crnn_loss_sum_delta"]) + float(row["teacher_abinet_loss_sum_delta"])
            )


def teacher_score(row: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        -float(row["teacher_string_delta_sum"]),
        -float(row["teacher_loss_mean_delta_sum"]),
        int(row["teacher_string_improve_count"]),
        -int(row["topk_rank"]),
    )


def select_teacher_best(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    nearest = nearest_row(rows)
    candidates = [row for row in rows if int(row["code_index"]) != int(row["nearest_code"]) and predicate(row)]
    if not candidates:
        return nearest
    return max(candidates, key=teacher_score)


def policy_predicates() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "any_string_improve_all_string_no_worse": lambda row: int(row["teacher_string_improve_count"]) >= 1
        and int(row["teacher_string_worsen_count"]) == 0,
        "two_string_improve_all_string_no_worse": lambda row: int(row["teacher_string_improve_count"]) >= 2
        and int(row["teacher_string_worsen_count"]) == 0,
        "parseq_improve_crnn_abinet_no_worse": lambda row: int(row["teacher_parseq_delta"]) < 0
        and int(row["teacher_crnn_delta"]) <= 0
        and int(row["teacher_abinet_delta"]) <= 0,
        "string_sum_improve_all_string_no_worse": lambda row: int(row["teacher_string_delta_sum"]) < 0
        and int(row["teacher_string_worsen_count"]) == 0,
        "string_sum_improve_loss_mean_no_worse": lambda row: int(row["teacher_string_delta_sum"]) < 0
        and float(row["teacher_crnn_loss_mean_delta"]) <= 0.0
        and float(row["teacher_abinet_loss_mean_delta"]) <= 0.0,
        "any_string_improve_all_string_and_loss_no_worse": lambda row: int(row["teacher_string_improve_count"]) >= 1
        and int(row["teacher_string_worsen_count"]) == 0
        and float(row["teacher_crnn_loss_mean_delta"]) <= 0.0
        and float(row["teacher_abinet_loss_mean_delta"]) <= 0.0,
        "teacher_string_sum_argmax_if_improves": lambda row: int(row["teacher_string_delta_sum"]) < 0,
    }


def init_metrics() -> dict[str, Any]:
    return {
        "groups": 0,
        "oracle_change_groups": 0,
        "changed_groups": 0,
        "exact": 0,
        "correct_nearest": 0,
        "exact_changed_groups": 0,
        "false_change": 0,
        "wrong_change": 0,
        "missed_oracle": 0,
        "tesseract_delta_vs_nearest": 0,
        "parseq_delta_vs_nearest": 0,
        "teacher_parseq_delta": 0,
        "teacher_crnn_delta": 0,
        "teacher_abinet_delta": 0,
        "teacher_loss_mean_delta_sum": 0.0,
        "tesseract_worse_groups": 0,
        "parseq_worse_groups": 0,
    }


def update_metrics(metrics: dict[str, Any], selected: dict[str, Any], nearest: dict[str, Any], oracle: dict[str, Any]) -> None:
    status = classify(selected, nearest, oracle)
    changed = int(selected["code_index"]) != int(nearest["code_index"])
    oracle_changed = int(oracle["code_index"]) != int(nearest["code_index"])
    t_delta = int(selected["tesseract_delta_vs_nearest"]) if changed else 0
    p_delta = int(selected["parseq_delta_vs_nearest"]) if changed else 0
    metrics["groups"] += 1
    metrics["oracle_change_groups"] += int(oracle_changed)
    metrics["changed_groups"] += int(changed)
    metrics[status] += 1
    metrics["exact_changed_groups"] += int(status == "exact")
    metrics["tesseract_delta_vs_nearest"] += t_delta
    metrics["parseq_delta_vs_nearest"] += p_delta
    metrics["teacher_parseq_delta"] += int(selected["teacher_parseq_delta"]) if changed else 0
    metrics["teacher_crnn_delta"] += int(selected["teacher_crnn_delta"]) if changed else 0
    metrics["teacher_abinet_delta"] += int(selected["teacher_abinet_delta"]) if changed else 0
    metrics["teacher_loss_mean_delta_sum"] += float(selected["teacher_loss_mean_delta_sum"]) if changed else 0.0
    metrics["tesseract_worse_groups"] += int(t_delta > 0)
    metrics["parseq_worse_groups"] += int(p_delta > 0)


def evaluate_policies(groups: dict[GroupKey, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicates = policy_predicates()
    policy_names = ["nearest", "heldout_oracle"] + list(predicates)
    metrics = {
        partition: {policy: init_metrics() for policy in policy_names}
        for partition in ["train", "val", "all"]
    }
    audits: list[dict[str, Any]] = []

    for key, rows in sorted(groups.items()):
        nearest = nearest_row(rows)
        oracle = oracle_row(rows)
        selections = {
            "nearest": nearest,
            "heldout_oracle": oracle,
        }
        for name, predicate in predicates.items():
            selections[name] = select_teacher_best(rows, predicate)
        partition = key[1]
        for bucket in [partition, "all"]:
            for name, selected in selections.items():
                update_metrics(metrics[bucket][name], selected, nearest, oracle)
        if partition == "val":
            record = {
                "key": {
                    "real_seed": key[0],
                    "partition": key[1],
                    "source_index": key[2],
                    "candidate_index": key[3],
                },
                "source": nearest.get("source"),
                "reference": nearest.get("reference"),
                "nearest_code": int(nearest["code_index"]),
                "oracle_code": int(oracle["code_index"]),
                "oracle_topk_rank": int(oracle["topk_rank"]),
                "oracle_tesseract_delta": int(oracle["tesseract_delta_vs_nearest"]),
                "oracle_parseq_delta": int(oracle["parseq_delta_vs_nearest"]),
                "oracle_teacher_string_delta_sum": int(oracle["teacher_string_delta_sum"]),
                "oracle_teacher_string_improve_count": int(oracle["teacher_string_improve_count"]),
                "oracle_teacher_string_worsen_count": int(oracle["teacher_string_worsen_count"]),
                "oracle_teacher_loss_mean_delta_sum": float(oracle["teacher_loss_mean_delta_sum"]),
                "policy_selection": {},
            }
            for name, selected in selections.items():
                record["policy_selection"][name] = {
                    "code_index": int(selected["code_index"]),
                    "topk_rank": int(selected["topk_rank"]),
                    "status": classify(selected, nearest, oracle),
                    "tesseract_delta_vs_nearest": int(selected["tesseract_delta_vs_nearest"])
                    if int(selected["code_index"]) != int(nearest["code_index"])
                    else 0,
                    "parseq_delta_vs_nearest": int(selected["parseq_delta_vs_nearest"])
                    if int(selected["code_index"]) != int(nearest["code_index"])
                    else 0,
                    "teacher_string_delta_sum": int(selected["teacher_string_delta_sum"]),
                    "teacher_string_improve_count": int(selected["teacher_string_improve_count"]),
                    "teacher_string_worsen_count": int(selected["teacher_string_worsen_count"]),
                    "teacher_loss_mean_delta_sum": float(selected["teacher_loss_mean_delta_sum"]),
                }
            audits.append(record)
    return metrics, audits


def oracle_filter_coverage(groups: dict[GroupKey, list[dict[str, Any]]]) -> dict[str, Any]:
    predicates = policy_predicates()
    out: dict[str, Any] = {}
    for name, predicate in predicates.items():
        by_partition = {}
        for partition in ["train", "val", "all"]:
            counts = Counter()
            for key, rows in groups.items():
                if partition != "all" and key[1] != partition:
                    continue
                nearest = nearest_row(rows)
                oracle = oracle_row(rows)
                oracle_changed = int(oracle["code_index"]) != int(nearest["code_index"])
                if not oracle_changed:
                    counts["no_oracle_change_groups"] += 1
                    counts["no_oracle_groups_with_passing_candidate"] += int(
                        any(
                            int(row["code_index"]) != int(row["nearest_code"]) and predicate(row)
                            for row in rows
                        )
                    )
                    continue
                counts["oracle_change_groups"] += 1
                counts["oracle_row_passes"] += int(predicate(oracle))
                counts["any_passing_candidate"] += int(
                    any(int(row["code_index"]) != int(row["nearest_code"]) and predicate(row) for row in rows)
                )
            by_partition[partition] = dict(counts)
        out[name] = by_partition
    return out


def source_breakdown(audits: list[dict[str, Any]], policy: str) -> dict[str, int]:
    counter = Counter()
    for row in audits:
        status = row["policy_selection"][policy]["status"]
        counter[(str(row.get("source", "")), str(status))] += 1
    return {"/".join(key): value for key, value in sorted(counter.items())}


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# {result['experiment_id']}",
        "",
        "OCR/text teacher-consensus diagnostic for Eval300 top-8 assignment candidates.",
        "",
        "This is not a promotion result: policies use train-teacher OCR/text signals at selection time and no counted `.oscr` stream is exported.",
        "",
        "## Validation Policies",
        "",
        "| policy | changed | exact changed | false | wrong | missed | Tesseract delta | PARSeq delta | Tesseract worse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    val = result["metrics"]["val"]
    for policy, metrics in val.items():
        lines.append(
            f"| `{policy}` | {metrics['changed_groups']} | {metrics['exact_changed_groups']} | "
            f"{metrics['false_change']} | {metrics['wrong_change']} | {metrics['missed_oracle']} | "
            f"{metrics['tesseract_delta_vs_nearest']} | {metrics['parseq_delta_vs_nearest']} | "
            f"{metrics['tesseract_worse_groups']} |"
        )
    lines.extend(["", "## Oracle Filter Coverage", ""])
    for policy, coverage in result["oracle_filter_coverage"].items():
        cov = coverage["val"]
        lines.append(
            f"- `{policy}`: oracle row passes `{cov.get('oracle_row_passes', 0)}/"
            f"{cov.get('oracle_change_groups', 0)}`, no-headroom groups with any passing candidate "
            f"`{cov.get('no_oracle_groups_with_passing_candidate', 0)}/{cov.get('no_oracle_change_groups', 0)}`"
        )
    lines.extend(["", "## Interpretation", "", result["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--crnn", type=Path, required=True)
    parser.add_argument("--abinet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment-id", default="eval300_top8_trainval_teacher_consensus")
    parser.add_argument("--profile", default="unicode_strict_v1")
    args = parser.parse_args()

    table_rows = read_jsonl(args.table)
    crnn_rows = {row_key(row): row for row in read_jsonl(args.crnn)}
    abinet_rows = {row_key(row): row for row in read_jsonl(args.abinet)}
    rows = enrich_rows(table_rows, crnn_rows, abinet_rows, args.profile)
    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: (int(row["topk_rank"]), int(row["code_index"])))
    add_group_deltas(groups)

    metrics, audits = evaluate_policies(groups)
    coverage = oracle_filter_coverage(groups)
    best_val_policy = min(
        (name for name in metrics["val"] if name not in {"nearest", "heldout_oracle"}),
        key=lambda name: (
            metrics["val"][name]["tesseract_delta_vs_nearest"],
            metrics["val"][name]["false_change"] + metrics["val"][name]["wrong_change"],
            metrics["val"][name]["changed_groups"],
        ),
    )
    current_floor = -8
    best_val = metrics["val"][best_val_policy]
    interpretation = (
        f"Best fixed teacher-consensus validation policy is `{best_val_policy}` with "
        f"Tesseract delta {best_val['tesseract_delta_vs_nearest']} and "
        f"{best_val['false_change'] + best_val['wrong_change']} false/wrong changes. "
        f"It {'beats' if best_val['tesseract_delta_vs_nearest'] < current_floor else 'does not beat'} "
        "the current actual-bitstream floor of -8 edits, and it is non-promotable without "
        "distillation or explicit encoder OCR cost accounting."
    )
    result = {
        "experiment_id": args.experiment_id,
        "validity": "diagnostic_teacher_consensus_not_promoted",
        "profile": args.profile,
        "inputs": {
            "table": {"path": str(args.table), "sha256": sha256_file(args.table)},
            "crnn": {"path": str(args.crnn), "sha256": sha256_file(args.crnn)},
            "abinet": {"path": str(args.abinet), "sha256": sha256_file(args.abinet)},
        },
        "summary": {
            "groups": len(groups),
            "table_rows": len(table_rows),
            "enriched_rows": len(rows),
            "train_groups": sum(1 for key in groups if key[1] == "train"),
            "val_groups": sum(1 for key in groups if key[1] == "val"),
            "best_val_policy": best_val_policy,
            "best_val_policy_metrics": best_val,
        },
        "metrics": metrics,
        "oracle_filter_coverage": coverage,
        "val_source_breakdown_best_policy": source_breakdown(audits, best_val_policy),
        "val_audits": audits,
        "interpretation": interpretation,
        "hashes": {
            "script": sha256_file(Path(__file__)),
        },
    }
    write_json(args.output, result)
    write_report(args.report, result)
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "best_val_policy": best_val_policy,
                "best_val_policy_metrics": best_val,
                "output": str(args.output),
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
