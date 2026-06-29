#!/usr/bin/env python3
"""Build current-relative labels for an executed exception-center stream policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, normalize_text  # noqa: E402


SEED_RE = re.compile(r"seed(?P<seed>\d+)_val(?P<count>\d+)")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_seed_dir(path: Path) -> tuple[int, int]:
    match = SEED_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer seed/count from stream directory name: {path}")
    return int(match.group("seed")), int(match.group("count"))


def seed_offsets(stream_dirs: list[Path]) -> dict[int, int]:
    offset = 0
    result: dict[int, int] = {}
    for path in sorted(stream_dirs, key=lambda item: parse_seed_dir(item)[0]):
        seed, count = parse_seed_dir(path)
        result[seed] = offset
        offset += count
    return result


def load_stream_rows(stream_dirs: list[Path], offsets: dict[int, int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for directory in stream_dirs:
        seed, _count = parse_seed_dir(directory)
        for row in read_jsonl(directory / "results.jsonl"):
            global_index = offsets[seed] + int(row["index"])
            out = dict(row)
            out["stream_seed"] = seed
            out["global_index"] = global_index
            rows[global_index] = out
    return rows


def load_policy_rows(policy_dir: Path, offsets: dict[int, int]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(policy_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            if "selected_by_policy" in row and not bool(row["selected_by_policy"]):
                continue
            seed = int(row["real_seed"])
            global_index = offsets[seed] + int(row["source_index"])
            rows[global_index].append(row)
    return dict(rows)


def load_ocr_rows(directory: Path) -> list[dict[str, Any]]:
    return read_jsonl(directory / "results.jsonl")


def distance(reference: str, prediction: str, profile: str) -> int:
    return int(character_counts(reference, prediction, profile).distance)


def exact(reference: str, prediction: str, profile: str) -> bool:
    return normalize_text(reference, profile) == normalize_text(prediction, profile)


def action_name(current_selected: bool, exception_selected: bool) -> str:
    if current_selected and exception_selected:
        return "replace_with_exception"
    if current_selected and not exception_selected:
        return "drop_current"
    if (not current_selected) and exception_selected:
        return "add_exception"
    return "keep_base"


def aggregate_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pixels = sum(int(row["width"]) * int(row["height"]) for row in rows)
    total_bytes = sum(int(row["actual_total_bytes"]) for row in rows)
    enh_bytes = sum(float(row["enhancement_payload_bpp"]) * int(row["width"]) * int(row["height"]) / 8.0 for row in rows)
    return {
        "images": len(rows),
        "actual_total_bytes": int(total_bytes),
        "actual_total_bpp_weighted": None if pixels == 0 else float(8.0 * total_bytes / pixels),
        "actual_total_bpp_mean": None if not rows else float(sum(float(row["actual_total_bpp"]) for row in rows) / len(rows)),
        "enhancement_payload_bpp_mean": None
        if not rows
        else float(sum(float(row["enhancement_payload_bpp"]) for row in rows) / len(rows)),
        "enhancement_payload_bytes_estimated_from_bpp": float(enh_bytes),
    }


def summarize_label_rows(rows: list[dict[str, Any]], profiles: list[str]) -> dict[str, Any]:
    actions = sorted(set(row["action_taken"] for row in rows))
    label_counts = Counter(row["strict_label"] for row in rows)
    result: dict[str, Any] = {
        "rows": len(rows),
        "actions": dict(Counter(row["action_taken"] for row in rows)),
        "strict_labels": dict(label_counts),
        "rate": {
            "current": aggregate_rate([row["current_stream"] for row in rows]),
            "exception": aggregate_rate([row["exception_stream"] for row in rows]),
        },
        "by_action": {},
        "ocr": {},
    }
    for action in actions:
        subset = [row for row in rows if row["action_taken"] == action]
        result["by_action"][action] = {
            "rows": len(subset),
            "strict_labels": dict(Counter(row["strict_label"] for row in subset)),
            "rate": {
                "current": aggregate_rate([row["current_stream"] for row in subset]),
                "exception": aggregate_rate([row["exception_stream"] for row in subset]),
            },
        }
    for model in ["tesseract", "parseq"]:
        result["ocr"][model] = {}
        for profile in profiles:
            result["ocr"][model][profile] = summarize_ocr_delta(rows, model, profile)
            for action in actions:
                result["by_action"][action].setdefault("ocr", {}).setdefault(model, {})[profile] = summarize_ocr_delta(
                    [row for row in rows if row["action_taken"] == action],
                    model,
                    profile,
                )
    return result


def summarize_ocr_delta(rows: list[dict[str, Any]], model: str, profile: str) -> dict[str, Any]:
    deltas = [int(row["ocr"][model][profile]["delta_errors"]) for row in rows]
    current_errors = [int(row["ocr"][model][profile]["current_distance"]) for row in rows]
    exception_errors = [int(row["ocr"][model][profile]["exception_distance"]) for row in rows]
    exact_deltas = [int(row["ocr"][model][profile]["delta_exact"]) for row in rows]
    return {
        "rows": len(rows),
        "current_errors": int(sum(current_errors)),
        "exception_errors": int(sum(exception_errors)),
        "delta_errors": int(sum(deltas)),
        "improved_rows": int(sum(1 for value in deltas if value < 0)),
        "worsened_rows": int(sum(1 for value in deltas if value > 0)),
        "equal_rows": int(sum(1 for value in deltas if value == 0)),
        "delta_exact_matches": int(sum(exact_deltas)),
    }


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    profiles = args.profile or ["unicode_strict_v1", "latin_alnum_ci_v1"]
    offsets = seed_offsets(args.current_stream_dir)
    if offsets != seed_offsets(args.exception_stream_dir):
        raise ValueError("current and exception stream dirs infer different seed offsets")
    current_stream = load_stream_rows(args.current_stream_dir, offsets)
    exception_stream = load_stream_rows(args.exception_stream_dir, offsets)
    selected_policy = load_policy_rows(args.policy_dir, offsets)
    if set(current_stream) != set(exception_stream):
        raise ValueError("current and exception stream row indices differ")

    ocr = {
        "tesseract": {
            "current": load_ocr_rows(args.current_tesseract_dir),
            "exception": load_ocr_rows(args.exception_tesseract_dir),
        },
        "parseq": {
            "current": load_ocr_rows(args.current_parseq_dir),
            "exception": load_ocr_rows(args.exception_parseq_dir),
        },
    }
    count = len(current_stream)
    for model, pair in ocr.items():
        if len(pair["current"]) != count or len(pair["exception"]) != count:
            raise ValueError(f"{model} OCR row count mismatch with streams")

    rows = []
    for index in sorted(current_stream):
        cur_stream = current_stream[index]
        exc_stream = exception_stream[index]
        current_selected = int(cur_stream["selected_candidate_count"]) > 0
        exception_selected = int(exc_stream["selected_candidate_count"]) > 0
        action = action_name(current_selected, exception_selected)
        row: dict[str, Any] = {
            "global_index": index,
            "seed": int(cur_stream["stream_seed"]),
            "source_index": int(cur_stream["index"]),
            "source_image": cur_stream["image"],
            "width": int(cur_stream["width"]),
            "height": int(cur_stream["height"]),
            "current_selected_candidate_count": int(cur_stream["selected_candidate_count"]),
            "exception_selected_candidate_count": int(exc_stream["selected_candidate_count"]),
            "current_selected": current_selected,
            "exception_selected": exception_selected,
            "action_taken": action,
            "policy_rows": selected_policy.get(index, []),
            "policy_oracle_change_proxy_any": any(bool(item.get("oracle_change_proxy")) for item in selected_policy.get(index, [])),
            "policy_oracle_change_proxy_all": all(bool(item.get("oracle_change_proxy")) for item in selected_policy.get(index, []))
            if selected_policy.get(index)
            else False,
            "current_stream": compact_stream_row(cur_stream),
            "exception_stream": compact_stream_row(exc_stream),
            "rate_delta": {
                "actual_total_bytes": int(exc_stream["actual_total_bytes"]) - int(cur_stream["actual_total_bytes"]),
                "actual_total_bpp": float(exc_stream["actual_total_bpp"]) - float(cur_stream["actual_total_bpp"]),
                "enhancement_payload_bpp": float(exc_stream["enhancement_payload_bpp"])
                - float(cur_stream["enhancement_payload_bpp"]),
                "psnr_delta_db_exception_minus_current": float(exc_stream["psnr_enhanced_db"])
                - float(cur_stream["psnr_enhanced_db"]),
            },
            "ocr": {},
        }
        all_deltas: list[int] = []
        for model, pair in ocr.items():
            cur_ocr = pair["current"][index]
            exc_ocr = pair["exception"][index]
            if str(cur_ocr["reference"]) != str(exc_ocr["reference"]):
                raise ValueError(f"{model} reference mismatch at row {index}")
            row["reference"] = str(cur_ocr["reference"])
            row["ocr"][model] = {
                "current_prediction": str(cur_ocr["prediction"]),
                "exception_prediction": str(exc_ocr["prediction"]),
                "profiles": profiles,
            }
            for profile in profiles:
                cur_distance = distance(str(cur_ocr["reference"]), str(cur_ocr["prediction"]), profile)
                exc_distance = distance(str(exc_ocr["reference"]), str(exc_ocr["prediction"]), profile)
                cur_exact = exact(str(cur_ocr["reference"]), str(cur_ocr["prediction"]), profile)
                exc_exact = exact(str(exc_ocr["reference"]), str(exc_ocr["prediction"]), profile)
                delta = exc_distance - cur_distance
                all_deltas.append(delta)
                row["ocr"][model][profile] = {
                    "current_distance": cur_distance,
                    "exception_distance": exc_distance,
                    "delta_errors": delta,
                    "current_exact": cur_exact,
                    "exception_exact": exc_exact,
                    "delta_exact": int(exc_exact) - int(cur_exact),
                }
        harmful = any(delta > 0 for delta in all_deltas)
        beneficial = any(delta < 0 for delta in all_deltas)
        if harmful and beneficial:
            label = "mixed_harm_and_benefit"
        elif harmful:
            label = "harmful_any_profile"
        elif beneficial:
            label = "beneficial_no_profile_harm"
        else:
            label = "neutral_all_profiles"
        row["strict_label"] = label
        rows.append(row)
    return rows


def compact_stream_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "global_index",
        "stream_seed",
        "index",
        "image",
        "width",
        "height",
        "selected_candidate_count",
        "candidate_count",
        "codebook_size",
        "actual_total_bytes",
        "actual_total_bpp",
        "base_payload_bpp",
        "enhancement_payload_bpp",
        "psnr_base_db",
        "psnr_enhanced_db",
        "psnr_delta_db",
        "assignment_mode",
        "assignment_changed_count",
        "stream_path",
        "reconstruction_path",
    ]
    return {key: row.get(key) for key in keys if key in row}


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Current-Relative Exception Labels",
        "",
        "This artifact labels the executed N141 exception-center stream against the matched current compact stream.",
        "It is not a full counterfactual oracle over all possible replacement/drop actions.",
        "",
        "## Action Summary",
        "",
        "| action | rows | beneficial | neutral | harmful | mixed | current bpp | exception bpp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for action, data in sorted(result["summary"]["by_action"].items()):
        labels = data["strict_labels"]
        cur_bpp = data["rate"]["current"]["actual_total_bpp_weighted"]
        exc_bpp = data["rate"]["exception"]["actual_total_bpp_weighted"]
        lines.append(
            f"| {action} | {data['rows']} | {labels.get('beneficial_no_profile_harm', 0)} | "
            f"{labels.get('neutral_all_profiles', 0)} | {labels.get('harmful_any_profile', 0)} | "
            f"{labels.get('mixed_harm_and_benefit', 0)} | {cur_bpp:.6f} | {exc_bpp:.6f} |"
        )
    lines.extend(["", "## OCR Delta By Action", ""])
    for model in ["tesseract", "parseq"]:
        lines.extend(
            [
                f"### {model}",
                "",
                "| action | profile | rows | delta errors | improved | worsened | equal | delta exact |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for action, data in sorted(result["summary"]["by_action"].items()):
            for profile, metrics in data["ocr"][model].items():
                lines.append(
                    f"| {action} | {profile} | {metrics['rows']} | {metrics['delta_errors']} | "
                    f"{metrics['improved_rows']} | {metrics['worsened_rows']} | "
                    f"{metrics['equal_rows']} | {metrics['delta_exact_matches']} |"
                )
        lines.append("")
    lines.extend(["## Conclusion", "", result["conclusion"], "", "## Next Action", "", result["next_action"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def log_wandb(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any] | None:
    if not args.wandb_project:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_id,
            tags=["oscar-lic", "exception-centers", "current-relative-labels", "diagnostic"],
            config={"experiment_id": args.experiment_id},
        )
        wandb.log(
            {
                "rows": result["summary"]["rows"],
                "actions/replace_with_exception": result["summary"]["actions"].get("replace_with_exception", 0),
                "actions/drop_current": result["summary"]["actions"].get("drop_current", 0),
                "labels/harmful_any_profile": result["summary"]["strict_labels"].get("harmful_any_profile", 0),
                "labels/mixed_harm_and_benefit": result["summary"]["strict_labels"].get("mixed_harm_and_benefit", 0),
            }
        )
        for model, profiles in result["summary"]["ocr"].items():
            for profile, metrics in profiles.items():
                wandb.log({f"{model}/{profile}/delta_errors": metrics["delta_errors"]})
        run.finish()
        return {"run_id": run.id, "url": run.url}
    except Exception as exc:  # pragma: no cover - wandb availability is environment-dependent.
        return {"error": repr(exc)}


def write_manifest(path: Path, args: argparse.Namespace, result_path: Path, table_path: Path, report_path: Path) -> None:
    script = Path(__file__).resolve().relative_to(Path.cwd().resolve())
    files = [
        {"name": "script", "path": str(script), "sha256": sha256_file(script)},
        {"name": "result", "path": str(result_path), "sha256": sha256_file(result_path)},
        {"name": "label_table", "path": str(table_path), "sha256": sha256_file(table_path)},
        {"name": "report", "path": str(report_path), "sha256": sha256_file(report_path)},
    ]
    for name, directory in [
        ("current_tesseract", args.current_tesseract_dir),
        ("exception_tesseract", args.exception_tesseract_dir),
        ("current_parseq", args.current_parseq_dir),
        ("exception_parseq", args.exception_parseq_dir),
    ]:
        files.append({"name": f"{name}_summary", "path": str(directory / "summary.json"), "sha256": sha256_file(directory / "summary.json")})
        files.append({"name": f"{name}_results", "path": str(directory / "results.jsonl"), "sha256": sha256_file(directory / "results.jsonl")})
    for directory in args.current_stream_dir:
        files.append({"name": f"current_stream_{directory.name}_summary", "path": str(directory / "summary.json"), "sha256": sha256_file(directory / "summary.json")})
        files.append({"name": f"current_stream_{directory.name}_results", "path": str(directory / "results.jsonl"), "sha256": sha256_file(directory / "results.jsonl")})
    for directory in args.exception_stream_dir:
        files.append({"name": f"exception_stream_{directory.name}_summary", "path": str(directory / "summary.json"), "sha256": sha256_file(directory / "summary.json")})
        files.append({"name": f"exception_stream_{directory.name}_results", "path": str(directory / "results.jsonl"), "sha256": sha256_file(directory / "results.jsonl")})
    for policy in sorted(args.policy_dir.glob("*.jsonl")):
        files.append({"name": f"policy_{policy.name}", "path": str(policy), "sha256": sha256_file(policy)})
    manifest = {
        "experiment_id": args.experiment_id,
        "status": "completed_diagnostic",
        "command": " ".join(["scripts/build_current_relative_exception_labels.py", *sys.argv[1:]]),
        "code_commit": git_commit(),
        "inputs_and_outputs": files,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--current-stream-dir", type=Path, action="append", required=True)
    parser.add_argument("--exception-stream-dir", type=Path, action="append", required=True)
    parser.add_argument("--current-tesseract-dir", type=Path, required=True)
    parser.add_argument("--exception-tesseract-dir", type=Path, required=True)
    parser.add_argument("--current-parseq-dir", type=Path, required=True)
    parser.add_argument("--exception-parseq-dir", type=Path, required=True)
    parser.add_argument("--profile", action="append", default=None)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--experiment-id", default="eval300_exception_center_current_relative_labels_2026_06_26")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.profile = args.profile or ["unicode_strict_v1", "latin_alnum_ci_v1"]
    rows = build_rows(args)
    summary = summarize_label_rows(rows, args.profile)
    result = {
        "experiment_id": args.experiment_id,
        "hypothesis_id": "H4-current-relative-exception-action-labels",
        "status": "completed_diagnostic",
        "code_commit": git_commit(),
        "description": "Current-relative labels for the executed rel<=1.0 exception-center stream policy.",
        "summary": summary,
        "conclusion": (
            "The executed exception-center policy is mainly a sparse/drop policy rather than a proven replacement "
            "policy. Any next selector must optimize against current-relative OCR labels, not only nearest-vs-exception "
            "proxy labels."
        ),
        "next_action": (
            "Use these labels as an audit target, then build explicit replacement/drop counterfactuals for current "
            "selected units before training another exception-center selector."
        ),
    }
    result["wandb"] = log_wandb(args, result)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_table, rows)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.report, result)
    write_manifest(args.manifest, args, args.result, args.output_table, args.report)
    print(json.dumps({"result": str(args.result), "table": str(args.output_table), "report": str(args.report), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
