#!/usr/bin/env python3
"""Paired bootstrap CIs for OCR comparison JSON files with actual-bitstream rates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import character_counts, normalize_text  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected label=path")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("empty label")
    return label, Path(path)


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty bootstrap values")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


class CodecLookup:
    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, dict[str, Any]]] = {}

    def lookup(self, reconstruction_path: str) -> dict[str, Any]:
        recon = Path(reconstruction_path)
        if recon.parent.name != "reconstructions":
            raise ValueError(f"cannot infer codec result path from {reconstruction_path}")
        result_path = recon.parent.parent / "results.jsonl"
        if result_path not in self._cache:
            rows = read_jsonl(result_path)
            by_recon = {str(row["reconstruction_path"]): row for row in rows}
            by_name = {Path(row["reconstruction_path"]).name: row for row in rows}
            self._cache[result_path] = {**by_name, **by_recon}
        rows_by_recon = self._cache[result_path]
        if reconstruction_path in rows_by_recon:
            return rows_by_recon[reconstruction_path]
        if recon.name in rows_by_recon:
            return rows_by_recon[recon.name]
        raise KeyError(f"{reconstruction_path} not found in {result_path}")


def codec_sample(row: dict[str, Any]) -> dict[str, float]:
    width = int(row["width"])
    height = int(row["height"])
    pixels = width * height
    total_bytes = int(row["actual_total_bytes"])
    return {
        "bytes": float(total_bytes),
        "pixels": float(pixels),
        "actual_total_bpp": float(row.get("actual_total_bpp", 8.0 * total_bytes / pixels)),
        "enhancement_payload_bpp": float(row.get("enhancement_payload_bpp", 0.0)),
        "base_payload_bpp": float(row.get("base_payload_bpp", 0.0)),
        "selected_candidate_count": float(row.get("selected_candidate_count", 0.0)),
        "assignment_changed_count": float(row.get("assignment_changed_count", 0.0)),
    }


def load_samples(compare_path: Path, profiles: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    comparison = read_json(compare_path)
    lookup = CodecLookup()
    samples: list[dict[str, Any]] = []
    input_files = [str(compare_path)]
    for pair in comparison["pairs"]:
        baseline_path = Path(pair["baseline_results"])
        candidate_path = Path(pair["candidate_results"])
        input_files.extend([str(baseline_path), str(candidate_path)])
        baseline_rows = read_jsonl(baseline_path)
        candidate_rows = read_jsonl(candidate_path)
        if len(baseline_rows) != len(candidate_rows):
            raise ValueError(f"{pair['label']}: OCR row count mismatch")
        for index, (baseline_row, candidate_row) in enumerate(zip(baseline_rows, candidate_rows)):
            if str(baseline_row["reference"]) != str(candidate_row["reference"]):
                raise ValueError(f"{pair['label']}: reference mismatch at row {index}")
            base_codec = codec_sample(lookup.lookup(str(baseline_row["image"])))
            cand_codec = codec_sample(lookup.lookup(str(candidate_row["image"])))
            if base_codec["pixels"] != cand_codec["pixels"]:
                raise ValueError(f"{pair['label']}: pixel mismatch at row {index}")
            reference = str(baseline_row["reference"])
            base_pred = str(baseline_row["prediction"])
            cand_pred = str(candidate_row["prediction"])
            profile_stats: dict[str, dict[str, Any]] = {}
            for profile in profiles:
                base_counts = character_counts(reference, base_pred, profile)
                cand_counts = character_counts(reference, cand_pred, profile)
                base_exact = normalize_text(reference, profile) == normalize_text(base_pred, profile)
                cand_exact = normalize_text(reference, profile) == normalize_text(cand_pred, profile)
                profile_stats[profile] = {
                    "reference_chars": float(base_counts.reference_length),
                    "baseline_errors": float(base_counts.distance),
                    "candidate_errors": float(cand_counts.distance),
                    "baseline_exact": float(base_exact),
                    "candidate_exact": float(cand_exact),
                }
            samples.append(
                {
                    "pair_label": pair["label"],
                    "row_index": index,
                    "reference": reference,
                    "baseline_image": str(baseline_row["image"]),
                    "candidate_image": str(candidate_row["image"]),
                    "baseline_codec": base_codec,
                    "candidate_codec": cand_codec,
                    "profiles": profile_stats,
                }
            )
    return samples, sorted(set(input_files))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate_stats(samples: list[dict[str, Any]], indices: list[int]) -> dict[str, float]:
    base_bytes = sum(samples[i]["baseline_codec"]["bytes"] for i in indices)
    cand_bytes = sum(samples[i]["candidate_codec"]["bytes"] for i in indices)
    pixels = sum(samples[i]["baseline_codec"]["pixels"] for i in indices)
    base_bpp_values = [samples[i]["baseline_codec"]["actual_total_bpp"] for i in indices]
    cand_bpp_values = [samples[i]["candidate_codec"]["actual_total_bpp"] for i in indices]
    base_enh_values = [samples[i]["baseline_codec"]["enhancement_payload_bpp"] for i in indices]
    cand_enh_values = [samples[i]["candidate_codec"]["enhancement_payload_bpp"] for i in indices]
    return {
        "baseline_pixel_weighted_bpp": 8.0 * base_bytes / pixels if pixels else 0.0,
        "candidate_pixel_weighted_bpp": 8.0 * cand_bytes / pixels if pixels else 0.0,
        "delta_pixel_weighted_bpp": 8.0 * (cand_bytes - base_bytes) / pixels if pixels else 0.0,
        "baseline_mean_image_bpp": mean(base_bpp_values),
        "candidate_mean_image_bpp": mean(cand_bpp_values),
        "delta_mean_image_bpp": mean(cand_bpp_values) - mean(base_bpp_values),
        "baseline_mean_enhancement_bpp": mean(base_enh_values),
        "candidate_mean_enhancement_bpp": mean(cand_enh_values),
        "delta_mean_enhancement_bpp": mean(cand_enh_values) - mean(base_enh_values),
        "delta_bytes": cand_bytes - base_bytes,
    }


def profile_stats(samples: list[dict[str, Any]], indices: list[int], profile: str) -> dict[str, float]:
    base_errors = sum(samples[i]["profiles"][profile]["baseline_errors"] for i in indices)
    cand_errors = sum(samples[i]["profiles"][profile]["candidate_errors"] for i in indices)
    ref_chars = sum(samples[i]["profiles"][profile]["reference_chars"] for i in indices)
    base_exact = sum(samples[i]["profiles"][profile]["baseline_exact"] for i in indices)
    cand_exact = sum(samples[i]["profiles"][profile]["candidate_exact"] for i in indices)
    deltas = [
        samples[i]["profiles"][profile]["candidate_errors"] - samples[i]["profiles"][profile]["baseline_errors"]
        for i in indices
    ]
    n = len(indices)
    return {
        "baseline_char_errors": base_errors,
        "candidate_char_errors": cand_errors,
        "reference_characters": ref_chars,
        "delta_char_errors": cand_errors - base_errors,
        "baseline_cer_micro": base_errors / ref_chars if ref_chars else 0.0,
        "candidate_cer_micro": cand_errors / ref_chars if ref_chars else 0.0,
        "delta_cer_micro": (cand_errors - base_errors) / ref_chars if ref_chars else 0.0,
        "baseline_exact_rate": base_exact / n if n else 0.0,
        "candidate_exact_rate": cand_exact / n if n else 0.0,
        "delta_exact_rate": (cand_exact - base_exact) / n if n else 0.0,
        "improved_rate": sum(1 for value in deltas if value < 0) / n if n else 0.0,
        "worsened_rate": sum(1 for value in deltas if value > 0) / n if n else 0.0,
        "unchanged_rate": sum(1 for value in deltas if value == 0) / n if n else 0.0,
        "delta_exact_matches": cand_exact - base_exact,
    }


def summarize_with_bootstrap(
    samples: list[dict[str, Any]],
    profiles: list[str],
    bootstrap_samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    n = len(samples)
    all_indices = list(range(n))
    observed_rate = rate_stats(samples, all_indices)
    rate_boot: dict[str, list[float]] = {
        key: []
        for key in (
            "delta_pixel_weighted_bpp",
            "delta_mean_image_bpp",
            "delta_mean_enhancement_bpp",
            "delta_bytes",
        )
    }
    profile_boot: dict[str, dict[str, list[float]]] = {
        profile: {
            key: []
            for key in (
                "delta_char_errors",
                "delta_cer_micro",
                "delta_exact_rate",
                "improved_rate",
                "worsened_rate",
                "delta_exact_matches",
            )
        }
        for profile in profiles
    }
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(n) for _ in range(n)]
        boot_rate = rate_stats(samples, indices)
        for key in rate_boot:
            rate_boot[key].append(float(boot_rate[key]))
        for profile in profiles:
            boot_profile = profile_stats(samples, indices, profile)
            for key in profile_boot[profile]:
                profile_boot[profile][key].append(float(boot_profile[key]))

    rate = {
        "observed": observed_rate,
        "ci95": {key: [quantile(values, 0.025), quantile(values, 0.975)] for key, values in rate_boot.items()},
    }
    profiles_out: dict[str, Any] = {}
    for profile in profiles:
        observed_profile = profile_stats(samples, all_indices, profile)
        values_by_key = profile_boot[profile]
        profiles_out[profile] = {
            "observed": observed_profile,
            "ci95": {key: [quantile(values, 0.025), quantile(values, 0.975)] for key, values in values_by_key.items()},
            "bootstrap_std": {
                key: float(statistics.pstdev(values)) if values else 0.0 for key, values in values_by_key.items()
            },
        }
    return {
        "samples": n,
        "rate": rate,
        "profiles": profiles_out,
    }


def markdown_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    widths = [len(str(cell)) for cell in header]
    for row in rows[1:]:
        widths = [max(width, len(str(cell))) for width, cell in zip(widths, row)]
    def fmt(row: list[Any]) -> str:
        return "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(header), sep, *(fmt(row) for row in rows[1:])])


def f6(value: float) -> str:
    return f"{value:.6f}"


def write_report(output: dict[str, Any], path: Path, primary_profile: str) -> None:
    rows = [["comparison", "samples", "dCER obs [95% CI]", "dChars obs [95% CI]", "dBPP obs [95% CI]"]]
    for label, item in output["comparisons"].items():
        profile = item["profiles"][primary_profile]
        rate = item["rate"]
        dcer = profile["observed"]["delta_cer_micro"]
        dcer_ci = profile["ci95"]["delta_cer_micro"]
        dchars = profile["observed"]["delta_char_errors"]
        dchars_ci = profile["ci95"]["delta_char_errors"]
        dbpp = rate["observed"]["delta_mean_image_bpp"]
        dbpp_ci = rate["ci95"]["delta_mean_image_bpp"]
        rows.append(
            [
                label,
                item["samples"],
                f"{f6(dcer)} [{f6(dcer_ci[0])}, {f6(dcer_ci[1])}]",
                f"{dchars:.0f} [{dchars_ci[0]:.0f}, {dchars_ci[1]:.0f}]",
                f"{f6(dbpp)} [{f6(dbpp_ci[0])}, {f6(dbpp_ci[1])}]",
            ]
        )
    lines = [
        "# Eval300 OCR Comparison Bootstrap CI",
        "",
        f"- Bootstrap samples: `{output['bootstrap_samples']}`",
        f"- Seed: `{output['seed']}`",
        f"- Primary profile: `{primary_profile}`",
        "- Rate is read from codec `results.jsonl` rows backing each OCR reconstruction path.",
        "- Negative OCR deltas mean the candidate has fewer edit errors.",
        "",
        markdown_table(rows),
        "",
        "## Inputs",
        "",
    ]
    for label, item in output["comparisons"].items():
        lines.append(f"- `{label}`: `{item['comparison_path']}` SHA256 `{item['comparison_sha256']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=parse_labeled_path, action="append", required=True)
    parser.add_argument("--profiles", nargs="+", default=["unicode_strict_v1", "latin_alnum_ci_v1", "raw_exact_v1"])
    parser.add_argument("--primary-profile", default="unicode_strict_v1")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.primary_profile not in args.profiles:
        raise SystemExit("--primary-profile must be in --profiles")

    rng = random.Random(args.seed)
    output: dict[str, Any] = {
        "description": "Paired bootstrap CIs for OCR comparison JSONs; negative OCR deltas favor the candidate.",
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "profiles": args.profiles,
        "primary_profile": args.primary_profile,
        "comparisons": {},
    }
    for label, compare_path in args.comparison:
        samples, input_files = load_samples(compare_path, args.profiles)
        item = summarize_with_bootstrap(samples, args.profiles, args.bootstrap_samples, rng)
        item["comparison_path"] = str(compare_path)
        item["comparison_sha256"] = sha256_file(compare_path)
        item["input_files"] = [{"path": path, "sha256": sha256_file(Path(path))} for path in input_files]
        output["comparisons"][label] = item

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        write_report(output, args.report, args.primary_profile)
    print(json.dumps({"output": str(args.output), "report": str(args.report) if args.report else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
