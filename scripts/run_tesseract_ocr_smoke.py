#!/usr/bin/env python3
"""Run Tesseract CLI OCR on OSCAR-LIC crop splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import evaluate_pairs  # noqa: E402


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_image_path(split_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    split_relative = (split_path.parent / path).resolve()
    if split_relative.is_file():
        return split_relative
    return (ROOT / path).resolve()


def read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = resolve_image_path(path, record["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_no}: missing image {image_path}")
            record["image_path"] = image_path
            record["text"] = str(record["text"])
            records.append(record)
    if not records:
        raise ValueError(f"empty OCR split: {path}")
    return records


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def tesseract_version() -> str | None:
    return command_output(["tesseract", "--version"])


def dpkg_versions() -> dict[str, str]:
    packages = ["tesseract-ocr", "tesseract-ocr-eng", "tesseract-ocr-osd", "libtesseract5", "libleptonica6"]
    completed = subprocess.run(
        ["dpkg-query", "-W", "-f=${Package}\t${Version}\n", *packages],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    versions = {}
    for line in completed.stdout.splitlines():
        if "\t" in line:
            package, version = line.split("\t", 1)
            versions[package] = version
    return versions


def run_one(args, record: dict) -> dict:
    command = [
        args.tesseract,
        str(record["image_path"]),
        "stdout",
        "--oem",
        str(args.oem),
        "--psm",
        str(args.psm),
        "-l",
        args.lang,
    ]
    if args.disable_dawg:
        command.extend(["-c", "load_system_dawg=0", "-c", "load_freq_dawg=0"])
    start = perf_counter()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    seconds = perf_counter() - start
    prediction = completed.stdout.replace("\f", "").strip()
    return {
        "image": project_path(record["image_path"]),
        "reference": record["text"],
        "prediction": prediction,
        "source": record.get("source"),
        "split": record.get("split"),
        "method_id": record.get("method_id"),
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "infer_seconds": seconds,
    }


def maybe_log_wandb(args, summary: dict) -> str | None:
    if args.no_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["ocr", "tesseract", "heldout", "smoke"],
        config={
            "split": str(args.split),
            "psm": args.psm,
            "oem": args.oem,
            "lang": args.lang,
            "disable_dawg": args.disable_dawg,
            "profiles": args.profiles,
        },
    )
    wandb.log({f"ocr/{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
    run_id = run.id
    run.finish()
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--psm", type=int, default=7)
    parser.add_argument("--oem", type=int, default=1)
    parser.add_argument("--disable-dawg", action="store_true")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["unicode_strict_v1", "latin_alnum_ci_v1", "raw_exact_v1"],
    )
    parser.add_argument("--tessdata", type=Path, default=Path("/usr/share/tesseract-ocr/5/tessdata/eng.traineddata"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oscar-lic")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    records = read_records(args.split)
    rows = [run_one(args, record) for record in records]
    metrics_by_profile = {
        profile: evaluate_pairs([(row["reference"], row["prediction"]) for row in rows], profile).to_dict()
        for profile in args.profiles
    }
    summary = {
        "split": project_path(args.split),
        "samples": len(rows),
        "model": "tesseract",
        "tesseract_version": tesseract_version(),
        "dpkg_versions": dpkg_versions(),
        "tessdata_path": str(args.tessdata),
        "tessdata_sha256": sha256_file(args.tessdata),
        "lang": args.lang,
        "psm": args.psm,
        "oem": args.oem,
        "disable_dawg": args.disable_dawg,
        "failed_samples": sum(1 for row in rows if row["returncode"] != 0),
        "mean_infer_seconds": sum(row["infer_seconds"] for row in rows) / len(rows),
        "metrics": metrics_by_profile,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary["wandb_run_id"] = maybe_log_wandb(args, summary)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "results_path": str(results_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
