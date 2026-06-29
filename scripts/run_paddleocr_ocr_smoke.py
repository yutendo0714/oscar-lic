#!/usr/bin/env python3
"""Run PaddleOCR text recognition on crop splits and report OCR metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import paddle
import paddleocr
from paddleocr import TextRecognition


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_dir(path: Path) -> list[dict]:
    if not path.is_dir():
        return []
    rows: list[dict] = []
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if ".cache" in file_path.parts:
            continue
        rows.append(
            {
                "path": str(file_path.relative_to(path)),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return rows


def run_batches(recognizer, records: list[dict], batch_size: int) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = [str(record["image_path"]) for record in batch]
        infer_start = perf_counter()
        outputs = recognizer.predict(paths, batch_size=batch_size)
        infer_seconds = perf_counter() - infer_start
        if len(outputs) != len(batch):
            raise RuntimeError(f"PaddleOCR output count mismatch: {len(outputs)} != {len(batch)}")
        for record, output in zip(batch, outputs):
            rows.append(
                {
                    "image": project_path(record["image_path"]),
                    "reference": record["text"],
                    "prediction": str(output.get("rec_text", "")),
                    "confidence": (
                        None
                        if output.get("rec_score") is None
                        else float(output.get("rec_score"))
                    ),
                    "source": record.get("source"),
                    "split": record.get("split"),
                    "method_id": record.get("method_id"),
                    "infer_seconds_batch_amortized": infer_seconds / len(batch),
                }
            )
    return rows


def maybe_log_wandb(args, summary: dict) -> str | None:
    if args.no_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["ocr", "paddleocr", "heldout", "smoke"],
        config={
            "split": str(args.split),
            "model_name": args.model_name,
            "model_dir": str(args.model_dir) if args.model_dir else None,
            "device": args.device,
            "batch_size": args.batch_size,
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
    parser.add_argument("--model-name", default="PP-OCRv6_medium_rec")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--enable-mkldnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-model-source-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["unicode_strict_v1", "latin_alnum_ci_v1", "raw_exact_v1"],
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oscar-lic")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.disable_model_source_check:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    records = read_records(args.split)
    recognizer = TextRecognition(
        model_name=args.model_name,
        model_dir=str(args.model_dir) if args.model_dir else None,
        device=args.device,
        enable_mkldnn=args.enable_mkldnn,
        cpu_threads=args.cpu_threads,
    )
    rows = run_batches(recognizer, records, args.batch_size)

    metrics_by_profile = {
        profile: evaluate_pairs([(row["reference"], row["prediction"]) for row in rows], profile).to_dict()
        for profile in args.profiles
    }
    cache_model_dir = Path("/root/.paddlex/official_models") / args.model_name
    model_dir = args.model_dir if args.model_dir is not None else cache_model_dir
    summary = {
        "split": project_path(args.split),
        "samples": len(rows),
        "model": "paddleocr_text_recognition",
        "model_name": args.model_name,
        "model_dir": str(model_dir),
        "model_files": hash_model_dir(model_dir),
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "enable_mkldnn": args.enable_mkldnn,
        "paddle": paddle.__version__,
        "paddleocr": getattr(paddleocr, "__version__", None),
        "mean_infer_seconds": sum(row["infer_seconds_batch_amortized"] for row in rows) / len(rows),
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
    if args.quiet:
        print(json.dumps({"summary_path": str(summary_path), "results_path": str(results_path)}, ensure_ascii=False))
    else:
        print(json.dumps({"summary": summary, "results_path": str(results_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
