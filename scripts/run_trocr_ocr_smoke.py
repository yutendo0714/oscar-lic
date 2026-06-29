#!/usr/bin/env python3
"""Run a pinned TrOCR recognizer on word crops and report OCR metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import evaluate_pairs  # noqa: E402


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    base = path.parent
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = Path(record["image_path"])
            if not image_path.is_absolute():
                split_relative = (base / image_path).resolve()
                project_relative = (ROOT / image_path).resolve()
                image_path = split_relative if split_relative.is_file() else project_relative
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


def resolve_snapshot(model_id: str, revision: str) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(model_id, revision=revision)
    resolved_revision = info.sha
    snapshot = Path(snapshot_download(repo_id=model_id, revision=resolved_revision))
    return snapshot, resolved_revision


def hash_snapshot_files(snapshot: Path) -> list[dict]:
    suffixes = {".json", ".txt", ".model", ".bin", ".safetensors", ".vocab", ".merges"}
    rows = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file() and p.suffix in suffixes):
        rows.append(
            {
                "path": str(path.relative_to(snapshot)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


@torch.inference_mode()
def run_batches(model, processor, records: list[dict], batch_size: int, device: str, num_beams: int, max_new_tokens: int) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = [Image.open(record["image_path"]).convert("RGB") for record in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_start = perf_counter()
        generated_ids = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        infer_seconds = perf_counter() - infer_start
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for record, prediction in zip(batch, predictions):
            rows.append(
                {
                    "image": project_path(record["image_path"]),
                    "reference": record["text"],
                    "prediction": prediction,
                    "confidence": None,
                    "source": record.get("source"),
                    "split": record.get("split"),
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
        tags=["ocr", "trocr", "heldout", "smoke"],
        config={
            "split": str(args.split),
            "model_id": args.model_id,
            "revision": args.revision,
            "resolved_revision": summary["resolved_revision"],
            "normalization_profiles": args.profiles,
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
    parser.add_argument("--model-id", default="microsoft/trocr-small-printed")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cuda", action="store_true")
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

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device != "cuda":
        raise SystemExit("CUDA requested but unavailable")

    records = read_records(args.split)
    snapshot, resolved_revision = resolve_snapshot(args.model_id, args.revision)
    processor = TrOCRProcessor.from_pretrained(snapshot)
    model = VisionEncoderDecoderModel.from_pretrained(snapshot).eval().to(device)
    rows = run_batches(model, processor, records, args.batch_size, device, args.num_beams, args.max_new_tokens)

    metrics_by_profile = {
        profile: evaluate_pairs([(row["reference"], row["prediction"]) for row in rows], profile).to_dict()
        for profile in args.profiles
    }
    summary = {
        "split": project_path(args.split),
        "samples": len(rows),
        "model": "trocr",
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot),
        "snapshot_files": hash_snapshot_files(snapshot),
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
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
        print(json.dumps({"summary": {k: v for k, v in summary.items() if k != "snapshot_files"}, "results_path": str(results_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
