#!/usr/bin/env python3
"""Run a StrHub recognizer on word crops and report OCR metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import evaluate_pairs  # noqa: E402
from strhub.data.module import SceneTextDataModule  # noqa: E402
from strhub.models.utils import create_model  # noqa: E402


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
            record["text"] = str(record.get("text", record.get("reference", "")))
            if not record["text"]:
                raise ValueError(f"{path}:{line_no}: missing text/reference")
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


def find_checkpoint(model_name: str) -> Path | None:
    checkpoints = sorted((Path(torch.hub.get_dir()) / "checkpoints").glob(f"{model_name}-*.pt"))
    return checkpoints[0] if checkpoints else None


def create_strhub_model(args: argparse.Namespace):
    if args.model_name == "parseq":
        return create_model(
            "parseq",
            pretrained=True,
            decode_ar=args.decode_ar,
            refine_iters=args.refine_iters,
        )
    if args.model_name == "parseq_tiny":
        return create_model(
            "parseq-tiny",
            pretrained=True,
            decode_ar=args.decode_ar,
            refine_iters=args.refine_iters,
        )
    if args.model_name == "abinet":
        return create_model("abinet", pretrained=True, iter_size=args.iter_size)
    return create_model(args.model_name, pretrained=True)


@torch.inference_mode()
def run_batches(model, transform, records: list[dict], batch_size: int, device: str) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = [transform(Image.open(record["image_path"]).convert("RGB")) for record in batch]
        tensor = torch.stack(images).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_start = perf_counter()
        logits = model(tensor)
        probs = logits.softmax(-1)
        predictions, token_probs = model.tokenizer.decode(probs)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_seconds = perf_counter() - infer_start
        for record, prediction, probs_i in zip(batch, predictions, token_probs):
            confidence = float(probs_i.prod().item()) if probs_i.numel() else 0.0
            row = {
                "image": project_path(record["image_path"]),
                "reference": record["text"],
                "prediction": prediction,
                "confidence": confidence,
                "token_count": int(probs_i.numel()),
                "source": record.get("source"),
                "split": record.get("split"),
                "method_id": record.get("method_id"),
                "infer_seconds_batch_amortized": infer_seconds / len(batch),
            }
            for key in [
                "seed",
                "real_seed",
                "original_seed",
                "source_index",
                "candidate_index",
                "candidate_slot",
                "code_index",
                "nearest_code",
                "topk_rank",
                "is_nearest",
            ]:
                if key in record:
                    row[key] = record[key]
            rows.append(row)
    return rows


def maybe_log_wandb(args, summary: dict) -> str | None:
    if args.no_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["ocr", "strhub", args.model_name, "teacher-lock"],
        config={
            "split": str(args.split),
            "model_name": args.model_name,
            "decode_ar": args.decode_ar,
            "refine_iters": args.refine_iters,
            "iter_size": args.iter_size,
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
    parser.add_argument("--model-name", choices=["parseq", "parseq_tiny", "crnn", "abinet", "trba", "vitstr"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--decode-ar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--iter-size", type=int, default=3)
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
    model = create_strhub_model(args).eval().to(device)
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    rows = run_batches(model, transform, records, args.batch_size, device)

    metrics_by_profile = {
        profile: evaluate_pairs([(row["reference"], row["prediction"]) for row in rows], profile).to_dict()
        for profile in args.profiles
    }
    checkpoint = find_checkpoint(args.model_name.replace("_", "-"))
    if checkpoint is None and args.model_name == "parseq_tiny":
        checkpoint = find_checkpoint("parseq-tiny")
    summary = {
        "split": project_path(args.split),
        "samples": len(rows),
        "model": "strhub",
        "model_name": args.model_name,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint else None,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "decode_ar": args.decode_ar if args.model_name.startswith("parseq") else None,
        "refine_iters": args.refine_iters if args.model_name.startswith("parseq") else None,
        "iter_size": args.iter_size if args.model_name == "abinet" else None,
        "mean_confidence": sum(row["confidence"] for row in rows) / len(rows),
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
