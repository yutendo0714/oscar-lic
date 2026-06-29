#!/usr/bin/env python3
"""Run PaddleOCR detector+recognizer on OCR splits.

This is held-out evaluation only. Do not feed these outputs into training,
selection, utility labels or threshold tuning under the current protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import paddle
import paddleocr
from paddleocr import PaddleOCR


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


def read_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = resolve_image_path(path, row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_no}: missing image {image_path}")
            row["image_path"] = image_path
            row["text"] = str(row["text"])
            rows.append(row)
    if not rows:
        raise ValueError(f"empty OCR split: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_dir(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        if ".cache" in file_path.parts:
            continue
        out.append(
            {
                "path": str(file_path.relative_to(path)),
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return out


def as_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    getter = getattr(result, "get", None)
    if getter is not None:
        keys = [
            "input_path",
            "dt_polys",
            "rec_texts",
            "rec_scores",
            "rec_polys",
            "rec_boxes",
            "text_det_params",
            "model_settings",
        ]
        return {key: getter(key) for key in keys}
    return {}


def box_sort_key(poly_or_box: Any) -> tuple[float, float]:
    try:
        import numpy as np

        arr = np.asarray(poly_or_box, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 4:
            return (float(arr[1]), float(arr[0]))
        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            return (float(arr[:, 1].min()), float(arr[:, 0].min()))
    except Exception:  # noqa: BLE001 - best-effort reading order for diagnostics
        pass
    return (0.0, 0.0)


def prediction_from_result(result: Any) -> dict[str, Any]:
    data = as_mapping(result)
    texts = [str(text) for text in (data.get("rec_texts") or [])]
    scores_raw = data.get("rec_scores") or []
    scores = [None if score is None else float(score) for score in scores_raw]
    boxes = data.get("rec_boxes")
    polys = data.get("rec_polys") or data.get("dt_polys") or []
    order = list(range(len(texts)))
    if boxes is not None:
        order.sort(key=lambda idx: box_sort_key(boxes[idx]))
    elif polys:
        order.sort(key=lambda idx: box_sort_key(polys[idx]))
    ordered_texts = [texts[idx] for idx in order]
    ordered_scores = [scores[idx] if idx < len(scores) else None for idx in order]
    return {
        "prediction": " ".join(text for text in ordered_texts if text),
        "rec_texts": ordered_texts,
        "rec_scores": ordered_scores,
        "detection_count": len(data.get("dt_polys") or []),
        "recognition_count": len(texts),
        "text_det_params": data.get("text_det_params"),
    }


def run_records(ocr: PaddleOCR, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for record in records:
        start = perf_counter()
        result_list = ocr.predict(str(record["image_path"]))
        infer_seconds = perf_counter() - start
        if not result_list:
            parsed = {
                "prediction": "",
                "rec_texts": [],
                "rec_scores": [],
                "detection_count": 0,
                "recognition_count": 0,
                "text_det_params": None,
            }
        else:
            parsed = prediction_from_result(result_list[0])
        outputs.append(
            {
                "image": project_path(record["image_path"]),
                "reference": record["text"],
                "prediction": parsed["prediction"],
                "rec_texts": parsed["rec_texts"],
                "rec_scores": parsed["rec_scores"],
                "detection_count": parsed["detection_count"],
                "recognition_count": parsed["recognition_count"],
                "source": record.get("source"),
                "split": record.get("split"),
                "method_id": record.get("method_id"),
                "infer_seconds": infer_seconds,
            }
        )
    return outputs


def maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["ocr", "paddleocr", "heldout", "e2e"],
        config={
            "split": str(args.split),
            "text_detection_model_name": args.text_detection_model_name,
            "text_recognition_model_name": args.text_recognition_model_name,
            "device": args.device,
            "enable_mkldnn": args.enable_mkldnn,
            "cpu_threads": args.cpu_threads,
            "profiles": args.profiles,
        },
    )
    wandb.log({f"ocr/{key}": value for key, value in summary.items() if isinstance(value, (int, float))})
    run_id = run.id
    run.finish()
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-detection-model-name", default="PP-OCRv6_medium_det")
    parser.add_argument("--text-detection-model-dir", type=Path)
    parser.add_argument("--text-recognition-model-name", default="PP-OCRv6_medium_rec")
    parser.add_argument("--text-recognition-model-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--enable-mkldnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--text-recognition-batch-size", type=int, default=1)
    parser.add_argument("--text-det-limit-side-len", type=int, default=64)
    parser.add_argument("--text-det-limit-type", default="min")
    parser.add_argument("--disable-model-source-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["unicode_strict_v1", "latin_alnum_ci_v1", "raw_exact_v1"],
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oscar-lic")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.disable_model_source_check:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    records = read_records(args.split)
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=args.text_detection_model_name,
        text_detection_model_dir=str(args.text_detection_model_dir) if args.text_detection_model_dir else None,
        text_recognition_model_name=args.text_recognition_model_name,
        text_recognition_model_dir=str(args.text_recognition_model_dir) if args.text_recognition_model_dir else None,
        text_recognition_batch_size=args.text_recognition_batch_size,
        text_det_limit_side_len=args.text_det_limit_side_len,
        text_det_limit_type=args.text_det_limit_type,
        device=args.device,
        enable_mkldnn=args.enable_mkldnn,
        cpu_threads=args.cpu_threads,
    )
    rows = run_records(ocr, records)
    metrics_by_profile = {
        profile: evaluate_pairs([(row["reference"], row["prediction"]) for row in rows], profile).to_dict()
        for profile in args.profiles
    }
    det_dir = (
        args.text_detection_model_dir
        if args.text_detection_model_dir is not None
        else Path("/root/.paddlex/official_models") / args.text_detection_model_name
    )
    rec_dir = (
        args.text_recognition_model_dir
        if args.text_recognition_model_dir is not None
        else Path("/root/.paddlex/official_models") / args.text_recognition_model_name
    )
    summary: dict[str, Any] = {
        "split": project_path(args.split),
        "samples": len(rows),
        "model": "paddleocr_detector_recognizer",
        "text_detection_model_name": args.text_detection_model_name,
        "text_detection_model_dir": str(det_dir),
        "text_detection_model_files": hash_model_dir(det_dir),
        "text_recognition_model_name": args.text_recognition_model_name,
        "text_recognition_model_dir": str(rec_dir),
        "text_recognition_model_files": hash_model_dir(rec_dir),
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "enable_mkldnn": args.enable_mkldnn,
        "text_det_limit_side_len": args.text_det_limit_side_len,
        "text_det_limit_type": args.text_det_limit_type,
        "paddle": paddle.__version__,
        "paddleocr": getattr(paddleocr, "__version__", None),
        "mean_infer_seconds": sum(row["infer_seconds"] for row in rows) / len(rows),
        "zero_detection_samples": sum(1 for row in rows if int(row["detection_count"]) == 0),
        "multi_detection_samples": sum(1 for row in rows if int(row["detection_count"]) > 1),
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
    payload = {"summary_path": str(summary_path), "results_path": str(results_path), "summary": summary}
    if args.quiet:
        payload = {"summary_path": str(summary_path), "results_path": str(results_path)}
    print(json.dumps(payload, indent=None if args.quiet else 2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
