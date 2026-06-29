#!/usr/bin/env python3
"""Run a small PaddleOCR full-image smoke on local TextOCR annotations.

This is an evaluation-protocol diagnostic for O002. It uses local files only,
does not download or copy dataset bytes, and reports a simple axis-aligned
IoU-based text spotting approximation rather than the official TextOCR metric.
PaddleOCR outputs remain held-out evaluator outputs and must not be used for
training, selection, utility labels, feature extraction or threshold tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import paddle
import paddleocr
from paddleocr import PaddleOCR


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.text_metrics import evaluate_pairs, normalize_text  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_textocr_records(
    annotation_json: Path,
    image_root: Path,
    *,
    max_images: int,
    min_words: int,
    max_words: int,
    seed: int,
) -> list[dict[str, Any]]:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for image_id, image in data["imgs"].items():
        ann_ids = data["imgToAnns"].get(image_id, [])
        words = []
        for ann_id in ann_ids:
            ann = data["anns"][ann_id]
            text = str(ann.get("utf8_string", "")).strip()
            if not text or text == "###":
                continue
            bbox = [float(v) for v in ann["bbox"]]
            if bbox[2] <= 1 or bbox[3] <= 1:
                continue
            words.append(
                {
                    "id": ann_id,
                    "text": text,
                    "bbox_xywh": bbox,
                    "points": [float(v) for v in ann.get("points", [])],
                    "area": float(ann.get("area", bbox[2] * bbox[3])),
                }
            )
        if not (min_words <= len(words) <= max_words):
            continue
        rel = str(image.get("file_name", ""))
        if rel.startswith("train/"):
            rel = rel[len("train/") :]
        image_path = image_root / rel
        if not image_path.is_file():
            continue
        candidates.append(
            {
                "dataset": "textocr",
                "image_id": image_id,
                "image_path": str(image_path),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "set": image.get("set"),
                "gt_words": words,
            }
        )
    rng = random.Random(seed)
    candidates.sort(key=lambda row: row["image_id"])
    rng.shuffle(candidates)
    selected = candidates[:max_images]
    selected.sort(key=lambda row: row["image_id"])
    if not selected:
        raise SystemExit("no TextOCR full-image records matched selection constraints")
    return selected


def as_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    getter = getattr(result, "get", None)
    if getter is None:
        return {}
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


def bbox_xyxy_from_any(value: Any) -> list[float] | None:
    try:
        import numpy as np

        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 4:
            x0, y0, x1, y1 = [float(v) for v in arr[:4]]
            if x1 < x0 or y1 < y0:
                x0, y0, w, h = [float(v) for v in arr[:4]]
                x1, y1 = x0 + max(w, 0.0), y0 + max(h, 0.0)
            return [x0, y0, x1, y1]
        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            xs = arr[..., 0].reshape(-1)
            ys = arr[..., 1].reshape(-1)
            return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    except Exception:  # noqa: BLE001 - best-effort evaluator parsing
        return None
    return None


def bbox_xyxy_from_xywh(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + max(w, 0.0), y + max(h, 0.0)]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def parse_predictions(result: Any) -> list[dict[str, Any]]:
    data = as_mapping(result)
    texts = [str(text) for text in (data.get("rec_texts") or [])]
    scores_raw = data.get("rec_scores") or []
    scores = [None if score is None else float(score) for score in scores_raw]
    boxes = data.get("rec_boxes")
    polys = data.get("rec_polys") or data.get("dt_polys") or []
    predictions: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        bbox = None
        if boxes is not None and idx < len(boxes):
            bbox = bbox_xyxy_from_any(boxes[idx])
        if bbox is None and idx < len(polys):
            bbox = bbox_xyxy_from_any(polys[idx])
        if bbox is None:
            continue
        predictions.append(
            {
                "index": idx,
                "text": text,
                "score": scores[idx] if idx < len(scores) else None,
                "bbox_xyxy": bbox,
            }
        )
    predictions.sort(key=lambda row: (row["bbox_xyxy"][1], row["bbox_xyxy"][0]))
    return predictions


def greedy_match(gt: list[dict[str, Any]], pred: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        g_box = bbox_xyxy_from_xywh(g["bbox_xywh"])
        for pi, p in enumerate(pred):
            iou = bbox_iou(g_box, p["bbox_xyxy"])
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append({"gt_index": gi, "pred_index": pi, "iou": float(iou)})
    return matches


def evaluate_image(record: dict[str, Any], predictions: list[dict[str, Any]], iou_threshold: float, profile: str) -> dict[str, Any]:
    gt = record["gt_words"]
    matches = greedy_match(gt, predictions, iou_threshold)
    matched_gt = {int(item["gt_index"]) for item in matches}
    matched_pred = {int(item["pred_index"]) for item in matches}
    gt_pairs = []
    matched_pairs = []
    exact_matches = 0
    for match in matches:
        ref = str(gt[match["gt_index"]]["text"])
        hyp = str(predictions[match["pred_index"]]["text"])
        gt_pairs.append((ref, hyp))
        matched_pairs.append((ref, hyp))
        if normalize_text(ref, profile) == normalize_text(hyp, profile):
            exact_matches += 1
    for gi, g in enumerate(gt):
        if gi not in matched_gt:
            gt_pairs.append((str(g["text"]), ""))
    gt_metrics = evaluate_pairs(gt_pairs, profile).to_dict()
    matched_metrics = evaluate_pairs(matched_pairs, profile).to_dict() if matched_pairs else None
    pred_count = len(predictions)
    gt_count = len(gt)
    false_positive_count = pred_count - len(matched_pred)
    exact_precision = None if pred_count == 0 else exact_matches / pred_count
    exact_recall = None if gt_count == 0 else exact_matches / gt_count
    if exact_precision is None or exact_recall is None or (exact_precision + exact_recall) == 0:
        exact_hmean = None
    else:
        exact_hmean = 2 * exact_precision * exact_recall / (exact_precision + exact_recall)
    return {
        "gt_words": gt_count,
        "pred_words": pred_count,
        "matched_words": len(matches),
        "false_positive_words": false_positive_count,
        "exact_matches": exact_matches,
        "detection_recall": None if gt_count == 0 else len(matches) / gt_count,
        "detection_precision": None if pred_count == 0 else len(matches) / pred_count,
        "exact_recall": exact_recall,
        "exact_precision": exact_precision,
        "exact_hmean": exact_hmean,
        "gt_coverage_metrics": gt_metrics,
        "matched_only_metrics": matched_metrics,
        "matches": matches,
    }


def aggregate(rows: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    gt_words = sum(int(row["metrics"]["gt_words"]) for row in rows)
    pred_words = sum(int(row["metrics"]["pred_words"]) for row in rows)
    matched_words = sum(int(row["metrics"]["matched_words"]) for row in rows)
    false_positive_words = sum(int(row["metrics"]["false_positive_words"]) for row in rows)
    exact_matches = sum(int(row["metrics"]["exact_matches"]) for row in rows)
    pairs = []
    matched_pairs = []
    for row in rows:
        matched_pred = {int(item["pred_index"]) for item in row["metrics"]["matches"]}
        gt_to_pred = {int(item["gt_index"]): int(item["pred_index"]) for item in row["metrics"]["matches"]}
        for gi, gt in enumerate(row["gt_words"]):
            hyp = row["predictions"][gt_to_pred[gi]]["text"] if gi in gt_to_pred else ""
            pairs.append((gt["text"], hyp))
        for item in row["metrics"]["matches"]:
            matched_pairs.append(
                (
                    row["gt_words"][int(item["gt_index"])]["text"],
                    row["predictions"][int(item["pred_index"])]["text"],
                )
            )
        # False-positive texts are not folded into CER; they are represented in precision counts.
        _ = matched_pred
    gt_metrics = evaluate_pairs(pairs, profile).to_dict()
    matched_metrics = evaluate_pairs(matched_pairs, profile).to_dict() if matched_pairs else None
    exact_precision = None if pred_words == 0 else exact_matches / pred_words
    exact_recall = None if gt_words == 0 else exact_matches / gt_words
    if exact_precision is None or exact_recall is None or (exact_precision + exact_recall) == 0:
        exact_hmean = None
    else:
        exact_hmean = 2 * exact_precision * exact_recall / (exact_precision + exact_recall)
    return {
        "images": len(rows),
        "gt_words": gt_words,
        "pred_words": pred_words,
        "matched_words": matched_words,
        "false_positive_words": false_positive_words,
        "exact_matches": exact_matches,
        "zero_prediction_images": sum(1 for row in rows if int(row["metrics"]["pred_words"]) == 0),
        "mean_infer_seconds": sum(float(row["infer_seconds"]) for row in rows) / max(len(rows), 1),
        "detection_recall": None if gt_words == 0 else matched_words / gt_words,
        "detection_precision": None if pred_words == 0 else matched_words / pred_words,
        "exact_recall": exact_recall,
        "exact_precision": exact_precision,
        "exact_hmean": exact_hmean,
        "gt_coverage_metrics": gt_metrics,
        "matched_only_metrics": matched_metrics,
    }


def hash_model_dir(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        return []
    return [
        {"path": str(file.relative_to(path)), "size_bytes": file.stat().st_size, "sha256": sha256_file(file)}
        for file in sorted(item for item in path.rglob("*") if item.is_file() and ".cache" not in item.parts)
    ]


def maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any]) -> str | None:
    if args.no_wandb:
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=["ocr", "paddleocr", "heldout", "e2e", "textocr", "full-image"],
        config={
            "annotation_json": str(args.annotation_json),
            "image_root": str(args.image_root),
            "max_images": args.max_images,
            "iou_threshold": args.iou_threshold,
            "profile": args.profile,
            "text_det_limit_side_len": args.text_det_limit_side_len,
            "text_det_limit_type": args.text_det_limit_type,
        },
    )
    metrics = summary["aggregate"]
    wandb.log(
        {
            "fullimage/gt_words": metrics["gt_words"],
            "fullimage/pred_words": metrics["pred_words"],
            "fullimage/matched_words": metrics["matched_words"],
            "fullimage/detection_recall": metrics["detection_recall"],
            "fullimage/detection_precision": metrics["detection_precision"],
            "fullimage/exact_recall": metrics["exact_recall"],
            "fullimage/exact_precision": metrics["exact_precision"],
            "fullimage/exact_hmean": metrics["exact_hmean"],
            "fullimage/gt_coverage_cer": metrics["gt_coverage_metrics"]["cer_micro"],
            "fullimage/mean_infer_seconds": metrics["mean_infer_seconds"],
        }
    )
    run_id = run.id
    run.finish()
    return run_id


def write_report(path: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# PaddleOCR TextOCR Full-Image Smoke",
        "",
        "This is an O002 evaluation-protocol diagnostic, not a publication benchmark. It uses local TextOCR train annotations only, no dataset download or image copying, and an axis-aligned IoU 0.5 matcher.",
        "",
        "## Aggregate",
        "",
        f"- images: {agg['images']}",
        f"- GT words: {agg['gt_words']}",
        f"- predicted words: {agg['pred_words']}",
        f"- matched words: {agg['matched_words']}",
        f"- false-positive words: {agg['false_positive_words']}",
        f"- detection recall / precision: {agg['detection_recall']:.6f} / {agg['detection_precision']:.6f}",
        f"- exact recall / precision / hmean: {agg['exact_recall']:.6f} / {agg['exact_precision']:.6f} / {agg['exact_hmean']:.6f}",
        f"- GT-coverage CER: {agg['gt_coverage_metrics']['cer_micro']:.6f}",
        f"- matched-only CER: {agg['matched_only_metrics']['cer_micro']:.6f}" if agg["matched_only_metrics"] else "- matched-only CER: n/a",
        f"- zero-prediction images: {agg['zero_prediction_images']}",
        f"- mean inference seconds: {agg['mean_infer_seconds']:.4f}",
        "",
        "## Interpretation",
        "",
        "Full-image detector+recognizer behavior is much stricter than oracle word-crop recognition. This smoke establishes the local matcher/logging path and quantifies detector coverage on a small fixed sample; it should not be used for model selection or final claims until dataset license and official-metric protocol are frozen.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-json", type=Path, default=Path("/dpl/veritext_raw/textocr/TextOCR_0.1_train.json"))
    parser.add_argument("--image-root", type=Path, default=Path("/dpl/veritext_raw/textocr/train_images"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--profile", default="unicode_strict_v1")
    parser.add_argument("--text-detection-model-name", default="PP-OCRv6_medium_det")
    parser.add_argument("--text-detection-model-dir", type=Path)
    parser.add_argument("--text-recognition-model-name", default="PP-OCRv6_medium_rec")
    parser.add_argument("--text-recognition-model-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--enable-mkldnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--text-recognition-batch-size", type=int, default=1)
    parser.add_argument("--text-det-limit-side-len", type=int, default=960)
    parser.add_argument("--text-det-limit-type", default="max")
    parser.add_argument("--disable-model-source-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oscar-lic")
    parser.add_argument("--wandb-run-name", default="paddleocr-textocr-fullimage-smoke-2026-06-26")
    args = parser.parse_args()

    if args.disable_model_source_check:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    records = load_textocr_records(
        args.annotation_json,
        args.image_root,
        max_images=args.max_images,
        min_words=args.min_words,
        max_words=args.max_words,
        seed=args.seed,
    )
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")

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

    rows = []
    for record in records:
        image_path = Path(record["image_path"])
        start = perf_counter()
        result_list = ocr.predict(str(image_path))
        infer_seconds = perf_counter() - start
        predictions = parse_predictions(result_list[0]) if result_list else []
        metrics = evaluate_image(record, predictions, args.iou_threshold, args.profile)
        rows.append(
            {
                "dataset": "textocr",
                "image_id": record["image_id"],
                "image": str(image_path),
                "image_sha256": sha256_file(image_path),
                "width": record["width"],
                "height": record["height"],
                "gt_words": record["gt_words"],
                "predictions": predictions,
                "metrics": metrics,
                "infer_seconds": infer_seconds,
            }
        )

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
        "description": "Held-out PaddleOCR full-image TextOCR smoke with simple axis-aligned IoU matching.",
        "validity": "diagnostic_smoke_not_publication_benchmark",
        "annotation_json": str(args.annotation_json),
        "annotation_json_sha256": sha256_file(args.annotation_json),
        "split_output": str(args.split_output),
        "split_output_sha256": sha256_file(args.split_output),
        "matcher": {"type": "axis_aligned_bbox_iou_greedy", "iou_threshold": args.iou_threshold},
        "profile": args.profile,
        "samples": len(rows),
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
        "aggregate": aggregate(rows, args.profile),
    }
    summary["wandb_run_id"] = maybe_log_wandb(args, summary)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    results_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary["results_path"] = str(results_path)
    summary["results_sha256"] = sha256_file(results_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(args.report, summary)

    print(json.dumps({"summary": str(summary_path), "results": str(results_path), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
