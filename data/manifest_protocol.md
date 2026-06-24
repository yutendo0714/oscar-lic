# Dataset Manifest Protocol

## Purpose

A dataset name is not a reproducible input. Every run uses a JSONL manifest whose order, split and hashes are frozen.

## Minimum image record

```json
{
  "sample_id": "dataset/split/000001",
  "dataset_id": "textocr",
  "split": "validation",
  "relative_path": "images/000001.jpg",
  "width": 1280,
  "height": 720,
  "sha256": "...",
  "annotations_path": "annotations/000001.json",
  "license_record_id": "..."
}
```

Do not put machine-specific absolute paths in the manifest. The dataset root is a local run parameter; the immutable relative path and hash establish identity.

## Annotation records

Each text region should preserve:

- region ID
- polygon and/or bbox
- raw transcription
- illegible/do-not-care flag
- script/language if known
- reading order/line grouping if known
- source annotation version

Normalization is performed at metric time, not destructively in the raw annotation manifest.

## Derived fields

Derived text height, contrast, blur, detector confidence and utility labels belong in versioned side manifests keyed by source image SHA256 and transform/config hash. Never overwrite source annotations.

## Split integrity

Generate and store:

- exact train/validation/test manifest hashes
- perceptual duplicate audit across splits
- provenance overlap audit across datasets
- OCR-pretraining overlap notes where known

The final paper reports dataset counts after all exclusions, not advertised download counts.
