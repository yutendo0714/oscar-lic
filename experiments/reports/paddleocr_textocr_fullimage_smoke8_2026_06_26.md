# PaddleOCR TextOCR Full-Image Smoke

This is an O002 evaluation-protocol diagnostic, not a publication benchmark. It uses local TextOCR train annotations only, no dataset download or image copying, and an axis-aligned IoU 0.5 matcher.

## Aggregate

- images: 8
- GT words: 68
- predicted words: 30
- matched words: 22
- false-positive words: 8
- detection recall / precision: 0.323529 / 0.733333
- exact recall / precision / hmean: 0.191176 / 0.433333 / 0.265306
- GT-coverage CER: 0.645756
- matched-only CER: 0.179487
- zero-prediction images: 0
- mean inference seconds: 2.9127

## Interpretation

Full-image detector+recognizer behavior is much stricter than oracle word-crop recognition. This smoke establishes the local matcher/logging path and quantifies detector coverage on a small fixed sample; it should not be used for model selection or final claims until dataset license and official-metric protocol are frozen.
