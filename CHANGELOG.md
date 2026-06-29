# Changelog

## 0.2.0-research-pass — 2026-06-25

- Set up working venvs, external code pins, dataset links and W&B logging for OSCAR-LIC research.
- Acquired corrected MLIC++ lambda 0.025 checkpoint, recorded SHA256, and verified real compress/decompress smoke with complete stream-byte bpp.
- Built Eval100/Eval300 OCR utility splits, PARSeq/Tesseract/TrOCR smoke evaluators, actual `.oscr` accounting, and compact latent/codebook candidate diagnostics.
- Promoted the current learned frozen-base assignment baseline: pooled image-diff group gate over corrected MLIC++, actual mean bpp `3.849649`, PARSeq unchanged, Tesseract `-8/-8` edit characters versus nearest K64 assignment.
- Added tensor, latent-context and structured latent-window assignment reranker artifacts and documented negative controls where high AUC/AP did not improve policy-level OCR.
- Added the next group/no-op assignment config and 12h reproducibility manifest for checkpoint/artifact/W&B/validation tracking.

## 0.1.0-bootstrap — 2026-06-24

- Created self-bootstrapping research repository.
- Added current LIC/ICM/OCR-aware/generative/RAW/robustness survey.
- Added OSCAR-LIC method, utility formulation and experiment gates.
- Added baseline, repository and dataset registries.
- Added evaluation, bitstream and reproducibility protocols.
- Added executable validation and metric utilities with tests.
