# Next Experiment Plan

**Date:** 2026-06-25  
**Principle:** do not start large training until actual-bitstream baseline reproduction and oracle utility evidence are clean.

## G1: Base Codec Reproduction

Goal: produce actual compressed files and bpp for MLIC++ and CompressAI smoke baselines.

Steps:

1. Download corrected MLIC++ checkpoint at lambda 0.0250.
2. Record checkpoint URL, size, SHA256, architecture variant, and expected command.
3. Run MLIC++ test/compress/decompress on a tiny CLIC subset.
4. Compute bpp from complete produced files, not likelihood estimates.
5. Store one JSON result per image using `evaluation/REPORTING_SCHEMA.json`.
6. Log run to wandb with environment lock and external commit lock.

Exit criteria:

- decoded images are valid;
- complete-file bpp is available;
- output is deterministic for repeated decode;
- GPU is visible before running.

## G2: OCR Evaluation Foundation

Goal: lock one training teacher and one held-out evaluator.

Initial split:

- Train teacher: PARSeq.
- Held-out evaluator: Tesseract or PaddleOCR first, TrOCR later if dependency cost is acceptable.

Steps:

1. Download PARSeq pretrained checkpoint via the pinned repo mechanism or torch hub equivalent.
2. Hash checkpoint and tokenizer/charset configuration.
3. Run OCR upper bound on original images for a small text subset.
4. Store raw and normalized OCR outputs with model revision.
5. Never use held-out OCR logits/confidence for training or oracle labels.

## G3: Adapter And Text-Crop Smoke

Goal: create the base codec adapter and a crop-level evaluation loop.

Steps:

1. Wrap MLIC++ behind `BaseCodecAdapter` without modifying upstream files broadly.
2. Define `CodecResult` with actual total bytes, section bytes, encode/decode time, metadata, and image shape.
3. Build a small text-rich validation manifest from TextOCR/ICDAR/TotalText local data.
4. Verify original/reconstructed OCR metrics on compressed outputs.

## G4: Counterfactual Oracle Feasibility

Goal: test the central OSCAR-LIC hypothesis before large model training.

Candidate layout:

- start with latent spatial tiles and channel groups matching MLIC++ feature shape;
- keep candidate count small for exact greedy trajectories.

Controls:

- random allocation;
- uniform text-mask allocation;
- detector confidence allocation;
- base uncertainty/residual energy;
- cheap stroke/edge energy.

Measurements:

- delta OCR loss per actual incremental serialized bit;
- held-out OCR CER/WER after allocation;
- side-index overhead;
- oracle interaction variance.

Exit criteria:

- oracle beats controls at matched total bytes on held-out OCR;
- gains survive gate/index/header accounting;
- improvement is not confined to one synthetic domain.

## Training Direction After G4

If G4 passes:

1. Train a lightweight utility predictor with ranking and calibrated benefit heads.
2. Train conditional feature-residual enhancement stream with frozen base.
3. Add multi-teacher invariance and topology only if needed.
4. Jointly fine-tune a small base tail only after frozen-base gains are stable.

If G4 fails:

1. Recheck candidate layout and exact serialization overhead.
2. Test a simpler detector-mask residual stream as a negative/diagnostic result.
3. Do not scale a learned predictor on an uninformative oracle.

