# Eval600/Eval300 STRHub Source-Pseudo-Label Loss Transfer Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream assignment diagnostic; no counted `.oscr` promotion stream

## Question

N056 used ground-truth text as the teacher-loss target, which is useful for training diagnostics but unavailable at encoder inference. This follow-up tests a deployable-style train-teacher verifier:

> If CRNN/ABINet read the source image first, can candidate reconstructions be selected by preserving that source OCR pseudo-label?

This uses only train-teacher OCR at the encoder side. Held-out Tesseract is used only for evaluation and upper-bound rows.

## Setup

- Source pseudo-label split: 548 unique source groups from the hard-negative-balanced Eval600 plus Eval300 seed1 top8 assignment table.
- Pseudo-labels:
  - CRNN source prediction for CRNN candidate loss.
  - ABINet source prediction for ABINet candidate loss.
- Candidate loss:
  - CRNN CTC loss against CRNN source prediction.
  - ABINet cross-entropy loss against ABINet source prediction.
- Empty source predictions are allowed and treated as valid pseudo-labels for this diagnostic.
- W&B runs:
  - CRNN source OCR: `4cx9p4rf`
  - ABINet source OCR: `l24e394t`
  - CRNN pseudo-loss extraction: `vcrn793q`
  - ABINet pseudo-loss extraction: `bu151m61`
  - Aggregate analysis: `smxrubsn`

## Signal Density

| Split | Groups | CRNN pseudo-loss improve groups | ABINet pseudo-loss improve groups | Either improve groups | Both teacher loss no-worse groups | Tesseract oracle-change groups |
|---|---:|---:|---:|---:|---:|---:|
| Eval600 train-like | 473 | 430 | 433 | 466 | 359 | 95 |
| Eval300 seed1 val75 | 75 | 66 | 66 | 75 | 48 | 17 |
| All | 548 | 496 | 499 | 541 | 407 | 112 |

Source pseudo-label loss is still much denser than true held-out OCR utility.

## Validation Results

Policies below are applied to Eval300 seed1 val75. Tesseract deltas are edit-character deltas versus nearest; lower is better.

| Policy | Changed | Exact oracle | False changes | Tesseract delta | Tesseract worsen groups | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|
| nearest | 0 | 0 | 0 | 0 | 0 | 0 |
| held-out Tesseract oracle | 17 | 17 | 0 | -24 | 0 | 0 |
| CRNN pseudo-loss mean argmax | 66 | 2 | 51 | -1 | 5 | 0 |
| ABINet pseudo-loss mean argmax | 66 | 5 | 50 | -1 | 8 | 0 |
| mean pseudo-loss mean argmax | 65 | 3 | 50 | -3 | 6 | 0 |
| mean pseudo-loss sum argmax | 64 | 3 | 49 | -2 | 6 | 0 |
| pseudo-loss no-worse held-out oracle | 4 | 4 | 0 | -7 | 0 | 0 |

The last row is a non-deployable upper bound because it uses held-out Tesseract oracle choice and only applies train-teacher pseudo-loss no-worse as a filter. It reaches `-7`, still below the current actual-bitstream learned baseline and far below the top-4 shortlist oracle.

## Interpretation

Source OCR pseudo-labeling is more deployable than ground-truth teacher loss, but it does not solve the selector problem. Direct pseudo-loss argmax remains dominated by false changes. The no-worse upper bound is slightly better than N056 (`-7` vs `-6`) but still insufficient: source self-consistency preserves only a small subset of the held-out Tesseract oracle headroom.

This suggests that a train-teacher OCR verifier alone is too conservative for the useful OSCAR-LIC assignment changes. It may serve as an auxiliary safety term, but the main selector needs candidate-local code-effect evidence or a learned abstention policy that is not reducible to "preserve source OCR pseudo-label."

## Decision

This is N057, a valid negative result for direct source-pseudo-label CRNN/ABINet loss argmax and hard pseudo-loss no-worse filtering as assignment policy rules.

The infrastructure is useful: `scripts/build_strhub_pseudolabel_splits.py` now creates source-image OCR pseudo-label tables that can be reused for future costed encoder-side verifier diagnostics.

## Files

- Pseudo-label builder: `scripts/build_strhub_pseudolabel_splits.py`
- Teacher-loss runner: `scripts/run_strhub_teacher_loss.py`
- Transfer analyzer: `scripts/analyze_strhub_teacher_loss_transfer.py`
- Source split: `data/splits/eval600_eval300_seed1_assignment_top8_source_images_for_strhub_pseudolabel_2026_06_26.jsonl`
- CRNN pseudo-label table: `experiments/results/eval600_eval300_seed1_assignment_top8_crnn_source_pseudolabel_table_2026_06_26.jsonl`
- ABINet pseudo-label table: `experiments/results/eval600_eval300_seed1_assignment_top8_abinet_source_pseudolabel_table_2026_06_26.jsonl`
- Aggregate result: `experiments/results/eval600_eval300_strhub_source_pseudolabel_loss_transfer_2026_06_26.json`
