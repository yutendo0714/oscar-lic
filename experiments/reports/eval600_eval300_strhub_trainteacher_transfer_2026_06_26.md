# Eval600/Eval300 STRHub Train-Teacher Transfer Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream assignment diagnostic; no counted `.oscr` promotion stream

## Question

PARSeq-only assignment labels are too sparse on the current top-8 substrate. This diagnostic locks two additional train-teacher OCR families, CRNN and ABINet from the STRHub/PARSeq model hub, and asks:

> Can train-teacher confidence or edit-distance signals identify the Tesseract-useful assignment code changes without using held-out OCR at inference or for training labels?

Tesseract remains held-out in this analysis. Policies named `heldout_tesseract_oracle` or `trainteacher_no_worse_heldout_oracle` are upper-bound diagnostics only and must not be used for training, feature generation, threshold tuning, or selection.

## Setup

- Assignment table: `experiments/results/eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`.
- Train-like split: 473 Eval600 groups from the hard-negative-balanced mixed table.
- Validation split: fixed Eval300 seed1 val75 groups.
- OCR runners:
  - CRNN train teacher, checkpoint SHA `679d0e315af22d813c6d5a86287ee5705a9a8dfe37378349359442b8b8953264`.
  - ABINet train teacher, checkpoint SHA `1d1e373e6d08bde09500a015436b87ae0f4c9bbd844c8c4fc7a8d9e3628325a7`.
- Normalization: `unicode_strict_v1` for primary transfer analysis.

Original Eval300 recognizer-only upper bounds after locking:

| Model | Unicode CER | Unicode exact | Latin CER | Latin exact |
|---|---:|---:|---:|---:|
| CRNN | 0.1101 | 0.7267 | 0.0938 | 0.7567 |
| ABINet | 0.0634 | 0.8633 | 0.0452 | 0.8933 |

## Signal Sparsity

| Split | Groups | CRNN improve groups | ABINet improve groups | Either train teacher improve groups | Tesseract oracle-change groups |
|---|---:|---:|---:|---:|---:|
| Eval600 train-like | 473 | 4 | 6 | 10 | 95 |
| Eval300 seed1 val75 | 75 | 1 | 0 | 1 | 17 |
| All | 548 | 5 | 6 | 11 | 112 |

Direct edit-distance improvements for CRNN/ABINet are too sparse to be the main assignment label. They are useful as model-diversity checks, but not as dense supervision for the current selector.

## Validation Results

Policies below are applied to Eval300 seed1 val75. Tesseract deltas are edit-character deltas versus nearest; lower is better.

| Policy | Changed | Exact oracle | False changes | Tesseract delta | Tesseract worsen groups | PARSeq delta | CRNN delta | ABINet delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| heldout Tesseract oracle | 17 | 17 | 0 | -24 | 0 | 0 | 0 | 0 |
| CRNN confidence | 67 | 0 | 54 | +2 | 6 | 0 | -1 | 0 |
| ABINet confidence | 68 | 3 | 52 | +3 | 9 | 0 | 0 | 0 |
| mean confidence | 69 | 3 | 52 | +1 | 7 | 0 | 0 | 0 |
| CRNN distance+confidence | 67 | 0 | 54 | +2 | 6 | 0 | -1 | 0 |
| ABINet distance+confidence | 68 | 3 | 52 | +3 | 9 | 0 | 0 | 0 |
| mean distance+confidence | 69 | 3 | 52 | +1 | 7 | 0 | -1 | 0 |
| train-teacher no-worse held-out oracle | 17 | 17 | 0 | -24 | 0 | 0 | 0 | 0 |

## Interpretation

CRNN and ABINet are now usable as locked train teachers, but their direct confidence heuristics are badly calibrated for OSCAR assignment selection. They change almost every group, produce 52-54 false changes on the 75-group Eval300 validation split, and worsen held-out Tesseract by `+1` to `+3` edit characters despite preserving PARSeq.

The upper-bound `trainteacher_no_worse_heldout_oracle` row is informative but non-deployable: it shows that the Tesseract-useful code changes are not excluded by a CRNN/ABINet no-worse constraint, but it still uses held-out Tesseract oracle choice to decide which groups to change. This supports using train teachers as constraints or auxiliary terms, not as direct dense confidence selectors.

## Decision

This is N055, a valid negative result for direct CRNN/ABINet confidence or distance-confidence policies. Do not retry confidence argmax, mean-confidence ensembling, or direct train-teacher edit-improvement labels as the main assignment target on this substrate.

The next viable route should create a dense train-teacher loss or verifier that is independent of held-out OCR labels, for example candidate-level teacher NLL/margin against ground-truth text, synthetic perturbation labels, or a high-precision text-aware verifier whose model weights and inference cost are explicitly accounted for.

W&B offline run: `ccwg6z4j`.

## Files

- OCR runner: `scripts/run_strhub_ocr_smoke.py`
- Transfer analyzer: `scripts/analyze_strhub_teacher_transfer.py`
- CRNN original Eval300 OCR: `experiments/results/crnn_eval300_original_2026_06_26/summary.json`
- ABINet original Eval300 OCR: `experiments/results/abinet_eval300_original_2026_06_26/summary.json`
- CRNN candidate OCR: `experiments/results/crnn_eval600_balanced210_plus_eval300_seed1_top8_candidates_2026_06_26/results.jsonl`
- ABINet candidate OCR: `experiments/results/abinet_eval600_balanced210_plus_eval300_seed1_top8_candidates_2026_06_26/results.jsonl`
- Transfer summary: `experiments/results/eval600_eval300_strhub_trainteacher_transfer_2026_06_26.json`
