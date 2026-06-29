# Eval600 Source-OOF to Eval300 Top-4 PARSeq Encoder Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream selector diagnostic; no counted `.oscr` promotion stream

## Question

N051 and N052 left the top-4 assignment substrate in an interesting state:

> The shortlist oracle is still strong at unchanged code-index payload size, but scalar score/rank, image-diff, generic patch, AlexNet and summary features do not select the useful rows safely.

This diagnostic asks whether direct OCR-teacher representation can explain the missing selector signal. PARSeq is the registered training teacher, so these features are allowed as teacher-evidence diagnostics. They are not held-out evaluation signals and are not deployable unless the OCR encoder cost and protocol role are explicitly accounted.

## Setup

- Candidate set: nearest/no-op plus the union top-4 candidates from source-OOF oracle-choice and safe-improve score files.
- Train/validation split: train on non-Eval300-seed1 groups, validate on Eval300 seed1.
- Rows: `4384`.
- Feature cache:
  - base selector features;
  - PARSeq encoder memory pooled by mean/std/max/min into `1536` dimensions;
  - candidate-minus-nearest, absolute difference, norm/cosine/mean/max relative features;
  - final feature matrix shape: `(4384, 4755)`.
- PARSeq checkpoint: `/root/.cache/torch/hub/checkpoints/parseq-bb5792a6.pt`.
- PARSeq checkpoint SHA256: `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.
- Device: CUDA.

Two selector variants were run over the same PARSeq encoder feature cache:

- `ce`: standard group CE, no additional margins.
- `balanced`: oracle-change group weight `8`, positive margin weight `1`, no-headroom and harmful margins `0.2`.

## Results

Means are over 3 selector seeds on Eval300 seed1 val75.

| Policy / variant | Changed groups | Tesseract delta | Tesseract worsen groups | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| shortlist oracle | 15.00 | -20.00 | 0.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| PARSeq-encoder CE argmax | 0.67 | 0.00 | 0.00 | 0.67 | 0.67 | 0.00 | 16.33 | 0.00 |
| PARSeq-encoder CE train-tuned margin | 0.67 | 0.00 | 0.00 | 0.67 | 0.67 | 0.00 | 16.33 | 0.00 |
| PARSeq-encoder balanced argmax | 4.33 | +1.67 | 1.33 | 4.33 | 4.33 | 0.00 | 15.67 | 0.00 |
| PARSeq-encoder balanced train-tuned margin | 4.33 | +1.67 | 1.33 | 4.33 | 4.33 | 0.00 | 15.67 | 0.00 |

Per-seed argmax behavior is also revealing:

- CE: selector seeds `0/1/2` change `1/0/1` groups, recover `0` exact oracle changes and produce no net Tesseract gain.
- Balanced: selector seeds `0/1/2` change `2/9/2` groups, still recover `0` exact oracle changes, and seed `1` worsens Tesseract by `+4` edit characters.

## Interpretation

This is a valid negative diagnostic for direct PARSeq encoder feature injection into the top-4 selector.

The teacher features do not close the policy gap. The standard CE model effectively collapses to no-op and misses almost all shortlist headroom. The balanced variant forces movement, but the movement is harmful: it selects false/wrong non-nearest changes and worsens Tesseract while preserving PARSeq only because the selected rows are not PARSeq-sensitive.

The result should not be read as "OCR teacher evidence is useless." It says the simplest direct encoder-memory feature cache, flattened into the existing MLP selector, is not a promotable route. The better use of teacher evidence is likely candidate-local auxiliary regularization or a distilled OCR/text-specific verifier with explicit no-op precision, not inference-time PARSeq feature concatenation.

## Next Action

Do not promote this selector. Do not run counted `.oscr` streams for it. Record it as N053 and avoid direct PARSeq encoder feature selectors unless the experiment changes materially: for example, a structured OCR/text verifier, auxiliary distillation into deployable features, or a policy objective with source/domain-disjoint calibration.

W&B offline run: `mw4lraiv`.

## Files

- Feature extractor: `scripts/extract_parseq_topk_feature_cache.py`
- Selector trainer: `scripts/train_assignment_topk_balanced_selector.py`
- Feature cache: `experiments/features/eval600_source_oof_to_eval300_top4_parseq_encoder_features_2026_06_26.npz`
- CE result: `experiments/results/eval600_source_oof_to_eval300_top4_parseq_encoder_selector_ce_2026_06_26.json`
- Balanced result: `experiments/results/eval600_source_oof_to_eval300_top4_parseq_encoder_selector_balanced_2026_06_26.json`
