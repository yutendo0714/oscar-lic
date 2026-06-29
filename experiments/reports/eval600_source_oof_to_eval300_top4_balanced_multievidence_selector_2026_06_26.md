# Eval600 Source-OOF to Eval300 Top-4 Balanced Multi-Evidence Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Invalidation Note

This run is invalid as a model result. It consumed the N046 multi-evidence feature cache before sanitization. That cache contained one all-NaN feature column (`200`, `4384` NaN values total), and all runs reported `train_margin=nan`.

The files are retained for audit, but the policy numbers below should not be interpreted as valid negative evidence. N048 sanitizes the feature cache and reruns the standard and balanced objectives as the corrected result.

## Question

N046 fused source-OOF scores, scalar image/code features, cached AlexNet embeddings and latent/codebook summaries, but the learned selector collapsed to nearest/no-op. This diagnostic asks:

> Is that collapse mainly caused by nearest-dominant label imbalance, and can hard-positive weighting make the same multi-evidence features recover useful top-4 oracle changes?

## Setup

- Feature cache: N046 multi-evidence features (`461` dimensions per row)
- Candidate set: nearest/no-op plus union top-4 from N041 oracle-choice and safe-improve source-OOF scores
- Split: train all non-Eval300-seed1 groups, validate Eval300 seed1
- Train groups: `70` oracle-change groups and `403` no-headroom/no-change groups
- Selector: same explicit-nearest top-k MLP, but with weighted group objective and positive oracle-vs-nearest margin

Two variants were run:

- `balanced`: oracle-change CE weight `8`, positive-margin weight `1`, no-headroom and harmful margins `0.2`
- `positive-only`: oracle-change CE weight `32`, positive-margin weight `2`, no no-headroom/harmful suppression

## Results

Means are over 3 selector seeds.

| Policy / variant | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| shortlist oracle | 15.00 | -20.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| balanced argmax | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| balanced train-tuned margin | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| positive-only argmax | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| positive-only train-tuned margin | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |

## Interpretation

This is an invalid diagnostic artifact because the feature matrix contained NaNs. The apparent no-op collapse is not reliable.

The next model should stop treating the top-4 selector as a flat summary-feature MLP. The remaining plausible paths are:

- keep richer local latent/code-token structure instead of summary statistics;
- use an OCR/text-specific verifier with explicit provenance and no test leakage;
- or improve the first-stage candidate generation/scoring so oracle codes are not dependent on a fragile post-hoc top-4 selector.

## Next Action

Do not promote or use this run for model conclusions. Use the corrected N048 clean-feature rerun instead.

W&B offline runs: balanced `sybz8479`, positive-only `n8b6dxc9`.

## Files

- Script: `scripts/train_assignment_topk_balanced_selector.py`
- Balanced result: `experiments/results/eval600_source_oof_to_eval300_top4_balanced_multievidence_selector_2026_06_26.json`
- Positive-only result: `experiments/results/eval600_source_oof_to_eval300_top4_balanced_multievidence_selector_posonly_2026_06_26.json`
- Feature cache: `experiments/features/eval600_source_oof_to_eval300_top4_multievidence_features_2026_06_26.npz`
