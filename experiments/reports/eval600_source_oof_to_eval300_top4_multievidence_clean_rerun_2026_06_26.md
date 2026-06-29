# Eval600 Source-OOF to Eval300 Top-4 Multi-Evidence Clean Rerun

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N046/N047 were invalid because the multi-evidence feature cache contained one all-NaN feature column. This corrected run asks:

> After sanitizing the non-finite feature column, do summary-level multi-evidence features and hard-positive weighting produce a useful top-4 selector?

## Setup

- Input feature cache: N046 multi-evidence cache
- Sanitization: replace NaN/inf with `0.0`
- Sanitization audit: `4384` NaN values in feature column `200`, no inf values
- Clean feature cache: `eval600_source_oof_to_eval300_top4_multievidence_features_clean_2026_06_26.npz`
- Candidate set: nearest/no-op plus union top-4 from source-OOF oracle-choice and safe-improve scores
- Split: train all non-Eval300-seed1 groups, validate Eval300 seed1
- Train groups: `70` oracle-change groups and `403` no-headroom groups

Two valid clean variants were run:

- `standard`: standard group CE, no extra margins
- `balanced`: oracle-change CE weight `8`, positive margin weight `1`, no-headroom and harmful margins `0.2`

## Results

Means are over 3 selector seeds.

| Policy / variant | Changed groups | Tesseract delta | Tesseract worsen groups | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| shortlist oracle | 15.00 | -20.00 | 0.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| clean standard argmax | 3.67 | +1.67 | 1.67 | 3.67 | 3.67 | 0.00 | 16.00 | 0.00 |
| clean standard train-tuned margin | 3.67 | +1.67 | 1.67 | 3.67 | 3.67 | 0.00 | 16.00 | 0.00 |
| clean balanced argmax | 6.00 | +1.33 | 2.00 | 5.33 | 5.33 | 0.67 | 14.67 | 0.00 |
| clean balanced train-tuned margin | 6.00 | +1.33 | 2.00 | 5.33 | 5.33 | 0.67 | 14.67 | 0.00 |

## Interpretation

This is the corrected valid negative result for N046/N047. Sanitizing the NaN column makes the selector move away from nearest/no-op, but the selections are mostly false/wrong changes and worsen Tesseract. Hard-positive weighting recovers a small amount of exact oracle signal, but false/wrong changes dominate and aggregate Tesseract is still worse than nearest.

The useful conclusion is sharper than the invalid no-op runs: summary-level multi-evidence features contain some signal, but the flat MLP selector cannot separate safe hard positives from harmful non-nearest changes under Eval600-to-Eval300 transfer. The next attempt should preserve structured local latent/code-token information or use an OCR/text-specific verifier, not add more scalar summaries or simple loss weights.

## Next Action

Do not promote this selector. Do not run counted `.oscr` streams. Retire the summary-feature top-4 MLP family for now; move to structured token/window modeling or first-stage candidate scoring changes.

W&B offline runs: standard `x0dbl88o`, balanced `pr68eztu`.

## Files

- Sanitizer: `scripts/sanitize_feature_cache.py`
- Training script: `scripts/train_assignment_topk_balanced_selector.py`
- Sanitization summary: `experiments/results/eval600_source_oof_to_eval300_top4_multievidence_features_clean_summary_2026_06_26.json`
- Standard result: `experiments/results/eval600_source_oof_to_eval300_top4_multievidence_selector_clean_standard_2026_06_26.json`
- Balanced result: `experiments/results/eval600_source_oof_to_eval300_top4_multievidence_selector_clean_balanced_2026_06_26.json`
- Clean feature cache: `experiments/features/eval600_source_oof_to_eval300_top4_multievidence_features_clean_2026_06_26.npz`
