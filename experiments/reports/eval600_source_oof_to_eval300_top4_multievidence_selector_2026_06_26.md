# Eval600 Source-OOF to Eval300 Top-4 Multi-Evidence Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Invalidation Note

This run is invalid as a model result. Post-run audit found that the generated feature cache contained one non-finite feature column: column `200` had `4384` NaN values, one for every row. The source was the upstream latent-window artifact's `baseline_score` array, which is all NaN in this merged artifact. All selector runs also had `train_margin=nan`.

The files are retained for audit, but the policy numbers below should not be interpreted as valid negative evidence. N048 sanitizes the feature cache and reruns the standard and balanced objectives as the corrected result.

## Question

N041-N045 show that scalar score/rank features, reward losses, strict budgets, scratch diff-crop CNNs and fixed AlexNet patch embeddings do not recover the top-4 oracle under Eval600-to-Eval300 transfer. This diagnostic asks:

> Does fusing weak evidence families - source-OOF scores, scalar image/code features, fixed AlexNet patch embeddings, and OSCAR-LIC latent/codebook summaries - produce a safer high-precision selector?

## Setup

- Table: `eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`
- Scores: N041 source-OOF mixed oracle-choice and safe-improve score files
- Candidate set: nearest/no-op plus union top-4 from both score files
- Latent/code evidence: 70 summary features from the merged top8 latent-window artifact
- Visual evidence: cached fixed AlexNet diff-crop features from N045
- Selector: same explicit-nearest top-k MLP selector
- Split: train all non-Eval300-seed1 groups, validate Eval300 seed1
- Seeds: selector seeds `0,1,2`

Two variants were run:

- `safe`: harmful-margin weight `1.5`
- `recall`: harmful-margin weight `0.0`

The latent/code join had `0` missing rows over all 4384 table rows.

## Results

Means are over 3 selector seeds.

| Policy / variant | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| shortlist oracle | 15.00 | -20.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| multi-evidence safe argmax | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| multi-evidence safe train-tuned margin | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| multi-evidence recall argmax | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| multi-evidence recall train-tuned margin | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |

## Interpretation

This is an invalid diagnostic artifact because the feature matrix contained NaNs. The apparent no-op collapse is not reliable.

The important conclusion is that summary-level feature fusion is still not a candidate-local glyph/code-effect model. The added evidence is too compressed or too weak to lift hard positives above the nearest anchor under external calibration. Future work should not add more summary scalars to this selector family. The next useful representation needs to preserve richer local structure, for example a jointly trained latent/code-token verifier or OCR/text-specific verifier, and should address the nearest-dominant label imbalance explicitly rather than relying on a standard listwise CE over heavily no-op-skewed groups.

## Next Action

Do not promote or use this run for model conclusions. Use the corrected N048 clean-feature rerun instead.

W&B offline runs: safe `chn5v4td`, recall `mkjhaz2x`.

## Files

- Script: `scripts/train_assignment_topk_multievidence_selector.py`
- Safe result: `experiments/results/eval600_source_oof_to_eval300_top4_multievidence_selector_2026_06_26.json`
- Recall result: `experiments/results/eval600_source_oof_to_eval300_top4_multievidence_selector_recall_2026_06_26.json`
- Feature cache: `experiments/features/eval600_source_oof_to_eval300_top4_multievidence_features_2026_06_26.npz`
