# Eval600 Source-OOF to Eval300 Top-4 Reward-Policy Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N041 showed that source-OOF score calibration alone does not rescue the learned top-4 selector. This diagnostic changes the second-stage objective:

> Keep the same nearest-plus-top-4 shortlist and OOF score inputs, but train the selector with train-side OCR edit reward/cost instead of only `label_assignment_oracle_choice` classification.

The intended fix is to suppress false non-nearest changes and recover hard positives by optimizing the policy-level OCR tradeoff more directly.

## Setup

Input substrate:

- assignment table: `eval600_hardneg_balanced210_plus_eval300_seed1_assignment_top8_table_imgdiff_2026_06_26.jsonl`;
- first-stage score rows: N041 source-modulo OOF Eval600 scores merged with external Eval300 scores;
- score model seed: `2`;
- shortlist: nearest/no-op plus union of top-4 oracle-choice and safe-improve scorer rows.

Reward:

- nearest/no-op reward is `0`;
- non-nearest reward starts as `-tesseract_delta_vs_nearest`;
- false/non-improving changes are penalized;
- PARSeq worsening is penalized heavily;
- Tesseract worsening receives an additional penalty.

Training:

- selector seeds `0,1,2`;
- 400 epochs, hidden dim `32`, dropout `0.2`;
- soft target distribution from reward, expected-reward term, positive margin, and nonpositive-vs-nearest margin.

## Results

Means are over 3 selector seeds on Eval300 seed1.

| Policy | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| label oracle | 15.00 | -20.00 | 0.00 | 0.00 | 15.00 | 2.00 | 0.00 |
| reward oracle | 16.00 | -21.00 | 1.00 | 1.00 | 15.00 | 1.00 | 0.00 |
| reward argmax | 14.67 | +4.00 | 14.33 | 14.33 | 0.33 | 12.33 | 0.00 |
| train-tuned safety margin | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 | 0.00 |
| train-tuned reward margin | 10.67 | +3.33 | 10.33 | 10.33 | 0.33 | 13.33 | 0.00 |

Reward-positive group counts:

- train: `83`;
- Eval300 seed1: `16`.

## Interpretation

This is a valid negative result for changing only the top-4 selector loss to an OCR reward/cost objective.

The useful part is the oracle gap:

- `reward_oracle` is slightly stronger than the existing label oracle (`-21` vs `-20` Tesseract edits);
- therefore the reward definition does not remove the headroom.

The failure is the learned policy:

- argmax becomes too aggressive and selects many false/wrong non-nearest rows;
- the original train-tuned safety margin collapses to no-op;
- the reward-tuned margin still transfers badly and worsens Tesseract.

This says the current feature evidence is still not separable enough for policy optimization to identify hard positives. The next branch should not keep changing scalar losses over the same feature set.

## Next Action

Do not promote or actual-OSCR this selector. Do not sweep small reward weights as the next primary path. The top-4 shortlist remains the validated upper-bound substrate, but the next useful change should alter representation or calibration more substantially:

- richer candidate-local code-effect evidence;
- uncertainty-aware abstention with calibrated confidence;
- source/domain-disjoint calibration data with new evidence;
- or a group-level verifier that sees more than scalar score/rank features.

W&B offline run: `s7wqdw5n`.

## Files

- Script: `scripts/train_assignment_topk_reward_policy_selector.py`
- Result: `experiments/results/eval600_source_oof_to_eval300_top4_reward_policy_selector_2026_06_26.json`
