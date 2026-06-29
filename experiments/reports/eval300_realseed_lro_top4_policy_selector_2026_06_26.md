# Eval300 Leave-Real-Seed-Out Top-4 Policy Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

The score-seed-2 top-4 shortlist oracle is now validated through actual counted `.oscr` streams, but learned selectors trained on Eval600 score outputs fail on Eval300. This diagnostic asks a narrower question:

> If calibration labels come from the same Eval300 distribution but a different `real_seed`, can the current top-4 policy selector recover the shortlist oracle?

This is not a promotion result. It uses Eval300 oracle labels for calibration and evaluates only table-level assignment choices. No new actual `.oscr` stream is claimed.

## Setup

Rows are from the hard-negative-balanced Eval600 + Eval300 seed1 assignment table with image-diff features. The candidate set per group is:

- nearest/no-op row;
- top-4 rows by N034 `oracle_choice` diff-crop score;
- top-4 rows by N035 `safe_improve` diff-crop score;
- duplicate candidates merged.

The corrected run uses only score model seed `2`, matching the shortlist oracle that was validated in actual `.oscr`.

Validation splits:

- validate `real_seed=0`, train on Eval300 `real_seed=1,2`;
- validate `real_seed=1`, train on Eval300 `real_seed=0,2`;
- validate `real_seed=2`, train on Eval300 `real_seed=0,1`.

Selector seeds: `0,1,2`.

## Invalid First Run

`experiments/results/eval300_realseed_lro_top4_policy_selector_scoreseed2_2026_06_26.json` is retained as an invalid diagnostic artifact. The script initially used `argparse(action="append", default=[0,1,2])`, so passing `--score-model-seed 2` appended to the default instead of replacing it. The corrected script sets append defaults to `None`.

## Corrected Results

Means are over 3 validation real-seeds x 3 selector seeds.

| Policy | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 5.67 | 0.00 |
| shortlist oracle | 5.00 | -6.67 | 0.00 | 0.00 | 5.00 | 0.67 | 0.00 |
| learned argmax | 1.78 | +0.33 | 1.67 | 1.67 | 0.11 | 5.11 | 0.00 |
| train-tuned margin | 1.78 | +0.33 | 1.67 | 1.67 | 0.11 | 5.11 | 0.00 |

Per split:

| Validation split | Shortlist oracle Tesseract delta | Learned argmax Tesseract delta | Learned exact changes | Learned false changes | Learned missed changes |
|---|---:|---:|---:|---:|---:|
| real_seed 0 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 |
| real_seed 1 | -7.00 | +0.67 | 0.33 | 2.67 | 4.67 |
| real_seed 2 | -13.00 | +0.33 | 0.00 | 1.67 | 10.67 |

## Interpretation

This is a valid negative result for the current top-4 MLP selector even under same-Eval300 leave-real-seed-out calibration.

The important conclusion is sharper than N037:

- the top-4 shortlist remains a strong substrate;
- the selector failure is not explained only by Eval600-to-Eval300 distribution shift;
- with only neighboring Eval300 calibration labels, the current score-feature MLP still cannot identify hard-positive code changes and keeps opening false non-nearest changes;
- train-tuned margin does not improve over argmax, so the current logit margin is not a reliable abstention signal.

The next path should avoid another shallow selector over the same score features. More promising directions are:

- out-of-fold/cross-fit first-stage scores so selector calibration is not trained on in-sample score distributions;
- candidate-local deployable evidence beyond scalar score/rank features;
- explicit uncertainty/no-op modeling at the policy level;
- source/domain-disjoint calibration with a high-precision false-change objective.

W&B offline run: `vtyajbnv`

## Files

- Corrected result JSON: `experiments/results/eval300_realseed_lro_top4_policy_selector_scoreseed2_corrected_2026_06_26.json`
- Invalid seed-mixed diagnostic JSON: `experiments/results/eval300_realseed_lro_top4_policy_selector_scoreseed2_2026_06_26.json`
- Script: `scripts/train_assignment_topk_policy_selector_split_diagnostic.py`
