# eval300_top8_oof_latent_proposal_risk_2026_06_26

N117 OOF shortlist with a latent/code/window non-nearest proposal model and separate N120-style risk abstention. Diagnostic only; no `.oscr` stream is exported.

W&B: `rlldzgi3`

## Proposal Seeds

| seed | loss | rank1 | rank<=4 | all proposals |
|---:|---:|---:|---:|---|
| 20260626 | 0.0000 | 4 | 13 | 4/17 exact, false 58, wrong 13, miss 0, T 2, P 0 |
| 20260627 | 0.0000 | 4 | 13 | 4/17 exact, false 58, wrong 13, miss 0, T -1, P 0 |
| 20260628 | 0.0000 | 4 | 10 | 4/17 exact, false 58, wrong 13, miss 0, T 1, P 0 |

## Ensemble Risk

| risk | budget0 | budget1 | budget2 | proposal rank1/rank<=4 |
|---|---|---|---|---:|
| risk_logistic_c05 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 4/13 |
| risk_hist_gradient | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 4/13 |
| risk_random_forest | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 0/17 exact, false 0, wrong 0, miss 17, T 0, P 0 | 4/13 |

## Guardrails

- Inference features use only deployable table scalars plus latent/code/window tensors listed in `feature_guard.used_npz_keys`.
- `teacher_parseq_*`, OCR deltas, OCR strings, references, predictions, and oracle labels are excluded from inference features.
- Risk thresholds are tuned on train groups and reported on validation groups.

## Interpretation

This is the first N117/N120-style proposal-risk diagnostic whose proposal stage sees candidate-local latent/code/window evidence. It is a valid next step after N122 because it does not sweep scalar sample weights or risk thresholds over the same feature family. It remains diagnostic-only unless validation table metrics beat the current actual-bitstream floor before any counted `.oscr` export.
