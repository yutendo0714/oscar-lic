# eval600_to_eval300_top8_topk_prior_oof_calibrated_verifier_2026_06_26

Group-OOF calibrated top-k-prior verifier over local latent/code tensors.
This is a table-level diagnostic and does not export counted `.oscr` streams.

## Results

| target | loss | pair | seed | oof changed | oof exact | oof false+wrong | oof T | val ens changed | val ens exact | val ens false+wrong | val ens T | val full changed | val full exact | val full false+wrong | val full T | val full P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact_changed` | `standard` | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `exact_changed` | `standard` | 0.25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `exact_changed` | `hardneg4` | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `exact_changed` | `hardneg4` | 0.25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `exact_changed` | `hardneg8` | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `exact_changed` | `hardneg8` | 0.25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `safe_improve` | `standard` | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `safe_improve` | `standard` | 0.25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `safe_improve` | `hardneg4` | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `safe_improve` | `hardneg4` | 0.25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `safe_improve` | `hardneg8` | 0.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `safe_improve` | `hardneg8` | 0.25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Interpretation

This tests whether train-group out-of-fold calibration and explicit hard-negative weighting can make the top-k/rate prior plus latent/code evidence safer than the in-sample N099 verifier. Promotion would still require beating the current counted actual-bitstream -8 Tesseract floor with low false/wrong changes.
