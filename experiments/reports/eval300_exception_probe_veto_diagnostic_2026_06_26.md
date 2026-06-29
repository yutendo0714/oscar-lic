# Exception Probe Veto Diagnostic Verifier

This is a leave-one-out diagnostic over 53 actual-labeled rows, not a deployable policy.

## Baseline

Rows `53`, labels `{"beneficial_no_profile_harm": 16, "harmful_any_profile": 4, "neutral_all_profiles": 33}`, Tesseract `-15` unicode / `-11` latin, rate delta bytes `-1290`.

## logistic

| max retained harmful | threshold | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00010874 | 0 | 4 | 16 | 0 | 0 | 0 | 53 |
| 1 | 0.00010874 | 0 | 4 | 16 | 0 | 0 | 0 | 53 |
| 2 | 0.00010874 | 0 | 4 | 16 | 0 | 0 | 0 | 53 |
| 3 | 0.00010874 | 0 | 4 | 16 | 0 | 0 | 0 | 53 |
| 4 | 0.890461 | 4 | 0 | 0 | -15 | -11 | -1247 | 2 |

## forest

| max retained harmful | threshold | retained harmful | vetoed harmful | vetoed beneficial | tess unicode | tess latin | rate bytes | vetoed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -1e-09 | 0 | 4 | 16 | 0 | 0 | 0 | 53 |
| 1 | -1e-09 | 0 | 4 | 16 | 0 | 0 | 0 | 53 |
| 2 | -1e-09 | 0 | 4 | 16 | 0 | 0 | 0 | 53 |
| 3 | -1e-09 | 0 | 4 | 16 | 0 | 0 | 0 | 53 |
| 4 | 0.499666 | 4 | 0 | 0 | -15 | -11 | -1290 | 0 |

## Conclusion

Leave-one-out harmful-row scoring is a tiny diagnostic only. It tests whether deployable image/rate/action features carry enough signal to approximate the N154 veto frontier without source/reference metadata.

## Next Action

If the LOO frontier is competitive with N154, fold these scores into a larger hard-negative collection; otherwise collect more labels before training any verifier.
