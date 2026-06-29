# Exception-Center Selector Audit

This diagnostic evaluates whether K80 exception-center distance features can separate oracle-change groups from no-headroom groups.
It does not decode new streams or claim OCR improvement.

## Thresholds

### `train`

Groups: `225`; oracle-change groups: `63`.

| threshold | changed | oracle covered | no-headroom false | recall | precision |
|---:|---:|---:|---:|---:|---:|
| 1.000 | 45 | 44 | 1 | 0.6984 | 0.9778 |
| 1.010 | 45 | 44 | 1 | 0.6984 | 0.9778 |
| 1.020 | 45 | 44 | 1 | 0.6984 | 0.9778 |
| 1.030 | 46 | 44 | 2 | 0.6984 | 0.9565 |
| 1.040 | 47 | 45 | 2 | 0.7143 | 0.9574 |
| 1.050 | 47 | 45 | 2 | 0.7143 | 0.9574 |
| 1.080 | 47 | 45 | 2 | 0.7143 | 0.9574 |
| 1.100 | 47 | 45 | 2 | 0.7143 | 0.9574 |
| 1.150 | 48 | 46 | 2 | 0.7302 | 0.9583 |
| 1.200 | 53 | 47 | 6 | 0.7460 | 0.8868 |

### `val`

Groups: `75`; oracle-change groups: `17`.

| threshold | changed | oracle covered | no-headroom false | recall | precision |
|---:|---:|---:|---:|---:|---:|
| 1.000 | 22 | 9 | 13 | 0.5294 | 0.4091 |
| 1.010 | 26 | 9 | 17 | 0.5294 | 0.3462 |
| 1.020 | 29 | 10 | 19 | 0.5882 | 0.3448 |
| 1.030 | 35 | 12 | 23 | 0.7059 | 0.3429 |
| 1.040 | 39 | 13 | 26 | 0.7647 | 0.3333 |
| 1.050 | 43 | 14 | 29 | 0.8235 | 0.3256 |
| 1.080 | 51 | 14 | 37 | 0.8235 | 0.2745 |
| 1.100 | 54 | 16 | 38 | 0.9412 | 0.2963 |
| 1.150 | 61 | 16 | 45 | 0.9412 | 0.2623 |
| 1.200 | 70 | 17 | 53 | 1.0000 | 0.2429 |

## Train-Tuned Thresholds

| budget | threshold | val changed | val oracle covered | val false |
|---:|---:|---:|---:|---:|
| 0 | none | 0 | 0 | 0 |
| 1 | 1.02 | 29 | 10 | 19 |
| 3 | 1.15 | 61 | 16 | 45 |
| 5 | 1.15 | 61 | 16 | 45 |

## Learned Score Audit

| model | budget | threshold | val changed | val oracle covered | val false |
|---|---:|---:|---:|---:|---:|
| logistic | 0 | 0.6249051260755815 | 0 | 0 | 0 |
| logistic | 1 | 0.6249051260755815 | 0 | 0 | 0 |
| logistic | 3 | 0.5456974395762767 | 0 | 0 | 0 |
| logistic | 5 | 0.5456974395762767 | 0 | 0 | 0 |
| hist_gradient | 0 | 0.866495572385456 | 0 | 0 | 0 |
| hist_gradient | 1 | 0.866495572385456 | 0 | 0 | 0 |
| hist_gradient | 3 | 0.866495572385456 | 0 | 0 | 0 |
| hist_gradient | 5 | 0.866495572385456 | 0 | 0 | 0 |
| random_forest | 0 | 0.6423319215074024 | 0 | 0 | 0 |
| random_forest | 1 | 0.6423319215074024 | 0 | 0 | 0 |
| random_forest | 3 | 0.6423319215074024 | 0 | 0 | 0 |
| random_forest | 5 | 0.6423319215074024 | 0 | 0 | 0 |

## Conclusion

Distance-only exception-center eligibility is not source-robust: thresholds that look high-precision on train leak many no-headroom validation groups. Exception centers therefore need a materially stronger no-headroom verifier before actual OCR streams are worth generating.

## Next Action

Use this audit as a stop rule for simple distance selectors. A future exception-center selector should add candidate-local visual/code-effect evidence and prove nonzero useful validation changes with near-zero no-headroom false changes before `.oscr` OCR evaluation.
