# Exception Action Supervision Gap

This audit uses the N143 action-readiness table. It is a supervision diagnostic, not a policy result.

## Replacement Balance

| partition | groups | replacement candidates | proxy true | proxy false | positive rate | current-only drop risk |
|---|---:|---:|---:|---:|---:|---:|
| train | 225 | 45 | 44 | 1 | 0.9778 | 180 |
| val | 75 | 22 | 9 | 13 | 0.4091 | 53 |

## Source Breakdown

### train

| source | replacement candidates | proxy true | proxy false |
|---|---:|---:|---:|
| iam_words | 22 | 22 | 0 |
| icdar2013 | 4 | 3 | 1 |
| icdar2015 | 8 | 8 | 0 |
| icdar_mlt2019_words | 5 | 5 | 0 |
| synthtext_words | 2 | 2 | 0 |
| textocr | 4 | 4 | 0 |

### val

| source | replacement candidates | proxy true | proxy false |
|---|---:|---:|---:|
| iam_words | 12 | 4 | 8 |
| icdar2013 | 2 | 2 | 0 |
| icdar2015 | 7 | 3 | 4 |
| textocr | 1 | 0 | 1 |

## Conclusion

The rel<=1.0 replacement cohort is not a usable standalone training target: train has only 1 proxy-false replacement candidate, while validation has 13. This explains why distance-only exception policies look high-precision on train and false-heavy on validation.

## Next Action

Collect or derive explicit current-relative replacement/drop labels, or add hard replacement-negative supervision, before training a replace/drop verifier.
