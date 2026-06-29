# eval300_interaction_subset3_smoke40_stop_baselines_2026_06_26

Target: `label_tesseract_improve_parseq_safe_unicode_strict_v1`
Transition positive rates: `{'0_to_1': 0.9666666666666667, '1_to_2': 0.04583333333333333, '2_to_3': 0.21666666666666667}`

## Fixed Policies

| policy | selected | TP/FP/FN | precision | recall | F1 | Tess delta | PARSeq delta |
|---|---:|---|---:|---:|---:|---:|---:|
| none | 0 | 0/0/153 | n/a | 0.0000 | n/a | 0 | 0 |
| all | 480 | 153/327/0 | 0.3187 | 1.0000 | 0.4834 | -7 | -13 |
| first_only | 120 | 116/4/37 | 0.9667 | 0.7582 | 0.8498 | -179 | -13 |
| first_or_third | 240 | 142/98/11 | 0.5917 | 0.9281 | 0.7226 | -185 | -17 |
| continue_only | 360 | 37/323/116 | 0.1028 | 0.2418 | 0.1442 | 172 | 0 |

## Cross-Fit Transition Thresholds

| policy | selected | TP/FP/FN | precision | recall | F1 | Tess delta | PARSeq delta |
|---|---:|---|---:|---:|---:|---:|---:|
| transition_rate_ge_0.05 | 288 | 142/146/11 | 0.4931 | 0.9281 | 0.6440 | -149 | -13 |
| transition_rate_ge_0.10 | 240 | 142/98/11 | 0.5917 | 0.9281 | 0.7226 | -185 | -17 |
| transition_rate_ge_0.20 | 216 | 133/83/20 | 0.6157 | 0.8693 | 0.7209 | -175 | -17 |
| transition_rate_ge_0.50 | 120 | 116/4/37 | 0.9667 | 0.7582 | 0.8498 | -179 | -13 |
| transition_rate_ge_0.80 | 120 | 116/4/37 | 0.9667 | 0.7582 | 0.8498 | -179 | -13 |
| transition_rate_ge_0.95 | 120 | 116/4/37 | 0.9667 | 0.7582 | 0.8498 | -179 | -13 |

Interpretation: this is a tiny diagnostic over OCR-derived labels. A strong first-only baseline means the next model needs transition/context awareness before capacity.
