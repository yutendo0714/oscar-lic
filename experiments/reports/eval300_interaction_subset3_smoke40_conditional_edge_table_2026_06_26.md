# eval300_interaction_subset3_smoke40_conditional_edge_table_2026_06_26

This table is a non-deployable diagnostic label substrate for conditional stop/continue modeling.
Labels use OCR outcomes and must not be treated as inference-time features.

| transition | rows | Tess unicode delta | Tess improved/worse | PARSeq worse | safe Tess-improve/PARSeq-safe | added enh bpp | added PSNR |
|---|---:|---:|---|---:|---:|---:|---:|
| 0_to_1 | 120 | -179 | 116/0 | 0 | 116 | 2.1017 | 0.0641 |
| 1_to_2 | 240 | 178 | 11/121 | 2 | 11 | 2.0165 | 0.0648 |
| 2_to_3 | 120 | -6 | 26/31 | 0 | 26 | 2.0151 | 0.0615 |

Interpretation: a useful stop/continue target must separate `0_to_1` from `1_to_2`; the same visual residual evidence has very different OCR effect once a first candidate has already been sent.
