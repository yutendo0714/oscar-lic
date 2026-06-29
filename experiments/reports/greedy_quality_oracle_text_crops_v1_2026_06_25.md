# Greedy Quality Oracle: Text-Crop v1

**Date:** 2026-06-25 JST  
**Candidate set:** CompressAI `bmshj2018-hyperprior` q1-q6 reconstructions  
**Teacher objective:** PARSeq `unicode_strict_v1` character edit distance improvement per added actual byte  
**Held-out evaluator:** Tesseract PSM8, same selected qualities  
**Rate:** corpus-level complete stream bytes from actual `compress/decompress` outputs  
**wandb offline run:** `2eirkfmm`

## Result

| Selection | Actual bpp | Quality counts | PARSeq CER | PARSeq exact | Tesseract CER | Tesseract exact |
|---|---:|---|---:|---:|---:|---:|
| uniform q1 | 0.5564381271 | `{1: 24}` | 0.1488095238 | 0.6250000000 | 0.4761904762 | 0.1250000000 |
| uniform q2 | 0.6806020067 | `{2: 24}` | 0.1071428571 | 0.5833333333 | 0.3809523810 | 0.1666666667 |
| uniform q3 | 0.8419732441 | `{3: 24}` | 0.0952380952 | 0.7083333333 | 0.3928571429 | 0.2500000000 |
| uniform q4 | 1.0664715719 | `{4: 24}` | 0.0833333333 | 0.7500000000 | 0.3690476190 | 0.2500000000 |
| uniform q5 | 1.3448996656 | `{5: 24}` | 0.0892857143 | 0.7083333333 | 0.3928571429 | 0.2500000000 |
| uniform q6 | 1.6747491639 | `{6: 24}` | 0.0714285714 | 0.7500000000 | 0.3571428571 | 0.2500000000 |
| greedy to q2 bytes | 0.6321070234 | `{1: 17, 2: 4, 3: 2, 6: 1}` | 0.0654761905 | 0.7916666667 | 0.4464285714 | 0.1666666667 |

The same greedy selection is returned for q2-q6 budgets because no remaining positive PARSeq edit-distance improvements are available after the selected upgrades; the oracle leaves extra budget unused rather than spending bits that do not help the teacher.

## Interpretation

This is the strongest early evidence so far for the OSCAR-LIC idea: nonuniform actual-byte allocation can beat uniform quality on the teacher OCR at lower total rate.

It is also a sharp warning. The PARSeq-only oracle degrades held-out Tesseract relative to uniform q2-q6. This supports the planned design choice: the eventual OSCAR-LIC utility objective must be recognizer-invariant or explicitly regularized against held-out-like failures, not just optimized for one differentiable teacher.

## Frozen Artifacts

- Summary: `experiments/results/greedy_quality_oracle_parseq_unicode_text_crops_v1_2026_06_25/summary.jsonl`
- Summary SHA256: `1aea887225c28f02879d2b25110ff89a567f7de6d8e4225c94c4612c2644c20b`
- Representative selection SHA256: `89f35a359e7177eb31c9ff6acabd0ee03fa410d9f8cc4c62470f662ec0e31e09`
- Script: `scripts/greedy_quality_oracle.py`

## Next

1. Repeat the oracle with a multi-teacher objective, for example PARSeq plus Tesseract edit loss, before building a learned predictor.
2. Add low-confidence strata and prevent the oracle from exploiting teacher instability on originally wrong crops.
3. Replace quality-level candidates with true latent/enhancement candidate units once the MLIC++ adapter boundary is ready.
