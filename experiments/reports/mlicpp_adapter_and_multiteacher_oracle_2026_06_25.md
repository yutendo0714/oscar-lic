# MLIC++ Adapter And Multi-Teacher Oracle

**Date:** 2026-06-25 JST  
**Base checkpoint:** MLIC++ corrected lambda 0.0250, epoch 612  
**Rate rule:** reported codec bpp uses complete serialized stream bytes. Likelihood rates below are diagnostics only.

## MLIC++ Adapter Boundary

The corrected MLIC++ base uses 10 uniform `y` slices of 32 channels. Its upstream compressor emits one main `y` ANS stream after appending all anchor and non-anchor symbols across all slices, plus separate entropy-bottleneck `z` streams. Therefore, the current upstream bitstream cannot provide actual per-slice or per-tile byte counts.

Decision: keep MLIC++ frozen as the base codec first and add OSCAR-LIC as independently counted enhancement sections. A split-stream MLIC++ base can be tested later, but it changes the base serialization and needs a separate ablation.

Design note: `docs/MLICPP_ADAPTER_DESIGN_2026_06_25.md`  
Diagnostic script: `scripts/inspect_mlicpp_latents.py`

## Latent Likelihood Diagnostics

| Split | Images | Actual bpp | Estimated bpp | Actual - estimated | Main observation |
|---|---:|---:|---:|---:|---|
| text crops | 24 | 2.7408453511 | 2.8592704086 | -0.1184250575 | `y` bits concentrate in slices 4, 5, 6, 8 |
| CLIC small | 9 | 0.7335005317 | 0.7300482534 | 0.0034522784 | `y` bits are much flatter across slices |

Text-crop estimated `y` bit shares: slice 4 = 0.1845, slice 5 = 0.1565, slice 6 = 0.1850, slice 8 = 0.1624.  
CLIC small estimated `y` bit shares are more even: the largest slices are 2, 6, 7 and 4 at roughly 0.12 each.

This suggests a first OSCAR candidate search should pay special attention to mid/late MLIC++ slices on text-rich crops, while still keeping controls that do not assume this slice prior.

## Confidence Strata

PARSeq original confidence was used only to define strata:

- low: confidence < 0.75, 6 samples;
- mid: 0.75 <= confidence < 0.98, 5 samples;
- high: confidence >= 0.98, 13 samples.

PARSeq candidates preserve the high-confidence stratum once quality reaches q3: high-stratum CER is 0.0 for q3-q6 and MLIC++. The low-confidence stratum remains unstable even at high quality. This supports using original-recognizable and confidence-aware filtering for oracle labels.

Tesseract behavior differs: even in the PARSeq high-confidence stratum, Tesseract CER remains high, but q4/q5/q6 improve exact rate relative to q1. This confirms that one differentiable teacher is insufficient as the sole training target.

## Multi-Teacher Quality Oracle

Candidate set: CompressAI `bmshj2018-hyperprior` q1-q6 reconstructions.  
Objective: weighted edit-distance loss, PARSeq 1.0 + Tesseract PSM8 1.0, `unicode_strict_v1`.  
Rate: corpus-level complete stream bytes from actual compressed files.

| Selection | Actual bpp | Quality counts | Weighted loss | PARSeq CER | PARSeq exact | Tesseract CER | Tesseract exact |
|---|---:|---|---:|---:|---:|---:|---:|
| uniform q1 | 0.5564381271 | `{1: 24}` | 105.0 | 0.1488095238 | 0.6250000000 | 0.4761904762 | 0.1250000000 |
| uniform q2 | 0.6806020067 | `{2: 24}` | 82.0 | 0.1071428571 | 0.5833333333 | 0.3809523810 | 0.1666666667 |
| uniform q3 | 0.8419732441 | `{3: 24}` | 82.0 | 0.0952380952 | 0.7083333333 | 0.3928571429 | 0.2500000000 |
| uniform q4 | 1.0664715719 | `{4: 24}` | 76.0 | 0.0833333333 | 0.7500000000 | 0.3690476190 | 0.2500000000 |
| uniform q6 | 1.6747491639 | `{6: 24}` | 72.0 | 0.0714285714 | 0.7500000000 | 0.3571428571 | 0.2500000000 |
| multi-teacher greedy to q2 bytes | 0.6722408027 | `{1: 12, 2: 7, 3: 4, 6: 1}` | 67.0 | 0.0833333333 | 0.7083333333 | 0.3154761905 | 0.2916666667 |
| multi-teacher greedy to q3+ bytes | 0.8035117057 | `{1: 10, 2: 5, 3: 3, 4: 3, 5: 1, 6: 2}` | 61.0 | 0.0773809524 | 0.7500000000 | 0.2857142857 | 0.3333333333 |

The equal-weight multi-teacher oracle is materially better than the previous PARSeq-only oracle on held-out Tesseract. At lower bpp than uniform q2, it improves both PARSeq and Tesseract. At lower bpp than uniform q3/q4/q6, it gives the best Tesseract CER among the tested selections while preserving strong PARSeq metrics.

## Overhead-Aware Multi-Teacher Oracle

The same oracle was rerun with a simple transmission-overhead simulation: 40 bytes for every selected enhancement candidate plus one 40-byte global gate overhead. Uniform q baselines are kept as ordinary full-codec streams, so the overhead applies only to mixed nonuniform selections.

| Selection | Actual bpp incl. overhead | Selection overhead | Quality counts | Weighted loss | PARSeq CER | Tesseract CER |
|---|---:|---:|---|---:|---:|---:|
| uniform q2 | 0.6806020067 | 0 | `{2: 24}` | 82.0 | 0.1071428571 | 0.3809523810 |
| overhead greedy to q2 bytes | 0.6760033445 | 440 | `{1: 14, 2: 6, 3: 2, 4: 2}` | 72.0 | 0.0833333333 | 0.3452380952 |
| uniform q3 | 0.8419732441 | 0 | `{3: 24}` | 82.0 | 0.0952380952 | 0.3928571429 |
| overhead greedy to q3 bytes | 0.8348662207 | 560 | `{1: 11, 2: 5, 3: 3, 4: 2, 5: 1, 6: 2}` | 62.0 | 0.0773809524 | 0.2916666667 |
| uniform q4 | 1.0664715719 | 0 | `{4: 24}` | 76.0 | 0.0833333333 | 0.3690476190 |
| overhead greedy to q4 bytes | 0.8662207358 | 600 | `{1: 10, 2: 5, 3: 3, 4: 3, 5: 1, 6: 2}` | 61.0 | 0.0773809524 | 0.2857142857 |

The margin shrinks at q2 when overhead is counted, but the nonuniform multi-teacher allocation still beats uniform q2 on both recognizers at slightly lower total bpp. This is not yet a real OSCAR enhancement stream, but it keeps the central hypothesis alive under a first-order packet overhead penalty.

## Non-Oracle Controls

Four non-oracle policies were tested with the same 40-byte/candidate plus 40-byte global overhead:

- `random`;
- `low_confidence`, using lower original PARSeq confidence first;
- `high_confidence`, using higher original PARSeq confidence first;
- `uniform_cost`, using equal score and therefore favoring cheap upgrades.

At the q2 budget, all controls underperform the overhead-aware multi-teacher oracle:

| Selection | Actual bpp | PARSeq CER | Tesseract CER |
|---|---:|---:|---:|
| overhead multi-teacher oracle to q2 | 0.6760033445 | 0.0833333333 | 0.3452380952 |
| random to q2 | 0.6806020067 | 0.1369047619 | 0.4702380952 |
| low-confidence to q2 | 0.6793478261 | 0.1369047619 | 0.4404761905 |
| high-confidence to q2 | 0.6751672241 | 0.1369047619 | 0.4702380952 |
| uniform-cost to q2 | 0.6797658863 | 0.1428571429 | 0.4464285714 |

At the q3 budget, the oracle remains ahead: PARSeq/Tesseract CER = 0.0774/0.2917, while the controls range from 0.1190/0.4107 to 0.1250/0.4583. This is useful negative evidence: naive confidence allocation is not enough.

## Frozen Artifacts

- Text latent summary: `experiments/results/mlicpp_latent_inspect_text24_2026_06_25/summary.json`
- Text latent summary SHA256: `b7dd832c1470310800290e39f590d2b205b1feb7dd51b49a5602a03395a21389`
- CLIC latent summary: `experiments/results/mlicpp_latent_inspect_clic9_2026_06_25/summary.json`
- CLIC latent summary SHA256: `b9efb382bf510193c00059b923d2faa0ae6b8f2fd2b96d03e2e2e760e1e2f550`
- PARSeq confidence strata SHA256: `227a2bde02b61f04b43a745bf7b7e8347323afdbfa9867d7ac9f97467587c587`
- Tesseract confidence strata SHA256: `21524184814ae8c500e9c513813afcf60ba42c784c8fbae49efeb23f0278526d`
- Multi-teacher oracle summary: `experiments/results/greedy_quality_oracle_multiteacher_equal_unicode_text_crops_v1_2026_06_25/summary.jsonl`
- Multi-teacher oracle summary SHA256: `df993c86ec5d6d94763ce93726c0ccefd3dfa8dd7b9dc801a7da56f103e2b1cf`
- Multi-teacher q2 selection SHA256: `8dedaeaaab609fb91d6738af37d7f0add20f4030ba5379c92f7459bea8c967a2`
- Overhead-aware multi-teacher oracle summary SHA256: `fe188e84a43a0e381e5a56cf09d5f90ad806a6a0d0e58ea229fb7547049cd564`
- Overhead-aware multi-teacher q2 selection SHA256: `7dd44f8ed98ee68c2896fce9ad078d7a862a379f0cc60fba0dbcefc929427315`
- Non-oracle controls summary SHA256: `bb8f6a7a404a62b541ef7b980d828bba2660bdec259f78cb598f39f561f11d0b`

## Next Decisions

1. Promote multi-teacher utility to the default oracle objective for the next stage.
2. Use confidence and original-recognizable strata to filter or downweight unstable oracle labels.
3. Carry explicit packet/gate overhead into every oracle and learned-allocation comparison.
4. Implement a frozen-base OSCAR enhancement prototype before modifying MLIC++ base serialization.
