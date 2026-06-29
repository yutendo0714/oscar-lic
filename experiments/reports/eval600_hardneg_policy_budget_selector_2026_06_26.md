# Eval600 Hard-Negative Policy-Budget Selector

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

The matched pairwise assignment head showed useful top-M coverage but poor deployed precision. This diagnostic asks whether a separate policy-budget selector can turn existing score JSONL files into a safer no-op-versus-change policy without training a larger model.

The new script `scripts/analyze_assignment_policy_budget_selector.py` tunes, on Eval600 train groups only:

- candidate shortlist size `top_m`;
- candidate ranking metric;
- change budget or transferred score threshold;
- train-side false-change, wrong-change, Tesseract-worsen and PARSeq-worsen limits.

It then evaluates the selected policy on Eval300 seed1 clean-val groups.

Script SHA256: `2b39a3c34c11c95cbba2c7adcb460f96e0e61e15a8474d6655bfa137f1371461`.

## Inputs

Primary score files:

- hard-negative oracle-choice baseline scores: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_oraclechoice_scores_2026_06_26.jsonl`
- pairwise group+row scores: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_pairwise_g05_r05_scores_2026_06_26.jsonl`
- pairwise row-heavy scores: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_pairwise_row2_scores_2026_06_26.jsonl`

Eval300 validation oracle remains unchanged:

- 75 groups
- 17 oracle-change groups
- PARSeq delta `0`
- Tesseract delta `-24`

## Results

All strict/relaxed/capped/top4 variants preserve PARSeq because selection was run with `--parseq-safe`. None is promotable.

| Selector | Train budget | Transfer mode | Val changed | Val Tesseract delta | False | Wrong | Tesseract worsen |
|---|---:|---|---:|---:|---:|---:|---:|
| oracle-choice strict | 70/243 | threshold | 12 | 0 | 6 | 3 | 3 |
| oracle-choice strict | 70/243 | budget fraction | 22 | -2 | 12 | 5 | 5 |
| pairwise g05/r05 strict | 70/243 | threshold | 0 | 0 | 0 | 0 | 0 |
| pairwise g05/r05 strict | 70/243 | budget fraction | 22 | -1 | 14 | 4 | 4 |
| pairwise row2 strict | 70/243 | budget fraction | 22 | +1 | 14 | 4 | 5 |
| oracle-choice budget10 | 24/243 | budget fraction | 7 | +1 | 4 | 2 | 2 |
| pairwise g05/r05 budget10 | 24/243 | budget fraction | 7 | -1 | 5 | 1 | 1 |
| oracle-choice budget15 | 36/243 | budget fraction | 11 | +1 | 7 | 2 | 3 |
| pairwise g05/r05 budget15 | 36/243 | budget fraction | 11 | -2 | 7 | 2 | 1 |
| oracle-choice top4 budget15 | 36/243 | budget fraction | 11 | -1 | 5 | 3 | 2 |
| pairwise g05/r05 top4 budget15 | 36/243 | budget fraction | 11 | -2 | 7 | 2 | 1 |

Relaxed train limits (`0.02` and `0.05` false/wrong/worsen rates) chose the same `70/243` train solution as the strict run because the training substrate is already perfectly separable under the score ranking. Relaxation therefore did not improve Eval300 transfer.

## Interpretation

This is a useful negative result. On Eval600 train, the score space can select all 70 oracle-change groups with zero false/wrong/worsen events. On Eval300 seed1, the transferred thresholds either collapse to no-op for the pairwise scorer or emit many false/wrong changes. Budget-fraction transfer recovers at most `-2` Tesseract edit characters and remains far below the promoted actual-bitstream gate (`-8` with much cleaner precision) and the Eval300 oracle (`-24`).

The bottleneck is not just the absence of a budget knob. It is external calibration of the change/no-op boundary and candidate identity under distribution shift. This argues against spending more time on post-hoc thresholding of the same scores.

## Decision

Record as N029. Do not promote policy-budget post-processing over the current score files. The next productive direction is richer deployable evidence for candidate-local code effects and/or source/domain-disjoint calibration data, followed by actual counted OSCR export only after table-level policy precision exceeds the promoted baseline.
