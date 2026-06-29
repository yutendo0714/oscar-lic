# Eval600 Hard-Negative Pairwise Assignment Objective

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

The previous Eval600 hard no-headroom substrate reduced Eval300 false assignment changes, but the current top-8 group/no-op head still missed most hard positives. This run tests whether explicit matched positive-vs-hard-negative pairwise losses can convert the improved substrate into a promotable Eval300 policy.

The matched negatives are built from nearest-row core features: reference length, PARSeq nearest distance and Tesseract nearest distance. The trainer now supports:

- `--matched-pairwise-group-weight`
- `--matched-pairwise-row-weight`
- `--matched-pairwise-margin`
- `--matched-pairwise-pairs-per-positive`

Trainer SHA256: `799685b914aba922c2db976c7727d535174b1e8ecc34e3aab446506c30c0fe77`.

## Data

Feature artifact:

- `experiments/features/eval600_oraclesafe123_hardnoheadroom120_plus_eval300_seed1val_top8_windows_2026_06_26.npz`
- SHA256: `4818e593dcd76e70ef6bc131dabc4641c19971c6691351f620a4375e0e8270f2`
- Training groups: 243 Eval600 groups, made of 123 oracle-safe positives and 120 hard no-headroom negatives
- Validation groups: 75 Eval300 seed1 clean-val groups
- Eval300 validation oracle: 17 changed groups, PARSeq delta `0`, Tesseract edit delta `-24`

## Runs

### Pairwise Group+Row

Output:

- Metrics: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_pairwise_g05_r05_2026_06_26.json`
- Scores: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_pairwise_g05_r05_scores_2026_06_26.jsonl`
- W&B offline run: `nfy2esqe`

Config delta:

- `row_target=oracle_choice`
- `group_change_weight=0.5`
- `false_change_margin_weight=1.0`
- `matched_pairwise_group_weight=0.5`
- `matched_pairwise_row_weight=0.5`
- `matched_pairwise_pairs_per_positive=3`

Eval300 seed1 aggregate:

| Selector | Val AUC | Val AP | Changed | Tesseract delta | False changes | Wrong changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| threshold 0.5 | 0.8961 | 0.6962 | 13.33 | -1.00 | 8.00 | 3.33 | 0 |
| train-tuned | 0.8961 | 0.6962 | 17.67 | -1.33 | 11.33 | 4.00 | 0 |
| score consensus | n/a | n/a | 5 | -1 | 3 | 1 | 0 |

### Row-Heavy Pairwise

Output:

- Metrics: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_pairwise_row2_2026_06_26.json`
- Scores: `experiments/results/eval600_hardneg_to_eval300_assignment_group_noop_window_pairwise_row2_scores_2026_06_26.jsonl`
- W&B offline run: `4yx9140t`

Config delta:

- `matched_pairwise_group_weight=0.0`
- `matched_pairwise_row_weight=2.0`
- `matched_pairwise_pairs_per_positive=5`

Eval300 seed1 aggregate:

| Selector | Val AUC | Val AP | Changed | Tesseract delta | False changes | Wrong changes | PARSeq delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| threshold 0.5 | 0.8858 | 0.6576 | 15.00 | -1.00 | 10.00 | 3.00 | 0 |
| train-tuned | 0.8858 | 0.6576 | 19.00 | -2.33 | 12.67 | 3.33 | 0 |
| score consensus | n/a | n/a | 25 | +4 | 17 | 5 | 0 |

## Non-Deployable Top-M Upper Control

The top-M upper control is not deployable because it uses the known oracle code when the oracle appears inside the scorer's top-M list. It is useful only to locate the bottleneck.

| Scorer | Top-1 recovered / 17 | Top-1 delta | Top-2 recovered / 17 | Top-2 delta | Top-4 recovered / 17 | Top-4 delta | Top-8 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| hardneg oracle-choice baseline | 3 | -3 | 7 | -12 | 11 | -16 | -24 |
| pairwise group+row | 1 | -1 | 4 | -7 | 13 | -18 | -24 |
| row-heavy pairwise | 2 | -3 | 5 | -8 | 10 | -15 | -24 |

The important signal is that top-4 contains 10 to 13 of the 17 useful Eval300 code changes, while top-1 and gated policies still fail. This means the current scalar assignment score is good enough as a shortlist generator, but not good enough as a final selector.

## Conclusion

Matched pairwise losses are a valid negative result for the current single-score group/no-op head. They improve row AP and in one setting improve the top-4 oracle upper control, but they do not produce a safe deployable policy and remain below the promoted actual-bitstream baseline (`-8` Tesseract edits with unchanged actual bpp and PARSeq unchanged).

The next objective should not be another scalar loss sweep. The next useful experiment is a two-stage policy:

1. use the current scorer only to create a top-4 shortlist plus explicit nearest/no-op anchor;
2. train a small reranker or budgeted policy over that shortlist to choose no-op versus one code;
3. optimize the selection objective directly against false changes, wrong changes, missed oracle groups, Tesseract edit delta and PARSeq no-worse constraints;
4. promote only after counted OSCR export and actual OCR comparison.

## Decision

Record as N028. Continue with top-4 shortlist reranking or policy-budget training over the hard-negative-balanced Eval600 substrate. Do not rerun the same pairwise weights, seed-sweep the current single-score head, or promote this diagnostic to counted OSCR.
