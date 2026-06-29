# Eval600 Source-OOF to Eval300 Top-4 Budget Selector Diagnostic

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N038 showed that strict high-precision budget rules over in-sample diff-crop scores do not transfer. N041 generated source-modulo OOF first-stage scores. This diagnostic asks:

> If the same budget rule is tuned on source-OOF Eval600 scores, does it safely transfer useful top-4 changes to Eval300?

## Setup

Scores:

- oracle-choice source-OOF mixed score file;
- safe-improve source-OOF mixed score file.

Selector:

- top-m: `2,3,4`;
- rank metrics: `score_mean`, `score_min`, `margin_mean`, `margin_min`, `vote_then_score`;
- train constraints: zero false changes, zero wrong changes, zero Tesseract-worsen groups, zero PARSeq-worsen groups;
- max budget rate: `0.3`;
- transfer modes: learned score threshold and learned budget fraction.

## Results

OOF train tuning no longer looks overconfident:

- selected changes: `1/473`;
- train Tesseract delta: `-2`;
- false/wrong/worsen groups: `0`;
- tuned rule: top-2, `margin_mean`, threshold `0.9106`.

Eval300 transfer is safe but useless:

| Policy | Changed groups | Tesseract delta | Exact oracle | Missed oracle | False changes | Worsen groups |
|---|---:|---:|---:|---:|---:|---:|
| nearest | 0 | 0 | 0 | 17 | 0 | 0 |
| oracle | 17 | -24 | 17 | 0 | 0 | 0 |
| threshold transfer | 0 | 0 | 0 | 17 | 0 | 0 |
| fraction transfer | 0 | 0 | 0 | 17 | 0 | 0 |

## Interpretation

This is a valid negative result for source-OOF score budget rules.

The result is useful because it separates two failures:

- in-sample scores created false train confidence in N038;
- OOF scores remove that overconfidence, but the strict rule then has almost no recall.

The top-4 shortlist still has strong headroom, but score-only abstention cannot identify Eval300 hard positives at the needed precision/recall tradeoff.

## Next Action

Do not promote or actual-OSCR this rule. Do not rerun score-only budget/threshold selectors over the same OOF score files. The next branch needs materially different evidence, not another score threshold.

W&B offline run: `p4nohfm9`.

## Files

- Result: `experiments/results/eval600_source_oof_to_eval300_top4_budget_selector_2026_06_26.json`
- Threshold policy: `experiments/results/eval600_source_oof_to_eval300_top4_budget_selector_policy_2026_06_26.threshold.jsonl`
- Fraction policy: `experiments/results/eval600_source_oof_to_eval300_top4_budget_selector_policy_2026_06_26.fraction.jsonl`
