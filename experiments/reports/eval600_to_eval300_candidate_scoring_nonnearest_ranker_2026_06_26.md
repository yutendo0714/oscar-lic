# Eval600 to Eval300 Candidate Scoring and Non-Nearest Ranker

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N048 showed that summary-level multi-evidence top-4 selectors move away from nearest but mostly pick false/wrong changes. This follow-up asks:

1. Is the current first-stage source-OOF score already ranking the oracle changed candidate above nearest?
2. If nearest over-scoring is the bottleneck, does a non-nearest-only patch ranker fix Eval600-to-Eval300 transfer?

## Setup

- Assignment table: Eval600 hard-negative/balanced plus Eval300 seed1 top-8 image-diff table
- Eval split: Eval300 seed1, 75 groups
- Scores: source-OOF mixed oracle-choice and safe-improve scores, score model seed 2
- New diagnostic model: non-nearest-only diff-crop CNN ranker
- Model seeds: 0, 1, 2
- No actual `.oscr` stream was produced; this is a selector/candidate-scoring diagnostic only.

## N049 Candidate-Scoring Audit

The upper bound is still strong:

| Policy | Changed | Tesseract delta | Exact oracle | False/wrong | Missed oracle |
|---|---:|---:|---:|---:|---:|
| nearest | 0 | 0 | 0 | 0 | 17 |
| shortlist oracle | 15 | -20 | 15 | 0 | 2 |
| full oracle | 17 | -24 | 17 | 0 | 0 |
| oracle-score argmax | 4 | +1 | 0 | 4 | 17 |
| safe-score argmax | 23 | +3 | 2 | 21 | 13 |

Coverage is not the main issue on this split: 15 of 17 oracle changed groups are present in the top-4 shortlist. The sharper failure is rank calibration. For the 17 oracle changed groups, the oracle-choice score ranks the true changed candidate as rank 2 or worse in every case; its score margin against nearest is always negative, with median about `-0.91`. The safe-improve score finds 2 exact changes but also triggers 21 false/wrong changes and worsens aggregate Tesseract.

## N050 Non-Nearest-Only Ranker

Removing nearest from the row-level training objective did not fix cross-domain transfer.

| Target / policy | Val AUC | Val AP | Changed | Tesseract delta | Exact oracle | False/wrong | Missed oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle-choice, safe threshold | 0.590 | 0.041 | 6.00 | +0.33 | 0.00 | 6.00 | 17.00 |
| oracle-choice, utility threshold | 0.590 | 0.041 | 14.00 | +3.33 | 0.33 | 13.67 | 15.00 |
| safe-improve, safe threshold | 0.572 | 0.093 | 13.00 | +0.33 | 1.00 | 12.00 | 14.67 |
| safe-improve, utility threshold | 0.572 | 0.093 | 19.00 | -0.33 | 1.67 | 17.33 | 13.00 |

The rankers fit the Eval600 training side almost perfectly (`train_auc` about 0.997-1.000), but transfer weakly to Eval300. The best mean Tesseract direction is the safe-improve utility threshold at `-0.33`, but it requires about 17 false/wrong changes per 75 groups, so it is not deployable and should not be promoted to actual `.oscr` evaluation.

## Interpretation

The failure is now better localized. The top-4 shortlist often contains the oracle candidate, but deployable candidate scoring cannot distinguish rare safe text-improving edits from visually similar false changes under Eval600-to-Eval300 transfer. Nearest over-scoring is a symptom, not the whole cause. A small diff-crop CNN overfits the training source and does not provide a reliable no-op/abstention signal on Eval300.

The next branch should avoid more shallow diff-crop/scalar selector variants. Better options are:

- structured local latent/code-token modeling with source-disjoint calibration,
- OCR/text-specific verification features that do not use held-out evaluator labels,
- broader domain-balanced training data for candidate scoring, or
- a stricter two-stage design: high-recall candidate proposal first, then conservative verifier/abstainer.

W&B offline run: `rym86loh`.

## Files

- Audit script: `scripts/audit_topk_candidate_scoring.py`
- Non-nearest ranker: `scripts/train_assignment_nonnearest_patch_ranker.py`
- N049 result: `experiments/results/eval600_source_oof_to_eval300_top4_candidate_scoring_audit_2026_06_26.json`
- N049 details: `experiments/results/eval600_source_oof_to_eval300_top4_candidate_scoring_audit_details_2026_06_26.jsonl`
- N050 oracle-choice result: `experiments/results/eval600_to_eval300_nonnearest_oraclechoice_patch_ranker_2026_06_26.json`
- N050 safe-improve result: `experiments/results/eval600_to_eval300_nonnearest_safeimprove_patch_ranker_2026_06_26.json`
