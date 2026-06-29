# Eval600 Hard-Negative Balanced210 Top-4 Policy Selector

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic, no counted OSCR promotion stream

## Question

N036 showed that diff-crop candidate scores preserve useful candidates in top-4 but fail at top-1 calibration. This diagnostic tests a small policy selector that sees a shortlist instead of a single best row.

Candidate set per group:

- nearest/no-op row;
- top-4 rows by N034 `oracle_choice` diff-crop score;
- top-4 rows by N035 `safe_improve` diff-crop score;
- duplicate candidates merged.

The selector is trained with:

- explicit nearest/no-op option;
- listwise group target using `label_assignment_oracle_choice`;
- margin penalty when a harmful non-nearest candidate outranks nearest on train groups.

## Results

The shortlist itself is strong:

- shortlist oracle Tesseract delta: `-19.33` edit characters;
- shortlist oracle PARSeq delta: `0`;
- exact oracle changes in shortlist: `14.33/17`;
- missed full-oracle changes: `2.67/17`.

But the learned selector fails:

| Policy | Changed groups | Tesseract delta | False changes | Wrong changes | Exact oracle changes | Missed oracle changes |
|---|---:|---:|---:|---:|---:|---:|
| nearest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 17.00 |
| shortlist oracle | 14.33 | -19.33 | 0.00 | 0.00 | 14.33 | 2.67 |
| learned argmax | 9.56 | +1.11 | 9.56 | 9.56 | 0.00 | 16.78 |
| train-tuned margin | 9.56 | +1.11 | 9.56 | 9.56 | 0.00 | 16.78 |

All values are means over 3 diff-crop score model seeds x 3 selector seeds on Eval300 seed1.

## Interpretation

This is a valid negative result for the first learned top-4 policy selector.

Important positive diagnostic:

- the union shortlist almost closes the Eval300 oracle gap (`-19.33` versus full `-24`);
- therefore the candidate-generation problem is now much smaller than the selector problem.

Failure mode:

- the learned selector recovers essentially none of the useful shortlist oracle choices;
- it selects many non-oracle non-nearest candidates, producing false/wrong changes and net Tesseract harm;
- train-tuned margins do not help because the selector's train-side confidence boundary does not transfer.

The next selector should not simply reuse in-sample Eval600 learned score features. It needs stronger external calibration or a more conservative uncertainty/abstention objective, possibly with source/domain-disjoint calibration and explicit false-change control.

W&B offline run: `33xqtxjk`

## Decision

Do not promote or seed-sweep this first top-4 MLP selector.

Keep the shortlist construction as useful infrastructure. Future work should focus on calibrated high-precision selection within this shortlist, or on exporting shortlist-oracle upper bounds to guide a better policy objective.
