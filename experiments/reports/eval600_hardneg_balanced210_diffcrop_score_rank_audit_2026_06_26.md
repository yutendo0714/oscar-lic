# Eval600 Hard-Negative Balanced210 Diff-Crop Score-Rank Audit

Date: 2026-06-26 JST  
Stage: G5 diagnostic  
Track: A pure visual bitstream diagnostic

## Question

N034 and N035 both failed as deployed policies, but they might fail in different ways. This audit asks whether the diff-crop candidate scores lose oracle candidates entirely, or whether useful candidates remain in the shortlist but are misranked at top-1.

Inputs:

- N034 `oracle_choice` diff-crop scores;
- N035 `safe_improve` diff-crop scores;
- fixed Eval300 seed1 validation groups;
- top-k assignment groups with 17 oracle-change / safe-improve groups.

## Findings

The `oracle_choice` diff-crop scorer preserves shortlist recall but over-scores nearest/no-op rows:

- oracle-change top-4 hits: mean `11.67/17`;
- safe-improve top-4 hits: mean `14.33/17`;
- top-1 improving groups: `0.00`;
- top-1 harmful groups: `1.33`;
- oracle score median is tiny (`~2e-4` to `5e-4` by model seed) while nearest score median is high (`~0.92` to `0.96`).

The `safe_improve` diff-crop scorer partially moves safe rows upward but also raises harmful rows:

- safe-improve top-1 hits: mean `1.67/17`;
- safe-improve top-4 hits: mean `14.00/17`;
- top-1 harmful groups: mean `4.00`;
- oracle/safe margin versus nearest is much less negative, but harmful candidate scores have very high upper quantiles.

## Interpretation

This is useful positive diagnostic evidence, not a promotion result.

The diff-crop evidence is not useless: useful candidates remain in top-4 for most oracle groups. The failure is the final high-precision choice among nearest, safe useful candidates and harmful candidates. That points to a verifier/selector that sees candidate-local evidence and is explicitly trained for abstention and false/wrong-change suppression, rather than another top-1 row classifier.

For the next experiment, the design requirement is:

- keep a top-4 shortlist rather than trusting top-1;
- include an explicit no-op decision;
- penalize harmful top-1 candidates directly;
- validate externally before any counted `.oscr` promotion.

W&B offline run: `ccm9qx83`
