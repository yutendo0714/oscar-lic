# Exception-Center Replacement Plan

Date: 2026-06-26

## Purpose

N138-N142 changed the interpretation of utility-positive exception centers. They are not a standalone selector fix. They are a compact-symbol proposal mechanism that may be useful only if a policy preserves useful current selections and makes high-precision replace/drop decisions.

The next exception-center experiment should therefore compare against the current compact policy, not only against nearest assignment. Lower enhancement rate is useful only if OCR is preserved or improved under actual serialized bytes.

## Evidence Summary

| Evidence | Key result | Implication |
|---|---:|---|
| N138 exception-center feasibility | K16 reaches `9/17` useful validation oracle residuals but creates `13` no-headroom close opportunities | Extra centers can cover useful residuals, but they create false-change pressure. |
| N139 K80 actual compact stream audit | `75` selected candidates, `0` assignment changes | Appending centers alone is a no-op under nearest assignment. |
| N140 selector audit | train `rel<=1.0` precision `0.9778`, validation precision `0.4091` | Distance/margin/center-utility selection is not source-robust. |
| N141 actual OCR | current `50` selected candidates, exception policy `22`; total bpp `3.823677 -> 3.793834`; Tesseract worsens `+3` unicode / `+6` latin; PARSeq unchanged | Sparse exception-only selection lowers rate by dropping current selections, but is not an OCR win. |
| N141 selected rows | Tesseract unicode net `0`; latin `+2`; PARSeq unchanged | Selected exception replacements are mostly neutral, with one improvement and one regression. |
| N142 contact sheet | mostly near-identical current/exception pairs, one visible OCR improvement and one OCR regression | Future features should compare current-vs-exception local code effects, not only exception distance. |
| N143 action-readiness table | validation has `22` direct replacement candidates and `53` current-only drop-risk rows; no add-exception rows in this substrate | This artifact is a replace/drop problem over current-selected units. |

N141 also warns that the proxy labels in N143 are not sufficient replacement targets. Among the `9` selected proxy-oracle-change rows, actual current-relative Tesseract unicode changes are only `-1` char total, while latin-alnum worsens by `+1`. Among the `13` selected proxy-noheadroom rows, Tesseract worsens by `+1` under both profiles. Use proxy labels for cohort construction, not as final replacement utility labels.

## Required Decision Structure

For each current selected group, define an explicit action set:

- `keep_current`: transmit the current compact K64 utility-weighted assignment.
- `replace_with_exception`: transmit the K80 exception-center code for the same candidate.
- `drop_current`: omit the optional enhancement for this group and keep base/nearest reconstruction.
- `keep_nearest`: no optional enhancement for groups that current would not select.

The policy should be current-preserving by default. It may replace or drop only when the verifier has evidence that the action improves the OCR/rate objective without opening no-headroom false changes.

## Candidate Evidence

Use candidate-local evidence that directly compares the reconstructions, not only codebook geometry:

- source crop, nearest reconstruction, current reconstruction and exception reconstruction;
- `current - nearest`, `exception - current`, `source - current`, `source - exception` patches;
- local latent windows for `y_hat`, current target/code residual and exception target/code residual;
- codebook metadata: nearest/current/exception code index, relative error, distance margin, center utility;
- source-domain and text-size metadata allowed by the protocol;
- optional train-teacher auxiliary losses from PARSeq/CRNN/ABINet, only as training supervision or auxiliary regularization, never as held-out evaluator features.

Do not include held-out Tesseract/PaddleOCR/TrOCR outputs, reference strings, edit deltas or oracle labels as inference features.

## Objective

Train or select with group-level costs, not row AUC:

- reward true OCR-improving replacement relative to current;
- penalize false replacement on no-headroom groups;
- penalize dropping a useful current selection;
- allow a small rate credit for `drop_current`, but never allow rate credit to mask held-out OCR regression;
- keep PARSeq no-worse as a hard validation constraint until a better multi-teacher objective is justified.

At table level, report at least:

- exact useful replacements;
- false replacements;
- wrong replacements;
- useful current drops;
- harmful current drops;
- missed useful replacements;
- Tesseract/PARSeq edit deltas relative to current and nearest;
- estimated selected bytes only as diagnostic.

## Promotion Gate

Before any actual `.oscr` generation:

- validation policy must beat or tie current Tesseract relative to current, not only reduce bpp;
- PARSeq must be unchanged or better;
- false plus wrong replacements should stay near the current precision floor;
- selected-row Tesseract delta must be non-positive under both `unicode_strict_v1` and `latin_alnum_ci_v1`;
- if the policy is primarily a rate-reduction policy, label it as secondary and compare against current at matched OCR, not as an OSCAR utility gain.

Only after the table precheck passes:

- export explicit policy rows;
- generate compact `TEXT_SYMBOLS` actual `.oscr` streams;
- compute bpp from complete file bytes;
- run PARSeq and Tesseract row-wise OCR first;
- add TrOCR/PaddleOCR evaluation if PARSeq/Tesseract pass;
- run paired bootstrap against nearest/current/top8 where sample size permits.

## Stop Rules

Stop the branch if any of the following occurs:

- replacement/dropping improves rate but worsens Tesseract versus current;
- the policy selects fewer candidates than current while missing known useful current selections;
- validation false replacement count exceeds useful replacement count;
- PARSeq worsens on table precheck or actual `.oscr`;
- gains are visible only in distance proxies, center utility, row AUC/AP or estimated rate.

## Next Concrete Experiment

Build a train-only table with one row per action in `{keep_current, replace_with_exception, drop_current}` for current-selected groups, plus `keep_nearest` for no-current groups. Train a small group policy with explicit action costs and source-held-out calibration. The first report should be table-only and should not create streams unless it improves or ties current OCR while reducing selected bytes.

For the existing N143 top8 selected-current substrate, the no-current case is absent. The immediate action set is therefore `{keep_current, replace_with_exception, drop_current}`. Add no-current/add-exception actions only if a later artifact includes groups that the current listwise gate did not select.
