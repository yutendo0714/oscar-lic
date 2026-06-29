# Eval300 Assignment Interaction Applicability

**Date:** 2026-06-25  
**W&B:** offline run `nwso1h4q` (`eval300_assignment_interaction_applicability_2026_06_25`)  
**Scope:** prerequisite audit for sampled subset / approximate Shapley diagnostics.

## Result

The current assignment/codebook promotion path is not yet a meaningful target for interaction-error estimation. Both assignment tables used for promotion have exactly one distinct candidate unit per source image:

| table | rows | source groups | sources with >1 distinct candidate | max distinct candidates/source | negative utility rows |
|---|---:|---:|---:|---:|---:|
| assignment top8 trainval | 2400 | 300 | 0 | 1 | 264 |
| assignment top8 val | 600 | 75 | 0 | 1 | 40 |
| raw candidate Eval300 | 4760 | 300 | 300 | 50 | 409 |

Because the actual promoted policies send at most one distinct enhancement candidate per image, there is no within-image subset interaction to estimate on that path. The raw candidate table does have sufficient candidate density, so a future Shapley smoke should sample multi-candidate policies from `candidate_utility_table_eval300_tile4_scale025_q005_2026_06_25.jsonl`, generate explicit counted `.oscr` streams, and then compare observed pair/subset OCR deltas with the additive single-candidate prediction.

## Decision

Do not mark interaction/Shapley as completed for the actual assignment path. Record this as an applicability diagnostic and defer the real estimate until we generate explicit multi-candidate policies from the raw candidate table.
