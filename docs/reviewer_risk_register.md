# Anticipated Reviewer Critiques

## “This is a collection of existing modules.”

Response requirement: show that the utility formulation changes the allocation problem and beats mask/uncertainty/random controls. Keep diffusion optional.

## “The codec attacks one OCR rather than preserving information.”

Response requirement: held-out OCR main table, ground-truth sequences, teacher dropout, worst-OCR metric and human readability.

## “Side information is hidden.”

Response requirement: experimental file format and per-section actual byte breakdown.

## “Oracle crops exaggerate results.”

Response requirement: end-to-end detector+recognizer is primary; oracle crop only upper bound.

## “Synthetic data are unrealistic.”

Response requirement: scene/screen/document real domains and synthetic-to-real split.

## “Utility labels are too expensive.”

Response requirement: cost table, pruned candidate recall, proxy-vs-exact ranking ablation and deployment predictor latency.

## “Generative refinement changes characters.”

Response requirement: no-refiner main result, exact-match/seed variance, deterministic critical mode.

## “BD-rate values across papers are incomparable.”

Response requirement: rerun baselines in one pipeline and identify author-reported-only rows.
