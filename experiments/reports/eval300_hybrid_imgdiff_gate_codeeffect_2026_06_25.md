# Eval300 Hybrid Image-Diff Gate + Code-Effect Reranker Diagnostic

Date: 2026-06-25

## Purpose

Test whether the promoted high-precision pooled image-diff group gate can be kept as the change/no-op selector while swapping only the assignment code choice to newer tensor/group-no-op reranker scores.

This is a table-level precheck only. No actual `.oscr` streams are promoted from this diagnostic.

## Inputs

- Gate policy: `experiments/results/eval300_assignment_group_gate_oraclechoice_model0_pooled_imgdiff_model1_threshold05_explicit_codes_2026_06_25.jsonl`
- Score sets:
  - h96 code-effect 10 seeds: `experiments/results/eval300_assignment_top4_codeeffect_reranker_codeonly_listwise_h96_scores_seed0_9_2026_06_25.jsonl`
  - h128 code-effect 10 seeds: `experiments/results/eval300_assignment_top4_codeeffect_reranker_codeonly_listwise_h128_scores_seed0_9_2026_06_25.jsonl`
  - group/no-op code-only pure-listwise 3 seeds: `experiments/results/eval300_assignment_group_noop_codeonly_purelistwise_scores_seed0_1_2_2026_06_25.jsonl`

## Result

The promoted gate alone remains `-8` Tesseract edits with PARSeq unchanged over 75 Eval300 val groups.

Hybrid replacement does not improve it:

- h96 10-seed reranker:
  - best include-nearest per seed: `-8`, 3 changed groups, 0 false/wrong changes, 0 Tesseract-worsen groups;
  - best force-change per seed: `-8`, 5 changed groups, 1 false change, 1 Tesseract-worsen group.
- h128 10-seed reranker:
  - best include-nearest per seed: `-7`;
  - best force-change per seed: `-8`, still with 1 false change and 1 Tesseract-worsen group.
- group/no-op code-only scores:
  - best include-nearest per seed: `-7`;
  - best force-change per seed: `-8`, again not cleaner than the promoted gate.

The gate-conditioned top4 oracle is only `-9`; the full top4 oracle over all groups is `-15`. This means the promoted gate's current change set itself is now a bottleneck. Swapping code rankers only inside those 5 changed groups cannot deliver a meaningful next jump.

## Conclusion

Negative diagnostic. Do not run actual `.oscr` streams for these hybrid policies.

The next useful direction is to expand the change set safely with a candidate-conditioned selector that can recover more of the full top4/top8 oracle headroom while retaining the promoted gate's precision floor. A fixed high-precision gate plus ranker substitution is too constrained.

W&B offline run: `qd8ol5zr` (`eval300_hybrid_imgdiff_gate_codeeffect_negative_2026_06_25`).
