# Eval300 Score-Consensus Selector Diagnostic

Date: 2026-06-25

## Purpose

Test whether multiple assignment reranker seeds can safely expand the promoted pooled image-diff gate's change set. The selector tunes score, margin and vote thresholds on train groups, then evaluates once on clean Eval300 val groups (`seed=1`). All selection is parseq-safe at the candidate level.

## Result

The diagnostic is negative.

Train-tuned consensus policies did not beat the promoted actual-bitstream gate (`-8` Tesseract edits, PARSeq unchanged):

- h96 code-effect 10 seeds:
  - false-change-rate max `0.02`: val `-5`, 18 changed groups, 14 false changes, 3 Tesseract-worsen groups;
  - false-change-rate max `0.0`: val `-4`, 15 changed groups, 12 false changes, 3 Tesseract-worsen groups.
- h128 code-effect 10 seeds:
  - both settings select 25 changed groups, produce only `-2`, and create 22 false changes.
- h96+h128 20 voters:
  - false-change-rate max `0.02`: val `-3`, 8 changed groups, 6 false changes;
  - false-change-rate max `0.0`: val `-3`, 8 changed groups, 6 false changes.
- group/no-op code-only 3 seeds:
  - val `-4`, 24 changed groups, 20 false changes.

The full top4 oracle on the same score tables remains `-15` with 10 changed groups and no false changes, so headroom is real. However, score consensus and score margins do not identify the safe extra groups reliably under the clean split.

## Conclusion

Negative diagnostic. Do not promote or run actual `.oscr` streams for these consensus policies.

The next improvement likely requires a better image/latent/text-aware selector, not only agreement among current reranker scores. In particular, train-side false-change constraints do not transfer to the clean val groups for these score features.

W&B offline run: `rm9fssf0` (`eval300_score_consensus_selector_negative_2026_06_25`).
