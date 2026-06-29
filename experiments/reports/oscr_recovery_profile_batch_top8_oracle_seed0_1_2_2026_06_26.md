# OSCR Recovery Profile Batch Audit: Top-8 Oracle Seeds 0/1/2

**Date:** 2026-06-26  
**Experiment ID:** `oscr_recovery_profile_batch_top8_oracle_seed0_1_2_2026_06_26`  
**W&B:** offline run `l5lc49qa` (`oscr-recovery-profile-batch-top8-oracle-seed0-1-2-2026-06-26`)  
**Scope:** structural section-level recovery audit over all top-8 oracle actual `.oscr` validation streams for Eval300 seeds 0/1/2. This aggregates N110-style per-seed audits.

## Summary

| metric | count |
|---|---:|
| batches | 3 |
| streams audited | 225 |
| streams with `TEXT_MAIN` | 75 |
| default parser rejects `TEXT_MAIN` first-byte corruption | 75 / 75 |
| recovery profile recovers `TEXT_MAIN` first-byte corruption | 75 / 75 |
| default parser rejects `BASE_MAIN` first-byte corruption | 225 / 225 |
| recovery profile rejects `BASE_MAIN` first-byte corruption | 225 / 225 |

Per seed, each 75-stream validation split has 25 streams with `TEXT_MAIN`. In all three splits, optional `TEXT_MAIN` first-byte corruption is quarantined by the recovery profile while preserving base payload identity, and required `BASE_MAIN` first-byte corruption is rejected.

## Interpretation

The section-level recovery profile behaves consistently across the full 225-stream top-8 actual validation set. This supports a narrow container-level robustness claim: after optional enhancement corruption, a decoder using the explicit recovery parser can recover verified base sections and refuse corrupted optional sections. It does not establish channel-level bit-error robustness, correction probability, or UEP effectiveness.

For submission writing, this belongs in an implementation/robustness paragraph or appendix table, not in the main RD/OCR comparison table.

## Artifact Hashes

```text
13c988e65ae5a6a3b627cb38e7c19e9dee80be3daa99dcf5b1cefd6a608250e2  scripts/audit_oscr_recovery_profile_batch.py
fe24c1edf5b04eb7cc3a01965ee76b6b3298c83f6b31d5268172f549f9d709a0  scripts/summarize_oscr_recovery_profile_batches.py
dcab444a069cd09f5722b7cddb170b86d5c0f8cfa0c6137a99dfed3bb76c758c  src/oscarlic/bitstream.py
032b89bc4b2c77e8debc0ba6cdec783c680963b86c41b8fdb6be298bedf38f50  experiments/results/oscr_recovery_profile_batch_top8_oracle_seed0_2026_06_26.json
5ec0358be67d82714abf5609d9ca77eec05d71f4b52394faa85004a3cb6bdea6  experiments/results/oscr_recovery_profile_batch_top8_oracle_seed1_2026_06_26.json
c8f3a93b334d2d0d863e910cbc7236d9b5c2502820bf0b082692238c4760aa0d  experiments/results/oscr_recovery_profile_batch_top8_oracle_seed2_2026_06_26.json
f3ae9a71c6382f83da4d5fe1cda5b00880bbefd916c6536ffbc891813ecdb36a  experiments/results/oscr_recovery_profile_batch_top8_oracle_seed0_1_2_2026_06_26.json
0dab42430c6dc221c14d5f2f1f78f37a612ae66dd9e229c2c8c1c0026ca6cc5b  experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/results.jsonl
56171440de91d96bc7ecdc45b146a81f249e1f37ec6c8154f8b5679d25b38afd  experiments/results/mlicpp_latent_codebook_seed1_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/results.jsonl
0dccf8e2c01290026ad4874dfa01c3bf052bb201a4f774d8e2b56599bebaf1be  experiments/results/mlicpp_latent_codebook_seed2_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/results.jsonl
```
