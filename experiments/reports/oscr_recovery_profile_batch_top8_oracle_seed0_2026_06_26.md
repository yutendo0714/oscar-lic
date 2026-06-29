# OSCR Recovery Profile Batch Audit: Top-8 Oracle Seed0

**Date:** 2026-06-26  
**Experiment ID:** `oscr_recovery_profile_batch_top8_oracle_seed0_2026_06_26`  
**W&B:** offline run `n0g68a63` (`oscr-recovery-profile-batch-top8-oracle-seed0-2026-06-26`)  
**Scope:** structural recovery-profile audit over all 75 actual `.oscr` streams from the top-8 oracle seed0 validation export. This is not a bit-error-rate benchmark and does not decode every recovered base image; N109 provides the decoded single-stream smoke.

## Inputs

- Stream directory: `experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/streams`
- Parent stream manifest: `experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/results.jsonl`
- Parent stream manifest SHA256: `0dab42430c6dc221c14d5f2f1f78f37a612ae66dd9e229c2c8c1c0026ca6cc5b`

## Summary

| metric | count |
|---|---:|
| streams audited | 75 |
| streams with `TEXT_MAIN` | 25 |
| default parser rejects `TEXT_MAIN` first-byte corruption | 25 / 25 |
| recovery profile recovers `TEXT_MAIN` first-byte corruption | 25 / 25 |
| default parser rejects `BASE_MAIN` first-byte corruption | 75 / 75 |
| recovery profile rejects `BASE_MAIN` first-byte corruption | 75 / 75 |

All `TEXT_MAIN` corruptions were recovered by quarantining the corrupt optional section while preserving the base payload SHA. All `BASE_MAIN` corruptions were rejected by the recovery profile because `BASE_MAIN` type `2` is required.

## Interpretation

The new section-level recovery profile is consistent across the seed0 top-8 actual stream set: optional text-enhancement damage can be isolated structurally, while required base damage is not passed downstream. This is stronger evidence than the previous single-stream smoke for the container behavior, but it remains a CRC-based detection/recovery-boundary audit, not error correction.

For a paper claim, phrase this as: verified base sections remain recoverable after optional section corruption under an explicit recovery parser. Do not phrase it as guaranteed bit-error robustness or UEP without adding counted packetization/FEC and a channel model.

## Artifact Hashes

```text
13c988e65ae5a6a3b627cb38e7c19e9dee80be3daa99dcf5b1cefd6a608250e2  scripts/audit_oscr_recovery_profile_batch.py
dcab444a069cd09f5722b7cddb170b86d5c0f8cfa0c6137a99dfed3bb76c758c  src/oscarlic/bitstream.py
032b89bc4b2c77e8debc0ba6cdec783c680963b86c41b8fdb6be298bedf38f50  experiments/results/oscr_recovery_profile_batch_top8_oracle_seed0_2026_06_26.json
0dab42430c6dc221c14d5f2f1f78f37a612ae66dd9e229c2c8c1c0026ca6cc5b  experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/results.jsonl
76523e26f09797b499879ec80997d1d0e6f41172cc1851749bc977bc3b83b9e5  experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/summary.json
```
