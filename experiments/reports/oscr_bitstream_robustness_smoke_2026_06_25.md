# OSCR Bitstream Robustness Smoke

**Date:** 2026-06-25  
**W&B:** offline run `4369gwqs` (`oscr_bitstream_robustness_smoke_2026_06_25`)  
**Scope:** one promoted `.oscr` stream with `BASE_MAIN`, `TEXT_GATE` and `TEXT_MAIN` sections. This is a smoke test, not a full bit-error benchmark.

## Stream

- Stream: `experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_pooled_imgdiff_model1_threshold05_explicit_q005_2026_06_25/streams/006_509194_12_hubble_1_88.oscr`
- SHA256: `e875264aacd534105e0a34f75000e8bb47895df36ff59534161c977131ed7cfa`
- Size/bpp: `1003` bytes, `2.089583` bpp
- Sections:
  - `BASE_MAIN`: type `2`, codec `1001`, `812` payload bytes
  - `TEXT_GATE`: type `10`, codec `1`, `16` payload bytes
  - `TEXT_MAIN`: type `12`, codec `3`, `7` payload bytes

## Findings

| case | expected behavior | observed |
|---|---|---|
| original stream | verifies and parses | OK, 3 sections |
| base-only repack | verifies and decodes as same base reconstruction | OK, `900` bytes / `1.875` bpp, max abs delta vs original base `0.0` |
| flip first byte of `TEXT_MAIN` | default verifier rejects corruption | OK, `BitstreamError: file CRC mismatch` |
| parse `TEXT_MAIN` flip with CRC disabled | base payload remains independently decodable | OK, base payload SHA unchanged, base decode max abs delta `0.0` |
| flip first byte of `BASE_MAIN` | default verifier rejects corruption | OK, `BitstreamError: file CRC mismatch` |
| decode corrupted `BASE_MAIN` with CRC disabled | unsafe and must not be used | MLIC++ attempted a huge allocation and raised CUDA OOM |
| truncate before `TEXT_MAIN` payload | verifier rejects truncated stream | OK, declared length mismatch |

## Interpretation

The current OSCR CRC design detects optional text-enhancement corruption, and verified `BASE_MAIN` can be repacked into a smaller base-only stream with identical base reconstruction. This supports the base-layer independence claim.

However, the full-container CRC currently causes the default parser to reject any payload corruption, even if only optional enhancement bytes are affected. A future robustness profile should support explicit per-section recovery or unequal error protection if the desired behavior is "decode base after enhancement damage."

Security note: never decode corrupted `BASE_MAIN` with CRC disabled. This smoke triggered an MLIC++ CUDA out-of-memory path after one flipped byte, consistent with the repository security rule that malicious checkpoints/bitstreams must not be treated as safe inputs.

## Artifact Hashes

```text
cf90810c6cca0d5865bb3c4254465a0f1d270bc35f79b677e0024c08f91bfdf8  scripts/smoke_oscr_bitstream_robustness.py
6eac281bdd1cfe01ad6583a49e51b37a806fb27c32cf037adde215ad0229216b  experiments/results/oscr_bitstream_robustness_smoke_seed0_stream006_2026_06_25.json
e875264aacd534105e0a34f75000e8bb47895df36ff59534161c977131ed7cfa  experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_pooled_imgdiff_model1_threshold05_explicit_q005_2026_06_25/streams/006_509194_12_hubble_1_88.oscr
```
