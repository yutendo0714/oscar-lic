# OSCR Bitstream Robustness Smoke: Top-8 Oracle Actual Stream

**Date:** 2026-06-26  
**Experiment ID:** `oscr_bitstream_robustness_top8_oracle_seed0_stream011_2026_06_26`  
**W&B:** offline run `uv6gko0x` (`oscr-bitstream-robustness-top8-oracle-seed0-stream011-2026-06-26`)  
**Scope:** one top-8 shortlist-oracle actual `.oscr` stream with `BASE_MAIN`, `TEXT_GATE`, and `TEXT_MAIN` sections. This is a bitstream safety smoke, not a full bit-error-rate benchmark. The top-8 policy remains an oracle diagnostic and is not deployable.

## Stream

- Stream: `experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/streams/011_a01-058x-00-00.oscr`
- Stream SHA256: `400700fdbcfee34f22eb0fd0c3fbb0f331163096624a537f5746d03b6ec564e5`
- Checkpoint: `experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar`
- Checkpoint SHA256: `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`
- Device: `cuda`, `NVIDIA GeForce RTX 4070 Ti SUPER`
- Size/bpp: `1715` bytes, `2.977431` bpp
- Sections:
  - `BASE_MAIN`: type `2`, codec `1001`, `1524` payload bytes
  - `TEXT_GATE`: type `10`, codec `1`, `16` payload bytes
  - `TEXT_MAIN`: type `12`, codec `3`, `7` payload bytes

## Findings

| case | expected behavior | observed |
|---|---|---|
| original stream | verifies and parses | OK, 3 sections |
| base-only repack | verifies and decodes as same base reconstruction | OK, `1612` bytes / `2.798611` bpp, max abs delta vs original base `0.0` |
| flip first byte of `TEXT_MAIN` | default verifier rejects corruption | OK, `BitstreamError: file CRC mismatch` |
| parse `TEXT_MAIN` flip with CRC disabled | base payload remains independently decodable | OK, base payload SHA unchanged, base decode max abs delta `0.0` |
| flip first byte of `BASE_MAIN` | default verifier rejects corruption | OK, `BitstreamError: file CRC mismatch` |
| decode corrupted `BASE_MAIN` with CRC disabled | unsafe and should not be attempted in this profile | skipped intentionally with `--skip-corrupt-base-decode` |
| truncate before `TEXT_MAIN` payload | verifier rejects truncated stream | OK, declared total length mismatch |

## Interpretation

This confirms the same structural safety boundary on a top-8 oracle actual stream as the earlier promoted-stream smoke: verified `BASE_MAIN` can be repacked and decoded as an independent base layer, and optional `TEXT_MAIN` damage does not alter the base payload when parsed in a no-CRC salvage mode.

The default full-container CRC still rejects any payload damage, including optional enhancement corruption. That is a conservative security posture, not a recovery profile. If OSCAR-LIC claims base-layer recovery after enhancement-layer damage, the container needs explicit per-section verification plus a policy that allows verified required sections to decode while quarantining failed optional sections.

Security note: do not feed CRC-failed `BASE_MAIN` payloads to MLIC++ entropy decoding. A previous smoke triggered a CUDA OOM on a one-byte-corrupted base payload, so this run records the default CRC rejection and skips the unsafe decode path.

## Artifact Hashes

```text
0fd22faccde815c2d1c172afd993ec656bf633c73a2db1e489af68ca10fc1611  scripts/smoke_oscr_bitstream_robustness.py
6feab1f60df28274ec2533734eaa5a183bfb4e0e4a61c07f6c9240cb0484b791  experiments/results/oscr_bitstream_robustness_top8_oracle_seed0_stream011_2026_06_26.json
400700fdbcfee34f22eb0fd0c3fbb0f331163096624a537f5746d03b6ec564e5  experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/streams/011_a01-058x-00-00.oscr
67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559  experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar
```
