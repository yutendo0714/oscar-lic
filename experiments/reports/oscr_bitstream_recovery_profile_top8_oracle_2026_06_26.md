# OSCR Section-Level Recovery Profile Smoke

**Date:** 2026-06-26  
**Experiment ID:** `oscr_bitstream_recovery_profile_top8_oracle_seed0_stream011_2026_06_26`  
**W&B:** offline run `zqz4zdmu` (`oscr-bitstream-recovery-profile-top8-oracle-seed0-stream011-2026-06-26`)  
**Scope:** implementation and smoke test of an explicit section-level recovery parser for one top-8 oracle actual `.oscr` stream. The default parser remains fail-closed.

## Implementation

- Added `parse_container_recovery()` in `src/oscarlic/bitstream.py`.
- Added `SectionRecoveryIssue` and `RecoveredContainer`.
- Recovery requires valid magic/version/dimensions, matching `total_bytes`, valid header/table CRC, in-range non-overlapping sections, and per-section payload CRC checks.
- Payloads with failed section CRC are quarantined.
- Required sections default to `BASE_MAIN` type `2`; corrupt required sections raise `BitstreamError`.
- Updated `scripts/smoke_oscr_bitstream_robustness.py` so optional `TEXT_MAIN` salvage uses the recovery profile rather than relying on generic `verify_crc=False` parsing.
- Documented the profile in `docs/BITSTREAM_SPEC.md`.

## Stream

- Stream: `experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/streams/011_a01-058x-00-00.oscr`
- Stream SHA256: `400700fdbcfee34f22eb0fd0c3fbb0f331163096624a537f5746d03b6ec564e5`
- Checkpoint: `experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar`
- Checkpoint SHA256: `67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559`
- Device: `cuda`, `NVIDIA GeForce RTX 4070 Ti SUPER`
- Size/bpp: `1715` bytes, `2.977431` bpp

## Findings

| case | observed |
|---|---|
| original stream | default CRC parse OK, 3 sections |
| base-only repack | verifies at `1612` bytes / `2.798611` bpp, max abs delta vs original base `0.0` |
| `TEXT_MAIN` first-byte flip | default parse rejects with file CRC mismatch |
| `TEXT_MAIN` recovery profile | OK, returns 2 valid sections, quarantines section index `2` / type `12`, `file_crc_ok=false` |
| recovered base from `TEXT_MAIN` flip | base payload SHA unchanged, base decode max abs delta `0.0` |
| `BASE_MAIN` first-byte flip | default parse rejects with file CRC mismatch |
| `BASE_MAIN` recovery profile | rejects with `required section 0 payload CRC mismatch` |
| truncated before `TEXT_MAIN` payload | rejected by length check |

## Interpretation

This converts the earlier no-CRC salvage observation into a safer explicit recovery profile. The default parser still rejects any corrupted file; recovery must be opted into and only returns payloads whose own section CRC verifies. Optional enhancement damage can now be represented as a quarantined section while keeping verified `BASE_MAIN` available for base-only reconstruction.

This is not FEC and does not correct bit errors. It supports a defensible robustness statement for base-layer independence under optional-section corruption, but publication claims about recovery probability or unequal error protection still require a packetization/FEC design with counted overhead.

## Artifact Hashes

```text
dcab444a069cd09f5722b7cddb170b86d5c0f8cfa0c6137a99dfed3bb76c758c  src/oscarlic/bitstream.py
95bfc3dbab19e793288b296c3eb81dd72ba4239f0e776523754b589c78963216  tests/test_bitstream.py
69d4230885c446e3a4fd404c7cdf336ae15792d0cbf1538cf8397f080d7e9b81  scripts/smoke_oscr_bitstream_robustness.py
d884106626381ca23ae526bb08c8cefab27f69b77787d634c6951ac9f3ad710c  docs/BITSTREAM_SPEC.md
99265da2569e867ccc46d3e8b712119416fba939cee3c18b53c322f96691636a  experiments/results/oscr_bitstream_recovery_profile_top8_oracle_seed0_stream011_2026_06_26.json
400700fdbcfee34f22eb0fd0c3fbb0f331163096624a537f5746d03b6ec564e5  experiments/results/mlicpp_latent_codebook_seed0_val75_top25_k64_utilityw_a8_shortlist_oracle_top8_scoreseed2_explicit_q005_2026_06_26/streams/011_a01-058x-00-00.oscr
67af8c950a4e8ae03da9bc95b87d13fd7831063ad12b7df1f872154e0082c559  experiments/checkpoints/mlicpp/mlicpp_mse_0025_corrected.pth.tar
```
