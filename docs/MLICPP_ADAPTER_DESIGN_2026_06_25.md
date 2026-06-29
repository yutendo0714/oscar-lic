# MLIC++ Adapter Design Notes

**Date:** 2026-06-25  
**Base:** corrected MLIC++ lambda 0.0250 checkpoint  
**Purpose:** define a conservative adapter boundary for OSCAR-LIC without weakening the base-model novelty axis.

## What The Corrected MLIC++ Base Emits

The active corrected MLIC++ configuration uses `N=192`, `M=320`, `slice_num=10`, and therefore 10 uniform `y` slices of 32 channels. The base codec has the following order:

1. `g_a(x)` produces `y`.
2. `h_a(y)` produces `z`; `z` is encoded by the entropy bottleneck.
3. `h_s(z_hat)` produces hyper scales and means.
4. `y` is chunked into 10 channel slices.
5. Each slice is encoded in checkerboard order: anchor first, non-anchor second.
6. Slices after index 0 use channel context, global inter context, global intra context, local context and hyper parameters.
7. Latent residual prediction is applied after each anchor/non-anchor quantization step.
8. `g_s(y_hat)` reconstructs the image.

The upstream `compress()` implementation appends every `y` anchor and non-anchor symbol into one `symbols_list` and one `indexes_list`, then calls `BufferedRansEncoder.encode_with_indexes(...)` once and `flush()` once. The output therefore contains one opaque main `y_string` plus the entropy-bottleneck `z_strings`.

## Critical Rate-Accounting Consequence

MLIC++ does not expose actual per-slice, per-anchor, per-tile, or per-candidate bytes in the emitted bitstream. Likelihoods can estimate where bits are likely spent, but they are not the same as transmitted bytes.

For OSCAR-LIC claims, all bpp must be computed from complete serialized files or from independently serialized packets whose headers, indexes, gates and protection bytes are included. Any per-candidate oracle that uses MLIC++ likelihoods must be labeled as a design diagnostic until the payload is serialized as independently countable sections.

## Recommended Adapter Boundary

Keep upstream MLIC++ as a frozen base codec first:

- `BaseCodecAdapter.compress_base(x) -> BaseCodecResult`
- `BaseCodecResult` records complete base bytes, base stream path, padded/original shape, decode timing, checkpoint hash and model commit.
- The OSCAR enhancement stream should be a separate counted section in `docs/BITSTREAM_SPEC.md`.
- Base-only decoding must remain valid when all OSCAR sections are dropped.

This preserves the strong MLIC++ base while allowing controlled enhancement experiments.

## Candidate Unit Recommendation

The safest first candidate geometry is:

- latent domain: `y` at `H/16 x W/16`;
- channel grouping: one MLIC++ slice at a time, optionally subdivided into 8-channel groups later;
- spatial grouping: small latent tiles, initially `4x4` or `8x8` cells;
- decode dependency: later slices depend on earlier reconstructed slices, so candidates should be ranked but serialized as independent enhancement residual packets rather than by reordering the base `y` stream.

For the current checkpoint, each slice has 32 channels. A `slice x 4x4 latent-tile` candidate is a good first compromise: it is small enough to localize text but not so small that packet headers dominate immediately.

## Two Valid Implementation Paths

Path A: frozen-base enhancement stream.

- Do not modify the upstream main `y_string`.
- Predict utility over latent tiles and send a learned residual/enhancement packet for selected candidates.
- Count gate, hyper, main enhancement and CRC/header bytes in `.oscr`.
- This is the preferred near-term path because it preserves base reproducibility.

Path B: split-stream base reserialization.

- Modify MLIC++ compression to flush multiple ANS streams, e.g. per slice or per tile group.
- Modify decompression to consume those streams in exactly the same dependency order.
- Count every stream header and table entry.
- This enables true per-base-candidate actual bytes but changes the base bitstream and needs careful ablation against unmodified MLIC++.

## Immediate Diagnostics

`scripts/inspect_mlicpp_latents.py` records complete actual MLIC++ stream bpp and likelihood-derived slice/anchor/non-anchor estimates. The estimate is useful for choosing candidate layout and for detecting whether high-rate regions correlate with OCR utility, but it is not a reported compression rate.

`src/oscarlic/candidates.py` now defines tested utilities for latent `y` tiling, MLIC++ slice/channel-group indexing, and gate payload byte estimates for bitset vs sparse index-list syntax. For a tiny word-crop latent grid such as `2x5`, 10 whole-slice candidates already need about 16-18 gate payload bytes depending on tile size. For a natural-image latent grid such as `64x64`, a `4x4` tile layout gives 2560 candidates and about 334 bytes for a 10% selected bitset gate. Tile granularity is therefore a rate-control decision, not just a localization decision.

`scripts/run_mlicpp_oscr_smoke.py` can now append counted dummy `TEXT_GATE` and `TEXT_MAIN` sections while decoding only `BASE_MAIN`. This verifies the intended forward-compatible behavior: optional sections increase transmitted bytes but do not affect frozen-base reconstruction. `scripts/analyze_mlicpp_candidate_overheads.py` estimates the incremental gate/main section cost directly from image dimensions and MLIC++ latent shapes. On the CLIC9 smoke split, slice-level `4x4` latent tiles at 10% selection cost about 0.088 incremental bpp with the current 40-byte global + 40-byte/selected dummy enhancement model; `8x8` reduces this to about 0.025 bpp. On word-crop splits the same syntax costs roughly 0.6-0.7 incremental bpp because the fixed two-section table and small image area dominate.

`scripts/inspect_mlicpp_decoded_yhat.py` provides the first true latent adapter. It reproduces the upstream `decompress()` dependency order, returns decoded `y_hat`, and verifies that `g_s(y_hat)` matches upstream `x_hat` exactly. The smoke result is exact on CLIC2, Text24 and Eval100. Top-RMS activation candidates concentrate strongly in slice 2, whereas likelihood-derived bit diagnostics were more mid/late-slice heavy on text crops. This means activation magnitude alone is not a sufficient utility proxy; the first learned allocator should combine decoded-latent statistics, likelihood/uncertainty diagnostics and OCR-derived utility labels.

`scripts/run_mlicpp_latent_residual_stub.py` validates the first non-dummy `TEXT_MAIN` path by sending quantized latent residuals for selected candidates. The path works end to end, but raw int8 residual tensors are too expensive: on Text24, 5% selected candidates cost 4.804 total bpp and only improve PSNR by 0.04 dB; 10% costs 5.890 bpp and improves PSNR by 0.066 dB. OCR effects are mixed and do not justify the raw payload. This should be treated as a negative control and a decoder-path validation, not as the final OSCAR enhancement design.

`scripts/run_mlicpp_latent_codebook_stub.py` tests a compact decoder-known codebook variant. On Text24, a split-trained K16 codebook at 5% selection reduces enhancement payload from 1.61 bpp to 0.066 bpp, but it gives only +0.011 dB PSNR and no OCR gain over MLIC++ base. The conclusion is useful: the payload should indeed be codebook/low-dimensional, but residual-MSE codebooks and residual-magnitude selection are not enough. The next version needs OCR-utility labels and a learned selector/codebook objective.

Initial questions to answer before large training:

- Do high estimated-bit latent regions overlap OCR-sensitive crops?
- Are OCR-sensitive gains concentrated in early or late slices?
- Does PARSeq-only utility disagree with held-out Tesseract utility at the same latent regions?
- How large is the gate and packet overhead for plausible candidate counts?

## Near-Term Decision

Proceed with Path A first. It is cleaner for international-paper evidence because base bpp remains exactly reproducible and every OSCAR byte can be isolated in a counted section. Only move to Path B if the enhancement stream cannot beat strong controls after gate overhead and held-out OCR evaluation.
