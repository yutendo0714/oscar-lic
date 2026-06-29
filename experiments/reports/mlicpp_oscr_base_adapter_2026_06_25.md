# MLIC++ OSCR Base Adapter Smoke

**Date:** 2026-06-25 JST  
**Goal:** verify a frozen-base MLIC++ adapter that can be decoded from an `.oscr` container with exact total-byte accounting.

## Implementation

`scripts/run_mlicpp_oscr_smoke.py` serializes the upstream MLIC++ body with `write_body(shape, strings)` and stores it as one `.oscr` `BASE_MAIN` section:

- section type: `2` (`BASE_MAIN`);
- codec: `1001` (`MLICPP_UPSTREAM_BODY`);
- model id: `1001`;
- model version: `1`;
- sections: one required base payload only.

The `.oscr` file is then parsed, CRC-checked, and decoded by extracting `BASE_MAIN`, running upstream `read_body`, then calling `MLICPlusPlus.decompress`. This verifies base-only independent decoding through the OSCAR container.

With one section, the fixed container overhead is 88 bytes: 44-byte header + 40-byte section table + 4-byte file CRC.

## Results

| Split | Images | OSCR total bpp | Base payload bpp | Fixed overhead | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| CLIC small | 9 | 0.7347308907 | 0.7334314760 | 88 bytes/image | 35.0448878656 | 0.9871003959 |
| text crops | 24 | 2.9644138327 | 2.7137528068 | 88 bytes/image | 35.1379361749 | null |

CLIC overhead is tiny because images are large. On small word crops, 88 bytes is already about 0.25 bpp on average. This strongly constrains OSCAR packet design: tiny independent packets will lose unless their utility gain is large enough to cover header/index/CRC overhead.

## Implications

1. Frozen-base enhancement sections are now technically viable without modifying MLIC++ entropy coding.
2. Candidate packet groups should be coarse enough to amortize section/subpacket metadata.
3. Per-candidate gate/index bytes must be included in the next oracle, not only payload bytes.
4. For crop-level studies, report both corpus-level bpp and per-image mean bpp because fixed overhead changes interpretation.
5. `src/oscarlic/candidates.py` now provides tested candidate layout and gate-byte utilities for the next implementation step.

## Frozen Artifacts

- Script: `scripts/run_mlicpp_oscr_smoke.py`
- CLIC result dir: `experiments/results/mlicpp_oscr_clic9_smoke_2026_06_25`
- CLIC results SHA256: `56375675e46180f2ead93c46fa369360f4754bbac7010d06fe7af0d2c007bc59`
- CLIC summary SHA256: `1ea110ffe4f5c091a62e121507d827b2e99f0e4f58a89cebaceaa5f221b3b143`
- Text result dir: `experiments/results/mlicpp_oscr_text24_smoke_2026_06_25`
- Text results SHA256: `7ddbf41ae1a3263d8a9ce2df6b0d54d8453a13e0dfaa744cf960a353a4ceee64`
- Text summary SHA256: `c34eed02818a2cc4e60cb7e29526e01da750c29537f1ad6d9f3d53a0bd202feb`
