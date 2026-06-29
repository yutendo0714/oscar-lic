# eval300_oscr_section_accounting_current_top4_top8_2026_06_26

W&B: `po2m40fu`

Actual `.oscr` section accounting over existing MLIC++ latent-codebook streams. This is an accounting diagnostic only; no streams are regenerated.

| set | streams | selected streams | mean total bpp | mean enhancement payload bpp | mean optional table bpp | selected optional bytes | merge+drop-header saving bpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 225 | 75 | 3.849649 | 0.028989 | 0.100831 | 103.00 | 0.064242 |
| top4_oracle | 225 | 75 | 3.849649 | 0.028989 | 0.100831 | 103.00 | 0.064242 |
| top8_oracle | 225 | 75 | 3.849649 | 0.028989 | 0.100831 | 103.00 | 0.064242 |

## Interpretation

The current/top4/top8 streams use the same selected-candidate budget, so OCR gains are not from extra payload. For selected word-crop streams, optional section-table and fixed gate/code headers dominate the few logical code/index bytes. A compact combined optional section would not change the selector problem, but it can reduce actual bpp and should be part of the next compact-symbol implementation.
