# eval300_compact_section_accounting_nearest_current_top4_top8_2026_06_26

W&B: `fjr97j6p`

Actual `.oscr` section accounting over existing MLIC++ latent-codebook streams. This is an accounting diagnostic only; no streams are regenerated.

| set | streams | selected streams | mean total bpp | mean enhancement payload bpp | mean optional table bpp | selected optional bytes | merge+drop-header saving bpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest_split | 225 | 75 | 3.849649 | 0.028989 | 0.100831 | 103.00 | 0.064242 |
| nearest_compact | 225 | 75 | 3.774026 | 0.003781 | 0.050416 | 43.00 | 0.000000 |
| current_split | 225 | 75 | 3.849649 | 0.028989 | 0.100831 | 103.00 | 0.064242 |
| current_compact | 225 | 75 | 3.774026 | 0.003781 | 0.050416 | 43.00 | 0.000000 |
| top4_split | 225 | 75 | 3.849649 | 0.028989 | 0.100831 | 103.00 | 0.064242 |
| top4_compact | 225 | 75 | 3.774026 | 0.003781 | 0.050416 | 43.00 | 0.000000 |
| top8_split | 225 | 75 | 3.849649 | 0.028989 | 0.100831 | 103.00 | 0.064242 |
| top8_compact | 225 | 75 | 3.774026 | 0.003781 | 0.050416 | 43.00 | 0.000000 |

## Interpretation

Split and compact rows use the same selected-candidate decisions and base payloads, so any difference here is pure transmitted-format accounting rather than model or selector quality. The compact TEXT_SYMBOLS section removes fixed per-stream gate/code payload headers and one optional section-table entry for selected streams; the selector problem is unchanged, but actual bpp is lower and should be used for future reported OSCAR-LIC points.
