# mlicpp_latent_codebook_compact_section_smoke5_2026_06_26

W&B: `8wefk30c`

Split-section versus compact-section OSCR smoke. Both runs use the same image split, selected candidates and loaded codebook; only optional enhancement packing changes.

| streams | split mean bytes | compact mean bytes | mean saving bytes | dataset saving bpp | split payload bytes | compact payload bytes | recon identical |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 1178.20 | 1118.20 | 60.00 | 0.156250 | 23.00 | 3.00 | True |

## Interpretation

The compact optional section preserves base payloads and reconstructions while saving exactly the expected one optional section table entry plus split payload headers for these one-candidate streams. This improves actual bpp accounting, but it does not change candidate selection or OCR utility.
