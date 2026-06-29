# eval300_top8_topk_prior_latent_verifier_2026_06_26

Top-k-prior candidate verifier over local latent/code tensors.
This is a table-level diagnostic and does not export counted `.oscr` streams.

## Results

| target | seed | train positives | val positives | train changed | train exact | train false+wrong | train T | val changed | val exact | val false+wrong | val T | val P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact_changed` | 0 | 30 | 6 | 30 | 30 | 0 | -41 | 3 | 1 | 2 | -3 | 0 |
| `exact_changed` | 1 | 30 | 6 | 30 | 30 | 0 | -41 | 4 | 1 | 3 | -1 | 0 |
| `exact_changed` | 2 | 30 | 6 | 30 | 30 | 0 | -41 | 3 | 1 | 2 | -3 | 0 |
| `safe_improve` | 0 | 34 | 6 | 8 | 7 | 1 | -11 | 0 | 0 | 0 | 0 | 0 |
| `safe_improve` | 1 | 34 | 6 | 11 | 10 | 1 | -14 | 0 | 0 | 0 | 0 | 0 |
| `safe_improve` | 2 | 34 | 6 | 9 | 8 | 1 | -11 | 0 | 0 | 0 | 0 | 0 |

## Interpretation

This tests whether materially richer local latent/code evidence can turn the N097 top-k prior into a safer verifier. Promotion still requires beating the current actual-bitstream -8 Tesseract floor with low false/wrong changes and PARSeq no worse.
