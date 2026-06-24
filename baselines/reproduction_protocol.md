# Baseline Reproduction Protocol

## 1. Immutable inputs

Record repository commit, patches, environment lock, checkpoint SHA256, dataset manifest SHA256 and command. Do not use a branch name alone in an experiment.

## 2. Image handling

- Load as RGB 8-bit unless normative baseline specifies another representation.
- Preserve original dimensions in metadata.
- Apply model-required padding deterministically.
- Compute primary bpp using total file bytes divided by original `H×W`.
- Save decoded PNG only for inspection; do not include PNG container size in codec rate.
- Record whether metric is on clipped/rounded 8-bit reconstruction or float output. Primary RGB PSNR is on rounded 8-bit unless paper protocol requires otherwise; keep both if needed.

## 3. Actual versus estimated rate

Run both when available:

- `estimated_bits = -sum(log2 likelihood)`
- `actual_bits = 8 * serialized_file_size`

Save their difference per image and stream. At low bpp, header/index overhead can dominate.

## 4. Timing

- 10 warm-up images or iterations.
- synchronize GPU before/after timed region.
- report median, p10, p90 and mean over images.
- separate transform, entropy coding, I/O and OCR/utility prediction.
- report CPU model and thread count for entropy coder.

## 5. Reproduction levels

- L0: code imports and toy round trip
- L1: public checkpoint on 5-image smoke set
- L2: full standard evaluation set
- L3: training curve/checkpoint reproduction

OSCAR implementation may start after MLIC++ reaches L1 and the metric pipeline passes, but publication requires at least L2 for main baselines.

## 6. VTM/JPEG AI

Record exact software tag, configuration file, color conversion, chroma format, bit depth, QP/quality points and command. A BD-rate comparison is invalid if these differ from the cited reference without disclosure.
