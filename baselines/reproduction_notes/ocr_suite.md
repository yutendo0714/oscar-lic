# OCR Suite Reproduction and Separation

## Train teacher snapshot

PARSeq model hub:

- repository commit: `1902db043c029a7e03a3818c616c06600af574be`
- main repository license: Apache-2.0
- reproduced ABINet component: BSD notice
- reproduced CRNN component: MIT notice
- stated runtime generation: Python >=3.9, PyTorch >=2.0
- model hub names include `parseq`, `abinet`, `crnn`, `trba`, `vitstr`

Torch Hub is convenient but a moving-cache risk. At first download:

1. checkout the pinned repository locally;
2. load from the local path or force the pinned source;
3. record every downloaded weight file path, URL, size and SHA256;
4. archive resolved model hyperparameters, input transform and tokenizer charset;
5. run a fixed five-word inference regression.

## Required separation

- PARSeq/ABINet/CRNN may generate training gradients or oracle labels.
- TrOCR/Tesseract/PaddleOCR are held out and may not influence checkpoint selection.
- If a held-out OCR is consulted during development, it becomes validation and a new untouched evaluator is required.
- Full model bundles include detector, recognizer, dictionary/tokenizer and preprocessing versions.

## Canonical teacher outputs

The adapter should expose:

```python
OCRTeacherOutput(
    logits,                 # projected or model-native with charset metadata
    visual_features,        # named layer and shape
    sequence_lengths,
    decoded_text,
    confidence,
    charset_id,
    preprocessing_id,
)
```

Do not naively average logits from incompatible charsets. Map to a canonical Unicode target only where mapping is unambiguous; otherwise combine normalized losses/features at the model level.

## First regressions

- exact five-word predictions and confidences
- blank/empty crop behavior
- punctuation/case behavior
- image resizing and aspect-ratio padding
- deterministic repeated inference
- mixed precision drift
- unsupported character behavior

## End-to-end OCR

Recognizer-only crops cannot support the main deployment claim. Freeze a detector and additionally evaluate each held-out OCR's native detector+recognizer pipeline under `evaluation/PROTOCOL.md`.
