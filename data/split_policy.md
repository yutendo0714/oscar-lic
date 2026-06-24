# Split and Leakage Policy

## Primary rule

Test images and their crops, OCR outputs, utility labels, captions and metadata must not affect training, model selection, budget selection or stopping.

## Required holdouts

- image/document identity
- source website/application where possible
- font family for synthetic tests
- background image source
- language/script subset
- camera/sensor for RAW
- OCR architecture family

## Duplicate audit

Compute exact SHA256, perceptual hash and resized-image near-duplicate candidates across all splits. Review near duplicates manually or with a logged threshold. Crops from one source image inherit the source split.

## OCR pretraining overlap

Most pretrained OCR systems have incompletely documented training corpora. Treat overlap as a limitation, use multiple model families, and add a human/ground-truth evaluation rather than claiming a clean unseen distribution.

## Hyperparameter selection

Use validation only. The final test suite is executed after configs and operating points are frozen. Additional test-driven debugging creates a new versioned test phase and must be disclosed.
