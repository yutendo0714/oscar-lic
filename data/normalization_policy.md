# Image and Annotation Normalization

## Images

- preserve raw originals
- decode with recorded library/version
- convert to RGB with explicit color-management policy
- no implicit EXIF rotation; either apply and update boxes or preserve consistently
- record bit depth and ICC profile
- do not resize evaluation images unless the protocol defines it

## Text annotations

Store both raw transcription and normalized variants. Never overwrite raw labels. Each metric result records the normalization profile from `evaluation/OCR_NORMALIZATION.md`.

## Geometry

Canonical representation is a polygon in original-image coordinates. Derived axis-aligned boxes and rectified crops are cached with transformation matrices.
