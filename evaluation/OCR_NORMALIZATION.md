# OCR Text Normalization Specification

Normalization can reverse method rankings. Therefore profiles are versioned and fixed before test evaluation.

## 1. General principles

- Preserve the raw reference and hypothesis.
- Apply the same deterministic function to every method.
- Never repair a prediction using a lexicon unless the OCR model's declared standard decoder uses that lexicon for all methods.
- Do not remove characters merely because one method fails on them.
- Unicode normalization does not replace script-aware evaluation.

## 2. Profiles

### `raw_exact_v1`

Use for exact integrity and security-sensitive text.

1. Convert CRLF/CR to LF only.
2. No Unicode compatibility normalization.
3. Preserve case, punctuation, repeated whitespace and line breaks.

### `unicode_strict_v1` — primary

1. Convert CRLF/CR to LF.
2. Apply Unicode NFKC.
3. Replace each run of Unicode whitespace with one ASCII space.
4. Strip leading/trailing whitespace.
5. Preserve case.
6. Preserve punctuation, symbols, diacritics and script.

Rationale: NFKC removes representation-level variants while retaining semantic distinctions that compression may corrupt.

### `unicode_casefold_v1`

Apply `unicode_strict_v1`, then Unicode `casefold()`. Use only as a secondary case-insensitive view.

### `latin_alnum_ci_v1` — legacy STR compatibility

1. Apply NFKC and Unicode casefold.
2. Retain Unicode letters and numbers.
3. Convert whitespace runs to one space.
4. Remove punctuation and symbols.
5. Strip.

This profile can hide punctuation, currency and sign errors and is never the sole primary metric.

### `digits_exact_v1`

1. Apply NFKC.
2. Preserve only positions whose reference belongs to a numeric-field annotation.
3. Preserve signs, decimal separators, currency marks and date separators according to the dataset schema.
4. Require exact match.

This is a dataset-field evaluation profile, not a generic filter over arbitrary text.

## 3. Character unit

Default CER in the bootstrap tools uses Unicode code points. For languages with combining sequences or emoji, final multilingual experiments should additionally report extended grapheme-cluster CER using a locked implementation such as the `regex` package's `\X`. Name that metric `gCER`, not CER, and record package/version.

## 4. Word unit

Default WER uses runs separated by normalized ASCII whitespace. It is inappropriate as the only sequence metric for scripts without explicit word spaces.

For Chinese, Japanese, Thai and similar scripts:

- CER/gCER remains primary.
- If a word metric is required, lock a language-specific segmenter and dictionary version.
- Segmentation is applied to both reference and hypothesis after normalization.
- Results get a distinct profile, e.g. `ja_sudachi_v1`.

## 5. Empty references

- Empty/illegible annotations are excluded according to the dataset manifest, not method-by-method.
- If an intentionally empty reference is retained, an empty prediction is exact; any nonempty prediction consists of insertions.
- Aggregate denominators and excluded counts must be reported.

## 6. Line and reading order

For full-page OCR:

- Preserve dataset-defined reading order when available.
- Otherwise use one frozen ordering algorithm for ground truth and predictions.
- Report layout-independent word matching separately from ordered line CER.
- Do not let a method-specific OCR engine choose the ground-truth reading order.

## 7. Required stored fields

Each recognition record should contain:

```json
{
  "sample_id": "...",
  "region_id": "...",
  "reference_raw": "...",
  "hypothesis_raw": "...",
  "normalization_profile": "unicode_strict_v1",
  "reference_normalized": "...",
  "hypothesis_normalized": "...",
  "script": "Latn",
  "language": "und",
  "ocr_model_id": "...",
  "recognition_mode": "end_to_end"
}
```

Where licenses or privacy rules prohibit storing strings, store salted hashes and aggregate edit counts; document the reduced auditability.

## 8. Regression examples

The unit tests must preserve these expectations:

| Reference | Hypothesis | Profile | Expected |
|---|---|---|---|
| `ＡＢＣ  12` | `ABC 12` | unicode_strict_v1 | exact |
| `Price: $8` | `price 8` | latin_alnum_ci_v1 | exact |
| `Price: $8` | `price 8` | unicode_strict_v1 | not exact |
| `Straße` | `STRASSE` | unicode_casefold_v1 | exact |
| `8` | `3` | all profiles | one substitution |
