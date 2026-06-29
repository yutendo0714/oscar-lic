# eval300_interaction_smoke20_detail_audit_2026_06_26

This is a diagnostic re-analysis of existing counted `.oscr` interaction smoke outputs.
It does not train or promote a selector.

## Summary

### tesseract

| profile | additive delta | pair delta | residual | pair vs best single | pair/additive gain | pair/best-single gain | pair beats/ties/worse best |
|---|---:|---:|---:|---:|---:|---:|---|
| unicode_strict_v1 | -59 | -10 | 49 | 21 | 0.1695 | 0.3226 | 0/9/11 |
| latin_alnum_ci_v1 | -40 | -6 | 34 | 17 | 0.1500 | 0.2609 | 0/13/7 |

Conditional second-candidate totals:

| profile | B after A delta | A after B delta |
|---|---:|---:|
| unicode_strict_v1 | 21 | 18 |
| latin_alnum_ci_v1 | 17 | 11 |

### parseq

| profile | additive delta | pair delta | residual | pair vs best single | pair/additive gain | pair/best-single gain | pair beats/ties/worse best |
|---|---:|---:|---:|---:|---:|---:|---|
| unicode_strict_v1 | -8 | -4 | 4 | 0 | 0.5000 | 1.0000 | 0/20/0 |
| latin_alnum_ci_v1 | -2 | -1 | 1 | 0 | 0.5000 | 1.0000 | 0/20/0 |

Conditional second-candidate totals:

| profile | B after A delta | A after B delta |
|---|---:|---:|
| unicode_strict_v1 | 0 | 0 |
| latin_alnum_ci_v1 | 0 | 0 |

## Interpretation

Negative deltas mean fewer OCR edit errors. A positive interaction residual means the pair underperforms the additive single-candidate utility estimate.

The important allocator signal is `pair vs best single`: if this is positive, sending two individually useful candidates is worse than sending only the better single candidate. That directly argues for conditional marginal utility or subset-aware allocation rather than independent score summation.
