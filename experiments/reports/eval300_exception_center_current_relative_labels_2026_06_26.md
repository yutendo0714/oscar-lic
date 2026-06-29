# Current-Relative Exception Labels

This artifact labels the executed N141 exception-center stream against the matched current compact stream.
It is not a full counterfactual oracle over all possible replacement/drop actions.

## Action Summary

| action | rows | beneficial | neutral | harmful | mixed | current bpp | exception bpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| drop_current | 28 | 4 | 18 | 5 | 1 | 3.269558 | 3.141582 |
| keep_base | 100 | 0 | 100 | 0 | 0 | 3.388301 | 3.388301 |
| replace_with_exception | 22 | 1 | 19 | 2 | 0 | 3.154800 | 3.154800 |

## OCR Delta By Action

### tesseract

| action | profile | rows | delta errors | improved | worsened | equal | delta exact |
|---|---|---:|---:|---:|---:|---:|---:|
| drop_current | unicode_strict_v1 | 28 | 3 | 3 | 6 | 19 | 0 |
| drop_current | latin_alnum_ci_v1 | 28 | 4 | 3 | 5 | 20 | -1 |
| keep_base | unicode_strict_v1 | 100 | 0 | 0 | 0 | 100 | 0 |
| keep_base | latin_alnum_ci_v1 | 100 | 0 | 0 | 0 | 100 | 0 |
| replace_with_exception | unicode_strict_v1 | 22 | 0 | 1 | 1 | 20 | 0 |
| replace_with_exception | latin_alnum_ci_v1 | 22 | 2 | 0 | 2 | 20 | 0 |

### parseq

| action | profile | rows | delta errors | improved | worsened | equal | delta exact |
|---|---|---:|---:|---:|---:|---:|---:|
| drop_current | unicode_strict_v1 | 28 | 0 | 0 | 0 | 28 | 0 |
| drop_current | latin_alnum_ci_v1 | 28 | 0 | 0 | 0 | 28 | 0 |
| keep_base | unicode_strict_v1 | 100 | 0 | 0 | 0 | 100 | 0 |
| keep_base | latin_alnum_ci_v1 | 100 | 0 | 0 | 0 | 100 | 0 |
| replace_with_exception | unicode_strict_v1 | 22 | 0 | 0 | 0 | 22 | 0 |
| replace_with_exception | latin_alnum_ci_v1 | 22 | 0 | 0 | 0 | 22 | 0 |

## Conclusion

The executed exception-center policy is mainly a sparse/drop policy rather than a proven replacement policy. Any next selector must optimize against current-relative OCR labels, not only nearest-vs-exception proxy labels.

## Next Action

Use these labels as an audit target, then build explicit replacement/drop counterfactuals for current selected units before training another exception-center selector.
