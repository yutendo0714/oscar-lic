# Exception-Center Action Readiness Table

This is a table-only precheck artifact for ADR-044. It does not claim OCR improvement.

## Summary

| partition | groups | current selected | exception eligible | both | current only | exception only | neither | exception proxy true | exception proxy false |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 225 | 225 | 45 | 45 | 180 | 0 | 0 | 44 | 1 |
| val | 75 | 75 | 22 | 22 | 53 | 0 | 0 | 9 | 13 |

## Interpretation

- `current_only` rows are current selections that a sparse exception-only policy would drop.
- `exception_only` rows are new exception additions with no current selected enhancement.
- `current_and_exception` rows are the only direct replacement candidates.
- `exception_proxy_false` is the no-headroom pressure seen by distance-only exception eligibility.

Before actual streams, a verifier should reduce `exception_proxy_false`, preserve useful `current_only` rows, and justify every replacement or drop relative to current.
