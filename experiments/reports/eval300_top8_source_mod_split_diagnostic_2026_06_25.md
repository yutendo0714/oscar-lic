# Eval300 Top8 Source-Disjoint Split Diagnostic

Date: 2026-06-25

## Purpose

The clean Eval300 seed split has source-index overlap between train and validation. In the current top8 artifact:

- train seed `0`: 159 unique source indices;
- val seed `1`: 49 unique source indices;
- overlap: 31 source indices.

That means the standard clean val precheck is not a hard source-disjoint generalization test. Since the top8 models still fail on that easier split, this diagnostic asks how badly the same group/no-op model behaves when source indices are held out by construction.

## Configuration

- Feature artifact: `experiments/features/eval300_assignment_top8_codeeffect_latentctx_windows_parseqteacher_seed0_1_2_2026_06_25.npz`
- Split mode: `source_modulo`
- Validation rule: `source_index % 5 == 4`
- Val groups: `68`
- Val oracle-change groups: `17`
- Model seeds: `0,1,2`
- Epochs: `400`
- Architecture/loss: default top8 group/no-op window head

This is a table-level diagnostic only. It is not an actual `.oscr` promotion candidate.

## Results

| Policy | Tesseract edit delta | PARSeq delta | False change | Wrong change | Missed oracle | Exact oracle | Changed groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| row argmax | `+3.0` | `+1.0` | `7.33` | `6.0` | `10.33` | `0.67` | `14.0` |
| group threshold 0.5 | `+2.33` | `+1.0` | `6.67` | `5.67` | `10.67` | `0.67` | `13.0` |
| train-tuned group threshold | `+3.0` | `+1.0` | `8.67` | `6.33` | `10.0` | `0.67` | `15.67` |
| oracle | `-19.0` | `0.0` | `0.0` | `0.0` | `0.0` | `17.0` | `17.0` |

Row validation AUC/AP remains apparently respectable (`0.8676 / 0.6233`), and group AUC is `0.7243`, but policy-level OCR is harmful.

## Interpretation

This reinforces two points:

1. Row AUC/AP and group AUC are not sufficient model-selection criteria for assignment.
2. The current Eval300 training pool is too small or too correlated for robust source-level generalization of the code-effect assignment policy.

The standard clean split is still the current promoted-baseline comparison, because all existing actual-bitstream claims use it. However, future stronger claims should use source-disjoint or dataset-disjoint validation, and likely need more labeled groups before adding more model capacity.

## Decision

Do not promote or run counted `.oscr` streams for the source-modulo policy.

The trainer now supports `--split-mode source_modulo`, `--val-source-modulo` and `--val-source-remainder`. Keep this as a diagnostic split tool, not as a replacement for the frozen clean split until a new protocol decision is recorded.
