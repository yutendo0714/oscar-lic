# eval300_top8_trainval_teacher_consensus_2026_06_26

OCR/text teacher-consensus diagnostic for Eval300 top-8 assignment candidates.

This is not a promotion result: policies use train-teacher OCR/text signals at selection time and no counted `.oscr` stream is exported.

## Validation Policies

| policy | changed | exact changed | false | wrong | missed | Tesseract delta | PARSeq delta | Tesseract worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nearest` | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| `heldout_oracle` | 17 | 17 | 0 | 0 | 0 | -24 | 0 | 0 |
| `any_string_improve_all_string_no_worse` | 1 | 0 | 1 | 0 | 17 | 0 | 0 | 0 |
| `two_string_improve_all_string_no_worse` | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| `parseq_improve_crnn_abinet_no_worse` | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| `string_sum_improve_all_string_no_worse` | 1 | 0 | 1 | 0 | 17 | 0 | 0 | 0 |
| `string_sum_improve_loss_mean_no_worse` | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| `any_string_improve_all_string_and_loss_no_worse` | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| `teacher_string_sum_argmax_if_improves` | 1 | 0 | 1 | 0 | 17 | 0 | 0 | 0 |

## Oracle Filter Coverage

- `any_string_improve_all_string_no_worse`: oracle row passes `0/17`, no-headroom groups with any passing candidate `1/58`
- `two_string_improve_all_string_no_worse`: oracle row passes `0/17`, no-headroom groups with any passing candidate `0/58`
- `parseq_improve_crnn_abinet_no_worse`: oracle row passes `0/17`, no-headroom groups with any passing candidate `0/58`
- `string_sum_improve_all_string_no_worse`: oracle row passes `0/17`, no-headroom groups with any passing candidate `1/58`
- `string_sum_improve_loss_mean_no_worse`: oracle row passes `0/17`, no-headroom groups with any passing candidate `0/58`
- `any_string_improve_all_string_and_loss_no_worse`: oracle row passes `0/17`, no-headroom groups with any passing candidate `0/58`
- `teacher_string_sum_argmax_if_improves`: oracle row passes `0/17`, no-headroom groups with any passing candidate `1/58`

## Interpretation

Best fixed teacher-consensus validation policy is `two_string_improve_all_string_no_worse` with Tesseract delta 0 and 0 false/wrong changes. It does not beat the current actual-bitstream floor of -8 edits, and it is non-promotable without distillation or explicit encoder OCR cost accounting.
