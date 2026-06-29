# Current-Best Changed-Pair Teacher Veto Diagnostic

This diagnostic tests fixed train-teacher veto rules only on rows where the current actual-bitstream policy changes Tesseract output.
It is not a promotion experiment, because the rule outcomes are interpreted against held-out Tesseract deltas.

| rule | kept | vetoed | kept Tesseract delta | vetoed improvements | vetoed worsens |
| --- | ---: | ---: | ---: | ---: | ---: |
| veto_if_mean_loss_increases | 2 | 3 | -3 | 2 | 1 |
| veto_if_crnn_abinet_loss_both_increase | 3 | 2 | -6 | 1 | 1 |
| veto_if_all_confidence_decrease | 5 | 0 | -8 | 0 | 0 |
| veto_if_any_teacher_prediction_changes | 5 | 0 | -8 | 0 | 0 |
| veto_if_no_teacher_loss_improves | 5 | 0 | -8 | 0 | 0 |

## Per-Pair Deltas

- `seed1` row `63` ref `heart` Tesseract delta `-3`
  - crnn: loss_delta `-0.000067`, confidence_delta `0.000451`, pred `heart` -> `heart`
  - abinet: loss_delta `0.000064`, confidence_delta `-0.000374`, pred `heart` -> `heart`
  - parseq: loss_delta `0.000003`, confidence_delta `-0.000018`, pred `heart` -> `heart`
- `seed2` row `6` ref `SLUSH` Tesseract delta `-2`
  - crnn: loss_delta `-0.003498`, confidence_delta `-0.000134`, pred `SLUSH` -> `SLUSH`
  - abinet: loss_delta `-0.001626`, confidence_delta `0.000003`, pred `SLUSH` -> `SLUSH`
  - parseq: loss_delta `0.000582`, confidence_delta `-0.000015`, pred `SLUSH` -> `SLUSH`
- `seed2` row `13` ref `Mauro's` Tesseract delta `-3`
  - crnn: loss_delta `0.000481`, confidence_delta `0.000660`, pred `nawo's` -> `nawo's`
  - abinet: loss_delta `0.020715`, confidence_delta `0.000128`, pred `Nano''s` -> `Nano''s`
  - parseq: loss_delta `-0.001277`, confidence_delta `0.001788`, pred `nawon's` -> `nawon's`
- `seed2` row `51` ref `from` Tesseract delta `1`
  - crnn: loss_delta `0.000099`, confidence_delta `-0.000456`, pred `from` -> `from`
  - abinet: loss_delta `0.000010`, confidence_delta `-0.000049`, pred `from` -> `from`
  - parseq: loss_delta `-0.000000`, confidence_delta `0.000001`, pred `from` -> `from`
- `seed2` row `64` ref `JOHN` Tesseract delta `-1`
  - crnn: loss_delta `-0.015424`, confidence_delta `0.004581`, pred `JOHN` -> `JOHN`
  - abinet: loss_delta `0.002049`, confidence_delta `0.002437`, pred `JOHN` -> `JOHN`
  - parseq: loss_delta `-0.000671`, confidence_delta `0.000044`, pred `JOHN` -> `JOHN`
