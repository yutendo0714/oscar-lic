# Exception Counterfactual Probe Stream Inputs

| seed | split rows | current selections | exception selections | tags |
|---|---:|---:|---:|---|
| 0 | 17 | 17 | 8 | `{"train_close_current_only": 9, "train_iam_current_only_near": 2, "train_iam_proxy_true_control": 8}` |
| 1 | 17 | 17 | 6 | `{"train_close_current_only": 11, "train_iam_current_only_near": 3, "train_iam_proxy_true_control": 6}` |
| 2 | 19 | 19 | 9 | `{"train_close_current_only": 10, "train_iam_current_only_near": 3, "train_iam_proxy_true_control": 8, "train_replacement_proxy_false": 1}` |

## Next Action

Run current K64 nearest streams and K80 explicit exception/drop streams per seed, then OCR both recon sets.
