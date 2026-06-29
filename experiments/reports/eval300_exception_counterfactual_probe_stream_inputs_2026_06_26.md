# Exception Counterfactual Probe Stream Inputs

| seed | split rows | current selections | exception selections | tags |
|---|---:|---:|---:|---|
| 0 | 11 | 11 | 5 | `{"train_close_current_only": 5, "train_iam_current_only_near": 2, "train_iam_proxy_true_control": 5}` |
| 1 | 9 | 9 | 1 | `{"train_close_current_only": 5, "train_iam_current_only_near": 3, "train_iam_proxy_true_control": 1}` |
| 2 | 12 | 12 | 5 | `{"train_close_current_only": 4, "train_iam_current_only_near": 3, "train_iam_proxy_true_control": 4, "train_replacement_proxy_false": 1}` |

## Next Action

Run current K64 nearest streams and K80 explicit exception/drop streams per seed, then OCR both recon sets.
