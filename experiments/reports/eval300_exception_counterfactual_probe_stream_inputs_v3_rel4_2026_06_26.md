# Exception Counterfactual Probe Stream Inputs

| seed | split rows | current selections | exception selections | tags |
|---|---:|---:|---:|---|
| 0 | 21 | 21 | 8 | `{"train_close_current_only": 13, "train_iam_current_only_near": 4, "train_iam_proxy_true_control": 8}` |
| 1 | 21 | 21 | 6 | `{"train_close_current_only": 15, "train_iam_current_only_near": 5, "train_iam_proxy_true_control": 6}` |
| 2 | 21 | 21 | 9 | `{"train_close_current_only": 12, "train_iam_current_only_near": 3, "train_iam_proxy_true_control": 8, "train_replacement_proxy_false": 1}` |

## Next Action

Run current K64 nearest streams and K80 explicit exception/drop streams per seed, then OCR both recon sets.
