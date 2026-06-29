# eval300_top8_strhub_sequence_feature_selector_2026_06_26

CRNN/ABINet binned sequence-profile feature selector over Eval300 top-8 assignment candidates.
This is an OCR-aware diagnostic and does not export counted `.oscr` streams.

## Validation Policies

| target | model | budget | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | rank1 | rank<=4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact_oracle | logistic | 0 | 3 | 0 | 3 | 0 | 17 | 1 | 0 | 4 | 11 |
| exact_oracle | logistic | 1 | 3 | 0 | 3 | 0 | 17 | 1 | 0 | 4 | 11 |
| exact_oracle | logistic | 2 | 4 | 0 | 3 | 1 | 16 | 2 | 0 | 4 | 11 |
| exact_oracle | hist_gradient | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 4 | 12 |
| exact_oracle | hist_gradient | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 4 | 12 |
| exact_oracle | hist_gradient | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 4 | 12 |
| exact_oracle | random_forest | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 2 | 10 |
| exact_oracle | random_forest | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 2 | 10 |
| exact_oracle | random_forest | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 2 | 10 |
| safe_improve | logistic | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 4 | 8 |
| safe_improve | logistic | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 4 | 8 |
| safe_improve | logistic | 2 | 2 | 0 | 1 | 1 | 16 | 0 | 0 | 4 | 8 |
| safe_improve | hist_gradient | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 1 | 9 |
| safe_improve | hist_gradient | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 1 | 9 |
| safe_improve | hist_gradient | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 1 | 9 |
| safe_improve | random_forest | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 4 | 7 |
| safe_improve | random_forest | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 4 | 7 |
| safe_improve | random_forest | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 4 | 7 |

## Best Policy

```json
{
  "target": "exact_oracle",
  "model_name": "hist_gradient",
  "max_false_wrong": 0,
  "threshold": 0.9479566812515259,
  "train_metrics": {
    "groups": 225,
    "oracle_change_groups": 63,
    "changed_groups": 63,
    "exact": 225,
    "exact_changed_groups": 63,
    "false_change": 0,
    "wrong_change": 0,
    "missed_oracle": 0,
    "tesseract_delta_vs_nearest": -87,
    "parseq_delta_vs_nearest": -1,
    "tesseract_worse_groups": 0,
    "oracle_rank_le1": 63,
    "oracle_rank_le2": 63,
    "oracle_rank_le4": 63
  },
  "val_metrics": {
    "groups": 75,
    "oracle_change_groups": 17,
    "changed_groups": 0,
    "exact": 58,
    "exact_changed_groups": 0,
    "false_change": 0,
    "wrong_change": 0,
    "missed_oracle": 17,
    "tesseract_delta_vs_nearest": 0,
    "parseq_delta_vs_nearest": 0,
    "tesseract_worse_groups": 0,
    "oracle_rank_le1": 4,
    "oracle_rank_le2": 9,
    "oracle_rank_le4": 12
  },
  "val_audits": [
    {
      "group_key": [
        0,
        1,
        6,
        1
      ],
      "selected_code": 2,
      "nearest_code": 2,
      "oracle_code": 2,
      "top_code": 34,
      "top_score": 0.005157427862286568,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        10,
        3
      ],
      "selected_code": 32,
      "nearest_code": 32,
      "oracle_code": 32,
      "top_code": 20,
      "top_score": 0.0005250428221188486,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        11,
        0
      ],
      "selected_code": 34,
      "nearest_code": 34,
      "oracle_code": 34,
      "top_code": 39,
      "top_score": 0.001029545092023909,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        17,
        1
      ],
      "selected_code": 55,
      "nearest_code": 55,
      "oracle_code": 55,
      "top_code": 57,
      "top_score": 0.018935883417725563,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        20,
        0
      ],
      "selected_code": 34,
      "nearest_code": 34,
      "oracle_code": 34,
      "top_code": 1,
      "top_score": 0.01348321232944727,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        22,
        0
      ],
      "selected_code": 34,
      "nearest_code": 34,
      "oracle_code": 34,
      "top_code": 48,
      "top_score": 0.015881886705756187,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        29,
        0
      ],
      "selected_code": 34,
      "nearest_code": 34,
      "oracle_code": 34,
      "top_code": 30,
      "top_score": 0.08692151308059692,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        32,
        3
      ],
      "selected_code": 32,
      "nearest_code": 32,
      "oracle_code": 32,
      "top_code": 55,
      "top_score": 0.005748989526182413,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        34,
        2
      ],
      "selected_code": 55,
      "nearest_code": 55,
      "oracle_code": 55,
      "top_code": 57,
      "top_score": 0.06995182484388351,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        35,
        2
      ],
      "selected_code": 22,
      "nearest_code": 22,
      "oracle_code": 22,
      "top_code": 26,
      "top_score": 0.011401101015508175,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        39,
        1
      ],
      "selected_code": 34,
      "nearest_code": 34,
      "oracle_code": 34,
      "top_code": 2,
      "top_score": 0.03897703066468239,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        41,
        1
      ],
      "selected_code": 55,
      "nearest_code": 55,
      "oracle_code": 55,
      "top_code": 20,
      "top_score": 0.047955043613910675,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        48,
        0
      ],
      "selected_code": 34,
      "nearest_code": 34,
      "oracle_code": 34,
      "top_code": 44,
      "top_score": 0.04389842599630356,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        49,
        1
      ],
      "selected_code": 55,
      "nearest_code": 55,
      "oracle_code": 55,
      "top_code": 57,
      "top_score": 0.0018134708516299725,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        53,
        1
      ],
      "selected_code": 42,
      "nearest_code": 42,
      "oracle_code": 42,
      "top_code": 60,
      "top_score": 0.023890791460871696,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        55,
        0
      ],
      "selected_code": 34,
      "nearest_code": 34,
      "oracle_code": 34,
      "top_code": 48,
      "top_score": 0.005919613875448704,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        57,
        2
      ],
      "selected_code": 55,
      "nearest_code": 55,
      "oracle_code": 55,
      "top_code": 6,
      "top_score": 0.0006177305476740003,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        58,
        2
      ],
      "selected_code": 55,
      "nearest_code": 55,
      "oracle_code": 55,
      "top_code": 32,
      "top_score": 0.006325986701995134,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        61,
        0
      ],
      "selected_code": 34,
      "nearest_code": 34,
      "oracle_code": 34,
      "top_code": 46,
      "top_score": 0.005112046375870705,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_key": [
        0,
        1,
        64,
        2
      ],
      "selected_code": 55,
      "nearest_code": 55,
      "oracle_code": 55,
      "top_code": 18,
      "top_score": 0.0008306715753860772,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    }
  ]
}
```
