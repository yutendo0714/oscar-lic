# eval300_top8_strhub_logit_feature_selector_2026_06_26

CRNN/ABINet logit-summary feature selector over Eval300 top-8 assignment candidates.
This is an OCR-aware encoder-side diagnostic and does not export counted `.oscr` streams.

## Validation Policies

| target | model | budget | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | rank1 | rank<=4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact_oracle | logistic | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 5 | 11 |
| exact_oracle | logistic | 1 | 7 | 0 | 5 | 2 | 15 | 6 | 0 | 5 | 11 |
| exact_oracle | logistic | 2 | 7 | 0 | 5 | 2 | 15 | 6 | 0 | 5 | 11 |
| exact_oracle | hist_gradient | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 3 | 8 |
| exact_oracle | hist_gradient | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 3 | 8 |
| exact_oracle | hist_gradient | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 3 | 8 |
| exact_oracle | random_forest | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 3 | 10 |
| exact_oracle | random_forest | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 3 | 10 |
| exact_oracle | random_forest | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 3 | 10 |
| safe_improve | logistic | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 1 | 9 |
| safe_improve | logistic | 1 | 1 | 0 | 1 | 0 | 17 | 0 | 0 | 1 | 9 |
| safe_improve | logistic | 2 | 1 | 0 | 1 | 0 | 17 | 0 | 0 | 1 | 9 |
| safe_improve | hist_gradient | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 6 |
| safe_improve | hist_gradient | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 6 |
| safe_improve | hist_gradient | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 6 |
| safe_improve | random_forest | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 2 | 10 |
| safe_improve | random_forest | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 2 | 10 |
| safe_improve | random_forest | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 2 | 10 |

## Best Policy

```json
{
  "target": "exact_oracle",
  "model_name": "logistic",
  "max_false_wrong": 0,
  "threshold": 1.9994474053382874,
  "train_metrics": {
    "groups": 225,
    "oracle_change_groups": 63,
    "changed_groups": 0,
    "exact": 162,
    "exact_changed_groups": 0,
    "false_change": 0,
    "wrong_change": 0,
    "missed_oracle": 63,
    "tesseract_delta_vs_nearest": 0,
    "parseq_delta_vs_nearest": 0,
    "tesseract_worse_groups": 0,
    "oracle_rank_le1": 43,
    "oracle_rank_le2": 55,
    "oracle_rank_le4": 62
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
    "oracle_rank_le1": 5,
    "oracle_rank_le2": 8,
    "oracle_rank_le4": 11
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
      "top_code": 27,
      "top_score": 1.5320839565902133e-06,
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
      "top_code": 55,
      "top_score": 0.1621844470500946,
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
      "top_code": 3,
      "top_score": 0.12173932790756226,
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
      "top_code": 13,
      "top_score": 0.14698834717273712,
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
      "top_score": 0.3134341835975647,
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
      "top_code": 46,
      "top_score": 0.056986790150403976,
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
      "top_score": 0.0021493476815521717,
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
      "top_score": 0.8623464703559875,
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
      "top_code": 32,
      "top_score": 0.657401978969574,
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
      "top_score": 0.7877841591835022,
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
      "top_code": 39,
      "top_score": 0.044181402772665024,
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
      "top_code": 32,
      "top_score": 0.9998378753662109,
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
      "top_score": 0.9558818340301514,
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
      "top_code": 59,
      "top_score": 0.057624612003564835,
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
      "top_score": 0.725989580154419,
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
      "top_code": 44,
      "top_score": 0.6079666018486023,
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
      "top_score": 0.4606724679470062,
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
      "top_code": 18,
      "top_score": 0.052129536867141724,
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
      "top_score": 0.334467351436615,
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
      "top_code": 6,
      "top_score": 0.9096325635910034,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    }
  ]
}
```
