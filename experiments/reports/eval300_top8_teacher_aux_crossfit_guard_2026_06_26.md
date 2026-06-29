# eval300_top8_teacher_aux_crossfit_guard_2026_06_26

Source-OOF no-op guard over the CRNN/ABINet teacher-auxiliary latent/code ranker.
Teacher losses shape the ranker during training only; acceptors see only ranker-derived/deployable group features.

## Validation Policies

| teacher weight | seed | acceptor | budget | changed | exact changed | false | wrong | missed | Tesseract | PARSeq | T worse |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0 | logistic | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | logistic | 1 | 2 | 1 | 1 | 0 | 16 | -1 | 0 | 0 |
| 0.000 | 0 | logistic | 2 | 2 | 1 | 1 | 0 | 16 | -1 | 0 | 0 |
| 0.000 | 0 | hist_gradient | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | hist_gradient | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | hist_gradient | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | random_forest | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | random_forest | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | random_forest | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | direct_top_score | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | direct_top_score | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.000 | 0 | direct_top_score | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | logistic | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | logistic | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | logistic | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | hist_gradient | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | hist_gradient | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | hist_gradient | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | random_forest | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | random_forest | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | random_forest | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | direct_top_score | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | direct_top_score | 1 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |
| 0.200 | 0 | direct_top_score | 2 | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 |

## Interpretation

This tests whether source-OOF ranker scores can train a separate no-op guard for the teacher-auxiliary candidate ranker. It is a pre-promotion diagnostic; counted OSCR export remains blocked unless the validation policy exceeds the current actual-bitstream -8 Tesseract floor with low false/wrong changes.

## Best Policy

```json
{
  "teacher_weight": 0.0,
  "model_seed": 0,
  "acceptor": "logistic",
  "max_false_wrong": 1,
  "threshold": 0.9349152445793152,
  "train_metrics": {
    "groups": 225,
    "oracle_change_groups": 63,
    "changed_groups": 3,
    "exact": 163,
    "exact_changed_groups": 2,
    "false_change": 1,
    "wrong_change": 0,
    "missed_oracle": 61,
    "tesseract_delta_vs_nearest": -5,
    "parseq_delta_vs_nearest": 0,
    "tesseract_worse_groups": 0,
    "parseq_worse_groups": 0
  },
  "val_metrics": {
    "groups": 75,
    "oracle_change_groups": 17,
    "changed_groups": 2,
    "exact": 58,
    "exact_changed_groups": 1,
    "false_change": 1,
    "wrong_change": 0,
    "missed_oracle": 16,
    "tesseract_delta_vs_nearest": -1,
    "parseq_delta_vs_nearest": 0,
    "tesseract_worse_groups": 0,
    "parseq_worse_groups": 0
  },
  "val_audits": [
    {
      "group_index": 75,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 6,
        "candidate_index": 1,
        "partition": "val"
      },
      "accept_score": 0.919885516166687,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 2,
      "top_code": 34,
      "nearest_code": 2,
      "oracle_code": 2,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 76,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 10,
        "candidate_index": 3,
        "partition": "val"
      },
      "accept_score": 0.00018571916734799743,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 32,
      "top_code": 51,
      "nearest_code": 32,
      "oracle_code": 32,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 77,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 11,
        "candidate_index": 0,
        "partition": "val"
      },
      "accept_score": 0.08443039655685425,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 34,
      "top_code": 2,
      "nearest_code": 34,
      "oracle_code": 34,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 78,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 17,
        "candidate_index": 1,
        "partition": "val"
      },
      "accept_score": 0.5179013013839722,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 55,
      "top_code": 51,
      "nearest_code": 55,
      "oracle_code": 55,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 79,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 20,
        "candidate_index": 0,
        "partition": "val"
      },
      "accept_score": 0.13420745730400085,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 34,
      "top_code": 1,
      "nearest_code": 34,
      "oracle_code": 34,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 80,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 22,
        "candidate_index": 0,
        "partition": "val"
      },
      "accept_score": 0.5245299339294434,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 34,
      "top_code": 48,
      "nearest_code": 34,
      "oracle_code": 34,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 81,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 29,
        "candidate_index": 0,
        "partition": "val"
      },
      "accept_score": 0.6148162484169006,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 34,
      "top_code": 30,
      "nearest_code": 34,
      "oracle_code": 34,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 82,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 32,
        "candidate_index": 3,
        "partition": "val"
      },
      "accept_score": 0.028958283364772797,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 32,
      "top_code": 55,
      "nearest_code": 32,
      "oracle_code": 32,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 83,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 34,
        "candidate_index": 2,
        "partition": "val"
      },
      "accept_score": 0.00034132396103814244,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 55,
      "top_code": 57,
      "nearest_code": 55,
      "oracle_code": 55,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 84,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 35,
        "candidate_index": 2,
        "partition": "val"
      },
      "accept_score": 0.03934113308787346,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 22,
      "top_code": 36,
      "nearest_code": 22,
      "oracle_code": 22,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 85,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 39,
        "candidate_index": 1,
        "partition": "val"
      },
      "accept_score": 0.003192505333572626,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 34,
      "top_code": 30,
      "nearest_code": 34,
      "oracle_code": 34,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 86,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 41,
        "candidate_index": 1,
        "partition": "val"
      },
      "accept_score": 0.2363332211971283,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 55,
      "top_code": 59,
      "nearest_code": 55,
      "oracle_code": 55,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 87,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 48,
        "candidate_index": 0,
        "partition": "val"
      },
      "accept_score": 0.8629383444786072,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 34,
      "top_code": 44,
      "nearest_code": 34,
      "oracle_code": 34,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 88,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 49,
        "candidate_index": 1,
        "partition": "val"
      },
      "accept_score": 0.9304944276809692,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 55,
      "top_code": 59,
      "nearest_code": 55,
      "oracle_code": 55,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 89,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 53,
        "candidate_index": 1,
        "partition": "val"
      },
      "accept_score": 0.9588080048561096,
      "threshold": 0.9349152445793152,
      "accepted": true,
      "selected_code": 60,
      "top_code": 60,
      "nearest_code": 42,
      "oracle_code": 42,
      "status": "false_change",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 90,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 55,
        "candidate_index": 0,
        "partition": "val"
      },
      "accept_score": 0.2559788227081299,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 34,
      "top_code": 44,
      "nearest_code": 34,
      "oracle_code": 34,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 91,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 57,
        "candidate_index": 2,
        "partition": "val"
      },
      "accept_score": 0.21157318353652954,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 55,
      "top_code": 20,
      "nearest_code": 55,
      "oracle_code": 55,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 92,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 58,
        "candidate_index": 2,
        "partition": "val"
      },
      "accept_score": 0.00039168106741271913,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 55,
      "top_code": 57,
      "nearest_code": 55,
      "oracle_code": 55,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 93,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 61,
        "candidate_index": 0,
        "partition": "val"
      },
      "accept_score": 0.21059763431549072,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 34,
      "top_code": 30,
      "nearest_code": 34,
      "oracle_code": 34,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    },
    {
      "group_index": 94,
      "key": {
        "real_seed": 0,
        "seed": 1,
        "source_index": 64,
        "candidate_index": 2,
        "partition": "val"
      },
      "accept_score": 0.1044931709766388,
      "threshold": 0.9349152445793152,
      "accepted": false,
      "selected_code": 55,
      "top_code": 6,
      "nearest_code": 55,
      "oracle_code": 55,
      "status": "exact",
      "selected_tesseract_delta_vs_nearest": 0,
      "selected_parseq_delta_vs_nearest": 0
    }
  ]
}
```
