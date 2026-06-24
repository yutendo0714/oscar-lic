# Experiments

`EXPERIMENT_MATRIX.csv` is the authoritative execution queue. Each actual run gets a JSON or YAML manifest under `experiments/manifests/`, logs under `logs/`, metrics under `results/`, and a concise interpretation under `reports/`.

Do not change a completed row's meaning. Add a new experiment ID or record an ADR.

Status vocabulary: `ready`, `planned`, `running`, `completed`, `failed`, `invalid`, `blocked_data`, `blocked_models`, `blocked_checkpoint`, `deferred`.
