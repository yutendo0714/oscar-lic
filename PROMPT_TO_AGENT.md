# Minimal instruction to a research agent

Use the following instruction after giving the agent access to this directory:

> Read `AGENTS.md` and follow its authoritative read order. Treat this directory as the source of truth. Begin with G0 only; do not download license-restricted data or start GPU training until the corresponding gates and registries permit it. Execute the highest-priority unblocked task in `experiments/EXPERIMENT_MATRIX.csv`, record all provenance, update `STATUS.md`, `TASKS.md`, `BLOCKERS.md`, `DECISIONS.md` and `NEGATIVE_RESULTS.md` as applicable, and never report estimated rate as actual bitstream rate.

A more open-ended command such as “研究を進めて” is insufficient because it does not enforce baseline reproduction, data governance or stage gates.
