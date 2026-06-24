# Agent Workflow State Machine

```text
BOOTSTRAP
  → VALIDATE_REPO
  → RESOLVE_EXTERNALS
  → APPROVE_DATA
  → SMOKE_CODEC
  → SMOKE_OCR
  → REPRODUCE_BASELINE
  → IMPLEMENT_SINGLE_TEACHER
  → BUILD_ORACLE
  → G4_DECISION
       ├─ NO-GO → ANALYZE/REDEFINE
       └─ GO → TRAIN_UTILITY
  → BUILD_SCALABLE_STREAM
  → GENERALIZATION
  → OPTIONAL_EXTENSIONS
  → SUBMISSION_AUDIT
```

At every transition, update `STATUS.md`, `TASKS.md`, experiment matrix and decision log. The agent may not jump from bootstrap directly to full proposed training.

## Failure escalation

1. capture command/log/environment
2. classify dependency, data, numerical, design or resource failure
3. attempt the smallest diagnostic
4. record workaround as a patch/ADR
5. do not retry unchanged command more than twice
