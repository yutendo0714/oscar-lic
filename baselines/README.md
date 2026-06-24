# Baseline Policy

`registry.yaml` defines the role and evidential status of each baseline. A result cannot be called an official reproduction unless code, commit, checkpoint, preprocessing and evaluation protocol match sufficiently and differences are documented.

## Required comparison buckets

1. conventional/framework LIC: factorized, hyperprior, joint AR
2. strong modern LIC: MLIC++; DCAE/FTIC/LALIC as resources permit
3. text-aware loss: Lai-style single teacher
4. allocation controls: uniform mask, detector confidence, random ranking
5. proposed: multi-teacher + utility + scalable stream
6. semantic-assisted/generative: separate Track B
7. standards: JPEG AI VM, plus VTM/SCC where available

## Reimplementation naming

Use names such as `Lai-loss (our reimplementation on MLIC++)`, never simply `Lai et al.` when author code is unavailable. This avoids attributing implementation choices to the original authors.
