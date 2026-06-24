# Repository Completeness Checklist

This file distinguishes information completeness from external asset availability.

## Research definition

- [x] falsifiable primary hypothesis
- [x] secondary hypotheses
- [x] non-goals
- [x] contribution boundary
- [x] Go/No-Go gates
- [x] target metrics and domains

## Reproducibility

- [x] experiment manifest schema
- [x] configuration templates
- [x] rate accounting rule
- [x] OCR normalization profiles
- [x] environment separation policy
- [x] repository pins for priority assets where verified
- [ ] local checkpoint hashes — resolved only after download
- [ ] local dataset manifests — resolved only after approved acquisition
- [ ] reproduced baseline numbers — no compute run yet

## Operations

- [x] agent read order and hard rules
- [x] task queue
- [x] status, decision, blocker and negative-result logs
- [x] validation scripts
- [x] unit tests
- [x] license-risk registry
- [x] freshness policy

## Literature

- [x] systematic survey through snapshot date
- [x] paper registry with publication status
- [x] code availability status
- [x] caveats and comparison pitfalls
- [x] priority reading queue

## Meaning of “ready”

The repository is ready for an agent to **begin** research after G0. It does not claim that external licenses are approved, checkpoints are locally immutable, or baseline results have been reproduced. Those facts cannot be truthfully pre-filled without the target institution, hardware and downloaded bytes.
