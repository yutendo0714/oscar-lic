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
- [x] local checkpoint hashes for corrected MLIC++ lambda 0.025, PARSeq and TrOCR
- [ ] local dataset manifests — resolved only after approved acquisition
- [x] reproduced smoke/small baseline numbers for corrected MLIC++, CompressAI, OCR evaluators and actual `.oscr` accounting
- [ ] full paper RD-curve baseline numbers

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

The repository is ready for an agent to **continue** the corrected-MLIC++ frozen-base OSCAR-LIC assignment research after the 2026-06-25 G4 conditional-go pass. It does not claim that external licenses are approved, all checkpoints are locally immutable, full RD curves have been reproduced, or the promoted Eval300 pilot is publication-scale by itself. Those facts cannot be truthfully pre-filled without the target institution, hardware, approved datasets and downloaded bytes.
