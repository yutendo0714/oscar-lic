# Blockers

| ID | Blocker | Impact | Resolution condition | Status |
|---|---|---|---|---|
| B01 | Scientific dataset license/terms approval not recorded | Toy smoke tests can run, but benchmark training/final claims cannot | Add dated local approval and immutable local manifest; never bundle prohibited bytes | open |
| B02 | Checkpoint hashes unresolved | Reproduction is not immutable | Download from registry source, inspect format, record size/SHA256 | open |
| B03 | MLIC++ legacy dependency compatibility | Current CUDA/PyTorch may not reproduce official path | Build isolated environment/container and document minimal patch | open |
| B04 | Official code absent for several OCR-aware baselines | Comparison is implementation-sensitive | Contact authors or label transparent reimplementation | open |
| B05 | Utility oracle computational cost unknown | Could exceed project budget | Benchmark `U001` on 100 crops and extrapolate GPU-hours/storage | open |
| B06 | Multilingual OCR teacher/evaluator coverage unresolved | Cross-script claims would be invalid | Freeze script-specific model bundles and normalization/segmentation | open |
| B07 | Target GPU/hardware not supplied in artifact environment | Training schedule and memory limits are unknown | Run environment capture on target machine and select hardware profile | open |
| B08 | WebUI/SCI1K access and redistribution terms unresolved | Screen-domain G7 may be blocked | Verify source and terms or substitute an approved screen benchmark | open |
| B09 | MLIC++ corrected checkpoint only one prominently linked update point | Full six-point corrected curve may require old/new architecture care | Resolve every chosen checkpoint independently; never mix classes | open |
