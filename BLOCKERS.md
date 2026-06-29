# Blockers

| ID | Blocker | Impact | Resolution condition | Status |
|---|---|---|---|---|
| B01 | Scientific dataset license/terms approval not recorded | Toy smoke tests can run, but benchmark training/final claims cannot | Add dated local approval and immutable local manifest; never bundle prohibited bytes | open |
| B02 | Some checkpoint hashes unresolved | Future full-curve baselines are not immutable until each chosen checkpoint is pinned | Download each chosen checkpoint from registry source, inspect format, record size/SHA256 | partially resolved: corrected MLIC++ lambda 0.025, DCAE lambda 0.013, PARSeq, CRNN, ABINet, TrOCR and PaddleOCR `PP-OCRv6_medium_rec` are hashed; any additional operating points remain unresolved until individually pinned |
| B03 | MLIC++ legacy dependency compatibility | Current CUDA/PyTorch may not reproduce official path | Build isolated environment/container and document minimal patch | resolved for corrected MLIC++ lambda 0.025 smoke in `.venv-mlic`; reopen for other operating points |
| B04 | Official code absent for several OCR-aware baselines | Comparison is implementation-sensitive | Contact authors or label transparent reimplementation | open |
| B05 | Utility oracle computational cost can grow quickly | Could exceed project budget for larger OCR counterfactual tables | Maintain pilot cost/storage snapshots before scaling beyond Eval300 | partially resolved for Eval100/Eval300 pilots |
| B06 | Multilingual OCR teacher/evaluator coverage unresolved | Cross-script claims would be invalid | Freeze script-specific model bundles and normalization/segmentation | open |
| B07 | Target GPU/hardware can intermittently disappear in container | Training schedule and memory limits depend on stable GPU visibility | Check `nvidia-smi` before GPU work; stop if GPU disappears until container restart | partially resolved: RTX 4070 Ti SUPER visible in final validation |
| B08 | WebUI/SCI1K access and redistribution terms unresolved | Screen-domain G7 may be blocked | Verify source and terms or substitute an approved screen benchmark | open |
| B09 | MLIC++ corrected checkpoint only one prominently linked update point | Full six-point corrected curve may require old/new architecture care | Resolve every chosen checkpoint independently; never mix classes | open; N061 confirmed official `origin/main` still lists only corrected lambda 0.0250, while multi-rate MSE weights are old-class-only |
| B10 | Full paper RD curve not reproduced | International submission claims cannot compare full codec frontier yet | Reproduce selected operating-point grid with actual bitstream bpp and frozen splits | open |
| B11 | Current best result is an Eval300 text-crop assignment pilot | Claim scope is promising but narrow | Scale to approved larger scene/screen/document domains with held-out OCR families | open |
