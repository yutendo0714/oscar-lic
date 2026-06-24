# Dependency Pinning Policy

The bootstrap utilities use `requirements-bootstrap.txt`. Research models use separate Conda environments because MLIC++ and modern OCR stacks have incompatible historical pins.

A lock is considered valid only after installation and smoke tests on the target platform. Generated lock files must include platform, Python, CUDA, driver and package hashes where the package manager supports them. Never upgrade a dependency inside a completed experiment environment; create a new environment ID.

The exact versions in `requirements-bootstrap.txt` reflect the environment used to validate this repository snapshot, not a claim that they are optimal for model training.
