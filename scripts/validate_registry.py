#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscarlic.registry import (  # noqa: E402
    validate_baseline_registry,
    validate_data_registry,
    validate_experiment_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OSCAR-LIC registries")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    checks = {
        "data": validate_data_registry(args.root / "data/registry.yaml"),
        "baselines": validate_baseline_registry(args.root / "baselines/registry.yaml"),
        "experiments": validate_experiment_matrix(args.root / "experiments/EXPERIMENT_MATRIX.csv"),
    }
    errors = [(name, err) for name, values in checks.items() for err in values]
    for name, values in checks.items():
        print(f"[{name}] {'OK' if not values else 'FAILED'}")
        for value in values:
            print(f"  - {value}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
