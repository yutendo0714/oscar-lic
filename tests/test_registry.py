from pathlib import Path

from oscarlic.registry import (
    validate_baseline_registry,
    validate_data_registry,
    validate_experiment_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def test_data_registry():
    assert validate_data_registry(ROOT / "data/registry.yaml") == []


def test_baseline_registry():
    assert validate_baseline_registry(ROOT / "baselines/registry.yaml") == []


def test_experiment_matrix():
    assert validate_experiment_matrix(ROOT / "experiments/EXPERIMENT_MATRIX.csv") == []
