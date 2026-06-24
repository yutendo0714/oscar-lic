"""Validation helpers for YAML/CSV research registries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml


class RegistryError(ValueError):
    pass


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_unique_ids(items: list[dict], key: str, context: str) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RegistryError(f"{context}[{index}] is not a mapping")
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise RegistryError(f"{context}[{index}] has invalid {key!r}")
        if value in seen:
            raise RegistryError(f"duplicate {key}={value!r} in {context}")
        seen.add(value)


def validate_data_registry(path: str | Path) -> list[str]:
    data = load_yaml(path)
    errors: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("datasets"), list):
        return ["data registry must contain a datasets list"]
    try:
        require_unique_ids(data["datasets"], "id", "datasets")
    except RegistryError as exc:
        errors.append(str(exc))
    allowed_status = {"verified", "manual_review_required", "blocked_until_verified"}
    for item in data["datasets"]:
        if item.get("license_status") not in allowed_status:
            errors.append(f"dataset {item.get('id')}: invalid license_status")
        if not item.get("expected_path"):
            errors.append(f"dataset {item.get('id')}: missing expected_path")
        if item.get("evaluation_only") is True and item.get("train_allowed") not in {False, "false"}:
            errors.append(f"dataset {item.get('id')}: evaluation_only but train_allowed is not false")
    return errors


def validate_baseline_registry(path: str | Path) -> list[str]:
    data = load_yaml(path)
    errors: list[str] = []
    if not isinstance(data, dict) or not isinstance(data.get("baselines"), list):
        return ["baseline registry must contain a baselines list"]
    try:
        require_unique_ids(data["baselines"], "id", "baselines")
    except RegistryError as exc:
        errors.append(str(exc))
    for item in data["baselines"]:
        if item.get("priority") not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"baseline {item.get('id')}: invalid priority")
        if not item.get("implementation_status"):
            errors.append(f"baseline {item.get('id')}: missing implementation_status")
    return errors


def validate_experiment_matrix(path: str | Path) -> list[str]:
    errors: list[str] = []
    required = {
        "experiment_id", "hypothesis_id", "stage", "priority", "track",
        "description", "status", "depends_on", "gate", "datasets",
        "seeds", "primary_metrics", "notes",
    }
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            return [f"experiment matrix missing columns: {sorted(missing)}"]
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            eid = row["experiment_id"]
            if not eid:
                errors.append(f"line {line}: empty experiment_id")
            elif eid in seen:
                errors.append(f"line {line}: duplicate experiment_id {eid}")
            seen.add(eid)
            if row["track"] not in {"A", "B", "C"}:
                errors.append(f"line {line}: invalid track {row['track']!r}")
            if row["priority"] not in {"P0", "P1", "P2", "P3"}:
                errors.append(f"line {line}: invalid priority {row['priority']!r}")
    return errors
