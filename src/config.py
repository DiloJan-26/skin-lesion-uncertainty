"""Configuration loading and validation for the Phase 0 research scaffold."""

from pathlib import Path
from typing import Any

import yaml


EXPECTED_NUM_CLASSES = 7


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment configuration."""
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required Phase 0 experiment configuration invariants."""
    data = _require_mapping(config, "data")
    training = _require_mapping(config, "training")

    train_fraction = _require_number(data, "train_fraction")
    val_fraction = _require_number(data, "val_fraction")
    test_fraction = _require_number(data, "test_fraction")
    split_total = train_fraction + val_fraction + test_fraction
    if abs(split_total - 1.0) > 1e-8:
        raise ValueError(
            "Data split fractions must sum to 1.0 "
            f"(got {split_total:.6f})."
        )

    classes = data.get("classes")
    if not isinstance(classes, list) or not all(
        isinstance(label, str) for label in classes
    ):
        raise ValueError("data.classes must be a list of class-label strings.")
    if len(classes) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Exactly {EXPECTED_NUM_CLASSES} class labels are required "
            f"(got {len(classes)})."
        )

    num_classes = data.get("num_classes")
    if not isinstance(num_classes, int):
        raise ValueError("data.num_classes must be an integer.")
    if num_classes != len(classes):
        raise ValueError(
            "data.num_classes must match the number of labels in data.classes "
            f"(got {num_classes} and {len(classes)})."
        )

    seeds = training.get("ensemble_seeds")
    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("training.ensemble_seeds must be a list of integers.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("training.ensemble_seeds must contain unique values.")


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _require_number(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"data.{key} must be numeric.")
    return float(value)
