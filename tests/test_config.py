from pathlib import Path

import pytest

from src.config import load_config, validate_config
from src.paths import ensure_output_dirs


CONFIG_PATH = Path("configs/experiment.yaml")
EXPECTED_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_default_config_loads_successfully() -> None:
    config = load_config(CONFIG_PATH)

    assert config["project"]["name"] == "uncertainty-aware-skin-lesion-classification"


def test_split_fractions_sum_to_one() -> None:
    config = load_config(CONFIG_PATH)
    data = config["data"]

    assert (
        data["train_fraction"] + data["val_fraction"] + data["test_fraction"]
    ) == pytest.approx(1.0)


def test_seven_fixed_class_labels_are_present() -> None:
    config = load_config(CONFIG_PATH)

    assert config["data"]["classes"] == EXPECTED_CLASSES
    assert config["data"]["num_classes"] == len(EXPECTED_CLASSES)


def test_ensemble_seeds_are_unique() -> None:
    config = load_config(CONFIG_PATH)
    seeds = config["training"]["ensemble_seeds"]

    assert len(seeds) == len(set(seeds))


def test_invalid_split_fractions_raise_clear_error() -> None:
    config = load_config(CONFIG_PATH)
    config["data"]["test_fraction"] = 0.20

    with pytest.raises(ValueError, match="split fractions must sum to 1.0"):
        validate_config(config)


def test_duplicate_ensemble_seeds_raise_clear_error() -> None:
    config = load_config(CONFIG_PATH)
    config["training"]["ensemble_seeds"] = [42, 42, 999]

    with pytest.raises(ValueError, match="ensemble_seeds must contain unique values"):
        validate_config(config)


def test_output_directory_creation_works_in_temporary_directory(
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)

    output_dirs = ensure_output_dirs(config, root=tmp_path)

    assert output_dirs
    for output_path in output_dirs.values():
        assert output_path.exists()
        assert output_path.is_dir()
