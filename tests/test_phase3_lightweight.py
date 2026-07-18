from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    add_label_indices,
    compute_class_weights,
    get_label_mapping,
    load_split_csvs,
)


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_get_label_mapping_returns_correct_indices() -> None:
    label_mapping = get_label_mapping(CLASSES)

    assert label_mapping == {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6,
    }


def test_get_label_mapping_raises_for_duplicate_classes() -> None:
    with pytest.raises(ValueError, match="Class names must be unique"):
        get_label_mapping(["nv", "mel", "nv"])


def test_add_label_indices_adds_correct_label_index() -> None:
    df = pd.DataFrame({"dx": ["akiec", "mel", "vasc"]})

    indexed_df = add_label_indices(df, CLASSES)

    assert indexed_df["label_index"].tolist() == [0, 4, 6]
    assert "label_index" not in df.columns


def test_add_label_indices_raises_for_unknown_class() -> None:
    df = pd.DataFrame({"dx": ["nv", "unknown"]})

    with pytest.raises(ValueError, match="Unknown class label"):
        add_label_indices(df, CLASSES)


def test_compute_class_weights_returns_expected_classes_and_positive_weights() -> None:
    df = pd.DataFrame(
        {
            "dx": [
                "akiec",
                "bcc",
                "bkl",
                "df",
                "mel",
                "nv",
                "nv",
                "vasc",
            ]
        }
    )

    weights_df = compute_class_weights(df, CLASSES)

    assert weights_df["class"].tolist() == CLASSES
    assert (weights_df["weight"] > 0).all()
    assert weights_df.loc[weights_df["class"] == "nv", "count"].item() == 2


def test_load_split_csvs_loads_train_val_test_from_temporary_csvs(
    tmp_path: Path,
) -> None:
    for split_name in ("train", "val", "test"):
        pd.DataFrame({"image_id": [f"{split_name}_image"], "dx": ["nv"]}).to_csv(
            tmp_path / f"{split_name}_split.csv",
            index=False,
        )

    split_dfs = load_split_csvs(tmp_path)

    assert set(split_dfs) == {"train", "val", "test"}
    assert split_dfs["train"].loc[0, "image_id"] == "train_image"
