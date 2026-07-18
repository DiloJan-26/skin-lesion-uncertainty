from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    attach_image_paths,
    create_grouped_stratified_split,
    summarize_split,
    verify_no_group_leakage,
)


def test_attach_image_paths_correctly_attaches_paths_when_ids_match(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "ISIC_0000001.jpg"
    image_path.touch()
    metadata = pd.DataFrame(
        {"image_id": ["ISIC_0000001"], "lesion_id": ["lesion_1"], "dx": ["nv"]}
    )

    attached = attach_image_paths(metadata, {"ISIC_0000001": image_path})

    assert attached.loc[0, "image_path"] == image_path
    assert "image_path" not in metadata.columns


def test_attach_image_paths_raises_value_error_when_image_is_missing() -> None:
    metadata = pd.DataFrame(
        {"image_id": ["ISIC_0000001"], "lesion_id": ["lesion_1"], "dx": ["nv"]}
    )

    with pytest.raises(ValueError, match="Missing image files"):
        attach_image_paths(metadata, {})


def test_create_grouped_stratified_split_creates_expected_labels() -> None:
    metadata = _synthetic_metadata()

    split_df = create_grouped_stratified_split(metadata, seed=7)

    assert set(split_df["split"].unique()) == {"train", "val", "test"}


def test_verify_no_group_leakage_returns_true_for_valid_split() -> None:
    metadata = _synthetic_metadata()
    split_df = create_grouped_stratified_split(metadata, seed=11)

    assert verify_no_group_leakage(split_df) is True


def test_verify_no_group_leakage_raises_value_error_for_leakage() -> None:
    split_df = pd.DataFrame(
        {
            "lesion_id": ["lesion_1", "lesion_1", "lesion_2"],
            "split": ["train", "test", "val"],
            "dx": ["nv", "nv", "mel"],
        }
    )

    with pytest.raises(ValueError, match="Group leakage detected"):
        verify_no_group_leakage(split_df)


def test_summarize_split_returns_non_empty_summary_tables() -> None:
    metadata = _synthetic_metadata()
    split_df = create_grouped_stratified_split(metadata, seed=13)

    split_summary, class_distribution = summarize_split(split_df)

    assert not split_summary.empty
    assert not class_distribution.empty
    assert {"split", "count", "percentage"}.issubset(split_summary.columns)


def _synthetic_metadata() -> pd.DataFrame:
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    rows: list[dict[str, str]] = []
    for class_label in classes:
        for group_idx in range(18):
            lesion_id = f"{class_label}_lesion_{group_idx:02d}"
            for image_idx in range(2):
                rows.append(
                    {
                        "lesion_id": lesion_id,
                        "image_id": f"{lesion_id}_image_{image_idx}",
                        "dx": class_label,
                    }
                )
    return pd.DataFrame(rows)
