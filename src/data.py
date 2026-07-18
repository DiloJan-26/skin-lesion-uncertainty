"""HAM10000 dataset discovery, metadata handling, and leakage-safe splitting.

These utilities are intentionally framework-light so they can be used from
Kaggle notebooks before model training code is introduced. Grouped stratified
splits may produce approximate percentages because lesion-level groups cannot
always be divided into exact train/validation/test proportions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_METADATA_COLUMNS = {"lesion_id", "image_id", "dx"}
EXPECTED_SPLITS = ("train", "val", "test")


def find_ham10000_paths(input_root: str | Path) -> dict[str, Any]:
    """Find HAM10000 metadata and image directories below an input root."""
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"HAM10000 input root does not exist: {root}")

    metadata_matches = sorted(root.rglob("HAM10000_metadata.csv"))
    if not metadata_matches:
        raise FileNotFoundError(
            f"Could not find HAM10000_metadata.csv below input root: {root}"
        )

    image_dirs_by_name: dict[str, list[Path]] = {
        "HAM10000_images_part_1": [],
        "HAM10000_images_part_2": [],
    }
    for path in root.rglob("*"):
        if path.is_dir() and path.name in image_dirs_by_name:
            image_dirs_by_name[path.name].append(path)

    missing_dirs = [
        name for name, matches in image_dirs_by_name.items() if not matches
    ]
    if missing_dirs:
        raise FileNotFoundError(
            "Could not find required HAM10000 image folder(s) below "
            f"{root}: {', '.join(missing_dirs)}"
        )

    metadata_path = metadata_matches[0]
    image_dirs = [
        sorted(image_dirs_by_name["HAM10000_images_part_1"])[0],
        sorted(image_dirs_by_name["HAM10000_images_part_2"])[0],
    ]
    common_root = Path(os.path.commonpath([metadata_path, *image_dirs]))

    return {
        "metadata_path": metadata_path,
        "image_dirs": image_dirs,
        "dataset_root": common_root,
    }


def load_ham10000_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load HAM10000 metadata and validate required columns."""
    path = Path(metadata_path)
    metadata = pd.read_csv(path)
    missing_columns = REQUIRED_METADATA_COLUMNS.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            "HAM10000 metadata is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )
    return metadata


def build_image_path_map(image_dirs: list[str | Path]) -> dict[str, Path]:
    """Map image ID stems to image file paths from supplied directories."""
    image_path_map: dict[str, Path] = {}
    for image_dir in image_dirs:
        directory = Path(image_dir)
        for pattern in ("*.jpg", "*.JPG", "*.png", "*.PNG"):
            for image_path in directory.rglob(pattern):
                image_path_map[image_path.stem] = image_path
    return image_path_map


def attach_image_paths(
    metadata: pd.DataFrame,
    image_path_map: dict[str, Path],
    image_id_column: str = "image_id",
) -> pd.DataFrame:
    """Attach image file paths to metadata rows by image ID."""
    if image_id_column not in metadata.columns:
        raise ValueError(f"Metadata is missing image ID column: {image_id_column}")

    attached = metadata.copy()
    attached["image_path"] = attached[image_id_column].map(image_path_map)

    missing_mask = attached["image_path"].isna()
    if missing_mask.any():
        missing_ids = attached.loc[missing_mask, image_id_column].astype(str).tolist()
        preview = ", ".join(missing_ids[:10])
        suffix = "..." if len(missing_ids) > 10 else ""
        raise ValueError(
            "Missing image files for metadata image_id value(s): "
            f"{preview}{suffix}"
        )

    attached["image_path"] = attached["image_path"].map(Path)
    return attached


def create_grouped_stratified_split(
    metadata: pd.DataFrame,
    label_column: str = "dx",
    group_column: str = "lesion_id",
    train_fraction: float = 0.80,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Create train/val/test splits while keeping lesion groups intact.

    The implementation uses a 10-fold grouped stratified split for the test set
    and a 9-fold grouped stratified split on the remainder for validation, so
    final percentages are approximate when group sizes are uneven.
    """
    _validate_split_fractions(train_fraction, val_fraction, test_fraction)
    _validate_split_columns(metadata, label_column, group_column)

    split_df = metadata.reset_index(drop=True).copy()
    labels = split_df[label_column]
    groups = split_df[group_column]

    test_splitter = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=seed
    )
    _, test_indices = next(test_splitter.split(split_df, labels, groups))

    split_df["split"] = "train"
    split_df.loc[test_indices, "split"] = "test"

    remaining_df = split_df.loc[split_df["split"] != "test"].copy()
    remaining_labels = remaining_df[label_column]
    remaining_groups = remaining_df[group_column]

    val_splitter = StratifiedGroupKFold(
        n_splits=9, shuffle=True, random_state=seed
    )
    _, val_relative_indices = next(
        val_splitter.split(remaining_df, remaining_labels, remaining_groups)
    )
    val_indices = remaining_df.iloc[val_relative_indices].index
    split_df.loc[val_indices, "split"] = "val"

    verify_no_group_leakage(split_df, group_column=group_column)
    return split_df.reset_index(drop=True)


def verify_no_group_leakage(
    split_df: pd.DataFrame,
    group_column: str = "lesion_id",
    split_column: str = "split",
) -> bool:
    """Return True when no group appears in more than one split."""
    for column in (group_column, split_column):
        if column not in split_df.columns:
            raise ValueError(f"Split dataframe is missing required column: {column}")

    split_counts = split_df.groupby(group_column)[split_column].nunique()
    leaked_groups = split_counts[split_counts > 1].index.astype(str).tolist()
    if leaked_groups:
        preview = ", ".join(leaked_groups[:20])
        suffix = "..." if len(leaked_groups) > 20 else ""
        raise ValueError(f"Group leakage detected for group ID(s): {preview}{suffix}")

    return True


def summarize_split(
    split_df: pd.DataFrame,
    label_column: str = "dx",
    split_column: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize split sizes and class counts by split."""
    for column in (label_column, split_column):
        if column not in split_df.columns:
            raise ValueError(f"Split dataframe is missing required column: {column}")

    split_counts = split_df[split_column].value_counts().reindex(EXPECTED_SPLITS)
    split_counts = split_counts.dropna().astype(int)
    split_summary = split_counts.rename("count").reset_index()
    split_summary = split_summary.rename(columns={"index": split_column})
    split_summary["percentage"] = split_summary["count"] / len(split_df)

    class_distribution = pd.crosstab(split_df[split_column], split_df[label_column])
    class_distribution = class_distribution.reindex(EXPECTED_SPLITS).dropna(how="all")
    class_distribution = class_distribution.fillna(0).astype(int)
    class_distribution.index.name = split_column
    class_distribution_by_split = class_distribution.reset_index()

    return split_summary, class_distribution_by_split


def save_split_files(
    split_df: pd.DataFrame,
    output_dir: str | Path,
    label_column: str = "dx",
) -> dict[str, Path]:
    """Save split CSV files and summary tables to an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_summary, class_distribution = summarize_split(
        split_df, label_column=label_column
    )
    saved_paths = {
        "train_split": output_path / "train_split.csv",
        "val_split": output_path / "val_split.csv",
        "test_split": output_path / "test_split.csv",
        "split_summary": output_path / "split_summary.csv",
        "class_distribution_by_split": output_path
        / "class_distribution_by_split.csv",
    }

    for split_name in EXPECTED_SPLITS:
        split_df.loc[split_df["split"] == split_name].to_csv(
            saved_paths[f"{split_name}_split"], index=False
        )
    split_summary.to_csv(saved_paths["split_summary"], index=False)
    class_distribution.to_csv(
        saved_paths["class_distribution_by_split"], index=False
    )

    return saved_paths


def _validate_split_fractions(
    train_fraction: float, val_fraction: float, test_fraction: float
) -> None:
    fractions = {
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
    }
    invalid_names = [
        name
        for name, value in fractions.items()
        if not isinstance(value, int | float) or value <= 0 or value >= 1
    ]
    if invalid_names:
        raise ValueError(
            "Split fractions must be numeric values between 0 and 1: "
            f"{', '.join(invalid_names)}"
        )

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "Split fractions must sum to 1.0 "
            f"(got {total:.6f})."
        )


def _validate_split_columns(
    metadata: pd.DataFrame, label_column: str, group_column: str
) -> None:
    missing_columns = [
        column
        for column in (label_column, group_column)
        if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(
            "Metadata is missing required split column(s): "
            f"{', '.join(missing_columns)}"
        )
