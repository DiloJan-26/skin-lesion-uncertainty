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
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_METADATA_COLUMNS = {"lesion_id", "image_id", "dx"}
EXPECTED_SPLITS = ("train", "val", "test")


def get_label_mapping(classes: list[str]) -> dict[str, int]:
    """Return a fixed class-to-index mapping from an ordered class list."""
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique to build a label mapping.")
    return {class_name: index for index, class_name in enumerate(classes)}


def add_label_indices(
    df: pd.DataFrame,
    classes: list[str],
    label_column: str = "dx",
) -> pd.DataFrame:
    """Add label_index using a fixed class order."""
    if label_column not in df.columns:
        raise ValueError(f"Dataframe is missing label column: {label_column}")

    label_mapping = get_label_mapping(classes)
    unknown_labels = sorted(set(df[label_column]).difference(label_mapping))
    if unknown_labels:
        raise ValueError(
            "Unknown class label(s) found while adding label indices: "
            f"{', '.join(map(str, unknown_labels))}"
        )

    indexed_df = df.copy()
    indexed_df["label_index"] = indexed_df[label_column].map(label_mapping).astype(int)
    return indexed_df


def compute_class_weights(
    df: pd.DataFrame,
    classes: list[str],
    label_column: str = "dx",
) -> pd.DataFrame:
    """Compute inverse-frequency class weights in the supplied class order."""
    if label_column not in df.columns:
        raise ValueError(f"Dataframe is missing label column: {label_column}")

    get_label_mapping(classes)
    counts = df[label_column].value_counts()
    total_count = len(df)
    num_classes = len(classes)
    rows: list[dict[str, float | int | str]] = []
    for class_name in classes:
        class_count = int(counts.get(class_name, 0))
        if class_count <= 0:
            raise ValueError(
                "Cannot compute class weight for class with zero examples: "
                f"{class_name}"
            )
        rows.append(
            {
                "class": class_name,
                "count": class_count,
                "weight": total_count / (num_classes * class_count),
            }
        )

    return pd.DataFrame(rows, columns=["class", "count", "weight"])


class HAM10000ImageDataset:
    """Torch-compatible HAM10000 image dataset backed by a pandas dataframe."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        classes: list[str],
        transform: Any = None,
        image_path_column: str = "image_path",
        label_column: str = "dx",
        image_id_column: str = "image_id",
        lesion_id_column: str = "lesion_id",
    ) -> None:
        """Create a dataset that returns image, label, and metadata dictionaries."""
        required_columns = {
            image_path_column,
            label_column,
            image_id_column,
            lesion_id_column,
        }
        missing_columns = sorted(required_columns.difference(dataframe.columns))
        if missing_columns:
            raise ValueError(
                "Dataset dataframe is missing required column(s): "
                f"{', '.join(missing_columns)}"
            )

        self.dataframe = add_label_indices(dataframe, classes, label_column)
        self.classes = classes
        self.transform = transform
        self.image_path_column = image_path_column
        self.label_column = label_column
        self.image_id_column = image_id_column
        self.lesion_id_column = lesion_id_column

    def __len__(self) -> int:
        """Return the number of image rows."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one RGB image and return model input plus row metadata."""
        row = self.dataframe.iloc[index]
        image_path = Path(row[self.image_path_column])
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        transformed_image = self.transform(image) if self.transform else image
        return {
            "image": transformed_image,
            "label": int(row["label_index"]),
            "image_id": str(row[self.image_id_column]),
            "lesion_id": str(row[self.lesion_id_column]),
            "class_name": str(row[self.label_column]),
        }


def get_image_transforms(image_size: int, train: bool) -> Any:
    """Create torchvision transforms for train or validation/test images."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to create image transforms. "
            "Install/use it in the Kaggle GPU environment."
        ) from exc

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def create_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Any:
    """Create a PyTorch DataLoader with lazy torch import."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "torch is required to create DataLoaders. "
            "Use this function in the Kaggle GPU environment."
        ) from exc

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_split_csvs(split_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test split CSV files from a directory."""
    directory = Path(split_dir)
    split_paths = {
        "train": directory / "train_split.csv",
        "val": directory / "val_split.csv",
        "test": directory / "test_split.csv",
    }

    missing_paths = [str(path) for path in split_paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required split CSV file(s): " f"{', '.join(missing_paths)}"
        )

    return {
        split_name: pd.read_csv(path)
        for split_name, path in split_paths.items()
    }


def build_datasets_and_loaders(
    split_dir: str | Path,
    classes: list[str],
    image_size: int,
    batch_size: int,
    fallback_batch_size: int | None = None,
    num_workers: int = 2,
) -> tuple[dict[str, HAM10000ImageDataset], dict[str, Any], pd.DataFrame]:
    """Build datasets, dataloaders, and class weights from saved split CSVs."""
    if batch_size <= 0:
        if fallback_batch_size is None or fallback_batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        batch_size = fallback_batch_size

    split_dfs = {
        split_name: add_label_indices(split_df, classes)
        for split_name, split_df in load_split_csvs(split_dir).items()
    }
    transforms = {
        "train": get_image_transforms(image_size=image_size, train=True),
        "val": get_image_transforms(image_size=image_size, train=False),
        "test": get_image_transforms(image_size=image_size, train=False),
    }
    datasets = {
        split_name: HAM10000ImageDataset(
            dataframe=split_df,
            classes=classes,
            transform=transforms[split_name],
        )
        for split_name, split_df in split_dfs.items()
    }
    dataloaders = {
        "train": create_dataloader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": create_dataloader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": create_dataloader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    class_weights_df = compute_class_weights(split_dfs["train"], classes)

    return datasets, dataloaders, class_weights_df


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
