"""Matplotlib plotting helpers for HAM10000 exploratory split checks."""

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt


def plot_overall_class_distribution(
    metadata: pd.DataFrame,
    label_column: str,
    output_path: str | Path,
) -> Path:
    """Save a PNG bar chart of overall class counts."""
    if label_column not in metadata.columns:
        raise ValueError(f"Metadata is missing label column: {label_column}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    counts = metadata[label_column].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("Overall HAM10000 Class Distribution")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Image count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_class_distribution_by_split(
    class_distribution_by_split: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a PNG grouped bar chart of class counts by split."""
    if "split" not in class_distribution_by_split.columns:
        raise ValueError("Class distribution table must include a split column.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    plot_df = class_distribution_by_split.set_index("split")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.T.plot(kind="bar", ax=ax)
    ax.set_title("HAM10000 Class Distribution by Split")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Image count")
    ax.legend(title="Split")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def denormalize_imagenet_tensor(image_tensor: Any) -> Any:
    """Convert an ImageNet-normalized CHW tensor to display range."""
    mean = image_tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = image_tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image_tensor.detach().cpu() * std + mean).clamp(0.0, 1.0)


def plot_batch_preview(
    batch: dict[str, Any],
    output_path: str | Path,
    max_images: int = 8,
) -> Path:
    """Save a small grid preview of a batch returned by HAM10000ImageDataset."""
    if "image" not in batch:
        raise ValueError("Batch must include an image entry.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    images = batch["image"]
    class_names = batch.get("class_name", [])
    num_images = min(max_images, len(images))
    if num_images <= 0:
        raise ValueError("Batch preview requires at least one image.")

    columns = min(4, num_images)
    rows = (num_images + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for image_index in range(num_images):
        ax = axes_list[image_index]
        image = denormalize_imagenet_tensor(images[image_index])
        image_array = image.permute(1, 2, 0).numpy()
        ax.imshow(image_array)
        ax.axis("off")
        if len(class_names) > image_index:
            ax.set_title(str(class_names[image_index]))

    for ax in axes_list[num_images:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
