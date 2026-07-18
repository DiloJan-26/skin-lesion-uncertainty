"""Matplotlib plotting helpers for HAM10000 exploratory split checks."""

from pathlib import Path
from typing import Any

import numpy as np
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


def plot_training_curves(history_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save baseline training loss and validation macro F1 curves."""
    required_columns = {"epoch", "train_loss", "val_loss", "val_macro_f1"}
    missing_columns = required_columns.difference(history_df.columns)
    if missing_columns:
        raise ValueError(
            "history_df is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        history_df["epoch"],
        history_df["val_macro_f1"],
        label="val macro F1",
        color="#F58518",
    )
    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_confusion_matrix(confusion_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a confusion matrix heatmap using matplotlib only."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    values = confusion_df.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(confusion_df.columns)))
    ax.set_yticks(np.arange(len(confusion_df.index)))
    ax.set_xticklabels(confusion_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(confusion_df.index)

    threshold = values.max() / 2 if values.size else 0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            count = int(values[row_index, column_index])
            color = "white" if count > threshold else "black"
            ax.text(
                column_index,
                row_index,
                str(count),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_reliability_diagram(
    uncalibrated_curve_df: pd.DataFrame,
    calibrated_curve_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a reliability diagram comparing uncalibrated and calibrated curves."""
    required_columns = {"confidence", "accuracy"}
    for name, curve_df in {
        "uncalibrated_curve_df": uncalibrated_curve_df,
        "calibrated_curve_df": calibrated_curve_df,
    }.items():
        missing_columns = required_columns.difference(curve_df.columns)
        if missing_columns:
            raise ValueError(
                f"{name} is missing required column(s): "
                f"{', '.join(sorted(missing_columns))}"
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", label="perfect")
    ax.plot(
        uncalibrated_curve_df["confidence"],
        uncalibrated_curve_df["accuracy"],
        marker="o",
        label="uncalibrated",
    )
    ax.plot(
        calibrated_curve_df["confidence"],
        calibrated_curve_df["accuracy"],
        marker="o",
        label="calibrated",
    )
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination
