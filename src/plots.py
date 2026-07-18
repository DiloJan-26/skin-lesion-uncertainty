"""Matplotlib plotting helpers for HAM10000 exploratory split checks."""

from pathlib import Path

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
