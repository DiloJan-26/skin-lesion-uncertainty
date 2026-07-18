"""Inference export helpers for baseline HAM10000 experiments."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import softmax_numpy


def load_checkpoint(model: Any, checkpoint_path: str | Path, device: Any) -> dict[str, Any]:
    """Load a torch checkpoint into a model and return the checkpoint payload."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to load checkpoints. Use this on Kaggle GPU."
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def collect_logits(
    model: Any,
    dataloader: Any,
    device: Any,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Collect logits, labels, and row metadata from a dataloader."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to collect logits. Use this on Kaggle GPU."
        ) from exc

    device_obj = torch.device(device)
    amp_enabled = use_amp and device_obj.type == "cuda" and torch.cuda.is_available()
    logits_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    metadata_rows: list[dict[str, str]] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device_obj, non_blocking=True)
            labels = batch["label"].to(device_obj, non_blocking=True).long()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)

            logits_batches.append(logits.detach().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            for image_id, lesion_id, class_name in zip(
                batch["image_id"],
                batch["lesion_id"],
                batch["class_name"],
            ):
                metadata_rows.append(
                    {
                        "image_id": str(image_id),
                        "lesion_id": str(lesion_id),
                        "class_name": str(class_name),
                    }
                )

    return (
        np.concatenate(logits_batches, axis=0),
        np.concatenate(label_batches, axis=0),
        pd.DataFrame(metadata_rows),
    )


def save_logits_and_labels(
    logits: Any,
    labels: Any,
    output_dir: str | Path,
    split_name: str,
    seed: int,
) -> dict[str, Path]:
    """Save logits and labels as NumPy arrays for later calibration."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths = {
        "logits": destination / f"{split_name}_logits_seed{seed}.npy",
        "labels": destination / f"{split_name}_labels_seed{seed}.npy",
    }
    np.save(saved_paths["logits"], np.asarray(logits))
    np.save(saved_paths["labels"], np.asarray(labels))
    return saved_paths


def prediction_dataframe_from_logits(
    logits: Any,
    labels: Any,
    metadata_df: pd.DataFrame,
    classes: list[str],
) -> pd.DataFrame:
    """Build a prediction table with probabilities and correctness flags."""
    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels, dtype=int)
    probabilities = softmax_numpy(logits_array)
    predicted_labels = np.argmax(probabilities, axis=1).astype(int)

    if len(metadata_df) != len(labels_array):
        raise ValueError(
            "metadata_df length must match logits and labels length "
            f"({len(metadata_df)} != {len(labels_array)})."
        )

    predictions_df = metadata_df[["image_id", "lesion_id"]].copy()
    predictions_df["true_label"] = labels_array
    predictions_df["true_class"] = [classes[index] for index in labels_array]
    predictions_df["predicted_label"] = predicted_labels
    predictions_df["predicted_class"] = [
        classes[index] for index in predicted_labels
    ]
    predictions_df["confidence"] = probabilities.max(axis=1)
    predictions_df["correct"] = labels_array == predicted_labels
    for class_index, class_name in enumerate(classes):
        predictions_df[f"prob_{class_name}"] = probabilities[:, class_index]

    return predictions_df
