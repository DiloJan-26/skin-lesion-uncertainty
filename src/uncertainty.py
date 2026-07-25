"""Uncertainty scoring utilities for calibrated baseline predictions."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def confidence_scores(probs: Any) -> np.ndarray:
    """Return maximum predicted probability for each sample."""
    probabilities = np.asarray(probs, dtype=np.float64)
    return np.max(probabilities, axis=1)


def confidence_uncertainty(probs: Any) -> np.ndarray:
    """Return confidence-based uncertainty, defined as 1 - max probability."""
    return 1.0 - confidence_scores(probs)


def predictive_entropy(probs: Any, eps: float = 1e-12) -> np.ndarray:
    """Return predictive entropy for each probability row."""
    probabilities = np.asarray(probs, dtype=np.float64)
    clipped = np.clip(probabilities, eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def prediction_correctness(probs: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class indices and correctness flags."""
    probabilities = np.asarray(probs, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=int)
    predictions = np.argmax(probabilities, axis=1).astype(int)
    return predictions, predictions == labels_array


def uncertainty_scores_dataframe(
    probs: Any,
    labels: Any,
    classes: list[str],
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-sample uncertainty scoring table."""
    probabilities = np.asarray(probs, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=int)
    predictions, correct = prediction_correctness(probabilities, labels_array)

    scores_df = pd.DataFrame()
    if metadata_df is not None:
        if len(metadata_df) != len(labels_array):
            raise ValueError(
                "metadata_df length must match probabilities and labels length "
                f"({len(metadata_df)} != {len(labels_array)})."
            )
        for column in ("image_id", "lesion_id"):
            if column in metadata_df.columns:
                scores_df[column] = metadata_df[column].astype(str).to_numpy()

    scores_df["true_label"] = labels_array
    scores_df["true_class"] = [classes[index] for index in labels_array]
    scores_df["predicted_label"] = predictions
    scores_df["predicted_class"] = [classes[index] for index in predictions]
    scores_df["confidence"] = confidence_scores(probabilities)
    scores_df["confidence_uncertainty"] = confidence_uncertainty(probabilities)
    scores_df["predictive_entropy"] = predictive_entropy(probabilities)
    scores_df["correct"] = correct
    return scores_df


def error_detection_auroc(uncertainty_scores: Any, correct: Any) -> float:
    """Compute AUROC for detecting incorrect predictions from uncertainty."""
    scores = np.asarray(uncertainty_scores, dtype=np.float64)
    correct_array = np.asarray(correct, dtype=bool)
    error_targets = (~correct_array).astype(int)
    if len(np.unique(error_targets)) < 2:
        return float("nan")
    return float(roc_auc_score(error_targets, scores))


def uncertainty_error_detection_table(
    scores_df: pd.DataFrame,
    score_columns: list[str],
) -> pd.DataFrame:
    """Compute error-detection AUROC for multiple uncertainty score columns."""
    if "correct" not in scores_df.columns:
        raise ValueError("scores_df must include a correct column.")

    rows: list[dict[str, float | str]] = []
    for score_column in score_columns:
        if score_column not in scores_df.columns:
            raise ValueError(f"scores_df is missing score column: {score_column}")
        rows.append(
            {
                "score": score_column,
                "error_detection_auroc": error_detection_auroc(
                    scores_df[score_column],
                    scores_df["correct"],
                ),
            }
        )
    return pd.DataFrame(rows, columns=["score", "error_detection_auroc"])
