"""Classification metrics for baseline HAM10000 experiments."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable row-wise softmax."""
    logits_array = np.asarray(logits, dtype=np.float64)
    shifted_logits = logits_array - np.max(logits_array, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def classification_metrics_from_logits(
    logits: Any,
    labels: Any,
    classes: list[str],
) -> dict[str, float]:
    """Compute core classification metrics from logits and integer labels."""
    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels)
    predictions = np.argmax(logits_array, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions,
        labels=list(range(len(classes))),
        average="macro",
        zero_division=0,
    )
    melanoma_index = classes.index("mel")
    melanoma_recall = recall_score(
        labels_array,
        predictions,
        labels=[melanoma_index],
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "melanoma_recall": float(melanoma_recall),
    }


def classification_report_dataframe(
    labels: Any,
    predictions: Any,
    classes: list[str],
) -> pd.DataFrame:
    """Return a sklearn classification report as a dataframe."""
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_matrix_dataframe(
    labels: Any,
    predictions: Any,
    classes: list[str],
) -> pd.DataFrame:
    """Return a confusion matrix dataframe with class names on both axes."""
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(classes))),
    )
    return pd.DataFrame(matrix, index=classes, columns=classes)


def negative_log_likelihood_from_probs(probs: Any, labels: Any) -> float:
    """Compute mean negative log-likelihood from probabilities."""
    probabilities = np.asarray(probs, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=int)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    sample_probs = clipped[np.arange(len(labels_array)), labels_array]
    return float(-np.mean(np.log(sample_probs)))


def multiclass_brier_score(probs: Any, labels: Any, num_classes: int) -> float:
    """Compute the multiclass Brier score with one-hot labels."""
    probabilities = np.asarray(probs, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=int)
    one_hot = np.zeros((len(labels_array), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels_array)), labels_array] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probs: Any,
    labels: Any,
    n_bins: int = 15,
) -> float:
    """Compute standard confidence-bin expected calibration error."""
    curve_df = reliability_curve_dataframe(probs, labels, n_bins=n_bins)
    total_count = int(curve_df["count"].sum())
    if total_count == 0:
        return 0.0

    weighted_gap = (
        curve_df["count"]
        * (curve_df["accuracy"] - curve_df["confidence"]).abs()
    ).sum()
    return float(weighted_gap / total_count)


def calibration_metrics_from_logits(
    logits: Any,
    labels: Any,
    classes: list[str],
    temperature: float | None = None,
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute accuracy, NLL, Brier score, and ECE from logits."""
    logits_array = np.asarray(logits, dtype=np.float64)
    if temperature is not None:
        from src.calibration import apply_temperature_scaling

        logits_array = apply_temperature_scaling(logits_array, temperature)

    labels_array = np.asarray(labels, dtype=int)
    probabilities = softmax_numpy(logits_array)
    predictions = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "nll": negative_log_likelihood_from_probs(probabilities, labels_array),
        "brier_score": multiclass_brier_score(
            probabilities,
            labels_array,
            num_classes=len(classes),
        ),
        "ece": expected_calibration_error(
            probabilities,
            labels_array,
            n_bins=n_bins,
        ),
    }


def reliability_curve_dataframe(
    probs: Any,
    labels: Any,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Return bin-level accuracy and confidence for reliability diagrams."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    probabilities = np.asarray(probs, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=int)
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == labels_array

    rows: list[dict[str, float | int]] = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_index in range(n_bins):
        lower = float(bin_edges[bin_index])
        upper = float(bin_edges[bin_index + 1])
        if bin_index == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        count = int(in_bin.sum())
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_center": (lower + upper) / 2.0,
                "count": count,
                "accuracy": float(correct[in_bin].mean()) if count else 0.0,
                "confidence": float(confidences[in_bin].mean()) if count else 0.0,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "bin_lower",
            "bin_upper",
            "bin_center",
            "count",
            "accuracy",
            "confidence",
        ],
    )
