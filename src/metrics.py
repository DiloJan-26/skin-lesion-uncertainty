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
