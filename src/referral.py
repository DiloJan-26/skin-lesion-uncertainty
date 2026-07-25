"""Selective human referral evaluation for calibrated predictions."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def referral_results_at_coverages(
    probs: Any,
    labels: Any,
    uncertainty_scores: Any,
    coverages: list[float],
    classes: list[str],
    method_name: str,
) -> pd.DataFrame:
    """Evaluate selective prediction by accepting least-uncertain samples."""
    probabilities = np.asarray(probs, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=int)
    uncertainties = np.asarray(uncertainty_scores, dtype=np.float64)
    predictions = np.argmax(probabilities, axis=1).astype(int)
    total_count = len(labels_array)
    order = np.argsort(uncertainties, kind="mergesort")

    rows: list[dict[str, float | int | str]] = []
    for coverage in coverages:
        _validate_coverage(coverage)
        accepted_count = total_count if coverage == 1.0 else int(np.ceil(coverage * total_count))
        accepted_count = min(max(accepted_count, 1), total_count)
        accepted_indices = order[:accepted_count]
        rows.append(
            _referral_metrics_for_indices(
                labels_array=labels_array,
                predictions=predictions,
                accepted_indices=accepted_indices,
                total_count=total_count,
                classes=classes,
                method_name=method_name,
                threshold=None,
            )
        )

    return pd.DataFrame(rows)


def select_uncertainty_threshold_for_coverage(
    uncertainty_scores: Any,
    target_coverage: float,
) -> float:
    """Select an uncertainty threshold that accepts approximately a coverage."""
    _validate_coverage(target_coverage)
    uncertainties = np.asarray(uncertainty_scores, dtype=np.float64)
    if len(uncertainties) == 0:
        raise ValueError("uncertainty_scores must contain at least one value.")

    accepted_count = (
        len(uncertainties)
        if target_coverage == 1.0
        else int(np.ceil(target_coverage * len(uncertainties)))
    )
    accepted_count = min(max(accepted_count, 1), len(uncertainties))
    sorted_scores = np.sort(uncertainties)
    return float(sorted_scores[accepted_count - 1])


def evaluate_referral_threshold(
    probs: Any,
    labels: Any,
    uncertainty_scores: Any,
    threshold: float,
    classes: list[str],
    method_name: str,
) -> dict[str, float | int | str]:
    """Evaluate a fixed validation-selected uncertainty threshold."""
    probabilities = np.asarray(probs, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=int)
    uncertainties = np.asarray(uncertainty_scores, dtype=np.float64)
    predictions = np.argmax(probabilities, axis=1).astype(int)
    accepted_indices = np.flatnonzero(uncertainties <= threshold)

    result = _referral_metrics_for_indices(
        labels_array=labels_array,
        predictions=predictions,
        accepted_indices=accepted_indices,
        total_count=len(labels_array),
        classes=classes,
        method_name=method_name,
        threshold=float(threshold),
    )
    result["threshold"] = float(threshold)
    return result


def add_referral_decision(
    scores_df: pd.DataFrame,
    uncertainty_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Add accept/refer decisions using an uncertainty threshold."""
    if uncertainty_column not in scores_df.columns:
        raise ValueError(f"scores_df is missing uncertainty column: {uncertainty_column}")

    decision_df = scores_df.copy()
    decision_df["referral_threshold"] = float(threshold)
    decision_df["decision"] = np.where(
        decision_df[uncertainty_column] <= threshold,
        "accept",
        "refer",
    )
    return decision_df


def _referral_metrics_for_indices(
    labels_array: np.ndarray,
    predictions: np.ndarray,
    accepted_indices: np.ndarray,
    total_count: int,
    classes: list[str],
    method_name: str,
    threshold: float | None,
) -> dict[str, float | int | str]:
    accepted_count = int(len(accepted_indices))
    rejected_count = int(total_count - accepted_count)
    coverage = accepted_count / total_count if total_count else 0.0

    if accepted_count:
        accepted_labels = labels_array[accepted_indices]
        accepted_predictions = predictions[accepted_indices]
        selective_accuracy = float(accuracy_score(accepted_labels, accepted_predictions))
        macro_f1 = float(
            f1_score(
                accepted_labels,
                accepted_predictions,
                labels=list(range(len(classes))),
                average="macro",
                zero_division=0,
            )
        )
    else:
        selective_accuracy = float("nan")
        macro_f1 = float("nan")

    result: dict[str, float | int | str] = {
        "method": method_name,
        "coverage": float(coverage),
        "rejection_rate": float(1.0 - coverage),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "selective_accuracy": selective_accuracy,
        "selective_risk": float(1.0 - selective_accuracy)
        if not np.isnan(selective_accuracy)
        else float("nan"),
        "macro_f1": macro_f1,
    }
    if threshold is not None:
        result["threshold"] = threshold
    return result


def _validate_coverage(coverage: float) -> None:
    if not isinstance(coverage, int | float) or coverage <= 0 or coverage > 1:
        raise ValueError(f"Coverage must be in the interval (0, 1], got {coverage}.")
