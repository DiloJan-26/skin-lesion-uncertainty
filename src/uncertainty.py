"""Uncertainty scoring utilities for calibrated baseline predictions."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.metrics import softmax_numpy


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


def stack_member_probabilities(member_logits: Any) -> np.ndarray:
    """Convert member logits to probabilities and stack as [M, N, C]."""
    logits_list = [np.asarray(logits, dtype=np.float64) for logits in member_logits]
    if not logits_list:
        raise ValueError("member_logits must contain at least one member array.")

    reference_shape = logits_list[0].shape
    if len(reference_shape) != 2:
        raise ValueError(
            "Each member logits array must have shape [N, C], "
            f"got {reference_shape}."
        )
    mismatched_shapes = [
        logits.shape for logits in logits_list if logits.shape != reference_shape
    ]
    if mismatched_shapes:
        raise ValueError(
            "All member logits arrays must have identical shape. "
            f"Expected {reference_shape}, got {mismatched_shapes[0]}."
        )

    return np.stack([softmax_numpy(logits) for logits in logits_list], axis=0)


def ensemble_mean_probabilities(member_probabilities: Any) -> np.ndarray:
    """Average member probabilities over the model axis."""
    probabilities = _validate_member_probabilities(member_probabilities)
    return probabilities.mean(axis=0)


def expected_entropy(member_probabilities: Any, eps: float = 1e-12) -> np.ndarray:
    """Average member predictive entropy for each sample."""
    probabilities = _validate_member_probabilities(member_probabilities)
    clipped = np.clip(probabilities, eps, 1.0)
    member_entropies = -np.sum(clipped * np.log(clipped), axis=2)
    return member_entropies.mean(axis=0)


def ensemble_predictive_entropy(
    member_probabilities: Any,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute entropy of ensemble mean probabilities for each sample."""
    return predictive_entropy(
        ensemble_mean_probabilities(member_probabilities),
        eps=eps,
    )


def mutual_information(member_probabilities: Any, eps: float = 1e-12) -> np.ndarray:
    """Compute ensemble mutual information uncertainty for each sample."""
    information = ensemble_predictive_entropy(
        member_probabilities,
        eps=eps,
    ) - expected_entropy(member_probabilities, eps=eps)
    return np.clip(information, 0.0, None)


def probability_variance(member_probabilities: Any) -> np.ndarray:
    """Compute mean class-probability variance across ensemble members."""
    probabilities = _validate_member_probabilities(member_probabilities)
    return probabilities.var(axis=0).mean(axis=1)


def ensemble_uncertainty_scores_dataframe(
    member_probabilities: Any,
    labels: Any,
    classes: list[str],
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-sample uncertainty table for ensemble predictions."""
    probabilities = _validate_member_probabilities(member_probabilities)
    mean_probabilities = ensemble_mean_probabilities(probabilities)
    labels_array = np.asarray(labels, dtype=int)
    predictions, correct = prediction_correctness(mean_probabilities, labels_array)

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
    scores_df["confidence"] = confidence_scores(mean_probabilities)
    scores_df["confidence_uncertainty"] = confidence_uncertainty(mean_probabilities)
    scores_df["predictive_entropy"] = ensemble_predictive_entropy(probabilities)
    scores_df["expected_entropy"] = expected_entropy(probabilities)
    scores_df["mutual_information"] = mutual_information(probabilities)
    scores_df["probability_variance"] = probability_variance(probabilities)
    scores_df["correct"] = correct
    return scores_df


def ensemble_error_detection_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ensemble error-detection AUROC for standard uncertainty scores."""
    return uncertainty_error_detection_table(
        scores_df,
        score_columns=[
            "confidence_uncertainty",
            "predictive_entropy",
            "mutual_information",
            "probability_variance",
        ],
    )


def mc_dropout_uncertainty_scores_dataframe(
    stochastic_probabilities: Any,
    labels: Any,
    classes: list[str],
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build MC Dropout uncertainty scores from [T, N, C] probabilities."""
    return ensemble_uncertainty_scores_dataframe(
        stochastic_probabilities,
        labels,
        classes,
        metadata_df=metadata_df,
    )


def mc_dropout_error_detection_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Compute MC Dropout error-detection AUROC for standard uncertainty scores."""
    return ensemble_error_detection_table(scores_df)


def _validate_member_probabilities(member_probabilities: Any) -> np.ndarray:
    probabilities = np.asarray(member_probabilities, dtype=np.float64)
    if probabilities.ndim != 3:
        raise ValueError(
            "member_probabilities must have shape [M, N, C], "
            f"got {probabilities.shape}."
        )
    return probabilities
