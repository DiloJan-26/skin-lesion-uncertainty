import numpy as np
import pandas as pd
import pytest

from src.calibration import temperature_scale_logits_numpy
from src.inference import prediction_dataframe_from_probabilities
from src.metrics import (
    calibration_metrics_from_logits,
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood_from_probs,
    reliability_curve_dataframe,
)


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_temperature_scale_logits_numpy_divides_logits_correctly() -> None:
    logits = np.array([[2.0, 4.0], [6.0, 8.0]])

    scaled = temperature_scale_logits_numpy(logits, temperature=2.0)

    assert np.allclose(scaled, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_temperature_scale_logits_numpy_raises_for_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="Temperature must be positive"):
        temperature_scale_logits_numpy(np.array([[1.0, 2.0]]), temperature=0.0)


def test_negative_log_likelihood_from_probs_returns_finite_positive_value() -> None:
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    labels = np.array([0, 1])

    nll = negative_log_likelihood_from_probs(probs, labels)

    assert np.isfinite(nll)
    assert nll > 0


def test_multiclass_brier_score_returns_finite_non_negative_value() -> None:
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
    labels = np.array([0, 1])

    score = multiclass_brier_score(probs, labels, num_classes=3)

    assert np.isfinite(score)
    assert score >= 0


def test_expected_calibration_error_returns_value_between_zero_and_one() -> None:
    probs = np.array([[0.8, 0.2], [0.4, 0.6], [0.55, 0.45]])
    labels = np.array([0, 1, 1])

    ece = expected_calibration_error(probs, labels, n_bins=5)

    assert 0.0 <= ece <= 1.0


def test_reliability_curve_dataframe_returns_expected_columns() -> None:
    probs = np.array([[0.8, 0.2], [0.4, 0.6]])
    labels = np.array([0, 1])

    curve_df = reliability_curve_dataframe(probs, labels, n_bins=4)

    assert curve_df.columns.tolist() == [
        "bin_lower",
        "bin_upper",
        "bin_center",
        "count",
        "accuracy",
        "confidence",
    ]
    assert len(curve_df) == 4


def test_calibration_metrics_from_logits_returns_expected_keys() -> None:
    logits = np.array(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        ]
    )
    labels = np.array([0, 4, 5])

    metrics = calibration_metrics_from_logits(
        logits,
        labels,
        CLASSES,
        temperature=1.5,
        n_bins=5,
    )

    assert {"accuracy", "nll", "brier_score", "ece"}.issubset(metrics)


def test_prediction_dataframe_from_probabilities_works_with_synthetic_probs() -> None:
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.05, 0.02, 0.02, 0.01],
            [0.1, 0.1, 0.1, 0.1, 0.45, 0.1, 0.05],
        ]
    )
    labels = np.array([0, 4])
    metadata_df = pd.DataFrame(
        {
            "image_id": ["img_1", "img_2"],
            "lesion_id": ["lesion_1", "lesion_2"],
            "class_name": ["akiec", "mel"],
        }
    )

    predictions_df = prediction_dataframe_from_probabilities(
        probs,
        labels,
        metadata_df,
        CLASSES,
    )

    assert predictions_df["predicted_class"].tolist() == ["akiec", "mel"]
    assert predictions_df["correct"].tolist() == [True, True]
    assert "prob_vasc" in predictions_df.columns
