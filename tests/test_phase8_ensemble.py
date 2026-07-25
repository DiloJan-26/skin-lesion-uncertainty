import numpy as np
import pandas as pd
import pytest

from src.calibration import apply_temperature_to_probabilities
from src.metrics import (
    calibration_metrics_from_probabilities,
    classification_metrics_from_probabilities,
)
from src.uncertainty import (
    ensemble_error_detection_table,
    ensemble_mean_probabilities,
    ensemble_predictive_entropy,
    ensemble_uncertainty_scores_dataframe,
    expected_entropy,
    mutual_information,
    probability_variance,
    stack_member_probabilities,
)


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_stack_member_probabilities_returns_member_sample_class_shape() -> None:
    member_logits, _ = _synthetic_member_logits_and_labels()

    member_probabilities = stack_member_probabilities(member_logits)

    assert member_probabilities.shape == (3, 5, 7)


def test_stack_member_probabilities_requires_matching_shapes() -> None:
    member_logits, _ = _synthetic_member_logits_and_labels()
    member_logits[1] = member_logits[1][:4]

    with pytest.raises(ValueError, match="identical shape"):
        stack_member_probabilities(member_logits)


def test_ensemble_mean_probabilities_sums_to_one_per_sample() -> None:
    member_probabilities = _synthetic_member_probabilities()

    mean_probabilities = ensemble_mean_probabilities(member_probabilities)

    assert np.allclose(mean_probabilities.sum(axis=1), 1.0)


def test_expected_entropy_is_finite_and_non_negative() -> None:
    entropy = expected_entropy(_synthetic_member_probabilities())

    assert np.isfinite(entropy).all()
    assert (entropy >= 0).all()


def test_predictive_entropy_is_finite_and_non_negative() -> None:
    entropy = ensemble_predictive_entropy(_synthetic_member_probabilities())

    assert np.isfinite(entropy).all()
    assert (entropy >= 0).all()


def test_mutual_information_is_finite_and_non_negative() -> None:
    scores = mutual_information(_synthetic_member_probabilities())

    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_probability_variance_is_finite_and_non_negative() -> None:
    scores = probability_variance(_synthetic_member_probabilities())

    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_ensemble_uncertainty_dataframe_contains_required_columns() -> None:
    member_probabilities = _synthetic_member_probabilities()
    _, labels = _synthetic_member_logits_and_labels()
    metadata_df = pd.DataFrame(
        {
            "image_id": [f"img_{index}" for index in range(len(labels))],
            "lesion_id": [f"lesion_{index}" for index in range(len(labels))],
        }
    )

    scores_df = ensemble_uncertainty_scores_dataframe(
        member_probabilities,
        labels,
        CLASSES,
        metadata_df=metadata_df,
    )

    expected_columns = {
        "image_id",
        "lesion_id",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "confidence",
        "confidence_uncertainty",
        "predictive_entropy",
        "expected_entropy",
        "mutual_information",
        "probability_variance",
        "correct",
    }
    assert expected_columns.issubset(scores_df.columns)


def test_apply_temperature_to_probabilities_returns_valid_probabilities() -> None:
    probabilities = ensemble_mean_probabilities(_synthetic_member_probabilities())

    calibrated = apply_temperature_to_probabilities(probabilities, temperature=1.5)

    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_classification_metrics_from_probabilities_returns_required_metrics() -> None:
    probabilities = ensemble_mean_probabilities(_synthetic_member_probabilities())
    _, labels = _synthetic_member_logits_and_labels()

    metrics = classification_metrics_from_probabilities(
        probabilities,
        labels,
        CLASSES,
    )

    assert {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "melanoma_recall",
    }.issubset(metrics)


def test_calibration_metrics_from_probabilities_returns_required_metrics() -> None:
    probabilities = ensemble_mean_probabilities(_synthetic_member_probabilities())
    _, labels = _synthetic_member_logits_and_labels()

    metrics = calibration_metrics_from_probabilities(
        probabilities,
        labels,
        CLASSES,
        n_bins=5,
    )

    assert {"accuracy", "nll", "brier_score", "ece"}.issubset(metrics)


def test_ensemble_error_detection_table_contains_all_four_methods() -> None:
    member_probabilities = _synthetic_member_probabilities()
    _, labels = _synthetic_member_logits_and_labels()
    scores_df = ensemble_uncertainty_scores_dataframe(
        member_probabilities,
        labels,
        CLASSES,
    )

    table = ensemble_error_detection_table(scores_df)

    assert table["score"].tolist() == [
        "confidence_uncertainty",
        "predictive_entropy",
        "mutual_information",
        "probability_variance",
    ]


def _synthetic_member_probabilities() -> np.ndarray:
    member_logits, _ = _synthetic_member_logits_and_labels()
    return stack_member_probabilities(member_logits)


def _synthetic_member_logits_and_labels() -> tuple[list[np.ndarray], np.ndarray]:
    base_logits = np.array(
        [
            [4.0, 0.5, 0.2, 0.1, 0.1, 0.0, 0.0],
            [0.2, 3.0, 0.4, 0.1, 0.2, 0.0, 0.1],
            [0.1, 0.2, 2.8, 0.2, 0.3, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.1, 2.5, 0.5, 0.2],
            [0.2, 0.1, 0.0, 0.2, 0.4, 0.5, 1.8],
        ]
    )
    member_logits = [
        base_logits,
        base_logits + np.array(
            [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.0, -0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, -0.1],
            ]
        ),
        base_logits + np.array(
            [
                [-0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.1, 0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.1, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.2],
            ]
        ),
    ]
    labels = np.array([0, 1, 2, 4, 6])
    return member_logits, labels
