import numpy as np
import pandas as pd
import pytest

from src.inference import collect_mc_dropout_probabilities
from src.uncertainty import (
    ensemble_mean_probabilities,
    mc_dropout_error_detection_table,
    mc_dropout_uncertainty_scores_dataframe,
    mutual_information,
    probability_variance,
)


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_mc_dropout_uncertainty_scores_dataframe_accepts_stochastic_probs() -> None:
    stochastic_probabilities, labels = _synthetic_stochastic_probs_and_labels()

    scores_df = mc_dropout_uncertainty_scores_dataframe(
        stochastic_probabilities,
        labels,
        CLASSES,
    )

    assert len(scores_df) == len(labels)


def test_mc_dropout_output_contains_required_uncertainty_columns() -> None:
    stochastic_probabilities, labels = _synthetic_stochastic_probs_and_labels()
    metadata_df = pd.DataFrame(
        {
            "image_id": [f"img_{index}" for index in range(len(labels))],
            "lesion_id": [f"lesion_{index}" for index in range(len(labels))],
        }
    )

    scores_df = mc_dropout_uncertainty_scores_dataframe(
        stochastic_probabilities,
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


def test_mc_dropout_mean_probabilities_sum_to_one() -> None:
    stochastic_probabilities, _ = _synthetic_stochastic_probs_and_labels()

    mean_probabilities = ensemble_mean_probabilities(stochastic_probabilities)

    assert np.allclose(mean_probabilities.sum(axis=1), 1.0)


def test_mc_dropout_mutual_information_is_finite_and_non_negative() -> None:
    stochastic_probabilities, _ = _synthetic_stochastic_probs_and_labels()

    scores = mutual_information(stochastic_probabilities)

    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_mc_dropout_probability_variance_is_finite_and_non_negative() -> None:
    stochastic_probabilities, _ = _synthetic_stochastic_probs_and_labels()

    scores = probability_variance(stochastic_probabilities)

    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_mc_dropout_error_detection_table_contains_all_four_methods() -> None:
    stochastic_probabilities, labels = _synthetic_stochastic_probs_and_labels()
    scores_df = mc_dropout_uncertainty_scores_dataframe(
        stochastic_probabilities,
        labels,
        CLASSES,
    )

    table = mc_dropout_error_detection_table(scores_df)

    assert table["score"].tolist() == [
        "confidence_uncertainty",
        "predictive_entropy",
        "mutual_information",
        "probability_variance",
    ]


def test_invalid_stochastic_probability_dimensions_raise_value_error() -> None:
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="shape \\[M, N, C\\]"):
        mc_dropout_uncertainty_scores_dataframe(
            np.array([[0.7, 0.3], [0.2, 0.8]]),
            labels,
            classes=["akiec", "bcc"],
        )


def test_collect_mc_dropout_probabilities_rejects_invalid_num_passes() -> None:
    with pytest.raises(ValueError, match="num_passes must be greater than 1"):
        collect_mc_dropout_probabilities(
            model=None,
            dataloader=[],
            device="cpu",
            num_passes=1,
        )


def _synthetic_stochastic_probs_and_labels() -> tuple[np.ndarray, np.ndarray]:
    base = np.array(
        [
            [0.78, 0.05, 0.04, 0.03, 0.04, 0.03, 0.03],
            [0.10, 0.62, 0.08, 0.05, 0.06, 0.04, 0.05],
            [0.06, 0.05, 0.68, 0.05, 0.06, 0.05, 0.05],
            [0.05, 0.04, 0.05, 0.08, 0.60, 0.10, 0.08],
            [0.14, 0.10, 0.08, 0.09, 0.22, 0.14, 0.23],
        ]
    )
    stochastic = np.stack(
        [
            base,
            _renormalize(base + 0.01),
            _renormalize(
                base
                + np.array(
                    [
                        [0.02, -0.01, 0.00, 0.00, 0.00, 0.00, -0.01],
                        [0.00, -0.02, 0.01, 0.00, 0.01, 0.00, 0.00],
                        [0.00, 0.00, -0.02, 0.00, 0.02, 0.00, 0.00],
                        [0.00, 0.00, 0.00, 0.02, -0.02, 0.00, 0.00],
                        [0.03, 0.00, 0.00, 0.00, -0.02, 0.00, -0.01],
                    ]
                )
            ),
            _renormalize(
                base
                + np.array(
                    [
                        [-0.02, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00],
                        [0.00, 0.02, -0.01, 0.00, -0.01, 0.00, 0.00],
                        [0.00, 0.00, 0.02, 0.00, -0.02, 0.00, 0.00],
                        [0.00, 0.00, 0.00, -0.02, 0.02, 0.00, 0.00],
                        [-0.02, 0.00, 0.00, 0.00, 0.03, 0.00, -0.01],
                    ]
                )
            ),
        ],
        axis=0,
    )
    labels = np.array([0, 1, 2, 4, 6])
    return stochastic, labels


def _renormalize(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, None)
    return clipped / clipped.sum(axis=1, keepdims=True)
