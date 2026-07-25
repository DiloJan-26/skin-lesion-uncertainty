import numpy as np
import pandas as pd

from src.referral import (
    add_referral_decision,
    evaluate_referral_threshold,
    referral_results_at_coverages,
    select_uncertainty_threshold_for_coverage,
)
from src.uncertainty import (
    confidence_scores,
    confidence_uncertainty,
    error_detection_auroc,
    predictive_entropy,
    uncertainty_scores_dataframe,
)


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_confidence_scores_returns_max_probability() -> None:
    probs, _ = _synthetic_probs_and_labels()

    scores = confidence_scores(probs)

    assert np.allclose(scores[:3], [0.8, 0.6, 0.7])


def test_confidence_uncertainty_returns_one_minus_confidence() -> None:
    probs, _ = _synthetic_probs_and_labels()

    uncertainty = confidence_uncertainty(probs)

    assert np.allclose(uncertainty, 1.0 - confidence_scores(probs))


def test_predictive_entropy_returns_finite_non_negative_values() -> None:
    probs, _ = _synthetic_probs_and_labels()

    entropy = predictive_entropy(probs)

    assert np.isfinite(entropy).all()
    assert (entropy >= 0).all()


def test_uncertainty_scores_dataframe_returns_required_columns() -> None:
    probs, labels = _synthetic_probs_and_labels()
    metadata_df = pd.DataFrame(
        {
            "image_id": [f"img_{index}" for index in range(len(labels))],
            "lesion_id": [f"lesion_{index}" for index in range(len(labels))],
        }
    )

    scores_df = uncertainty_scores_dataframe(probs, labels, CLASSES, metadata_df)

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
        "correct",
    }
    assert expected_columns.issubset(scores_df.columns)


def test_error_detection_auroc_returns_value_between_zero_and_one_or_nan() -> None:
    probs, labels = _synthetic_probs_and_labels()
    scores_df = uncertainty_scores_dataframe(probs, labels, CLASSES)

    auroc = error_detection_auroc(
        scores_df["confidence_uncertainty"],
        scores_df["correct"],
    )

    assert np.isnan(auroc) or 0.0 <= auroc <= 1.0


def test_referral_results_at_coverages_returns_rows_for_all_coverages() -> None:
    probs, labels = _synthetic_probs_and_labels()
    uncertainties = confidence_uncertainty(probs)
    coverages = [1.0, 0.75, 0.5]

    results_df = referral_results_at_coverages(
        probs,
        labels,
        uncertainties,
        coverages,
        CLASSES,
        method_name="confidence_uncertainty",
    )

    assert len(results_df) == len(coverages)
    assert results_df.loc[results_df["coverage"] == 1.0, "accepted_count"].item() == len(labels)


def test_select_uncertainty_threshold_for_coverage_returns_finite_threshold() -> None:
    probs, _ = _synthetic_probs_and_labels()
    uncertainties = confidence_uncertainty(probs)

    threshold = select_uncertainty_threshold_for_coverage(
        uncertainties,
        target_coverage=0.75,
    )

    assert np.isfinite(threshold)


def test_evaluate_referral_threshold_returns_coverage_and_selective_accuracy() -> None:
    probs, labels = _synthetic_probs_and_labels()
    uncertainties = confidence_uncertainty(probs)
    threshold = select_uncertainty_threshold_for_coverage(uncertainties, 0.75)

    result = evaluate_referral_threshold(
        probs,
        labels,
        uncertainties,
        threshold,
        CLASSES,
        method_name="confidence_uncertainty",
    )

    assert "coverage" in result
    assert "selective_accuracy" in result
    assert 0.0 < result["coverage"] <= 1.0


def test_add_referral_decision_adds_accept_refer_decisions() -> None:
    probs, labels = _synthetic_probs_and_labels()
    scores_df = uncertainty_scores_dataframe(probs, labels, CLASSES)
    threshold = select_uncertainty_threshold_for_coverage(
        scores_df["confidence_uncertainty"],
        target_coverage=0.5,
    )

    decision_df = add_referral_decision(
        scores_df,
        uncertainty_column="confidence_uncertainty",
        threshold=threshold,
    )

    assert {"referral_threshold", "decision"}.issubset(decision_df.columns)
    assert set(decision_df["decision"]).issubset({"accept", "refer"})


def _synthetic_probs_and_labels() -> tuple[np.ndarray, np.ndarray]:
    probs = np.array(
        [
            [0.8, 0.05, 0.03, 0.02, 0.04, 0.03, 0.03],
            [0.1, 0.6, 0.08, 0.05, 0.07, 0.05, 0.05],
            [0.05, 0.05, 0.7, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.55, 0.1, 0.1, 0.1],
            [0.05, 0.05, 0.05, 0.05, 0.62, 0.1, 0.08],
            [0.2, 0.05, 0.05, 0.05, 0.1, 0.45, 0.1],
            [0.15, 0.1, 0.1, 0.1, 0.3, 0.15, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4],
        ]
    )
    labels = np.array([0, 1, 2, 4, 4, 5, 6, 6])
    return probs, labels
