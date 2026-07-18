import numpy as np
import pandas as pd

from src.inference import prediction_dataframe_from_logits
from src.metrics import (
    classification_metrics_from_logits,
    classification_report_dataframe,
    confusion_matrix_dataframe,
    softmax_numpy,
)


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_softmax_numpy_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 999.0]])

    probabilities = softmax_numpy(logits)

    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_classification_metrics_from_logits_returns_accuracy_and_macro_f1() -> None:
    logits = _example_logits()
    labels = np.array([0, 4, 5, 6])

    metrics = classification_metrics_from_logits(logits, labels, CLASSES)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert isinstance(metrics["accuracy"], float)


def test_melanoma_recall_is_present() -> None:
    metrics = classification_metrics_from_logits(
        _example_logits(),
        np.array([0, 4, 5, 6]),
        CLASSES,
    )

    assert "melanoma_recall" in metrics


def test_classification_report_dataframe_returns_dataframe() -> None:
    report_df = classification_report_dataframe(
        labels=np.array([0, 4, 5, 6]),
        predictions=np.array([0, 4, 5, 5]),
        classes=CLASSES,
    )

    assert isinstance(report_df, pd.DataFrame)
    assert "precision" in report_df.columns


def test_confusion_matrix_dataframe_returns_square_dataframe_with_class_labels() -> None:
    confusion_df = confusion_matrix_dataframe(
        labels=np.array([0, 4, 5, 6]),
        predictions=np.array([0, 4, 5, 5]),
        classes=CLASSES,
    )

    assert confusion_df.shape == (len(CLASSES), len(CLASSES))
    assert confusion_df.index.tolist() == CLASSES
    assert confusion_df.columns.tolist() == CLASSES


def test_prediction_dataframe_from_logits_does_not_require_torch() -> None:
    metadata_df = pd.DataFrame(
        {
            "image_id": ["img_1", "img_2"],
            "lesion_id": ["lesion_1", "lesion_2"],
            "class_name": ["akiec", "mel"],
        }
    )
    logits = np.array(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ]
    )
    labels = np.array([0, 4])

    predictions_df = prediction_dataframe_from_logits(
        logits,
        labels,
        metadata_df,
        CLASSES,
    )

    assert predictions_df["predicted_class"].tolist() == ["akiec", "mel"]
    assert "prob_mel" in predictions_df.columns


def _example_logits() -> np.ndarray:
    return np.array(
        [
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
        ]
    )
