"""Model construction and lightweight model summaries.

The timm dependency is imported lazily so local tests can run without the full
Kaggle GPU stack installed.
"""

from typing import Any

import pandas as pd


def create_efficientnet_classifier(
    backbone: str = "efficientnet_b0",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_probability: float = 0.2,
) -> Any:
    """Create an EfficientNet classifier with a configurable output head."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required to create EfficientNet classifiers. "
            "Install/use it in the Kaggle GPU environment."
        ) from exc

    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_probability,
    )


def count_trainable_parameters(model: Any) -> int:
    """Count parameters that require gradient updates."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model: Any) -> int:
    """Count all model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def summarize_model(
    model: Any,
    backbone: str,
    num_classes: int,
    pretrained: bool,
) -> pd.DataFrame:
    """Return a one-row model summary table."""
    return pd.DataFrame(
        [
            {
                "backbone": backbone,
                "num_classes": num_classes,
                "pretrained": pretrained,
                "total_parameters": count_total_parameters(model),
                "trainable_parameters": count_trainable_parameters(model),
            }
        ]
    )
