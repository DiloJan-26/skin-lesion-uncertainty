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
    """Create EfficientNet with explicit classifier dropout.

    The dropout module is attached inside a ``Linear`` subclass so the final
    parameters retain timm's ``classifier.weight`` and ``classifier.bias``
    state-dict keys. This keeps checkpoints from the previous factory
    loadable while making dropout discoverable for MC Dropout inference.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to create EfficientNet classifiers. "
            "Use this in the Kaggle GPU environment."
        ) from exc

    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required to create EfficientNet classifiers. "
            "Install/use it in the Kaggle GPU environment."
        ) from exc

    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError(
            "dropout_probability must be in the interval [0, 1), "
            f"got {dropout_probability}."
        )

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        # Explicit module dropout below replaces timm's functional dropout.
        drop_rate=0.0,
    )

    existing_classifier = getattr(model, "classifier", None)
    if not isinstance(existing_classifier, torch.nn.Linear):
        raise ValueError(
            f"Expected {backbone!r} to expose torch.nn.Linear as "
            "model.classifier; checkpoint-compatible replacement is not possible."
        )

    class _DropoutLinear(torch.nn.Linear):
        """Linear classifier that applies an explicit parameter-free dropout."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            probability: float,
        ) -> None:
            super().__init__(in_features, out_features, bias=bias)
            self.dropout = torch.nn.Dropout(p=probability)

        def forward(self, inputs: Any) -> Any:
            return super().forward(self.dropout(inputs))

    classifier = _DropoutLinear(
        in_features=existing_classifier.in_features,
        out_features=existing_classifier.out_features,
        bias=existing_classifier.bias is not None,
        probability=dropout_probability,
    )
    classifier.load_state_dict(existing_classifier.state_dict(), strict=True)
    model.classifier = classifier
    return model


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
