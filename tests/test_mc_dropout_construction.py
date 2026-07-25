"""Torch-optional regression tests for corrected MC Dropout construction."""

from typing import Any

import pytest

from src.inference import (
    collect_mc_dropout_probabilities,
    enable_mc_dropout,
    inspect_dropout_modules,
)


torch = pytest.importorskip("torch")


def test_enable_mc_dropout_enables_dropout_and_keeps_batchnorm_in_eval() -> None:
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(8),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(8, 3),
    )
    model.train()

    enable_mc_dropout(model)

    assert model[1].training is True
    assert model[0].training is False


def test_repeated_forward_passes_differ_with_mc_dropout_enabled() -> None:
    model = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(16, 7),
    )
    inputs = torch.ones(32, 16)

    enable_mc_dropout(model)
    first = model(inputs)
    second = model(inputs)

    assert not torch.equal(first, second)


def test_collection_rejects_model_without_active_dropout() -> None:
    model = torch.nn.Linear(4, 2)

    with pytest.raises(ValueError, match="at least one supported dropout"):
        collect_mc_dropout_probabilities(
            model=model,
            dataloader=[],
            device="cpu",
            num_passes=2,
            use_amp=False,
        )


def test_collection_rejects_identical_stochastic_passes() -> None:
    class UnusedDropoutModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.5)
            self.classifier = torch.nn.Linear(4, 2)

        def forward(self, inputs: Any) -> Any:
            return self.classifier(inputs)

    batch = {
        "image": torch.ones(3, 4),
        "label": torch.tensor([0, 1, 0]),
        "image_id": ["a", "b", "c"],
        "lesion_id": ["la", "lb", "lc"],
        "class_name": ["akiec", "bcc", "akiec"],
    }

    with pytest.raises(RuntimeError, match="identical probabilities"):
        collect_mc_dropout_probabilities(
            model=UnusedDropoutModel(),
            dataloader=[batch],
            device="cpu",
            num_passes=2,
            use_amp=False,
        )


def test_efficientnet_has_explicit_dropout_and_correct_output_shape() -> None:
    pytest.importorskip("timm")
    from src.models import create_efficientnet_classifier

    model = create_efficientnet_classifier(
        pretrained=False,
        num_classes=7,
        dropout_probability=0.2,
    )
    dropout_modules = inspect_dropout_modules(model)

    assert any(module.p > 0 for module in dropout_modules)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 7)


def test_existing_timm_checkpoint_keys_remain_strictly_compatible() -> None:
    timm: Any = pytest.importorskip("timm")
    from src.models import create_efficientnet_classifier

    previous_model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=7,
        drop_rate=0.2,
    )
    previous_state = previous_model.state_dict()
    corrected_model = create_efficientnet_classifier(
        pretrained=False,
        num_classes=7,
        dropout_probability=0.2,
    )

    assert set(corrected_model.state_dict()) == set(previous_state)
    assert "classifier.weight" in previous_state
    assert "classifier.bias" in previous_state
    incompatible = corrected_model.load_state_dict(previous_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
