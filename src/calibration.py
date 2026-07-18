"""Temperature Scaling utilities for post-hoc calibration."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def temperature_scale_logits_numpy(logits: Any, temperature: float) -> np.ndarray:
    """Divide logits by a positive scalar temperature."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")
    return np.asarray(logits, dtype=np.float64) / float(temperature)


def fit_temperature_scaling(
    val_logits: Any,
    val_labels: Any,
    max_iter: int = 1000,
    lr: float = 0.01,
    initial_temperature: float = 1.0,
    device: str | None = None,
) -> float:
    """Fit a positive Temperature Scaling scalar on validation logits.

    This optimizes only a standalone log-temperature parameter with validation
    negative log-likelihood; model weights are not modified.
    """
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be positive.")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to fit Temperature Scaling. "
            "Use this function in the Kaggle GPU environment."
        ) from exc

    device_obj = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logits_tensor = torch.as_tensor(
        np.asarray(val_logits), dtype=torch.float32, device=device_obj
    )
    labels_tensor = torch.as_tensor(
        np.asarray(val_labels), dtype=torch.long, device=device_obj
    )
    log_temperature = torch.nn.Parameter(
        torch.tensor(
            np.log(float(initial_temperature)),
            dtype=torch.float32,
            device=device_obj,
        )
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([log_temperature], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature)
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        optimizer.step()

    return float(torch.exp(log_temperature).detach().cpu().item())


def apply_temperature_scaling(logits: Any, temperature: float) -> np.ndarray:
    """Apply Temperature Scaling to logits using the NumPy implementation."""
    return temperature_scale_logits_numpy(logits, temperature)


def save_temperature(temperature: float, output_path: str | Path) -> Path:
    """Save a fitted temperature value as a small JSON file."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump({"temperature": float(temperature)}, file, indent=2)
        file.write("\n")
    return destination
