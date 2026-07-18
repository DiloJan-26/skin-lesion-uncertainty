"""Kaggle-oriented baseline training and validation utilities."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import classification_metrics_from_logits


def create_weighted_cross_entropy_loss(class_weights_df: pd.DataFrame, device: Any) -> Any:
    """Create torch CrossEntropyLoss from a class-weight dataframe."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to create weighted CrossEntropyLoss. "
            "Use this function in the Kaggle GPU environment."
        ) from exc

    required_columns = {"class", "count", "weight"}
    missing_columns = required_columns.difference(class_weights_df.columns)
    if missing_columns:
        raise ValueError(
            "class_weights_df is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )

    weights = torch.tensor(
        class_weights_df["weight"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    return torch.nn.CrossEntropyLoss(weight=weights)


def create_optimizer(model: Any, learning_rate: float, weight_decay: float) -> Any:
    """Create an AdamW optimizer for model training."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to create optimizers. "
            "Use this function in the Kaggle GPU environment."
        ) from exc

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    use_amp: bool = True,
) -> dict[str, float]:
    """Train a model for one epoch and return mean loss and accuracy."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for training. Use this function on Kaggle GPU."
        ) from exc

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    device_obj = torch.device(device)
    amp_enabled = use_amp and device_obj.type == "cuda" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    iterator = tqdm(dataloader, desc="train", leave=False) if tqdm else dataloader

    for batch in iterator:
        images = batch["image"].to(device_obj, non_blocking=True)
        labels = batch["label"].to(device_obj, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def evaluate_model(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    classes: list[str],
    use_amp: bool = True,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a model and return metrics, logits, and labels."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for evaluation. Use this function on Kaggle GPU."
        ) from exc

    device_obj = torch.device(device)
    amp_enabled = use_amp and device_obj.type == "cuda" and torch.cuda.is_available()

    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device_obj, non_blocking=True)
            labels = batch["label"].to(device_obj, non_blocking=True).long()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total_examples += batch_size
            logits_batches.append(logits.detach().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())

    logits_array = np.concatenate(logits_batches, axis=0)
    labels_array = np.concatenate(label_batches, axis=0)
    metrics = classification_metrics_from_logits(logits_array, labels_array, classes)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics, logits_array, labels_array


def train_baseline_model(
    model: Any,
    dataloaders: dict[str, Any],
    criterion: Any,
    optimizer: Any,
    device: Any,
    classes: list[str],
    max_epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
    seed: int,
    use_amp: bool = True,
) -> pd.DataFrame:
    """Train a baseline model with early stopping on validation macro F1."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for baseline training. Use this on Kaggle GPU."
        ) from exc

    checkpoint_destination = Path(checkpoint_path)
    history_destination = Path(history_path)
    checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    history_destination.parent.mkdir(parents=True, exist_ok=True)

    best_val_macro_f1 = -float("inf")
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
        )
        val_metrics, _, _ = evaluate_model(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            classes=classes,
            use_amp=use_amp,
        )

        val_macro_f1 = val_metrics["macro_f1"]
        improved = val_macro_f1 > best_val_macro_f1
        if improved:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "best_val_macro_f1": best_val_macro_f1,
                    "classes": classes,
                },
                checkpoint_destination,
            )
        else:
            epochs_without_improvement += 1

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_macro_f1,
            "val_melanoma_recall": val_metrics["melanoma_recall"],
            "best_val_macro_f1": best_val_macro_f1,
        }
        history_rows.append(history_row)
        pd.DataFrame(history_rows).to_csv(history_destination, index=False)

        print(
            f"Epoch {epoch}/{max_epochs} - "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    return pd.DataFrame(history_rows)
