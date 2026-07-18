"""Reproducibility utilities for lightweight local and Kaggle runs."""

import importlib.util
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed common RNGs; deterministic PyTorch mode may reduce performance."""
    random.seed(seed)
    np.random.seed(seed)

    if importlib.util.find_spec("torch") is None:
        return

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
