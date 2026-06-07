"""Reproducibility utilities: seed every relevant RNG in one call."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA), and PYTHONHASHSEED.

    Call at the very start of any training or evaluation script to guarantee
    deterministic behaviour across runs with the same seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
