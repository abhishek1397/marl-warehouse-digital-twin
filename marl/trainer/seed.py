"""Seeding management for multi-framework reproducible execution."""

import random
from typing import Optional

import numpy as np
import torch

from marl.utils import set_seed as set_marl_seed


def seed_everything(seed: int, torch_deterministic: bool = True) -> None:
    """Seeds Python, NumPy, PyTorch, Gymnasium, PettingZoo, and simulator.

    Args:
        seed: Integer seed value.
        torch_deterministic: If True, configures PyTorch cuDNN to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch_deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Seed Gymnasium / PettingZoo / Simulator RNGs
    set_marl_seed(seed)
