"""Randomness management and seeding utilities for deterministic execution."""

import random
from typing import Optional, Tuple

import numpy as np


def set_seed(seed: Optional[int] = None) -> Tuple[random.Random, np.random.Generator]:
    """Sets random seeds for Python random module and NumPy.

    Args:
        seed: Optional integer seed value.

    Returns:
        Tuple of (random.Random instance, np.random.Generator instance).
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    return py_rng, np_rng
