"""Data normalization functions for advantages, rewards, and observations."""

from typing import Tuple

import torch


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalizes advantage estimates to zero mean and unit variance."""
    if len(advantages) <= 1:
        return advantages
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)


def normalize_rewards(
    rewards: torch.Tensor,
    running_mean: float = 0.0,
    running_std: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalizes step rewards by running std."""
    return (rewards - running_mean) / (running_std + eps)


def normalize_observations(
    obs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalizes observation tensors using running mean and std tensors."""
    return (obs - mean) / (std + eps)
