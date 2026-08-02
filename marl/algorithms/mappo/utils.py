"""Helper utilities for Multi-Agent PPO (MAPPO)."""

from typing import Dict, List

import numpy as np
import torch


def format_mappo_summary(mean_reward: float, critic_loss: float, fairness: float) -> str:
    """Formats MAPPO training progress as a clean string summary."""
    return f"Reward: {mean_reward:.2f} | Critic Loss: {critic_loss:.4f} | Fairness: {fairness:.2f}"
