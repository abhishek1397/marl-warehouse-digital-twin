"""Helper utilities for Spatial MAPPO (S-MAPPO)."""

from typing import Dict, List


def format_spatial_mappo_summary(mean_reward: float, cnn_critic_loss: float, fairness: float) -> str:
    """Formats Spatial MAPPO training progress as a clean string summary."""
    return f"Reward: {mean_reward:.2f} | CNN Critic Loss: {cnn_critic_loss:.4f} | Fairness: {fairness:.2f}"
