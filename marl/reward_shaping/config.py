"""Configuration dataclass for Potential-Based Reward Shaping (PBRS)."""

from dataclasses import dataclass


@dataclass
class RewardShapingConfig:
    """Configuration settings for potential-based reward shaping following Ng et al. (1999)."""

    enable_reward_shaping: bool = True
    potential_function: str = "manhattan"  # "manhattan", "euclidean", "chebyshev"
    shaping_scale: float = 1.0
    gamma: float = 0.99
