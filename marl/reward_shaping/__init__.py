"""Potential-Based Reward Shaping (PBRS) Subsystem Package."""

from marl.reward_shaping.config import RewardShapingConfig
from marl.reward_shaping.distance_metrics import (
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
)
from marl.reward_shaping.potential import ManhattanPotential, PotentialFunction
from marl.reward_shaping.reward_engine import ShapedRewardEngine, ShapedRewardOutput
from marl.reward_shaping.utils import calculate_goal_progress, calculate_shaping_reward

__all__ = [
    "RewardShapingConfig",
    "PotentialFunction",
    "ManhattanPotential",
    "ShapedRewardEngine",
    "ShapedRewardOutput",
    "manhattan_distance",
    "euclidean_distance",
    "chebyshev_distance",
    "calculate_shaping_reward",
    "calculate_goal_progress",
]
