"""SpatialMAPPOConfig dataclass holding hyperparameters for Spatial MAPPO (S-MAPPO) with CNN Centralized Critic."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SpatialMAPPOConfig:
    """Hyperparameter configuration settings for Spatial MAPPO (S-MAPPO)."""

    num_agents: int = 2
    cnn_channels: int = 5  # Robots, Shelves, Obstacles, Charging Stations, Packages
    conv_filters: Tuple[int, ...] = (32, 64, 128)
    hidden_dim: int = 128
    shared_policy: bool = True  # Parameter sharing across actors
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    epochs: int = 4
    batch_size: int = 400
    mini_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    enable_action_masking: bool = True
    enable_reward_shaping: bool = True
    eval_interval: int = 1000
    eval_episodes: int = 5
    seed: Optional[int] = 42
    device: str = "cpu"
