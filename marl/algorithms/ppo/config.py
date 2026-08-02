"""Configuration dataclass for single-agent PPO algorithm."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """Configuration hyperparameters for PPO algorithm."""

    learning_rate: float = 0.0003
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    batch_size: int = 2048
    mini_batch_size: int = 64

    # Loss coefficients
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Scheduler and execution
    scheduler_type: str = "linear"  # "constant", "linear", "cosine"
    device: str = "cpu"
    seed: int = 42

    # Evaluation and checkpoint intervals
    eval_interval: int = 2000
    eval_episodes: int = 5
    checkpoint_interval: int = 5000
