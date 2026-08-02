"""IPPOConfig dataclass holding hyperparameters for Independent PPO training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IPPOConfig:
    """Hyperparameter configuration settings for Independent PPO (IPPO)."""

    num_agents: int = 2
    shared_policy: bool = False  # False = Independent policy per agent, True = Shared weights
    learning_rate: float = 3e-4
    epochs: int = 4
    batch_size: int = 400
    mini_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.015
    enable_action_masking: bool = True
    enable_reward_shaping: bool = True
    eval_interval: int = 1000
    eval_episodes: int = 5
    seed: Optional[int] = 42
    device: str = "cpu"
