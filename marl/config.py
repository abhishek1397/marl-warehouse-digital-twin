"""Configuration dataclasses for the Warehouse Gymnasium environment."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    """Configuration settings for WarehouseGymEnv."""

    grid_width: int = 20
    grid_height: int = 20
    max_episode_steps: int = 200
    observation_radius: int = 3
    robot_count: int = 1
    task_count: int = 5
    seed: Optional[int] = 42
    render_mode: Optional[str] = None

    # Configurable reward values
    successful_delivery_reward: float = 100.0
    package_pickup_reward: float = 20.0
    collision_penalty: float = -50.0
    invalid_action_penalty: float = -10.0
    waiting_penalty: float = -1.0
    step_time_penalty: float = -0.1
    battery_empty_penalty: float = -100.0
    successful_charging_reward: float = 5.0

    # Potential-Based Reward Shaping options
    enable_reward_shaping: bool = False
    shaping_scale: float = 1.0
    shaping_gamma: float = 0.99

    # Dynamic Action Masking options
    enable_action_masking: bool = False
