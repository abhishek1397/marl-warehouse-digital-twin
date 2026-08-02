"""Multi-agent configuration settings for PettingZoo WarehouseParallelEnv."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MultiAgentEnvConfig:
    """Configuration options for PettingZoo WarehouseParallelEnv."""

    grid_width: int = 20
    grid_height: int = 20
    num_robots: int = 3
    num_tasks: int = 10
    max_episode_steps: int = 200

    # Observation configuration
    observation_radius: int = 3
    observation_mode: str = "local"  # "local", "global", "hybrid"

    # Reward configuration
    reward_mode: str = "individual"  # "individual", "team", "hybrid"
    team_reward_weight: float = 0.5

    # Communication configuration
    comm_mode: str = "none"  # "none", "broadcast", "radius"
    comm_radius: int = 5
    comm_msg_dim: int = 4

    seed: Optional[int] = 42
    render_mode: Optional[str] = None
    enable_reward_shaping: bool = True
    enable_action_masking: bool = True

    # Reward values
    successful_delivery_reward: float = 100.0
    package_pickup_reward: float = 20.0
    collision_penalty: float = -50.0
    invalid_action_penalty: float = -10.0
    waiting_penalty: float = -1.0
    step_time_penalty: float = -0.1
    battery_empty_penalty: float = -100.0
    successful_charging_reward: float = 5.0
