"""Gymnasium action and observation space specifications for WarehouseGymEnv."""

import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np

from marl.config import EnvConfig


def get_action_space() -> Discrete:
    """Returns the discrete action space (8 actions).

    Actions:
        0: Move Up (North)
        1: Move Down (South)
        2: Move Left (West)
        3: Move Right (East)
        4: Wait
        5: Pick Package
        6: Drop Package
        7: Go Charge
    """
    return Discrete(8)


def get_observation_space(config: EnvConfig) -> Dict:
    """Returns the Gymnasium Dict observation space for an agent based on environment config."""
    max_dim = max(config.grid_width, config.grid_height)
    win_size = config.observation_radius * 2 + 1

    return Dict(
        {
            "robot_position": Box(low=0, high=max_dim, shape=(2,), dtype=np.int32),
            "goal_position": Box(low=0, high=max_dim, shape=(2,), dtype=np.int32),
            "battery_level": Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "package_status": Box(low=0, high=3, shape=(1,), dtype=np.int32),
            "local_occupancy": Box(low=0, high=5, shape=(win_size, win_size), dtype=np.int32),
            "charging_station_distance": Box(low=0.0, high=float(max_dim * 2), shape=(1,), dtype=np.float32),
            "task_status": Box(low=0, high=4, shape=(1,), dtype=np.int32),
        }
    )
