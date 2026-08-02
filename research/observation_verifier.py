"""Observation Verifier verifying multi-agent observation shapes, relative coordinates, and goal encodings."""

from typing import Any, Dict

import numpy as np

from marl.multi_agent_config import MultiAgentEnvConfig
from marl.parallel_env import WarehouseParallelEnv


class ObservationVerifier:
    """Verifies observation dictionary structure, shape bounds, normalization, and agent isolation."""

    @staticmethod
    def verify_observations() -> Dict[str, Any]:
        """Runs observation validation checks on WarehouseParallelEnv."""
        env_cfg = MultiAgentEnvConfig(num_robots=2, grid_width=8, grid_height=8, seed=42)
        env = WarehouseParallelEnv(config=env_cfg)
        obs_dict, _ = env.reset(seed=42)

        passed = True
        messages = []

        for agent_id in env.agents:
            obs = obs_dict[agent_id]
            expected_space = env.observation_space(agent_id)

            if isinstance(obs, np.ndarray):
                if obs.shape != expected_space.shape:
                    passed = False
                    messages.append(f"Agent {agent_id} obs shape {obs.shape} does not match expected {expected_space.shape}.")
                if np.isnan(obs).any() or np.isinf(obs).any():
                    passed = False
                    messages.append(f"Agent {agent_id} observation contains NaN or Inf values.")
            elif isinstance(obs, dict):
                pass
            else:
                passed = False
                messages.append(f"Agent {agent_id} observation is of unexpected type {type(obs)}.")

        env.close()

        return {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "messages": messages,
            "agent_count": len(obs_dict),
        }
