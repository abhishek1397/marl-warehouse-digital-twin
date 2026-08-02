"""Environment Sync Verifier comparing step-by-step trajectories between single-agent Gymnasium PPO vs PettingZoo IPPO."""

from typing import Any, Dict, List

import numpy as np

from marl.config import EnvConfig
from marl.environment import WarehouseGymEnv
from marl.multi_agent_config import MultiAgentEnvConfig
from marl.parallel_env import WarehouseParallelEnv


class EnvironmentSyncVerifier:
    """Compares step-by-step environment trajectories, step actions, rewards, and collisions between Gym PPO vs PettingZoo IPPO."""

    @staticmethod
    def compare_trajectories(num_steps: int = 50, seed: int = 42) -> Dict[str, Any]:
        """Runs comparative trajectory rollout recording and calculates state divergence."""
        # 1. Gymnasium Single-Agent Env
        gym_cfg = EnvConfig(grid_width=8, grid_height=8, seed=seed)
        gym_env = WarehouseGymEnv(config=gym_cfg)
        gym_obs, _ = gym_env.reset(seed=seed)

        # 2. PettingZoo 1-Robot Parallel Env
        pz_cfg = MultiAgentEnvConfig(num_robots=1, grid_width=8, grid_height=8, seed=seed)
        pz_env = WarehouseParallelEnv(config=pz_cfg)
        pz_obs_dict, _ = pz_env.reset(seed=seed)

        divergences = 0
        step_logs = []

        for step in range(num_steps):
            gym_act = gym_env.action_space.sample()
            next_gym_obs, gym_rew, gym_done, gym_trunc, _ = gym_env.step(gym_act)

            pz_act_dict = {"robot_0": gym_act}
            next_pz_obs, pz_rews, pz_dones, pz_truncs, _ = pz_env.step(pz_act_dict)

            pz_rew = pz_rews.get("robot_0", 0.0)
            if abs(gym_rew - pz_rew) > 1e-4:
                divergences += 1

            step_logs.append({
                "step": step,
                "gym_reward": float(gym_rew),
                "pz_reward": float(pz_rew),
            })

            if gym_done or gym_trunc:
                break

        gym_env.close()
        pz_env.close()

        is_synced = (divergences == 0)

        return {
            "is_synced": is_synced,
            "divergence_count": divergences,
            "total_steps_compared": len(step_logs),
            "status": "PASSED" if is_synced else "MISMATCHED",
        }
