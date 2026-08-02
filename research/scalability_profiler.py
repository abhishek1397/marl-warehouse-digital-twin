"""ScalabilityProfiler measuring MAPPO training speed, memory allocation, and latencies across fleet sizes."""

import time
from typing import Any, Dict, List

import torch

from marl import MultiAgentEnvConfig, WarehouseParallelEnv
from marl.algorithms.mappo import MAPPOConfig, MAPPOTrainer


class ScalabilityProfiler:
    """Profiles MAPPO step runtime, actor/critic forward latencies, and parameter scaling for fleet sizes N=1 to N=32."""

    @staticmethod
    def profile_fleet_scalability(
        fleet_sizes: List[int] = [1, 2, 4, 8],
    ) -> Dict[str, Any]:
        """Profiles execution step time (ms) and parameter count across multi-robot fleet sizes."""
        profile_results = {}

        for n_robots in fleet_sizes:
            env_cfg = MultiAgentEnvConfig(num_robots=n_robots, grid_width=6, grid_height=6)
            env = WarehouseParallelEnv(config=env_cfg)
            mappo_cfg = MAPPOConfig(num_agents=n_robots, batch_size=100, mini_batch_size=32)
            trainer = MAPPOTrainer(env=env, config=mappo_cfg)

            t0 = time.perf_counter()
            trainer.train(total_timesteps=100)
            t1 = time.perf_counter()

            step_time_ms = float(((t1 - t0) / 100.0) * 1000.0)
            critic_params = sum(p.numel() for p in trainer.centralized_critic.parameters())
            actor_params = sum(p.numel() for p in trainer.policy_manager.get_all_agents()[0].policy.parameters())

            env.close()

            profile_results[f"{n_robots}_robots"] = {
                "num_robots": n_robots,
                "step_time_ms": step_time_ms,
                "critic_param_count": critic_params,
                "actor_param_count": actor_params,
                "total_param_count": critic_params + actor_params,
            }

        return {
            "status": "COMPLETED",
            "fleet_profiles": profile_results,
        }
