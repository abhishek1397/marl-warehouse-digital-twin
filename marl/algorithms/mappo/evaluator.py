"""MAPPOEvaluator conducting multi-episode evaluations for multi-robot fleets."""

from typing import Dict, List, Optional

import numpy as np
import torch

from marl.algorithms.mappo.metrics import MAPPOMetricsTracker
from marl.algorithms.mappo.shared_policy import SharedPolicyManager
from marl.parallel_env import WarehouseParallelEnv


class MAPPOEvaluator:
    """Evaluates MAPPO multi-agent fleet policy performance during decentralized execution."""

    def __init__(self, env: WarehouseParallelEnv) -> None:
        self.env: WarehouseParallelEnv = env

    def evaluate(
        self,
        policy_manager: SharedPolicyManager,
        num_episodes: int = 5,
        seed: Optional[int] = 42,
    ) -> Dict[str, float]:
        """Evaluates policy manager performance and computes aggregated metrics."""
        ep_rewards: List[float] = []
        ep_lengths: List[float] = []
        total_collisions = 0
        total_deliveries = 0

        for ep_idx in range(num_episodes):
            ep_seed = seed + ep_idx if seed is not None else None
            obs_dict, info_dict = self.env.reset(seed=ep_seed)

            done = False
            step_count = 0
            ep_reward_sum = 0.0

            while not done:
                actions_dict = {}

                with torch.no_grad():
                    for agent_id in self.env.agents:
                        if agent_id in obs_dict:
                            agent = policy_manager.get_agent(agent_id)
                            mask = info_dict.get(agent_id, {}).get("action_mask", None)
                            action = agent.predict(obs_dict[agent_id], mask=mask, deterministic=True)
                            act_int = int(action.item() if isinstance(action, torch.Tensor) and action.numel() == 1 else action)
                            actions_dict[agent_id] = act_int

                next_obs, rewards, terminations, truncations, infos = self.env.step(actions_dict)
                step_count += 1
                step_rew = sum(rewards.values()) if rewards else 0.0
                ep_reward_sum += step_rew

                # Count collisions & deliveries
                for agent_id, info in infos.items():
                    if not info.get("action_valid", True):
                        total_collisions += 1
                    if "Delivered package" in str(info.get("action_message", "")):
                        total_deliveries += 1

                obs_dict, info_dict = next_obs, infos
                done = not self.env.agents or all(terminations.values()) or all(truncations.values())

            ep_rewards.append(ep_reward_sum)
            ep_lengths.append(float(step_count))

        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_length = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        fairness = MAPPOMetricsTracker.compute_jains_fairness(ep_rewards)

        return {
            "eval_mean_reward": mean_reward,
            "eval_mean_length": mean_length,
            "eval_total_collisions": float(total_collisions),
            "eval_total_deliveries": float(total_deliveries),
            "eval_throughput": float(total_deliveries / max(1.0, sum(ep_lengths))),
            "eval_jains_fairness": fairness,
        }
