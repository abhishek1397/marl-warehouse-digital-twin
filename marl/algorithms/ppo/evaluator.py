"""PPOEvaluator performing deterministic evaluation runs without exploration."""

from typing import Dict, Optional

import numpy as np
import torch

from marl.environment import WarehouseGymEnv
from marl.networks.policy_network import PolicyNetwork


class PPOEvaluator:
    """Evaluates policy deterministically on WarehouseGymEnv and collects performance telemetry."""

    def evaluate(
        self,
        env: WarehouseGymEnv,
        policy: PolicyNetwork,
        num_episodes: int = 5,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Runs evaluation episodes with deterministic action selection.

        Args:
            env: Target WarehouseGymEnv environment.
            policy: PolicyNetwork instance.
            num_episodes: Number of evaluation episodes to execute.
            seed: Optional random seed.

        Returns:
            Dictionary of evaluation metrics.
        """
        policy.eval()

        ep_rewards = []
        ep_lengths = []
        ep_successes = []
        ep_deliveries = []
        ep_collisions = []

        for ep_idx in range(num_episodes):
            ep_seed = seed + ep_idx if seed is not None else None
            obs, info = env.reset(seed=ep_seed)

            done = False
            total_reward = 0.0
            steps = 0
            collisions = 0

            while not done:
                mask = info.get("action_mask", None)
                with torch.no_grad():
                    action = policy.predict(obs, mask=mask, deterministic=True)

                if isinstance(action, torch.Tensor):
                    action_int = int(action.item() if action.numel() == 1 else action[0].item())
                else:
                    action_int = int(action)

                obs, reward, terminated, truncated, info = env.step(action_int)
                total_reward += reward
                steps += 1
                if not info.get("action_valid", True):
                    collisions += 1

                done = terminated or truncated

            ep_rewards.append(total_reward)
            ep_lengths.append(steps)
            ep_successes.append(1.0 if info.get("is_success", False) else 0.0)
            ep_deliveries.append(info.get("completed_deliveries", 0))
            ep_collisions.append(collisions)

        policy.train()

        total_steps = sum(ep_lengths)
        throughput = (sum(ep_deliveries) / total_steps * 100.0) if total_steps > 0 else 0.0

        return {
            "eval_mean_reward": float(np.mean(ep_rewards)),
            "eval_mean_length": float(np.mean(ep_lengths)),
            "eval_success_rate": float(np.mean(ep_successes)),
            "eval_total_deliveries": float(sum(ep_deliveries)),
            "eval_throughput": float(throughput),
            "eval_mean_collisions": float(np.mean(ep_collisions)),
        }
