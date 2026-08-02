"""IPPOMetricsTracker module calculating multi-agent rewards, throughput, collisions, and Jain's fairness index."""

from typing import Dict, List

import numpy as np


class IPPOMetricsTracker:
    """Tracks multi-agent training telemetry, per-agent rewards, throughput, and fairness index."""

    @staticmethod
    def compute_jains_fairness(values: List[float]) -> float:
        """Calculates Jain's Fairness Index across agent values: J = (sum x_i)^2 / (N * sum x_i^2)."""
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        if n == 0:
            return 1.0

        # Shift to non-negative if necessary
        min_v = np.min(arr)
        if min_v < 0:
            arr = arr - min_v + 1e-5

        sum_val = np.sum(arr)
        sum_sq = np.sum(arr ** 2)

        if sum_sq <= 1e-12:
            return 1.0

        jain = (sum_val ** 2) / (n * sum_sq)
        return float(np.clip(jain, 0.0, 1.0))

    @staticmethod
    def aggregate_multi_agent_metrics(
        agent_rewards: Dict[str, float],
        agent_losses: Dict[str, float],
        collisions: int = 0,
        throughput: float = 0.0,
    ) -> Dict[str, float]:
        """Aggregates per-agent metrics into multi-agent summary metrics."""
        rewards_list = list(agent_rewards.values())
        mean_reward = float(np.mean(rewards_list)) if rewards_list else 0.0
        fairness = IPPOMetricsTracker.compute_jains_fairness(rewards_list)

        metrics = {
            "mean_reward": mean_reward,
            "jains_fairness": fairness,
            "total_collisions": float(collisions),
            "throughput": float(throughput),
        }

        for agent_id, rew in agent_rewards.items():
            metrics[f"reward_{agent_id}"] = float(rew)
        for agent_id, loss in agent_losses.items():
            metrics[f"loss_{agent_id}"] = float(loss)

        return metrics
