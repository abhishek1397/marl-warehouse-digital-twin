"""BufferStatistics providing telemetry analytics over rollout buffer data."""

from typing import Dict, List, Optional

import numpy as np
import torch

from marl.storage.transition import Transition


class BufferStatistics:
    """Computes statistical diagnostics over stored transitions and advantage estimates."""

    @staticmethod
    def compute(
        transitions: List[Transition],
        advantages: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        capacity: int = 10000,
    ) -> Dict[str, float]:
        """Calculates statistical summary dictionary for stored buffer data."""
        if not transitions:
            return {
                "num_transitions": 0.0,
                "utilization": 0.0,
                "mean_reward": 0.0,
                "mean_advantage": 0.0,
                "std_advantage": 0.0,
                "mean_return": 0.0,
            }

        rewards = [t.reward for t in transitions]
        stats = {
            "num_transitions": float(len(transitions)),
            "utilization": float(len(transitions) / max(capacity, 1)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }

        if advantages is not None and len(advantages) > 0:
            adv_np = advantages.detach().cpu().numpy()
            stats["mean_advantage"] = float(np.mean(adv_np))
            stats["std_advantage"] = float(np.std(adv_np))
            stats["min_advantage"] = float(np.min(adv_np))
            stats["max_advantage"] = float(np.max(adv_np))

        if returns is not None and len(returns) > 0:
            ret_np = returns.detach().cpu().numpy()
            stats["mean_return"] = float(np.mean(ret_np))
            stats["std_return"] = float(np.std(ret_np))

        return stats
