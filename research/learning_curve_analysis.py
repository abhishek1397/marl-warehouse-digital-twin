"""LearningCurveAnalyzer generating training trajectory metrics for IPPO vs MAPPO."""

from typing import Any, Dict, List

import numpy as np


class LearningCurveAnalyzer:
    """Analyzes training curve dynamics, plateau detection, and comparative convergence speed."""

    @staticmethod
    def analyze_learning_curve(reward_history: List[float]) -> Dict[str, Any]:
        """Calculates reward growth slope, final plateau value, and variance."""
        arr = np.asarray(reward_history, dtype=float)
        if len(arr) == 0:
            return {"mean_reward": 0.0, "plateau_reached": False}

        final_reward = float(arr[-1])
        mean_reward = float(np.mean(arr))
        reward_std = float(np.std(arr))

        return {
            "mean_reward": mean_reward,
            "final_reward": final_reward,
            "reward_std": reward_std,
            "plateau_reached": bool(reward_std < abs(mean_reward) * 0.2),
        }
