"""CriticAnalyzer analyzing MAPPO Centralized Value Network loss, explained variance, and prediction errors."""

from typing import Any, Dict, List

import numpy as np
import torch


class CriticAnalyzer:
    """Analyzes MAPPO Centralized Critic V(S) convergence, explained variance, and prediction errors."""

    @staticmethod
    def compute_explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculates explained variance R^2 = 1 - Var(y_true - y_pred) / Var(y_true)."""
        y_t = y_true.detach().cpu().numpy().flatten()
        y_p = y_pred.detach().cpu().numpy().flatten()

        var_y = float(np.var(y_t))
        if var_y <= 1e-12:
            return 1.0

        var_diff = float(np.var(y_t - y_p))
        ev = 1.0 - (var_diff / var_y)
        return float(np.clip(ev, -1.0, 1.0))

    @staticmethod
    def analyze_critic_trajectory(
        value_losses: List[float],
        returns: List[float],
        predictions: List[float],
    ) -> Dict[str, Any]:
        """Analyzes value loss trajectory and evaluates critic health indicators."""
        y_t = torch.tensor(returns, dtype=torch.float32)
        y_p = torch.tensor(predictions, dtype=torch.float32)

        ev = CriticAnalyzer.compute_explained_variance(y_t, y_p)
        mean_loss = float(np.mean(value_losses)) if value_losses else 0.0
        loss_std = float(np.std(value_losses)) if value_losses else 0.0

        is_overfitting = (ev < 0.0) and (mean_loss > 100.0)
        is_converging = (ev > 0.5) and (loss_std < mean_loss * 0.5)

        return {
            "status": "PASSED" if not is_overfitting else "FAILED",
            "explained_variance": ev,
            "mean_value_loss": mean_reward_loss if (mean_reward_loss := mean_loss) else 0.0,
            "value_loss_std": loss_std,
            "is_overfitting": is_overfitting,
            "is_converging": is_converging,
        }
