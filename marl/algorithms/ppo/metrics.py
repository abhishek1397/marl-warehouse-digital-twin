"""PPOMetricsTracker recording policy, value, entropy, and optimization metrics."""

from collections import deque
from typing import Any, Dict

import numpy as np

from marl.algorithms.ppo.loss import PPOLossOutput


class PPOMetricsTracker:
    """Tracks PPO optimization loss metrics, learning rates, and gradient norms."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size: int = window_size

        self.policy_losses: deque[float] = deque(maxlen=window_size)
        self.value_losses: deque[float] = deque(maxlen=window_size)
        self.entropy_losses: deque[float] = deque(maxlen=window_size)
        self.total_losses: deque[float] = deque(maxlen=window_size)
        self.approx_kls: deque[float] = deque(maxlen=window_size)
        self.clip_fractions: deque[float] = deque(maxlen=window_size)
        self.grad_norms: deque[float] = deque(maxlen=window_size)
        self.learning_rates: deque[float] = deque(maxlen=window_size)

    def record_update(self, loss_out: PPOLossOutput, grad_norm: float, lr: float) -> None:
        """Records diagnostic metrics for an optimization update step."""
        self.policy_losses.append(float(loss_out.policy_loss.item()))
        self.value_losses.append(float(loss_out.value_loss.item()))
        self.entropy_losses.append(float(loss_out.entropy_loss.item()))
        self.total_losses.append(float(loss_out.total_loss.item()))
        self.approx_kls.append(float(loss_out.approx_kl.item()))
        self.clip_fractions.append(float(loss_out.clip_fraction.item()))
        self.grad_norms.append(float(grad_norm))
        self.learning_rates.append(float(lr))

    def get_summary(self) -> Dict[str, float]:
        """Returns summary dictionary of mean metrics."""
        if not self.policy_losses:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "grad_norm": 0.0,
                "learning_rate": 0.0,
            }

        return {
            "policy_loss": float(np.mean(self.policy_losses)),
            "value_loss": float(np.mean(self.value_losses)),
            "entropy": float(np.mean(self.entropy_losses)),
            "total_loss": float(np.mean(self.total_losses)),
            "approx_kl": float(np.mean(self.approx_kls)),
            "clip_fraction": float(np.mean(self.clip_fractions)),
            "grad_norm": float(np.mean(self.grad_norms)),
            "learning_rate": float(self.learning_rates[-1]),
        }

    def reset(self) -> None:
        """Resets stored metric histories."""
        self.policy_losses.clear()
        self.value_losses.clear()
        self.entropy_losses.clear()
        self.total_losses.clear()
        self.approx_kls.clear()
        self.clip_fractions.clear()
        self.grad_norms.clear()
        self.learning_rates.clear()
