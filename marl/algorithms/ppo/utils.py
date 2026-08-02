"""PPO utility helper functions for advantage clipping and explained variance."""

import torch


def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Computes explained variance metric: 1 - Var(y_true - y_pred) / Var(y_true)."""
    var_y = torch.var(y_true)
    if var_y == 0.0:
        return 0.0
    var_diff = torch.var(y_true - y_pred)
    return float(1.0 - (var_diff / var_y).item())


def clip_advantages(advantages: torch.Tensor, max_val: float = 10.0) -> torch.Tensor:
    """Clips advantage estimates to [-max_val, max_val]."""
    return torch.clamp(advantages, min=-max_val, max=max_val)
