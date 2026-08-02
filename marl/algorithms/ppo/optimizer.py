"""PPOOptimizer class wrapping PyTorch optimizers with automatic gradient norm clipping."""

from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer


class PPOOptimizer:
    """Optimizer wrapper handling Adam/AdamW setup, gradient clipping, and norm tracking."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        lr: float = 0.0003,
        max_grad_norm: float = 0.5,
        opt_type: str = "adam",
        eps: float = 1e-5,
    ) -> None:
        self.max_grad_norm: float = max_grad_norm
        self.param_list = list(parameters)

        if opt_type.lower() == "adamw":
            self.optimizer: Optimizer = AdamW(self.param_list, lr=lr, eps=eps)
        else:
            self.optimizer: Optimizer = Adam(self.param_list, lr=lr, eps=eps)

    def zero_grad(self) -> None:
        """Zeros parameter gradients."""
        self.optimizer.zero_grad()

    def step(self, loss: torch.Tensor) -> float:
        """Performs backpropagation, gradient norm clipping, and optimizer step.

        Returns:
            Calculated scalar gradient norm before clipping.
        """
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm > 0.0:
            grad_norm = float(nn.utils.clip_grad_norm_(self.param_list, max_norm=self.max_grad_norm))
        else:
            grad_norm = 0.0

        self.optimizer.step()
        return grad_norm

    def set_lr(self, lr: float) -> None:
        """Updates learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Returns current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
