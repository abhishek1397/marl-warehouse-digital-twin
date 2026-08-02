"""CentralizedValueNetwork processing global warehouse state for MAPPO Centralized Critic V(S)."""

import torch
import torch.nn as nn


class CentralizedValueNetwork(nn.Module):
    """Centralized Value Network V(S) mapping global state tensors to scalar value estimates."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim: int = state_dim
        self.hidden_dim: int = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Computes global value estimate V(S) for input state tensor."""
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        return self.net(state)
