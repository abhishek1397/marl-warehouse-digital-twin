"""ValueNetwork for standalone state-value V(s) estimation."""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from gymnasium.spaces import Space

from marl.networks.base import BaseNetwork
from marl.networks.critic import CriticNetwork


class ValueNetwork(BaseNetwork):
    """Standalone Value Network wrapper for state-value V(s) estimation."""

    def __init__(
        self,
        observation_space: Union[Space, int],
        feature_dim: int = 128,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        init_type: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.critic: CriticNetwork = CriticNetwork(
            observation_space=observation_space,
            action_dim=None,  # Output 1 scalar value V(s)
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            init_type=init_type,
        )

    def forward(self, obs: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        """Forward pass returning scalar state-value V(s) tensor of shape (B, 1)."""
        return self.critic(obs)
