"""CriticNetwork for state-value V(s) and action-value Q(s, a) estimation."""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from gymnasium.spaces import Space

from marl.networks.base import BaseNetwork
from marl.networks.feature_extractor import FeatureExtractor
from marl.networks.mlp import MLP


class CriticNetwork(BaseNetwork):
    """Critic Network evaluating scalar state-value V(s) or state-action value Q(s, a)."""

    def __init__(
        self,
        observation_space: Union[Space, int],
        action_dim: Optional[int] = None,
        feature_dim: int = 128,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        init_type: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.action_dim: Optional[int] = action_dim

        self.feature_extractor: FeatureExtractor = FeatureExtractor(
            observation_space=observation_space,
            output_dim=feature_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            init_type=init_type,
        )

        value_out_dim = 1 if action_dim is None else action_dim
        self.value_head: MLP = MLP(
            input_dim=feature_dim,
            output_dim=value_out_dim,
            hidden_dims=[],
            activation=activation,
            init_type=init_type,
            gain=1.0,
        )

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing state-value V(s) or Q(s, a).

        Returns:
            Scalar value tensor of shape (B, 1) or Q-values of shape (B, action_dim).
        """
        features = self.feature_extractor(obs)
        val = self.value_head(features)

        if self.action_dim is not None and action is not None:
            # Extract specific Q(s, a) value if discrete action tensor is provided
            if action.dim() == 1:
                action = action.unsqueeze(-1)
            val = val.gather(1, action)

        return val
