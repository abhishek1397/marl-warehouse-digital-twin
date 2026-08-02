"""ActorNetwork for policy action selection and probability distribution generation."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from gymnasium.spaces import Space

from marl.networks.base import BaseNetwork
from marl.networks.distribution import CategoricalDistribution
from marl.networks.feature_extractor import FeatureExtractor
from marl.networks.mlp import MLP


class ActorNetwork(BaseNetwork):
    """Actor Network computing action logits and action probability distributions."""

    def __init__(
        self,
        observation_space: Union[Space, int],
        action_dim: int = 8,
        hidden_dims: Optional[list[int]] = None,
        feature_dim: int = 128,
        activation: str = "relu",
        init_type: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.action_dim: int = action_dim

        self.feature_extractor: FeatureExtractor = FeatureExtractor(
            observation_space=observation_space,
            output_dim=feature_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            init_type=init_type,
        )

        self.logits_head: MLP = MLP(
            input_dim=feature_dim,
            output_dim=action_dim,
            hidden_dims=[],  # Direct linear projection from feature_dim -> action_dim
            activation=activation,
            init_type=init_type,
            gain=0.01,  # Small gain for action logits initialization
        )

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        temperature: float = 1.0,
    ) -> CategoricalDistribution:
        """Forward pass generating action probability distribution."""
        features = self.feature_extractor(obs)
        logits = self.logits_head(features)
        return CategoricalDistribution(logits=logits, temperature=temperature)

    def get_distribution(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        temperature: float = 1.0,
    ) -> CategoricalDistribution:
        """Returns CategoricalDistribution object."""
        return self.forward(obs, temperature=temperature)

    def sample_action(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples an action and computes its log probability.

        Returns:
            Tuple of (action_tensor, log_prob_tensor).
        """
        dist = self.get_distribution(obs, temperature=temperature)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
