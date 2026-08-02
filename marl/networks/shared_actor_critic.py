"""SharedActorCritic network combining a shared feature backbone with actor and critic heads."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from gymnasium.spaces import Space

from marl.networks.base import BaseNetwork
from marl.networks.distribution import CategoricalDistribution
from marl.networks.feature_extractor import FeatureExtractor
from marl.networks.mlp import MLP


class SharedActorCritic(BaseNetwork):
    """Shared Actor-Critic network with joint feature backbone and separate actor/critic heads."""

    def __init__(
        self,
        observation_space: Union[Space, int],
        action_dim: int = 8,
        feature_dim: int = 128,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        init_type: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.action_dim: int = action_dim

        # Shared representation backbone
        self.feature_extractor: FeatureExtractor = FeatureExtractor(
            observation_space=observation_space,
            output_dim=feature_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            init_type=init_type,
        )

        # Actor head for policy action selection
        self.actor_head: MLP = MLP(
            input_dim=feature_dim,
            output_dim=action_dim,
            hidden_dims=[],
            activation=activation,
            init_type=init_type,
            gain=0.01,
        )

        # Critic head for state-value V(s) estimation
        self.critic_head: MLP = MLP(
            input_dim=feature_dim,
            output_dim=1,
            hidden_dims=[],
            activation=activation,
            init_type=init_type,
            gain=1.0,
        )

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        temperature: float = 1.0,
    ) -> Tuple[CategoricalDistribution, torch.Tensor]:
        """Forward pass returning action distribution and state value V(s).

        Returns:
            Tuple of (CategoricalDistribution, state_values_tensor).
        """
        features = self.feature_extractor(obs)
        logits = self.actor_head(features)
        values = self.critic_head(features)

        dist = CategoricalDistribution(logits=logits, temperature=temperature)
        return dist, values

    def evaluate_actions(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates state values, log probabilities, and distribution entropy for given actions.

        Returns:
            Tuple of (values, log_probs, entropy).
        """
        dist, values = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_probs, entropy
