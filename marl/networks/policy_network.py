"""PolicyNetwork high-level policy wrapper supporting act(), evaluate_actions(), and predict()."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Space

from marl.networks.actor import ActorNetwork
from marl.networks.base import BaseNetwork
from marl.networks.distribution import CategoricalDistribution
from marl.networks.shared_actor_critic import SharedActorCritic


class PolicyNetwork(BaseNetwork):
    """High-level Policy Network wrapper supporting stochastic/deterministic inference and evaluation."""

    def __init__(
        self,
        observation_space: Union[Space, int],
        action_dim: int = 8,
        use_shared_critic: bool = False,
        feature_dim: int = 128,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        init_type: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.use_shared_critic: bool = use_shared_critic

        if use_shared_critic:
            self.backbone: SharedActorCritic = SharedActorCritic(
                observation_space=observation_space,
                action_dim=action_dim,
                feature_dim=feature_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                init_type=init_type,
            )
        else:
            self.backbone: ActorNetwork = ActorNetwork(
                observation_space=observation_space,
                action_dim=action_dim,
                feature_dim=feature_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                init_type=init_type,
            )

    def forward(
        self, obs: Union[torch.Tensor, Dict[str, Any]], temperature: float = 1.0
    ) -> Union[CategoricalDistribution, Tuple[CategoricalDistribution, torch.Tensor]]:
        """Forward pass through underlying policy backbone."""
        return self.backbone(obs, temperature=temperature)

    def act(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects action and returns (action, log_prob)."""
        if self.use_shared_critic:
            dist, _ = self.backbone(obs, temperature=temperature)
        else:
            dist = self.backbone.get_distribution(obs, temperature=temperature)

        if mask is not None:
            from marl.action_masking import MaskedPolicyWrapper
            from marl.networks.distribution import CategoricalDistribution
            masked_logits = MaskedPolicyWrapper.apply_mask(dist.logits, mask)
            dist = CategoricalDistribution(logits=masked_logits, temperature=temperature)

        action = dist.mode() if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def predict(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Inference prediction helper returning action tensor."""
        action, _ = self.act(obs, mask=mask, deterministic=deterministic)
        return action

    def evaluate_actions(
        self, obs: Union[torch.Tensor, Dict[str, Any]], actions: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Evaluates values (if shared critic present), log probabilities, and entropy for given actions."""
        if self.use_shared_critic:
            return self.backbone.evaluate_actions(obs, actions)
        else:
            dist = self.backbone.get_distribution(obs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return None, log_probs, entropy
