"""PPO Loss module implementing Schulman et al. clipped surrogate objective and value loss."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from marl.algorithms.ppo.config import PPOConfig
from marl.storage.batch import Batch


@dataclass
class PPOLossOutput:
    """Dataclass storing scalar loss components and diagnostic metrics."""

    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    total_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


class PPOLoss:
    """Computes Clipped Policy Loss, Value Function Loss, and Entropy Bonus for PPO."""

    def __init__(self, config: PPOConfig) -> None:
        self.config: PPOConfig = config

    def compute_loss(
        self,
        policy_network: nn.Module,
        batch: Batch,
    ) -> PPOLossOutput:
        """Calculates PPO loss components for a single mini-batch.

        Args:
            policy_network: Network module exposing evaluate_actions(obs, action).
            batch: Batch tensor object containing obs, actions, advantages, returns, values, old_log_probs.

        Returns:
            PPOLossOutput dataclass.
        """
        obs = batch.observations
        actions = batch.actions
        advantages = batch.advantages
        returns = batch.returns
        old_log_probs = batch.old_log_probs

        # 1. Evaluate current policy on batch observations and actions
        if hasattr(policy_network, "evaluate_actions"):
            values, new_log_probs, entropy = policy_network.evaluate_actions(obs, actions)
        else:
            dist = policy_network(obs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            values = torch.zeros_like(returns)

        if values is None:
            values = torch.zeros_like(returns)

        values = values.squeeze(-1) if values.dim() > 1 else values

        # 2. Probability Ratio r_t(theta) = exp(new_log_prob - old_log_prob)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Approximate KL divergence and clip fraction tracking
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (torch.abs(ratio - 1.0) > self.config.clip_eps).float().mean()

        # 3. Clipped Surrogate Objective L^CLIP
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 4. Value Function Loss L^VF
        value_loss = 0.5 * nn.functional.mse_loss(values, returns)

        # 5. Entropy Bonus S[pi]
        entropy_loss = entropy.mean()

        # 6. Combined Total Loss L = L^CLIP + c1 * L^VF - c2 * S
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy_loss
        )

        return PPOLossOutput(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            total_loss=total_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )
