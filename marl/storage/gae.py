"""Generalized Advantage Estimation (GAE) computation module following Schulman et al."""

from typing import Tuple, Union

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Generalized Advantage Estimation (GAE) computation.

    Formulas:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
        Returns_t = A_t + V(s_t)

    Args:
        rewards: 1D Tensor of step rewards (T,).
        values: 1D Tensor of state value estimates V(s_t) (T,).
        next_values: 1D Tensor of next state values V(s_{t+1}) (T,).
        dones: 1D Tensor of boolean episode termination masks (T,).
        gamma: Discount factor.
        gae_lambda: GAE bias-variance trade-off parameter.

    Returns:
        Tuple of (advantages, returns) Tensors of shape (T,).
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)

    last_gae_lam = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_val = next_values[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

    returns = advantages + values
    return advantages, returns


def compute_gae_reference(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference step-by-step loop GAE computation for unit test validation."""
    T = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32, device=rewards.device)

    gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        adv[t] = gae

    ret = adv + values
    return adv, ret
