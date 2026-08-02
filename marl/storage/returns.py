"""Return computation engines for discounted, Monte Carlo, and bootstrapped returns."""

from typing import List

import torch

from marl.storage.trajectory import Trajectory


def compute_discounted_returns(
    rewards: torch.Tensor, dones: torch.Tensor, gamma: float = 0.99
) -> torch.Tensor:
    """Computes discounted cumulative returns R_t = sum_{k=0}^{T} gamma^k * r_{t+k}."""
    T = len(rewards)
    returns = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    running_return = 0.0

    for t in reversed(range(T)):
        if dones[t]:
            running_return = 0.0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def compute_mc_returns(
    trajectories: List[Trajectory], gamma: float = 0.99
) -> List[torch.Tensor]:
    """Computes Monte Carlo returns for a list of trajectory objects."""
    mc_returns_list: List[torch.Tensor] = []

    for traj in trajectories:
        rewards = torch.tensor([t.reward for t in traj.transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in traj.transitions], dtype=torch.bool)
        ret = compute_discounted_returns(rewards, dones, gamma=gamma)
        mc_returns_list.append(ret)

    return mc_returns_list


def compute_bootstrapped_returns(
    advantages: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Computes bootstrapped returns R_t = A_t + V(s_t)."""
    return advantages + values
