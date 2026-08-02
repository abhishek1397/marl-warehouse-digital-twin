"""RolloutBuffer for single and multi-agent trajectory collection and mini-batch generation."""

from typing import Any, Dict, Iterator, List, Optional, Union

import torch

from marl.storage.batch import Batch
from marl.storage.buffer_utils import (
    compute_mask,
    convert_obs_to_tensor,
    stack_observations,
)
from marl.storage.gae import compute_gae
from marl.storage.normalization import normalize_advantages
from marl.storage.sampler import MiniBatchSampler
from marl.storage.statistics import BufferStatistics
from marl.storage.transition import Transition


class RolloutBuffer:
    """Rollout Buffer storing transitions, computing GAE advantages, and generating PyTorch mini-batches."""

    def __init__(
        self,
        capacity: int = 10000,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.capacity: int = capacity
        self.device: torch.device = (
            torch.device(device) if isinstance(device, str) else device
        )

        self.transitions: List[Transition] = []
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    def insert(self, transition: Transition) -> None:
        """Inserts a single step transition into the buffer."""
        if len(self.transitions) >= self.capacity:
            # Buffer overflow protection: remove oldest transition
            self.transitions.pop(0)
        self.transitions.append(transition)

    def insert_joint(self, joint_transitions: Dict[str, Transition]) -> None:
        """Inserts a dictionary of multi-agent joint step transitions."""
        for t in joint_transitions.values():
            self.insert(t)

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_adv: bool = True,
        last_value: float = 0.0,
    ) -> None:
        """Computes GAE advantage estimates and target returns across stored transitions."""
        if not self.transitions:
            return

        T = len(self.transitions)
        rewards = torch.tensor([t.reward for t in self.transitions], dtype=torch.float32, device=self.device)
        values = torch.tensor([t.value_estimate for t in self.transitions], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in self.transitions], dtype=torch.bool, device=self.device)

        # Construct next state values
        next_values = torch.zeros(T, dtype=torch.float32, device=self.device)
        for t in range(T - 1):
            next_values[t] = self.transitions[t + 1].value_estimate
        next_values[-1] = last_value

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            next_values=next_values,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        if normalize_adv:
            advantages = normalize_advantages(advantages)

        self.advantages = advantages
        self.returns = returns

    def get_generator(
        self, mini_batch_size: int, num_epochs: int = 4
    ) -> Iterator[Batch]:
        """Yields Batch objects sliced into mini-batches over num_epochs."""
        if not self.transitions:
            return

        N = len(self.transitions)
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call compute_returns_and_advantages() before get_generator().")

        obs_list = [t.observation for t in self.transitions]
        stacked_obs = stack_observations(obs_list)

        actions = torch.tensor([t.action for t in self.transitions], device=self.device)
        values = torch.tensor([t.value_estimate for t in self.transitions], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.transitions], dtype=torch.float32, device=self.device)
        masks = torch.tensor([compute_mask(t.terminated, t.truncated) for t in self.transitions], dtype=torch.float32, device=self.device)
        agent_ids = [t.agent_id for t in self.transitions]

        sampler = MiniBatchSampler(
            dataset_size=N,
            mini_batch_size=mini_batch_size,
            num_epochs=num_epochs,
            shuffle=True,
        )

        for indices in sampler.sample_indices():
            if isinstance(stacked_obs, dict):
                sub_obs = {k: v[indices].to(self.device) for k, v in stacked_obs.items()}
            else:
                sub_obs = stacked_obs[indices].to(self.device)

            sub_agent_ids = [agent_ids[i] for i in indices.tolist()]

            yield Batch(
                observations=sub_obs,
                actions=actions[indices].to(self.device),
                advantages=self.advantages[indices].to(self.device),
                returns=self.returns[indices].to(self.device),
                values=values[indices].to(self.device),
                old_log_probs=old_log_probs[indices].to(self.device),
                masks=masks[indices].to(self.device),
                agent_ids=sub_agent_ids,
            )

    def to_device(self, device: Union[str, torch.device]) -> None:
        """Transfers buffer and computed tensors to target device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.advantages is not None:
            self.advantages = self.advantages.to(self.device)
        if self.returns is not None:
            self.returns = self.returns.to(self.device)

    def get_statistics(self) -> Dict[str, float]:
        """Returns statistical diagnostics of stored buffer data."""
        return BufferStatistics.compute(
            transitions=self.transitions,
            advantages=self.advantages,
            returns=self.returns,
            capacity=self.capacity,
        )

    def clear(self) -> None:
        """Clears all transitions and computed tensors."""
        self.transitions.clear()
        self.advantages = None
        self.returns = None

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def __len__(self) -> int:
        """Returns total number of transitions stored."""
        return len(self.transitions)
