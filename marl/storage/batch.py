"""Batch dataclass storing PyTorch tensor mini-batches for RL optimizers."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class Batch:
    """Tensor batch container storing experience tensors for mini-batch SGD optimization."""

    observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    old_log_probs: torch.Tensor
    masks: torch.Tensor
    agent_ids: Optional[List[str]] = None

    def to_device(self, device: torch.device) -> "Batch":
        """Moves all batch tensors to specified PyTorch device."""
        if isinstance(self.observations, dict):
            obs_dev = {k: v.to(device) for k, v in self.observations.items()}
        else:
            obs_dev = self.observations.to(device)

        return Batch(
            observations=obs_dev,
            actions=self.actions.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            values=self.values.to(device),
            old_log_probs=self.old_log_probs.to(device),
            masks=self.masks.to(device),
            agent_ids=self.agent_ids,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts Batch into dictionary format."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "advantages": self.advantages,
            "returns": self.returns,
            "values": self.values,
            "old_log_probs": self.old_log_probs,
            "masks": self.masks,
            "agent_ids": self.agent_ids,
        }
