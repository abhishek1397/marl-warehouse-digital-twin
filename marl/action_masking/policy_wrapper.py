"""MaskedPolicyWrapper module wrapping policy logits with dynamic action masking."""

from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MaskedPolicyWrapper:
    """Modifies policy logits with dynamic action masks before Categorical distribution sampling."""

    @staticmethod
    def apply_mask(
        logits: torch.Tensor,
        mask: Union[torch.Tensor, np.ndarray],
        mask_fill_value: float = -1e9,
    ) -> torch.Tensor:
        """Modifies policy logits by filling invalid action entries with -1e9.

        Args:
            logits: Policy logits tensor of shape (8,) or (B, 8).
            mask: Boolean tensor of shape (8,) or (B, 8).
            mask_fill_value: Large negative float value (default -1e9).

        Returns:
            Masked logits tensor of matching shape.
        """
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=logits.device)

        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        if mask.device != logits.device:
            mask = mask.to(logits.device)

        # Expand mask if batch sizes differ
        if logits.dim() == 2 and mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(logits.shape[0], -1)

        masked_logits = logits.masked_fill(~mask, mask_fill_value)
        return masked_logits

    @staticmethod
    def get_masked_distribution(
        logits: torch.Tensor,
        mask: Union[torch.Tensor, np.ndarray],
    ) -> Categorical:
        """Creates a PyTorch Categorical distribution over masked logits."""
        masked_logits = MaskedPolicyWrapper.apply_mask(logits, mask)
        return Categorical(logits=masked_logits)

    @staticmethod
    def sample_masked_action(
        policy_net: nn.Module,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through policy_net with action mask injection.

        Returns:
            Tuple of (action_tensor, log_prob_tensor, entropy_tensor).
        """
        res = policy_net(obs)
        if isinstance(res, tuple):
            dist_obj = res[0]
            logits = dist_obj.logits if hasattr(dist_obj, "logits") else res[0]
        elif hasattr(res, "logits"):
            logits = res.logits
        elif isinstance(res, torch.Tensor):
            logits = res
        else:
            logits = res

        dist = MaskedPolicyWrapper.get_masked_distribution(logits, mask)

        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy
