"""PPOCheckpointHandler wrapping CheckpointManager for PPO policy and optimizer state persistence."""

from typing import Any, Dict, Optional

import torch

from marl.algorithms.ppo.optimizer import PPOOptimizer
from marl.networks.policy_network import PolicyNetwork
from marl.trainer.checkpoint_manager import CheckpointManager


class PPOCheckpointHandler:
    """Handles saving and loading of PPO policy state dicts, optimizer states, and metadata."""

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        self.ckpt_manager: CheckpointManager = checkpoint_manager

    def save_checkpoint(
        self,
        policy: PolicyNetwork,
        optimizer: PPOOptimizer,
        step: int,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Saves current PPO training checkpoint."""
        state_dict = {
            "policy": policy.state_dict(),
            "optimizer": optimizer.optimizer.state_dict(),
        }
        return self.ckpt_manager.save_checkpoint(
            state_dict=state_dict,
            step=step,
            is_best=is_best,
            metadata=metadata,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str,
        policy: PolicyNetwork,
        optimizer: Optional[PPOOptimizer] = None,
    ) -> Dict[str, Any]:
        """Restores policy and optimizer states from checkpoint file."""
        payload = self.ckpt_manager.load_checkpoint(checkpoint_path)
        state_dict = payload["state_dict"]

        if "policy" in state_dict:
            policy.load_state_dict(state_dict["policy"])

        if optimizer is not None and "optimizer" in state_dict:
            optimizer.optimizer.load_state_dict(state_dict["optimizer"])

        return payload
