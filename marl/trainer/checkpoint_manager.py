"""CheckpointManager handling PyTorch model state saving, loading, best-model tracking, and resume."""

import glob
import os
from typing import Any, Dict, List, Optional

import torch

from marl.trainer.config import CheckpointSubConfig


class CheckpointManager:
    """Manages PyTorch state_dict checkpoint saving, loading, best tracking, and pruning."""

    def __init__(
        self,
        checkpoint_dir: str,
        config: Optional[CheckpointSubConfig] = None,
    ) -> None:
        self.checkpoint_dir: str = os.path.abspath(checkpoint_dir)
        self.config: CheckpointSubConfig = config or CheckpointSubConfig()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.latest_path: str = os.path.join(self.checkpoint_dir, "latest.pt")
        self.best_path: str = os.path.join(self.checkpoint_dir, "best_model.pt")

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        step: int,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Saves a training checkpoint state dictionary.

        Args:
            state_dict: Dictionary containing PyTorch model, optimizer, or trainer states.
            step: Current step or episode integer index.
            is_best: If True, updates best_model.pt.
            metadata: Additional metadata dictionary.

        Returns:
            Absolute filepath of saved checkpoint.
        """
        payload = {
            "step": step,
            "state_dict": state_dict,
            "metadata": metadata or {},
        }

        # Step checkpoint path: checkpoint_000100.pt
        step_filename = f"checkpoint_{step:06d}.pt"
        step_path = os.path.join(self.checkpoint_dir, step_filename)

        torch.save(payload, step_path)
        torch.save(payload, self.latest_path)

        if is_best and self.config.save_best:
            torch.save(payload, self.best_path)

        self._prune_old_checkpoints()
        return step_path

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Loads a checkpoint payload from filepath."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Loads latest.pt checkpoint if it exists."""
        if os.path.exists(self.latest_path):
            return self.load_checkpoint(self.latest_path)
        return None

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Loads best_model.pt checkpoint if it exists."""
        if os.path.exists(self.best_path):
            return self.load_checkpoint(self.best_path)
        return None

    def _prune_old_checkpoints(self) -> None:
        """Prunes older step checkpoints exceeding max_checkpoints limit."""
        if self.config.max_checkpoints <= 0:
            return

        pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.pt")
        ckpt_files = sorted(glob.glob(pattern))

        if len(ckpt_files) > self.config.max_checkpoints:
            files_to_delete = ckpt_files[: -self.config.max_checkpoints]
            for filepath in files_to_delete:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
