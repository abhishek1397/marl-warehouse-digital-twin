"""Abstract BaseNetwork class providing standard PyTorch module extensions."""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class BaseNetwork(nn.Module, ABC):
    """Abstract base class for all PyTorch neural network modules in MARL framework."""

    def __init__(self) -> None:
        super().__init__()
        self._device: torch.device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Returns current device of the network."""
        return self._device

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract forward pass method."""
        pass

    def to_device(self, device: Union[str, torch.device]) -> "BaseNetwork":
        """Moves network to specified device and updates internal reference."""
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self.to(device)

    def save(self, path: str) -> None:
        """Saves PyTorch state_dict to a file path."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: Optional[Union[str, torch.device]] = None) -> None:
        """Loads PyTorch state_dict from a file path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Network state file not found: {path}")

        target_device = device or self._device
        state_dict = torch.load(path, map_location=target_device, weights_only=True)
        self.load_state_dict(state_dict)
        if device is not None:
            self.to_device(device)

    def count_parameters(self) -> Dict[str, int]:
        """Returns dictionary of total, trainable, and non-trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }

    def weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculates statistical summary (mean, std, min, max) for each named parameter."""
        stats: Dict[str, Dict[str, float]] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                data = param.data
                stats[name] = {
                    "mean": float(data.mean().item()),
                    "std": float(data.std().item()) if data.numel() > 1 else 0.0,
                    "min": float(data.min().item()),
                    "max": float(data.max().item()),
                }
        return stats

    def get_summary(self) -> str:
        """Returns string representation of network structure and parameter counts."""
        counts = self.count_parameters()
        return (
            f"Network Architecture: {self.__class__.__name__}\n"
            f"Device: {self._device}\n"
            f"Total Parameters: {counts['total']:,}\n"
            f"Trainable Parameters: {counts['trainable']:,}\n"
            f"Non-Trainable Parameters: {counts['non_trainable']:,}\n"
            f"{'='*50}\n"
            f"{super().__repr__()}"
        )
