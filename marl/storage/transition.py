"""Transition dataclass storing a single environment step experience tuple."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


@dataclass
class Transition:
    """Dataclass storing step transition data for single or multi-agent RL algorithms."""

    observation: Union[np.ndarray, Dict[str, np.ndarray], torch.Tensor]
    action: Union[int, float, np.ndarray, torch.Tensor]
    reward: float
    next_observation: Optional[Union[np.ndarray, Dict[str, np.ndarray], torch.Tensor]] = None
    terminated: bool = False
    truncated: bool = False
    value_estimate: float = 0.0
    log_prob: float = 0.0
    hidden_state: Optional[Any] = None
    agent_id: str = "agent_0"
    episode_id: int = 0
    timestep: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Returns True if episode ended via termination or truncation."""
        return self.terminated or self.truncated
