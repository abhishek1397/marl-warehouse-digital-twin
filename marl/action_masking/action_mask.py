"""ActionMask dataclass representing boolean action validity masks."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class ActionMask:
    """Dataclass holding boolean action validity mask and diagnostic metrics."""

    mask_array: np.ndarray
    mask_tensor: torch.Tensor
    valid_indices: List[int] = field(default_factory=list)
    num_valid: int = 0
    mask_entropy: float = 0.0

    def __post_init__(self) -> None:
        if not self.valid_indices:
            self.valid_indices = [int(i) for i in np.where(self.mask_array)[0]]
        self.num_valid = len(self.valid_indices)
        if self.num_valid > 0:
            probs = self.mask_array.astype(float) / self.num_valid
            probs = probs[probs > 0]
            self.mask_entropy = float(-np.sum(probs * np.log(probs)))
        else:
            self.mask_entropy = 0.0
