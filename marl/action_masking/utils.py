"""Action masking utility helper functions for debugging and diagnostic visualization."""

from typing import Dict, List, Union

import numpy as np
import torch


def calculate_mask_entropy(mask: Union[torch.Tensor, np.ndarray]) -> float:
    """Computes uniform probability entropy of valid actions in mask."""
    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    valid_count = int(np.sum(arr))
    if valid_count <= 0:
        return 0.0
    p = 1.0 / valid_count
    return float(-valid_count * (p * np.log(p)))


def compute_mask_utilization(mask: Union[torch.Tensor, np.ndarray]) -> float:
    """Computes percentage of actions eliminated by mask."""
    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    num_actions = len(arr)
    num_masked = num_actions - int(np.sum(arr))
    return float((num_masked / num_actions) * 100.0)


def format_mask_visualization(
    raw_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    mask: torch.Tensor,
    selected_action: int,
) -> str:
    """Renders diagnostic ASCII visualization of raw logits, masked logits, and selected action."""
    action_names = ["Up", "Down", "Left", "Right", "Wait", "Pick", "Drop", "Charge"]
    r_logits = raw_logits.squeeze().detach().cpu().numpy()
    m_logits = masked_logits.squeeze().detach().cpu().numpy()
    m_arr = mask.squeeze().detach().cpu().numpy()

    lines = ["Action Masking Visualization:"]
    lines.append(f"{'Action':<8} | {'Valid':<5} | {'Raw Logit':<10} | {'Masked Logit':<12} | {'Selected'}")
    lines.append("-" * 55)

    for i in range(8):
        sel_str = "  <--" if i == selected_action else ""
        lines.append(
            f"{action_names[i]:<8} | {str(bool(m_arr[i])):<5} | {r_logits[i]:<10.2f} | {m_logits[i]:<12.2f} |{sel_str}"
        )

    return "\n".join(lines)
