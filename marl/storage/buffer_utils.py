"""Helper functions for observation tensor conversion, stacking, and mask computation."""

from typing import Any, Dict, List, Union

import numpy as np
import torch


def convert_obs_to_tensor(
    obs: Union[np.ndarray, Dict[str, np.ndarray], torch.Tensor, Dict[str, torch.Tensor]]
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Converts NumPy or dict observation into PyTorch float32 tensor."""
    if isinstance(obs, torch.Tensor):
        return obs.to(torch.float32)
    elif isinstance(obs, dict):
        return {
            k: (v.to(torch.float32) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
            for k, v in obs.items()
        }
    elif isinstance(obs, (np.ndarray, list, tuple)):
        return torch.tensor(obs, dtype=torch.float32)
    raise TypeError(f"Cannot convert observation of type {type(obs)} to tensor.")


def stack_observations(
    obs_list: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Stacks a list of observation tensors or dict of tensors along batch dimension 0."""
    if not obs_list:
        raise ValueError("Cannot stack empty observation list.")

    first = obs_list[0]
    if isinstance(first, dict):
        stacked_dict = {}
        for key in first.keys():
            tensors = [convert_obs_to_tensor(o[key]) for o in obs_list]
            stacked_dict[key] = torch.stack(tensors, dim=0)
        return stacked_dict
    else:
        tensors = [convert_obs_to_tensor(o) for o in obs_list]
        return torch.stack(tensors, dim=0)


def compute_mask(terminated: bool, truncated: bool) -> float:
    """Computes episode continuation mask (0.0 if terminated else 1.0)."""
    return 0.0 if terminated else 1.0
