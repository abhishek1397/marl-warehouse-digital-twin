"""PyTorch network utility functions for gradient inspection, tensor movement, and layer details."""

from typing import Any, Dict, List, Union

import torch
import torch.nn as nn


def compute_grad_norm(module: nn.Module) -> float:
    """Calculates total L2 gradient norm across all parameter gradients."""
    total_norm_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm_sq += param_norm ** 2
    return total_norm_sq ** 0.5


def inspect_layers(module: nn.Module) -> List[Dict[str, Any]]:
    """Returns list of layer inspection diagnostics (name, type, numel, shape)."""
    layers_info = []
    for name, layer in module.named_modules():
        if len(list(layer.children())) == 0:  # Leaf layer
            num_params = sum(p.numel() for p in layer.parameters())
            layers_info.append(
                {
                    "name": name,
                    "type": layer.__class__.__name__,
                    "num_parameters": num_params,
                }
            )
    return layers_info


def to_device(
    data: Union[torch.Tensor, Dict[str, Any]], device: torch.device
) -> Union[torch.Tensor, Dict[str, Any]]:
    """Recursively moves PyTorch tensors or dictionary of tensors to specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data
